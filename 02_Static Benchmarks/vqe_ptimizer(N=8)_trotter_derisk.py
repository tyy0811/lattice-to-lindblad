import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.optimize import minimize
from functools import reduce
import time
import pandas as pd
import warnings


# ============================================================
# 0) Global config (tune here)
# ============================================================
m = 0.1
g = 0.5
w = 1.0
E0 = 0.0              # ground state / sanity checks
t_max = 2.0

DT_LIST = [0.1, 0.05] # de-risking
T_FINE = np.linspace(0.0, t_max, 81)

# N=8 VQE sweep settings (fast + stable)
DEPTHS_N8 = [2, 4, 6]
STAGE1_RESTARTS = 12
STAGE2_TOPK = 2
BOUNDS_RANGE = 10.0

# Iteration budgets (with analytic gradients these are usually enough)
STAGE1_MAXITER = 180
STAGE2_MAXITER = 450

# Early stop per depth (stops restarts once |E - ED| below threshold)
EARLY_STOP_ERR = {2: 5e-2, 4: 2e-2, 6: 2e-3}

# MOD: optional warm-start across depths
WARM_START_ACROSS_DEPTHS = True


# ============================================================
# 1) System construction (dense, then project)
# ============================================================
def get_paulis():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z

def get_vacuum_state(N):
    vac = np.array([1, 0], dtype=complex)
    occ = np.array([0, 1], dtype=complex)
    psi = reduce(np.kron, [vac if i % 2 == 0 else occ for i in range(N)])
    psi = psi / np.linalg.norm(psi)
    return psi

def get_sector_basis_half_filling(N):
    dim = 2**N
    indices = [i for i in range(dim) if bin(i).count("1") == N // 2]
    P = np.zeros((dim, len(indices)), dtype=complex)
    for j, idx in enumerate(indices):
        P[idx, j] = 1.0
    return P

def proj_op(H, P):
    return P.T.conj() @ H @ P

def build_full_ops(N, m, g, w, E0=0.0):
    I, X, Y, Z = get_paulis()
    Id = [I] * N
    def mk(ops): return reduce(np.kron, ops)

    hop_terms = []
    for n in range(N - 1):
        oXX = Id.copy(); oXX[n] = X; oXX[n + 1] = X
        oYY = Id.copy(); oYY[n] = Y; oYY[n + 1] = Y
        hop_terms.append(-(w / 2) * (mk(oXX) + mk(oYY)))

    mass_terms = []
    for n in range(N):
        o = Id.copy(); o[n] = 0.5 * (I - Z)
        mass_terms.append(m * ((-1) ** n) * mk(o))

    elec_terms = []
    for n in range(N - 1):
        # E_n = E0 + sum_{i<=n} q_i, q_i = ((-1)^i - Z_i)/2
        E_op = E0 * np.eye(2**N, dtype=complex)
        for i in range(n + 1):
            o = Id.copy()
            o[i] = 0.5 * (((-1) ** i) * I - Z)
            E_op += mk(o)
        elec_terms.append((g**2 / 2) * (E_op @ E_op))

    ops = {"hop_terms": hop_terms, "mass_terms": mass_terms, "elec_terms": elec_terms}
    H_total = sum(hop_terms) + sum(mass_terms) + sum(elec_terms)
    return H_total, ops

def build_ops(N, m, g, w, project=False, E0=0.0):
    H_full, ops_full = build_full_ops(N, m, g, w, E0=E0)
    psi0_full = get_vacuum_state(N)

    if not project:
        return H_full, ops_full, psi0_full, None

    P = get_sector_basis_half_filling(N)
    ops_proj = {
        "hop_terms":  [proj_op(h, P) for h in ops_full["hop_terms"]],
        "mass_terms": [proj_op(h, P) for h in ops_full["mass_terms"]],
        "elec_terms": [proj_op(h, P) for h in ops_full["elec_terms"]],
    }
    H_proj = proj_op(H_full, P)
    psi0_proj = P.T.conj() @ psi0_full
    psi0_proj /= np.linalg.norm(psi0_proj)
    return H_proj, ops_proj, psi0_proj, P


# ============================================================
# 2) Semi-local grouping
# ============================================================
def split_by_parity_sum(terms, parity):
    out = None
    for i, t in enumerate(terms):
        if i % 2 == parity:
            out = t if out is None else (out + t)
    if out is None:
        out = np.zeros_like(terms[0])
    return out

def partition_indices(n_items, n_groups):
    edges = np.linspace(0, n_items, n_groups + 1, dtype=int)
    return [list(range(a, b)) for a, b in zip(edges[:-1], edges[1:])]

def make_generators_semilocal(ops, n_elec_groups):
    hop_terms  = ops["hop_terms"]
    mass_terms = ops["mass_terms"]
    elec_terms = ops["elec_terms"]

    G = []
    labels = []

    G.append(split_by_parity_sum(hop_terms, 0));  labels.append("hop_odd")
    G.append(split_by_parity_sum(hop_terms, 1));  labels.append("hop_even")
    G.append(split_by_parity_sum(mass_terms, 0)); labels.append("mass_even")
    G.append(split_by_parity_sum(mass_terms, 1)); labels.append("mass_odd")

    parts = partition_indices(len(elec_terms), n_elec_groups)
    for gi, idxs in enumerate(parts):
        block = None
        for j in idxs:
            block = elec_terms[j] if block is None else (block + elec_terms[j])
        if block is None:
            block = np.zeros_like(elec_terms[0])
        G.append(block)
        labels.append(f"elec_block_{gi+1}")

    return G, labels


# ============================================================
# 3) Fast unitary application via pre-diagonalization
# ============================================================
def hermitize(A):
    return 0.5 * (A + A.conj().T)

# MOD: check hermiticity (warn + hermitize) with label context
def ensure_hermitian(G, label, atol=1e-12):
    if not np.allclose(G, G.conj().T, atol=atol):
        max_dev = np.max(np.abs(G - G.conj().T))
        print(f"WARNING: Generator not Hermitian (label={label}); max |G-G†|={max_dev:.3e}. Hermitizing.")
    return hermitize(G)

def prep_eig_generators(generators, labels=None):
    eig = []
    for i, G in enumerate(generators):
        label = labels[i] if labels is not None else f"G{i}"
        Gh = ensure_hermitian(G, label, atol=1e-12)
        evals, evecs = eigh(Gh)
        eig.append((Gh, evals, evecs, label))
    return eig

def apply_eig_unitary(state, theta, evals, evecs, dagger=False):
    phase = np.exp((1j if dagger else -1j) * theta * evals)
    tmp = evecs.conj().T @ state
    tmp = phase * tmp
    return evecs @ tmp


# ============================================================
# 4) VQE with analytic gradient (adjoint method)
# ============================================================
def make_x0_list(n_params, n, seed=1234, include_zero=True, scale=0.1, warm_start=None):
    rng = np.random.default_rng(seed)
    xs = []

    # MOD: warm start first if provided
    if warm_start is not None:
        ws = np.array(warm_start, dtype=float).copy()
        if ws.shape != (n_params,):
            raise ValueError(f"warm_start has shape {ws.shape}, expected {(n_params,)}")
        xs.append(ws)

    if include_zero:
        z = np.zeros(n_params)
        # avoid duplicates if warm_start is already zero-ish
        if not (len(xs) > 0 and np.allclose(xs[0], z, atol=1e-15)):
            xs.append(z)

    while len(xs) < n:
        xs.append(rng.uniform(-scale, scale, n_params))
    return xs

def vqe_minimize_with_grad(H, psi0, eig_gens, layers, bounds_range, x0, maxiter, ftol):
    nG = len(eig_gens)
    n_params = layers * nG
    bounds = [(-bounds_range, bounds_range)] * n_params

    def fun_and_grad(params):
        # Forward pass: store states after each unitary
        states = [psi0]
        psi = psi0
        for k in range(n_params):
            gi = k % nG
            _, evals, evecs, _ = eig_gens[gi]
            psi = apply_eig_unitary(psi, params[k], evals, evecs, dagger=False)
            states.append(psi)

        # Energy
        Hpsi = H @ psi
        E = np.real(np.vdot(psi, Hpsi))

        # Backward pass
        grad = np.zeros(n_params, dtype=float)
        costate = Hpsi  # H |psi_M>

        for k in range(n_params - 1, -1, -1):
            gi = k % nG
            Gmat, evals, evecs, _ = eig_gens[gi]
            sk = states[k + 1]  # state after kth unitary
            grad[k] = -2.0 * np.imag(np.vdot(sk, Gmat @ costate))
            costate = apply_eig_unitary(costate, params[k], evals, evecs, dagger=True)

        return E, grad

    res = minimize(
        fun_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": ftol}
    )
    return res.fun, res.x

def run_vqe_two_stage(
    H, psi0, generators, layers,
    labels=None,
    ed_ref=None, early_stop_err=None,
    bounds_range=10.0,
    stage1_restarts=12, stage1_maxiter=180, stage1_ftol=1e-10,
    stage2_topk=2, stage2_maxiter=450, stage2_ftol=1e-12,
    seed=2025,
    warm_start=None
):
    eig_gens = prep_eig_generators(generators, labels=labels)
    n_params = layers * len(generators)

    x0_list = make_x0_list(
        n_params,
        stage1_restarts,
        seed=seed,
        include_zero=True,
        scale=0.1,
        warm_start=warm_start
    )

    t0 = time.time()
    stage1_results = []
    used_stage1 = 0

    for x0 in x0_list:
        used_stage1 += 1
        E, xopt = vqe_minimize_with_grad(
            H, psi0, eig_gens, layers,
            bounds_range=bounds_range,
            x0=x0, maxiter=stage1_maxiter, ftol=stage1_ftol
        )
        stage1_results.append((E, xopt))

        # MOD: early stop only limits stage-1 restarts; stage-2 still runs
        if (ed_ref is not None) and (early_stop_err is not None):
            if abs(E - ed_ref) < early_stop_err:
                break

    stage1_results.sort(key=lambda z: z[0])
    best_E, best_x = stage1_results[0]

    # MOD: Always do stage-2 refinement on top-k (even if early-stop hit)
    used_stage2 = 0
    for i in range(min(stage2_topk, len(stage1_results))):
        used_stage2 += 1
        E2, x2 = vqe_minimize_with_grad(
            H, psi0, eig_gens, layers,
            bounds_range=bounds_range,
            x0=stage1_results[i][1], maxiter=stage2_maxiter, ftol=stage2_ftol
        )
        if E2 < best_E:
            best_E, best_x = E2, x2

    return best_E, best_x, (time.time() - t0), (used_stage1, used_stage2)


# ============================================================
# 5) De-risking: exact vs trotter observable with inset |Δ|
# ============================================================
def build_charge_operator(N, site, project=False, P=None):
    I, _, _, Z = get_paulis()
    Id = [I] * N
    def mk(ops): return reduce(np.kron, ops)
    ops = Id.copy()
    ops[site] = 0.5 * (((-1) ** site) * I - Z)
    Q_full = mk(ops)
    if not project:
        return Q_full
    assert P is not None
    return proj_op(Q_full, P)

def exact_states_via_eig(H, psi0, times):
    evals, evecs = eigh(H)
    c0 = evecs.conj().T @ psi0
    out = []
    for t in times:
        psi_t = evecs @ (np.exp(-1j * evals * t) * c0)
        out.append(psi_t)
    return out

def trotter_states(ops, psi0, dt, t_max):
    hop_terms  = ops["hop_terms"]
    mass_terms = ops["mass_terms"]
    elec_terms = ops["elec_terms"]

    H_diag = sum(mass_terms) + sum(elec_terms)
    H_odd  = split_by_parity_sum(hop_terms, 0)
    H_even = split_by_parity_sum(hop_terms, 1)

    from scipy.linalg import expm
    U_d = expm(-1j * dt * H_diag)
    U_o = expm(-1j * dt * H_odd)
    U_e = expm(-1j * dt * H_even)
    U_step = U_e @ (U_o @ U_d)

    steps = int(round(t_max / dt))
    times = np.array([k * dt for k in range(steps + 1)], dtype=float)

    psi = psi0.copy()
    states = [psi.copy()]
    for _ in range(steps):
        psi = U_step @ psi
        states.append(psi.copy())
    return times, states

def expval_series(states, O):
    return np.array([np.real(np.vdot(psi, O @ psi)) for psi in states], dtype=float)

def plot_observable(title, outfile, t_exact, y_exact, traces, dt_list, t_max):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_exact, y_exact, linewidth=2, label="Exact")

    for dt in dt_list:
        t_tr, y_tr = traces[dt]
        ax.plot(t_tr, y_tr, "o--", markersize=4, label=f"Trotter dt={dt}")

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\langle q_{N/2}(t)\rangle$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    caption = f"1st-order Trotter; dt ∈ {{{', '.join(str(d) for d in dt_list)}}}; t_max={t_max}"
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(outfile)
    plt.close(fig)


def plot_deviation_standalone(title, outfile, t_exact, y_exact, traces, dt_list, t_max, logy=True):
    fig, ax = plt.subplots(figsize=(7, 5))

    for dt in dt_list:
        t_tr, y_tr = traces[dt]
        y_ex = np.interp(t_tr, t_exact, y_exact)
        dev = np.abs(y_ex - y_tr)

        if logy:
            dev = np.clip(dev, 1e-14, None)
            ax.semilogy(t_tr, dev, "o--", markersize=4, label=f"dt={dt}")
        else:
            ax.plot(t_tr, dev, "o--", markersize=4, label=f"dt={dt}")

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$|\Delta q(t)|$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    caption = f"|Δ| vs time; 1st-order Trotter; dt ∈ {{{', '.join(str(d) for d in dt_list)}}}; t_max={t_max}"
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(outfile)
    plt.close(fig)


# ============================================================
# 6) Main: N=8 VQE sweep + plots + de-risking plots (N=4 + N=8)
# ============================================================
if __name__ == "__main__":
    # ---------- N=8 projected build ----------
    N = 8
    H8, ops8, psi08, P8 = build_ops(N, m, g, w, project=True, E0=E0)
    ed8 = eigh(H8, eigvals_only=True)[0]
    print(f"N=8 sector dim = {H8.shape[0]}")
    print(f"ED E0 = {ed8:.8f}, ED density = {ed8/N:.6f}")

    schemes = {
        "E1-block (5 gen/layer)": 1,
        "E3-block (7 gen/layer)": 3,
    }

    rows = []
    best_record = None

    for scheme_name, n_elec_groups in schemes.items():
        G, labels = make_generators_semilocal(ops8, n_elec_groups=n_elec_groups)
        gen_per_layer = len(G)

        # MOD: warm-start state for this scheme across depths
        prev_best_x = None
        prev_L = None

        for L in DEPTHS_N8:
            early = EARLY_STOP_ERR.get(L, None)

            warm_start = None
            if WARM_START_ACROSS_DEPTHS and (prev_best_x is not None) and (prev_L is not None):
                if L >= prev_L:
                    pad = np.zeros((L - prev_L) * gen_per_layer, dtype=float)
                    warm_start = np.concatenate([prev_best_x, pad])
                else:
                    warm_start = prev_best_x[: L * gen_per_layer]

            E, xbest, rt, (s1_used, s2_used) = run_vqe_two_stage(
                H8, psi08, G, layers=L,
                labels=labels,
                ed_ref=ed8, early_stop_err=early,
                bounds_range=BOUNDS_RANGE,
                stage1_restarts=STAGE1_RESTARTS,
                stage1_maxiter=STAGE1_MAXITER,
                stage2_topk=STAGE2_TOPK,
                stage2_maxiter=STAGE2_MAXITER,
                seed=2025 + 1000 * L + (10 * n_elec_groups),
                warm_start=warm_start
            )

            err = abs(E - ed8)
            dens = E / N
            print(f"[{scheme_name}] L={L}: E={E:.8f}, err={err:.2e}, time={rt:.2f}s (stage1={s1_used}, stage2={s2_used})")

            rows.append({
                "scheme": scheme_name,
                "elec_blocks": n_elec_groups,
                "layers": L,
                "gen_per_layer": gen_per_layer,
                "params": gen_per_layer * L,
                "E_vqe": E,
                "err": err,
                "density": dens,
                "time_s": rt,
                "stage1_used": s1_used,
                "stage2_used": s2_used
            })

            if (best_record is None) or (err < best_record["err"]):
                best_record = {
                    "scheme": scheme_name,
                    "layers": L,
                    "generators": G,
                    "params": xbest,
                    "err": err
                }

            # MOD: update warm-start reference
            prev_best_x = xbest
            prev_L = L

    df = pd.DataFrame(rows).sort_values(["scheme", "layers"]).reset_index(drop=True)
    print("\nSummary table:")
    print(df.to_string(index=False, float_format="%.6e"))

    # ---------- Plot: Energy density vs depth ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    for scheme_name in schemes.keys():
        sub = df[df["scheme"] == scheme_name].sort_values("layers")
        ax.plot(sub["layers"], sub["density"], "o--", label=scheme_name)
    ax.axhline(ed8 / N, linestyle="--", label="ED")
    ax.set_xlabel("Layers")
    ax.set_ylabel(r"Energy density $E/N$")
    ax.set_title("N=8 Energy Density vs Depth (Semi-local grouping)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("n8_energy_density_vs_depth.png")
    plt.close(fig)

    # ---------- Plot: Error vs depth (log) + annotate runtime ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    for scheme_name in schemes.keys():
        sub = df[df["scheme"] == scheme_name].sort_values("layers")
        ax.semilogy(sub["layers"], sub["err"], "o--", label=scheme_name)
        for _, r in sub.iterrows():
            ax.text(r["layers"] + 0.03, r["err"] * 1.05, f"{r['time_s']:.0f}s", fontsize=9)
    ax.set_xlabel("Layers")
    ax.set_ylabel(r"Energy error $|E - E_{\rm ED}|$")
    ax.set_title("N=8 Projected VQE Error vs Depth (Semi-local grouping)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("n8_vqe_error_vs_depth_grouped.png")
    plt.close(fig)

    # ---------- Plot: Pareto runtime vs error (log y), label (scheme,L) ----------
    fig, ax = plt.subplots(figsize=(7, 5))
    for scheme_name in schemes.keys():
        sub = df[df["scheme"] == scheme_name]
        ax.semilogy(sub["time_s"], sub["err"], "o", label=scheme_name)
        for _, r in sub.iterrows():
            ax.text(r["time_s"] * 1.02, r["err"] * 1.02, f"(L={int(r['layers'])})", fontsize=9)
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel(r"Energy error $|E - E_{\rm ED}|$")
    ax.set_title("N=8 Pareto: Runtime vs Error (Semi-local grouping)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("n8_pareto_runtime_vs_error.png")
    plt.close(fig)

    # ---------- De-risking plot (N=4 full space): restore primary deliverable + summary ----------
    N4 = 4
    H4, ops4, psi04, _ = build_ops(N4, m, g, w, project=False, E0=E0)
    ed4 = eigh(H4, eigvals_only=True)[0]
    print(f"\nN=4 full dim = {H4.shape[0]}")
    print(f"N=4 ED E0 = {ed4:.8f}, ED density = {ed4/N4:.6f}")

    Q_mid4 = build_charge_operator(N4, site=N4 // 2, project=False)

    exact_states4 = exact_states_via_eig(H4, psi04, T_FINE)
    q_exact4 = expval_series(exact_states4, Q_mid4)

    traces4 = {}
    for dt in DT_LIST:
        t_tr, states_tr = trotter_states(ops4, psi04, dt=dt, t_max=t_max)
        q_tr = expval_series(states_tr, Q_mid4)
        traces4[dt] = (t_tr, q_tr)

        # Print dt summary vs exact
        q_ex_interp = np.interp(t_tr, T_FINE, q_exact4)
        dev = np.abs(q_ex_interp - q_tr)
        print(f"N=4 de-risk dt={dt}: max|Δq|={dev.max():.3e}, mean|Δq|={dev.mean():.3e}")

    plot_observable(
        title="De-risking: Observable vs time (Full space, N=4)",
        outfile="derisk_observable_vs_time_N4_full.png",
        t_exact=T_FINE,
        y_exact=q_exact4,
        traces=traces4,
        dt_list=DT_LIST,
        t_max=t_max
    )

    plot_deviation_standalone(
        title="De-risking: |Δq(t)| vs time (Full space, N=4)",
        outfile="derisk_delta_q_vs_time_N4_full.png",
        t_exact=T_FINE,
        y_exact=q_exact4,
        traces=traces4,
        dt_list=DT_LIST,
        t_max=t_max,
        logy=True
)


    # ---------- De-risking plot (N=8 projected): observable vs time + standalone |Δ| ----------
    Q_mid8 = build_charge_operator(N, site=N // 2, project=True, P=P8)

    exact_states8 = exact_states_via_eig(H8, psi08, T_FINE)
    q_exact8 = expval_series(exact_states8, Q_mid8)

    traces8 = {}
    for dt in DT_LIST:
        t_tr, states_tr = trotter_states(ops8, psi08, dt=dt, t_max=t_max)
        q_tr = expval_series(states_tr, Q_mid8)
        traces8[dt] = (t_tr, q_tr)

    plot_observable(
        title="De-risking: Observable vs time (Projected, N=8)",
        outfile="derisk_observable_vs_time_N8_projected.png",
        t_exact=T_FINE,
        y_exact=q_exact8,
        traces=traces8,
        dt_list=DT_LIST,
        t_max=t_max
    )

    plot_deviation_standalone(
        title="De-risking: |Δq(t)| vs time (Projected, N=8)",
        outfile="derisk_delta_q_vs_time_N8_projected.png",
        t_exact=T_FINE,
        y_exact=q_exact8,
        traces=traces8,
        dt_list=DT_LIST,
        t_max=t_max,
        logy=True
    )

    print("\nSaved figures:")
    print("  n8_energy_density_vs_depth.png")
    print("  n8_vqe_error_vs_depth_grouped.png")
    print("  n8_pareto_runtime_vs_error.png")
    print("  derisk_observable_vs_time_N4_full.png")
    print("  derisk_delta_q_vs_time_N4_full.png")
    print("  derisk_observable_vs_time_N8_projected.png")
    print("  derisk_delta_q_vs_time_N8_projected.png")

