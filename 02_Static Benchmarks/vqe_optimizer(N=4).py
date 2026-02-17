import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, eigh
from scipy.optimize import minimize
from functools import reduce
import time
import pandas as pd


# =========================
# 1) System Construction
# =========================
def get_paulis():
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    return I, X, Y, Z

def get_vacuum_state(N):
    vac = np.array([1, 0], dtype=complex)
    occ = np.array([0, 1], dtype=complex)
    return reduce(np.kron, [vac if i%2==0 else occ for i in range(N)])

def get_sector_basis(N):
    dim = 2**N
    indices = [i for i in range(dim) if bin(i).count("1") == N//2]
    P = np.zeros((dim, len(indices)), dtype=complex)
    for j, idx in enumerate(indices):
        P[idx, j] = 1.0
    return P

def proj_op(H, P):
    return P.T.conj() @ H @ P

def build_ops(N, m, g, w, project=False):
    I, X, Y, Z = get_paulis()
    Id = [I]*N
    def mk(ops): return reduce(np.kron, ops)

    hop_ops, mass_ops, elec_ops = [], [], []

    for n in range(N-1):
        oXX=Id.copy(); oXX[n]=X; oXX[n+1]=X
        oYY=Id.copy(); oYY[n]=Y; oYY[n+1]=Y
        hop_ops.append(-(w/2)*(mk(oXX)+mk(oYY)))

    for n in range(N):
        o=Id.copy(); o[n]=0.5*(I-Z)
        mass_ops.append(m*((-1)**n)*mk(o))

    for n in range(N-1):
        E_op = np.zeros((2**N, 2**N), dtype=complex)
        for i in range(n+1):
            o=Id.copy(); o[i]=0.5*((-1)**i*I - Z)
            E_op += mk(o)
        elec_ops.append((g**2/2)*(E_op @ E_op))

    ops = {"hop": hop_ops, "mass": mass_ops, "elec": elec_ops}
    psi0 = get_vacuum_state(N)

    if project:
        P = get_sector_basis(N)
        ops["hop"]  = [proj_op(h, P) for h in hop_ops]
        ops["mass"] = [proj_op(h, P) for h in mass_ops]
        ops["elec"] = [proj_op(h, P) for h in elec_ops]
        psi0 = P.T.conj() @ psi0
        psi0 /= np.linalg.norm(psi0)

    H_total = sum(ops["hop"]) + sum(ops["mass"]) + sum(ops["elec"])
    return H_total, ops, psi0


# =========================
# 1b) Observable builders (charge operator)
# =========================
def build_charge_op_full(N, site):
    """
    q_site = ((-1)^site - Z_site)/2  (in your conventions)
    """
    I, X, Y, Z = get_paulis()
    Id = [I]*N
    def mk(ops): return reduce(np.kron, ops)

    ops = Id.copy()
    ops[site] = 0.5 * (((-1)**site) * I - Z)
    return mk(ops)


# =========================
# 2) VQE Engine (+ early stop)
# =========================
def make_x0_list(n_params, restarts, seed=1234, include_zero=True, scale=0.1):
    rng = np.random.default_rng(seed)
    x0_list = []
    if include_zero:
        x0_list.append(np.zeros(n_params))
    while len(x0_list) < restarts:
        x0_list.append(rng.uniform(-scale, scale, n_params))
    return x0_list

def run_vqe_experiment(H, psi0, ops, layers, ansatz_type="local",
                       x0_list=None, bounds_range=10.0, maxiter=800, ftol=1e-12,
                       ed_ref=None, early_stop_err=None, patience=3):
    N_hop = len(ops["hop"])
    N_mass = len(ops["mass"])
    N_elec = len(ops["elec"])

    H_hop_odd  = sum(ops["hop"][i] for i in range(0, N_hop, 2))
    H_hop_even = sum(ops["hop"][i] for i in range(1, N_hop, 2))
    H_diag     = sum(ops["mass"]) + sum(ops["elec"])

    ppl = 3 if ansatz_type == "global" else (N_hop + N_mass + N_elec)
    n_params = layers * ppl
    bounds = [(-bounds_range, bounds_range)] * n_params

    def ansatz(params):
        state = psi0.copy()
        for l in range(layers):
            p = params[l*ppl:(l+1)*ppl]
            if ansatz_type == "global":
                th_odd, th_even, th_diag = p
                state = expm(-1j * th_diag * H_diag) @ state
                state = expm(-1j * th_odd  * H_hop_odd) @ state
                state = expm(-1j * th_even * H_hop_even) @ state
            else:
                p_hop  = p[:N_hop]
                p_mass = p[N_hop:N_hop+N_mass]
                p_elec = p[N_hop+N_mass:]

                H_m   = sum(p_mass[i] * ops["mass"][i] for i in range(N_mass))
                H_el  = sum(p_elec[i] * ops["elec"][i] for i in range(N_elec))
                H_odd = sum(p_hop[i]  * ops["hop"][i]  for i in range(0, N_hop, 2))
                H_even= sum(p_hop[i]  * ops["hop"][i]  for i in range(1, N_hop, 2))

                state = expm(-1j * H_m) @ state
                state = expm(-1j * H_el) @ state
                state = expm(-1j * H_odd) @ state
                if N_hop > 1:
                    state = expm(-1j * H_even) @ state

            nrm = np.linalg.norm(state)
            if nrm > 1e-12:
                state /= nrm
        return state

    def cost(params):
        psi = ansatz(params)
        num = np.real(np.vdot(psi, H @ psi))
        den = np.real(np.vdot(psi, psi))
        return num / den

    if x0_list is None:
        x0_list = make_x0_list(n_params, restarts=5, seed=1234)

    best_E = np.inf
    best_improve = []
    t0 = time.time()

    used = 0
    for r, x0 in enumerate(x0_list, start=1):
        used = r
        res = minimize(cost, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "ftol": ftol})

        prev = best_E
        if res.fun < best_E:
            best_E = res.fun

        best_improve.append(abs(prev - best_E))

        if (ed_ref is not None) and (early_stop_err is not None):
            if abs(best_E - ed_ref) < early_stop_err:
                break

        if len(best_improve) >= patience and all(x < 1e-10 for x in best_improve[-patience:]):
            break

    return best_E, time.time() - t0, n_params, used



# =========================
# MAIN
# =========================
if __name__ == "__main__":
    m = 0.1; g = 0.5; w = 1.0

    # -------------------------
    # VQE sweep (N=4)
    # -------------------------
    N = 4
    H_full, ops_full, psi0_full = build_ops(N, m, g, w, project=False)
    H_proj, ops_proj, psi0_proj = build_ops(N, m, g, w, project=True)

    ed_full = eigh(H_full, eigvals_only=True)[0]
    ed_proj = eigh(H_proj, eigvals_only=True)[0]
    print(f"N={N}: full={2**N}, sector={H_proj.shape[0]}, ED={ed_full:.6f}")
    print(f"ED diff(full-proj) = {abs(ed_full-ed_proj):.3e}\n")

    layers_list = [2, 4]
    ansatz_list = ["global", "local"]
    results = []

    for ansatz_type in ansatz_list:
        for L in layers_list:
            if ansatz_type == "global":
                restarts = 50
                maxiter = 300
                early_stop = 1e-10
            else:
                restarts = 5
                maxiter = 800
                early_stop = 1e-8

            N_hop = len(ops_full["hop"]); N_mass = len(ops_full["mass"]); N_elec = len(ops_full["elec"])
            ppl = 3 if ansatz_type == "global" else (N_hop + N_mass + N_elec)
            n_params = L * ppl

            x0_list = make_x0_list(n_params, restarts=restarts, seed=2025, include_zero=True, scale=0.1)

            for space_name, H, ops, psi0 in [
                ("Full", H_full, ops_full, psi0_full),
                ("Projected", H_proj, ops_proj, psi0_proj),
            ]:
                E, rt, npar, used = run_vqe_experiment(
                    H, psi0, ops,
                    layers=L,
                    ansatz_type=ansatz_type,
                    x0_list=x0_list,
                    bounds_range=10.0,
                    maxiter=maxiter,
                    ftol=1e-12,
                    ed_ref=ed_full,                 # only safe for N=4
                    early_stop_err=early_stop,
                    patience=3
                )
                results.append({
                    "Space": space_name,
                    "Ansatz": ansatz_type,
                    "Layers": L,
                    "Params": npar,
                    "Restarts_used": used,
                    "Restarts_max": restarts,
                    "Energy": E,
                    "Err_vs_ED": abs(E - ed_full),
                    "Time_s": rt
                })

    df = pd.DataFrame(results).sort_values(["Ansatz","Layers","Space"]).reset_index(drop=True)
    print(df.to_string(index=False, float_format="%.6e"))

    # Save VQE plots to OUTDIR (avoid read-only)
    dfp = df[df["Space"] == "Projected"].copy()

    plt.figure(figsize=(6,4))
    for ans in ansatz_list:
        sub = dfp[dfp["Ansatz"] == ans].sort_values("Layers")
        plt.semilogy(sub["Layers"], sub["Err_vs_ED"], "o--", markersize=4, linewidth=1.5, label=ans)
    plt.axhline(1e-8, linestyle=":", linewidth=1.2, label="1e-8 target")
    plt.xlabel("Layers")
    plt.ylabel(r"Energy error $|E - E_{\rm ED}|$")
    plt.title(f"N=4 VQE Error vs Depth (Projected)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("N=4 VQE Error vs Depth (Projected).png")


    plt.figure(figsize=(6,4))
    for ans in ansatz_list:
        sub = dfp[dfp["Ansatz"] == ans].sort_values("Layers")
        plt.plot(sub["Layers"], sub["Time_s"], "o--", markersize=4, linewidth=1.5, label=ans)
    plt.xlabel("Layers")
    plt.ylabel("Runtime (s)")
    plt.title("N=4 VQE Runtime vs Depth (Projected)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("N=4 VQE Runtime vs Depth (Projected).png")


    

    
    
