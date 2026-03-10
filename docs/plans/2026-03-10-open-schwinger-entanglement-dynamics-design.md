# Open Schwinger Entanglement Dynamics Design

Date: 2026-03-10
Status: Approved for implementation
Owner: Codex + user

## Objective

Build a single experiment driver that unifies:

- Schwinger many-body quench dynamics (existing closed-system foundation), and
- open-system Lindblad evolution (weak dissipation),

to answer:

How does weak openness alter entanglement growth and effective tensor-network compressibility relative to the closed Schwinger quench?

Scientific framing:

We extend the closed-system Schwinger entanglement analysis to weakly open dynamics and track how dissipation modifies entropy growth, entanglement structure, and effective Schmidt compressibility.

## Scope Freeze (v1)

- Backend: exact Lindblad master-equation evolution in fixed-charge ED sector.
- Dissipation channel: local dephasing on staggered charge operators `Q_n`.
- Channel label (v1): `charge_dephasing` only.
- Benchmark defaults: `N=10`, `tmax=6.0`, `nt=61`, `gamma=0.0`, `gamma_ref=0.02`.
- Entanglement cut: configurable `--cut`, default `4`.
- Physical quench observable panel: mean electric-field magnitude `mean_abs_L(t)`.
- Comparison: closed (`gamma=0`) vs weak open (`gamma_ref>0`).
- Out of scope: multi-channel zoo, large parameter sweeps, trajectory methods, MPS/DMRG-time-evolution.

## Script Placement

`05_Entanglement_Structure_QI/open_schwinger_entanglement_dynamics.py`

Rationale: this is an application artifact centered on entanglement structure and compressibility.

## Architecture (Single Driver, 5 Real Functions)

The script is an orchestrator; no large anonymous procedural blocks in `main()`.

1. `build_sector_model(...)`
2. `prepare_quench_initial_state(...)`
3. `evolve_open_dynamics(...)`
4. `measure_timeseries(...)`
5. `write_outputs_and_plot(...)`

### 1) `build_sector_model(...) -> dict`

Builds and returns the reusable projected-sector payload:

- `basis`: fixed-charge sector basis.
- `dim`: sector dimension.
- `H_init`: pre-quench Hamiltonian.
- `H_evolve`: post-quench Hamiltonian.
- `Q_ops`: local staggered-charge operators in sector basis.
- `L_ops`: link-field operators in sector basis.
- `cut_maps`: precomputed index maps for reduced density matrix at chosen cut.
- `meta`: static model descriptors.

Constraint: all sector/basis bookkeeping lives here only.

### 2) `prepare_quench_initial_state(model, initial_state, quench) -> (psi0, rho0, prep_meta)`

Returns:

- `psi0`: complex vector, shape `(dim,)`.
- `rho0 = |psi0><psi0|`: complex matrix, shape `(dim, dim)`, Hermitian, trace 1.
- `prep_meta`: protocol diagnostics for metadata output.

### 3) `evolve_open_dynamics(model, rho0, times, gamma, channel, rtol, atol) -> dict`

Lindblad RHS (v1):

`d rho / dt = -i [H_evolve, rho] + gamma * sum_n (Q_n rho Q_n - 0.5 {Q_n^2, rho})`

Output:

- `rho_t`: dense complex trajectory, shape `(nt, dim, dim)`.
- `solver_meta`: numerical diagnostics and thresholds status.

### 4) `measure_timeseries(model, rho_t, times, cut, snapshot_times, gamma, channel, row_meta) -> (timeseries_rows, snapshot_rows, measurement_meta)`

Computes:

- `entropy_vn(t)` at selected cut from reduced density matrix `rho_A(t)`.
- `mean_abs_L(t)` from link expectations.
- Snapshot compressibility proxies from reduced-spectrum eigenvalues:
  - `schmidt_proxy_from_rhoA = sqrt(p_eig)`
  - cumulative retained weight.

Input hygiene:

- accepts narrow `row_meta` only (no broad run-args object).

### 5) `write_outputs_and_plot(timeseries_rows, snapshot_rows, meta, outdir, tag, show, force) -> dict`

Pure I/O layer only:

- writes CSV/PNG/JSON artifacts,
- performs no additional physics computation.

## Data Flow

1. Parse and validate CLI.
2. Build sector model once.
3. Prepare one quench initial state (`psi0`, `rho0`).
4. Run evolution for each gamma case (`0.0`, `gamma_ref`).
5. Measure timeseries and snapshots per run.
6. Concatenate rows.
7. Write artifacts and render 3-panel figure.

## CLI Contract (v1)

Required:

- `--N`
- `--mass`
- `--coupling`
- `--outdir`

Dynamics:

- `--tmax` (default `6.0`)
- `--nt` (default `61`)
- `--cut` (default `4`)
- `--initial-state`
- `--quench`

Open-system:

- `--gamma` (default `0.0`)
- `--gamma-ref` (default `0.02`)
- `--channel` (default/only `charge_dephasing`)

Output:

- `--snapshot-times` (default: `0.0,3.0,6.0`)
- `--tag`
- `--force`
- `--show`

Validation requirements at CLI layer:

- `N` must be even.
- `cut` must satisfy `0 <= cut <= N-2`.
- `tmax > 0`, `nt >= 2`.
- `gamma >= 0`, `gamma_ref >= 0`.
- `channel` must be exactly `charge_dephasing` (v1).
- `initial_state` must be one of explicitly enumerated choices.
- `quench` must be one of explicitly enumerated choices.
- each snapshot time must satisfy `0 <= t_snapshot <= tmax` (reject out-of-range; no clamping).

## Output Artifacts

### Main CSV

`open_schwinger_entanglement_dynamics.csv`

Columns:

`time,observable,value,cut,channel,gamma,model,N,m_over_g,x,initial_state,quench`

Observables (v1):

- `entropy_vn`
- `mean_abs_L`

### Snapshot CSV (optional)

`open_schwinger_entanglement_schmidt_snapshots.csv`

Columns:

`time,rank,schmidt_proxy_from_rhoA,p_eig,cum_weight,cut,channel,gamma,model,N,m_over_g,x`

Note: these are reduced-spectrum proxies from mixed-state `rho_A`, not literal global-state Schmidt values.

### Figure

`open_schwinger_entanglement_dynamics.png`

3 panels:

1. `S_vN(t)` closed vs open.
2. cumulative proxy weight at snapshot times.
3. `mean_abs_L(t)` closed vs open.

### Metadata JSON

`run_metadata.json`

Includes:

- arguments and resolved defaults,
- git commit hash,
- static model metadata,
- prep metadata,
- solver diagnostics and threshold outcomes,
- snapshot requested/actual index map,
- reduced-spectrum clipping stats,
- output paths.

## Numerical Safeguards

## Solver-level diagnostics (`solver_meta`)

- `max_abs_trace_error = max_t |Tr(rho_t)-1|`
- `max_hermiticity_error = max_t ||rho_t - rho_t^dagger||_F`
- `min_eig_real_over_checks` for positivity sanity.

Threshold policy:

- warn if trace/hermiticity exceeds `1e-8`; fail if exceeds `1e-5`.
- warn if minimum eigenvalue `< -1e-10`; fail if `< -1e-7`.

Positivity check sample set (reproducible):

- first time index,
- last time index,
- all snapshot indices,
- evenly spaced interior checks (fixed count).

## Reduced density matrix safeguards

Before eigensolve:

- enforce Hermitian symmetrization: `rho_A <- (rho_A + rho_A^dagger) / 2`.

Eigenvalue handling:

- clip negatives below `eig_clip = 1e-12` to zero,
- renormalize clipped spectrum to sum to 1 if needed,
- record clipping counts and max correction in `measurement_meta`.

Also record:

- `trace_rhoA` diagnostics in metadata (not CSV rows).

## Snapshot mapping

- map each requested snapshot time to nearest grid time index,
- record triplets `(requested_time, actual_time, index)` in metadata,
- reject duplicate mapped indices unless explicitly enabled in the future.

## Validation Plan (pre-completion gate)

1. Smoke run at default benchmark

- `N=10, tmax=6.0, nt=61, cut=4, gamma=0.0, gamma_ref=0.02`.
- verify all expected artifacts are generated and non-empty.

2. Physics consistency checks

- For `gamma=0`: purity `Tr(rho^2)` remains within tolerance of 1 (drift reported).
- For `gamma_ref>0`: purity decreases from 1 and remains within physical bounds (within tolerance).
- `entropy_vn(t)` is finite and non-negative up to numerical tolerance.
- Snapshot `p_eig` and `cum_weight` stay in `[0,1]` up to tolerance.

3. Schema checks

- Main CSV columns match contract exactly.
- Snapshot CSV columns match contract exactly.
- Metadata includes solver thresholds, clipping metadata, and snapshot index map.

## Risks and Mitigations

1. Runtime/memory growth in dense trajectory storage

- keep v1 benchmark small (`N=10`, `nt=61`),
- reserve streaming-measurement mode as future extension.

2. Visually trivial result (`entropy` only decays)

- include `mean_abs_L(t)` and snapshot compressibility panel to preserve multi-observable interpretation.

3. Scope drift into generic noise study

- keep one channel, two gamma values, one cut, one quench protocol in v1.

## Acceptance Criteria

Done when all are true:

- script runs closed and weak-open cases in one execution,
- outputs stable `entropy_vn(t)` comparison at selected cut,
- includes compressibility proxy snapshots and `mean_abs_L(t)`,
- produces four artifacts:
  - `open_schwinger_entanglement_dynamics.csv`
  - `open_schwinger_entanglement_schmidt_snapshots.csv` (or documented skip if disabled)
  - `open_schwinger_entanglement_dynamics.png`
  - `run_metadata.json`
- supports a concrete summary claim of dissipation-induced reshaping of entanglement and proxy compressibility.
