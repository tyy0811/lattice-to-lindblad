# Open Schwinger Entanglement Benchmark Summary

## Per-Gamma Summary

| gamma | peak_entropy_vn | final_entropy_vn | mean_entropy_vn | peak_mean_abs_L | final_mean_abs_L |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.942042 | 0.690262 | 0.762113 | 0.347057 | 0.226014 |
| 0.02 | 1.56277 | 1.42565 | 1.14114 | 0.346117 | 0.236624 |

## Snapshot Rank/Compressibility Summary

| gamma | time | rank_95 | rank_99 | top_p_eig | top2_cum_weight |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 2 | 2 | 0.897461 | 0.991284 |
| 0 | 3 | 2 | 4 | 0.587244 | 0.969821 |
| 0 | 6 | 2 | 4 | 0.756368 | 0.966631 |
| 0.02 | 0 | 2 | 2 | 0.897461 | 0.991284 |
| 0.02 | 3 | 6 | 14 | 0.534439 | 0.891498 |
| 0.02 | 6 | 10 | 17 | 0.612056 | 0.813557 |

## Two-Case Comparison

- Closed reference: `gamma=0`
- Open case: `gamma=0.02`
- Shared snapshot times: `0, 3, 6`
- Post-quench snapshot times used for compressibility verdict: `3, 6`

| entropy_increase_verdict | compressibility_reduction_verdict | field_perturbation_verdict |
|---:|---:|---:|
| ✓ | ✓ | ~ |

| delta_peak_entropy_vn | delta_final_entropy_vn | delta_mean_entropy_vn | delta_peak_mean_abs_L | delta_final_mean_abs_L |
|---:|---:|---:|---:|---:|
| 0.62073 | 0.735386 | 0.379028 | -0.000939296 | 0.0106104 |

- Compressibility verdict definition: Verdict based on post-quench shared snapshot times only (t > 0 within tolerance); t=0 is excluded because closed and open spectra are identical by construction.

## Interpretation Notes

- Panel 1 (`Subsystem von Neumann entropy`): closed case (`gamma=0`) is pure-state entanglement entropy.
- Open case (`gamma>0`): `S_vN(rho_A)` mixes entanglement with local mixedness/classical uncertainty.
- Panel 2 (`Reduced-state spectrum compressibility proxy`): derived from eigenvalues of `rho_A`.
