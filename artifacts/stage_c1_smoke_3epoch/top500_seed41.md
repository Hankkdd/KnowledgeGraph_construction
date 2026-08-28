# Stage C1 convergence/runtime smoke

This is an implementation smoke test; it is not a KG efficacy gate.

- fold: `fold12 test=2025 train=2022-01-03..2024-12-02 test=2025-01-02..2025-12-31`
- seed: `41`
- device: `cuda`
- total runtime: `100.72s`

| variant | train loss first | train loss last | test Rank IC | test MSE | runtime (s) |
|---|---:|---:|---:|---:|---:|
| no_graph | 0.00872363 | 0.00689422 | 0.0182722 | 0.00873337 | 4.77 |
| self | 0.00833462 | 0.00688605 | 0.0175395 | 0.00870632 | 5.75 |
| real | 0.00902297 | 0.0068995 | -0.000174566 | 0.00893551 | 28.29 |
| relation_shuffle | 0.00903892 | 0.0068943 | 0.00823902 | 0.00859207 | 28.38 |
| topology_shuffle | 0.00922652 | 0.00690041 | -0.0144834 | 0.00902583 | 33.22 |
