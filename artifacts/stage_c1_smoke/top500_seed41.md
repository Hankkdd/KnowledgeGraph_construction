# Stage C1 convergence/runtime smoke

This is an implementation smoke test; it is not a KG efficacy gate.

- fold: `fold12 test=2025 train=2022-01-03..2024-12-02 test=2025-01-02..2025-12-31`
- seed: `41`
- device: `cuda`
- total runtime: `37.42s`

| variant | train loss first | train loss last | test Rank IC | test MSE | runtime (s) |
|---|---:|---:|---:|---:|---:|
| no_graph | 0.00872363 | 0.00872363 | -0.00161882 | 0.00875913 | 2.60 |
| self | 0.00833462 | 0.00833462 | 0.0160884 | 0.00882448 | 2.49 |
| real | 0.00902452 | 0.00902452 | 0.0229587 | 0.00884905 | 10.10 |
| relation_shuffle | 0.00903847 | 0.00903847 | 0.0198821 | 0.00894127 | 10.03 |
| topology_shuffle | 0.00922704 | 0.00922704 | 0.0155422 | 0.00861554 | 11.88 |
