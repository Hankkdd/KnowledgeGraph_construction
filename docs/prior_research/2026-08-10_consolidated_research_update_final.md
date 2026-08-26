# Consolidated research update：SEC temporal event KG final gate

## Final evidence

- SEC metadata：4,018 rows、30/30 tickers、210/210 ticker-year。
- Source text：627/627 files archived with accession/SHA-256。
- DEF14A event manifest：799 evidence-backed `compensation_peer` rows。
- Month-matched event study：post-filing pair dispersion increased at h20/h60/h120, but signed spread was near zero.
- Dynamic dispersion gate：real MSE wins 4/5 vs price-only, but only 3/5 vs time-shuffle.
- Count/recency ablation：`count_only` MSE `0.013245` was better than real `0.013413`; real Spearman did not improve.

## Final interpretation

The remaining MSE effect is explainable by event intensity/recency or model capacity. Correct semantic/time-aware KG value was not demonstrated. No evidence supports improved return, Sharpe, max drawdown, or RL reward.

## Decision

Stop RL integration and hyperparameter sweeps for this relation. Preserve the artifacts as a negative/diagnostic result. Only an explicitly new event type with an independent evidence audit would justify another supervised gate.

Detailed records: `2026-08-10_SEC_filing_manifest_temporal_data_gate.md`, `2026-08-10_SEC_event_manifest_extraction.md`, `2026-08-10_dynamic_event_dispersion_gate_5seed.md`, `2026-08-10_event_count_recency_ablation.md`.
