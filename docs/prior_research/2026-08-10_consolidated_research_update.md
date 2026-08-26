# Consolidated research update：SEC temporal event KG

本文件是 `docs/2026-08-06_GPM_KG時間尺度與慢速交易驗證紀錄.md` 的 2026-08-10 addendum；所有結果與原始產物均在同一 branch `experiment/gpm-kg-consumer`。

## Gate 結果

- SEC metadata：4,018 rows、4,018 unique accessions、30/30 tickers、210/210 ticker-year、0 errors。
- Source text：bounded 627/627 files，accession/SHA-256 archived。
- DEF14A event manifest：799 `compensation_peer` rows、135 unique pairs、每筆有 evidence quote 與 `known_as=filing_date`。
- Month-matched event study：h20/h60/h120 observed absolute pair spread 比 placebo 高 +0.0058/+0.0116/+0.0232，signed spread 近 0。
- Dynamic dispersion gate：5 seeds。real vs price-only MSE wins 4/5；real vs time-shuffle MSE wins 3/5；real vs topology-shuffle MSE wins 4/5。Spearman 沒有穩定改善。

## 證明與限制

這證明資料封存、point-in-time graph 與 dynamic model pipeline 可重現，並提供「事件後 dispersion/risk」的探索性候選訊號；不證明 semantic/time-aware KG、方向性 alpha、portfolio return/Sharpe/MDD 或 RL 改善。

## 決策

暫不進 hierarchical RL。若還要追加，只跑 event-count/recency-only 與 section/type 分層 controls；若仍無法勝過 time/topology controls，就停止 RL sweep，將此 KG relation 記為未證實的慢速 risk candidate。

詳細紀錄：

- `docs/2026-08-10_SEC_filing_manifest_temporal_data_gate.md`
- `docs/2026-08-10_SEC_event_text_pilot.md`
- `docs/2026-08-10_SEC_event_manifest_extraction.md`
- `docs/2026-08-10_SEC_event_pair_movement_gate.md`
- `docs/2026-08-10_dynamic_event_dispersion_gate_5seed.md`
