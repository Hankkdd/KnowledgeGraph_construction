# 監督式 KG 風險 Gate：跨年度 replication

日期：2026-08-06

## 設計

沿用 `setup/31_supervised_graph_risk_gate.py` 的固定設計，只有時間窗與 KG snapshot 改變：

- 30-stock universe、horizon=60、435 pair residual absolute-correlation targets
- real、self、relation-permuted、topology-permuted、sector-only
- seeds 41–45、30 epochs、無 RL episodes
- 2023 graph/train → 2024 test
- artifact：`0601_atten_V2/artifacts/diagnostics/supervised_graph_risk_gate_2023_5seed/`

## 2023→2024 corrected paired results

| metric | control | real mean | control mean | real wins | one-sided p |
|---|---|---:|---:|---:|---:|
| MSE | self | 0.015835 | 0.016159 | 5/5 | 0.0313 |
| MSE | relation-permuted | 0.015835 | 0.016014 | 5/5 | 0.0313 |
| MSE | topology-permuted | 0.015835 | 0.016047 | 4/5 | 0.0625 |
| MSE | sector-only | 0.015835 | 0.016054 | 5/5 | 0.0313 |
| MAE | self | 0.097556 | 0.098015 | 4/5 | 0.0938 |
| MAE | relation-permuted | 0.097556 | 0.098121 | 5/5 | 0.0313 |
| MAE | topology-permuted | 0.097556 | 0.098031 | 4/5 | 0.1563 |
| MAE | sector-only | 0.097556 | 0.098049 | 4/5 | 0.0625 |
| Spearman | self | 0.078472 | 0.021585 | 5/5 | 0.0313 |
| Spearman | relation-permuted | 0.078472 | 0.053526 | 5/5 | 0.0313 |
| Spearman | topology-permuted | 0.078472 | 0.051427 | 4/5 | 0.0625 |
| Spearman | sector-only | 0.078472 | 0.044611 | 5/5 | 0.0313 |
| Top-20 recall | self | 0.262881 | 0.213556 | 5/5 | 0.0313 |
| Top-20 recall | relation-permuted | 0.262881 | 0.239363 | 5/5 | 0.0313 |
| Top-20 recall | topology-permuted | 0.262881 | 0.239537 | 4/5 | 0.0938 |
| Top-20 recall | sector-only | 0.262881 | 0.222167 | 5/5 | 0.0313 |

## 與 2024→2025 的合併解讀

2024→2025 的結果沒有同樣穩定地勝過 relation/topology controls。因此兩個 temporal windows 不能合併成「KG relation semantics 已通過」：

- real vs self：跨兩窗的 Spearman/top-20 recall 方向一致，表示 graph message 相對 no-graph 可能有可重現的 node-state benefit。
- real vs relation-permuted：跨窗方向不一致，不能證明 relation labels 本身有效。
- real vs topology-permuted：跨窗方向不一致，不能證明真實 ticker topology 勝過同容量的亂拓撲。
- sector-only 在部分指標接近或勝過 real，表示 sector/capacity proxy 仍不能排除。

## Decision

**confirm partial graph-state effect；relation-semantic gate 未通過。** 這是比單一年度更有說服力的結果，但仍不足以進入 RL。下一步應做 relation/topology decomposition 或 extraction coverage audit，而不是把這個結果直接接到 portfolio reward。
