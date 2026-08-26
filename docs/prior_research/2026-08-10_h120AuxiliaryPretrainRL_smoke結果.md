# h120 auxiliary-pretrain → daily RL smoke 結果

日期：2026-08-10

## 設計

- h120 cross-sectional relative-return target 預訓練 GPM encoder；再 freeze encoder，訓練 daily allocation head
- 2023→2024；fixed-5 portfolio；seed 41；pretrain 2 epochs、RL episodes 2
- variants：real、self、relation-permuted、topology-permuted
- 沒有把 trend score 直接加到 action；測試的是 representation interface

## 結果

| variant | total return | Sharpe | MDD | volatility |
|---|---:|---:|---:|---:|
| real | 0.6207 | 2.1586 | -0.1493 | 0.01498 |
| self | 0.5227 | 2.5022 | -0.0911 | 0.01102 |
| relation-permuted | 1.1272 | 1.8409 | -0.2626 | 0.02977 |
| topology-permuted | 0.6416 | 2.5198 | -0.0872 | 0.01298 |

Artifact：`0601_atten_V2/artifacts/diagnostics/h120_auxiliary_pretrain_rl_2023_smoke/`。

## 解讀

- real KG auxiliary pretraining 沒有改善 Sharpe、MDD 或 volatility；相對 self，Sharpe 下降、MDD/volatility 上升。
- relation-permuted 的高 return 伴隨最差風險，不能視為 KG 價值。
- 這是單 seed smoke，不是正式 RL 結論；但它與 raw slow-context、slow ranking prior 兩個 interface smoke 的方向一致：h120 ranking signal 尚未轉成 daily action 的 risk-adjusted value。

## Decision

**implementation smoke 通過；auxiliary representation interface 未通過。** 目前不跑正式 5-seed RL，也不再調 prior strength、episodes 或 reward。KG 的正面結果應定位在 slow graph/ranking diagnostic；daily RL integration 暫時結案，除非改成全新的 dynamic/event KG 或 hierarchical slow policy 問題。
