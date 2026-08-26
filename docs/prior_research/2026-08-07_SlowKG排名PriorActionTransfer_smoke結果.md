# Slow KG ranking prior action-transfer smoke

日期：2026-08-07

## 設計

- frozen daily GPM policy：同一個 self-graph policy，訓練一次
- h120 trend score：train-only supervised model；每 60 trading days 更新
- score：cross-sectional z-score/clip；固定 `prior_lambda=0.25`
- test：2023→2024；seed 41；pretrain 2 epochs、RL episodes 2
- 只有 action-transfer 的 slow prior 不同，沒有改 daily reward

## 結果

| variant | total return | Sharpe | MDD | volatility |
|---|---:|---:|---:|---:|
| base | 0.5231 | 2.5312 | -0.0893 | 0.01089 |
| price-only prior | 1.1758 | 2.6251 | -0.1031 | 0.01993 |
| real KG prior | 0.5229 | 1.8382 | -0.1277 | 0.01552 |
| relation-permuted prior | 1.7235 | 2.2099 | -0.2661 | 0.03248 |
| topology-permuted prior | 0.0450 | 0.3479 | -0.1916 | 0.01055 |

Artifact：`0601_atten_V2/artifacts/diagnostics/slow_kg_prior_action_transfer_2023_smoke/`。

## 解讀

- fixed logit-prior interface 可以執行，但 real KG prior 在單一 smoke seed 沒有改善 Sharpe 或 MDD。
- price-only、relation-permuted 的高 return 伴隨更高 volatility/MDD，不能視為 KG evidence。
- `prior_lambda=0.25` 是事前固定值；本結果不支持事後調 lambda 追求 return。

## Decision

**action-transfer smoke 通過；economic gate 未通過。** 目前 h120 ranking signal 尚未成功轉成穩定 portfolio risk-adjusted value。下一步應停止直接調 prior strength，改做 critic/auxiliary representation integration 或 risk-budget constraint；在新 interface 通過前不跑正式 5-seed RL。
