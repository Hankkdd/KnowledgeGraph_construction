# Slow KG Context → Daily RL Smoke 結果

日期：2026-08-06

## 設計

- train/test：2023→2024
- trend model：h120 cross-sectional relative return
- slow context：每 60 trading days 更新一次；每次更新後 cross-sectional z-score/clip 到 [-3, 3]
- policy：固定 self-graph GPM policy；只有 slow feature source 改變
- RL：pretrain 2 epochs、episodes 2、seed 41；僅 smoke，不作正式績效結論

## 結果

| variant | total return | Sharpe | MDD | volatility |
|---|---:|---:|---:|---:|
| no_slow | 0.5295 | 2.5409 | -0.0897 | 0.01096 |
| price_only_slow | 0.6450 | 2.6216 | -0.0980 | 0.01249 |
| real_slow | 0.6241 | 2.2011 | -0.1455 | 0.01473 |
| relation_permuted_slow | 1.0997 | 1.8770 | -0.2522 | 0.02843 |
| topology_permuted_slow | 0.4326 | 2.3689 | -0.0842 | 0.00994 |

Artifact：`0601_atten_V2/artifacts/diagnostics/slow_kg_context_rl_2023_zscore_smoke/`。

## 解讀

- slow feature 可以成功接入 daily policy，資料流與訓練流程正常。
- 但 real_slow 在單一 seed 降低 Sharpe、提高 volatility/MDD；relation-permuted 甚至產生更高 return 但更差 risk，顯示 raw slow score 直接串接到 action state 不具備可辨識的經濟價值。
- price-only slow context 的改善不能算 KG 證據，可能只是額外 feature/capacity effect。

## Decision

**implementation smoke 通過；naive slow-context interface 不通過。** 暫不跑 5-seed RL。下一版應改為讓 slow KG 只調節 risk budget、stock-score gate 或 critic auxiliary input，而不是直接把 raw trend score 串入 daily price encoder；並保留 no-slow、price-only-slow、relation/topology controls。
