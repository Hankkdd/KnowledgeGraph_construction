# Dynamic event graph dispersion supervised gate：5-seed 紀錄

日期：2026-08-10  
專案：`GraphRagTrading/0601_atten_V2`  
Branch：`experiment/gpm-kg-consumer`

## 研究問題與 frozen design

前一個 month-matched event study 顯示 DEF14A peer events 後 pair dispersion 增加，但 signed spread 接近零。因此這一 gate 不再測方向 alpha，而測 KG 是否能改善慢速 pair-dispersion 預測。

- train：2021-01-01–2023-01-01；test：2023-01-01–2025-12-31。
- target：未來 60 交易日兩家公司 cumulative-return 差的絕對值。
- dynamic graph：只使用 `known_as_of <= sample date` 的 event，365 日 exponential decay。
- variants：`price_only`、`real`、`time_shuffled`、`topology_shuffled`。
- seeds：41–45；15 epochs；CUDA。
- primary metrics：MSE/MAE（越低越好）；Spearman 作排序檢查。
- advance rule：5/5 seed 且同時勝過 controls 才能進下一個 gate。

腳本：`setup/43_supervised_event_dispersion_gate.py`。產物：`artifacts/diagnostics/sec_event_dispersion_gate_5seed_2021_2025/`。

## 結果

| metric | real vs control | real mean | control mean | wins | Wilcoxon |
|---|---|---:|---:|---:|---:|
| MSE | price-only | 0.013410 | 0.013544 | 4/5 | 0.1563 |
| MSE | time-shuffled | 0.013410 | 0.013419 | 3/5 | 0.4063 |
| MSE | topology-shuffled | 0.013410 | 0.013454 | 4/5 | 0.0938 |
| MAE | price-only | 0.083504 | 0.083922 | 4/5 | 0.2188 |
| MAE | time-shuffled | 0.083504 | 0.083599 | 4/5 | 0.0625 |
| MAE | topology-shuffled | 0.083504 | 0.083494 | 2/5 | 0.6875 |
| Spearman | price-only | 0.016171 | 0.020801 | 2/5 | 0.9063 |
| Spearman | time-shuffled | 0.016171 | 0.016391 | 2/5 | 0.5938 |
| Spearman | topology-shuffled | 0.016171 | 0.012628 | 4/5 | 0.0625 |

## 證明了什麼

- point-in-time dynamic graph implementation 可以穩定跑完 5 seeds。
- real graph 對 price-only 的 dispersion MSE/MAE 有探索性改善（MSE 4/5、MAE 4/5），與先前 event study 的 risk/dispersion 方向一致。

## 沒有證明什麼

- 沒有通過預先指定的 5/5 advance rule。
- real 沒有穩定勝過 time-shuffled；time-shuffled 的 MSE 幾乎相同，表示改善可能來自 graph capacity、事件數量或 regularization，而不是正確的事件時間。
- Spearman 沒有改善，因此沒有證明 pair ranking 或可交易方向訊號。
- 沒有證明 portfolio return、Sharpe 或 max drawdown 改善；本實驗完全沒有 RL。

## Decision

**Decision：stop/pivot，暫不進 hierarchical RL。** 目前最有說服力的結論是：「SEC peer-disclosure events 可能是慢速 dispersion/risk context，但 semantic/time-aware KG 增益尚未成立。」若要繼續，只做一個更嚴格且低成本的 quality/ablation：將事件按 filing type/section 分層，加入 event-count/recency-only control，確認 real 是否仍能勝過只看事件數量與新鮮度的 baseline。若仍無法勝過，停止把此 relation 稱為 KG alpha，保留為風險先驗負結果；不再投入 RL hyperparameter sweep。
