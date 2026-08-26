# Event count/recency ablation 紀錄

日期：2026-08-10  
專案：`GraphRagTrading/0601_atten_V2`  
Branch：`experiment/gpm-kg-consumer`

## 目的

dynamic event graph 的 real MSE 曾略優於 price-only，但沒有穩定勝過 time-shuffle。這個消融檢查改善是否其實只來自每檔股票收到的事件量或事件新鮮度，而非 pair topology/semantic relation。

## 設計

- 相同 2021–2022 train、2023–2025 test、horizon=60、365 日 decay、seeds 41–45。
- `count_only`：只輸入每檔股票的 decayed incident event strength。
- `recency_only`：只輸入每檔股票最近事件的 decayed recency。
- 對照：`price_only`、`real`、`time_shuffled`、`topology_shuffled`。
- 目標：未來 60 日 pair absolute cumulative-return spread。

腳本：`setup/44_event_count_recency_ablation.py`。產物：`artifacts/diagnostics/sec_event_count_recency_ablation_5seed/`。

## 結果

| metric | real vs control | real mean | control mean | wins |
|---|---|---:|---:|---:|
| MSE | price-only | 0.013413 | 0.013562 | 4/5 |
| MSE | count-only | 0.013413 | **0.013245** | 1/5 |
| MSE | recency-only | 0.013413 | 0.013577 | 4/5 |
| MSE | time-shuffled | 0.013413 | 0.013411 | 3/5 |
| MSE | topology-shuffled | 0.013413 | 0.013439 | 3/5 |
| Spearman | price-only | 0.011462 | 0.016039 | 1/5 |
| Spearman | count-only | 0.011462 | 0.027105 | 0/5 |
| Spearman | time-shuffled | 0.011462 | 0.013089 | 2/5 |

MAE 上 real 對 time-shuffle 勝 5/5（0.083539 vs 0.083606），但對 count-only 只有 2/5；而 count-only MSE 已優於 real。這不是 semantic KG 證據。

## 結論與決策

**Decision：stop RL integration / pivot to negative finding。** 在目前 SEC `compensation_peer` event relation、h60 dispersion target 與 dynamic graph 下，real graph 沒有勝過 count-only 或 time-shuffle；排序指標也沒有改善。最保守且可辯護的結論是：觀察到的少量 MSE 改善主要可由事件強度/模型容量解釋，尚未證明 KG relation semantic value。

後續不再跑 RL hyperparameter sweep。若教授要求最後補強，只做人工 evidence precision audit 或換一種明確事件類型；不把這條 relation 接入交易 RL 並宣稱績效提升。
