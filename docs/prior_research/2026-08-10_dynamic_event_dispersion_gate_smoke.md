# Dynamic event graph dispersion gate：smoke 紀錄

日期：2026-08-10  
專案：`GraphRagTrading/0601_atten_V2`  
Branch：`experiment/gpm-kg-consumer`

## 設計

為了把前一個「事件後 pair 分歧增加」結果轉成可檢驗的 graph-to-target 問題，建立 point-in-time dynamic graph：每個 sample date 只使用 `known_as_of <= sample date` 的事件，並以 365 日 exponential decay 形成 adjacency。模型使用 30 檔 OHLCV context，加上 dynamic event graph，預測未來 60 日所有公司 pair 的 absolute cumulative-return spread。

Variants：`price_only`、`real`、`time_shuffled`、`topology_shuffled`。這是 1 seed、5 epochs smoke，只驗證實作；不作研究結論，也不跑 RL。

執行：

```bash
PYTHONPATH=. /home/hankdd/.venv/bin/python setup/43_supervised_event_dispersion_gate.py \
  --events artifacts/sec_event_manifest_2019_2025_pilot/events.csv \
  --price-data data/price_ohlcv.csv \
  --train-start 2021-01-01 --train-end 2023-01-01 \
  --test-start 2023-01-01 --test-end 2025-12-31 \
  --horizon 60 --epochs 5 --seeds 1 --base-seed 41 \
  --output-dir artifacts/diagnostics/sec_event_dispersion_gate_smoke
```

## 結果

| variant | MSE | MAE | Spearman |
|---|---:|---:|---:|
| price_only | 0.014348 | 0.086071 | 0.006430 |
| real | 0.013780 | 0.083331 | 0.006296 |
| time_shuffled | 0.013806 | 0.083322 | 0.013687 |
| topology_shuffled | 0.013829 | 0.083353 | 0.003349 |

real 在此 smoke 的 MSE 優於 price-only，但 time-shuffled 幾乎相同，且 Spearman 沒有優於 controls。因此這只是「dynamic adjacency 實作可跑」與可能的容量/regularization 現象，不是 semantic KG 證據。

## Decision

**Decision：implementation，進入正式 paired 5-seed risk gate 前不接 RL。** 正式 gate 必須固定資料切分與 decay，跑 5 seeds，比較 real 是否同時勝過 price-only、time-shuffled、topology-shuffled；若只改善 MSE 而沒有排序/方向改善，結論只能是慢速風險/dispersion prior。

產物：`artifacts/diagnostics/sec_event_dispersion_gate_smoke/`。
