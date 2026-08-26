# SEC 事件 pair-movement gate 紀錄

日期：2026-08-10  
專案：`GraphRagTrading/0601_atten_V2`  
Branch：`experiment/gpm-kg-consumer`

## 問題與控制

799 筆 `compensation_peer` events 是否在 filing 後帶來可觀察的 pair movement？這是 semantic KG 接到 RL 前的低成本 gate，不是交易回測。

- target：source/target 未來累積報酬差 `pair_spread` 與其絕對值。
- horizons：20、60、120 個交易日。
- observed：effective trade date 為 filing date 後第一個交易日。
- placebo：同一公司 pair、同一 calendar month、排除事件前後 horizon 的隨機日期；每事件 200 個。
- script：`setup/42_temporal_event_month_matched_gate.py`。

## 結果

| horizon | events | observed abs spread | month-matched placebo | difference |
|---:|---:|---:|---:|---:|
| 20 | 799 | 0.0663 | 0.0604 | +0.0058 |
| 60 | 799 | 0.1160 | 0.1045 | +0.0116 |
| 120 | 797 | 0.1716 | 0.1485 | +0.0232 |

signed spread 仍接近零：h20 `+0.0026`、h60 `-0.0068`、h120 `-0.0108`。也就是事件後 pair 分歧/風險提高，但沒有穩定的 source-vs-target 方向。

完整產物：`artifacts/diagnostics/sec_event_month_matched_gate_2019_2025_pilot/`；先前未控制月份的結果在 `artifacts/diagnostics/sec_event_manifest_gate_2019_2025_pilot/`。

## 解讀

這個結果支持一個比「KG 預測方向」更保守的假設：DEF14A peer disclosure 的 filing window 可能對未來 pair dispersion 有條件資訊，但目前不能把它解讀成 return alpha，也不能直接說會改善 Sharpe 或 max drawdown。事件類型本身是年度 proxy/compensation disclosure，並非突發資訊；仍可能存在 filing-season、公司規模與公告選擇偏差。

## Decision

**Decision：pivot，方向性 trend gate 不通過；風險/dispersion 方向值得另做 supervised gate。** 不啟動 RL。下一步應把 target 改為 pair dispersion/volatility 或 drawdown proxy，並以 price-only、self、time-shuffle/topology-shuffle 做 paired supervised controls；若只在 dispersion MSE/校準勝出，就把 KG 定位為慢速風險先驗，不宣稱每日買賣訊號。
