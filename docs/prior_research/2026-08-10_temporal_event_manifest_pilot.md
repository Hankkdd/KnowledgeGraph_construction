# Temporal event manifest pilot

日期：2026-08-10

## 為什麼跑

目前 static KG 沒有事件時間，先使用 `event_window_scaleup_manifest.csv` 的 `known_as_of` 與 SEC accession，確認時間正確的 event snapshot 是否能對齊事件後價格變化。

## 設計

- manifest：16 筆有效 dated relations，2019-02～2021-02。
- 對每個 source-target pair，將 `known_as_of` 對齊到下一個交易日。
- 計算事件後 20、60、120 個交易日的 target-minus-source return spread。
- placebo：同一 pair、同一 horizon，將起始日期打亂並避開真實事件附近。
- 不接 RL、不宣稱 alpha；只做 temporal pipeline/event-window feasibility。

## 結果

實際價格 universe 只有 4 個事件 pair 可用，而且全部來自同一份 IBM 2020-02-25 filing（8 個 manifest rows 中只有 4 個 pair 同時在目前 universe 有效）。

| horizon | usable events | observed abs spread | placebo abs spread | difference |
|---:|---:|---:|---:|---:|
| 20 | 4 | 0.1599 | 0.0604 | +0.0994 |
| 60 | 4 | 0.2460 | 0.1144 | +0.1316 |
| 120 | 4 | 0.3411 | 0.1688 | +0.1723 |

事件後 spread 確實高於 pair-matched placebo，但這四個 pair 同屬一個 filing cluster，且 manifest 沒有 polarity，因此不能解釋為可交易方向訊號，也不能估計獨立事件的統計顯著性。

## 證明與限制

這次證明了：

1. `known_as_of` 可以正確對齊交易日。
2. 可以建立事件後 horizon 與 pair-matched time-shuffle control。
3. 現有 manifest 可能含有事件窗口訊號，但目前樣本不足以判斷其來源。

這次沒有證明：

- dynamic KG 具有方向性 alpha；
- KG 能改善 supervised return prediction；
- KG 能改善 RL 或 portfolio。

## Decision

**Temporal pipeline feasibility 通過；資料充分性 gate 未通過。** 不接 RL、不跑 5-seed event gate。下一步先補 2021–2025 的 SEC filing/event metadata，至少覆蓋多個 accession、公司與年度，再進行 supervised event-to-target gate。

原始結果：`0601_atten_V2/artifacts/diagnostics/temporal_event_manifest_gate/`；腳本：`0601_atten_V2/setup/37_temporal_event_manifest_gate.py`。
