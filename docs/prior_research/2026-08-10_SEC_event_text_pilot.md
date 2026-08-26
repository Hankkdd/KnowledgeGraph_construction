# SEC 事件全文 bounded pilot 紀錄

日期：2026-08-10  
專案：`GraphRagTrading/0601_atten_V2`  
Branch：`experiment/gpm-kg-consumer`

## 研究問題

metadata manifest 已確認 30 檔、2019–2025 每個 ticker-year 都有 filing，但 metadata 本身沒有公司關係、事件類型或極性。這一步要確認能否以 point-in-time 規則保存可重現的 source text，供後續事件 KG 抽取；不在此步驟宣稱有交易訊號。

## 設計

- 候選輸入：`artifacts/sec_event_candidates_2019_2025_pilot/event_candidates.csv`。
- 每個 ticker-year：全部 `DEF 14A`，以及按 filing date 等距取最多 2 筆 `8-K`。
- 候選數：627；每筆使用 `filing_date` 作為 `known_as_of`。
- 保存路徑：`text/{ticker}/{accession}.txt`；不覆蓋已有非空檔案。
- 每筆保存 `status`、字元/bytes、`sha256`、來源 URL 與 accession。

執行：

```bash
cd /home/hankdd/NYCU/GraphRagTrading/0601_atten_V2
PYTHONPATH=. /home/hankdd/.venv/bin/python \
  setup/40_download_sec_event_text_pilot.py \
  --delay 0.35 \
  --output-dir artifacts/sec_event_text_2019_2025_pilot
```

## 結果

| 檢查 | 結果 |
|---|---:|
| downloaded/reused | 627/627 |
| local text files | 627 |
| unique accession | 627 |
| HTTP errors | 0 |
| artifact size | 約 78 MB |
| text under 1,000 chars | 1 |

唯一極短檔是 NVDA 2022 的 `DEF 14A` definitive additional materials（`0001045810-22-000068`，964 chars），它是有效 SEC 文件但不是完整 proxy statement；後續需在 `event_type`/section quality gate 標記，不能默認視為完整 peer-group 文件。

## 證明了什麼

- SEC source text 可以依 accession 與 filing date 做可重現、可續跑的本地封存。
- 下一步可以建立 evidence sentence、relation type、event type、polarity、confidence 的事件 manifest。

## 沒有證明什麼

- 尚未證明文本抽取出的 KG relation 正確。
- 尚未證明事件對 h20/h60/h120 報酬有預測力。
- 尚未證明 KG 能改善 RL/portfolio 的 return、Sharpe 或 max drawdown。

## 決策與下一步

**Decision：implementation / pivot。** metadata 與 source-text gates 通過，但 semantic supervised gate 尚未開始。下一步只做 bounded event extraction：先從 `DEF 14A` peer-group sections 與 `8-K` 可辨識的 event sections 抽取事件，要求每筆 relation 有 evidence sentence、source URL、`known_as_of` 和 confidence；再以 `real`、`self/price-only`、`relation shuffle` 跑 h20/h60/h120 supervised gate。只有 real 在預先指定的 paired seeds 規則下勝過 controls，才進 slow-policy/hierarchical RL。

相關 metadata 紀錄：`docs/2026-08-10_SEC_filing_manifest_temporal_data_gate.md`。
