# SEC filing metadata：時間事件資料 Gate

日期：2026-08-10  
專案：`GraphRagTrading/0601_atten_V2`  
Branch：`experiment/gpm-kg-consumer`

## 1. 為什麼做這個 gate

前一個 temporal-event pilot 只有 16 筆 manifest rows，最後只有 4 個可配對事件，而且都集中在 IBM 的同一個 filing cluster。那個結果只能證明時間對齊與 target 計算流程可運作，不能證明事件 KG 有足夠的樣本或可泛化訊號。

因此先不下載全文，也不修改 RL；先建立 SEC submissions metadata manifest，確認資料是否能支撐跨公司、跨年度的事件抽取與 supervised gate。

## 2. 實驗設計

- Universe：Dow 30，30 檔股票。
- 日期：2019-01-01 至 2025-12-31。
- Form：`8-K`、`10-K`、`10-Q`、`DEF 14A`。
- 資料：SEC submissions metadata（recent 與 historical submission JSON），不下載全文。
- 可用時間：`known_as_of = filing_date`；`report_date`/period end 不可當作市場已知時間。
- 去重：`ticker + accession + form`；同時檢查 accession 唯一性。

執行腳本：

```bash
cd /home/hankdd/NYCU/GraphRagTrading/0601_atten_V2
PYTHONPATH=. /home/hankdd/.venv/bin/python \
  setup/38c_collect_sec_filing_manifest_full_fixed.py \
  --delay 0.15 \
  --output-dir artifacts/sec_filing_manifest_2019_2025_full_fixed
```

主要輸出：`artifacts/sec_filing_manifest_2019_2025_full_fixed/filings.csv` 與 `summary.json`。

## 3. 結果

| 檢查 | 結果 |
|---|---:|
| filing rows | 4,018 |
| unique accessions | 4,018 |
| unique tickers | 30/30 |
| ticker-year coverage | 210/210（每檔每年都有資料） |
| 8-K | 2,971 |
| 10-Q | 630 |
| 10-K | 210 |
| DEF 14A | 207 |
| collection errors | 0 |

年度筆數為：2019 `590`、2020 `629`、2021 `613`、2022 `564`、2023 `543`、2024 `540`、2025 `539`。每一檔股票的第一筆 filing 都落在 2019 年，最後一筆落在 2025 年；各 ticker-year 沒有空缺。

## 4. 這個結果證明什麼

1. SEC metadata 的時間覆蓋已足以支撐下一階段的跨公司、跨年度事件資料建置。
2. `filing_date`、`known_as_of`、accession、primary document 與 source URL 已被固定保存，後續可以做 point-in-time cutoff。
3. 事件候選量不是前一輪 pilot 的瓶頸：`8-K + DEF 14A = 3,178` 筆，足以先抽取一個有界的全文 pilot。

## 5. 這個結果沒有證明什麼

- 沒有證明 filing 內容包含正確的公司關係或事件極性。
- 沒有證明事件對未來報酬有預測力。
- 沒有證明 KG 能改善 GPM/RL 的 return、Sharpe 或 max drawdown。
- 仍可能有同一事件多次披露、公司公告類型混雜，以及全文 section/句子抽取品質問題。

## 6. 決策與下一個 gate

**Decision：implementation / pivot，metadata gate 通過；不直接進 RL。**

下一步先用固定且可重現的 bounded pilot：每個 ticker-year 保留全部 `DEF 14A`，再從 `8-K` 按日期等距選最多 2 筆，共約 627 筆候選。選取腳本：
`setup/39_select_sec_event_candidates.py`；輸出：
`artifacts/sec_event_candidates_2019_2025_pilot/event_candidates.csv`。

全文下載與抽取後，必須先跑：

1. `real event` vs `price-only/self` vs `relation/type shuffle`；
2. target horizons `{20, 60, 120}` 的 point-in-time supervised gate；
3. 5 個 paired seeds，預先指定至少 5/5 勝出才進 10-seed confirmation；
4. 只有 supervised gate 同時勝過 controls，才做 slow-policy 或 hierarchical RL。

若 relation shuffle 和 real 一樣好，停止宣稱 semantic relation value，回頭檢查事件類型、證據句與 target alignment。
