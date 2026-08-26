# DEF 14A compensation-peer event manifest 紀錄

日期：2026-08-10  
專案：`GraphRagTrading/0601_atten_V2`  
Branch：`experiment/gpm-kg-consumer`

## 研究問題

bounded SEC source text 已經保存，但全文不等於 KG。這一步用既有的保守 DEF14A peer-group parser，把可追溯的「補償同儕群組」事件抽成 temporal manifest；relation 明確命名為 `compensation_peer`，不把它偷換成 `competes_with`。

## 設計

- 輸入：207 筆候選 `DEF 14A` 與本地全文。
- 只掃描 `peer group`、`our peers`、`compensation comparison group` 等 anchor window。
- 只接受 Dow 30 已知公司 alias；排除 filing company 自己。
- 每筆事件保留：`source_ticker`、`target_ticker`、`relation_type`、`event_type`、`polarity`、`filed_date`、`known_as_of`、accession、SEC URL、本地檔案、evidence quote、confidence、extractor version。
- `known_as_of` 固定等於 `filing_date`；不使用 report/period end 作為可用時間。

執行腳本：`setup/41_extract_def14a_peer_event_manifest.py`。

## 結果

輸出：`artifacts/sec_event_manifest_2019_2025_pilot/`。

| 檢查 | 結果 |
|---|---:|
| candidate filings | 207 |
| filings with text | 207 |
| filings with peer windows | 202 |
| event rows | 799 |
| unique company pairs | 135 |
| unique source accessions | 157 |
| source tickers with events | 26 |
| evidence quote missing/empty | 0 |
| `known_as_of != filing_date` | 0 |

事件年度分布：2019 `99`、2020 `114`、2021 `107`、2022 `111`、2023 `109`、2024 `121`、2025 `138`。

## 證明了什麼

- 已有跨 30 檔、7 年的可追溯 temporal relation candidates，不再是前一輪只有 4 個 IBM cluster pairs 的資料量。
- 每個 relation 都能回到 SEC accession、filing date 與 evidence sentence/window，後續可做人工抽查與 point-in-time graph snapshot。

## 沒有證明什麼

- `compensation_peer` 不必然等於競爭、供應或客戶關係；它是明確的 proxy peer disclosure。
- rule-based alias/window extraction 仍可能有 false positive；799 rows 是候選事件，不是人工 verified ground truth。
- 尚未測試 h20/h60/h120 報酬預測，也尚未測試 portfolio/RL 經濟價值。

## 下一個 gate

先做 event-manifest quality audit（抽查 evidence、排除 definitive additional materials、確認事件在 cutoff 前可用），再建立依 `known_as_of` 動態 graph snapshots。比較 `real`、`price-only/self`、`topology/time shuffle`，以長期 target `{20, 60, 120}` 跑 supervised gate；5 個 paired seeds 全部勝出才進 10-seed confirmation。supervised gate 未通過前不跑 hierarchical RL。
