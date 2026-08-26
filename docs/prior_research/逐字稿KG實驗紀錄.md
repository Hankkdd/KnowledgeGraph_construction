# 逐字稿(Earnings Call Transcript)KG 實驗紀錄

日期:2026-07-21
範圍:5 家公司(AAPL、NVDA、AMZN、MSFT、AMD)、各 4 季(2019Q4~2020Q3)逐字稿
資料庫:`run_id 20260721_035555_af1f9a` → `snapshot_id 4`(最終版)→ `pykeen_run_id 2`

## 背景與動機

10-K 財報這條路遇到兩個結構性問題:

1. 財報常自稱「the Company」「we」而非公司全名,GraphRAG 抽不出公司對公司的直接關係。
2. 單一年度、無 point-in-time 時間戳記。

改用法說會逐字稿(`Transcripts/{ticker}/YYYY-Mon-DD-{ticker}.txt`)测試,因為:

- 分析師問答形式,公司會直接點名對手/供應商/客戶,不像 10-K 打太極。
- 檔名自帶精確日期,`filed_date` 直接可用,point-in-time 篩選立刻生效,不用等資料蒐集。
- AMD 用來暫代 IBM(逐字稿資料庫沒有 IBM)——**這只是這次實驗的暫代,正式版本仍維持道瓊 30 成分股、用 IBM**。

## 資料前處理

逐字稿本身有兩塊噪音,加了 `clean_transcript_text()`(`meta/graphrag/financial_reports.py`)處理:

- 開頭的 Thomson Reuters 版權/品牌區塊 + 分析師參與者名單(含任職券商)。
- 結尾的免責聲明(有固定錨點 `"Thomson Reuters reserves the right to make changes"`,兩種逐字稿格式都適用)。

砍掉後保留 91~95% 內容,drop 的部分主要是不含商業資訊的樣板文字。

## 抓到的兩個真實 Bug

### Bug 1:Projection 分組條件過嚴(`meta/graphrag/kg.py`)

`_build_projected_ticker_edges` 原本用 `groupby(["normalized_concept", "relation"])`——不只要求兩家公司連到同一個概念,還要求**關係措辭完全一致**才算連結。但 LLM 對同一件事的措辭本來就會不同(例如 `organization_serves_market` vs `organization_participates_in_market` 都是「涉足這個市場」)。

**實測影響**:AMD/NVIDIA 明明都連到 `DATA CENTER MARKET`、`GAMING INDUSTRY`、`TSMC`,但因措辭不同被拆到不同組,各自只有 1 家公司,永遠過不了「至少 2 家」的門檻——修復前這 5 家公司之間的 projected edges 是 **0 條**。

**修法**:改成只用 `groupby("normalized_concept")`,組內用最常見的關係字串當代表性標籤。

已加回歸測試:`tests/test_ticker_graph_projection.py`。

### Bug 2:分析師/券商名稱污染(`meta/graphrag/kg.py`)

逐字稿主持人會介紹「下一位提問來自 Morgan Stanley 的某某」,GraphRAG 把這類句子抽成看似公司關係的三元組(如 `company_analyses_company`、`organization_provides_research_for_organization`)。分析師任職的券商(Morgan Stanley、JPMorgan、Goldman Sachs 等)因此變成連接不相關公司的假「共享概念」。

**修法(兩層)**:
1. `_PROJECTED_RELATION_STOP_TERMS` 加入 `company_analyses_company`、`person_employed_by_organization`、`organization_employs_person`、`company_interacts_with_institution`。
2. `_PROJECTED_CONCEPT_STOP_SUBSTRINGS` 加入約 30 家已知券商/投行名稱(Morgan Stanley、Goldman Sachs、UBS、Credit Suisse、Barclays、Cowen、Jefferies 等)——概念層級擋比逐一擋關係字串更穩健,不受 LLM 用詞變化影響。

也發現:即使 `settings.yaml` 的 `entity_types` 已經拿掉 `product`/`person`/`event`,LLM(qwen2.5:14b)仍然抽出了這些型別的實體(PRODUCT 177 個、PERSON 54 個、EVENT 21 個)——**entity_types 設定對這個模型不是硬約束,只是提示**,這點在解讀「型別限制有沒有生效」時要留意。

## 修復前後對比(這 5 家公司之間的連結)

| 版本 | direct edges | projected edges(directed) | 連通的公司對 | 備註 |
|---|---|---|---|---|
| 未修復(僅 stop-relation 過濾前) | 1 | 0 | 0/10 | groupby 過嚴,真訊號被誤殺 |
| 修 Bug 1 後 | 1 | 46 | 8/10 | 但混有 3~4 對假的券商連結 |
| 修 Bug 1+2 後(最終) | 1 | 30 | **7/10** | 全部可驗證為真實產業/供應鏈關係 |

最終 7 對連結(皆有實質業務意義):

- AMD–NVDA:CLOUD GAMING MARKET、GAMING INDUSTRY、ARM、AWS、CHINA、DATA CENTER MARKET(GPU/CPU 直接競爭)
- AMD–AMZN、AMZN–NVDA:皆透過 AWS(供應鏈客戶關係)
- AMD–MSFT:CHINA、MICROSOFT AZURE(AMD 供應 Azure)
- MSFT–NVDA:SECURITY、MINECRAFT、CHINA
- AAPL–AMD、AAPL–AMZN:較弱(IBM、JAPAN,偏地理/泛用參照)

只剩 AAPL–MSFT、AAPL–NVDA 沒連上。

## KG 驗證套件結果(`setup/10_validate_kg.py`,`snapshot_id 4`)

- 關係型別數:330 種,accept rate 79.2%
- **連通度:0/5 孤立、最大連通分量 5/5(全連通)**
- **Link prediction:MRR 真實 KG = 0.0190,shuffled 對照組 = 0.0078**(真實 KG ≈ 對照組的 2.4 倍,方向正確但絕對值仍偏低,樣本量小(812 條 triples、330 種關係型別)是主因之一)
- Sector test / Mantel test:因 AMD 缺股價與產業別資料,自動跳過(預期中,AMD 是暫代不影響結論)
- Known-pairs recall:目前 0(`data/known_company_pairs.csv` 尚未人工審核 `verified=true`)

## 待辦 / 下一步

- [ ] `data/known_company_pairs.csv` 人工審核,補上 `verified=true`,讓 known-pairs recall 真正生效
- [ ] 考慮擴大到 9 家半導體供應鏈公司(AAPL/AMD/AMZN/ASML/CSCO/INTC/MSFT/MU/NVDA),看連結密度會不會進一步提升
- [ ] 正式版本改回道瓊 30 成分股 + IBM(AMD 僅為此次實驗暫代)
- [ ] `setup/08`、`09` 尚未加 `--tickers` 覆寫支援(這次驗證流程繞過了它們,直接用 `setup/10`)
- [ ] L1 層級(LLM 抽查邊忠實度、跨模型/跨 seed 抽取穩定性)仍未處理,留待資料蒐集方法定案後一併處理
