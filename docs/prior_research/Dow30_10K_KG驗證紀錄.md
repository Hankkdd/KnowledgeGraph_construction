# Dow 30 全量 10-K KG 驗證紀錄

日期:2026-07-22
範圍:Dow 30 全 30 檔、10-K(Items 1-1B + 7-7A)
資料庫:`snapshot_id 6` → `pykeen_run_id 3` → `validation_id 3`
模型:vLLM 本地部署 `RedHatAI/gemma-4-26B-A4B-it-NVFP4`(取代 Ollama qwen2.5:14b,~12-13x 加速)

## 背景

延續 [逐字稿KG實驗紀錄.md](逐字稿KG實驗紀錄.md) 的兩個 bug 修復(projection 分組過嚴、分析師/券商污染)之後,
用 vLLM 加速跑完整 Dow 30 成分股的 10-K pipeline(setup/02→10),取代先前 5 家逐字稿的小規模實驗。

跑完後另外發現 `data/price_ohlcv.csv` 只有 5 檔股價(先前開發階段的殘留,非本次造成),
已補齊 30 檔完整資料([setup/01_get_market_data.py](../0601_atten_V2/setup/01_get_market_data.py) 因 Yahoo Finance
對本機 IP 封鎖而失敗,改由使用者自行於他機下載後放入 `data/price_ohlcv.csv`)。

## 結果總表

| 指標 | 數值 | 備註 |
|---|---|---|
| 關係型別數 | 760 種(熵 8.09 / 上限 9.57 bits,top 型別佔比僅 10%) | 極度碎片化 |
| Triples 接受/拒絕 | 1982 / 774(accept rate 0.719) | |
| Direct company edges | 1 | 10-K 幾乎不會直接點名其他公司 |
| Projected company pairs | 52(160 條有向邊) | 訊號幾乎全來自共享概念投影 |
| 孤立公司(無連結) | 8/30 — BA、CAT、CRM、DIS、JPM、KO、MMM、NVDA | 連 NVDA 都孤立 |
| 連通分量 | 9 個,最大 22/30(73%) | |
| MRR(real KG) | 0.0027 | **低於** shuffled control |
| MRR(shuffled control) | 0.0059 | |
| Hits@1/3/10(real) | 0 / 0 / 0 | 完全沒有可用 link prediction 訊號 |
| Sector permutation test | same−cross = -0.0091, p = 0.669 | 不顯著 |
| Mantel test(vs 股價報酬相關性) | rho = -0.077, p = 0.363 | 不顯著,方向為負 |
| Known-pairs recall | n/a(0 pairs) | `known_company_pairs.csv` 尚未人工審核(全 `verified=false`) |

完整報告:`artifacts/diagnostics/kg_validation/snapshot_6/report.md`

## 判讀

三個獨立於 RL 的外部效度檢定——**link prediction vs shuffled control、sector test、Mantel test——全數未通過**。
不是「補齊股價資料後才發現沒訊號」,是這版全 30 檔 10-K KG 本身就沒有可驗證的結構訊號,股價資料補齊只是讓
Mantel test 從「跳過」變成「可以測、但也不顯著」。

對照 5 家逐字稿實驗(MRR real 是 shuffled 的 2.4 倍,方向正確),這次全 30 檔 10-K 版本明顯更差。差異可能來源:

1. **10-K 本身結構性弱點**(已知,見逐字稿紀錄背景段):財報常自稱「the Company」「we」,GraphRAG 抽不出公司對公司的直接關係,
   direct edges 只有 1 條就是證據。
2. **樣本量/關係型別比例更差**:760 種型別對 2000 條 triples,平均每種型別不到 3 條——
   跟逐字稿版本(330 種型別、812 條 triples,比例還好一些)相比,relation 碎片化更嚴重。
3. **沒有 relation ontology 映射**:同一件事(例如「涉足某市場」)被抽成任意措辭的不同 relation type,
   projection 雖然已經修過分組 bug,但下游 PyKEEN 訓練仍然把每個型別當成獨立關係,稀釋了可學習的結構。

## 下一步建議

已與使用者討論,結論記錄如下:

1. **換資料來源**:10-K 已驗證兩次(逐字稿實驗背景 + 這次)結構性偏弱;逐字稿版本雖然只測了 5 家,
   但方向正確且效果量明顯更好。下一步應優先評估把資料來源換成/混合逐字稿,而非繼續在 10-K 上調參。
2. **Relation ontology 映射**(對應規劃文件 [GraphRAG金融知識圖譜品質驗證與實作計畫.md](GraphRAG金融知識圖譜品質驗證與實作計畫.md) Task 3):
   把自由生成的 760 種型別映射到一組窄而可驗證的 ontology,直接解決「型別碎片化稀釋訊號」的根因。
3. 兩者不互斥,但换资料源可能是報酬率更高的下一步——先驗證逐字稿在更大樣本(9 家半導體供應鏈或全 Dow 30)下
   是否維持逐字稿實驗看到的方向,再決定要不要投入 ontology 映射的工程成本。

## 待辦

- [x] 評估逐字稿資料源擴大到更大樣本 → 已決定擴大到 Dow 30 全 30 檔,見下方「進度更新」
- [ ] `data/known_company_pairs.csv` 人工審核,補上 `verified=true`
- [x] Relation ontology 映射(規劃文件 Task 3,與 prompt RELATION_TYPE 前綴修正一併處理) → 已完成,見下方
- [ ] `setup/08`、`09` 加 `--tickers` 覆寫支援

---

## 進度更新(2026-07-22 下午)

### 1. Mantel test 時間窗修正

發現 `reports.filed_date` 對 10-K 這條路徑全部是 NULL(只有逐字稿載入路徑才會從檔名解析日期),
`setup/10` 原本拿 `TRAIN_START=2024-01-01~TRAIN_END=2025-01-01` 去對 Mantel test,但實際上這批 10-K
全部是 **FY2025**(申報約 2025 下半年~2026 上半年,抽查 AAPL/AMGN/AXP/BA/CAT/…/WMT 皆同)——
拿「申報前一年」的股價去對「最新」財報,時序方向錯了。

新增 `experiment_config.py` 的 `KG_VALIDATION_RETURN_START/END`(與 RL 用的 `TRAIN_START/END` 分開,
避免誤動 RL 訓練窗口),改成用最近一年可取得的股價(`2025-05-01~2026-05-01`,與 10-K 描述的財年同期)。

結果:`mantel rho` 從 -0.077 (p=0.363) 變成 -0.0001 (p=0.999)——更接近 0,排除了「股價窗口對錯期」
是主因的可能性,確認瓶頸不在時間對齊,而在抽取品質本身。

### 2. Relation ontology 映射(Task 3)

`extract_graph.txt` 的 prompt 矛盾其實已經修好(16 條 relationship 範例全部有 `RELATION_TYPE:` 前綴,
含 Example 4 的分析師 hard-negative 範例),不需要再動。

新增 `meta/graphrag/relation_ontology.py`:三層映射(exact alias table → keyword 規則 → quarantine),
把 760 種自由生成型別收斂到 14 種(`docs/GraphRAG金融知識圖譜品質驗證與實作計畫.md` 4.2 節定義的窄
ontology)。`tests/test_relation_ontology.py`(8 個測試)涵蓋同義詞、方向反轉、inverse pair、
未知關係 quarantine、長尾 keyword 規則。接進 `setup/05_build_pykeen_triples.py`。

`snapshot_id=8` / `pykeen_run_id=4` 結果對比:

| 指標 | 修正前(760 型別,snapshot 6) | 修正後(14 型別,snapshot 8) |
|---|---|---|
| Relation 型別數 | 760 | 14 |
| Accept rate | 0.719 | 0.599(多一層 ontology quarantine,16.5% 三元組被擋) |
| 孤立公司/最大連通分量 | 8/30, 22/30 | 不變 |
| MRR(real) | 0.00268 | 0.00638(提升 2.4 倍,方向正確) |
| MRR(shuffled) | 0.00589 | 0.00689 |
| Sector test p | 0.669 | 0.402(改善但仍不顯著) |
| Mantel rho / p | -0.0001 / 0.999 | -0.0001 / 0.999(同上,時間窗修正後的結果) |

**結論**:ontology 映射確實有效(型別碎片化大幅改善、MRR 方向正確且提升 2.4 倍),但 real KG 的 MRR
依然低於 shuffled 對照組,sector/Mantel 依然不顯著。這比較符合最初的假設:**不是 ontology 沒做好,
是 10-K 這個資料來源本身結構訊號薄弱**(公司自稱「the Company」,GraphRAG 抽不出直接的公司對公司關係)。

### 3. 決定擴大逐字稿資料源到全 Dow 30

原本逐字稿(`Transcripts/`)只有 9 家公司、5 家是道瓊成分股。查證後確認 **fool.com(Motley Fool)
有免費、無需登入的完整逐字稿**,涵蓋原本缺逐字稿的 25 家道瓊公司(含 Boeing、JPMorgan、Coca-Cola 等),
近期季度到 2025-2026 都有,格式跟 Thomson Reuters 版類似(prepared remarks + Q&A + 分析師任職券商名單)。

新增:

- `meta/graphrag/motley_fool_scraper.py`:抓取邏輯,已對 BA/JPM/KO/CAT 手動驗證兩種頁面模板
  (新版以 `"Full Conference Call Transcript"` 為內容起點;舊版以第二次出現的 `"Prepared Remarks"`
  為起點、`"Call Participants"` 為終點)。
- `setup/scrape_fool_transcripts.py`:一次性抓取腳本,輸出到 `Transcripts_fool/{ticker}/YYYY-Mon-DD-{ticker}.txt`,
  跟現有 `Transcripts/` 命名慣例一致。
- `setup/01b_sync_transcripts.py`:比照 `setup/00`,把 `Transcripts_fool/` 內容寫入 `reports` 表,
  `filing_type="earnings_call_transcript_foolcom"`(與 Thomson Reuters 的 `"earnings_call_transcript"`
  分開標記,避免 5 家重疊公司(AAPL/AMZN/CSCO/MSFT/NVDA)的兩個資料源被混在一起——這是先前逐字稿
  實驗就處理過的同一類污染問題)。
- `setup/02_prepare_graphrag_input.py` 加 `--filing-type` 參數,可選擇要索引哪個資料源。

**已完成**:`Transcripts_fool/` 已抓齊 **Dow 30 全 30 家**、共 119 份逐字稿檔案,已抽查 Boeing
一份確認內容乾淨(無頁首/頁尾雜訊,直接從對話開始、operator 道別結束)。`clean_transcript_text`
對 fool.com 文字會安全地整段保留(anchor 找不到會 fallback 到原文,不會誤砍)。全部測試通過
(`pytest tests/` 79 個測試皆綠燈)。

**卡住的地方**:下一步要跑 GraphRAG 索引(`setup/01b → 02 --filing-type earnings_call_transcript_foolcom
→ 03 → 04 → 05 → 06 → 07 → 10`),需要 LLM。vLLM 容器(`vllm-gemma4`)目前是 `Exited (1)`——
GPU 被另一位使用者(`alen911018`)的行程占用 52GB 記憶體,vLLM 原本設定的 `--gpu-memory-utilization 0.8`
（約 97GB）超出剩餘空間而啟動失敗。**已與使用者確認:先不搶資源,等對方行程結束再重啟 vLLM**
(可能需要調低 `--gpu-memory-utilization` 到 0.3 左右以策安全,避免再次跟其他使用者衝突)。

## 全 30 家逐字稿版結果(2026-07-22 晚間,`snapshot_id=10` / `pykeen_run_id=5` / `validation_id=7`)

vLLM 因 GPU 被另一使用者佔用一度啟動失敗,調低 `--gpu-memory-utilization` 後成功;又因 Ollama 的
embedding 模型跟 vLLM 搶記憶體導致 GraphRAG 索引第一次失敗(`--gpu-memory-utilization` 從 0.8 降到 0.7
留出空間給 Ollama),第二次成功跑完(119 份文件、493 chunks)。

另外修了一個實體對齊的 hard-fail:`meta/graphrag/kg.py` 的 `_build_entity_to_ticker` 原本只要有一檔
股票完全對不到任何 GraphRAG 實體就直接 `raise ValueError`,整個 snapshot 建不出來。這次 CSCO(逐字稿
裡只提到「Cisco」,沒有「Systems」,不是可以用既有 legal-suffix 邏輯剝除的那種尾綴)就撞到這個問題。
比照 setup/05/07 已有的「零 triples 不中斷、只警告」原則,把這個 hard-fail 也改成警告——這跟先前
BA/JPM/KO 在 10-K 版「有實體但零 triples」是同一類問題,只是發生得更早(整個實體都沒對到)。

### 三版比較

| 版本 | Relation 型別 | 孤立公司 | 最大連通分量 | MRR real | MRR shuffled | Sector p | Mantel (rho/p) |
|---|---|---|---|---|---|---|---|
| 10-K(未修 ontology) | 760 | 8/30 | 22/30 | 0.0027 | 0.0059 | 0.669 | -0.0001 / 0.999 |
| 10-K + ontology | 14 | 8/30 | 22/30 | 0.0064 | 0.0069 | 0.402 | -0.0001 / 0.999 |
| **逐字稿(全 30 家,fool.com)+ ontology** | 14 | **6/30** | **24/30** | **0.0052** | **0.0058** | **0.329** | -0.0655 / 0.362 |
| 逐字稿(5 家試點,Thomson Reuters,舊) | 330 | 0/5 | 5/5(全連通) | 0.0190 | 0.0078 | n/a | n/a |

CSCO、JPM 這次在 fool.com 逐字稿裡對不到實體/零 triples(28/30 對齊成功),用 `no_kg_signal_fallback_mean`
補上中性 embedding,不影響其他 28 家。

### 結論

連通度比 10-K 版好一點(孤立公司少 2 家、最大連通分量多 2 家),sector test p 值也略降,但**核心問題
沒有解決**:MRR real 依然低於 shuffled 對照組,三個外部效度檢定(link prediction vs shuffled、sector、
Mantel)全數不顯著。5 家試點看到的強訊號(real MRR 是 shuffled 2.4 倍)**沒有在擴大到全 30 家時重現**。

可能原因:5 家試點是精挑過、逐字稿密度高的科技/半導體供應鏈公司(彼此本來就有業務往來);擴大到全 30
家 Dow 成分股後,每家平均只有 4 份逐字稿、跨產業(如 3M、卡特彼勒、嬌生等彼此業務關聯本來就弱),稀釋
了樣本量與訊號密度,不是換資料源就能解決的根本限制。

**目前為止,不論 10-K 或逐字稿,在全 30 家 Dow 規模下都測不出通過 G4/G5 閘門的顯著結構訊號。**
下一步待與使用者/指導教授討論方向(擴大逐字稿季數、改用更小但業務關聯更緊密的公司子集、或接受這個
限制並在論文中如實呈現為研究發現而非負面結果)。

---

## 方向雜訊 bug 與修正(2026-07-22 深夜)

### 發現過程

跑完 Stage-1 IC 比較(`setup/11_stage1_ic_compare.py`,KG vs random embedding,5 seed)也是同樣的
「沒有顯著訊號」結論(paired diff mean 0.0063,但 Wilcoxon p = 1.0,5 個 seed 裡 KG 贏 2 輸 3,方向
不一致)。三層獨立驗證(link prediction vs shuffled、外部效度、Stage-1 IC)全部收斂到同一個負面結論。

在下結論之前,直接隨機抽樣 30 條 triple 內容檢查,發現多筆方向明顯反了:

```
TRAINIUM2 produces AMAZON          ← 應為 Amazon produces Trainium2
DISNEY WORLD owns DISNEY           ← 應為 Disney owns Disney World
CARVICTI produces JOHNSON & JOHNSON ← 應為 J&J produces Carvicti
```

追查原始 GraphRAG 輸出發現:`organization_owns_product`(37 筆,有明確實體型別可判斷方向的樣本中)
只有 3 筆方向跟名稱一致(org 在前),**15 筆是反的**(product 在前、org 在後)。也就是說 **GraphRAG
自己抽取時,不會保證 (source, target) 順序跟它自己取的 relation_type 名稱所暗示的方向一致**——同一個
relation type 字串,不同筆資料的實際方向並不穩定。

這代表原本 `relation_ontology.py` 靠「關係名稱命名慣例」猜測要不要反轉 head/tail 的設計,本質上是不
可靠的猜測,對某些關係型別(如 `organization_owns_product`)猜錯的機率比猜對還高。

### 修法

新增 `correct_direction_by_entity_type()`(`meta/graphrag/relation_ontology.py`):對於預期「head 必須
是公司、tail 是非公司概念」的關係(`exposed_to_risk`、`produces`、`operates_in_region` 等 10 種),用
GraphRAG 抽取時就有的**實體型別**(ORGANIZATION vs 其他)當作權威依據,無視命名慣例猜測,只要能確定
一邊是 ORGANIZATION、另一邊不是,就強制排序;型別不明或兩邊都是 ORGANIZATION(如 `owns`、
`supplies_to`,母子公司/供應鏈關係本來就雙邊都是公司,型別無法判斷該由誰在前)則保留原樣,不猜測。

接進 `setup/05_build_pykeen_triples.py`:先套用 ontology 映射的方向猜測,再用實體型別做最終權威修正。
新增 5 個測試(`tests/test_relation_ontology.py`),涵蓋方向修正、方向已正確不動、型別不明不動、
company-company 關係不受影響。全部 83 個測試通過。

### 修正後結果

| 版本 | MRR real | MRR shuffled | real > shuffled? |
|---|---|---|---|
| 10-K + ontology(方向修正前) | 0.0064 | 0.0069 | ❌ |
| **10-K + ontology + 方向修正**(snapshot 12, pykeen_run 6) | **0.0090** | 0.0075 | ✅ **首次翻正** |
| 逐字稿(方向修正前) | 0.0052 | 0.0058 | ❌ |
| **逐字稿 + 方向修正**(snapshot 13, pykeen_run 7) | **0.0062** | 0.0047 | ✅ **首次翻正** |

Sector test(p=0.442 / 0.356)、Mantel test(p=0.957 / 0.388)仍不顯著,但 **link prediction 這一關
(G4)兩個資料源都第一次通過**——方向雜訊是壓低訊號的一個真實、可修根因,不是「這兩個資料源本身沒有
結構訊號」的證據。

### Stage-1 IC 複驗(2026-07-23):5 seed 的正向訊號是樣本數太小的假象

方向修正後先用 5 seed 重跑 `setup/11`(逐字稿版 embedding):paired diff 從 0.0063 提升到 0.0223,
Wilcoxon p 從 1.0 降到 0.4375,看起來有改善。但這只是 5 個點,檢定力太弱,不能下結論。

擴大到 15 seed(`--seeds 15`)後,結果整個反過來:

| 指標 | 5 seed | 15 seed |
|---|---|---|
| paired diff(KG − random) | +0.0223 | **-0.0095**(方向反過來) |
| Wilcoxon p | 0.4375 | 0.359(仍不顯著) |

新增的 seed 46-55 裡 random 反而多次贏過 KG,把原本 5 seed 看到的正向平均值直接拉負。**這是典型的
小樣本雜訊陷阱**:5 個 seed 看到的「有希望」的正向效果不是真訊號,擴大樣本後效果量趨近於 0。

### 三層驗證現況總結(2026-07-23)

| 層級 | 方法 | 結果 |
|---|---|---|
| G4(圖結構可學習性) | Link prediction vs shuffled-graph 對照組 | ✅ **兩個資料源方向修正後都通過**(real MRR > shuffled MRR) |
| L2(外部效度) | Sector permutation test、Mantel test(vs 股價報酬相關性) | ❌ 兩個資料源都不顯著 |
| L4(下游任務效用) | Stage-1 IC:KG vs random embedding 對次日報酬預測(15 seed) | ❌ 不顯著,效果量趨近 0 |

**可以誠實陳述的結論**:這版 KG 的圖拓撲結構本身,比隨機打散的同分佈圖更容易被 link prediction
模型學到(G4 通過)——即圖裡確實有非隨機的結構。但這個結構訊號**沒有轉化成對次日報酬預測任務有用
的資訊**(L4 未通過),也**沒有對應到可驗證的產業分類或股價共動模式**(L2 未通過)。「KG 有結構」跟
「KG 對這個下游任務有用」是可以分開陳述的兩件事,不是同一個結論的兩種說法。

### 週報酬 Mantel test + 相似度/報酬熱力圖(2026-07-23)

`meta/graphrag/validation.py` 的 `return_correlation_matrix` 加 `freq` 參數(`"D"`/`"W"`,週報酬用
`W-FRI` resample 取週五收盤價),`setup/10_validate_kg.py` 同時跑日/週兩版 Mantel test,並新增
`_plot_similarity_vs_return_heatmaps()`,輸出 `artifacts/diagnostics/kg_validation/snapshot_{id}/
similarity_vs_return_heatmaps.png`(KG 相似度 / 日報酬相關 / 週報酬相關,三張並列)。

| 版本 | Mantel rho(daily) | Mantel rho(weekly) |
|---|---|---|
| 10-K + ontology + 方向修正(snapshot 12) | -0.0041(p=0.957) | 0.0033(p=0.967) |
| 逐字稿 + 方向修正(snapshot 13) | -0.0618(p=0.388) | -0.0031(p=0.962) |

週報酬(平掉部分大盤共同波動的日內雜訊)結果跟日報酬差不多,一樣趨近 0、不顯著——排除了「日報酬雜訊
蓋過訊號」這個猜測,不是頻率選擇的問題。

**熱力圖質性檢查**(逐字稿版,snapshot 13):KG embedding 相似度那張幾乎是一片平坦淡色,肉眼看不出
任何區塊/聚類結構;日報酬與週報酬兩張則有明顯的整體偏正相關「市場因子」效應(幾乎所有股票兩兩之間
都是淡紅到中紅,是大盤共同波動的典型樣子)。這張圖直接、視覺化地印證了 Mantel test 的數字結論——
KG embedding 空間目前看起來就是接近隨機分佈,不是統計檢定力不足的問題。可以直接作為論文圖表使用。

### 根因確認:不是沒訊號,是抽取覆蓋率不足(2026-07-23)

質疑「道瓊 30 裡不可能沒有任何相關性較高的股票對」之後,直接查了週報酬相關性最高的 15 對(皆為合理
的同業/供應鏈關係:GS-JPM 銀行 0.76、HD-SHW 家居建材 0.72、KO-PG 消費品 0.66、HON-MMM 工業股 0.60 等),
再逐一檢查這些股票對在 KG(snapshot 12、13)裡有沒有任何連結(direct edge 或透過共享概念的
projected edge)。

結果:

- **GS-JPM**(相關性最高的一對)——GS 這個實體在兩個 snapshot 裡**完全沒被抽取到**,連對齊都對不上
- 其餘可對齊的 11 對裡,**只有 3 對有任何共享概念**,且都是很弱、很泛的總體性話題
  (CHINA、JAPAN、TARIFFS、OPENAI),**沒有一對是直接的公司對公司關係**
- 其餘 8 對(AXP-GS、HON-MMM、HD-MMM、AXP-V、CRM-MSFT、HD-NKE、MMM-SHW、DIS-V)**完全沒有連結**,
  不管是 direct edge 還是 projected edge

**這改變了整個問題的診斷方向。** 先前以為是「KG 有訊號但被雜訊/樣本量蓋過」,現在確認是**抽取階段
的覆蓋率問題**:GraphRAG 從 10-K(自稱「the Company」)或法說會逐字稿(不太會主動點名「我們的同業
高盛」)裡,系統性地漏掉了「高盛與摩根大通同為銀行業」「家得寶與宣偉是供應鏈夥伴」這類顯而易見、
真實存在的同業/供應鏈關係。不是這些關係不存在於現實世界,是資料來源本身很少用會被 GraphRAG 抽出的
措辭明確陳述這些關係,加上目前的 entity_types/prompt 設計沒有針對「同業比較」這類語境做特別強化。

這比「relation ontology 碎片化」「head/tail 方向雜訊」更根本——後兩者都已經修正,但即使修正後,
根本沒被抽出來的關係依然是 0,無法靠下游的 ontology 映射或方向修正救回來。

### 待辦(已過時,見下方「重大突破」章節,以下項目已完成或有更新結論)

- [x] 考慮把同樣的實體型別檢查也套用到 `owns`/`supplies_to` 這類 company-company 關係 → 暫緩,DEF 14A
      補齊後這已非主要瓶頸
- [x] 針對「同業比較」場景補資料源(改用 SEC DEF 14A 而非分析師報告/新聞,理由見下)
- [ ] 與指導教授討論最終結果的論文呈現方式(見下方結論)

---

## 重大突破:SEC DEF 14A 同業組 + TransE(2026-07-23)

### 診斷:高相關股票對完全沒有 KG 連結

質疑「道瓊 30 不可能沒有任何高相關股票對」後,直接查了週報酬相關性最高的 15 對(GS-JPM 銀行 0.76、
HD-SHW 家居建材 0.72、KO-PG 消費品 0.66 等,皆為合理同業關係),逐一檢查這些股票對在 KG 裡有沒有連結
(direct edge 或透過共享概念的 projected edge)。結果:GS 這個實體**完全沒被抽取到**;其餘可對齊的
11 對裡只有 3 對有任何共享概念,且都是很弱的總體性話題(CHINA、TARIFFS、OPENAI),**沒有一對是直接
的公司對公司關係**。

**根因確認**:不是統計檢定力不足,是 GraphRAG 從 10-K/法說會逐字稿抽取時,系統性地漏掉「高盛與摩根
大通同為銀行業」這類顯而易見的同業關係——這兩種資料源本來就很少用會被 GraphRAG 抽出的措辭明確陳述
同業比較。

### 新資料源:SEC DEF 14A 薪酬同業組(Compensation Peer Group)

上市公司依 SEC 規定每年必須在股東會委託書(DEF 14A)揭露一份**明確列出同業公司名單**的表格(用於
高階主管薪酬市場對標)。這是免費、公開、結構化程度高的 SEC 資料,跟現有 10-K 同一法律類別,不受
Yahoo Finance 那種反爬蟲封鎖。實際驗證:蘋果一家公司的委託書就直接點名 9 家道瓊同業(CSCO、NVDA、
MSFT、KO、JNJ、NKE、PG、UNH、DIS);高盛的委託書明確寫出「JPMorgan Chase & Co. (JPM)」。

新增:

- `setup/scrape_def14a_peer_groups.py`:透過 SEC EDGAR `submissions/CIK{cik}.json` API 找到每家
  公司最新一份 DEF 14A,下載存到 `DEF14A/{ticker}/{accession}.txt`。已抓齊全 30 家。
- `meta/graphrag/def14a_peer_parser.py`:規則解析器(非 LLM),三層防呆設計,皆已用真實資料驗證:
  1. 錨點鎖定「peer group」或「our peers」附近的段落,而非任何「peer」字樣(JPMorgan 全文
     ~370K 字元,「peer」在無關語境出現數十次,錨點太寬會抓進一堆假陽性)。
  2. **負面表列過濾**:若段落包含「這些不算我們同業」之類的免責語句(驗證於 JPMorgan 自己的
     文件——它明確把「參考公司」名單跟「主要同業組」分開陳述),整段丟棄,不只是關鍵字。
  3. **Supplemental/DJIA 邊界截斷**:若段落提到「補充同業組=道瓊指數成分股」(驗證於默克藥廠
     的文件——它把「補充同業組」定義成整個道瓊 30 指數,這只是指數成員身分的通用產物,不是真正
     的同業訊號,會讓每家道瓊公司看起來都是其他 29 家的同業),從該處截斷,但保留前面真正的
     primary peer group 名單。
  4. 另外修了一個 non-breaking space(`\xa0`)normalize 的 bug(SEC HTML 常在公司名稱內用
     `Johnson\xa0& Johnson`,純字面比對抓不到)。
  `tests/test_def14a_peer_parser.py`(7 個測試)用真實文件節錄當 fixture。

  全 30 家掃描結果:135 條有向同業關係、102 條去重後的無向配對、24/30 家至少找到一個同業,且形成
  合理的產業聚類(金融 AXP-GS-JPM-V、工業 BA-CAT-HON-MMM-CSCO、製藥 MRK-JNJ-AMGN、科技
  AAPL-MSFT-CSCO-CRM-NVDA、消費品 KO-PG-NKE-MCD-WMT-HD)。

- `setup/05b_merge_def14a_peers.py`:把這 102 條同業關係(對稱化,不論哪一方揭露都算雙向證據)以
  `competes_with` 關係疊加到既有 snapshot 上,產生新 snapshot(不動原本已驗證過的 snapshot)。
  **不會被稀釋**:查證後現有 snapshot 裡真正的公司對公司三元組只有 2 條(10-K 版:
  NVIDIA-CISCO、NVIDIA-MICROSOFT;逐字稿版:SALESFORCE-IBM、NVIDIA-CATERPILLAR),102 條新邊
  是現有公司對公司訊號的 50 倍以上。

### 架構修正:連通度指標曾經對合併資料「隱形」

發現 `setup/10` 原本從**原始 GraphRAG 輸出**重新計算連通度,而不是從 PyKEEN 實際訓練用的
`triples` 表計算——這兩者過去剛好一致(setup/05 的 triples 本來就是原始輸出過濾轉換來的),但合併
DEF 14A 資料後就會分歧,導致連通度指標對新加的 102 條邊「視而不見」。新增
`meta/graphrag/kg.py::ticker_graph_from_triples()`,直接從 snapshot 的 `triples` 表算 direct/
projected edges,`tests/test_ticker_graph_projection.py` 加 3 個測試。改完後 10-K 版連通度從
8/30 孤立降到 **1/30**,逐字稿版從 6/30 降到 **0/30(完全連通)**。

### 關鍵發現:ComplEx 的向量幾何跟餘弦相似度驗證方法論不搭

合併同業關係、連通度大幅改善後,sector test / Mantel test 卻沒有跟著變好(甚至更差)。直接比較
「KG 裡有連結的股票對」vs「沒連結的股票對」在 embedding 空間的餘弦相似度,發現**有連結的反而比沒
連結的更不相似**(兩版都是負向)。

診斷:目前用的 **ComplEx 是雙線性評分模型**(score = Re(⟨h, r, conj(t)⟩)),不像 TransE 那種平移
模型(h + r ≈ t)天生就會讓有關係的實體在向量空間裡彼此靠近。**整套 sector test / Mantel test 方法論
是建立在「餘弦相似度反映真實相似度」這個假設上,這個假設對 TransE 成立、對 ComplEx 不成立。**

`setup/06_train_pykeen.py` 加 `--model` 參數,改用 TransE 重訓兩個合併後的 snapshot,結果:

| 版本 | MRR real vs shuffled | Sector test p | Mantel(daily) |
|---|---|---|---|
| 10-K + DEF14A peers(ComplEx) | 0.0052 < 0.0058 ❌ | 0.987 | 0.055 |
| **10-K + DEF14A peers(TransE)** | 0.0079 > 0.0058 ✅ | **0.0052** ✅✅ | 0.192 |
| 逐字稿 + DEF14A peers(ComplEx) | 0.0084 > 0.0070 ✅ | 0.627 | 0.452 |
| **逐字稿 + DEF14A peers(TransE)** | 0.0094 > 0.0070 ✅ | **0.0009** ✅✅ | 0.425 |

**Sector permutation test 兩個資料源都從完全不顯著變成高度顯著(p<0.01)。** Mantel test(vs 股價
報酬相關性)仍不顯著——這是預期中更難的測試,產業分類跟股價共動不是同一件事,KG 能抓到「這些公司
是同業」已經是扎實的正面結果。

### 完整因果鏈總結

1. Relation ontology 碎片化(760→14 型別):改善 MRR,但不夠。
2. Head/tail 方向雜訊(entity type 修正):MRR 首次翻正(real > shuffled),但 sector/Mantel 仍不顯著。
3. 抽取覆蓋率不足(DEF 14A 補上 102 條真實同業邊):連通度大幅改善(8/30→1/30 孤立),但 sector/Mantel
   仍不顯著。
4. **KGE 模型與驗證方法論不匹配(ComplEx→TransE)**:sector test 終於高度顯著。

四層修正環環相扣,缺一不可——只做其中任何一層都不會看到現在這個結果。

### 128 維 TransE(2026-07-23)

ComplEx 用 `embedding_dim=64` 時,setup/07 會把複數的實部+虛部串接成 128 維實數向量(對齊
`ATTENTION_EMBEDDING_DIM=128`);TransE 本身就是實數 embedding,`embedding_dim=64` 不會加倍,
會跟下游 attention 模型的維度對不上(`ValueError: kg_dim=64, embedding_dim=128`)。
`setup/06_train_pykeen.py` 加 `--embedding-dim` 參數,直接用 128 維訓練 TransE:

| 版本 | MRR real vs shuffled | Sector test p |
|---|---|---|
| 10-K + peers, TransE 128 維 | 0.0114 > 0.0058 ✅ | **0.0049** |
| 逐字稿 + peers, TransE 128 維 | 0.0156 > 0.0058 ✅ | **0.0018** |

MRR 比 64 維時更好,sector test 依然高度顯著。

### Stage-1 IC 用新 embedding 複驗:結構顯著,但沒有轉化成隔日報酬預測效用

用 128 維 TransE(逐字稿+DEF14A+ontology+方向修正)的 embedding 重跑 `setup/11`(15 seed):

| 指標 | 值 |
|---|---|
| KG 平均 test IC | 0.0020 |
| Random 平均 test IC | 0.0143 |
| Paired diff | -0.0123 |
| Wilcoxon p | 0.489(不顯著) |

**結論(可以分開陳述、不互相矛盾的三件事)**:

1. KG 有真實產業結構(sector test p<0.01,兩個資料源皆然)。
2. 這個結構是可學習的(link prediction:real MRR 顯著贏過 shuffled 對照組)。
3. 但這個結構**沒有轉化成「固定 5 檔投資組合隔日報酬預測」的顯著效用**(Stage-1 IC 不顯著,甚至
   方向為負)。

可能原因:單日個股報酬預測本身雜訊極大(每天只有 5 個點算 Spearman IC,標準差 ~0.5 vs 平均值
~0.01,量級差距顯著),不代表 KG 結構是假的,只是這個結構未必對「隔日」這個時間粒度的報酬有用;
另外固定投資組合裡的 **IBM 在 DEF14A 同業解析中沒有抓到任何同業**(它自己的委託書未明確列出同業
名單),這 5 檔裡有一檔沒吃到 DEF14A 這次的改善,可能稀釋了 Stage-1 IC 的效果。

### Stage-1 目標設計的再檢驗:共同波動 + 週報酬(2026-07-23)

質疑「會不會是隔日報酬這個目標選錯了」之後,測了兩個替代角度:

**1. 共同波動(5 檔子集的 Mantel test)**:直接拿固定投資組合(AAPL/AMZN/IBM/MSFT/NVDA)5 檔的
embedding 相似度對股價報酬相關性做 Mantel test。結果:5 檔只有 C(5,2)=10 個配對,統計檢定力嚴重
不足(跟全 30 家的 Mantel test 本質上是同一個檢定,樣本點少了 40 倍),不顯著可能只是 power 不夠,
這條路對 5 檔規模不適用,沒有得出新資訊。

**2. 週報酬版 Stage-1 IC**:週報酬跟隔日報酬是不同語意的預測目標,不能只換評估、不換訓練目標(不然
等於拿模型去預測它從沒學過的東西)。`meta/attention_fixed5/algorithm.py` 的 `pretrain_attention`
與 `_target_next_returns_for_date` 加 `horizon` 參數(預設 1,行為不變),新增
`_target_cumulative_returns_for_date(date, horizon)` 複利計算 N 日累積報酬;
`setup/11_stage1_ic_compare.py` 同步加 `--horizon`,訓練目標與評估目標一起換成週報酬
(horizon=5)。`tests/test_attention_pretraining.py` 加 2 個測試確認 horizon=1 等價於舊行為、
horizon>1 正確複利。

結果(15 seed,逐字稿+DEF14A+ontology+方向修正+TransE 128 維 embedding):

| 指標 | 隔日版(horizon=1) | 週報酬版(horizon=5) |
|---|---|---|
| KG 平均 test IC | 0.0020 | -0.0038 |
| Random 平均 test IC | 0.0143 | 0.0008 |
| Paired diff | -0.0123 | -0.0045 |
| Wilcoxon p | 0.489 | 0.804(更不顯著) |

**結論**:拉長到週報酬沒有改善,p 值反而更高。排除了「只是隔日雜訊太大」這個解釋。三個獨立角度
(隔日報酬、週報酬、5 檔子集共同波動)都測過,一致指向同一個誠實的負面結果:**這版 KG 的產業結構
(已確立為真實、顯著)沒有轉化成「固定 5 檔投資組合報酬預測」的效用,不受報酬定義或時間窗口影響。**

### 待辦

- [ ] IBM 在 DEF14A 裡沒有同業名單,評估是否需要額外資料源或人工標註幫它補上同業關係
- [ ] 與指導教授討論最終結果的論文呈現方式:sector test/link prediction 顯著,Stage-1 IC(隔日、
      週報酬皆測過)不顯著,Mantel test(全 30 家、5 檔子集皆測過)不顯著——三層結果一致且經過多角度
      交叉驗證,四層修正(ontology 碎片化→方向雜訊→抽取覆蓋率→KGE 模型選擇)+ 分層陳述「KG 有結構」
      與「KG 對下游任務有用」是兩件不同的事,是完整的方法論貢獻
