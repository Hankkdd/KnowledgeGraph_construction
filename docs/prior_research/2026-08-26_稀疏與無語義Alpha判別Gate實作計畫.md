# 稀疏 vs 無語義 Alpha 判別測試：H1/H2/H3 實作計畫

版本：v1.0  
日期：2026-08-26  
Branch：`experiment/gpm-kg-consumer`  
性質：`pivot`；純 supervised gate，不接 RL

## 1. 研究目的

目前的實驗已顯示：在現有 Dow 30、固定五檔、靜態公司關係與 GPM/RL 接法下，真實 KG 沒有穩健改善交易績效。但這個負結果仍有兩個可能解釋：

1. **H1：稀疏/橫截面不足。** 關係太少、固定投資組合太小，導致圖訊號無法傳到預測目標；換成較密集的圖與較大 universe 後，語義圖應能勝過容量匹配的 shuffle。
2. **H2：語義關係與報酬目標不匹配。** 即使使用密集圖與較大 universe，real 仍無法穩定勝過 shuffle；目前這類關係不適合報酬 alpha。
3. **H3：任務錯配。** 報酬排序沒有改善，但風險/共動目標能勝過 controls；KG 可能適合做慢速風險或分散化 prior，而非方向性 alpha。

本測試的目標是區分上述三種解釋，而不是再次尋找最佳 RL 超參數。

## 2. 先修正的判定邏輯

「密集圖勝過 shuffle」不能單獨證明原本失敗是因為稀疏，因為效果也可能來自 sector baseline、universe 變大或圖容量增加。因此 H1 必須觀察 universe × graph 的交互作用：

- 同一圖與同一模型，分別在 Dow 30 與較大 universe 測試。
- real 與 relation-shuffle、topology-shuffle 必須同時比較。
- 只有當較大 universe 的 real-versus-control 效果明顯增強，才把結果解讀為「稀疏/橫截面可能是主因」。

即使 H1 成立，也只能支持「密集的結構化圖有用」，不能直接宣稱 GraphRAG 抽取的公司語義 KG 有用。

## 3. Frozen research question

在相同價格資料、相同模型容量與時間正確的條件下，增加 universe 與圖密度，是否能讓正確的圖結構在 out-of-sample 報酬或風險目標上穩定勝過 no-graph、relation-shuffle 與 topology-shuffle controls？

## 4. 實驗範圍與分階段策略

### Stage 0：資料可行性與 PIT manifest（不訓練）

先建立並凍結 universe、價格與產業 metadata。若 PIT 成分資料無法可靠取得，不得直接宣稱「無 survivorship bias 的 S&P 500」。

### Stage 1：最小成本 supervised gate（先跑這個）

- Universe：Dow 30 與 S&P 100/200 proxy（二選一，以資料完整度較高者為準）。
- Graph：`G_gics`、`G_corr`。
- Horizon：h20、h60。
- Seeds：41–45。
- 目的：先判斷 sector/dynamic-correlation 圖是否存在可重現訊號。

### Stage 2：擴充 semantic graph（只有 Stage 1 有結果才做）

- 加入 `G_wiki`。
- 補做 `G_dense_random` 與完整 topology-matched controls。
- 將 promising target 擴到 10 seeds（41–50）。

### Stage 3：風險任務（若報酬 gate 失敗）

- 目標改為未來 h 日 realized covariance、absolute pair spread、volatility 或 drawdown proxy。
- 不接 RL；先確認 supervised risk gate。

### Stage 4：RL（預設不執行）

只有當 real 同時勝過 self、relation-shuffle、topology-shuffle，且在 supervised 與簡單經濟 backtest 都達到預先設定門檻，才重新考慮慢速 RL。

## 5. Universe 設計

### 5.1 主 universe

優先使用可取得歷史成分的 S&P 100 或 S&P 200，而不是直接假設 S&P 500。若 PIT 成分資料、價格資料或 ticker mapping 不完整，先縮小 universe，不以今日名單回填整段歷史。

### 5.2 Dow 30 對照

沿用現有 Dow 30 設定，與大 universe 使用相同：

- train/test 日期
- price feature
- target horizon
- model architecture
- seed
- controls

### 5.3 PIT 要求

每個 ticker 必須保存：

- `valid_from`、`valid_to`
- 成分來源與抓取日期
- ticker rename / delisting mapping
- universe manifest hash
- 缺失價格日與處理方式

若只能使用「測試期起點成分股」，報告中要明確標記為 historical proxy，並把 survivorship bias 列為限制。

## 6. 免費資料來源與 archive

### 6.1 價格

- 優先沿用現有 yfinance/Stooq pipeline。
- 保存原始下載檔、下載日期、ticker list、資料 hash。
- 所有 graph 與 target 只能使用決策日前資料。

### 6.2 GICS

- 保存 ticker、sector、industry、sub-industry、來源與版本日期。
- 若分類在期間內會變動，先凍結為研究起點版本並記錄限制；不可混用不同日期的分類。

### 6.3 Wikidata

- 保存 SPARQL query、query time、endpoint、raw response、解析後 edges 與 rejected rows。
- 只把 `parent/subsidiary`、`owned_by`、`industry` 等明確 typed relation 納入。
- 不預設 Wikidata 是密集圖；先報告 coverage、isolated nodes、degree distribution。

## 7. 四種圖的定義

### `G_gics`

- 同 sub-industry：主邊。
- 同 sector：可作弱邊或獨立 variant，不與主邊混為同一條 relation。
- 建議使用 typed undirected edges：`same_subindustry`、`same_sector`。

### `G_corr`

- 用過去 L 個交易日報酬建立 rolling correlation。
- 每個日期只使用截至當日的資料，取 top-k 或固定 threshold。
- 建議先固定 `L=60`、`k=5`，不要同時掃多個 window 與 k。
- 必須保存每一日 graph 的 edge list、density、degree、建圖截止日。

### `G_wiki`

- 使用 point-in-time 可用的 Wikidata typed relations。
- 缺失 relation 不得自動補成 negative edge。
- 需標記每條 edge 的 source、relation type、validity 與 coverage。

### `G_dense_random`

- 只作 sanity check。
- 優先使用 degree-preserving randomization，而非只匹配總 edge count。
- random graph 必須與被比較的 real graph 分別保存 seed 與 adjacency hash。

## 8. Controls 與公平性

每一個 graph family 都必須固定相同的模型容量與訓練設定，至少包含：

1. `no_graph`：只使用 price features。
2. `self`：只保留 self-loop。
3. `relation_shuffle`：保留 topology、degree 與 relation frequency，只打亂 relation label；只適用於 typed graph。
4. `topology_shuffle`：保留節點與 edge count，使用 degree-preserving rewiring；dynamic graph 要逐日處理。
5. `real`：未破壞的圖。

對 GICS 若只使用單一 relation，`relation_shuffle` 沒有語義意義，需改成明確標記為「不適用」，不可假造 typed control。此時至少保留 topology-shuffle 與 density/degree-matched random。

## 9. Target 與評估指標

### 9.1 報酬 target（H1/H2 主判定）

- 未來累積報酬：`h ∈ {20, 60}` 作為 Stage 1 主測試。
- primary：daily cross-sectional Rank IC。
- secondary：directional accuracy、top-k hit/recall、MSE。
- MSE 改善只能解讀為數值校準，不能單獨稱為 alpha。

### 9.2 風險/共動 target（H3）

- future realized covariance / correlation
- pairwise absolute return spread
- volatility 或 drawdown proxy

每個 target 必須在執行前指定：定義、方向、aggregation、primary metric 與 control。

### 9.3 時間切分

沿用現有 walk-forward 視窗；target horizon 為 h 時，train/test 邊界加入 purge gap，避免 forward window 重疊造成資訊污染。所有 scaler、graph normalization 與 target normalization 只在 train fit。

## 10. 模型與實作邊界

- 重用現有 typed R-GCN/GNN + supervised head。
- 所有 graph condition 使用完全相同的 hidden size、layer 數、epochs、optimizer、batching 與 early stopping。
- 不接 RL、不調 `action_blend`、不加入 reward shaping。
- `G_corr` 若為動態圖，模型 input API 必須明確接收 `graph[t]`，不能把測試期 rolling graph 預先整批建立後再回填。
- S&P 100/200 先作 scaling smoke；若要上 S&P 500，先確認 memory、runtime 與 missing ticker rate。

## 11. 統計設計與 gate 門檻

### Stage 1 exploratory

- seeds：41–45。
- real 對每個適用 control 做 paired comparison。
- 5/5 seed 方向一致才進 Stage 2；否則標記 exploratory。

### Stage 2 confirmation

- seeds：41–50。
- 至少 9/10 seeds 勝出。
- 同時報 paired mean difference、win count、effect size、Wilcoxon p-value。
- 主要決策不能只依 p-value。

### 依據 primary metric 的判定

- 報酬：real 必須在 Rank IC 上勝過 relation/topology controls；只勝 MSE 不算通過。
- 風險：real 必須在預先指定的 covariance/spread/volatility 指標上勝過 controls。
- 結果要同時在 Dow 與較大 universe 報告，避免只挑較好的 universe。

## 12. H1/H2/H3 決策樹

### H1：稀疏/橫截面不足

必要條件：

- 大 universe 的 real-vs-control Rank IC 改善明顯高於 Dow 30。
- 至少一個語義圖在 5 seeds 通過探索門檻，之後 10 seeds 達 9/10。
- 效果不能只來自 MSE 或圖邊數增加。

行動：投入更完整的 KG coverage、entity resolution 與 relation extraction；仍需把 GICS/sector baseline 與 GraphRAG KG 分開報告。

### H2：目前語義關係不適合報酬 alpha

必要條件：

- GICS、Wikidata 或其他 tested graph 在較大 universe 仍無法穩定勝 controls。
- Rank IC、direction 與簡單 ranking backtest 都沒有一致改善。

行動：停止「補更多同類公司關係即可提升報酬」主張，正式寫入負結果。

### H3：風險/共動任務較匹配

必要條件：

- 報酬 Rank IC 未通過。
- 至少一個 risk target 在 real 對 relation/topology controls 上達到預設 seed gate。

行動：轉向慢速 risk forecasting、covariance estimation 或 portfolio risk overlay；仍先停留在 supervised，不直接接 RL。

## 13. 執行順序與預估成本

### Phase 0：資料與 scaling smoke

1. 凍結 PIT/historical-proxy manifest。
2. 下載並驗證價格資料。
3. 產生 GICS 與第一版 rolling correlation graph。
4. 輸出 coverage、density、degree、isolated nodes 報告。

停止條件：universe 缺失率、PIT 對齊或價格 coverage 不合格時，不進模型。

### Phase 1：2 graph × 2 universe × 2 horizon × controls × 5 seeds

先跑 `G_gics`、`G_corr`，只測 h20/h60 與報酬 target。這一階段的重點是判斷是否有任何便宜且密集的結構能產生訊號。

### Phase 2：semantic graph confirmation

只有 Phase 1 出現一致候選，才加入 G_wiki、dense random 並補到 10 seeds。

### Phase 3：risk target

若報酬 gate 失敗，使用同一 frozen split 測 risk target；不重新調整 universe 與 graph 定義。

### Phase 4：停止或重新提出 RL spec

沒有 supervised gate 通過時，停止 RL。只有通過後，另立一份 slow-policy RL spec，不把本計畫結果直接接進既有 daily RL。

## 14. Artifact 與程式規劃

建議新增：

```text
0601_atten_V2/setup/50_build_dense_universe.py
0601_atten_V2/setup/51_build_gics_graph.py
0601_atten_V2/setup/52_build_rolling_corr_graph.py
0601_atten_V2/setup/53_build_wikidata_graph.py
0601_atten_V2/setup/54_validate_universe_and_graphs.py
0601_atten_V2/setup/55_run_sparse_semantic_gate.py
```

建議輸出：

```text
0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_YYYYMMDD/
├── config.json
├── universe_manifest.csv
├── universe_manifest.json
├── price_manifest.json
├── graph_manifest.json
├── graphs/<graph>/<date>.edges.csv
├── coverage_report.md
├── graph_statistics.csv
├── seed_results.csv
├── paired_comparisons.csv
├── summary.json
└── report.md
```

每次 run 必須記錄：

- git commit / branch
- command line
- seed list
- train/validation/test dates
- universe hash
- graph snapshot/hash
- target horizon
- model config
- controls
- runtime 與失敗樣本

## 15. 報告格式

每個 completed gate 都要回答：

```text
Research question:
Why this experiment is next:
Hypothesis:
Frozen data/KG split:
Variants and controls:
Primary metric and direction:
Seed/power rule:
Stop/advance criterion:
Result:
Paired effect and seed wins:
What this proves:
What it does not prove:
Decision: confirm / pivot / implementation / stop
Next gate:
Record updated: yes/no
```

完成後同步更新：

- `docs/2026-08-06_GPM_KG時間尺度與慢速交易驗證紀錄.md`
- `docs/2026-08-21_研究全歷程實驗清單.md`
- 新增 artifact `report.md` 與 `summary.json`

## 16. 預期研究結論

這個實驗的成功標準不是一定要得到正結果，而是讓目前的負結果可以被清楚拆解：

- H1：值得投資更高 coverage、更密集且更完整的 KG。
- H2：目前語義公司關係不適合報酬 alpha，停止同類資料工程與 RL sweep。
- H3：KG 可能適合慢速風險/共動任務，轉向 risk-aware downstream。

無論結果是哪一種，都不能把 dense sector/correlation graph 的結果直接等同於 GraphRAG financial KG 的語義價值。
