# KG 圖學習交易框架：研究定位與驗證規劃

更新日期：2026-07-27

## 1. 文件目的

本文件整理近期針對金融知識圖譜（Knowledge Graph, KG）、Attention、R-GCN、GRU、投資組合最佳化、Gate 0/1，以及跨公司組合與跨指數驗證的討論。目的包括：

1. 說明目前程式實際完成了什麼，而不是把規劃中的系統誤當成已完成成果。
2. 釐清 KG、圖編碼器與交易決策各層應分別驗證的問題。
3. 確定本研究可安全主張的研究貢獻與不可主張的內容。
4. 將原本固定五檔配置實驗重新定位為 controlled benchmark。
5. 規劃隨機子投資組合、完整 universe 與跨指數泛化驗證。
6. 提供後續修改《大專生計畫.md》及向教授說明時可使用的架構。

---

## 2. 核心結論

本研究不應將創新定位為「第一個結合 KG、GCN 與 RL 的交易系統」。KG+RL、GCN+RL、KG+GCN，以及使用 heterogeneous graph policy 的 portfolio RL 都已有直接或高度相鄰的工作。

較安全且更符合目前工作內容的定位是：

> 建立一套可審計、可歸因、逐階段驗證的金融 KG 到投資組合決策框架，分別驗證 KG 品質、圖結構利用、signal-to-action transfer 與真實市場經濟價值。

預期貢獻可分成三層：

1. **System contribution**：可重現的 point-in-time financial KG → relation-aware graph encoder → portfolio policy/optimizer pipeline。
2. **Methodological contribution**：以 Gate、oracle 與 graph corruption controls 分離不同系統層級，讓成功或失敗可以歸因。
3. **KG engineering/evaluation contribution**：提出具 provenance、時間正確性、relation ontology、direct/projected edge 區分，以及多層品質驗證的真實金融 KG 建立方法。

真正可能具有新穎性的部分，不一定是某一個神經網路模型，而是：

> 如何科學地判定金融 KG 是否含有可交易資訊、圖模型是否真的使用它，以及資訊是否真的傳遞到投資決策。

---

## 3. 目前系統的實際狀態

### 3.1 原始真實市場主線

目前《大專生計畫.md》描述的主線是：

```text
金融報告
  → GraphRAG 建立 KG
  → PyKEEN/ComplEx 產生靜態公司 embedding
  → KG embedding 作為 Attention query/key
  → Dow 30 OHLCV windows 作為 value
  → 固定資產集合的 allocation head
  → direct differentiable portfolio optimization
```

這條主線使用的是 **static KG embeddings + Attention**，尚不是每個交易日直接在 typed/directed KG 上執行 R-GCN message passing。

其交易訓練嚴格而言較接近 **direct differentiable portfolio optimization**：梯度由已實現的 portfolio growth 經可微分環境直接回傳至 policy，沒有 value function、advantage estimate，也不是 PPO、DQN 或標準 actor-critic。論文若繼續使用 RL 一詞，必須明確說明這個定義。

### 3.2 正式主線並不是每天從 30 檔挑五檔

目前程式預設的可交易集合固定為：

```text
MSFT, NVDA, GS, JPM, KO
```

模型每天只在這五檔之間輸出連續配置權重。預設設定為：

- 不允許 cash。
- 不允許 short selling。
- 五檔權重總和為 1。
- softmax temperature = 2.0。
- 單檔最大權重 = 0.4。
- action blend = 0.025（原始正式主線設定）。

因此它是 **fixed-universe allocation**，不是 stock selection，也不是每期由 Dow 30 排名後選 top 5。softmax 通常會讓五檔都有正權重。

目前《大專生計畫.md》仍寫舊組合 AAPL/NVDA/AMZN/IBM/MSFT，與程式設定不一致，後續需要修正。

### 3.3 Gate 0

Gate 0 是 positive control，將具有未來資訊的 oracle context 直接交給 consumer/reward/action pipeline，繞過原本 KG/price Attention。

Gate 0 支持的窄結論是：

> consumer、reward 與 action pipeline 有能力利用明確可用的排序訊號，整條交易管線並非完全失效。

Gate 0 不支持以下結論：

- 真實 KG 有效。
- Attention 有效。
- R-GCN 有效。
- relation type 或 edge direction 有效。
- GRU 是必要的。
- oracle 結果可以視為可部署績效。

### 3.4 Gate 1

Gate 1 是與原始真實市場主線分離的 synthetic dynamic-graph test bench，其資料生成過程為：

```text
r[t+1] = beta × A_t × x_t + gamma × f_t + epsilon[t]
```

policy 只能看到當期 node features `x_t` 與 typed/directed graph `A_t`，不能看到 future returns、private conditional alpha 或 relation coefficients。

Gate 1 的目的是回答：

> 當未來報酬確實由 typed、directed、dynamic graph 生成時，圖模型能否學會利用正確的圖結構並將訊號傳到 action？

主要 controls 包括：

- Correct graph
- No graph
- Degree-preserving shuffled graph
- Relation-shuffled graph
- Timestamp-shifted graph

`top1_hit` 只是檢查 action 中權重最高的資產，是否等於 synthetic ground-truth conditional alpha 最高的資產。它是診斷指標，不是正式交易策略的 top-k 選股規則。

---

## 4. Gate 1 已有結果與可支持的結論

### 4.1 Gate 1 v1 formal

40 個 formal cases 的 simultaneous wins 為：

```text
28/40 = 70%
```

門檻要求為：

```text
36/40 = 90%
```

因此 Gate 1 v1 的正式決策是 **FAIL**，不能因為個別 controls 勝率較高而改稱通過。

個別 correct-graph 勝率：

| Control | Correct graph 勝率 |
|---|---:|
| Degree-preserving shuffled | 38/40 = 95% |
| Timestamp-shifted | 35/40 = 87.5% |
| Relation-shuffled | 34/40 = 85% |
| No graph | 34/40 = 85% |

其他診斷：

- Mean Spearman：0.0818
- Mean top1 hit：0.2316
- Graph-mask action L1：0.0942
- Graph-mask log-growth drop：+0.659 bp/day

這些結果顯示模型可能利用了部分圖訊號，但尚未達到預先凍結的 simultaneous pass criterion。

### 4.2 Gate 1 v2 GRU

GRU 版本結果：

- Simultaneous wins：3/40 = 7.5%
- Spearman：0.00699
- top1 hit：0.20625
- Graph-mask action L1：0.29939
- Graph-mask performance drop：-0.08837 bp/day

這不是「差一點」，而是目前 GRU 版本對圖訊號的利用全面惡化。此結果只支持：

> 目前的 GRU 實作與訓練設定沒有改善 Gate 1，不能支持跨期記憶的必要性。

它不代表所有 GRU 架構永遠無效；但 Gate 1 的 DGP 在當期 `A_t,x_t` 已充分，理論上不需要記憶，因此目前不應優先投入 GRU 調參。

### 4.3 Action-blend sweep

Correct-graph-only sweep 使用五個 validation seeds 與四個 transitions，結果為：

| action_blend | Spearman | top1 hit | Sharpe | mask 後績效下降 |
|---:|---:|---:|---:|---:|
| 0.025 | 0.0866 | 0.2469 | 0.417 | 0.6049 bp/day |
| 0.10 | 0.2483 | 0.3063 | 2.225 | 3.5078 bp/day |
| 0.25 | 0.4597 | 0.3500 | 6.298 | 9.8971 bp/day |
| 1.00 | 0.8987 | 0.7195 | 20.235 | 44.0366 bp/day |

四個指標隨 action blend 單調改善，強烈支持：

> `action_blend=0.025` 會嚴重壓制 graph signal 傳到實際 action。

但這個 sweep 只跑 correct graph，沒有同時重訓四個 controls。因此它尚未證明 `action_blend=1.0` 能達成 36/40 simultaneous wins，也不能直接宣布 Gate 1 v1.1 通過。

合理的下一步是：

- 使用無 GRU 的 R-GCN。
- 固定 `action_blend=1.0` 作為候選設定。
- 使用全新 confirmation seeds。
- correct graph 與四個 controls 全部重新訓練。
- 使用原本凍結的 90% simultaneous criterion，不因結果修改門檻。

---

## 5. Attention、R-GCN 與 GRU 的角色

### 5.1 Attention 不應直接被放棄

原始 Attention 的負面或不穩定結果，不代表所有 Attention 架構無效。它目前應重新定位為：

- Static embedding + Attention baseline
- Dynamic-query Attention ablation
- Relation-aware Attention 候選
- R-GCN + Attention 混合模型候選

向教授說明時，不應說「Attention 失敗所以改用 R-GCN」，而應說：

> 原始實驗無法區分問題來自 KG、encoder、action transfer 或 portfolio objective，因此先建立 Gate 0/1；Attention 保留為 baseline，R-GCN 則針對 multi-relational KG 結構提供更直接的 inductive bias。

### 5.2 R-GCN 的角色

R-GCN 為每種 relation type 使用不同的 message transformation，適合 typed、directed、multi-relational KG。它能直接測試：

- relation type 是否重要。
- edge direction 是否重要。
- graph timing 是否重要。
- multi-hop message passing 是否提供額外資訊。

### 5.3 GRU 的角色

GRU 用來表示跨期記憶。只有當研究假設明確要求歷史 graph states、事件累積或 regime persistence 時，才應加入。若當期 observation 已充分，加入 GRU 只會增加參數與最佳化困難。

目前建議：

- Gate 1 v1.1 先使用無 GRU R-GCN。
- GRU 降為後續 temporal ablation。
- 真實動態 KG 顯示跨期記憶需求後，再重新評估。

---

## 6. 文獻定位

目前已核對的代表性工作如下：

1. [Sentiment and Knowledge Based Algorithmic Trading with Deep Reinforcement Learning](https://arxiv.org/abs/2001.09403)：結合價格、新聞 sentiment、KG 與 DRL；摘要未顯示使用 GCN。
2. [DeepPocket: Deep Graph Convolutional Reinforcement Learning for Financial Portfolio Management](https://arxiv.org/abs/2105.08664)：以資產 pairwise correlation graph、graph convolution 與 actor-critic 進行 portfolio management；其 graph 不是由金融文件建立的 semantic KG。
3. [Integrated GCN-LSTM stock prices movement prediction based on knowledge-incorporated graphs construction](https://link.springer.com/article/10.1007/s13042-023-01817-6)：knowledge-incorporated graph + GCN-LSTM，用於股價走勢預測，不是 portfolio RL。
4. [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)：R-GCN 原始方法，針對 multi-relational knowledge bases，但不是金融交易研究。
5. [SmartFolio, IJCAI 2025](https://www.ijcai.org/proceedings/2025/1054)：結合 financial knowledge、heterogeneous graph policy、hierarchical attention 與 inverse RL，與本研究高度接近，必須列為主要 related work。

因此不可主張：

> First KG+GCN+RL framework for portfolio management.

目前較安全的說法是：

> 在已核對的代表性工作中，尚未看到與本研究完全相同的「point-in-time 金融文件 KG、relation-aware graph encoder、portfolio policy optimization、Gate 0/1 causal controls 與多層真實 KG validation」完整流程。

即使使用 `to our knowledge`，也應等完成 Google Scholar、Scopus、IEEE Xplore、ACM Digital Library 等較完整檢索後再決定。

文獻內容均為重述整理，未逐字重製來源內容。

---

## 7. 固定五檔設計的問題與重新定位

### 7.1 為什麼固定五檔不適合作為最終交易框架

1. 股票由研究者事先選定，而非模型選擇。
2. 好績效可能來自人工挑到好股票，而不是 KG 或模型有效。
3. 無法證明 KG 能協助完整 universe 中的 stock selection。
4. 存在 selection bias 與 survivorship bias 風險。
5. R-GCN 對完整圖計算後只允許五個節點交易，會浪費圖模型能力。

因此，單一固定五檔只能支持：

> KG-derived context 是否可能改善預先指定資產集合內的配置。

不能支持：

> KG 能從市場中發現值得投資的公司。

### 7.2 固定五檔仍可保留為 controlled benchmark

固定集合的優點是能控制 action space，將 stock-selection error 與 allocation error 分離。可保留以下比較：

- Equal weight
- Price-only / no graph
- Static KG embedding + Attention
- GCN
- R-GCN
- Corrupted graph controls

但論文中應稱為：

> Fixed-universe allocation benchmark

而不是完整選股交易系統。

---

## 8. 隨機子投資組合與跨指數泛化驗證

### 8.1 核心研究問題

使用者提出的最終想法可形式化為：

> 不論抽到哪一組公司，只要提供的是具品質且時間正確的金融 KG，relation-aware portfolio model 是否能比沒有 KG 或錯誤 KG 做出更好的配置決策？

更安全的論文表述為：

> Across multiple pre-registered company subsets, market periods, and equity indices, the point-in-time financial KG provides incremental information over non-graph and corrupted-graph controls.

這是在測試框架的 generalization，而不是某五檔是否剛好有效。

### 8.2 Repeated random-subportfolio validation

每個 index 先建立完整 universe，再以預先登記的 random seeds 抽取多組 tradable subsets，例如：

- 每個 index 抽 30–50 組 portfolios。
- 比較 subset size = 5、10、15 或 all。
- random seeds 在看到結果前凍結。
- 不因績效不好而重新抽樣。
- 保留所有成功與失敗案例。

每個隨機 portfolio 都計算 paired difference：

```text
Delta_i = Performance(Real KG_i) - Performance(Control_i)
```

統計結果至少包括：

- Mean delta
- Median delta
- 95% confidence interval
- Real KG 勝出比例
- 不同年份與市場 regime 的效果
- Transaction-cost-adjusted return
- Sharpe、maximum drawdown、turnover 與 concentration

不應要求每組都勝出；較合理的是整體平均為正、多數組合方向一致、信賴區間排除零，而且不依賴單一股票或單一時期。

### 8.3 不應為每個 subset 建立孤立 KG

如果只為抽到的五家公司建圖，會遺漏其與其他未交易公司的重要關係。較好的設計是：

```text
完整 index universe KG = graph/context universe
隨機抽到的公司 = tradable subset
action = 只對 tradable subset 配置權重
```

例如 MSFT 與 NVDA 的關係可能有助於 MSFT 的決策，即使 NVDA 當期不在 tradable subset 中，也可以保留為 context node。

### 8.4 跨指數驗證

候選市場：

1. Dow 30
2. Nasdaq-100
3. S&P 100 或 S&P 500

若資源有限，可先完成 Dow 30 與 Nasdaq-100。所有實驗必須使用 **point-in-time constituent lists**，不能用現在的成分股回測過去，否則會有 survivorship bias。

跨指數驗證必須注意：

- 不同市場文件覆蓋率。
- entity/ticker alignment。
- 產業分布差異。
- index reconstitution。
- delisted firms。
- 文件發布時間與資料可得時間。

---

## 9. 最終 graph controls 與比較基準

每組公司、每個時間區段都應固定相同的：

- Price path
- Train/test dates
- Initialization seeds
- Model capacity
- Hyperparameters
- Reward/objective
- Transaction cost
- Action constraints

只替換 graph provider：

1. Real point-in-time financial KG
2. No graph
3. Random graph
4. Degree-preserving shuffled graph
5. Relation-shuffled graph
6. Timestamp-shifted graph
7. Sector graph
8. Price-correlation graph

Sector graph 與 correlation graph 特別重要，因為它們回答：

> 複雜的文件 KG 是否真的比廉價、容易取得的金融圖更有價值？

若 Real KG 只贏 no-graph，卻不優於 sector/correlation graph，則不能主張金融文件知識是績效來源。

---

## 10. 真實金融 KG 的建立與多層驗證

「好的 KG」不能以交易績效反向定義，否則會形成循環論證。KG 必須在進入交易模型前，使用獨立標準評估。

### 10.1 資料與時間正確性

- Point-in-time availability
- Filing/transcript publication timestamp
- Future-information leakage check
- Entity–ticker alignment accuracy
- Delisted company handling
- Index membership history
- Provenance coverage

### 10.2 結構與語義品質

- Relation ontology coverage
- Relation type diversity
- Direct/projected edge 比例
- Duplicate/conflicting edge rate
- Manual edge precision
- Known-pair recall
- Relation-specific precision
- High-degree generic concept contamination
- Link prediction vs. shuffled graph baseline

### 10.3 金融訊號一致性

- KG similarity vs. sector membership
- Graph distance vs. return correlation
- Known supply-chain/competitor relation recall
- Relation-specific future-return association
- Sector-label permutation
- Return-correlation permutation
- Timestamp permutation

### 10.4 下游因果驗證

固定 encoder、optimizer、reward、action 與市場路徑，只替換 graph provider，回答：

> 是真實金融知識有效，還是只要提供任意 graph 都有效？

### 10.5 KG 品質與下游增益的關係

可進一步檢驗：

```text
KG quality ↑  →  Real-KG-vs-control performance delta ↑
```

例如 manual precision、provenance coverage、temporal coverage 或 relation diversity 越高的圖，是否帶來較大的增量績效。若成立，才能支持：

> 不是「有圖就有效」，而是「金融知識品質越好，模型越能學到可用決策資訊」。

---

## 11. 建議的正式研究假設

### H1：正確 KG 提供增量價值

```text
Real KG > No graph
```

跨隨機 portfolios 的平均成本後績效較好。

### H2：效果來自正確的圖結構與語義

```text
Real KG > Degree shuffle, Relation shuffle, Timestamp shift
```

用以排除參數量增加或任意 graph regularization 的解釋。

### H3：文件 KG 提供簡單金融圖以外的資訊

```text
Real KG > Sector graph, Correlation graph
```

這是最困難但也最有價值的假設。

### H4：KG 品質可預測下游增益

```text
KG quality ↑ → Incremental portfolio value ↑
```

連結 KG 建立方法與交易結果。

### H5：效果可跨 subset、時期與 index 泛化

```text
Effect direction is stable across pre-registered subsets, walk-forward windows, and indices.
```

不要求每一組都勝出，但不能由少數樣本主導。

---

## 12. 最終交易框架建議

### 12.1 Controlled experiment

保留固定或隨機抽樣的子投資組合作為受控 allocation benchmark，目的是隔離 KG/encoder 效果。

### 12.2 Full-universe allocation

最終框架應讓 R-GCN 對完整 point-in-time universe 的所有公司產生 score，再轉成連續權重：

```text
Financial documents available by t
  → Point-in-time typed/directed KG G_t
  → Node/market features X_t
  → R-GCN or relation-aware encoder
  → Per-company scores
  → Portfolio weights
  → Cash/cap/turnover/transaction-cost constraints
  → Walk-forward evaluation
```

### 12.3 若需要稀疏持倉

不應由研究者事先指定股票。可以比較：

- All-universe continuous allocation
- Model-ranked top 5
- Model-ranked top 10
- Model-ranked top 15
- Sparsemax/entmax allocation
- Cash-enabled allocation

`K` 必須預先決定並進行 sensitivity analysis，避免只報告績效最佳的 K。

---

## 13. 可向教授使用的說明

建議說法：

> 我們不是因為 Attention 結果不好就任意改成 R-GCN，而是發現原本的實驗無法區分問題究竟來自 KG、encoder、action transfer 或 portfolio objective。因此先建立 Gate 0 與 Gate 1，逐層確認交易管線能使用 oracle，並確認模型能否識別 relation、direction 與 graph timing。Attention 會保留為 baseline；R-GCN 是針對 multi-relational KG 的主模型候選。最後在固定下游模型的條件下，以多組公司、不同指數及 graph corruption controls，驗證真實 KG 是否提供可泛化的增量資訊。

固定五檔的說明：

> 原本固定五檔是 controlled allocation experiment，用來分離配置與選股問題，不是最終完整交易系統。下一階段將擴展為預先登記的隨機子投資組合與完整 universe 配置。

---

## 14. 論文可安全使用與不可使用的表述

### 14.1 建議使用

> We present an auditable, stage-wise validated framework connecting point-in-time financial knowledge graphs, relational graph representation learning, and portfolio policy optimization.

> We introduce a causal validation protocol that separates KG quality, graph utilization, signal-to-action transfer, and downstream portfolio performance.

> We evaluate whether point-in-time financial knowledge provides incremental information over non-graph, corrupted-graph, sector-graph, and correlation-graph controls.

### 14.2 尚不可使用

> We propose the first KG+GCN+RL portfolio framework.

> Gate 1 proves that R-GCN works on real financial KGs.

> The action-blend sweep proves Gate 1 passes.

> KG improves trading across arbitrary companies and markets.

> The framework performs stock selection from the Dow 30.

以上敘述分別受到既有文獻、synthetic-to-real gap、controls 尚未重跑、跨市場實驗尚未完成，以及目前只做固定集合 allocation 的限制。

---

## 15. 建議的後續執行順序

### Phase 1：完成 synthetic causal validation

1. 凍結無 GRU R-GCN + `action_blend=1.0` 候選設定。
2. 使用全新 confirmation seeds。
3. correct graph 與四個 controls 全部重訓。
4. 依原定 36/40 simultaneous rule 判定 Gate 1。
5. 不再以 correct-only sweep 取代正式 gate decision。

### Phase 2：建立 point-in-time 真實 KG snapshots

1. 定義 relation ontology。
2. 保留 provenance。
3. 區分 direct/projected edges。
4. 驗證 publication timestamp 與 ticker alignment。
5. 完成 manual、structural、temporal 與 financial diagnostics。

### Phase 3：固定模型替換 graph provider

1. Real KG
2. No graph
3. Degree shuffle
4. Relation shuffle
5. Timestamp shift
6. Sector graph
7. Correlation graph

此階段不能同時更改 KG、encoder、reward 與 action rules，否則失去歸因能力。

### Phase 4：Repeated random-subportfolio validation

1. 預先登記 random seeds。
2. 每個 index 抽取多組 5/10/15-stock subsets。
3. 使用完整 index KG 作為 context。
4. 進行 paired graph-provider comparisons。
5. 報告所有樣本與信賴區間。

### Phase 5：Full-universe walk-forward validation

1. 使用 point-in-time constituents。
2. 完整 universe allocation。
3. Cash、weight cap、turnover 與 transaction costs。
4. 多 walk-forward windows。
5. 與 equal weight、MVO、price-only、sector graph、correlation graph 比較。

### Phase 6：跨指數外部驗證

在凍結架構與主要超參數後，移至第二個 index，避免針對每個市場重新大量調參後再宣稱泛化。

---

## 16. 目前限制清單

1. Gate 1 v1 formal 為 70%，未達 90% 門檻。
2. Action-blend sweep 只測 correct graph，尚未完成 confirmation controls。
3. GRU v2 目前失敗，不能宣稱 temporal memory 有效。
4. R-GCN 尚未接上 point-in-time 真實金融 KG。
5. 原始真實市場主線仍是 static PyKEEN embedding + Attention。
6. 現有正式策略只做固定五檔 allocation，不做全 universe stock selection。
7. 《大專生計畫.md》的五檔名單與程式不一致。
8. 真實 KG 的 publication-time snapshots 尚未完整驗證。
9. 跨隨機 subsets 與跨指數實驗尚未執行。
10. 尚未完成正式 systematic literature review，不能主張 first 或完整 novelty exclusion。

---

## 17. 最終建議研究問題

本研究可收斂為：

> 經過獨立品質驗證、具來源證據且時間正確的金融知識圖譜，是否能跨公司組合、跨市場時期與跨股票指數，為關係感知的投資組合模型提供穩定、可泛化且可歸因的增量決策資訊？

這個問題同時涵蓋：

- 如何建立可信的真實金融 KG。
- 如何證明模型真的使用 relation、direction 與 graph timing。
- 如何證明 graph signal 能傳到 portfolio action。
- 如何排除任意 graph、sector 或 correlation graph 的替代解釋。
- 如何證明效果不是特定五檔或單一指數的偶然結果。

若最終 Real KG 顯著優於所有 controls，可支持金融知識具有增量交易價值；若沒有優於，分層 Gate 與 controls 仍能指出問題發生於 KG 品質、graph encoder、action transfer 或市場訊號本身，使負面結果仍具有可解釋性與研究價值。
