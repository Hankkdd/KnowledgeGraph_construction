# KG → RL 方法缺陷 Audit 與文獻轉向規劃

日期：2026-08-06  
分支：`experiment/gpm-kg-consumer`

## 1. 這份文件要回答的問題

目前結果不能簡化成「KG 好、RL 好，只是接在一起不好」。較精確的研究問題是：

> 現有的 KG 表示是否被模型轉換成與投資 horizon、supervised target 和 portfolio reward 對齊的可交易訊號？

目前的答案是：**尚未證明**。

本文件的目的不是再做一次超參數搜尋，而是：

1. 盤點目前 KG → model → action → reward 的實際資料流。
2. 把已知限制和尚未驗證的假設分開。
3. 以文獻中的 temporal graph、multi-relational graph 和 graph-native policy 作為設計參考。
4. 在重新啟動長時間 RL 前，先定義一個便宜且可停止的 supervised gate。

## 2. 目前證據的正確解讀

| 層次 | 目前證據 | 可以宣稱 | 不能宣稱 |
|---|---|---|---|
| KG 結構 | sector homophily 高於 degree-preserving null；2023→2024、2024→2025 的 forward residual-correlation alignment 顯著 | KG topology 包含可重現的公司關係/慢速風險結構 | KG 關係能預測每日方向或報酬 |
| KG → supervised target | h60 magnitude/MSE 有窄幅證據；IC、direction 未穩定勝過 controls；30-stock gate 未通過 | KG 可能是慢速 magnitude/risk prior 候選 | KG 已具有 cross-sectional alpha |
| RL pipeline | oracle positive control 5/5 seed 勝 random/equal-weight | consumer、environment、PVM、gradient path 能使用強訊號 | 真實市場中的 RL 本身穩定產生超額報酬 |
| KG → portfolio/RL | fixed adjacency overlay 與 covariance-aware overlay 未穩定改善主要風險指標 | 現有 interface 尚未把 KG 轉成穩定交易價值 | 只要增加 RL episodes 或調 reward 就能解決 |

## 3. 現有程式資料流 audit

### 3.1 目前固定 attention 路徑

`run_attention_fixed5.py` 和 `meta/attention_fixed5/data.py` 的資料流如下：

```text
一份固定 PyKEEN embedding (30 x 128)
        │
        ├─ selected 5 rows → Q
        └─ all 30 rows     → K

每日 30 檔 OHLCV rolling window (30 x window x features)
        │
        └─ PriceValueEncoder → V (30 x 128)

MultiHeadAttention(Q, K, V)
        │
        └─ 5 個 selected stock context → allocation head → action
```

關鍵程式位置：

- `meta/attention_fixed5/data.py:192-251`：每個日期只重新建立 OHLCV window；`kg_all`/`kg_selected` 從單一 embedding 檔載入一次。
- `meta/attention_fixed5/models.py:33-125`：KG embedding 是 attention 的 Q/K；只有 `--dynamic-query` 時，query 才融合 selected stock 自己的價格表示。
- `meta/attention_fixed5/models.py:128-241`：attention context 進入 `stock_scorer`，另外串接上一期 action；預測報酬 head 不直接進 action logits。
- `meta/attention_fixed5/models.py:266-285`：`action_blend` 把新 action 與上一期 action 混合。
- `meta/attention_fixed5/algorithm.py:141-216`：pretrain target 預設為 next-day cumulative return；可改 horizon，但主流程必須明確傳入一致 horizon。
- `meta/attention_fixed5/algorithm.py:315-388`：RL loss 主要是 differentiable portfolio log-growth；KG diversification penalty 只有設定係數時才啟用。

### 3.2 GPM 路徑的實際限制

`meta/attention_fixed5/gpm.py` 已經比固定 embedding attention 更接近 graph-native policy，但仍有以下限制：

- `GPMAllocationPolicy` 的 `edge_index`、`edge_type`、`edge_weight` 是 registered buffers；一個 episode 內沒有每日 graph snapshot。
- `PriceValueEncoder` 先把價格 window 轉成 node state，再用固定 graph 做 message passing；KG 關係沒有自己的時間或事件特徵。
- `encode_states()` 的 `graph_gate` 是單一 scalar，不能表達不同股票、relation 或日期對 KG 的依賴程度。
- `attention_context()` 輸出的 edge weights 其實是 normalized `edge_weight`，不是 learned message attribution；不能用它宣稱模型學到某條關係。
- `set_graph()` 只在 test/walk-forward 時替換整張 graph，沒有讓 train/test 內的 graph 隨日期更新。

### 3.3 已確認的結構性問題

#### A. 時間尺度不匹配

KG snapshot 近似年度/靜態；價格輸入是每日 rolling window；pretrain 預設 next-day target；RL reward 每日結算。這四個時間尺度沒有自然的 state transition 對應。

因此，KG 即使正確描述 60 日共動，也不代表它能改善 next-day allocation。

#### B. KG 只是一個 side representation，不是可驗證的 knowledge channel

目前模型沒有明確的 relation-level prediction task，也沒有要求：

- 相同 relation 的鄰居產生可辨識的 message；
- KG message 能改善 h60 residual risk/return target；
- graph ablation 時 downstream action 必須改變且變得更差。

所以目前的 attention weight 或 graph state 不能作為「模型使用知識」的證據。

#### C. graph target 與 reward 沒有對齊

目前可能同時存在：

- h60 residual-correlation / magnitude 的慢速訊號；
- next-day return 的 pretrain target；
- daily log-growth 的 RL objective。

這不是同一個 learning problem。把相同 embedding 接到三個 objective，不會自動完成 horizon transfer。

#### D. reward 可能把 KG 的有效部分抵消

目前主要 reward 是 portfolio growth。若 KG 真正提供的是風險/共動資訊，return-only objective 沒有理由學會使用它；`action_blend` 又會把新 action 與上一期 action 混合，造成訊號梯度進入實際 action 的幅度變小。

#### E. graph coverage 與 relation semantics 仍是瓶頸

GraphRAG 的公司名稱解析、relation extraction、concept projection 可能漏掉大量直接公司關係。即使整體 topology 有結構，selected portfolio 中仍可能沒有足夠高訊號、可區分的 relation。

## 4. 文獻轉向：不是照抄模型，而是抽取設計原則

### 4.1 Temporal relational prediction

**Temporal Relational Ranking for Stock Prediction** 將股票 relation 與時間演化同時建模，並直接以 ranking 為投資相關 target，而不是先訓練靜態 node embedding 再串到 RL。[論文](https://arxiv.org/abs/1809.09441)

可借鑑：

- relation message 應該與時間序列 encoder 同步更新；
- target 要和 portfolio selection 的 ranking 對齊；
- 先證明 graph-to-ranking，再接 portfolio policy。

### 4.2 Dynamic multi-relational graph

**MDGNN** 將多種 relation 與 relation evolution 交由 temporal Transformer 建模，而不是把所有 relation 壓成一個固定 embedding。[論文](https://arxiv.org/abs/2402.06633)

可借鑑：

- 每個 relation type 保留獨立參數與 mask；
- graph snapshot 或 edge event 具有時間索引；
- 用 relation ablation 判斷究竟是 topology、relation semantics，還是單純 capacity 在產生效果。

### 4.3 Graph-native actor-critic

**DeepPocket** 把 graph convolution 放入 actor-critic，使用隨時間變化的資產關係，而不是把 graph embedding 當作固定 side feature。[論文](https://arxiv.org/abs/2105.08664)

可借鑑：

- graph encoder 應同時服務 actor 與 critic；
- policy 和 value function 都要能看到 graph message；
- graph 需要是 state 的一部分，而不是只在 pretrain 時出現。

### 4.4 Factor representation 與 policy 分離

**Factor Representation and Decision Making in Stock Markets Using Deep Reinforcement Learning** 將市場狀態 representation 與 portfolio decision 分成兩個階段。[論文](https://arxiv.org/abs/2108.01758)

可借鑑：

- 先建立可驗證的 factor/representation target；
- 再讓 RL 學如何使用 representation；
- 不要用 RL 的 noisy return 結果反推 representation 一定有資訊。

## 5. 新的研究設計：先 supervised，再決定是否 RL

### Research question

> 在固定、時間正確的 KG snapshot 下，relation-aware graph encoder 是否能預測下一個 h60 window 的 residual risk/magnitude，且穩定勝過 no-graph、relation-shuffle、topology-shuffle controls？

### Hypothesis

若 KG 的有效資訊確實是慢速風險先驗，則 real graph 應在 h60 residual-correlation 或 magnitude target 上勝過所有 controls；若只勝 MSE 而不勝 IC，結論只能是 calibration/risk prior。

### Frozen split

- graph snapshot：只使用 holdout 起點以前可取得的 filings/events。
- windows：先做 `2023→2024` 與 `2024→2025` 兩個 temporal replication。
- horizon：`h=60`；另外只做一個 `h=20` sensitivity，不同時掃多個 horizon。
- universe：先用 30-stock universe，避免 fixed-5 power 過低。

### Variants and controls

1. `real`：真實 ticker-level KG，保留 relation type、edge confidence、snapshot date。
2. `self`：只有 self-loop/no graph。
3. `relation_permuted`：保留 topology 與 relation count，打亂 relation labels。
4. `topology_permuted`：保留 edge count/relation capacity，打亂 node topology。
5. `sector_only`：只用 sector relation，區分 KG 是否只是 sector proxy。

### Model interface

先不接 RL，建立下列 supervised encoder：

```text
daily OHLCV window → temporal node encoder
                         │
KG snapshot + relation type + edge confidence + age
                         │
                 relation-aware message passing
                         │
                 per-stock h60 prediction head
                         │
                 pair risk / magnitude target
```

必須保存：

- 每日每股票 graph message norm；
- 每種 relation 的 gate/weight；
- graph dropout 與 relation ablation 結果；
- real 與每一個 control 的 paired predictions。

### Primary metrics

- primary：paired seed win rate on h60 target MSE；
- secondary：Spearman IC、direction accuracy、top-k pair recall；
- risk target：future 60-day residual absolute correlation；
- statistical report：每個 seed 的差值、median effect、Wilcoxon one-sided p-value、90% seed rule。

### Seed/power rule

- smoke：1 seed，只驗證 pipeline，不做結論；
- exploratory：5 paired seeds；
- confirmatory：10 paired seeds；
- advance criterion：real 至少 `5/5` exploratory 或 `9/10` confirmatory 勝過主要 controls，且 effect direction 在兩個 temporal windows 一致；
- 若只改善 MSE、不改善 IC：只能進入 risk/magnitude branch，不得宣稱 alpha。

### Stop/advance criterion

若 real graph 沒有穩定勝過 `self`、`relation_permuted`、`topology_permuted`，停止 RL integration，回到 KG extraction/coverage。若 supervised gate 通過，才設計 graph-native actor-critic；且先做 1-window、5-seed 小型 economic backtest，再決定是否正式 RL。

## 6. 只有 supervised gate 通過後，才允許的 RL 版本

下一版 RL 不應再是目前的「固定 embedding attention + lightweight allocation head」，而應具備：

1. graph encoder 同時輸入 actor 與 critic；
2. graph snapshot/edge age 作為 state；
3. auxiliary h60 risk/magnitude prediction loss；
4. 明確的 risk-aware reward（例如 residual volatility 或 drawdown penalty），而不是只加一個靜態 `w^T S w` penalty；
5. actor-only、critic-only、actor+critic、no-auxiliary 四個 ablation；
6. graph dropout、relation shuffle、topology shuffle 的 action divergence 與 portfolio metrics。

RL 的 advance criterion 必須同時要求：

- real KG 相對 controls 的 90% seed 勝率；
- Sharpe、MDD、volatility 至少兩項 primary risk metrics 改善；
- action 對 graph ablation 有可重現差異；
- 不依賴單一 seed 或單一市場年度。

## 7. 當前決策

**Decision：pivot + implementation，不啟動新的長時間 RL。**

下一個實際工作項目是實作上述 supervised graph-to-h60 risk/magnitude gate，並先產生 1-seed smoke artifact。只有 smoke 的資料流、時間索引、control 生成和 metrics 都正確，才進入 5-seed exploratory。

這個 pivot 不否定目前結果；它把主張收斂到可驗證的問題：KG 是否提供慢速風險/幅度資訊，以及目前 RL interface 為何無法把這種資訊轉成 action。

