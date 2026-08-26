# RL Attention 收斂診斷與 Walk-Forward 驗證紀錄

日期:2026-07-23
範圍:`run_attention_fixed5.py`(Stage-2 RL 資產配置),固定投資組合 AAPL/NVDA/AMZN/IBM/MSFT
KG:snapshot 15(逐字稿+DEF14A同業關係+ontology映射+方向修正),pykeen_run 13(TransE 128 維)
資料存放:`runs/kg_vs_random_15seed_20260723/`、`runs/rl_convergence_diagnostics_20260723/`、
`runs/walkforward_ep20_final_test_20260723/`(原本在 session scratchpad,已搬進專案 `runs/` 永久保存)

## 背景

在 KG 驗證(見 `Dow30_10K_KG驗證紀錄.md`)確立「KG 有真實產業結構」之後,想確認這個結構有沒有
在 Stage-2 RL 資產配置上產生實際效用。第一次比較(`--episodes 5` 預設值)KG vs random embedding
幾乎沒有差異,進一步檢查發現 **policy 幾乎沒有在交易**。

## 診斷 1:Policy 幾乎不動

測試期(2025 全年,250 個交易日)的權重紀錄:5 檔股票全部卡在 **19.8%~20.4%** 之間(等權重是
20%),標準差只有 0.0002~0.0012。

## 診斷 2:episode-level reward 沒有被記錄

`meta/attention_fixed5/algorithm.py` 原本只有 `env._portfolio_reward_memory`(每個 episode
`reset()` 時清空,只保留最後一個 episode 的逐日 reward),沒有任何地方記錄「每個 episode 的總
reward」,無法判斷訓練有沒有進步。新增:

- `AttentionPolicyGradient.__init__` 加 `self.episode_reward_history: list[dict] = []`
- `train()` 迴圈裡累積每個 episode 的 `total_reward`/`mean_reward`/`n_steps`
- `run_attention_fixed5.py` 存成 `train/episode_rewards.csv`

## 診斷 3:訓練 30 個 episode 後,reward 確實在漲,但幅度小

`total_reward` 從 0.446(ep1)漲到 0.472(ep30),跟 episode 數相關係數 0.987,後段(25-30)開始
趨緩。`stock_entropy` 從 1.609428 緩慢降到 1.606057(仍接近均勻分布上限 ln(5)=1.6094,但確定在
持續下降,不是雜訊)。`turnover` 從 0.0006 漲到 0.0070(增加超過 10 倍)。**結論:RL 確實有在學,
只是收斂極慢,5 個 episode 遠遠不夠看出變化。**

## 診斷 4:舊的參數掃描報告(`runs/PARAM_SWEEP_REPORT.md`,6/27,先前工作)有誤導性結論

舊報告主張「1 個 episode 比 5-20 個 episode 類化更好」。查證 `runs/param_sweep_summary.csv` 發現
**這個結論被混淆了**——所有 episodes>1 的紀錄(`rlstrong`、`momentum`、`unfrozen`、`cash`)同時
也改了 `lr`、`head_coef`,甚至 `allow_cash`/`unfrozen attention` 這些完全不同的設定,不是單純
只改 episode 數的乾淨對照。

用現在這版 KG、其他參數維持預設值、只改 episode 數(1/5/10/20),在**同一段測試期**(2025 全年)
上結果:total_return 0.2289→0.2291→02294→0.2319,Sharpe 0.9458→0.9461→0.9466→0.9507——
**沒有變差,反而微幅變好**。舊報告的「更多 episode 更差」是超參數混淆造成的假象,不是 episode
數本身的問題。

## Walk-Forward 驗證(修正舊報告的方法論限制)

舊報告自己承認的限制:「這些參數是直接在 2025 測試期上選出來的,不是真正的外推驗證」。這次改用
乾淨的驗證/測試切分:

- 訓練:2024-01-01 ~ 2025-01-01(不變)
- **驗證段**(用來選 episode 數,不看最終測試段):2025-01-01 ~ 2025-07-01
- **最終測試段**(選好參數後只評估一次):2025-07-01 ~ 2026-01-01

### 驗證段結果(episode 數掃描:1/5/10/20/50/100)

| episodes | Sharpe(驗證段) |
|---|---|
| 1 | 0.8088 |
| 5 | 0.8092 |
| 10 | 0.8098 |
| **20** | **0.8111**(最高) |
| 50 | 0.8020 |
| 100 | 0.7956 |

典型的「先進步、後過擬合」曲線,**episodes=20** 是驗證段上的最佳值,選定後不再調整。

### 最終測試段結果(2025 H2,只跑一次,5 個 seed 估變異度)

| Seed | KG 報酬 | Random 報酬 | KG Sharpe | Random Sharpe |
|---|---|---|---|---|
| 41 | 0.1184 | 0.1178 | 1.4589 | 1.4565 |
| 42 | 0.1180 | 0.1187 | 1.4575 | 1.4456 |
| 43 | 0.1183 | 0.1191 | 1.4630 | 1.4681 |
| 44 | 0.1184 | 0.1177 | 1.4618 | 1.4537 |
| 45 | 0.1195 | 0.1179 | 1.4677 | 1.4586 |
| **平均** | **0.1185** | **0.1183** | **1.4618** | **1.4565** |

Wilcoxon signed-rank(paired,5 seed):total_return p=1.0;sharpe p=0.1875。**兩者都不顯著。**

## 結論

用乾淨的 walk-forward 方法論(不在測試段調參)、用目前已確立為「有真實產業結構」的最佳 KG
版本,**Stage-2 RL 資產配置的表現跟餵隨機雜訊 embedding 幾乎一模一樣**,沒有顯著差異。這不是
訓練不足造成的(episode 數已經用驗證段選過最佳值),是這個 attention 機制目前的設計,沒有把
KG 帶來的產業結構訊號轉化成資產配置決策上的優勢。

搭配 KG 驗證的結果,三層結論完全一致:

| 層級 | 結果 |
|---|---|
| KG 結構本身(sector test、link prediction) | ✅ 顯著 |
| Stage-1(個股報酬預測,隔日+週報酬皆測過) | ❌ 不顯著 |
| Stage-2(RL 資產配置,乾淨 walk-forward) | ❌ 不顯著 |

## 解凍 attention encoder 測試(2026-07-23)

假設:`ATTENTION_FREEZE_AFTER_PRETRAIN=True` 讓 Stage-2 RL 只能在 Stage-1 就已經凍結、且已知
KG 與 random 沒有顯著差異的表示上訓練一層薄薄的配置頭,這可能是 Stage-2 測不出 KG 優勢的結構性
原因——如果讓 attention encoder 在 Stage-2 也能繼續跟著 RL 目標調整,KG 版有沒有機會展現優勢?

用 `--no-freeze-attention-after-pretrain`,同樣 episodes=20、驗證段(2025 H1)、5 個 seed:

| | KG 平均 | Random 平均 | Wilcoxon p |
|---|---|---|---|
| Sharpe | 0.8090 | 0.8093 | 0.8125 |
| Total return | 0.10698 | 0.10698 | 1.0 |

**假設不成立。** 解凍後 KG 跟 random 還是幾乎一模一樣,排除了「凍結太早」這個解釋。這代表問題不是
訓練時機,而是**現在這個「拿 KG embedding 當 Q/K 算 attention 權重」的做法,不管凍不凍結,都沒有
把 KG 裡確實存在的產業結構(sector test 已顯著)轉化成對交易決策有用的資訊**——比較像是架構本身
的限制,不是超參數或訓練時機問題。原始資料存於 `runs/unfreeze_attention_test_20260723/`。

## Reward Shaping 測試:把 KG 相似度直接寫進損失函數(2026-07-23)

前兩個測試(凍結/解凍)都只是讓 attention 隱含地「有沒有機會」學到 KG 結構,沒有強迫它一定要用。
第三個測試改成直接在損失函數裡加一項,明確用 KG 結構形塑 reward,不再依賴 attention 自己發現。

`_gradient_ascent()` 原本的損失是 `base_loss = -mean(log(portfolio_growth))`,完全從 `mu`(動作
權重)和 `price_variations` 算出來,不經過 `env.step()` 的 reward。新增
`kg_diversification_penalty(weights, similarity) = mean(w^T S w)`(`meta/attention_fixed5/
algorithm.py`),S 是固定投資組合 5 檔的 KG embedding 餘弦相似度矩陣(對角線恆為 1),懲罰「同時
重壓兩檔 KG 認為相似的股票」。`policy_loss = base_loss + kg_diversification_coef * penalty`。
`run_attention_fixed5.py` 加 `--kg-diversification-coef` 參數(預設 0,行為不變)。
`tests/test_attention_pretraining.py` 加 3 個測試驗證計算正確性。

驗證段(2025 H1)掃描係數:0→0.01→0.05→0.1→0.5→1.0→2.0→5.0,Sharpe 從 0.8111 緩升到 **2.0 時
的 0.8131**(高峰),5.0 略降回 0.8130。選 `coef=2.0`。

最終測試段(2025 H2,5 seed):

| | KG 平均 | Random 平均 |
|---|---|---|
| Total return | 0.1190 | 0.1182 |
| Sharpe | 1.4586 | 1.4605(random 反而略高) |

Wilcoxon p:return=0.3125,sharpe=0.8125,**仍不顯著**。

**發現一個設計層面的解釋**:「random」對照組的隨機 embedding(即使是雜訊)本身也會算出一個非零的
餘弦相似度矩陣,shaping 對 KG 版跟 random 版都會產生「鼓勵分散」的效果,不管矩陣裡的數字是不是真
的有意義。驗證段看到的 Sharpe 隨係數上升,很可能只是「分散化本身」的一般性正則化效果,不是 KG
內容 specifically 有用——這解釋了為什麼兩組都跟著提升,彼此差距卻沒有拉開。

## 三個獨立測試都指向同一個結論

| 測試 | 修改內容 | 結果 |
|---|---|---|
| 凍結(baseline) | Stage-2 只訓練配置頭,attention encoder 凍結 | KG ≈ random,不顯著 |
| 解凍 | attention encoder 在 Stage-2 也能調整 | KG ≈ random,不顯著 |
| Reward shaping | 損失函數直接加 KG 相似度懲罰項 | KG ≈ random,不顯著 |

三個從「訓練時機」到「明確用 KG 結構形塑 reward」的獨立嘗試,都沒有讓 KG 版顯著贏過 random 版。
這是相當收斂的證據:**問題不在訓練細節或超參數,是這個「用 attention 消費靜態 KG embedding」
的架構,目前這樣設計就是沒辦法把 KG 的內容轉化成 random 雜訊做不到的優勢。**

## Attention Score 溫度測試(2026-07-23)

第四個測試:`nn.MultiheadAttention` 內部固定用標準 `1/sqrt(head_dim)` 縮放,沒有額外可調的溫度。
KG embedding 的餘弦相似度數值普遍很接近(熱力圖偏淡),softmax 過差異很小的分數,權重可能太接近
均勻分布。新增 `attention_score_temperature` 參數(`meta/attention_fixed5/models.py`
`KGAttentionStateEncoder`):把 query 除以這個溫度再送進 attention,<1 讓分佈更尖銳、>1 更平均。
`Fixed5AllocationPolicy` 跟 `run_attention_fixed5.py --attention-score-temperature` 同步串接,
預設 1.0 不改變原行為。`tests/test_attention_score_temperature.py` 加 2 個測試驗證預設值等價於
不縮放、溫度越低權重分佈越尖銳。

驗證段(2025 H1)掃描溫度:0.05→0.1→0.2→0.5→1.0→2.0→5.0,Sharpe 在 **0.05 時最高(0.8138)**,
中段(0.1-0.5)反而比 1.0 差,不是單調趨勢(可能是單一 seed 雜訊)。選 `temperature=0.05`。

最終測試段(2025 H2,5 seed):

| | KG 平均 | Random 平均 |
|---|---|---|
| Total return | 0.1215 | 0.1198 |
| Sharpe | **1.4805** | 1.4602 |

Wilcoxon p:return=0.625,sharpe=0.8125,**仍不顯著**,但這是四個方法裡 KG 與 random 平均差距
最大的一次(Sharpe +0.0203)。拆開來看 5 個 seed 的 Sharpe 差值:-0.013、-0.008、+0.086、+0.066、
-0.029——2 個 seed 大幅偏向 KG(+0.086、+0.066),3 個 seed 小幅偏向 random,樣本數 5 個對這種
不對稱的分佈不夠有檢定力,不能說已經證實有效,但方向上比前三個方法更值得後續用更多 seed 追蹤。

### 擴大到 15 seed:5-seed 的「希望」是雜訊(2026-07-23)

跟先前 Stage-1 IC 一模一樣的模式:用 15 個 seed(41-55)重跑同一組設定(`temperature=0.05`,
`episodes=20`,最終測試段 2025 H2),原本 5-seed 看到的正向差距完全消失、甚至反轉:

| | 5 seed | 15 seed |
|---|---|---|
| KG 平均 Sharpe | 1.4805 | 1.4698 |
| Random 平均 Sharpe | 1.4602 | **1.4753**(現在反而略高) |
| 方向一致的 seed 數 | 3/5 偏 KG | 只有 5/15 偏 KG |
| Wilcoxon p(sharpe) | 0.8125 | 0.5245 |

**確認:attention 溫度調整也沒有產生真實、穩定的 KG 優勢。** 5-seed 的正向訊號是小樣本雜訊,
不是真效果。

## 四個獨立測試的總結(皆已用足夠 seed 數驗證)

| 測試 | 修改內容 | 最終測試段 Sharpe 差(KG−random) | 樣本數 | 顯著? |
|---|---|---|---|---|
| 凍結(baseline) | Stage-2 只訓練配置頭 | +0.0053 | 5 seed | 否 |
| 解凍 | attention encoder 可調整 | -0.0003(驗證段) | 5 seed | 否 |
| Reward shaping | 損失函數加 KG 相似度懲罰 | -0.0019 | 5 seed | 否 |
| Attention 溫度 | query 除以溫度再算 attention | -0.0054(15 seed 修正後) | 15 seed | 否 |

四個方法在充分樣本數下都沒有顯著效果,且都指向同一個結論。

## 架構層級的根本問題(2026-07-23 討論)

四個修改都沒用,回頭看架構本身的假設:**整個 attention 機制的前提是「KG 相似度應該拿來加權平均
其他公司的股價變動,當作這家公司的狀態表示」——這等於假設「KG 認為相似的公司,股價會一起動」。
但 Mantel test 已經證實這個假設不成立**(KG 相似度矩陣 vs 報酬相關性矩陣,日報酬、週報酬皆不
顯著)。用一個跟報酬共動無關的相似度去加權平均股價資料,某種程度上是在稀釋訊號而非加強訊號——
這可能才是四種修改方式都救不起來的根本原因,不是訓練細節。

架構還有幾個值得檢視的設計:

1. KG embedding 完全靜態(離線訓練一次,整個回測期間 Q/K 不變),V(股價特徵)卻是每天滾動更新
   ——用寫死的結構先驗去挑該看哪些動態資料,解凍測試也證實讓它動起來沒有幫助。
2. Reward 訊號離 attention 機制太遠:loss 是整個投資組合的對數成長率,這個訊號被大盤共動雜訊
   主導(報酬相關性熱力圖顯示幾乎所有股票都跟大盤一起漲跌),KG 微調配置能貢獻的訊噪比本來就小。
3. 除了 attention 的 Q/K/V,還有一個平行的「配置頭事後諸葛預訓練」(`_allocation_head_pretrain_loss`):
   用 `softmax(實際報酬 × scale)` 當事後最佳權重的假想目標,預訓練配置頭去模仿。但這個監督訊號
   一樣是從同一個 KG 加權出來的 state 起算,state 本身沒有攜帶有用資訊的話,這層預訓練也救不回來。

**若要重新設計,核心問題不是「attention vs GNN」的選擇,而是「拿什麼當 Q/K 的相似度基礎」**——
不管換不換成 GNN,只要還是假設「KG cosine similarity = 報酬共動」,同樣的邏輯漏洞都會存在。GNN
若是端到端訓練(直接用交易目標的梯度調整圖上的訊息傳遞方式),理論上能自己學出「哪些關係型別/
圖結構模式真的有助於預測報酬」,而非被迫假設 KG 相似度本身就是有用的量,這才是相對現在架構真正
的優勢所在。

## Dynamic Query 測試:修掉「Q/K 完全靜態」這個具體架構缺陷(2026-07-23)

回頭檢視程式碼,發現一個比前四個測試都更根本的問題:`KGAttentionStateEncoder.forward()` 裡
`query = kg_selected`、`key = kg_all` **完全是靜態的 KG embedding,不隨日期/市場狀況變化**——
只有 `value`(股價特徵)每天不同。這代表 `softmax(QKᵀ)` 算出來的 attention 權重,整個訓練+測試
期間永遠是同一組固定數字,數學上等價於一個係數寫死的線性層,不是真正會「根據當下市場狀況決定
關注誰」的 attention。

修法:新增 `dynamic_query` 選項(`meta/attention_fixed5/models.py`)。Q 不再只是
`kg_selected`,改成 `combine(kg_selected, 該公司自己最近的股價編碼)`(串接後過一層
`Linear→LayerNorm→ReLU→Linear` 融合),K 維持不變。`Fixed5DataBundle` 新增 `selected_indices`
欄位(原本 `load_fixed5_data` 內部就有算,只是沒有外露),用來從 30 檔的股價編碼裡挑出 5 檔
自己的部分。`run_attention_fixed5.py` 加 `--dynamic-query` 參數。`tests/
test_attention_score_temperature.py` 加 3 個測試,包含直接證實原本 bug 的
`test_static_query_gives_identical_weights_regardless_of_market_data`。

**驗證確實生效**:`action_variability` 的標準差從前四個方法的 0.0002-0.014 範圍,在 dynamic
query 下明顯變大(0.006-0.014),policy 真的做出更有分化的決策了。

驗證段(2025 H1)episode 數 sweep:1→5→10→20→30→50,Sharpe 單調遞減(0.8167→0.8090→0.8018→
0.8008→0.8003→0.7948),**episodes=1 最好**。

最終測試段(2025 H2,15 seed,`episodes=1` + `dynamic_query`):

| | KG 平均 Sharpe | Random 平均 Sharpe |
|---|---|---|
| 15 seed | 1.4619 | 1.4683(random 仍略高) |

Wilcoxon p=0.5245,只有 6/15 seed 偏向 KG。**修掉「Q/K 完全靜態」這個具體架構缺陷之後,依然
沒有產生顯著的 KG 優勢。**

## 五個獨立測試的總結(皆已用足夠 seed 數驗證)

| 測試 | 修改內容 | 最終測試段 Sharpe 差(KG−random) | 樣本數 | 顯著? |
|---|---|---|---|---|
| 凍結(baseline) | Stage-2 只訓練配置頭 | +0.0053 | 5 seed | 否 |
| 解凍 | attention encoder 可調整 | -0.0003(驗證段) | 5 seed | 否 |
| Reward shaping | 損失函數加 KG 相似度懲罰 | -0.0019 | 5 seed | 否 |
| Attention 溫度 | query 除以溫度再算 attention | -0.0054 | 15 seed | 否 |
| **Dynamic query** | **Q 融合當下市場狀況(修掉靜態 Q/K 的架構缺陷)** | **-0.0064** | **15 seed** | 否 |

連我們精準定位、直接對症下藥修掉的「Q/K 完全靜態」這個具體架構缺陷,修完之後依然測不出 KG 相對
random 的優勢。這把問題排除到比「attention 機制某個具體技術缺陷」更深的層次——回到最初那個
發現:**KG 相似度本身跟報酬共動沒有顯著關係(Mantel test 不顯著),不管用什麼方式把這個相似度
餵進模型、不管模型多有表達能力,這個 architecture 都沒有材料可以蓋出一個對交易有用的優勢。**

## 換投資組合與回測期間(2026-07-24)

原本固定投資組合(AAPL/NVDA/AMZN/IBM/MSFT)幾乎全部是同一個科技大集團(DEF14A 顯示彼此互相
連結),沒有涵蓋兩個明確不同的產業,KG 分散化的價值缺乏發揮舞台。改用 SEC DEF14A 交叉驗證後
的兩組配對:**MSFT/NVDA(科技,雙向連結)+ GS/JPM(金融,雙向連結,也是全道瓊報酬相關性最高
的一對,週報酬 rho=0.76)+ KO(消費品,低相關對照)**。

回測期間也從資料本身找,而非憑印象猜:計算 2018 年至今道瓊 30 家的月度報酬橫斷面標準差(離散度
越高代表產業分化/輪動越明顯),挑出**跟 KG 申報年份相近**(避免時序方向問題)、離散度又高的
窗口:訓練 2024-05-01~2025-05-01、驗證 2025-05-01~2025-11-01(含 2025-05 高離散月)、最終
測試 2025-11-01~2026-04-30(含 2026-01/02/04 三個高離散月)。

驗證段 5 seed 快速比較(`dynamic_query`,`episodes=20`,無 reward shaping):KG 平均 Sharpe
4.150,random 平均 4.160,只有 2/5 seed 偏向 KG——**跟前五個方法一樣,還是雜訊等級的差異**,
換投資組合、換期間都沒有改變結論。action_variability(0.005-0.018)確認模型確實有交易,不是
又卡在等權重不動。

## 直接檢查訓練好的 attention 權重(2026-07-24)

不重新訓練,直接把已訓練模型在測試期間的 `attention_weights.npy` 拉出來看(對日期、對 head
取平均),確認每家公司的 query 實際上把注意力放在誰身上:

| Query | 前 6 名被關注公司 |
|---|---|
| MSFT | BA、AAPL、KO、CVX、JNJ、AMZN(**沒有 NVDA**) |
| NVDA | BA、JNJ、IBM、AAPL、NVDA(自己)、CVX(**沒有 MSFT**) |
| **GS** | **JPM**(✓,對!)、KO、CVX、V、MCD、AAPL |
| JPM | AAPL、V、JPM(自己)、KO、AMZN、BA(**沒有 GS**,跟 GS→JPM 不對稱) |
| KO | V、SHW、JPM、WMT、HD、AAPL |

**混合結果**:GS 的 attention 正確地把 JPM 排第一(全道瓊真實相關性最高的一對,也是 DEF14A
確認的同業),證明機制有能力反映真實結構。但 MSFT-NVDA 雙向都沒有互相關注(儘管也是確認過的
強連結配對),GS→JPM 這個連結本身也是單向不對稱(JPM 自己沒有把 GS 排進前六),而且 **BA
(波音,跟這兩家完全不同產業)同時是 MSFT 跟 NVDA 關注度最高的公司**,比較像某種通用的熱門
吸引節點雜訊。整體權重分布也偏平坦(均勻分布基準約 0.033,大部分數字落在 0.036-0.046)。

**這解釋了為什麼整體 Sharpe 測不出顯著優勢**:機制偶爾學對(GS→JPM)、常常學錯或學到雜訊
(BA 熱門節點、MSFT-NVDA 沒連上),平均起來被抵銷掉,不是「完全沒學到東西」,也不是「學得很好」。

## 正控制組測試:整套訓練機制學不學得到「保證有用」的訊號?(2026-07-24)

前面所有測試都在問「KG 有沒有比 random 好」,但沒驗證過**整套 RL 訓練機制是不是連一個保證有用
的訊號都學不會利用**——如果連這個都做不到,那「KG 沒有顯著優勢」就可能是訓練機制的天花板,
跟 KG 內容好不好無關,前面五個修改也不可能測出真正的答案。

設計一個「作弊」embedding:用驗證段(2025-05-01~2025-11-01)5 檔投資組合的**真實已實現報酬
相關性矩陣**,透過特徵值分解反推出一組向量,讓這組向量的餘弦相似度精確等於真實報酬相關性
(GS-JPM=0.81,MSFT-KO=-0.14 等,見下方矩陣),其餘 25 檔universe填入符合真實 embedding 均值/
標準差的隨機雜訊。這是刻意用驗證段自己的未來報酬「作弊」,只用來測試架構的學習能力上限,不是
真實回測。

```
        MSFT   NVDA   GS     JPM    KO
MSFT    1.00   0.46   0.40   0.42  -0.14
NVDA    0.46   1.00   0.44   0.42  -0.26
GS      0.40   0.44   1.00   0.81  -0.10
JPM     0.42   0.42   0.81   1.00  -0.11
KO     -0.14  -0.26  -0.10  -0.11   1.00
```

15 seed、`episodes=20`、`dynamic_query`,三組(random 下限 / 真實 KG / 作弊上限)在同一驗證段
的完整比較:

| 組別 | 平均 Sharpe |
|---|---|
| Random(下限) | 4.1652 |
| 真實 KG | 4.1556(比 random 還低) |
| 作弊 embedding(上限) | 4.1817 |

作弊組 vs random:Sharpe 平均高 0.0165,11/15 seed 同方向,p=0.073(接近但未達 0.05 顯著)。
真實 KG vs random:效果量更小、方向更不一致。

**結論**:整套 RL 訓練機制**不是完全學不到任何東西**——餵它一個相似度=真實報酬相關性的完美
訊號,確實表現出比 random 更好、更一致的傾向(雖然只到邊緣顯著,還沒到能宣稱「顯著」的程度)。
這排除了「訓練機制本身有天花板,不管餵什麼都沒用」的最悲觀解釋,反過來確認:**真實 KG 沒有
顯著優勢,是因為 KG 的相似度結構本身沒有攜帶足夠接近真實報酬相關性的資訊,不是訓練/架構的問題。**

### 新的 KG 驗收指標:報酬相關性捕捉率

用「random 下限 → 真實 KG → 作弊上限」這三個錨點,可以定義一個比 Mantel test 的抽象 p 值更
直觀、可比較的量表:

```
KG 報酬相關性捕捉率 = (真實KG平均Sharpe − random下限) / (作弊上限 − random下限)
```

本次結果:`(4.1556 − 4.1652) / (4.1817 − 4.1652) = -58%`。

不只是「沒有捕捉到」,而是**負值**——真實 KG 這一版,在這個投資組合/期間下,比完全沒有結構
資訊的隨機雜訊還要差一點。這是一個誠實、可量化、未來每次改進 KG 建構方式都可以重新算一次來
追蹤進度的驗收指標,建議正式收進 KG 驗證套件,作為 Mantel test 之外的第二個「KG 是否具備報酬
相關性」的量測方向。

## 正控制組的第二次測試:換期間/組合,效果消失(2026-07-24)

為了確認第一次正控制組的結果(作弊 embedding 顯示比 random 更好、更一致的傾向)不是特定期間/
組合的偶然,換一組完全不同的設定重測:**COVID 崩盤+反彈期**(訓練 2019-02-01~2020-02-01,
測試 2020-02-01~2020-08-01),投資組合換成橫跨三個產業的 **AAPL(科技)、CSCO(網通)、
BA(航太)、CAT(工業)、CVX(能源)**。同樣手法:用測試期真實報酬相關性反推作弊 embedding。
`run_attention_fixed5.py` 新增 `--portfolio-tickers` 參數,不用再手動改 `experiment_config.py`
才能換投資組合。

這段期間的真實相關性矩陣本身跟先前不同——**所有配對相關性都偏高(0.50-0.79),沒有明顯分群**,
這是「危機時期相關性趨近於 1」的典型現象(系統性風險主導,而非產業別差異)。

15 seed 結果:

| | 平均 Sharpe | 同方向 seed 數 | Wilcoxon p |
|---|---|---|---|
| **COVID 期間**:作弊 − random | +0.0051 | 8/15(接近對半) | 0.72(完全不顯著) |
| (對照)2025 H2:作弊 − random | +0.0165 | 11/15 | 0.073 |

**這次作弊優勢幾乎消失。** 合理解釋:當所有股票的真實相關性都偏高且彼此接近時,就算完美知道
這個相關性結構,「該多關注誰」提供的差異化資訊也很有限——不管關注誰,混合出來的結果都差不多。
2025 H2 那段有清楚的兩個產業集群分化(科技 vs 金融),知道「誰跟誰是一掛」才真正有辨識度可以
利用。

**新發現**:正控制組能不能展現優勢,本身是**有條件的**——取決於當下市場有沒有清楚的分化結構
可以利用,不是任何時候「完美知道真實相關性」都必然帶來交易優勢。這代表就算 KG 完美反映報酬
相關性,也不保證任何時期都能轉化成交易效用;**KG 報酬相關性捕捉率**這個指標本身也應該視市場
regime(是否有清楚分化 vs 系統性危機)分開解讀,而非只看單一期間的單一數字。

## 換倉行為直接測試:產業輪動 + 空頭防禦(2026-07-25)

前面所有測試看的都是聚合績效(Sharpe/總報酬)。就算作弊 embedding 表現較好,也無法排除「模型
只是選對了一個固定的靜態配置,剛好在這段期間表現不錯」——不代表它真的隨市場變化調整權重。這次
直接設計兩個「答案很明顯要換倉」的場景,拿掉聚合指標,直接看 `action_weights.csv` 的逐日/逐月
權重軌跡。兩次都用作弊 embedding(相似度=測試期真實報酬相關性),`--dynamic-query`、
`episodes=20`、單一 seed=41(權重完全沒動,不需要統計檢定就能下結論)。

### 測試一:產業龍頭輪動

`NVDA/MSFT/CSCO`(科技)vs `JNJ/KO`(防禦,受限於 `Fixed5DataBundle._validate_tickers` 硬性
要求剛好 5 檔,原本想要的 3+3 改成 3+2)。訓練 2021-06-01~2022-06-01,測試
2022-06-01~2023-12-31——這段期間有真實驗證過的龍頭輪動:2022 到 2023 年 3 月防禦股領先,
2023 年 5 月起科技股(尤其 NVDA)決定性反超(逐月 `tech_minus_def` 累積報酬差已驗證)。

結果:`total_return=0.351`,`sharpe=1.015`(聚合數字不差)。但逐月平均 `tech_weight`
(NVDA+MSFT+CSCO):

| 2022-06 | 2022-09 | 2022-12 | 2023-03 | 2023-06 | 2023-09 | 2023-12 |
|---|---|---|---|---|---|---|
| 0.606 | 0.628 | 0.612 | 0.612 | 0.620 | 0.605 | 0.594 |

整整 18 個月,全區間 `tech_weight` 只在 **0.588–0.634** 之間微幅震盪,完全沒有反映 2023 年 5
月那次明顯的龍頭反轉。權重甚至從第一天(等權重 0.2/0.2/0.2/0.2/0.2)到最後一天幾乎沒有偏離。

### 測試二:高波動 vs 低波動的空頭防禦

`NVDA/BA/CRM`(高波動,依 2021 年日報酬標準差排序選出)vs `PG/JNJ`(低波動)。訓練
2021-01-01~2022-01-01,測試 2022 全年空頭。實現報酬落差極大:NVDA `-51.5%`、CRM `-48.1%`、
BA `-8.4%`、PG `-7.0%`、JNJ `+3.0%`——教科書等級的「該轉去防禦股」情境。

結果:`total_return=-0.230`,`sharpe=-0.722`。逐月平均 `highvol_weight`(NVDA+BA+CRM):

| 2022-01 | 2022-04 | 2022-07 | 2022-09 | 2022-12 |
|---|---|---|---|---|
| 0.598 | 0.595 | 0.616 | 0.604 | 0.617 |

全年只在 **0.580–0.629** 之間震盪,同樣完全沒有轉向防禦股,即使高波動這組全年跌了近一半、
低波動那組是正報酬。

### 兩次測試的共同發現與根因分析

兩個獨立設計的場景(產業龍頭反轉、空頭防禦輪動),用的是彼此不重疊的股票組合和期間,結果卻
高度一致:**權重曲線幾乎是一條平線**,不管背後餵的是「相似度=真實報酬相關性」的完美 KG 訊號,
也不管市場已經出現多戲劇性的落差。這比先前「KG 內容品質不夠好」的結論更根本:**問題不在於
攻擊 KG 有沒有攜帶對的資訊,而是整條 RL pipeline 的資產配置輸出本質上是靜態的,幾乎不隨行情
展開而調整**。

追出去看程式碼(`meta/attention_fixed5/models.py`),找到一個高度可疑的機制:

```python
ATTENTION_ACTION_BLEND = 0.025   # experiment_config.py
```

`Fixed5AllocationPolicy._blend_stock_weights_with_last_action`:

```python
blended = self.action_blend * weights + (1.0 - self.action_blend) * last_stock_weights
```

也就是說,`stock_scorer` 算出來的原始目標權重(`weights`),每天只能用 2.5% 的比例混進實際配置,
97.5% 沿用前一天的權重。這是一個半衰期約 **27 個交易日**(`ln(0.5)/ln(0.975)≈27.4`)的指數
平滑(EMA)。兩次測試都確認了這個參數:`run_config.json` 中 `action_blend=0.025`。

這個發現能解釋觀察到的現象,但要分兩種情況看:
- 若 `stock_scorer` 的原始目標本身**持續**偏向某一邊(例如反轉後應該持續押防禦股數月),27 天
  半衰期不至於完全抹平——18 個月測試期內應該早就收斂到新目標附近,但沒有看到這件事發生。
- 這代表更可能的情況是:**`stock_scorer` 的原始目標本身在任何時間點都相差不大**(即使 attention
  context 因為 `dynamic_query` 已經每天不同),`action_blend` 只是把這個原本就很小的日內雜訊
  再平滑掉,讓最終權重看起來更加靜止。換句話說,`dynamic_query` 修好的是 attention **輸入**的
  時變性,但沒有解決下游 `stock_scorer`→`softmax` 這段輸出端本身缺乏時變的問題。

這兩種可能都還沒有被排除——需要直接記錄 blend 前的原始 `weights`(目前只存了 blend 後的
`action_weights.csv`)才能確定是哪一種。這是下一步最優先、成本最低的診斷實驗。

### 追加發現:`action_blend` 不只平滑輸出,還在削弱 RL 梯度本身

追進 `_gradient_ascent`(`meta/attention_fixed5/algorithm.py:288`)發現一個比「輸出被平滑」更
根本的問題。RL 損失是這樣算的:

```python
mu = self.train_policy.mu(market_context, last_actions)   # 內部又呼叫了一次 blend
base_loss = -torch.mean(torch.log(portfolio_growth))       # portfolio_growth 由 mu 算出
```

`policy.mu()` 內部呼叫的正是同一個 `_blend_stock_weights_with_last_action`:
`blended = action_blend * weights + (1-action_blend) * last_stock_weights`。這代表反向傳播
時,`d(blended)/d(weights) = action_blend = 0.025` 這個常數會直接乘進鏈式法則裡——`stock_scorer`
(負責「幫哪一檔配更高權重」的那一段網路)拿到的梯度,天生就被這個超參數衰減成只剩 1/40。也就是
說 `action_blend=0.025` 不只是「推論時把輸出抹平」,連**訓練時教這段網路怎麼判斷輪動的梯度訊號
本身也被同一個係數打了 40 倍折扣**。這解釋了為什麼即使餵完美的作弊 embedding、`dynamic_query`
已經讓 attention 輸入逐日不同,`stock_scorer` 這一層還是學不太動——它收到的學習訊號本來就被
設計成很小。

這個參數是在舊版 `PARAM_SWEEP_REPORT`(已知有 confound,見前面章節)於**2025 年下半年這種波動
不大的期間**調出來的——在那種期間,壓低 blend、減少換倉噪音確實是對的方向;但同一個值套用到本
節這種「答案很明顯要換倉」的場景,反而直接把換倉需要的梯度也一起壓沒了。

## 對「RL 那邊修改甚麼有機會讓它學到要換倉」的具體建議

依可驗證成本由低到高排序,前兩項幾乎零成本、應該先做,能直接告訴我們問題出在「訊號被壓制」還是
「訊號根本沒被學到」:

1. **把 `action_blend` 調高甚至關掉(=1.0)重跑同兩個測試**——如果原始 `stock_scorer` 其實想
   換倉、只是被 blend 壓住,調高後應該會看到權重明顯跟著行情走(即使伴隨更高換手成本/雜訊);
   如果調高後權重還是死平,代表問題不在 blend,而在 `stock_scorer` 這段網路本身沒學到差異化。
2. **在 `mu()` 裡另外記錄 blend 前的原始 `weights`**(目前只存 blend 後的
   `action_weights.csv`),不需要重跑訓練就能直接檢查:對同一批已訓練好的模型,拿測試期資料
   forward 一次,比較 blend 前後的權重差異有多大。這是純診斷、不影響訓練,能最快排除/確認上面
   的 gradient-attenuation 假說。
3. **把 reward 從「絕對報酬」改成「相對於等權重基準的超額報酬」**:目前 `base_loss` 獎勵的是
   `-log(portfolio_growth)` 這個絕對值,換倉正確與否只是這個大數字裡的一個小修正項,訊號本來就
   容易被稀釋(尤其疊加上面第 40 倍的梯度衰減)。改成先扣掉等權重基準的當日報酬再算 loss,能讓
   梯度更直接、更大比例地指向「換倉這個決策本身值不值得」,而不是被同一批股票的共同漲跌淹沒。
4. **把 `ATTENTION_HEAD_PRETRAIN_COEF`(Stage-1.5 的 hindsight-imitation 熱身)在 RL 訓練過程
   中逐步退火到 0**,而不是全程固定 1.0。這個熱身是用整個訓練窗口的「事後最優」報酬 softmax 出
   目標權重,如果訓練窗口本身沒有单一持續的贏家,這個目標本身就偏向接近等權重——等於一開始就把
   網路釘在「不太想換倉」的初始化附近,而 RL 階段的梯度(還被 blend 衰減過)未必有能力把它拉出來。
5. **加入跨日的記憶(如 GRU/LSTM 疊在 state encoder 輸出後面)**,如果 1-4 都做了、原始訊號確實
   存在但還是無法穩定累積成持續的換倉行為——現在的架構每天都是獨立重新判斷(30 天價格窗+KG),
   沒有機制去累積「我已經連續好幾天看到反轉訊號,該真的換倉了」這種跨日信念。這是成本最高、
   也最接近使用者先前提過的「GNN/GRU 比較合邏輯」方向,建議留到 1-4 排除掉「超參數/reward 設計」
   這類低成本解釋之後再做。

## `action_blend=1.0` 消融實驗:推翻「訊號被壓住」假說(2026-07-25)

用完全相同的作弊 embedding、投資組合、訓練/測試期間,只把 `--action-blend` 從 0.025 調到 1.0
(關閉平滑),同 seed=41 重跑兩個換倉測試。Run 目錄:`runs/blend_ablation_20260725/`。

**日內變動幅度確實暴增**——單日權重標準差從原本 0.007-0.013 提高到 0.13-0.15(約 10 倍),
權重範圍從原本窄幅的 0.59-0.63 擴大到幾乎打滿上下限的 0.20-0.92。這證實反向傳播確實有把梯度
傳到 `stock_scorer`,原始輸出本身不是死的、也不是完全不隨市場變化。

**但逐月平均軌跡沒有變成預期中的規律換倉,反而更像雜訊**:

| | 2022-06~2022-12 平均 tech_weight | 2023-05~2023-12 平均 tech_weight(真實龍頭反超之後) |
|---|---|---|
| blend=0.025(原始) | ~0.615 | ~0.610(幾乎沒變) |
| blend=1.0 | ~0.625 | ~0.604(**不升反降**,方向與真實反轉相反) |

聚合績效也沒有變好,反而更差:

| | Sharpe(產業輪動) | Sharpe(空頭防禦) |
|---|---|---|
| blend=0.025 | 1.015 | -0.722 |
| blend=1.0 | 0.626(↓) | -0.960(↓,更差) |

**結論修正**:前一節提出的「`action_blend` 的梯度衰減把好訊號壓沒了」這個假說,被這次實驗
**推翻**。拿掉平滑後暴露出來的不是一個被隱藏的、跟真實龍頭反轉方向一致的訊號,而是**振幅更大
但方向沒有規律的雜訊**——`stock_scorer` 本身根本沒有學到「這是該換倉的時候」這種跨日持續性的
判斷能力,不管梯度有沒有被 `action_blend` 打折都一樣。`action_blend=0.025` 客觀上反而是在替
一個尚未學會規律、只會產生雜訊的原始輸出做**降噪**,調高它不是解方,反而讓雜訊直接反映到實際
配置上、拖累績效。

這把根因從「超參數設計不當」進一步收斂到**下游決策網路本身沒有學到時間上連貫的判斷依據**——
呼應前面「對 RL 修改的具體建議」清單中第 3-5 項(reward 相對化、pretrain 熱身退火、跨日記憶)
才是接下來該優先驗證的方向,而不是繼續在 `action_blend` 這類推論期後處理參數上打轉。

## 相對報酬 reward + 解凍 attention encoder:同樣沒有換倉(2026-07-25)

### 實作修正:單純減基準是無效改動

原計畫的「reward 改成相對等權重基準的超額報酬」,字面實作(從 loss 減掉一個不依賴 mu 的常數)
在數學上其實是**無效改動**——這個訓練迴圈是直接對 `mu` 做可微分反向傳播(不是 REINFORCE 那種
用 baseline 降低梯度方差的做法),對一個跟 `mu` 無關的常數項,`d(loss)/d(mu)` 不會因為减掉它
而改變。改用真正會影響梯度的版本:`relative_selection_reward()`
(`meta/attention_fixed5/algorithm.py`)把每天的原始報酬換成**當天橫截面 z-score**
(減去當天均值、除以當天標準差)再跟權重做內積,直接獎勵「有沒有把權重壓對當天真正的相對贏家」,
且用波動度正規化避免被普漲普跌的高波動日主導。新增 `--relative-reward-coef`(0=行為不變)。

### 結果:加了也沒用,不是量級問題

`relative_reward_coef=1.0`(其餘設定與 `action_blend=0.025` 基線相同,`freeze_attention_after_pretrain`
維持預設 `True`)結果幾乎沒變:

| | Sharpe(產業輪動) | Sharpe(空頭防禦) |
|---|---|---|
| coef=0.0(=原始基線) | 1.015 | -0.722 |
| coef=1.0 | 1.016 | -0.721 |

檢查 `loss.csv` 排除了「係數太小」的解釋:`relative_reward` 這項的量級其實比 `base_loss`
還大(中位數比值 `|base_loss|/|relative_reward| ≈ 0.32`),梯度大小不是瓶頸。

### 追查到凍結:解凍後仍然沒有換倉

`run_config.json` 確認這些 run 全部用預設 `freeze_attention_after_pretrain=True`——Stage-1
pretrain 完後,`state_encoder`(含 `dynamic_query` 的 `query_fusion` 模組)整個被凍結,RL 階段
不管用 `base_loss` 還是 `relative_reward`,能訓練的都只有一個 8644 參數的小 `stock_scorer` 頭,
接在**從沒被教過要區分「誰是今天的相對贏家」**的凍結表徵後面。加 `--no-freeze-attention-after-pretrain`
讓整個網路(含 `query_fusion`)端到端用 RL 微調,同樣測 `relative_reward_coef∈{0,1}`:

| | Sharpe(產業輪動) | Sharpe(空頭防禦) |
|---|---|---|
| 解凍,coef=0.0 | 1.001 | -0.727 |
| 解凍,coef=1.0 | 1.019 | -0.729 |

逐月權重軌跡(解凍+`coef=1.0`):產業輪動 `tech_weight` 全年只在 **0.575–0.634** 之間震盪
(範圍比凍結版還窄),空頭防禦 `highvol_weight` 在 **0.575–0.618** 之間——跟前面所有測試一樣,
完全平,沒有任何換倉跡象。

### 收斂結論

四種獨立的下游修改(拿掉 `action_blend` 平滑、加相對報酬 reward、解凍 encoder、以及兩兩組合)
**全部沒有解鎖換倉行為**。這排除了「超參數設計不當」「reward 設計不對」「encoder 被凍結沒機會學」
三種解釋,把根因進一步收斂到僅剩兩個候選:

1. **Stage-1.5 hindsight-imitation 熱身(`_allocation_head_pretrain_loss`)把網路釘死在接近
   等權重的初始化附近**,不管下游 RL 目標怎麼改,20 個 episode 的訓練量都不足以把它拉出來——
   需要測試把 `head_pretrain_coef` 降低或在 RL 訓練過程中退火到 0。
2. **架構本身缺乏跨日記憶/信念累積機制**——現在每天都是獨立重新判斷(30 天價格窗
   + 當天的 KG attention context),沒有辦法把「已經連續好幾天觀察到反轉訊號」累積成一個持續的
   換倉決策。這需要在 state encoder 輸出後面加 GRU/LSTM 這類跨日狀態,是目前候選中成本最高、
   但也是唯一還沒實測過的架構級改動。

## 待辦

- [ ] 測試把 `head_pretrain_coef` 調低(如 0.1)甚至關掉(0.0,搭配
      `--no-freeze-attention-after-pretrain` 才有意義),看拿掉/減弱 hindsight-imitation 熱身
      的錨定效果後,`relative_reward` RL 訓練是否終於能學出跨日連貫的換倉規律。這比 GRU/LSTM
      架構改動成本低,應該先測。
- [ ] 若熱身退火仍無法解鎖換倉行為,才上跨日記憶架構(GRU/LSTM 疊在 state encoder 輸出後面)——
      這是目前清單中成本最高的選項,也是本節四個消融實驗都排除掉其他解釋後,唯一還沒被測試過、
      理論上仍可能解決問題的方向。
- [ ] 依上面「對 RL 修改的具體建議」清單,由上而下(blend→reward 相對化→pretrain 退火→跨日記憶)
      依序測試,每做完一項就記錄結果,不要跳著做,避免又混入 confound
- [ ] 把「KG 報酬相關性捕捉率」正式寫進 `meta/graphrag/validation.py`/`setup/10_validate_kg.py`
      的驗證套件,讓每一版新 KG 都能算一次這個指標,追蹤是否有進步
- [ ] 檢查 reward 訊號量級(`base_loss` 相對其他梯度的量級)是否太小、訊號被稀釋——雖然正控制組
      證明訊號能傳導,但幅度不大(僅邊緣顯著),值得確認是否訊號傳導效率本身也有改善空間
- [ ] 五個獨立架構修改 + 換投資組合/期間 + attention 權重檢查 + 正控制組,完整證據鏈指向同一個
      根因:**KG 相似度與報酬共動關聯薄弱,不是 RL/attention 架構的問題**。下一步的重心應該回到
      「能不能找到/建構出跟報酬共動更相關的圖結構訊號」(更豐富的資料源、不同的圖建構方式),而非
      繼續在下游模型架構上打轉
- [ ] 與指導教授討論:KG 結構驗證(sector test、link prediction)是扎實的正面結果,RL/attention
      下游效用經過完整驗證流程(五種架構修改+換組合/期間+權重檢查+正控制組,含新的「報酬相關性
      捕捉率」量化指標 -58%)都未達顯著且方向為負,這是一個完整、一致、經得起檢驗的負面結果,
      論文應如何呈現這個發現鏈
