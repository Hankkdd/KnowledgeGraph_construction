# 研究規範：gate 設計、control 定義與報告格式

承接自 `GraphRagTrading/0601_atten_V2` 的實驗方法，是本 repo 所有實驗的共同約束。
執行任何 gate 之前先凍結設計，執行之後產出符合 §5 格式的紀錄。

## 1. 核心判定原則

只贏 no-graph 或只贏 self，不算圖有語義價值。模型容量差異、正則化效果、
邊數增加都可能造成表面改善。real 必須同時勝過保留結構但破壞語義的 controls。

MSE 改善只能解讀為數值校準，不能單獨稱為 alpha。主判定一律看橫斷面排序指標。

## 2. Controls

每個 graph family 固定相同的模型容量與訓練設定，至少包含：

| control | 定義 |
|---|---|
| `no_graph` | 只使用價格特徵，不傳訊息 |
| `self` | 只保留 self-loop，沒有跨公司邊 |
| `relation_shuffle` | 保留拓樸、度數與 relation 頻率，只打亂 relation label |
| `topology_shuffle` | 保留節點與邊數，使用 degree-preserving rewiring |
| `real` | 未破壞的圖 |

`relation_shuffle` 只適用於 typed graph。單一 relation 的圖（如 same-sector）
必須明確標記為「不適用」，不可假造 typed control。此時至少保留 topology-shuffle
與 density/degree-matched random。

dynamic graph 的 shuffle 要逐日處理，不可只打亂一次再套用到所有日期。

## 3. 時間正確性

- 圖與特徵在決策日 `t` 只能使用 `t` 當日以前的資料；target 從 `t+1` 起算。
- target horizon 為 `h` 時，train/test 邊界加入 purge gap，避免 forward window 重疊。
- 所有 scaler、graph normalization、target normalization 只在 train fit。
- dynamic graph 必須以 `graph[t]` 的形式傳入模型，不可預先整批建好再回填。
- 關係型資料保留 `known_as_of`（揭露日／filing date），不可用資料的 `datadate`
  當作可得日期——財報期末與揭露日之間有數月落差。

## 4. Seed 與統計門檻

| 階段 | seeds | 門檻 |
|---|---|---|
| exploratory | 41–45 | 5/5 方向一致才進 confirmation |
| confirmation | 41–50 | 至少 9/10 勝出 |

同時報 paired mean difference、win count、effect size 與 Wilcoxon p-value。
主要決策不能只依 p-value。

Primary metric 依任務指定：

- 報酬任務：daily cross-sectional Rank IC。secondary 為 directional accuracy、top-k hit、MSE。
- 風險任務：預先指定的 realized covariance／pair spread／volatility 指標。

## 5. 報告格式

每個完成的 gate 產出一份 `docs/YYYY-MM-DD_<name>.md`，回答：

```text
Research question:
Why this experiment is next:
Hypothesis:
Frozen data/universe/graph split:
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
```

每次 run 必須記錄 git commit、command line、seed list、train/test 日期、
universe hash、graph snapshot hash、target horizon、model config、controls、
runtime 與失敗樣本。

## 6. 已凍結的負結果

以下結論已由前身 repo 建立，不重複測試，也不作為新實驗的隱含前提：

| 結論 | 依據 |
|---|---|
| RL consumer pipeline 沒壞 | Gate 0 oracle 5/5 勝 random/equal-weight |
| 架構在訊號強且即時時能學會圖 | Gate 1 v1.1 合成動態圖 40/40 |
| GraphRAG 語義 KG 在 Dow 30 上無交易增益 | static/long-horizon/dynamic event KG 三路皆未過 |
| GICS 靜態圖在 44 檔上無 Rank IC 增益 | 2026-08-26 Phase 1，h20/h60 均 real 輸給 controls |
| 每日更新的相關圖在 44 檔上無 Rank IC 增益 | 2026-08-26 Phase 1，real vs self 1/5 |

因此本 repo 不再測試：Dow 30 universe、GraphRAG 開放域關係抽取、
現有 KG relation 的 RL 超參數 sweep。

## 7. 進入 RL 的條件

預設不執行 RL。只有當 real 同時勝過 self、relation-shuffle 與 topology-shuffle，
且在 supervised 與簡單經濟 backtest 都達到預先設定門檻，才另立 slow-policy RL spec。
不把 supervised 結果直接接進既有的 daily RL。
