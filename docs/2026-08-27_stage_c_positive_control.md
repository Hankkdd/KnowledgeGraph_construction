# Stage C 正控制：模型學得會植入的圖訊號嗎

日期：2026-08-27
Branch：`feat/stage-c-graph-gate`
性質：正控制，不是 gate——target 是合成的，結果與 KG 有沒有資訊無關

## Research question

在 C2 之前，這個模型與訓練設定能不能學會一個「確定存在、且必須靠圖才能取得」的訊號？

## Why this experiment is next

C1 smoke 以 loss 下降作為模型可收斂的證據。把最終 loss 與 target 變異數對照後，
那個推論站不住：

```text
訓練期 target 變異數（預測橫斷面均值的 MSE）   0.006849
C1 五個 variant 的最終 train loss              0.006886 – 0.006900
                                               比預測 0 還差 0.64%
```

模型收斂到的是常數預測器。五個 variant 的最終 loss 幾乎相同，正是因為它們都退化成
同一個解，而不是因為圖沒有差別。

這件事必須先釐清，否則 C2 的 1,365 個 run 全部是在量抽樣雜訊，而 real 與 control 的
任何差異都只是隨機——那正是前身 repo 的失敗模式，也是它當初設 Gate 0 的原因。

## Frozen design

| 項目 | 值 |
|---|---|
| Universe | top-500 |
| Fold | fold12，train 2022-01-03..2024-12-02，test 2025-01-02..2025-12-31 |
| 特徵 | 真實資料，60 日視窗的 4 維摘要 |
| 圖 | 真實 G_corr，k=5 |
| Target | 合成：鄰居特徵平均，橫斷面標準化後加雜訊 |
| 訊噪比 | 2:1（`noise=0.5`） |
| Variants | no_graph、self、real、relation_shuffle、topology_shuffle |
| Epochs / seed | 3 / 41 |

target 一律由**真實**圖產生；variant 只改模型看得到的圖。因此 `topology_shuffle`
看到的是錯的鄰居，理論上無法還原訊號。

## 判準

三項缺一不可：

1. `real` 的 test Rank IC > 0.5，且在訓練期與測試期都優於各自的常數預測器
   → 模型學得會
2. `real` − `topology_shuffle` > 0.2 → 學到的是圖結構，不是節點自身特徵
3. `real` 與 `relation_shuffle` 的差距小於它與 `topology_shuffle` 的差距
   → 量到的是拓樸訊號，不是 relation label 或 edge weight 的意外效果

第 3 項是因為合成訊號只依賴鄰居拓樸，完全不依賴 relation label。
`relation_shuffle` 因此應該接近 `real`；若它也大幅下降，代表這個測量抓到的
不是我們以為的東西。

## Result

train baseline 1.2482，test baseline 1.2550——兩者分開計算，
拿測試期的變異數去比訓練 loss 是不同資料區間，比較沒有意義。

| variant | train loss | test MSE | test Rank IC |
|---|---|---|---:|
| no_graph | 0.959 | 1.046 | +0.5148 |
| self | 0.959 | 1.039 | +0.5149 |
| real | **0.641** | 1.136 | **+0.8033** |
| relation_shuffle | 0.642 | 1.167 | **+0.8019** |
| topology_shuffle | 0.931 | 1.005 | +0.5254 |

五個 variant 在訓練期與測試期都勝過各自的常數預測器。

```text
real            +0.8033
relation_shuffle +0.8019   gap 0.0013  ← 打亂標籤幾乎沒有影響
topology_shuffle +0.5254   gap 0.2779  ← 打亂鄰居就掉回基準線
```

**PASS，三項判準全數成立。**

`no_graph` 拿到 0.51 是預期的：訊號是鄰居特徵的平均，與節點自身特徵有相關，
所以不看圖也能取得一部分。關鍵在於只有看得到正確鄰居的兩個 variant 能走到 0.80。

`relation_shuffle` 與 `real` 幾乎相同，正是合成訊號的設計所預測的結果。
這排除了「多出來的 0.28 來自 relation 或 weight 的意外效果」這個解釋。

## 一個附帶結論：3 epochs 的訓練預算夠用

正控制在同樣 3 epochs 下就把 `real` 學到 0.80。因此 C2 沿用 3 epochs 有證據支持，
不需要再調——調整訓練預算會多出一個研究者自由度。

## 訓練目標維持 MSE

主判定指標是 Rank IC，訓練卻用 MSE，因此另外比較了直接優化橫斷面相關的 IC loss。
同一折、同一容量、真實 target：

| objective | variant | test Rank IC | SE | t |
|---|---|---:|---:|---:|
| mse | no_graph | +0.0183 | 0.0103 | +1.78 |
| mse | real | +0.0002 | 0.0079 | +0.03 |
| mse | topology_shuffle | −0.0136 | 0.0087 | −1.55 |
| ic | no_graph | +0.0166 | 0.0119 | +1.40 |
| ic | real | −0.0313 | 0.0117 | −2.67 |
| ic | topology_shuffle | −0.0179 | 0.0097 | −1.84 |

IC loss 沒有改善，`real` 反而更差。維持 MSE，不改凍結設計。

## What this proves

1. 模型與訓練設定學得會圖訊號，C2 的負結果將可歸因於資料而非實作。
2. 模型會使用圖結構本身，不只是節點自身特徵。
3. 3 epochs 足夠，MSE 目標不需替換。

## What it does not prove

1. 真實 20 日報酬存在任何可學的訊號。合成 target 的訊噪比 2:1，真實報酬遠低於此。
2. real KG 優於任何 control。
3. 正控制通過不代表 C2 有足夠統計力；單折的 Rank IC SE 約 0.010，
   判別力來自 13 折 × 5 seeds 的聚合與配對。

## Decision

`implementation`。C2 的前提成立，依原設計執行。

## Next gate

C2：G_corr × 三個 universe × 五個 variant × 13 折 × 5 seeds，
主判定為 real-vs-control 的 paired Rank IC 效果是否隨 universe 規模增強。

## Artifacts

```
scripts/19_positive_control.py
artifacts/stage_c_positive_control/top500_seed41.json
```
