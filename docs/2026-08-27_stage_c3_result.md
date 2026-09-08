# Stage C3：top-500 穩健性確認——未通過

日期：2026-08-27
Branch：`feat/stage-c-graph-gate`
規模：520 個 run（top-500 × 10 seeds × 13 folds × 4 variants），0 失敗

## Research question

top-500 上 G_corr 相對 `topology_shuffle` 的微弱優勢，是否跨 seed、跨 fold、
以及在逐一排除任一 fold 後都穩健？

## Frozen design

判準在 `docs/2026-08-27_stage_c3_frozen_design.md` 凍結，
於看到 2016 稽核結果**之前**寫定。三項須同時成立：

| 判準 | 門檻 |
|---|---|
| seed-level 勝出 | ≥ 9/10 |
| fold-level 勝出 | ≥ 9/13 |
| leave-one-fold-out | 13 次都維持 seed ≥ 9/10 |

主要 control 固定 `topology_shuffle`。

## Result

| 判準 | 結果 | 門檻 | 通過 |
|---|---:|---:|---|
| seed-level 勝出 | 7/10 | ≥ 9/10 | 否 |
| fold-level 勝出 | 8/13 | ≥ 9/13 | 否 |
| leave-one-fold-out | 2/13 維持 | 13/13 | 否 |

平均效果 +0.00274，95% CI **[−0.00024, +0.00557]**，Wilcoxon p = 0.1055。
信賴區間包含 0。

**未通過，三項全部不成立。**

次要 control `relation_shuffle`：seed 8/10、fold 9/13、平均效果 +0.00143。
依凍結規則不參與判定，也不得在主要 control 失敗後改用它作為證據。

## 排除 2016 之後效果轉為負值

leave-one-fold-out 只有排除 2014 與 2015 時仍維持 9/10——那兩年本身是負貢獻，
拿掉它們當然會讓平均變好。排除任何其他年份都不足 9/10：

| 排除的年份 | seed 勝出 | 平均效果 |
|---|---:|---:|
| 2016 | **4/10** | **−0.00080** |
| 2014 | 9/10 | +0.00444 |
| 2015 | 9/10 | +0.00496 |
| 其餘 10 年 | 7–8/10 | +0.0018 – +0.0038 |

排除 2016 之後平均效果**轉為負值**。C2 觀察到的「85% 來自 2016」在
10 seeds 下不但成立，而且更強：那一年不只主導效果，是唯一支撐它的年份。

逐 fold 效果：

```text
2013 +0.00093   2016 +0.04526   2019 -0.00134   2022 ...
2014 -0.01766   2017 +0.00075   2020 -0.00935
2015 -0.02385   2018 +0.01343   2021 -0.00113
```

## C2 的 5/5 沒有在 10 seeds 下重現

C2 用 seeds 41–45 得到 5/5；C3 用 41–50 得到 7/10。
換言之 seeds 46–50 中有三個是負的。這本身就說明 5 seeds 的探索門檻
對這種量級的效果沒有判別力。

## 量測本身有非決定性

C3 重跑了 seeds 41–45，理應與 C2 完全一致。實際不然：

| variant | 同 seed 重跑的 \|Rank IC 差異\| 中位數 | 最大 |
|---|---:|---:|
| `self` | **0.00000** | 0.00000 |
| `real` | 0.00094 | 0.01677 |
| `relation_shuffle` | 0.00107 | 0.04443 |
| `topology_shuffle` | 0.00073 | 0.02250 |

`self` 完全決定性，其他三個不是。差別在於 `self` 的每個目標節點只有一條入邊，
其餘 variant 的訊息聚合會有多條邊指向同一節點：

```text
model.py 的 messages.index_add_(0, dst, transformed)
        │
        ├─ 索引不重複（self）  → 無碰撞，結果決定性
        └─ 索引重複（其他）    → CUDA atomicAdd，累加順序不保證
                                 → 浮點誤差 → 經 3 epochs 訓練放大
```

配對差 `real − topology_shuffle` 的合成雜訊每 (seed, fold) 約 0.0027，
**與量到的效果同量級**。跨 13 折平均後降到約 0.0007，所以 seed-level 的
+0.00274 仍是雜訊的 3.7 倍，但這足以解釋勝出計數為何不穩定。

這個發現強化而非削弱負結論：在固定 seed 下都無法重現的情況下，
0.003 量級的效果不具意義。

## What this proves

1. top-500 的 G_corr 優勢**不穩健**。三項凍結判準全部未通過，
   95% CI 包含 0。
2. 該優勢幾乎完全依賴 2016 一年，排除後轉為負值。
3. 目前的訊息聚合實作在 CUDA 上非決定性，雜訊量級與待測效果相當。

## What it does not prove

1. 動態相關圖在任何情況下都無用。2016 的 +0.045 經逐日稽核確認不是雜訊
   （見 `artifacts/stage_c2_audit_2016/daily_report.md`），
   它是一個真實但**單一年份**的現象。
2. 語義 KG 無效。G_corr 是價格統計圖，與抽取式知識圖譜無關。
3. 風險/共動任務無效。本輪只測報酬排序。

## Decision

`stop`，依凍結的決策流程：

> topology shuffle 勝出不足 → 不能宣稱模型使用真實共動結構，
> 應停止把這個訊號接入 RL。

不進 RL。不擴充 seeds。不改用 `self` 或 `relation_shuffle` 作為主要證據。

## H1 的最終狀態

H1（稀疏／橫斷面不足是前身 repo 負結果的主因）**不成立**。

C2 顯示效果不隨 universe 單調增強（−0.01152、−0.02176、+0.00322），
C3 顯示唯一通過探索門檻的 top-500 也不穩健。

前身 repo 在 Dow 30 上的負結果，不能歸因於 universe 太小。

## Next gate

轉 H3：以相同的凍結資料與 walk-forward 切分，改測風險/共動目標
（未來實現共變異數、pairwise absolute spread、波動度），
判準沿用 `RESEARCH_PROTOCOL.md` §2 的 controls 與 §4 的 seed 門檻。

在那之前必須先修掉非決定性，否則 H3 會重複同一個問題：

- `torch.use_deterministic_algorithms(True)` 加上 `CUBLAS_WORKSPACE_CONFIG`，或
- 把 `index_add_` 換成預先建好的稀疏鄰接矩陣乘法（`torch.sparse.mm`），
  它是決定性的，而且在這個密度下可能更快。

修完後應重跑一次 C3 的一小部分確認同 seed 可完全重現，再開始 H3。

## Artifacts

```
scripts/26_summarise_c3.py
artifacts/stage_c3_gate/          130 個工作單位的原始結果
artifacts/stage_c3_summary/       判定、leave-one-fold-out、重現性檢查
artifacts/stage_c3_sweep.log
```
