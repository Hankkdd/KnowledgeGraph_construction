# Stage C2：G_corr supervised gate（5 seeds，三個 universe 規模）

日期：2026-08-27
Branch：`feat/stage-c-graph-gate`
Commit：`bfabc3e`
規模：975 個 run，208 分鐘，0 失敗

## Research question

在相同價格資料、相同模型容量與時間正確的條件下，增加 universe 規模是否能讓
真實的動態相關圖在 out-of-sample 橫斷面報酬排序上穩定勝過 no-graph、self、
relation-shuffle 與 topology-shuffle？

## Why this experiment is next

前身 repo 的所有 gate 都在 30–44 檔上失敗，共用一個未檢驗的前提：universe 太小。
Stage B 建好三個規模點、Stage C0 確認 G_corr 的 in-degree 不隨規模變動、
正控制確認模型學得會圖訊號。此處第一次公平地測試規模這個維度。

## Frozen design

| 項目 | 值 |
|---|---|
| Universe | top-30 / top-100 / top-500 |
| Graph | G_corr，60 日報酬相關，每節點 top-k=5，typed（正負號） |
| Target | 未來 20 交易日累積報酬，橫斷面去均值 |
| 切分 | walk-forward 3 年 train / 1 年 test，13 折（2013–2025），purge 20 日 |
| Seeds | 41–45 |
| Epochs | 3（由正控制確認足夠） |
| Primary metric | 每日橫斷面 Rank IC |
| 統計單位 | seed（每個 seed 先跨 13 折平均） |
| 門檻 | 5/5 seeds，且 real 須同時勝過 self、relation_shuffle、topology_shuffle |

## Result

paired 效果（real 減 control），統計單位為 seed：

| universe | control | mean effect | seed wins | Wilcoxon p |
|---|---|---:|---:|---:|
| 30 | self | −0.00615 | 1/5 | 0.1875 |
| 30 | relation_shuffle | +0.01029 | 5/5 | 0.0625 |
| 30 | topology_shuffle | −0.01152 | 0/5 | 0.0625 |
| 100 | self | −0.02176 | 0/5 | 0.0625 |
| 100 | relation_shuffle | −0.00016 | 2/5 | 0.8125 |
| 100 | topology_shuffle | −0.01071 | 0/5 | 0.0625 |
| 500 | self | **+0.00989** | **5/5** | 0.0625 |
| 500 | relation_shuffle | **+0.00322** | **5/5** | 0.0625 |
| 500 | topology_shuffle | **+0.00469** | **5/5** | 0.0625 |

只有 top-500 同時勝過三個 control。`p = 0.0625` 是 n=5 時 5/5 全勝所能達到的最小值，
不是效果強度的指標。

## 這個結果對 H1 判定不成立

凍結的判準要求效果隨 universe 單調增強。各 universe 的最小效果為
top-30 −0.01152、top-100 −0.02176、top-500 +0.00322——**不是單調的**，
top-100 比 top-30 更差。

因此依預先凍結的規則，判定是「混合結果」，不是 H1 成立。
兩個較小的 universe 都是負值，它們之間的排序意義有限，但**不得因此事後放寬判準**——
那正是這套協定要防止的研究者自由度。

## 效果的 85% 來自 2016 年

逐折檢視 top-500 的 `real − topology_shuffle`：

```text
2013 +0.0001   2016 +0.0523   2019 +0.0017   2022 +0.0011   2025 +0.0178
2014 -0.0196   2017 +0.0091   2020 -0.0077   2023 +0.0012
2015 -0.0179   2018 +0.0141   2021 -0.0006   2024 +0.0095
```

13 折中 9 折為正，但 2016 一年就貢獻了 85% 的平均效果。排除 2016 之後：

| control | 全 13 折 | 排除 2016 | 折中位數 |
|---|---|---|---:|
| self | 5/5, +0.00989 | 5/5, +0.00734 | +0.01501 |
| relation_shuffle | 5/5, +0.00322 | 5/5, +0.00347 | +0.00190 |
| topology_shuffle | 5/5, +0.00469 | **3/5, +0.00072** | +0.00116 |

對 `self` 與 `relation_shuffle` 的優勢穩健，對 `topology_shuffle` 的優勢不穩健。
`topology_shuffle` 是三個 control 中最關鍵的一個——它保留邊數與兩個度數序列，
只打散連到誰。贏不過它，就無法宣稱模型用到的是真實的共動結構。

## 模型幾乎不擊敗常數預測器

| 診斷 | 值 |
|---|---:|
| 常數預測器的 run | 0 / 975 |
| 訓練期勝過 mean predictor | 69 / 975 |
| 測試期勝過 mean predictor | 20 / 975 |
| 預測標準差 / target 標準差（中位數） | 0.115 |
| degenerate days | 0 |

沒有任何 run 退化成常數預測器，但預測的橫斷面離散度只有 target 的 11.5%，
且幾乎不曾在 MSE 上勝過直接預測均值。Rank IC 在這種收縮預測下仍然可解讀
（排序資訊可以保留），但效果量級 0.003–0.010 必須放在這個背景下理解。

## What this proves

1. 只有 top-500 通過預先設定的 5/5 探索門檻，兩個較小的 universe 沒有。
2. real 相對 `relation_shuffle` 的優勢在 top-500 是穩健的（排除 2016 後仍 5/5）。
3. pipeline 沒有退化：零常數預測器、零 degenerate day、mask 與 control 定義一致。

## What it does not prove

1. **H1 不成立。** 效果沒有隨 universe 單調增強。
2. top-500 對 `topology_shuffle` 的優勢不穩健，85% 來自 2016 一年。
3. 沒有任何交易績效主張。Rank IC 差距 0.003–0.010，且模型幾乎不擊敗常數預測器。
4. 與語義 KG 無關。G_corr 是價格統計圖，不是抽取式知識圖譜。

## Decision

`pivot`。

依凍結規則，top-500 的 5/5 達到進入 10-seed confirmation 的探索門檻。
但增加 seeds 只降低 seed 變異，**無法處理這裡真正的脆弱性——它在 fold 之間**。
單純執行原定的 C3 會得到更精確的、對 2016 依賴的估計。

因此 C3 的設計需要修改後才執行，見下。

## Next gate

C3 修改為：

1. seeds 擴到 41–50，門檻 9/10。
2. **同時報告 fold-level 勝出數**，不只 seed-level。目前的脆弱性在折之間，
   只看 seed 會看不見。
3. 預先指定 leave-one-fold-out 敏感度：逐一排除每一折後重算，
   若移除任一折就使結論翻轉，則不得宣稱通過。
4. 判定仍以 `topology_shuffle` 為主要 control；只贏 `self` 或 `no_graph` 不算。

在 C3 執行前，這些修改必須先凍結並記錄，不得在看到結果後調整。

## Artifacts

```
scripts/22_run_gate_sweep.py
scripts/23_summarise_gate.py
artifacts/stage_c2_gate/            195 個工作單位的原始結果
artifacts/stage_c2_summary/         paired 比較與判定
artifacts/stage_c2_sweep.log        排程紀錄
```
