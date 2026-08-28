# Stage C1：固定容量 GNN 收斂與 runtime smoke

日期：2026-08-27  
Branch：`feat/stage-c-graph-gate`  
性質：implementation smoke，不是 KG efficacy gate

## Research question

確認 Stage C 的 supervised graph-to-target pipeline 是否能在 top-500、完整 walk-forward fold
與所有 G_corr controls 上正常訓練、反向傳播、計算 Rank IC，並量出單次 run 的 runtime。

這一步不回答 KG 是否有效，也不執行 RL episode。若模型本身無法收斂，後續 C2 的負結果
就無法歸因；若能收斂，才可以進入多 seed 的正式比較。

## Frozen design

- Universe：CRSP point-in-time top-500。
- Fold：test year 2025；train 2022-01-03..2024-12-02，test 2025-01-02..2025-12-31。
- Purge：20 個交易日。
- Feature：每檔股票最近 60 個有效 daily return 的四個摘要（last、mean、std、cumulative）。
- Target：`Panel.target_at()` 的未來 20 個交易日橫截面去均值累積報酬。
- Model：固定兩層 typed GNN、hidden dimension 64、兩個 relation channels 與 per-node linear head。
- 公平性：`no_graph` 使用同一模型與參數，只傳空邊集；`self`、real、relation shuffle、
  topology shuffle 不改模型容量。
- Seed：41；epoch=1 的 runtime smoke 及 epoch=3 的收斂 smoke。

Variant：`no_graph`、`self`、`real`、`relation_shuffle`、`topology_shuffle`。

## 邊界問題與修正

第一次執行把測試期最後約 20 個交易日納入評估，但這些日期沒有完整的未來 h=20 target，
因此 usable node 少於 2。runner 已修正為先依 asset-level target mask 篩掉無法計算 target
的日期；修正後測試期保留 230 個日期。這是資料切分問題，不是模型錯誤。

## 結果：epoch=1 runtime smoke

| variant | train loss | test Rank IC | test MSE | runtime |
|---|---:|---:|---:|---:|
| no_graph | 0.008724 | -0.001619 | 0.008759 | 2.60 s |
| self | 0.008335 | 0.016088 | 0.008824 | 2.49 s |
| real | 0.009025 | 0.022959 | 0.008849 | 10.10 s |
| relation_shuffle | 0.009038 | 0.019882 | 0.008941 | 10.03 s |
| topology_shuffle | 0.009227 | 0.015542 | 0.008616 | 11.88 s |
| **total** | — | — | — | **37.42 s** |

資料量：733 個 train dates、230 個 test dates、每期最多 500 個節點。上述 Rank IC 是單一 seed、
單一 epoch 的 smoke 數字，不具 gate 證據力。

## 結果：epoch=3 收斂 smoke

所有 variant 的 train loss 都下降：

| variant | loss first | loss last | test Rank IC | runtime |
|---|---:|---:|---:|---:|
| no_graph | 0.008724 | 0.006894 | 0.018272 | 4.77 s |
| self | 0.008335 | 0.006886 | 0.017539 | 5.75 s |
| real | 0.009023 | 0.006900 | -0.000175 | 28.29 s |
| relation_shuffle | 0.009039 | 0.006894 | 0.008239 | 28.38 s |
| topology_shuffle | 0.009227 | 0.006900 | -0.014483 | 33.22 s |
| **total** | — | — | — | **約 100 s** |

## Interpretation

這個結果證明：

1. 空圖、self-loop 與三種 G_corr graph 都能通過同一個 model interface。
2. 所有條件都能完成 forward、loss、backward 與 test Rank IC 計算。
3. graph variant 的額外成本約為 no-graph 的 4 倍；單一完整 1-epoch 五條件約 37 秒，
   可用來規劃 C2 平行執行。

這個結果不證明：

1. real graph 優於任何 control；單 seed、少量 epoch 的 IC 不可作為結論。
2. top-500 universe 存在可交易 alpha。
3. graph signal 能傳到 portfolio action 或改善 Sharpe、MDD。
4. sector graph 或語義供應鏈 KG 有效；本 C1 只跑 G_corr family。
5. **模型具備學習能力。** 見下節。

## 更正：loss 下降不等於學到東西

初稿把「3 個 epoch 中所有條件的 training loss 都下降」列為模型具備收斂能力的證據。
把最終 loss 與 target 變異數對照之後，這個推論不成立：

| | 值 |
|---|---:|
| 訓練期 target 變異數（= 直接預測橫斷面均值的 MSE） | 0.006849 |
| 五個 variant 的最終 train loss | 0.006886 – 0.006900 |

最終 loss 比預測 0 還高 0.64%。模型收斂到的是「預測橫斷面均值」這個平凡解——
五個 variant 得到幾乎相同的最終 loss，正是因為它們都退化成同一個常數預測器。
loss 從隨機初始化下降，只說明它離開了更差的起點。

模型是否真的學得動，改由 `docs/2026-08-27_stage_c_positive_control.md` 的
正控制回答；那份的結論是 PASS，因此 C2 的前提成立。

## Decision

`implementation`：C1 smoke 通過，進入 C2 exploratory supervised gate。

## C2 前提

- 使用相同的 target mask、purge、模型容量與 graph control 定義。
- H1 主判定只使用 G_corr；G_sector 保留為描述性／探索性圖，不納入 universe-size 的主要結論。
- 先跑 top-30、top-100、top-500 的 5 seeds（41–45），再依 5/5 結果決定是否進 10-seed confirmation。
- 每個正式 gate 必須報 daily Rank IC、paired mean difference、seed win count、effect size 與 Wilcoxon p-value。
- C2 仍然不接 RL；只有 supervised gate 與簡單 economic backtest 都通過，才另立 slow-policy RL spec。

## Reproducible artifacts

- Model：`src/kgc/model.py`
- Training/evaluation：`src/kgc/gate.py`
- Runner：`scripts/21_run_gate.py`
- 1 epoch：`artifacts/stage_c1_smoke/top500_seed41.{json,md}`
- 3 epoch：`artifacts/stage_c1_smoke_3epoch/top500_seed41.{json,md}`
- C0：`artifacts/stage_c_graphs/report.md`
