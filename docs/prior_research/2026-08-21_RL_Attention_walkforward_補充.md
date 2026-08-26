# RL Attention 收斂與 Walk-Forward：簡報補充資料

這份補充明確把 `RL_Attention收斂與Walk-Forward驗證紀錄.md` 納入總研究時間線。

## 必須呈現的結果

### Policy 收斂診斷

- 初始 policy 幾乎不交易：5 檔權重約 19.8%–20.4%。
- 補上 episode-level reward logging 後，30 episodes 的 total reward 由 0.446 升到 0.472，與 episode 數相關係數 0.987。
- stock entropy 由 1.609428 降至 1.606057；turnover 由 0.0006 升至 0.0070。
- 舊 parameter sweep 同時改動 lr、head coefficient、cash/unfrozen 等參數，不能用來宣稱「episodes 越多越差」。

### Clean Walk-Forward

- train：2024-01-01–2025-01-01。
- validation：2025-01-01–2025-07-01，用於選 episodes。
- final test：2025-07-01–2026-01-01，只評估一次。
- validation episodes sweep：1/5/10/20/50/100 的 Sharpe 為 0.8088/0.8092/0.8098/0.8111/0.8020/0.7956；選 episodes=20。
- final test：KG return 0.1185、Sharpe 1.4618；random return 0.1183、Sharpe 1.4565。
- paired Wilcoxon：return p=1.0、Sharpe p=0.1875；不顯著。

### Attention/RL 架構消融

| 測試 | 結果 |
|---|---|
| 解凍 attention encoder | KG≈random，Sharpe p=0.8125 |
| KG similarity reward shaping | KG 1.4586 vs random 1.4605，p=0.8125 |
| attention score temperature | 15 seeds KG 1.4698 vs random 1.4753，p=0.5245 |
| dynamic query | 15 seeds KG 1.4619 vs random 1.4683，p=0.5245 |
| 換 portfolio/期間 | validation 只有 2/5 seeds 偏向 KG |
| action_blend=1.0 | 只放大噪聲；產業輪動 Sharpe 1.015→0.626，空頭防禦 -0.722→-0.960 |
| relative reward + unfreeze | Sharpe 約 1.015→1.016，沒有解鎖換倉 |

## 這組實驗對總結論的作用

這份紀錄不是單純「RL 沒有效果」；它完成了三件重要工作：

1. 證明 policy 確實會慢慢學習，排除「完全沒有梯度/沒有收斂」的誤判。
2. 用 clean walk-forward 修正舊 parameter sweep 的方法論混淆。
3. 連續排除 freeze、reward、temperature、dynamic query、action blend 等局部解釋，將問題定位到 KG similarity 與報酬共動薄弱，以及缺少可用跨日決策訊號。

因此它應放在簡報前半部，作為「RL/Attention 診斷與負結果收斂」的獨立段落，而不是只在 baseline slide 用一句話帶過。
