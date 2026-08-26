# KG/RL 實驗優先順序與定期反思流程

本文件是專案的實驗管理規則，搭配 user-local Codex skill `kg-research-priority-review` 使用。目的不是增加文書工作，而是避免在結果不穩定時繼續盲跑 RL，並讓每一個實驗都能回答主研究問題。

## 核心研究問題

> 在固定資料切分與公平控制組下，時間正確、語義正確的金融 KG 是否提供 no-graph 與 corrupted-graph 以外的增量投資價值？

把問題分成四層：

1. KG factual/provenance/temporal quality
2. Graph utilization 與 semantic attribution
3. Out-of-sample supervised signal
4. Portfolio/RL economic value

後一層不能替前一層背書。sector/link prediction 通過，不等於 trading 通過；平均 Sharpe 上升，也不等於 KG semantic value 已證明。

## 什麼時候必須先反思

以下請求一律先做 priority review，再決定是否執行：

- 「繼續」「下一步」「再跑一次」
- 「這結果代表什麼」
- 「要不要改 RL/KG」
- 新增 horizon、relation、portfolio 或 reward
- 任何可能消耗長時間 GPU/CPU 的實驗

## Priority review checklist

### 1. 先讀最新證據

讀取：

- 最新 `report.md`、`summary.json`、`results.csv`
- branch 與 commit
- [整合研究紀錄](2026-08-06_GPM_KG時間尺度與慢速交易驗證紀錄.md)

不要只看平均值；必須看 seed-level paired differences。

### 2. 明確寫出未解問題

例如：

- 關係語義是否比任意圖拓撲好？
- h60 MSE 改善是否能轉成 Sharpe？
- 固定五檔是否覆蓋不足？
- 圖訊號是否只改善 calibration，而非方向？
- RL action transfer 是否是瓶頸？

### 3. 將工作分類

| 類型 | 意義 |
|---|---|
| `confirm` | 對 promising result 做窄範圍 replication |
| `pivot` | 換 target、universe、horizon 或 control 以解決已診斷問題 |
| `implementation` | 實作已批准的測試所需程式 |
| `stop` | 重複失敗或統計力不足以支持新主張，停止同類實驗 |

### 4. 執行前檢查

每次執行前記錄：

```text
Research question:
Why now:
Hypothesis:
Frozen data/KG split:
Variants and controls:
Primary metric:
Seed/power rule:
Stop/advance criterion:
Expected artifacts:
```

最低控制組：

- `self` / no-relation
- `relation_permuted`
- `topology_permuted` 或 degree-preserving topology control

5 seeds 只能作探索；若採用 90% seed gate：5 seeds 必須 5/5，10 seeds 至少 9/10。報告 effect size、勝率與 Wilcoxon，不只報 p-value。

### 5. 優先選最便宜且最能區分假設的實驗

優先順序：

1. supervised graph-to-target
2. graph structure/semantics controls
3. 大 universe 或 pair/residual target
4. 簡單慢速 backtest
5. 最後才是完整 RL

在 supervised signal 沒有勝過 controls 前，不跑長時間 RL hyperparameter sweep。

## 解讀規則

- real > self，但不 > shuffled：可能是 topology/capacity regularization，不是 semantic KG 證明。
- real 只改善 MSE：代表 calibration/magnitude，不能稱為 directional alpha。
- real 改善 IC，但沒有 portfolio 改善：signal-to-action transfer 尚未成立。
- 只有平均值改善、seed 勝率不穩定：標記為 exploratory。
- 多個 horizon 與 control 都失敗：pivot 或 stop，不再盲調 RL。
- static 年度關係預測隔日報酬：先檢查時間尺度，不直接下結論說 KG 無效。

## 完成後的強制記錄

每個 gate 完成後，必須在同一個 change 中更新：

1. 為什麼跑
2. 實驗設計與控制組
3. command、seed、資料/圖 snapshot
4. 結果與 paired effect
5. 證明了什麼
6. 沒有證明什麼
7. 決策：confirm / pivot / implementation / stop
8. 下一個 gate
9. artifact report 與本整合文件連結

## 目前專案停止條件

只有在以下條件同時滿足後，才進入兩時間尺度 RL：

- real 勝 self
- real 勝 relation-permuted
- real 勝 topology-permuted
- 至少 90% seeds 方向一致
- supervised 結果能在簡單慢速 backtest 中轉成一致的 Sharpe 或 max drawdown 改善

若最後只剩 h60 MSE 改善，研究定位應是「KG 對長期報酬幅度/風險估計的 partial evidence」，而不是「KG 已提升 RL 交易績效」。

## 目前狀態（2026-08-06）

- Gate 0 oracle：通過
- 真實 KG 圖結構：有 sector/link/provenance 證據
- 隔日/週期 downstream：未通過
- h60 MSE：real 對 relation-permuted 5/5，但只有 5 seeds，待確認
- 慢速交易：平均 Sharpe 正向，但 paired 不顯著
- 下一步：補跑 h60 relation-permuted seeds 46–50，再決定 pivot 到 30-stock residual/volatility task 或停止 KG-RL 績效路線
