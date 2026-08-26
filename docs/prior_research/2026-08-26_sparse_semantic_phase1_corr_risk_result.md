# 稀疏與無語義 Alpha 判別 Gate：動態相關圖風險結果

日期：2026-08-26  
階段：Phase 1 H3 supervised risk gate（不執行 RL episode）  
前置：報酬目標 gate 未通過；本實驗檢查同一張 dynamic correlation graph 是否較適合描述風險／波動，而非未來報酬。

## 1. 為什麼改測風險

前一個 h=20 報酬測試中，dynamic correlation real 沒有穩定優於 self-loop 與 topology permutation。這表示它不能被當成已驗證的報酬 alpha。

但共動關係的直觀用途不一定是預測哪檔股票會上漲；它可能更適合描述「哪些股票會一起波動」或「哪檔股票未來風險較高」。因此本實驗固定資料、模型、日期與圖建構方式，只把 supervised target 改為每檔股票未來 20 個交易日的 realized volatility，測試 H3：

> 圖對風險任務有用，但對報酬排序任務不一定有用。

本階段仍然不接 RL。這能把「圖是否含有風險訊息」與「RL 是否學會交易」分開。

## 2. 實驗設計

- Universe：44 檔可取得完整價格的 S&P 100 current-constituent proxy；不是 point-in-time 成分股，存在 survivorship/selection bias。
- 訓練期：2024-01-01 至 2025-01-01。
- 測試期：2025-01-01 至 2026-01-01。
- Graph：每個日期 `t` 只使用 `t` 以前最近 60 個交易日的 close 報酬，top-k=5；不使用未來價格建圖。
- Target：從 `t+1` 開始的未來 20 個交易日，每檔股票的 log-return 標準差。
- 模型：同一個 GPM/R-GCN encoder 與 return head，固定容量；只訓練 volatility prediction head，未使用 allocation-head loss。
- Seeds：41–45，共 5 個。
- 主要指標：橫截面 risk Rank IC（預測波動排序與實際未來波動排序的 Spearman correlation）；輔助指標為 MSE。

控制組定義與報酬 gate 相同：

| 條件 | 意義 |
|---|---|
| real | 真實 rolling correlation graph |
| self | 每檔股票只有 self-loop |
| relation_permuted | 保留拓樸但反轉正／負相關型別 |
| topology_permuted | 保留邊數／大致度數但重連節點 |

## 3. 結果

### 3.1 五個 seed 平均值

| 條件 | Risk Rank IC | Risk MSE |
|---|---:|---:|
| real | 0.05467 | 0.00011434 |
| self | 0.05667 | 0.00011369 |
| relation_permuted | 0.04791 | 0.00013206 |
| topology_permuted | 0.02345 | 0.00013110 |

### 3.2 配對比較

| 比較 | real 勝出 seed | 平均 IC 優勢 | 平均 MSE 優勢 | Wilcoxon p（IC） |
|---|---:|---:|---:|---:|
| real vs self | 1/5 | -0.00199 | -0.00000066 | 0.6250 |
| real vs relation_permuted | 4/5 | +0.00677 | +0.00001771 | 0.6250 |
| real vs topology_permuted | 5/5 | +0.03123 | +0.00001675 | 0.0625 |

MSE 優勢定義為 control MSE − real MSE，正值表示 real 較低。

## 4. 這些結果代表什麼

### 4.1 對 H3 的支持

real 對 topology permutation 的 risk Rank IC 是 5/5 seed 勝出，平均高出約 0.031。這表示在打亂節點連接、但保留大致圖密度後，風險排序明顯下降；至少有一個跡象顯示真實 correlation topology 對風險任務可能有幫助。

real 對 relation permutation 也有 4/5 seed 勝出，MSE 也較低，方向與 H3 一致。

因此，與報酬 gate 的「幾乎沒有訊號」相比，風險任務出現了較有一致性的圖結構差異。這支持後續研究把 KG／graph 當成風險暴露或共動輸入，而不是直接當成報酬預測器。

### 4.2 為什麼還不能宣布 H3 通過

real 對 self-loop 只有 1/5 seed 勝出，而且 self 的平均 IC 略高於 real。self-loop 沒有跨公司邊，若 real 真正依賴跨公司關係，理應也要穩定擊敗 self。

此外，topology 比較雖然 5/5 勝出，但 n=5 的 Wilcoxon p=0.0625，略高於預先設定的 0.05。這是「值得確認」而不是「已完成統計確認」的證據。

所以目前最準確的結論是：

> H3 有初步支持，尤其是相對 topology permutation；但尚未證明跨公司圖語義在風險預測上穩定優於沒有跨公司關係的 baseline。

### 4.3 方向準確率不能用來宣稱成功

波動度本身幾乎永遠是正值，因此只要模型輸出正值，directional accuracy 就會接近 100%。本實驗的方向準確率在各組幾乎都是 1.0，沒有辨識力；真正應看的指標是 risk Rank IC、MSE，以及後續的風險排序／投資組合暴露效果。

## 5. 對整個研究主軸的意義

目前證據鏈可以清楚分成兩部分：

1. **報酬 alpha：未通過。** 擴大 universe 並使用 point-in-time dynamic correlation graph，仍沒有穩定的未來報酬排序優勢。
2. **風險訊息：有希望但未定案。** real graph 相對 topology permutation 的風險 IC 5/5 勝出，但 self-loop 結果不支持完全歸因於跨公司關係。

這比直接把失敗歸因於 RL 更清楚：目前較可能的問題不是 RL reward 寫錯，而是圖的資訊更接近「共同風險結構」，不一定是「可交易方向 alpha」。

## 6. 下一步決策

優先做一次較小但正式的 confirmation，而不是直接接回 RL：

- 固定 h=20、window=60，只把 seeds 擴到 10；至少跑 real、self、topology 三組。
- 預先設定通過條件：real 對 topology 與 self 都至少 9/10 勝出，且配對檢定 p<0.05。
- 若通過，再把 risk graph 作為 RL 的風險輸入（例如 volatility-aware position cap、risk parity 或 drawdown penalty），而不是當作報酬預測訊號。
- 若不通過，H3 也不成立；應停止擴大此 correlation graph，回到資料／任務重新定義。

## 7. 可重現產物

- Runner：`0601_atten_V2/setup/57_run_dynamic_corr_risk_gate.py`
- Smoke：`0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/phase1_smoke_corr_risk_real_h20/`
- Formal real：`.../phase1_corr_risk_real_h20_5seed/`
- Formal self：`.../phase1_corr_risk_self_h20_5seed/`
- Formal relation permutation：`.../phase1_corr_risk_relation_h20_5seed/`
- Formal topology permutation：`.../phase1_corr_risk_topology_h20_5seed/`
