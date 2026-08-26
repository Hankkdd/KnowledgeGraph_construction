# 稀疏與無語義 Alpha 判別 Gate：動態相關圖結果

日期：2026-08-26  
階段：Phase 1 supervised gate（不執行 RL episode）  
前置：Phase 0 已確認 44 檔可用 proxy universe 的價格覆蓋完整；GICS 靜態圖在 h=20、h=60 均未通過報酬語義 gate。

## 1. 為什麼要做這個實驗

前面的負結果有兩種可能解釋：

1. 圖太稀疏、股票數太少，所以模型沒有足夠的跨公司訊息（H1）。
2. 即使圖變密集，這類關係本身也不包含可預測未來報酬的訊息（H2）。

本實驗選用 44 檔價格覆蓋完整的 S&P 100 current-constituent proxy，並以過去 60 個交易日的報酬建立 rolling correlation kNN 圖。它測試的是「較大的 universe + 每日更新的共動圖」是否能改善報酬預測，而不是把靜態圖硬套到每日交易。

這是一個 supervised gate：模型只學習預測未來 20 個交易日的累積報酬，不接 RL、不做 action_blend 掃描。因此若此處沒有訊號，不應把問題歸因於 RL reward 或 portfolio execution。

## 2. 實驗設計

- Universe：44 檔可取得完整歷史價格的 S&P 100 current-constituent proxy；不是 point-in-time 成分股，因此有 survivorship/selection bias，結果只能視為方法診斷，不是正式投資回測。
- 訓練期：2024-01-01 至 2025-01-01。
- 測試期：2025-01-01 至 2026-01-01。
- Graph：每個日期 `t` 都只使用 `t` 當日以前最近 60 個交易日的 close 報酬；預測目標從 `t+1` 開始，避免把未來資料放入圖中。
- 圖邊：每檔最多保留 top-k=5 的相關邊；邊型別表示相關係數正負，邊權重為相關係數絕對值。
- 模型：沿用已通過架構 smoke test 的 GPM/R-GCN supervised return head；固定模型容量。
- Seeds：41–45，共 5 個。
- 主指標：每日橫截面 Rank IC（預測排序與實際未來報酬排序的 Spearman correlation）。輔助指標：MSE、方向準確率。

每一個條件都使用相同的價格、模型容量、訓練日期與 seed。差異只在圖的內容：

| 條件 | 圖的意義 |
|---|---|
| real | 真實 rolling correlation graph |
| self | 每檔股票只有 self-loop；沒有跨公司邊 |
| relation_permuted | 保留真實拓樸，但反轉正/負關係型別 |
| topology_permuted | 保留邊數與大致度數，但以固定 permutation 重連節點 |

判定標準沿用本研究的配對 gate：real 必須在主要指標上相對 control 穩定勝出；探索門檻為 5/5 seed 勝出。只贏 no-graph 不算語義證明，因為模型容量或正則化差異也可能造成表面改善。

## 3. 五個 seed 的結果

### 3.1 平均值

| 條件 | Rank IC mean | MSE | Directional accuracy |
|---|---:|---:|---:|
| real | 0.00010 | 0.008842 | 0.5531 |
| self | 0.00932 | 0.008981 | 0.5534 |
| relation_permuted | -0.00980 | 0.008955 | 0.5530 |
| topology_permuted | 0.00842 | 0.008943 | 0.5517 |

Rank IC 的絕對值都很小。real 的平均 IC 幾乎為零，並沒有因為換成動態相關圖而產生清楚的報酬排序訊號。

### 3.2 配對比較（real − control）

| 比較 | IC 勝出 seed 數 | 平均 IC 差 | MSE 較佳 seed 數 | Wilcoxon p（IC） |
|---|---:|---:|---:|---:|
| real vs self | 1/5 | -0.00922 | 4/5 | 0.4375 |
| real vs relation_permuted | 3/5 | +0.00990 | 4/5 | 0.3125 |
| real vs topology_permuted | 1/5 | -0.00832 | 2/5 | 0.3125 |

MSE 越低越好；雖然 real 對 self 與 relation permutation 的 MSE 分別有 4/5 seed 較低，但差距很小，且未達統計顯著。更重要的是，MSE 的改善沒有轉成 IC 的穩定改善；因此不能把它解讀成可交易的 alpha。

## 4. 結果如何解讀

### 4.1 對 H1「只是太稀疏」的判定

本測試不支持 H1。理由是我們已經把 universe 從原本的 Dow 30 擴到 44 檔可用 proxy，並把靜態關係換成每日 point-in-time rolling correlation graph；但 real 沒有在 Rank IC 上同時穩定勝過 self 與 topology permutation：

- 對 self：只有 1/5 seed 勝出，平均 IC 反而低 0.00922。
- 對 topology permutation：只有 1/5 seed 勝出，平均 IC 反而低 0.00832。
- 對 relation permutation：3/5 勝出，但離 5/5 gate 很遠，Wilcoxon p=0.3125。

所以「只要補更多節點、把圖做密一點，就會自然出現報酬訊號」目前沒有證據支持。

### 4.2 對 H2「關係本身不預測報酬」的判定

結果與 H2 一致，但措辭要保守：在本次 44 檔 proxy、2024–2025 時間窗、60 日 rolling correlation、h=20 目標與目前 GNN head 下，沒有觀察到穩定的報酬 alpha。這不是數學上證明所有 correlation graph 都無效，而是說在本研究設定與資料品質下，繼續單純增加圖密度不值得優先投入。

### 4.3 relation permutation 比較好的細節

real 的 IC 平均高於 relation_permuted，且 3/5 seed 勝出；這表示正負相關型別可能攜帶一些資訊，或至少打亂型別會破壞部分可學訊號。然而：

- 沒有達到 5/5 exploratory gate。
- 沒有同時擊敗 self 與 topology permutation。
- p=0.3125，樣本只有 5 個 seed，不能稱為穩定證據。

因此最多只能說「關係符號值得在風險/共動目標進一步檢查」，不能說已證明它能改善報酬交易。

### 4.4 方向準確率的限制

方向準確率約 55% 在四組都相近，而且 self 條件在每個 seed 顯示相同的 0.5534。這說明模型可能學到共同的市場方向或輸出近似常數，而不是利用跨公司圖結構做橫截面排序。因此本 gate 以 Rank IC 為主，不以方向準確率單獨宣稱圖有效。

另有 real seed 45 的 `n_test_days=217`（其餘多為 230），原因是部分日期的預測向量為常數，Spearman IC 無法定義；runner 已略過這些 NaN IC。這是模型訊號退化的診斷警告，並非額外優勢。

## 5. 目前結論與下一步

**Phase 1 的動態 correlation 報酬 gate 未通過。** 這把證據鏈補得更完整：即使使用較大的可用 universe、較密集的圖、每日更新且無明顯 look-ahead 的共動圖，仍無法在目前 supervised head 上穩定產生報酬排序 alpha。

因此目前不應直接投入更昂貴的 LLM-KG 或 RL 長時間 sweep。較合理的下一步是二選一：

1. **轉換任務**：用同一批 dynamic graph 預測未來共動、pairwise spread、realized covariance 或風險暴露，檢查 H3「圖對風險有用、但對報酬無用」。
2. **縮小且預先註冊的敏感度實驗**：只改 rolling window（20/60/120）與 horizon（20/60），確認是否存在明確的時間尺度匹配；若仍無法擊敗 controls，正式接受 H2 型負結果。

在 supervised gate 沒有通過前，不進入 RL portfolio performance 比較，因為那會把「圖沒有可預測訊號」與「RL 沒有學好」混在一起。

## 6. 可重現產物

- Runner：`0601_atten_V2/setup/56_run_dynamic_corr_gate.py`
- Smoke：`0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/phase1_smoke_corr_real_h20/`
- Formal real：`.../phase1_corr_real_h20_5seed/`
- Formal self：`.../phase1_corr_self_h20_5seed/`
- Formal relation permutation：`.../phase1_corr_relation_h20_5seed/`
- Formal topology permutation：`.../phase1_corr_topology_h20_5seed/`
- 價格資料：`0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/price_ohlcv_sp100_available_proxy.csv`
- Universe manifest：`0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/sp100_available_proxy_manifest.csv`
