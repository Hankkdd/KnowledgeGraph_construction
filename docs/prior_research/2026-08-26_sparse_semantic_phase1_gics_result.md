# 稀疏 vs 無語義 Alpha 判別測試：GICS Phase 1 結果

日期：2026-08-26  
Branch：`experiment/gpm-kg-consumer`  
Decision：`H1 not supported for the tested GICS graph; do not advance to RL`

## Research question

在 44 檔較大 universe 中，增加同 sector 的 GICS graph 是否能讓 real graph 在 out-of-sample 報酬預測上穩定勝過 self 與 topology-shuffle controls？

## Frozen design

- Universe：`SP100_AVAILABLE_PROXY`，44 tickers
- Membership：current S&P100 source snapshot 與 project price intersection；不是 PIT 成分股
- Price：`price_ohlcv_sp100_available_proxy.csv`
- Graph：GICS same-sector，133 undirected edges；loader 後 266 directed edges
- Train/test：2024-01-01～2025-01-01 / 2025-01-01～2026-01-01
- Model：既有 typed GPM/R-GCN supervised head，所有 variant 固定容量
- Pretraining：10 epochs
- Seeds：41–45
- Variants：real、self、topology_permuted
- RL episodes：0
- Primary metric：daily cross-sectional Rank IC
- Secondary：MSE、directional accuracy

## Results

### Horizon 20

| variant | Rank IC mean | MSE mean | directional accuracy |
|---|---:|---:|---:|
| real | 0.002232 | 0.012234 | 0.518063 |
| self | 0.005091 | 0.011421 | 0.516779 |
| topology_permuted | 0.011934 | 0.011837 | 0.513538 |

Paired real-minus-control results：

| metric | control | effect (higher-is-better direction) | real wins | Wilcoxon p |
|---|---|---:|---:|---:|
| Rank IC | self | -0.002858 | 3/5 | 1.0000 |
| Rank IC | topology | -0.009702 | 1/5 | 0.3125 |
| MSE | self | -0.000814 | 0/5 | 0.0625 |
| MSE | topology | -0.000397 | 0/5 | 0.0625 |

### Horizon 60

| variant | Rank IC mean | MSE mean | directional accuracy |
|---|---:|---:|---:|
| real | 0.032328 | 0.036962 | 0.554354 |
| self | 0.040494 | 0.036164 | 0.546818 |
| topology_permuted | 0.023534 | 0.036715 | 0.543780 |

Paired real-minus-control results：

| metric | control | effect (higher-is-better direction) | real wins | Wilcoxon p |
|---|---|---:|---:|---:|
| Rank IC | self | -0.008166 | 2/5 | 0.6250 |
| Rank IC | topology | +0.008794 | 3/5 | 0.6250 |
| MSE | self | -0.000798 | 1/5 | 0.1250 |
| MSE | topology | -0.000247 | 1/5 | 0.4375 |
| directional accuracy | self | +0.007536 | 5/5 | 0.0625 |
| directional accuracy | topology | +0.010574 | 4/5 | 0.1250 |

## Interpretation

### 證明了什麼

1. 44 檔較大 universe 與密集 GICS graph 的 supervised pipeline 可以正常執行。
2. GICS graph 的圖密度高於原始稀疏公司關係圖，但在 h20/h60 報酬 target 上沒有穩定 Rank IC 增益。
3. h60 的 real directional accuracy 有正向探索性結果，但未達 primary metric 與整體 gate 門檻。

### 沒有證明什麼

1. 不能判定所有 dense graph 或所有 universe 都無效。
2. 不能把 GICS sector graph 的失敗等同於 GraphRAG semantic KG 的失敗。
3. 44 檔 proxy 不是 PIT S&P100，結果有 selection/survivorship 限制。

## Decision

**H1 not supported for the tested GICS graph.** 不進入 RL，也不因 h60 direction accuracy 的 5/5 單項結果而調整 primary gate。GICS real 沒有同時勝過 self 與 topology controls。

## Next gate

若要繼續 Phase 1，優先實作真正逐日的 `G_corr` dynamic graph runner，使用每個日期前 60 日資料建圖，並先測 h20/h60 Rank IC。若 G_corr 也失敗，再轉 H3 risk target；不再追加 GICS 超參數或 RL sweep。

## Artifacts

- `0601_atten_V2/setup/55_run_sparse_semantic_gate.py`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/phase1_gics_real_h20_5seed/`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/phase1_gics_self_h20_5seed/`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/phase1_gics_topology_h20_5seed/`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/phase1_gics_real_h60_5seed/`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/phase1_gics_self_h60_5seed/`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/phase1_gics_topology_h60_5seed/`
