# 稀疏 vs 無語義 Alpha 判別測試：Phase 1 supervised smoke

日期：2026-08-26  
Branch：`experiment/gpm-kg-consumer`  
Decision：`implementation smoke passed; no statistical decision`

## Research question

確認 44 檔 `SP100_AVAILABLE_PROXY`、GICS graph 與既有 GPM supervised head 能否端到端執行，且 real、self、topology-shuffle 使用相同模型與資料流程。

## Frozen design

- Universe：`SP100_AVAILABLE_PROXY`，44 tickers
- Price：`price_ohlcv_sp100_available_proxy.csv`
- Graph：GICS same-sector graph，133 undirected edges（loader 後 266 directed edges）
- Train/test：2024-01-01～2025-01-01 / 2025-01-01～2026-01-01
- Target：future cumulative return，horizon=20
- Seeds：`[41]`
- Pretraining：1 epoch
- Variants：real、self、topology_permuted
- RL episodes：0

## Result

| variant | test Rank IC mean | test MSE | directional accuracy | test days | asset-days |
|---|---:|---:|---:|---:|---:|
| real | 0.000599 | 0.014685 | 0.518577 | 230 | 10120 |
| self | 0.000126 | 0.016474 | 0.522431 | 230 | 10120 |
| topology_permuted | -0.001294 | 0.015028 | 0.527569 | 230 | 10120 |

## Interpretation

### 證明了什麼

1. 44 檔 proxy 的 GPM supervised pipeline 可正常建立資料、圖、模型、訓練與測試輸出。
2. real、self、topology-shuffle 使用相同的 44-node input 與 target 流程。
3. smoke 產生了 230 test days、10,120 asset-days，時間切分與維度正常。

### 沒有證明什麼

1. 單一 seed、1 epoch 不足以判斷 real 是否勝出。
2. Rank IC 差異未做 paired seed test，也不能宣稱 alpha。
3. 尚未測試 h60、G_corr、relation-shuffle 或 risk target。

## Decision

**Implementation smoke passed.** 不把 smoke 數字升格為 H1/H2/H3 結論；進入正式 Phase 1 前，先固定 runner 與 artifact schema，再跑 5 seeds × h20/h60 × real/self/topology controls。

## Artifacts

- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/phase1_smoke_gics_real_h20/`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/phase1_smoke_gics_self_h20/`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/phase1_smoke_gics_topology_h20/`

## Reproduction command pattern

```bash
cd /home/hankdd/NYCU/GraphRagTrading/0601_atten_V2
PYTHONPATH=.venv-aarch64/lib/python3.12/site-packages python3 setup/55_run_sparse_semantic_gate.py \
  --manifest artifacts/diagnostics/sparse_semantic_gate_20260826/sp100_available_proxy_manifest.csv \
  --prices artifacts/diagnostics/sparse_semantic_gate_20260826/price_ohlcv_sp100_available_proxy.csv \
  --graph-edges artifacts/diagnostics/sparse_semantic_gate_20260826/available_proxy_validation/graphs/G_gics/SP100_AVAILABLE_PROXY.edges.csv \
  --output-dir artifacts/diagnostics/sparse_semantic_gate_20260826/phase1_smoke_gics_real_h20 \
  --horizon 20 --seeds 1 --pretrain-epochs 1 --variant real --device cuda
```
