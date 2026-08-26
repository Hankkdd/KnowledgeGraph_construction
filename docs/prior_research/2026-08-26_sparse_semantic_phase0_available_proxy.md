# 稀疏 vs 無語義 Alpha 判別測試：Phase 0 v2 執行紀錄

日期：2026-08-26  
Branch：`experiment/gpm-kg-consumer`  
Decision：`PASS data coverage; proceed to supervised scaling smoke with explicit proxy limitation`

## 背景

原始 Phase 0 的 S&P100 current proxy 只有 27/101 檔有主價格資料，因此停止進入 supervised gate。檢查 repository 後發現 `data/price_ohlcv.csv`、`data/new_ticker_prices.csv` 與 `data/holdout_ticker_prices.csv` 三份價格檔 ticker 不重疊，可合併成較大的可追溯 proxy。

## Frozen inputs

- S&P100 source：Wikipedia page snapshot `2026-08-26`
- Price sources：三份既有 project bundles；各自保存 SHA-256
- train：`2024-01-01`–`2025-01-01`
- test：`2025-01-01`–`2026-01-01`
- graph snapshots：`2024-01-01`、`2025-01-01`
- rolling correlation：window `60`、top-k `5`

## Result

| Universe | Nodes | Price-covered | Missing | Shared train days | Shared test days | Status |
|---|---:|---:|---|---:|---:|---|
| SP100_AVAILABLE_PROXY | 44 | 44 | none | 252 | 250 | PASS |

Graph statistics：

- GICS same-sector：133 edges、density `0.1406`、mean degree `6.0455`、isolated nodes `1`
- Rolling correlation at 2024-01-01：156 edges、density `0.1649`、mean degree `7.0909`
- Rolling correlation at 2025-01-01：151 edges、density `0.1596`、mean degree `6.8636`

合併價格資料共 `181,718` rows，涵蓋 S&P100 source snapshot 中 44 檔 ticker。三份來源沒有日期/ticker duplicate。

## Interpretation

### 證明了什麼

1. 目前 repository 可以組成 44 檔、完整覆蓋既定 train/test window 的較大 universe proxy。
2. GICS 與 point-in-time rolling-correlation 圖可以在 44 檔 universe 建立。
3. Phase 1 scaling smoke 現在具備可執行的價格與圖輸入。

### 沒有證明什麼

1. 尚未測試 Rank IC、方向率、MSE 或任何風險 target。
2. 尚未判斷 H1/H2/H3。
3. `SP100_AVAILABLE_PROXY` 是「目前來源成分股與可用價格的交集」，不是歷史 point-in-time 成分股；結果必須標記 survivorship/selection bias 限制。

## Decision

Coverage gate PASS，可進入最小成本的 supervised scaling smoke，但不得把結果稱為無 survivorship bias 的 S&P100。若後續需要正式論文級結論，仍應補齊 PIT 成分與缺失 ticker 的歷史價格。

## Next gate

先固定 `SP100_AVAILABLE_PROXY` 與 Dow30，執行 `G_gics`、`G_corr` 的 h20/h60 報酬 Rank IC gate；使用 no-graph、self、topology-matched controls，5 seeds。只在 real 通過預設門檻後，才加入 Wikidata 與 10-seed confirmation。

## Artifacts

- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/sp100_available_proxy_manifest.csv`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/sp100_available_proxy_manifest.json`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/price_ohlcv_sp100_available_proxy.csv`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/available_proxy_validation/coverage_report.md`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/available_proxy_validation/gics_graph_statistics.csv`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/available_proxy_validation/corr_graph_statistics.csv`

## Reproduction commands

```bash
cd /home/hankdd/NYCU/GraphRagTrading/0601_atten_V2
python3 setup/50b_build_available_proxy.py
python3 setup/54_validate_universe_and_graphs.py \
  --manifest artifacts/diagnostics/sparse_semantic_gate_20260826/sp100_available_proxy_manifest.csv \
  --prices artifacts/diagnostics/sparse_semantic_gate_20260826/price_ohlcv_sp100_available_proxy.csv \
  --output-dir artifacts/diagnostics/sparse_semantic_gate_20260826/available_proxy_validation
python3 setup/51_build_gics_graph.py \
  --manifest artifacts/diagnostics/sparse_semantic_gate_20260826/sp100_available_proxy_manifest.csv \
  --output-dir artifacts/diagnostics/sparse_semantic_gate_20260826/available_proxy_validation
python3 setup/52_build_rolling_corr_graph.py \
  --manifest artifacts/diagnostics/sparse_semantic_gate_20260826/sp100_available_proxy_manifest.csv \
  --prices artifacts/diagnostics/sparse_semantic_gate_20260826/price_ohlcv_sp100_available_proxy.csv \
  --output-dir artifacts/diagnostics/sparse_semantic_gate_20260826/available_proxy_validation
```
