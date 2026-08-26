# 稀疏 vs 無語義 Alpha 判別測試：Phase 0 執行紀錄

日期：2026-08-26  
Branch：`experiment/gpm-kg-consumer`  
Decision：`STOP before supervised gate`

## Research question

較大 universe 與密集圖能否在時間正確、資料完整的前提下建立，作為後續 H1/H2/H3 supervised gate 的輸入？

## Why this experiment was next

既有固定五檔/Dow30 結果無法區分圖太稀疏、橫截面太小與語義關係沒有報酬訊號。因此先做不訓練模型的資料可行性檢查，避免把價格缺失或 survivorship bias 誤判成模型結果。

## Frozen design

- Train：`2024-01-01`–`2025-01-01`
- Test：`2025-01-01`–`2026-01-01`
- Universe：既有 Dow30；Wikipedia S&P100 current constituent proxy
- Graph：GICS same-sector；rolling correlation（window=60、top-k=5）
- 建圖只使用截至 `as_of` 的價格；manifest、source HTML、hash 與 graph statistics 均保存
- 本階段不訓練 supervised model，也不跑 RL

## Result

| Universe | Nodes | Price-covered | Shared train days | Shared test days | Status |
|---|---:|---:|---:|---:|---|
| Dow30 | 30 | 30 | 252 | 250 | PASS |
| SP100_CURRENT_PROXY | 101 | 27 | 252 | 250 | FAIL |

GICS graph：

- Dow30：47 edges，density 0.1080，isolated nodes 2
- S&P100 proxy：620 edges，density 0.1228，isolated nodes 1

Rolling-correlation graph：

- Dow30 的 `2024-01-01` 與 `2025-01-01` snapshots 成功建立
- S&P100 proxy 因 74 檔缺少價格而跳過

S&P100 source HTML SHA-256：`a7e2adab4ed530b652f91ca74103094065cd82b0da0f4ddfcf3efb8d9da541a2`  
Price file SHA-256：`08bac9cdaba0ffde82fac4f43d4f8b50714d8e57ef96cf93f89e712d8e435140`

## What this proves

1. Phase 0 的 universe manifest、GICS graph 與 Dow30 rolling graph 建構流程可執行。
2. Dow30 資料仍可重現既有 walk-forward 時間切分。
3. 目前 repository 的價格資料不能直接支援較大的 S&P universe。

## What this does not prove

1. 尚未測試任何 supervised 報酬或風險 target。
2. 尚未判斷 H1、H2 或 H3。
3. `SP100_CURRENT_PROXY` 不是 point-in-time 成分股，不能當作無 survivorship bias 的歷史 S&P100。

## Decision

不使用只有 27 檔重疊股票的 S&P proxy 進入模型，也不把資料不完整的結果降級成不對等比較。先補齊可追溯的大 universe 歷史價格 bundle、ticker rename/delisting mapping 與 coverage audit。

## Next gate

取得合格的大 universe 價格資料後，先跑 `G_gics` 與 `G_corr` 的 h20/h60 supervised gate；若仍無訊號，再測 risk target。沒有 supervised gate 通過前，不重新接 RL。

## Artifacts

- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/coverage_report.md`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/phase0_summary.json`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/universe_manifest.csv`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/gics_graph_statistics.csv`
- `0601_atten_V2/artifacts/diagnostics/sparse_semantic_gate_20260826/corr_graph_statistics.csv`

## Reproduction commands

```bash
cd /home/hankdd/NYCU/GraphRagTrading/0601_atten_V2
python3 setup/50_build_dense_universe.py
python3 setup/51_build_gics_graph.py
python3 setup/52_build_rolling_corr_graph.py
python3 setup/54_validate_universe_and_graphs.py
```
