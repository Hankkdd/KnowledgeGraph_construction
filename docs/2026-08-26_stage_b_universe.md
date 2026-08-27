# Stage B：CRSP point-in-time universe 與價格資料

日期：2026-08-26
Branch：`feat/stage-b-universe`
性質：資料建置，非 gate——不做任何 real-vs-control 比較，因此不產生 gate 結論

## Research question

能不能在沒有指數成分名單的情況下，用 CRSP 建出真正 point-in-time、含下市證券的
大型股 universe，並取得後續所有 gate 所需的價格資料？

## Why this experiment is next

前身 repo 的所有 gate 都在 30–44 檔上執行且全部未通過，包含 2026-08-26 跑完的
GICS 靜態圖與每日相關圖。這些測試共用一個未被檢驗的前提：universe 只有 30–44 檔。
universe 規模是 H1 唯一還沒被測過的維度。

同時舊實驗一律使用 current-constituent proxy，bias 無法量化。

## Frozen data

| 項目 | 值 |
|---|---|
| 來源 | `crsp.dsf_v2`（CIZ 格式；舊 SIZ 的 `crsp.dsf` 凍結在 2024-12-31，不可用） |
| 期間 | 2010-01-01 – 2025-12-31，192 個月末 rebalance 日 |
| 合格條件 | `sharetype='NS'`、`securitytype='EQTY'`、`securitysubtype='COM'`、`usincflg='Y'`、`primaryexch in ('N','A','Q')`、`dlycap > 0` |
| 排序 | `dlycap` 由大到小，平手以 `permno` 決勝 |
| Universe | top-30 / top-100 / top-500 |
| 價格期間 | 2009-08-03 – 2025-12-31（含 60 交易日 warm-up） |
| 報酬 | `dlyret`，CIZ 已整合下市報酬 |

合格月末列數 765,164，涵蓋 8,182 檔相異證券。

## Result

| | top-30 | top-100 | top-500 |
|---|---:|---:|---:|
| rebalance 日數 | 192 | 192 | 192 |
| 每日成員數皆為 N | 是 | 是 | 是 |
| 相異成員 | 61 | 222 | 1,136 |
| 資格區間重疊 | 0 | 0 | 0 |
| 中斷後再入選 | 126 次 | 405 次 | 1,502 次 |
| 入選期間內價格缺口 | 0 天 | 0 天 | 0 天 |
| 月換手率 平均 / 最高 | 2.74% / 13.33% | 2.76% / 7.00% | 2.24% / 5.40% |
| 60 個有效 return 完整比率 | 99.948% | 99.896% | 99.728% |
| 需 mask 的 member-date | 3 | 20 | 261 |
| 成員資格期間內停止交易 | 0 | 13 | 157 |
| 曾入選、樣本結束前停止交易 | 0 | 26 (11.7%) | 307 (27.0%) |
| `dlydelflg='Y'` 的成員 | 0 | 26 | 307 |
| 前 10 大市值佔比 平均 / 期末 | 55.2% / 71.4% | 34.5% / 51.8% | 22.9% / 38.8% |

覆蓋率以「有效 `dlyret`」計算，不是「有價格列」——一檔股票可能 60 天都有價格列，
但其中幾天的 return 是空的，相關係數就算不出來。

下市數字由兩個獨立來源計算：最後觀測日推斷與 CRSP 的 `dlydelflg`。
三個 universe 的兩個來源都完全一致，代表沒有資料中斷被誤判成下市。

價格檔 3,551,681 列、4,130 個交易日、1,136 檔證券；`dlyret` 缺失 3,633 列（0.102%），
多為上市首日無前收盤價。

同一 rebalance 日滿足 top-30 ⊂ top-100 ⊂ top-500，192 個日期無例外。

## survivorship bias 的量級

曾進入 top-500 的 1,136 檔中，**307 檔（27.0%）在樣本結束前就停止交易**。
其中 157 檔是在仍具成員資格時下市，其餘先跌出 top-500、之後才下市。

用今日成分名單回填歷史會靜靜漏掉這 307 檔。前身 repo 的 44 檔 proxy 正是這種構造，
這也是那些結果只能標為 historical proxy 的原因。

top-30 的對應數字是 0——十六年間沒有任何一檔在市值前 30 名內下市。
規模越大、下市比例越高，這個梯度本身就是 H1 值得測的旁證。

## 集中度隨規模下降

```text
前 10 大市值佔該 universe 的比重（期末 2025-12-31）
        top-30   71.4%   ← 等權配置也幾乎是在賭前十大
        top-100  51.8%
        top-500  38.8%
```

top-30 的投資組合經濟暴露高度集中在前十檔，且成員之間共振較強，
統計上的有效獨立樣本數低於名目值。Rank IC 的名目橫截面樣本仍是 30 檔——
集中度影響的是估計的雜訊水準與經濟暴露，不是計算 IC 時的樣本數。

## warm-up 不足採 asset-level mask

top-500 有 142 個 rebalance 日、共 261 個 member-date 在往前 60 個交易日內
拿不到滿 60 個有效 return。兩種處理方式的代價差距懸殊：

```text
要修正的對象：96,000 個 member-date 中的 261 個（0.272%）
        │
        ├─ 剔除整個 rebalance 日  → 只剩 50/192 日（26.0%）
        └─ 只遮蔽該成員            → 保留 192/192 日
                                     平均每期少 1.4/500 檔，最多一期少 7 檔
```

**凍結為 asset-level mask**：該 (rebalance 日, 成員) 從建圖與評估中排除，
整個日期保留，不補值。mask 由資料本身決定，因此在 real 與所有 control 之間
完全一致——這是它能用的前提，Stage C 必須驗證這一點。

## What this proves

1. 不需要指數成分名單也能建出 point-in-time universe，且完全不依賴外部資料。
2. 三個規模點的建構規則完全一致，規模是唯一的差異，適合作為 H1 的自變數。
3. 資料完整性足以支撐後續 gate：無資格區間重疊、無入選期間內缺口、
   warm-up 覆蓋率最低 99.761%。

## What it does not prove

1. 沒有做任何 real-vs-control 比較，因此對 H1／H2／H3 都沒有證據力。
2. 市值前 N 不等於 S&P N。兩者成員高度重疊但不相同，不可互相代稱。
3. asset-level mask 只在資料層凍結。它與 control 一致這件事要在 Stage C
   實際驗證，不能只靠設計上的推論。

## Decision

`implementation`。資料層完成，進入 Stage C 的圖建構。

## Next gate

Stage C：以 `siccd`／`naics` 建 sector graph、以 60 日報酬建 rolling correlation graph，
在三個 universe 規模上跑 supervised Rank IC gate，controls 依 `RESEARCH_PROTOCOL.md` §2。
主判定是 real-vs-control 效果是否隨 universe 規模單調增強。

## Artifacts

```
src/kgc/universe.py                             選取邏輯與 SQL
src/kgc/validation.py                           完整性檢查
src/kgc/wrds_io.py                              存取層、manifest、型別正規化
scripts/10_build_universe.py                    成員表
scripts/11_pull_prices.py                       日頻價格
scripts/12_validate_universe.py                 驗證報告
tests/test_universe.py                          13 項單元測試
data/universe/members_top{30,100,500}.parquet   （不進版控）
data/prices/daily_prices.parquet                （不進版控）
data/**/*.manifest.json                         （進版控）
artifacts/stage_b_validation/report.md          （進版控）
artifacts/stage_b_validation/summary.json       （進版控）
```

`.gitignore` 按副檔名忽略而不是按目錄——git 無法在被忽略的目錄底下重新納入檔案，
若忽略 `data/` 整個目錄，manifest 就跟著消失，其他人拿到 branch 會看不到 SQL、
SHA-256 與抽取時間。

每份資料旁有 `.manifest.json`，記錄 SQL、參數、列數、SHA-256、抽取時間與 git commit。
`wrds_io.load()` 回讀時驗證 hash，資料被改動過會直接報錯，並統一把
`datetime.date` 轉 datetime64、`Decimal` 轉 float64。後者不轉的話
`np.log`、`np.corrcoef`、`torch.tensor` 都會拒絕接受，而排序與比較不受影響，
問題會一路潛伏到建圖才爆。
