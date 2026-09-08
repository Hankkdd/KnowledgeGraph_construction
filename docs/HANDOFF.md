# 交接：KG + 資產配置研究的現況

日期：2026-08-27
對象：接手或平行進行這個題目的人
前身 repo：`GraphRagTrading/0601_atten_V2`（歷程見 `docs/prior_research/`）

這份文件自足，不需要口頭補充。讀完應該能回答三件事：
之前做了什麼、為什麼換資料源、現在資料怎麼拿。

## 1. 一句話現況

前身 repo 在 Dow 30 上證明了 GraphRAG 抽取的語義 KG 沒有交易增量價值；
本 repo 換到 WRDS/CRSP、把 universe 擴到 500 檔重測，
結論是**規模不是主因**，圖的優勢只在特定年份出現。

## 2. 前身 repo 的結論（不要重做）

完整清單在 `docs/prior_research/2026-08-21_研究全歷程實驗清單.md`。摘要：

| 已證明 | 依據 |
|---|---|
| RL pipeline 沒壞 | Gate 0 oracle 5/5 勝 random 與 equal-weight |
| 架構在訊號強且即時時學得會圖 | Gate 1 v1.1 合成動態圖 40/40 |
| GraphRAG 語義 KG 在 Dow 30 無交易增益 | static、long-horizon、dynamic event KG 三路皆未過 |

GraphRAG 抽取本身的天花板：held-out recall 39.1%、precision 15.3%。
沒有任何文獻在這種品質的抽取關係上做交易。

**因此本 repo 不再測試**：Dow 30 universe、GraphRAG 開放域關係抽取、
現有 KG relation 的 RL 超參數 sweep。

## 3. 為什麼換 WRDS

舊 repo 用 yfinance，有兩個無法修補的問題：

- **拿不到下市公司的價格。** 2010–2025 曾進入市值前 500 的 1,136 檔中，
  有 307 檔（27.0%）在樣本結束前停止交易。用今日名單回填歷史會靜靜漏掉它們。
- **沒有 point-in-time 成分股名單**，只能用 current-constituent proxy。

CRSP 兩個都解決。

## 4. WRDS 有什麼（NYCU 訂閱，2026-08-26 實測）

用 `select 1 from <table> limit 1` 逐表驗證，不是看目錄——
`information_schema` 會列出沒有權限的表。

### 可用

| Table | 內容 | 範圍 |
|---|---|---|
| `crsp.dsf_v2`、`crsp.stkdlysecuritydata` | 日頻價格、報酬、市值，含已下市 | 1925-12-31 – 2025-12-31 |
| `crsp.stksecurityinfohist` | PIT 證券屬性、`siccd`、`naics`、交易所 | 全期 |
| `compseg.seg_customer` | 供應商→客戶關係，含 `salecs` 銷售額 | 1976 – 2026，734,842 列 |
| `wrdsapps.wrds_relationships` | SEC EX-21 子公司，含 `cik`/`gvkey`/`fdate` | 1996 起 |
| `comp.co_hgic` | 歷史 GICS | 1999 – 2026 |
| `wrdsapps.id` | `permno`/`gvkey`/`ticker`/`cusip` 對照 | 靜態 |

### 不可用

`crsp.dsp500list` 等 index 檔、`crsp.ccmxpf_*`（CCM 連結）、
`wrdsapps.seglink`（Supply Chain with IDs）、`factset_revere_supply_chain`
都是 InsufficientPrivilege；`wrdssec`、`ibes` 是 NotSubscribed。

兩個直接後果：拿不到 PIT S&P 500 成分（改用市值排序自建）、
供應鏈的客戶名稱→GVKEY 對映要自建。

### 一個會讓人踩到的坑

`crsp.dsf`（舊 SIZ 格式）**凍結在 2024-12-31**，CRSP 已於 2025 年初停止更新。
一定要用 CIZ 的 `dsf_v2`，欄位名是 `dlycaldt` / `dlyret` / `dlyprc` / `dlycap`。
用錯的話 2025 整年是空的，而且不會報錯。

## 5. 怎麼重抓資料

### 環境

```bash
git clone git@github.com:Hankkdd/KnowledgeGraph_construction.git
cd KnowledgeGraph_construction
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env        # 填入自己的 WRDS_USER
```

密碼放 `~/.pgpass`，權限必須是 400 或 600，否則 libpq 會**安靜地忽略它**
然後認證失敗：

```
wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>
```

WRDS 帳號若啟用 Duo 2FA，先 `ssh <username>@wrds-cloud.wharton.upenn.edu`
完成一次驗證，之後同一台機器的 pgdata 連線才會通。

### 取數

```bash
PYTHONPATH=src .venv/bin/python scripts/10_build_universe.py    # 成員表
PYTHONPATH=src .venv/bin/python scripts/11_pull_prices.py       # 日頻價格
PYTHONPATH=src .venv/bin/python scripts/12_validate_universe.py # 驗證報告
```

約 10 分鐘。輸出在 `data/`，每份資料旁有 `.manifest.json`
記錄 SQL、列數、SHA-256、抽取時間與 git commit。
`wrds_io.load()` 回讀時驗證 hash，資料被改動過會直接報錯。

**WRDS 資料受授權限制不得進版控。** `.gitignore` 按副檔名擋
（`*.parquet` 等），不是按目錄——git 無法在被忽略的目錄底下重新納入檔案，
若忽略 `data/` 整個目錄，manifest 會跟著消失。

## 6. universe 怎麼建

不使用任何指數成分名單。每個月最後交易日取合格普通股、依市值排序取前 N：

```
篩選條件  sharetype='NS'、securitytype='EQTY'、securitysubtype='COM'
          usincflg='Y'、primaryexch in ('N','A','Q')、dlycap > 0

2010-2025  192 個 rebalance 日
           相異成員 top-30 61 檔、top-100 222 檔、top-500 1,136 檔
           月換手率 2.2% – 2.8%
```

這是完全 point-in-time、survivorship-free 的，且不依賴外部資料。
成員在期間內下市時仍留在當期 universe，`dlyret` 已含下市報酬。

三個規模點的建構規則完全一致，規模是唯一差異——這是為了測
「負結果是不是因為 universe 太小」而設計的。

## 7. 到目前為止的結果

### 圖的定義

- **G_corr**：過去 60 日報酬相關，每節點取 top-k=5，relation 是相關係數正負號。
  in-degree 恆為 5，**不隨 universe 大小變動**。
- **G_sector**：SIC 2 位數同組相連。in-degree 從 top-30 的 2.1 到 top-500 的 24.4，
  差一個數量級，**因此無法分辨「規模變大」與「每個節點收到更多訊息」**，
  已降為描述性圖，不參與判定。

### C2 主 gate（975 runs）

| universe | vs self | vs relation_shuffle | vs topology_shuffle |
|---|---|---|---|
| top-30 | 1/5 | 5/5 | 0/5 |
| top-100 | 0/5 | 2/5 | 0/5 |
| top-500 | 5/5 | 5/5 | 5/5 |

只有 top-500 同時勝過三個 control。但**假設 H1（稀疏是主因）判定不成立**：
最小效果為 top-30 −0.01152、top-100 −0.02176、top-500 +0.00322，
不是單調增強。依預先凍結的規則，這是混合結果。

### 效果的 85% 來自 2016 年

排除 2016 後，top-500 對 `topology_shuffle` 的 seed 勝出從 5/5 掉到 3/5，
平均效果從 +0.00469 掉到 +0.00072。

2016 稽核的結論：**不是資料錯誤**（拆股/股利調整、邊更替率都正常），
是真實的 regime——該年有 26 檔成員停止交易，其他年份平均 9.8 檔。
逐日檢查顯示效果全年普遍存在（去頭尾 5% 後不變、移除最極端 40 天仍保留 41%、
5 個 seed 同向），不是少數幾天造成的。

### C3：未通過（520 runs）

只跑 top-500、seeds 41–50、13 折。判準在看到 2016 稽核結果前凍結。

| 判準 | 結果 | 門檻 |
|---|---:|---:|
| seed-level 勝出 | 7/10 | ≥ 9/10 |
| fold-level 勝出 | 8/13 | ≥ 9/13 |
| leave-one-fold-out | 2/13 維持 | 13/13 |

平均效果 +0.00274，95% CI [−0.00024, +0.00557]——包含 0。三項全部不成立。

**排除 2016 後效果轉為負值**（−0.00080，seed 勝出 4/10）。
2016 不只主導效果，是唯一支撐它的年份。

C2 用 seeds 41–45 得到 5/5，C3 用 41–50 只有 7/10——5 seeds 的探索門檻
對這種量級的效果沒有判別力。

### 量測本身有非決定性

C3 重跑 seeds 41–45 理應與 C2 完全一致，實際不然：`self` 完全一致
（0.00000），其他三個 variant 中位差 0.0007–0.0011、最大 0.044。

原因是 `model.py` 的 `messages.index_add_(0, dst, transformed)`：
`self` 每個目標節點只有一條入邊、無碰撞；其餘 variant 索引重複，
CUDA 用 atomicAdd，累加順序不保證，浮點誤差經 3 epochs 訓練放大。

配對差的合成雜訊每 (seed, fold) 約 0.0027，**與待測效果同量級**。
H3 開始前必須先修（`torch.use_deterministic_algorithms(True)`，
或把 `index_add_` 換成稀疏矩陣乘法）。

### H1 的最終狀態

**不成立。** 效果不隨 universe 單調增強，唯一通過探索門檻的 top-500 也不穩健。
前身 repo 在 Dow 30 上的負結果，不能歸因於 universe 太小。

## 8. 已知限制

模型幾乎不擊敗常數預測器：975 個 run 中只有 20 個在測試期 MSE 上勝過直接
預測橫斷面均值，預測的離散度只有 target 的 11.5%。效果量 0.003–0.010 Rank IC
必須放在這個背景理解。沒有任何交易績效主張。

mask 要求未來 20 日 target 完整，等於**以存活 20 天為條件篩選橫斷面**。
這個 forward-looking selection 對所有 variant 一致，但在高下市年份較強
（2016 為典型年份的 2.5 倍）。更正確的做法是保留中途下市的股票、
用 delisting return 補完部分視窗。

## 9. 下一步的三條路

1. **供應鏈 KG（Stage D）**——`compseg.seg_customer` 是文獻上唯一有正面前例的
   關係型別（Cohen–Frazzini customer momentum），有 `salecs` 可當 edge weight、
   有天然方向性。2015 年後具名客戶 60,391 筆，需自建名稱→GVKEY 對映，
   可用本機 vLLM 批次處理。
2. **風險/共動目標（H3）**——若報酬 gate 最終失敗，改測未來實現共變異數、
   pair spread 或波動度。
3. **regime-conditional 的正面結論**——若 C3 顯示效果只在高下市年份出現，
   那本身是與 limited-attention 機制一致的發現，可以寫。

三條路都不接 RL。依 `docs/RESEARCH_PROTOCOL.md` §7，
supervised gate 與簡單經濟 backtest 都通過才另立 slow-policy RL spec。

## 10. 讀哪些文件

```
docs/RESEARCH_PROTOCOL.md                     gate 設計、control 定義、報告格式
docs/2026-08-26_WRDS資料可得性盤點.md          訂閱範圍與逐表探測結果
docs/2026-08-26_stage_b_universe.md            universe 與價格資料
docs/2026-08-27_stage_c1_smoke.md              實作 smoke（含一條被更正的結論）
docs/2026-08-27_stage_c_positive_control.md    模型學得會圖訊號的證明
docs/2026-08-27_stage_c2_gcorr_gate.md         主 gate 結果
docs/2026-08-27_stage_c3_frozen_design.md      C3 的凍結判準
docs/prior_research/                           前身 repo 的 41 份紀錄
```

方法論的核心在 `RESEARCH_PROTOCOL.md`：real 必須同時勝過保留結構但破壞語義的
controls，只贏 no-graph 或 self 不算；MSE 改善只能解讀為數值校準，不是 alpha。
