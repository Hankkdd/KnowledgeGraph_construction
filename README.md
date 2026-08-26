# KnowledgeGraph_construction

以 WRDS/CRSP 為資料基礎，檢驗圖結構（sector、動態相關、供應鏈關係）是否能為橫斷面報酬預測提供
相對於 no-graph 與 shuffle controls 的增量價值。

## 為什麼開這個 repo

前身 `GraphRagTrading/0601_atten_V2` 的結論是：在 Dow 30 與 44 檔 S&P 100 proxy 上，
GraphRAG 抽取的語義 KG、GICS 靜態圖、每日更新的相關圖，都沒有通過 paired control gate。
完整歷程見 `docs/2026-08-21_研究全歷程實驗清單.md`。

那些測試共用一個未被檢驗的前提：**universe 只有 30–44 檔**。
文獻上有訊號的 graph-based 報酬預測研究，universe 中位數在 300–500 檔，強結果多在 1000 檔以上；
KG alpha 的理論機制（limited attention）在超大型股上本來就最弱。

本 repo 的第一個目的，是用 WRDS/CRSP 建立真正 point-in-time、survivorship-free 的大型 universe，
把 universe 規模這個維度補上，讓 H1（稀疏／橫斷面不足）第一次得到公平的測試。

## 資料來源與已知邊界

可用（NYCU 訂閱已確認，2026-08-26）：

| 用途 | Table |
|---|---|
| 日頻價格／報酬（含下市） | `crsp.dsf_v2`、`crsp.stkdlysecuritydata` |
| PIT 證券屬性、SIC／NAICS | `crsp.stksecurityinfohist` |
| GVKEY ↔ PERMNO | `wrdsapps.id` |
| 歷史 GICS | `comp.co_hgic` |
| 供應商→客戶關係 | `compseg.seg_customer` |
| SEC EX-21 子公司（entity resolution 字典） | `wrdsapps.wrds_relationships` |

不可用（無訂閱權限）：

- `crsp.dsp500list` 等 index 檔、`crsp.ccmxpf_*`（CCM 連結）
- `wrdsapps.seglink`（Supply Chain with IDs）、`factset_revere_supply_chain.supply_chain`
- `wrdssec`（SEC Analytics Suite）、`ibes`

兩個直接後果：

1. **拿不到 PIT S&P 500 成分名單**（`comp.idxcst_his` 只有現任成分，departed=0）。
   改用 CRSP 自建 universe：每個 rebalance 日取市值前 N，套用普通股／美國籍／三大所篩選。
   這比成分名單更乾淨，且不需要任何外部資料。
2. **供應鏈的客戶名稱→GVKEY 對映要自建**。`compseg.seg_customer` 的 `cnms` 是自由文字，
   2015 年後 `ctype='COMPANY'` 共 117,634 列，其中具名 60,391 列（51.3%），
   其餘是 `Not Reported`／`N Customers` 匿名彙總。

## 授權限制

WRDS 資料不得進入版控或公開產出。`data/`、`artifacts/` 已在 `.gitignore`。
manifest 只記錄 SQL、抽取日期、列數與 hash，不記錄原始資料。

## 環境

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env    # 填入 WRDS_USER
```

密碼放 `~/.pgpass`（`chmod 400`），不放 repo：

```
wrds-pgdata.wharton.upenn.edu:9737:wrds:<username>:<password>
```

## 研究規範

實驗設計、control 定義、seed 門檻與報告格式見 `docs/RESEARCH_PROTOCOL.md`。
任何 gate 在執行前必須凍結設計，執行後必須產出符合該格式的紀錄。
