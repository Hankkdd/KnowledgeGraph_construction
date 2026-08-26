# KG snapshot archive 操作說明

PostgreSQL 是 lineage/index，不應是唯一的 KG 保存位置。每個 snapshot 都要保留一份可攜式、不可覆寫的檔案 archive。

## 建立 filtered KG 時自動 archive

```bash
cd NYCU/GraphRagTrading/0601_atten_V2
python setup/05_build_pykeen_triples_archived.py --run-id <run_id>
```

這個 wrapper 會先執行既有的 `05_build_pykeen_triples.py`，成功建立 `snapshot_id` 後立即匯出：

```text
artifacts/kg_snapshots/snapshot_<snapshot_id>_<run_id>/
├── snapshot.json
├── snapshot_manifest.json
├── README.md
├── triples.csv
├── triples.parquet
├── rejected_triples.csv
├── triples_with_provenance.csv
├── input_reports.csv
├── entities.parquet
├── relationships.parquet
└── graphrag_output/
```

## 匯出既有 snapshot

```bash
python setup/11_export_kg_snapshot.py --snapshot-id <snapshot_id>
```

不指定 `--snapshot-id` 時使用 DB 中最新 snapshot。archive 目錄若已存在會故意失敗，避免覆寫歷史實驗輸入。

`snapshot_manifest.json` 包含每個檔案的 SHA-256；實驗啟動前應保存該 manifest fingerprint。

## 重要規則

- PostgreSQL row 與檔案 archive 都保留；不要只保存最新的 `graphrag_project/output/`。
- 下游實驗應指定 archive 路徑或 `snapshot_id`，不要使用 mutable output 目錄。
- 新的 prompt、設定、輸入報告或 cutoff 都建立新 run/snapshot，不在原目錄覆寫。
