# KG reproducibility package

建立 filtered KG snapshot 後，若已完成 PyKEEN 訓練，將模型與 embeddings 一起封裝：

```bash
cd NYCU/GraphRagTrading/0601_atten_V2
python setup/12c_finalize_kg_snapshot_package_clean.py \
  --snapshot-id <snapshot_id> \
  --pykeen-run-id <id1> <id2> <id3>
```

輸出位於：

```text
artifacts/kg_packages/snapshot_<snapshot>_<run>_pykeen_<ids>/
├── snapshot_29.../          # KG snapshot files and provenance
├── pykeen/run_<id>/embeddings.csv
├── pykeen/run_<id>/model/
├── pykeen_runs.json
├── package_manifest.json
└── README.md
```

`package_manifest.json` 是 write-once package 的完整 SHA-256 清單。若要重新建立同一組內容，必須產生新 package 目錄，不覆寫舊目錄。
