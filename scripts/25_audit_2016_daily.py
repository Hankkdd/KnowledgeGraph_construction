"""2016 稽核第二部分：逐日與逐股票的效果分布。

資料層稽核（`24_audit_2016.py`）顯示 2016 沒有拆股/股利調整異常，但下市檔數
是其他年份的 2.6 倍，因而存活條件篩選較強。那說明 2016 的橫斷面組成不同，
但不足以判斷效果是真實 regime 還是少數幾天的雜訊。

這支腳本重跑 2016 折並保存每日 Rank IC，回答兩個問題：

    real 相對 topology_shuffle 的優勢
            │
            ├─ 全年普遍存在                → 傾向真實 regime
            └─ 集中在少數幾天或少數股票    → 傾向抽樣雜訊

同時對照 2017 作為基準年，避免把「任何一年逐日看起來都很吵」誤讀成 2016 特殊。

用法：
    PYTHONPATH=src .venv/bin/python scripts/25_audit_2016_daily.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from kgc import dataset, gate, wrds_io
from kgc.model import GraphReturnModel, node_features

UNIVERSE = 500
YEARS = (2016, 2017)
SEEDS = (41, 42, 43, 44, 45)
VARIANTS = ("real", "topology_shuffle")
EPOCHS = 3
OUT_DIR = wrds_io.REPO_ROOT / "artifacts" / "stage_c2_audit_2016"


def daily_ic(panel, train_dates, test_dates, variant, seed, device):
    """訓練後回傳逐日 Rank IC 與逐股票預測。"""
    torch.manual_seed(seed)
    model = GraphReturnModel(input_dim=4, hidden_dim=64, n_relations=2).to(device)
    gate.train_one(model, panel, train_dates, variant, seed, EPOCHS, device, 1e-3)

    model.eval()
    records, preds = [], {}
    with torch.no_grad():
        for date in test_dates:
            graph, kept = gate.graph_for(panel, date, variant, seed)
            window, _ = panel.window_at(date, kept)
            target, ok = panel.target_at(date, kept)
            if not ok.all():
                continue
            p = model(torch.from_numpy(node_features(window)).to(device),
                      graph).cpu().numpy()
            if np.std(p) == 0:
                continue
            records.append({"date": pd.Timestamp(date), "seed": seed,
                            "variant": variant,
                            "ic": float(spearmanr(p, target).statistic)})
            preds[(pd.Timestamp(date))] = dict(zip(panel.permnos[kept].tolist(),
                                                   p.tolist()))
    return pd.DataFrame(records), preds


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = dataset.Panel.load(UNIVERSE)
    folds = {f.test_year: f for f in dataset.walk_forward(panel.calendar)}

    def window_dates(lo, hi):
        sel = panel.calendar[(panel.calendar >= np.datetime64(lo)) &
                             (panel.calendar <= np.datetime64(hi))]
        return np.array([d for d in sel
                         if panel.usable_mask(d, panel.nodes_at(d)).sum() >= 2])

    all_ic, pred_store = [], {}
    for year in YEARS:
        fold = folds[year]
        tr = window_dates(fold.train_start, fold.train_end)
        te = window_dates(fold.test_start, fold.test_end)
        print(f"{year}: train={len(tr)} test={len(te)}", flush=True)
        for seed in SEEDS:
            for variant in VARIANTS:
                df, preds = daily_ic(panel, tr, te, variant, seed, device)
                df["year"] = year
                all_ic.append(df)
                pred_store[(year, seed, variant)] = preds
                print(f"  {year} seed{seed} {variant:18s} "
                      f"IC {df['ic'].mean():+.4f} over {len(df)} days", flush=True)

    ic = pd.concat(all_ic, ignore_index=True)
    wide = ic.pivot_table(index=["year", "date", "seed"], columns="variant",
                          values="ic").reset_index()
    wide["diff"] = wide["real"] - wide["topology_shuffle"]

    lines = ["# 2016 逐日稽核", "",
             "重跑 2016 與 2017 折，保存每日 Rank IC。"
             "2017 作為基準年，避免把「逐日本來就很吵」誤讀成 2016 特殊。", ""]

    summary = {}
    for year in YEARS:
        sub = wide[wide["year"] == year]
        by_date = sub.groupby("date")["diff"].mean()
        total = by_date.sum()
        ranked = by_date.reindex(by_date.abs().sort_values(ascending=False).index)
        top5_share = float(ranked.head(5).sum() / total) if total else float("nan")
        top20_share = float(ranked.head(20).sum() / total) if total else float("nan")

        summary[year] = {
            "mean_diff": float(by_date.mean()),
            "median_diff": float(by_date.median()),
            "positive_days": int((by_date > 0).sum()),
            "n_days": int(len(by_date)),
            "top5_day_share": top5_share,
            "top20_day_share": top20_share,
        }
        s = summary[year]
        lines += [
            f"## {year}", "",
            f"- 每日 `real − topology_shuffle` 平均 {s['mean_diff']:+.5f}、"
            f"中位數 {s['median_diff']:+.5f}",
            f"- 正值天數 {s['positive_days']}/{s['n_days']}"
            f"（{s['positive_days'] / s['n_days']:.1%}）",
            f"- 絕對值最大的 5 天佔全年總和 {s['top5_day_share']:.1%}",
            f"- 絕對值最大的 20 天佔全年總和 {s['top20_day_share']:.1%}", "",
        ]

    verdict = ("集中在少數幾天，傾向抽樣雜訊"
               if summary[2016]["top5_day_share"] > 0.5
               else "全年普遍存在，傾向真實 regime")
    lines += ["## 判讀", "", f"2016：{verdict}", "",
              "中位數與平均差距越大、少數天數佔比越高，越像雜訊而非穩定現象。", ""]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "daily_report.md").write_text("\n".join(lines), encoding="utf-8")
    ic.to_json(OUT_DIR / "daily_ic.json", orient="records", date_format="iso")
    (OUT_DIR / "daily_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
