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
    cache = OUT_DIR / "daily_ic.json"
    if "--from-cache" in __import__("sys").argv:
        if not cache.exists():
            raise SystemExit(f"{cache} 不存在，先完整跑一次")
        ic = pd.read_json(cache)
        return report(ic)

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
    ic.to_json(OUT_DIR / "daily_ic.json", orient="records", date_format="iso")
    return report(ic)


def report(ic: pd.DataFrame) -> int:
    wide = ic.pivot_table(index=["year", "date", "seed"], columns="variant",
                          values="ic").reset_index()
    wide["diff"] = wide["real"] - wide["topology_shuffle"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# 2016 逐日稽核", "",
             "重跑 2016 與 2017 折，保存每日 Rank IC。"
             "2017 作為基準年，避免把「逐日本來就很吵」誤讀成 2016 特殊。", ""]

    # 集中度不能用「最大 N 天佔全年總和的比例」衡量：總和接近 0 時分母趨近零，
    # 比例會變成無意義的大數或負數。改用移除最極端天數後殘留的效果，
    # 那個量在總和接近零時仍然可解讀。
    summary = {}
    for year in YEARS:
        by_date = wide[wide["year"] == year].groupby("date")["diff"].mean()
        order = by_date.abs().sort_values(ascending=False).index
        trimmed = by_date[(by_date >= by_date.quantile(0.05)) &
                          (by_date <= by_date.quantile(0.95))].mean()
        drop40 = float(by_date.drop(order[:40]).mean())
        per_seed = wide[wide["year"] == year].groupby("seed")["diff"].mean()

        summary[year] = {
            "mean_diff": float(by_date.mean()),
            "median_diff": float(by_date.median()),
            "trimmed_mean_5_95": float(trimmed),
            "positive_days": int((by_date > 0).sum()),
            "n_days": int(len(by_date)),
            "mean_after_dropping_10": float(by_date.drop(order[:10]).mean()),
            "mean_after_dropping_40": drop40,
            "retained_after_dropping_40": drop40 / float(by_date.mean()),
            "seed_wins": int((per_seed > 0).sum()),
            "n_seeds": int(len(per_seed)),
        }
        s = summary[year]
        lines += [
            f"## {year}", "",
            f"- 每日 `real − topology_shuffle` 平均 {s['mean_diff']:+.5f}、"
            f"中位數 {s['median_diff']:+.5f}、去頭尾 5% 後 {s['trimmed_mean_5_95']:+.5f}",
            f"- 正值天數 {s['positive_days']}/{s['n_days']}"
            f"（{s['positive_days'] / s['n_days']:.1%}）",
            f"- 移除絕對值最大的 10 天後 {s['mean_after_dropping_10']:+.5f}、"
            f"40 天後 {s['mean_after_dropping_40']:+.5f}"
            f"（保留原效果的 {s['retained_after_dropping_40']:.0%}）",
            f"- seed 勝出 {s['seed_wins']}/{s['n_seeds']}", "",
        ]

    # 去頭尾後效果不減、且移除 40 天仍保留可觀比例，才算全年普遍存在。
    s16 = summary[2016]
    broad = (s16["trimmed_mean_5_95"] >= 0.8 * s16["mean_diff"]
             and s16["retained_after_dropping_40"] >= 0.3
             and s16["seed_wins"] == s16["n_seeds"])
    verdict = ("全年普遍存在，且五個 seed 一致，傾向真實 regime" if broad
               else "集中在少數幾天或 seed 之間不一致，傾向抽樣雜訊")
    lines += ["## 判讀", "", f"2016：{verdict}", "",
              "判準是三項同時成立：去頭尾 5% 後效果不低於原值的八成、",
              "移除絕對值最大的 40 天後仍保留三成以上、五個 seed 方向一致。",
              "任一項不成立就代表效果依賴少數觀測或特定初始化。", ""]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "daily_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "daily_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
