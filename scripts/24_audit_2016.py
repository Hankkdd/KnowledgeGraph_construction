"""Stage C2 後續：2016 年的資料稽核。

C2 的 top-500 對 `topology_shuffle` 的優勢有 85% 來自 2016 一年，排除該年後
seed 勝出從 5/5 掉到 3/5。在補 seeds 之前必須先分清楚三件事：

    2016 的效果
        │
        ├─ 資料異常（拆股/股利調整、下市、缺值）  → 修正後重跑受影響的 fold
        ├─ 真實的 regime                          → 保留，但不得只憑它宣稱通過
        └─ 少數幾天或少數股票造成的抽樣雜訊        → 視為雜訊

這支腳本只做資料層檢查，不訓練。逐日與逐股票的預測差異需要重跑模型，
放在 `25_audit_2016_daily.py`。

用法：
    PYTHONPATH=src .venv/bin/python scripts/24_audit_2016.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from kgc import dataset, graphs, wrds_io

UNIVERSE = 500
SUSPECT_YEAR = 2016
OUT_DIR = wrds_io.REPO_ROOT / "artifacts" / "stage_c2_audit_2016"


def per_year_frame(panel: dataset.Panel) -> pd.DataFrame:
    """每個 test 年的資料層統計。"""
    years = pd.DatetimeIndex(panel.calendar).year
    rows = []
    for year in sorted(set(years[years >= 2013])):
        dates = panel.calendar[years == year]
        nodes_n, masked, target_sd, extreme = [], [], [], 0
        edge_turnover, prev_edges = [], None

        for date in dates:
            nodes = panel.nodes_at(date)
            mask = panel.usable_mask(date, nodes)
            kept = nodes[mask]
            if len(kept) < 2:
                continue
            nodes_n.append(len(kept))
            masked.append(len(nodes) - len(kept))

            target, ok = panel.target_at(date, kept)
            if ok.all():
                target_sd.append(float(target.std()))
            window, _ = panel.window_at(date, kept)
            # 拆股或股利調整出錯會表現成單日極端報酬。
            extreme += int((np.abs(window) > 0.5).sum())

            edges = graphs.corr_edges(window, k=5)
            current = set(zip(edges.src.tolist(), edges.dst.tolist()))
            if prev_edges is not None:
                kept_frac = len(current & prev_edges) / max(len(current), 1)
                edge_turnover.append(1.0 - kept_frac)
            prev_edges = current

        rows.append({
            "year": year,
            "test_days": len(nodes_n),
            "nodes_mean": float(np.mean(nodes_n)),
            "nodes_min": int(np.min(nodes_n)),
            "masked_mean": float(np.mean(masked)),
            "target_sd_mean": float(np.mean(target_sd)),
            "target_sd_max": float(np.max(target_sd)),
            "extreme_return_cells": extreme,
            "edge_turnover_mean": float(np.mean(edge_turnover)),
        })
    return pd.DataFrame(rows)


def membership_events(panel: dataset.Panel, members: pd.DataFrame) -> pd.DataFrame:
    """每年的成員進出與期間內停止交易的檔數。"""
    years = sorted(members["rebal_dt"].dt.year.unique())
    by_period = {d: set(g["permno"]) for d, g in members.groupby("rebal_dt")}
    dates = sorted(by_period)

    last_obs = {}
    for pos, permno in enumerate(panel.permnos):
        col = panel.returns[:, pos]
        seen = np.flatnonzero(~np.isnan(col))
        if len(seen):
            last_obs[permno] = panel.calendar[seen[-1]]
    final = panel.calendar[-1]

    rows = []
    for year in years:
        in_year = [d for d in dates if pd.Timestamp(d).year == year]
        entered = departed = 0
        for d in in_year:
            prev = [x for x in dates if x < d]
            if not prev:
                continue
            before, now = by_period[prev[-1]], by_period[d]
            entered += len(now - before)
            departed += len(before - now)
        stopped = sum(1 for p in set().union(*[by_period[d] for d in in_year])
                      if p in last_obs and pd.Timestamp(last_obs[p]).year == year
                      and last_obs[p] < final)
        rows.append({"year": year, "entered": entered, "departed": departed,
                     "stopped_trading": stopped})
    return pd.DataFrame(rows)


def main() -> int:
    panel = dataset.Panel.load(UNIVERSE)
    members = wrds_io.load(f"members_top{UNIVERSE}", subdir="universe")

    yearly = per_year_frame(panel)
    events = membership_events(panel, members)
    merged = yearly.merge(events, on="year", how="left")

    suspect = merged[merged["year"] == SUSPECT_YEAR].iloc[0]
    others = merged[merged["year"] != SUSPECT_YEAR]

    flags = []
    for col in ("nodes_mean", "masked_mean", "target_sd_mean",
                "extreme_return_cells", "edge_turnover_mean",
                "entered", "departed", "stopped_trading"):
        mu, sd = others[col].mean(), others[col].std()
        z = (suspect[col] - mu) / sd if sd > 0 else 0.0
        if abs(z) > 2:
            flags.append(f"`{col}`：2016 為 {suspect[col]:.4g}，"
                         f"其餘年份 {mu:.4g} ± {sd:.4g}（z = {z:+.1f}）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 2016 資料稽核", "",
        f"universe top-{UNIVERSE}，逐年比較資料層指標。"
        f"目的是分辨 2016 的效果是資料異常、真實 regime，還是抽樣雜訊。", "",
        "## 逐年統計", "",
        merged.round(4).to_markdown(index=False), "",
        "## 2016 是否偏離其他年份", "",
    ]
    if flags:
        lines += ["以下指標偏離其他年份超過 2 個標準差：", ""] + [f"- {f}" for f in flags]
    else:
        lines += ["沒有任何資料層指標偏離其他年份超過 2 個標準差。",
                  "2016 在成員數、遮蔽數、target 離散度、極端報酬、邊更替率、",
                  "成員進出與停止交易上都屬正常範圍。", "",
                  "因此 2016 的效果不能歸因於資料異常；"
                  "剩下的解釋是真實 regime 或抽樣雜訊，需要逐日與逐股票的分析區分。"]

    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "yearly.json").write_text(
        json.dumps({"yearly": merged.to_dict("records"), "flags": flags},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(merged.round(4).to_string(index=False))
    print()
    if flags:
        print("偏離 2 個標準差的指標：")
        for f in flags:
            print("  -", f)
    else:
        print("沒有資料層指標偏離其他年份 2 個標準差以上。")
    print(f"\n寫出 {OUT_DIR}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
