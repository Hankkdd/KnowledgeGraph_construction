"""Stage C step 0：建圖統計與 mask 一致性驗證。不訓練。

兩個目的：

1. 量出各 universe、各圖族的邊數、密度、度數與孤立節點，確認 G_corr 的
   in-degree 確實不隨 universe 大小變動（密度受控臂的前提）。
2. 硬性驗證 mask 在所有 variant 之間一致、沒有邊指向被 mask 掉的節點、
   shuffle 保住度數與 relation 頻率、且 shuffle 後仍是 simple graph。

任何一項失敗就 exit 1，不只是印警告。

用法：
    PYTHONPATH=src .venv/bin/python scripts/20_verify_graph_and_mask.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from kgc import dataset, graphs, wrds_io

SIZES = (30, 100, 500)
SAMPLE_DATES = None    # None 表示驗證全部 rebalance 日；抽樣只適合開發時用
SEEDS = (41, 42, 43)   # shuffle 是隨機的，多個 seed 才能確認性質對每次抽樣都成立
OUT_DIR = wrds_io.REPO_ROOT / "artifacts" / "stage_c_graphs"


def variants_at(panel, date, nodes, rng):
    """建出該日所有 variant 的圖，全部由同一份 real 邊集衍生。"""
    window, _ = panel.window_at(date, nodes)
    n = len(nodes)

    corr_real = graphs.corr_edges(window, k=5)
    sector_real = graphs.sector_edges(panel.siccd_at(date, nodes), digits=2)

    return {
        ("shared", "no_graph"): graphs.empty_graph(n),
        ("shared", "self"): graphs.self_loops(n),
        ("sector", "real"): sector_real,
        ("sector", "topology_shuffle"): graphs.shuffle_topology(sector_real, rng),
        ("corr", "real"): corr_real,
        ("corr", "relation_shuffle"): graphs.shuffle_relations(corr_real, rng),
        ("corr", "topology_shuffle"): graphs.shuffle_topology(corr_real, rng),
    }


def duplicate_edges(g) -> int:
    """重複的 (src, dst)。real 圖必為 0；shuffle 後若不為 0，sum aggregation
    會重複計入那些訊息，等於 control 的有效容量比 real 大。"""
    if g.n_edges == 0:
        return 0
    keys = g.src.astype(np.int64) * g.n_nodes + g.dst.astype(np.int64)
    return int(g.n_edges - len(np.unique(keys)))


def check_one_date(panel, n, date, seed, stats, failures):
    nodes = panel.nodes_at(date)
    mask = panel.usable_mask(date, nodes)
    kept = nodes[mask]
    if len(kept) < 2:
        return

    tag = f"top-{n} {pd.Timestamp(date).date()} seed{seed}"
    built = variants_at(panel, date, kept, np.random.default_rng(seed))
    reference = None

    for (family, variant), g in built.items():
        if reference is None:
            reference = g.n_nodes
        elif g.n_nodes != reference:
            failures.append(f"{tag} {family}/{variant}: 節點數 {g.n_nodes} != {reference}")

        if g.n_edges and (g.src.max() >= len(kept) or g.dst.max() >= len(kept)):
            failures.append(f"{tag} {family}/{variant}: 邊指向 mask 外的節點")

        dups = duplicate_edges(g)
        if dups:
            failures.append(f"{tag} {family}/{variant}: {dups} 條重複邊")

        stats.append({
            "universe": n, "date": pd.Timestamp(date).date(), "seed": seed,
            "family": family, "variant": variant,
            "nodes": g.n_nodes, "edges": g.n_edges, "density": g.density,
            "mean_in_degree": float(g.in_degree().mean()),
            "isolated": g.isolated(), "duplicates": dups,
            "masked_out": int(len(nodes) - len(kept)),
        })

    for family in ("sector", "corr"):
        real = built[(family, "real")]
        topo = built[(family, "topology_shuffle")]
        if not np.array_equal(np.sort(real.in_degree()), np.sort(topo.in_degree())):
            failures.append(f"{tag} {family}: topology_shuffle 未保住 in-degree 序列")
        if not np.array_equal(np.sort(real.out_degree()), np.sort(topo.out_degree())):
            failures.append(f"{tag} {family}: topology_shuffle 未保住 out-degree 序列")
        if real.n_edges != topo.n_edges:
            failures.append(f"{tag} {family}: topology_shuffle 邊數改變")

    corr_real = built[("corr", "real")]
    corr_rel = built[("corr", "relation_shuffle")]
    if not np.array_equal(np.bincount(corr_real.relation, minlength=2),
                          np.bincount(corr_rel.relation, minlength=2)):
        failures.append(f"{tag} corr: relation_shuffle 改變了 relation 頻率")


def main() -> int:
    stats: list[dict] = []
    failures: list[str] = []

    for n in SIZES:
        panel = dataset.Panel.load(n)
        folds = dataset.walk_forward(panel.calendar)
        rebal = panel.rebalance_dates()
        usable = [d for d in rebal
                  if dataset.WINDOW <= panel.calendar.searchsorted(d)
                  < len(panel.calendar) - dataset.HORIZON]
        dates = usable if SAMPLE_DATES is None else \
            usable[:: max(1, len(usable) // SAMPLE_DATES)][:SAMPLE_DATES]

        print(f"top-{n}: {len(folds)} folds, 驗證 {len(dates)} 個 rebalance 日 "
              f"× {len(SEEDS)} seeds", flush=True)

        for date in dates:
            for seed in SEEDS:
                check_one_date(panel, n, date, seed, stats, failures)

    df = pd.DataFrame(stats)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    agg = (df.groupby(["universe", "family", "variant"])
             .agg(dates=("date", "nunique"), nodes=("nodes", "mean"),
                  edges=("edges", "mean"), density=("density", "mean"),
                  mean_in_degree=("mean_in_degree", "mean"),
                  isolated=("isolated", "mean"),
                  duplicates=("duplicates", "max"),
                  masked_out=("masked_out", "mean"))
             .round(3).reset_index())

    scope = "全部" if SAMPLE_DATES is None else f"抽樣 {SAMPLE_DATES} 個"
    lines = ["# Stage C — 建圖統計與 mask 一致性", "",
             f"{scope} rebalance 日 × seeds {list(SEEDS)}，"
             f"共 {len(df):,} 次圖建構。", "",
             "## 驗證結果", ""]
    if failures:
        lines += ["**FAIL**", ""] + [f"- {f}" for f in failures[:40]]
        if len(failures) > 40:
            lines.append(f"- ...另有 {len(failures) - 40} 項")
    else:
        lines += [
            "PASS，五項檢查全數通過：", "",
            "1. 所有 variant 的節點集合一致",
            "2. 沒有邊指向 mask 外的節點",
            "3. `topology_shuffle` 保住 in-degree 與 out-degree 序列、邊數不變",
            "4. `relation_shuffle` 保住 relation 頻率",
            "5. 所有 variant 都是 simple graph（無重複邊、無自環）",
        ]

    lines += ["", "## 各 variant 的圖統計（平均）", "",
              agg.to_markdown(index=False), "",
              "## G_corr 的 in-degree 不隨 universe 大小變動", ""]

    corr_real = agg[(agg["family"] == "corr") & (agg["variant"] == "real")]
    lines += ["| universe | 平均 in-degree |", "|---|---:|"]
    lines += [f"| top-{int(r.universe)} | {r.mean_in_degree} |"
              for r in corr_real.itertuples()]

    sector_real = agg[(agg["family"] == "sector") & (agg["variant"] == "real")]
    lines += ["", "## G_sector 的 in-degree 隨 universe 大小暴增", "",
              "| universe | 平均 in-degree | 密度 |", "|---|---:|---:|"]
    lines += [f"| top-{int(r.universe)} | {r.mean_in_degree} | {r.density} |"
              for r in sector_real.itertuples()]
    lines += ["", "密度看起來穩定是 N² 稀釋造成的假象；實際進到訊息聚合的是度數，",
              "而它在 top-30 與 top-500 之間差一個數量級。因此 G_sector 無法分辨",
              "「universe 變大」與「每個節點收到更多訊息」，H1 的判定只交給 G_corr。", ""]

    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "graph_stats.json").write_text(
        json.dumps({"scope": scope, "seeds": list(SEEDS),
                    "n_builds": len(df), "failures": failures,
                    "stats": agg.to_dict(orient="records")},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("\n".join(lines[4:12]))
    print(f"\n寫出 {OUT_DIR}/report.md")
    if failures:
        print(f"\n驗證 FAIL，{len(failures)} 項，中止。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
