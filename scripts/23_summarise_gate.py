"""Stage C2：彙整 gate 結果，做 paired 比較與 H1 判定。

統計單位是 seed，不是 (fold, seed)。13 個 fold 的時間區間彼此關聯，把 65 個
組合當成獨立樣本會高估統計力。因此先在每個 seed 內把 13 個 fold 平均，
再以 seed 為勝負單位套 5/5 門檻；同時列出逐 fold 結果，檢查效果是不是
只來自少數年份。

腳本在資料不完整時直接報錯，不會默默產生表格：缺 run、重複 run、
設定不一致、或退化的 run 都會擋下來。

用法：
    PYTHONPATH=src .venv/bin/python scripts/23_summarise_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
IN_DIR = REPO / "artifacts" / "stage_c2_gate"
OUT_DIR = REPO / "artifacts" / "stage_c2_summary"

UNIVERSES = (30, 100, 500)
TEST_YEARS = tuple(range(2013, 2026))
SEEDS = (41, 42, 43, 44, 45)
VARIANTS = ("no_graph", "self", "real", "relation_shuffle", "topology_shuffle")
# real 必須同時勝過這兩個；只贏 no_graph 或 self 不算語義證據。
CONTROLS = ("self", "relation_shuffle", "topology_shuffle")
FIXED_FIELDS = ("epochs", "hidden_dim", "window", "horizon", "purge_days")


def load() -> tuple[pd.DataFrame, list[str]]:
    rows, problems, seen = [], [], set()
    for path in sorted(IN_DIR.glob("top*_y*_seed*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            key = (r["universe"], r["test_year"], r["seed"], r["variant"])
            if key in seen:
                problems.append(f"重複的 run：{key}")
            seen.add(key)
            rows.append({**r, "git_commit": data.get("git_commit"),
                         "manifest_members": data.get("data_manifests", {}).get("members"),
                         "manifest_prices": data.get("data_manifests", {}).get("prices")})

    df = pd.DataFrame(rows)
    if df.empty:
        return df, ["沒有任何結果檔"]

    expected = {(u, y, s, v) for u in UNIVERSES for y in TEST_YEARS
                for s in SEEDS for v in VARIANTS}
    missing = expected - seen
    if missing:
        problems.append(f"缺少 {len(missing)} 個 run，例如 {sorted(missing)[:5]}")

    for field in FIXED_FIELDS:
        if field in df and df[field].nunique() > 1:
            problems.append(f"`{field}` 在不同 run 之間不一致：{sorted(df[field].unique())}")
    for field in ("manifest_members", "manifest_prices"):
        if df[field].nunique(dropna=True) > len(UNIVERSES if field.endswith("members") else [1]):
            problems.append(f"`{field}` 有多個版本，run 用到不同資料")

    degenerate = df[df["constant_predictor"]]
    if len(degenerate):
        problems.append(
            f"{len(degenerate)} 個 run 是常數預測器，其 Rank IC 無意義："
            f"{degenerate[['universe', 'test_year', 'seed', 'variant']].head().to_dict('records')}")
    return df, problems


def paired_by_seed(df: pd.DataFrame) -> pd.DataFrame:
    """每個 seed 先跨 13 個 fold 平均，再算 real 減 control。"""
    per_seed = (df.groupby(["universe", "seed", "variant"])["test_rank_ic_mean"]
                  .mean().unstack("variant"))
    rows = []
    for control in CONTROLS:
        effect = (per_seed["real"] - per_seed[control]).rename("effect").reset_index()
        for universe, g in effect.groupby("universe"):
            wins = int((g["effect"] > 0).sum())
            try:
                p = float(wilcoxon(g["effect"]).pvalue)
            except ValueError:
                p = float("nan")
            rows.append({
                "universe": universe, "control": control,
                "mean_effect": float(g["effect"].mean()),
                "seed_wins": wins, "n_seeds": len(g),
                "wilcoxon_p": p,
                "passes_exploratory": wins == len(g),
            })
    return pd.DataFrame(rows)


def h1_verdict(paired: pd.DataFrame) -> dict:
    """H1 要的是效果隨 universe 增強，不只是 real 贏。"""
    passing = {u: all(paired[(paired.universe == u)]["passes_exploratory"])
               for u in UNIVERSES}
    effects = {u: float(paired[(paired.universe == u)]["mean_effect"].min())
               for u in UNIVERSES}
    monotone = effects[500] > effects[100] > effects[30]

    if passing[500] and monotone:
        verdict = "H1 成立：效果隨 universe 增強，進 C3 補到 10 seeds"
    elif all(passing.values()):
        verdict = "圖有用但 H1 不成立：三個規模都贏，稀疏不是主因"
    elif not any(passing.values()):
        verdict = "H2：三個規模都沒贏，轉 H3 風險 target"
    else:
        verdict = "混合結果，需逐項檢視，不得挑選有利的 universe 陳述"
    return {"passing": passing, "min_effects": effects,
            "monotone_in_universe": monotone, "verdict": verdict}


def main() -> int:
    df, problems = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if problems:
        report = ["# Stage C2 彙整：資料不完整，未產生判定", ""] + \
                 [f"- {p}" for p in problems]
        (OUT_DIR / "summary.md").write_text("\n".join(report), encoding="utf-8")
        print("\n".join(report), file=sys.stderr)
        return 1

    paired = paired_by_seed(df)
    verdict = h1_verdict(paired)

    per_fold = (df.groupby(["universe", "test_year", "variant"])["test_rank_ic_mean"]
                  .mean().unstack("variant").round(4).reset_index())

    lines = [
        "# Stage C2：G_corr supervised gate 彙整", "",
        f"{len(df):,} 個 run，{df['seed'].nunique()} seeds × "
        f"{df['test_year'].nunique()} folds × {df['universe'].nunique()} universes "
        f"× {df['variant'].nunique()} variants。", "",
        f"git commit `{df['git_commit'].iloc[0][:12]}`。", "",
        "統計單位是 seed：每個 seed 先跨 13 個 fold 平均，再以 seed 為勝負單位。",
        "13 個 fold 的時間區間彼此關聯，當成 65 個獨立樣本會高估統計力。", "",
        "## Paired 比較（real 減 control）", "",
        paired.round(5).to_markdown(index=False), "",
        "## H1 判定", "",
        f"- 三個 universe 是否都同時勝過所有 control：{verdict['passing']}",
        f"- 各 universe 的最小效果："
        + ", ".join(f"top-{u} {e:+.5f}" for u, e in verdict["min_effects"].items()),
        f"- 效果是否隨 universe 單調增強：{verdict['monotone_in_universe']}", "",
        f"**{verdict['verdict']}**", "",
        "## 逐 fold 的 Rank IC", "",
        "檢查效果是不是只來自少數年份。", "",
        per_fold.to_markdown(index=False), "",
    ]

    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "paired.json").write_text(
        json.dumps({"paired": paired.to_dict("records"), "verdict": verdict},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n".join(lines[:20]))
    print(f"\n寫出 {OUT_DIR}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
