"""Stage C3 彙整：三項凍結判準與 leave-one-fold-out。

判準在 `docs/2026-08-27_stage_c3_frozen_design.md` 凍結，於看到 2016 稽核結果
之前寫定。這支腳本只執行那些判準，不引入新的門檻。

    通過 = 三項同時成立
            │
            ├─ seed-level  ≥ 9/10
            ├─ fold-level  ≥ 9/13
            └─ leave-one-fold-out 13 次結論都不翻轉

主要 control 固定為 `topology_shuffle`。`self` 只是價格模型基準，贏它不構成
語義證據，因此列出但不參與判定。

用法：
    PYTHONPATH=src .venv/bin/python scripts/26_summarise_c3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO = Path(__file__).resolve().parents[1]
IN_DIR = REPO / "artifacts" / "stage_c3_gate"
C2_DIR = REPO / "artifacts" / "stage_c2_gate"
OUT_DIR = REPO / "artifacts" / "stage_c3_summary"

UNIVERSE = 500
TEST_YEARS = tuple(range(2013, 2026))
SEEDS = tuple(range(41, 51))
VARIANTS = ("self", "real", "relation_shuffle", "topology_shuffle")
PRIMARY = "topology_shuffle"
SECONDARY = "relation_shuffle"

SEED_THRESHOLD = 9
FOLD_THRESHOLD = 9
BOOTSTRAP = 10_000


def load(directory: Path, variants=VARIANTS) -> tuple[pd.DataFrame, list[str]]:
    rows, problems, seen = [], [], set()
    for path in sorted(directory.glob("top*_y*_seed*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for r in data.get("results", []):
            key = (r["universe"], r["test_year"], r["seed"], r["variant"])
            if key in seen:
                problems.append(f"重複的 run：{key}")
            seen.add(key)
            rows.append({**r, "git_commit": data.get("git_commit")})

    df = pd.DataFrame(rows)
    if df.empty:
        return df, ["沒有任何結果檔"]

    expected = {(UNIVERSE, y, s, v) for y in TEST_YEARS
                for s in SEEDS for v in variants}
    missing = expected - seen
    if missing:
        problems.append(f"缺少 {len(missing)} 個 run，例如 {sorted(missing)[:5]}")
    for field in ("epochs", "hidden_dim", "window", "horizon", "purge_days"):
        if field in df and df[field].nunique() > 1:
            problems.append(f"`{field}` 不一致：{sorted(df[field].unique())}")
    if df["constant_predictor"].any():
        n = int(df["constant_predictor"].sum())
        problems.append(f"{n} 個 run 是常數預測器，其 Rank IC 無意義")
    return df, problems


def effects(df: pd.DataFrame, control: str) -> pd.DataFrame:
    """每個 (seed, fold) 的 real 減 control。"""
    wide = (df.pivot_table(index=["seed", "test_year"], columns="variant",
                           values="test_rank_ic_mean").reset_index())
    wide["effect"] = wide["real"] - wide[control]
    return wide


def leave_one_fold_out(eff: pd.DataFrame) -> pd.DataFrame:
    """逐一排除每折，重算 seed-level 勝出數與平均效果。"""
    rows = []
    for year in sorted(eff["test_year"].unique()):
        kept = eff[eff["test_year"] != year]
        per_seed = kept.groupby("seed")["effect"].mean()
        rows.append({
            "excluded_year": year,
            "seed_wins": int((per_seed > 0).sum()),
            "mean_effect": float(per_seed.mean()),
            "still_passes": bool((per_seed > 0).sum() >= SEED_THRESHOLD),
        })
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, rng) -> tuple[float, float]:
    draws = rng.choice(values, size=(BOOTSTRAP, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def assess(df: pd.DataFrame, control: str, rng) -> dict:
    eff = effects(df, control)
    per_seed = eff.groupby("seed")["effect"].mean()
    per_fold = eff.groupby("test_year")["effect"].mean()
    lofo = leave_one_fold_out(eff)

    lo, hi = bootstrap_ci(per_seed.to_numpy(), rng)
    try:
        p = float(wilcoxon(per_seed).pvalue)
    except ValueError:
        p = float("nan")

    return {
        "control": control,
        "mean_effect": float(per_seed.mean()),
        "ci95": [lo, hi],
        "wilcoxon_p": p,
        "seed_wins": int((per_seed > 0).sum()),
        "n_seeds": int(len(per_seed)),
        "fold_wins": int((per_fold > 0).sum()),
        "n_folds": int(len(per_fold)),
        "seed_criterion": bool((per_seed > 0).sum() >= SEED_THRESHOLD),
        "fold_criterion": bool((per_fold > 0).sum() >= FOLD_THRESHOLD),
        "lofo_criterion": bool(lofo["still_passes"].all()),
        "lofo_failures": lofo.loc[~lofo["still_passes"], "excluded_year"].tolist(),
        "per_fold": per_fold.round(5).to_dict(),
        "lofo": lofo.round(5).to_dict("records"),
    }


def main() -> int:
    df, problems = load(IN_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if problems:
        text = ["# Stage C3 彙整：資料不完整，未產生判定", ""] + \
               [f"- {p}" for p in problems]
        (OUT_DIR / "summary.md").write_text("\n".join(text), encoding="utf-8")
        print("\n".join(text), file=sys.stderr)
        return 1

    rng = np.random.default_rng(41)
    primary = assess(df, PRIMARY, rng)
    secondary = assess(df, SECONDARY, rng)
    passed = (primary["seed_criterion"] and primary["fold_criterion"]
              and primary["lofo_criterion"])

    # 與 C2 的重現性檢查：seeds 41-45 的 top-500 應該完全一致。
    reproduced = None
    if C2_DIR.exists():
        c2, c2_problems = load(C2_DIR, variants=VARIANTS + ("no_graph",))
        if not c2_problems or "缺少" in " ".join(c2_problems):
            c2 = c2[(c2.universe == UNIVERSE) & (c2.seed <= 45)]
            merged = df.merge(c2, on=["universe", "test_year", "seed", "variant"],
                              suffixes=("_c3", "_c2"))
            if len(merged):
                delta = (merged["test_rank_ic_mean_c3"]
                         - merged["test_rank_ic_mean_c2"]).abs()
                reproduced = {"n_compared": int(len(merged)),
                              "max_abs_diff": float(delta.max()),
                              "identical": bool(delta.max() < 1e-9)}

    lines = [
        "# Stage C3：top-500 穩健性確認", "",
        f"{len(df):,} 個 run，{df['seed'].nunique()} seeds × "
        f"{df['test_year'].nunique()} folds × {len(VARIANTS)} variants。", "",
        "判準在 `docs/2026-08-27_stage_c3_frozen_design.md` 凍結，",
        "於看到 2016 稽核結果之前寫定。", "",
        "## H1 的狀態不因本輪改變", "",
        "H1 要求效果隨 universe 規模增強。C2 量到的最小效果為 top-30 −0.01152、",
        "top-100 −0.02176、top-500 +0.00322，不是單調，**H1 已判定不成立**。",
        "C3 只檢驗 top-500 的微弱優勢是否跨 seed、fold 與 leave-one-fold-out 穩健。", "",
        "## 主要判定（control = `topology_shuffle`）", "",
        f"| 判準 | 結果 | 門檻 | 通過 |",
        f"|---|---:|---:|---|",
        f"| seed-level 勝出 | {primary['seed_wins']}/{primary['n_seeds']} | "
        f"≥ {SEED_THRESHOLD}/10 | {'是' if primary['seed_criterion'] else '否'} |",
        f"| fold-level 勝出 | {primary['fold_wins']}/{primary['n_folds']} | "
        f"≥ {FOLD_THRESHOLD}/13 | {'是' if primary['fold_criterion'] else '否'} |",
        f"| leave-one-fold-out | "
        f"{13 - len(primary['lofo_failures'])}/13 維持 | 13/13 | "
        f"{'是' if primary['lofo_criterion'] else '否'} |", "",
        f"平均效果 {primary['mean_effect']:+.5f}，"
        f"95% CI [{primary['ci95'][0]:+.5f}, {primary['ci95'][1]:+.5f}]，"
        f"Wilcoxon p = {primary['wilcoxon_p']:.4f}", "",
        f"**{'通過' if passed else '未通過'}**（三項須同時成立）", "",
    ]

    if primary["lofo_failures"]:
        lines += [f"排除以下年份會使結論翻轉：{primary['lofo_failures']}", "",
                  "依凍結規則，移除任一折即翻轉者不得稱為穩健通過。", ""]

    lines += ["## leave-one-fold-out 明細", "",
              pd.DataFrame(primary["lofo"]).to_markdown(index=False), "",
              "## 逐 fold 效果", "",
              pd.Series(primary["per_fold"]).to_frame("effect")
                .to_markdown(), "",
              "## 次要 control（`relation_shuffle`）", "",
              f"seed {secondary['seed_wins']}/{secondary['n_seeds']}、"
              f"fold {secondary['fold_wins']}/{secondary['n_folds']}、"
              f"平均效果 {secondary['mean_effect']:+.5f}", "",
              "列出供參考，不參與判定。", ""]

    if reproduced:
        lines += ["## 與 C2 的重現性", "",
                  f"seeds 41–45 的 top-500 共 {reproduced['n_compared']} 個 run 對照，"
                  f"最大絕對差 {reproduced['max_abs_diff']:.2e}"
                  f"（{'完全一致' if reproduced['identical'] else '不一致，需檢查非決定性來源'}）。", ""]

    lines += ["## 已知限制", "",
              "沿用 Stage B 的 mask：要求未來 20 日 target 完整，等於以存活 20 天",
              "為條件篩選橫斷面。此 forward-looking selection 對所有 variant 一致，",
              "但在高下市年份較強（2016 為典型年份的 2.5 倍）。", ""]

    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "verdict.json").write_text(
        json.dumps({"passed": passed, "primary": primary,
                    "secondary": secondary, "reproduced": reproduced},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
