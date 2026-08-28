"""Stage C1: one-fold supervised convergence and runtime smoke.

This is deliberately a smoke test, not a KG result. It runs the same fixed
capacity model over the G_corr controls on one walk-forward fold and records
loss, Rank IC and wall-clock time. No portfolio or RL code is used.

Example (using the repository's torch-enabled environment)::

    PYTHONPATH=src /path/to/python scripts/21_run_gate.py --universe 500 \
        --test-year 2025 --epochs 1 --seed 41 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from kgc import dataset, gate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", type=int, choices=(30, 100, 500), default=500)
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--train-stride", type=int, default=1,
                   help="Use every k-th training date; 1 means full fold.")
    p.add_argument("--test-stride", type=int, default=1,
                   help="Use every k-th test date; 1 means full fold.")
    p.add_argument("--output-dir", type=Path,
                   default=Path("artifacts/stage_c1_smoke"))
    return p.parse_args()


def _git_commit() -> str | None:
    import subprocess
    out = subprocess.run(["git", "rev-parse", "--verify", "HEAD"],
                         capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[1])
    return out.stdout.strip() or None if out.returncode == 0 else None


def _manifest_hashes(universe: int) -> dict:
    """上游資料的 SHA-256，讓每個 run 綁定到它實際用的資料版本。"""
    root = Path(__file__).resolve().parents[1] / "data"
    wanted = {
        "members": root / "universe" / f"members_top{universe}.manifest.json",
        "prices": root / "prices" / "daily_prices.manifest.json",
    }
    out = {}
    for key, path in wanted.items():
        if path.exists():
            out[key] = json.loads(path.read_text(encoding="utf-8"))["sha256"]
    return out


def main() -> int:
    args = parse_args()
    if args.train_stride < 1 or args.test_stride < 1:
        raise SystemExit("stride must be >= 1")
    device = ("cuda" if args.device == "auto" and torch.cuda.is_available()
              else args.device)
    if device == "auto":
        device = "cpu"

    panel = dataset.Panel.load(args.universe)
    folds = dataset.walk_forward(panel.calendar)
    fold = next((f for f in folds if f.test_year == args.test_year), None)
    if fold is None:
        raise SystemExit(f"no walk-forward fold for test year {args.test_year}")
    train_dates = panel.calendar[
        (panel.calendar >= np.datetime64(fold.train_start)) &
        (panel.calendar <= np.datetime64(fold.train_end))
    ][::args.train_stride]
    test_dates = panel.calendar[
        (panel.calendar >= np.datetime64(fold.test_start)) &
        (panel.calendar <= np.datetime64(fold.test_end))
    ][::args.test_stride]

    # The final horizon days have no complete future target.  Apply the same
    # data-only asset mask used by the gate and retain a date only when at
    # least two assets remain for a cross-sectional comparison.
    def has_cross_section(date) -> bool:
        nodes = panel.nodes_at(date)
        return int(panel.usable_mask(date, nodes).sum()) >= 2

    train_dates = np.array([d for d in train_dates if has_cross_section(d)])
    test_dates = np.array([d for d in test_dates if has_cross_section(d)])
    if not len(train_dates) or not len(test_dates):
        raise SystemExit("no usable train/test dates after target mask")

    variants = ("no_graph", "self", "real", "relation_shuffle", "topology_shuffle")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    for variant in variants:
        result = gate.run_config(
            panel, train_dates, test_dates, variant, args.seed,
            epochs=args.epochs, hidden_dim=args.hidden_dim,
            device=device, lr=args.lr,
        )
        result.update({
            "universe": args.universe,
            "test_year": args.test_year,
            "train_dates": len(train_dates),
            "test_dates": len(test_dates),
            "device": device,
            "train_stride": args.train_stride,
            "test_stride": args.test_stride,
            "window": dataset.WINDOW,
            "horizon": dataset.HORIZON,
            "purge_days": dataset.PURGE_DAYS,
        })
        results.append(result)
        print(json.dumps(result, indent=2))

    stem = f"top{args.universe}_y{args.test_year}_seed{args.seed}"
    summary = {
        "kind": "supervised G_corr gate run",
        "universe": args.universe,
        "test_year": args.test_year,
        "seed": args.seed,
        "variants": variants,
        "fold": str(fold),
        "results": results,
        "git_commit": _git_commit(),
        "data_manifests": _manifest_hashes(args.universe),
        "total_runtime_seconds": time.perf_counter() - started,
    }
    (args.output_dir / stem).with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Stage C1 convergence/runtime smoke", "",
             "This is an implementation smoke test; it is not a KG efficacy gate.", "",
             f"- fold: `{fold}`", f"- seed: `{args.seed}`",
             f"- device: `{device}`", f"- total runtime: `{summary['total_runtime_seconds']:.2f}s`", "",
             "| variant | train loss first | train loss last | test Rank IC | test MSE | runtime (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(
            f"| {r['variant']} | {r['train_loss_first']:.6g} | {r['train_loss_last']:.6g} | "
            f"{r['test_rank_ic_mean']:.6g} | {r['test_mse']:.6g} | {r['runtime_seconds']:.2f} |"
        )
    (args.output_dir / stem).with_suffix(".md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
