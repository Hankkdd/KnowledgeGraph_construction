"""派工並平行執行 G_corr gate 格點（C2 與 C3 共用）。

每個 (universe, test_year, seed) 是一個獨立 subprocess，內含所有 variant。
這個粒度讓失敗只損失一個工作單位，而 Panel 每個 subprocess 只載入一次。

排程器本身不做任何科學判斷，只負責：完成的跳過、失敗的保留 stderr、
結束時檢查有沒有缺漏或殘檔。彙整與判定在 `23_summarise_gate.py`（C2）
與 `26_summarise_c3.py`（C3）。

用法：
    # C2：三個 universe、5 seeds、五個 variant
    PYTHONPATH=src .venv/bin/python scripts/22_run_gate_sweep.py --workers 6

    # C3：只有 top-500、10 seeds、去掉 no_graph
    PYTHONPATH=src .venv/bin/python scripts/22_run_gate_sweep.py \\
        --universes 500 --seeds 41-50 \\
        --variants self,real,relation_shuffle,topology_shuffle \\
        --out-dir artifacts/stage_c3_gate --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "21_run_gate.py"
TEST_YEARS = tuple(range(2013, 2026))


def parse_seeds(spec: str) -> tuple[int, ...]:
    """接受 `41-50` 或 `41,42,43`。"""
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return tuple(range(int(lo), int(hi) + 1))
    return tuple(int(x) for x in spec.split(",") if x.strip())


def configs(universes, seeds) -> list[dict]:
    return [{"universe": u, "test_year": y, "seed": s}
            for u in universes for y in TEST_YEARS for s in seeds]


def output_path(cfg: dict, out_dir: Path) -> Path:
    return out_dir / f"top{cfg['universe']}_y{cfg['test_year']}_seed{cfg['seed']}.json"


def is_complete(cfg: dict, out_dir: Path, n_variants: int) -> bool:
    """已完成 = 檔案存在、可解析、且所有 variant 都在。

    只檢查存在會把中途被砍的殘檔當成完成，那種錯誤要到彙整階段才會爆。
    """
    path = output_path(cfg, out_dir)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return len(data.get("results", [])) == n_variants


def run_one(cfg: dict, python: str, epochs: int, variants: str,
            out_dir: Path) -> tuple[dict, int, float]:
    started = time.perf_counter()
    cmd = [python, str(RUNNER),
           "--universe", str(cfg["universe"]),
           "--test-year", str(cfg["test_year"]),
           "--seed", str(cfg["seed"]),
           "--epochs", str(epochs),
           "--variants", variants,
           "--output-dir", str(out_dir)]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": "src"})
    if proc.returncode != 0:
        stem = f"top{cfg['universe']}_y{cfg['test_year']}_seed{cfg['seed']}"
        (out_dir / "logs" / f"{stem}.stderr.txt").write_text(
            proc.stderr or "(no stderr)", encoding="utf-8")
    return cfg, proc.returncode, time.perf_counter() - started


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universes", default="30,100,500")
    p.add_argument("--seeds", default="41-45")
    p.add_argument("--variants",
                   default="no_graph,self,real,relation_shuffle,topology_shuffle")
    p.add_argument("--out-dir", type=Path,
                   default=REPO / "artifacts" / "stage_c2_gate")
    p.add_argument("--workers", type=int, default=6,
                   help="瓶頸是逐日迴圈的 kernel launch 而非 GPU 記憶體，"
                        "可以開到核心數的一半左右。")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="只跑前 N 個未完成的工作單位，用於排程驗證")
    args = p.parse_args()

    universes = tuple(int(x) for x in args.universes.split(","))
    seeds = parse_seeds(args.seeds)
    n_variants = len([v for v in args.variants.split(",") if v.strip()])
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    planned = configs(universes, seeds)
    done = [c for c in planned if is_complete(c, out_dir, n_variants)]
    pending = [c for c in planned if c not in done]
    if args.limit:
        pending = pending[: args.limit]

    print(f"格點 {len(planned)} 個工作單位（{len(planned) * n_variants} 個 run），"
          f"已完成 {len(done)}，待跑 {len(pending)}")
    print(f"universes={universes} seeds={seeds} variants={args.variants}")
    print(f"輸出 {out_dir}")
    if args.dry_run:
        for c in pending[:10]:
            print("  ", c)
        if len(pending) > 10:
            print(f"   ...另有 {len(pending) - 10} 個")
        return 0
    if not pending:
        print("沒有待跑的工作單位。")
        return 0

    failures, durations = [], []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, c, args.python, args.epochs,
                               args.variants, out_dir): c for c in pending}
        for i, future in enumerate(as_completed(futures), 1):
            cfg, code, secs = future.result()
            durations.append(secs)
            if code != 0:
                failures.append(cfg)
            rate = (time.perf_counter() - started) / i
            print(f"[{i:3d}/{len(pending)}] {'ok ' if code == 0 else 'FAIL'} "
                  f"top{cfg['universe']:<3d} y{cfg['test_year']} seed{cfg['seed']} "
                  f"{secs:5.0f}s  剩餘約 {(len(pending) - i) * rate / 60:.0f} 分",
                  flush=True)

    # 結束檢查：用 --limit 時只檢查這一批派出去的，其餘本來就還沒輪到。
    scope = pending if args.limit else planned
    incomplete = [c for c in scope if not is_complete(c, out_dir, n_variants)]
    print(f"\n總耗時 {(time.perf_counter() - started) / 60:.1f} 分，"
          f"單位平均 {sum(durations) / len(durations):.0f}s")
    if failures:
        print(f"失敗 {len(failures)} 個，stderr 在 {out_dir / 'logs'}")
    if incomplete:
        print(f"這批有 {len(incomplete)} 個工作單位未完成：")
        for c in incomplete[:10]:
            print("  ", c)
        return 1

    if args.limit:
        remaining = len([c for c in planned
                         if not is_complete(c, out_dir, n_variants)])
        print(f"這批 {len(pending)} 個完成；整個格點還剩 {remaining} 個。")
    else:
        print(f"全部 {len(planned)} 個工作單位完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
