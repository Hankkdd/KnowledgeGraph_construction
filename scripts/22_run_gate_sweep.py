"""Stage C2：派工並平行執行完整的 G_corr gate 格點。

每個 (universe, test_year, seed) 是一個獨立 subprocess，內含五個 variant。
這個粒度讓失敗只損失一個工作單位，而 Panel 每個 subprocess 只載入一次。

排程器本身不做任何科學判斷，只負責：完成的跳過、失敗的保留 stderr、
結束時檢查有沒有缺漏或重複。彙整與判定在 `23_summarise_gate.py`。

用法：
    PYTHONPATH=src .venv/bin/python scripts/22_run_gate_sweep.py --workers 6
    PYTHONPATH=src .venv/bin/python scripts/22_run_gate_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "artifacts" / "stage_c2_gate"
LOG_DIR = OUT_DIR / "logs"
RUNNER = REPO / "scripts" / "21_run_gate.py"

UNIVERSES = (30, 100, 500)
TEST_YEARS = tuple(range(2013, 2026))
SEEDS = (41, 42, 43, 44, 45)
EPOCHS = 3


def configs() -> list[dict]:
    return [{"universe": u, "test_year": y, "seed": s}
            for u in UNIVERSES for y in TEST_YEARS for s in SEEDS]


def output_path(cfg: dict) -> Path:
    return OUT_DIR / f"top{cfg['universe']}_y{cfg['test_year']}_seed{cfg['seed']}.json"


def is_complete(cfg: dict) -> bool:
    """已完成 = 檔案存在、可解析、且五個 variant 都在。

    只檢查存在會把中途被砍的殘檔當成完成，那種錯誤在彙整時才會爆。
    """
    path = output_path(cfg)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return len(data.get("results", [])) == 5


def run_one(cfg: dict, python: str, epochs: int) -> tuple[dict, int, float]:
    started = time.perf_counter()
    cmd = [python, str(RUNNER),
           "--universe", str(cfg["universe"]),
           "--test-year", str(cfg["test_year"]),
           "--seed", str(cfg["seed"]),
           "--epochs", str(epochs),
           "--output-dir", str(OUT_DIR)]
    env_note = LOG_DIR / f"top{cfg['universe']}_y{cfg['test_year']}_seed{cfg['seed']}"
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONPATH": "src"})
    if proc.returncode != 0:
        env_note.with_suffix(".stderr.txt").write_text(
            proc.stderr or "(no stderr)", encoding="utf-8")
    return cfg, proc.returncode, time.perf_counter() - started


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=6,
                   help="同時執行的 subprocess 數。瓶頸是逐日迴圈的 kernel launch，"
                        "不是 GPU 記憶體，所以可以開到核心數的一半左右。")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--python", default=str(REPO / ".venv" / "bin" / "python"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="只跑前 N 個未完成的 config，用於排程驗證")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    planned = configs()
    pending = [c for c in planned if not is_complete(c)]
    if args.limit:
        pending = pending[: args.limit]

    print(f"格點 {len(planned)} 個工作單位（{len(planned) * 5} 個 run），"
          f"已完成 {len(planned) - len([c for c in planned if not is_complete(c)])}，"
          f"待跑 {len(pending)}")
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
        futures = {pool.submit(run_one, c, args.python, args.epochs): c
                   for c in pending}
        for i, future in enumerate(as_completed(futures), 1):
            cfg, code, secs = future.result()
            durations.append(secs)
            status = "ok " if code == 0 else "FAIL"
            if code != 0:
                failures.append(cfg)
            elapsed = time.perf_counter() - started
            rate = elapsed / i
            print(f"[{i:3d}/{len(pending)}] {status} "
                  f"top{cfg['universe']:<3d} y{cfg['test_year']} seed{cfg['seed']} "
                  f"{secs:5.0f}s  剩餘約 {(len(pending) - i) * rate / 60:.0f} 分",
                  flush=True)

    # 結束檢查：缺漏與殘檔都要明說，不能讓彙整階段才發現。
    # 用 --limit 時只檢查這一批派出去的，其餘本來就還沒輪到。
    scope = pending if args.limit else planned
    incomplete = [c for c in scope if not is_complete(c)]
    print(f"\n總耗時 {(time.perf_counter() - started) / 60:.1f} 分，"
          f"單位平均 {sum(durations) / len(durations):.0f}s")
    if failures:
        print(f"失敗 {len(failures)} 個，stderr 在 {LOG_DIR}")
    if incomplete:
        print(f"這批有 {len(incomplete)} 個工作單位未完成：")
        for c in incomplete[:10]:
            print("  ", c)
        return 1

    if args.limit:
        remaining = len([c for c in planned if not is_complete(c)])
        print(f"這批 {len(pending)} 個工作單位完成；整個格點還剩 {remaining} 個。")
    else:
        print(f"全部 {len(planned)} 個工作單位完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
