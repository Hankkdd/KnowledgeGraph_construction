"""Stage B step 1：建立 top-30 / top-100 / top-500 的 point-in-time 成員表。

用法：
    PYTHONPATH=src .venv/bin/python scripts/10_build_universe.py
"""

from __future__ import annotations

import sys

from kgc import universe, wrds_io

START = "2010-01-01"
END = "2025-12-31"
SIZES = (30, 100, 500)


def main() -> int:
    snapshots, snap_manifest = wrds_io.extract(
        "month_end_snapshots",
        universe.MONTH_END_SQL,
        params=(START, END),
        subdir="universe",
        note=f"{START}..{END} 每月最後交易日的合格普通股與市值",
    )
    n_dates = snapshots["rebal_dt"].nunique()
    print(f"month_end_snapshots: {snap_manifest.rows:,} 列，"
          f"{n_dates} 個 rebalance 日，"
          f"{snapshots['permno'].nunique():,} 檔相異證券")

    if n_dates == 0:
        print("沒有取到任何 rebalance 日，中止。", file=sys.stderr)
        return 1

    for n in SIZES:
        members = universe.select_top_n(snapshots, n)

        counts = members.groupby("rebal_dt").size()
        short = counts[counts != n]
        turnover = universe.monthly_turnover(members)

        wrds_io.save_derived(
            f"members_top{n}",
            members,
            derived_from=["month_end_snapshots"],
            subdir="universe",
            note=f"每月末依 dlycap 取前 {n} 名，平手以 permno 決勝",
        )

        print(f"\nmembers_top{n}:")
        print(f"  rebalance 日數      {counts.size}")
        print(f"  相異成員            {members['permno'].nunique():,}")
        print(f"  月換手率 平均/最高   {turnover['turnover'].mean():.2%} / "
              f"{turnover['turnover'].max():.2%}")
        if len(short):
            print(f"  成員數不足 {n} 的日期：{len(short)} 個 -> "
                  f"{list(short.index[:5])}")
        else:
            print(f"  每個 rebalance 日都剛好 {n} 名成員")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
