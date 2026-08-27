"""Stage B step 2：抽取 universe 成員的日頻價格序列。

只拉 top-500 的成員聯集——它涵蓋 top-30 與 top-100，不必分開拉。
起點比 universe 起點早 5 個月，供 60 個交易日的相關窗 warm-up。

用法：
    PYTHONPATH=src .venv/bin/python scripts/11_pull_prices.py
"""

from __future__ import annotations

from kgc import universe, wrds_io

WARMUP_START = "2009-08-01"
END = "2025-12-31"


def main() -> int:
    members = wrds_io.load("members_top500", subdir="universe")
    permnos = universe.member_union(members)
    print(f"成員聯集：{len(permnos):,} 檔 permno")

    prices, manifest = wrds_io.extract(
        "daily_prices",
        universe.PRICE_SQL,
        params=(permnos, WARMUP_START, END),
        subdir="prices",
        note=f"top-500 成員聯集的日頻序列，{WARMUP_START}..{END}（含 warm-up）",
    )

    print(f"daily_prices: {manifest.rows:,} 列")
    print(f"  日期範圍  {prices['dlycaldt'].min()} .. {prices['dlycaldt'].max()}")
    print(f"  交易日數  {prices['dlycaldt'].nunique():,}")
    print(f"  相異證券  {prices['permno'].nunique():,}")
    print(f"  dlyret 缺失 {prices['dlyret'].isna().sum():,} 列")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
