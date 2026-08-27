"""Stage B step 3：驗證 universe 與價格資料，輸出報告。

不連 WRDS。只讀回 step 1/2 的落檔（`wrds_io.load()` 會驗 SHA-256），
檢查邏輯在 `kgc.validation`，這裡只負責讀檔與產出報告。

用法：
    PYTHONPATH=src .venv/bin/python scripts/12_validate_universe.py
"""

from __future__ import annotations

import json

from kgc import universe, validation, wrds_io

SIZES = (30, 100, 500)
OUT_DIR = wrds_io.REPO_ROOT / "artifacts" / "stage_b_validation"
W = validation.WARMUP_DAYS


def main() -> int:
    prices = wrds_io.load("daily_prices", subdir="prices")
    calendar, date_pos, permno_pos, presence, cum = validation.build_presence(prices)

    summary = {
        "prices": {
            "rows": int(len(prices)),
            "trading_days": int(len(calendar)),
            "first_day": str(calendar[0])[:10],
            "last_day": str(calendar[-1])[:10],
            "distinct_permno": int(len(permno_pos)),
            "dlyret_missing": int(prices["dlyret"].isna().sum()),
        },
        "warmup_policy": (
            f"asset-level mask：成員在 rebalance 日往前 {W} 個交易日內"
            f"有效 dlyret 不足 {W} 筆時，該 (日期, 成員) 從建圖與評估中排除，"
            "整個 rebalance 日仍保留。mask 在 real 與所有 control 之間完全一致。"
        ),
        "universes": {},
    }

    lines = [
        "# Stage B 驗證報告", "",
        f"價格：{len(prices):,} 列，{len(calendar):,} 個交易日"
        f"（{str(calendar[0])[:10]} .. {str(calendar[-1])[:10]}），"
        f"{len(permno_pos):,} 檔證券。",
        f"`dlyret` 缺失 {prices['dlyret'].isna().sum():,} 列"
        f"（{prices['dlyret'].isna().mean():.3%}），多為上市首日無前收盤價。",
        "",
        "## warm-up 處理方式（已凍結）", "",
        summary["warmup_policy"], "",
        f"替代方案是剔除整個 rebalance 日，但那會為了修正 0.3% 的 member-date "
        f"而丟掉約四分之三的日期，代價不成比例。", "",
    ]

    for n in SIZES:
        members = wrds_io.load(f"members_top{n}", subdir="universe")
        counts = members.groupby("rebal_dt").size()
        turnover = universe.monthly_turnover(members)
        cov = validation.warmup_coverage(members, date_pos, permno_pos, cum)
        integ = validation.interval_overlaps(members)
        comp = validation.member_completeness(
            members, prices, calendar, date_pos, permno_pos, presence)
        conc = validation.concentration(members)

        distinct = int(members["permno"].nunique())
        entry = {
            "rebalance_dates": int(counts.size),
            "exact_n_all_dates": bool((counts == n).all()),
            "distinct_members": distinct,
            "interval_overlaps": integ["overlaps"],
            "re_entries": integ["re_entries"],
            "turnover_mean": float(turnover["turnover"].mean()),
            "turnover_max": float(turnover["turnover"].max()),
            "warmup_full_ratio": float(cov["full"].sum() / cov["members"].sum()),
            "warmup_incomplete_dates": int((cov["full"] < cov["members"]).sum()),
            "masked_member_dates": int((cov["members"] - cov["full"]).sum()),
            "gap_days": comp["gap_days"],
            "ended_in_membership": comp["ended_in_membership"],
            "ever_member_ended": comp["ever_member_ended"],
            "ever_member_ended_ratio": comp["ever_member_ended"] / distinct,
            "delflg_delisted": comp["delflg_delisted"],
            "delisting_sources_agree": comp["sources_agree"],
            "top10_share_mean": float(conc.mean()),
            "top10_share_last": float(conc.iloc[-1]),
        }
        summary["universes"][f"top{n}"] = entry

        lines += [
            f"## top-{n}", "",
            "| 檢查 | 結果 |",
            "|---|---|",
            f"| rebalance 日數 | {entry['rebalance_dates']} |",
            f"| 每日成員數皆為 {n} | {'是' if entry['exact_n_all_dates'] else '否'} |",
            f"| 相異成員 | {distinct:,} |",
            f"| 資格區間重疊 | {entry['interval_overlaps']} |",
            f"| 中斷後再入選 | {entry['re_entries']:,} 次 |",
            f"| 月換手率 平均 / 最高 | {entry['turnover_mean']:.2%} / {entry['turnover_max']:.2%} |",
            f"| {W} 個有效 return 完整比率 | {entry['warmup_full_ratio']:.3%} |",
            f"| warm-up 未滿的 rebalance 日 | {entry['warmup_incomplete_dates']} |",
            f"| 需 mask 的 member-date | {entry['masked_member_dates']:,} |",
            f"| 入選期間內的價格缺口 | {entry['gap_days']:,} 天 |",
            f"| 成員資格期間內停止交易 | {entry['ended_in_membership']:,} 檔 |",
            f"| 曾入選、樣本結束前停止交易 | {entry['ever_member_ended']:,}"
            f"（{entry['ever_member_ended_ratio']:.1%}）|",
            f"| `dlydelflg='Y'` 的成員 | {entry['delflg_delisted']:,}"
            f"（與上一列{'一致' if entry['delisting_sources_agree'] else '不一致'}）|",
            f"| 前 10 大市值佔比 平均 / 期末 | {entry['top10_share_mean']:.1%} / {entry['top10_share_last']:.1%} |",
            "",
        ]

        incomplete = cov[cov["full"] < cov["members"]]
        if len(incomplete):
            lines += [
                f"需 mask 的 rebalance 日（{len(incomplete)} 個）：", "",
                "| rebalance 日 | 成員 | 完整 | 遮蔽 | 平均覆蓋 |",
                "|---|---:|---:|---:|---:|",
            ]
            for _, r in incomplete.iterrows():
                lines.append(
                    f"| {str(r['rebal_dt'])[:10]} | {r['members']} | {r['full']} | "
                    f"{r['members'] - r['full']} | {r['mean_cov']:.1%} |")
            lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print(f"寫出 {OUT_DIR}/report.md 與 summary.json")
    for n in SIZES:
        e = summary["universes"][f"top{n}"]
        print(f"  top-{n:<3d} 成員 {e['distinct_members']:>5,}  "
              f"mask {e['masked_member_dates']:>4,}  "
              f"下市 {e['ever_member_ended']:>3,}  "
              f"來源一致 {e['delisting_sources_agree']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
