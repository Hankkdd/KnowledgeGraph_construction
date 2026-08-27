"""從 CRSP 建立 point-in-time 的市值排序 universe。

不依賴任何指數成分名單。每個 rebalance 日只用當日與之前的資料，
成員在期間內下市時仍留在當期 universe——`dlyret` 已含下市報酬。

NYCU 的訂閱沒有 `crsp.dsp500list` 與 CCM 連結表，成分名單這條路是封死的；
細節見 `docs/2026-08-26_WRDS資料可得性盤點.md`。
"""

from __future__ import annotations

import pandas as pd

# 標準學術普通股篩選。這裡是唯一定義處，其他地方一律引用，不要各寫一份。
ELIGIBILITY = """
    d.sharetype = 'NS'
    and d.securitytype = 'EQTY'
    and d.securitysubtype = 'COM'
    and d.usincflg = 'Y'
    and d.primaryexch in ('N', 'A', 'Q')
    and d.dlycap is not null
    and d.dlycap > 0
"""

# 一次取回所有月末的合格證券。192 個月末約 765,164 列，一次拉完，
# 不要每個月末各發一次查詢。
MONTH_END_SQL = f"""
with month_end as (
    select max(dlycaldt) as rebal_dt
    from crsp.dsf_v2
    where dlycaldt between %s and %s
    group by date_trunc('month', dlycaldt)
)
select
    d.dlycaldt as rebal_dt,
    d.permno,
    d.dlycap,
    d.dlyprc,
    d.ticker,
    d.siccd,
    d.primaryexch
from crsp.dsf_v2 d
join month_end m on m.rebal_dt = d.dlycaldt
where {ELIGIBILITY}
order by d.dlycaldt, d.dlycap desc, d.permno
"""

PRICE_SQL = """
select
    d.permno,
    d.dlycaldt,
    d.dlyprc,
    d.dlyret,
    d.dlyretx,
    d.dlycap,
    d.dlyvol,
    d.dlyprcvol,
    d.dlyopen,
    d.dlyhigh,
    d.dlylow,
    d.dlyclose,
    d.dlycumfacpr,
    d.dlydelflg,
    d.ticker,
    d.siccd,
    d.primaryexch
from crsp.dsf_v2 d
where d.permno = any(%s)
  and d.dlycaldt between %s and %s
order by d.permno, d.dlycaldt
"""


def select_top_n(snapshots: pd.DataFrame, n: int) -> pd.DataFrame:
    """每個 rebalance 日依 dlycap 取前 n 名。

    回傳含 `valid_from` / `valid_to` 的成員表。區間是半開的
    `[valid_from, valid_to)`：資格在 rebalance 日收盤判定，持續到下一個
    rebalance 日為止，因此相鄰期間既不重疊也不留缺口。

    平手時以 permno 決勝，確保重跑結果一致。
    """
    ranked = (
        snapshots.sort_values(["rebal_dt", "dlycap", "permno"],
                              ascending=[True, False, True])
        .groupby("rebal_dt", sort=True)
        .head(n)
        .copy()
    )
    ranked["rank"] = ranked.groupby("rebal_dt").cumcount() + 1

    rebal_dates = sorted(ranked["rebal_dt"].unique())
    next_date = dict(zip(rebal_dates[:-1], rebal_dates[1:]))

    ranked["valid_from"] = ranked["rebal_dt"]
    ranked["valid_to"] = ranked["rebal_dt"].map(next_date)
    # 最後一期沒有下一個 rebalance 日，留 NaT 表示開放區間。

    ranked["n_target"] = n
    return ranked.reset_index(drop=True)[
        ["n_target", "rebal_dt", "rank", "permno", "dlycap", "dlyprc",
         "ticker", "siccd", "primaryexch", "valid_from", "valid_to"]
    ]


def member_union(members: pd.DataFrame) -> list[int]:
    """成員表裡出現過的所有 permno，供價格抽取使用。"""
    return sorted(int(p) for p in members["permno"].unique())


def monthly_turnover(members: pd.DataFrame) -> pd.DataFrame:
    """相鄰 rebalance 日之間的成員更替比率。

    turnover = 新進成員數 / N。第一期沒有前一期，不計算。
    """
    by_date = {
        d: set(g["permno"])
        for d, g in members.groupby("rebal_dt", sort=True)
    }
    dates = sorted(by_date)
    n = int(members["n_target"].iloc[0])

    rows = []
    for prev, curr in zip(dates[:-1], dates[1:]):
        entered = by_date[curr] - by_date[prev]
        rows.append({
            "rebal_dt": curr,
            "entered": len(entered),
            "turnover": len(entered) / n,
        })
    return pd.DataFrame(rows)
