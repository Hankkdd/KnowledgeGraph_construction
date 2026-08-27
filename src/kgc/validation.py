"""universe 與價格資料的完整性檢查。

純邏輯放這裡，`scripts/12_validate_universe.py` 只負責讀檔與產出報告。

覆蓋率一律以「有效 `dlyret`」為準，不是「有價格列」。一檔股票可能 60 天都有
價格列，但其中幾天 `dlyret` 是空的，60 日相關係數就算不出來——只檢查價格列
會讓這種情況通過。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WARMUP_DAYS = 60


def build_presence(prices: pd.DataFrame):
    """回傳 (交易日曆, 日期索引, permno 索引, 出現矩陣, 有效 return 累積矩陣)。

    `cum[p, i]` 是 permno p 在日曆前 i+1 天中有有效 `dlyret` 的天數，
    任意視窗的有效筆數以兩點相減取得。
    """
    calendar = np.array(sorted(prices["dlycaldt"].unique()))
    permnos = np.array(sorted(prices["permno"].unique()))
    date_pos = {d: i for i, d in enumerate(calendar)}
    permno_pos = {p: i for i, p in enumerate(permnos)}

    rows = prices["permno"].map(permno_pos).to_numpy()
    cols = prices["dlycaldt"].map(date_pos).to_numpy()

    presence = np.zeros((len(permnos), len(calendar)), dtype=np.int32)
    presence[rows, cols] = 1

    valid = np.zeros((len(permnos), len(calendar)), dtype=np.int32)
    valid[rows, cols] = prices["dlyret"].notna().to_numpy().astype(np.int32)

    return calendar, date_pos, permno_pos, presence, np.cumsum(valid, axis=1)


def warmup_coverage(members, date_pos, permno_pos, cum, window: int = WARMUP_DAYS):
    """每個 (rebalance 日, 成員) 在該日往前 `window` 個交易日的有效 return 筆數。"""
    records = []
    for rebal_dt, group in members.groupby("rebal_dt", sort=True):
        i = date_pos.get(rebal_dt)
        if i is None:
            records.append({"rebal_dt": rebal_dt, "members": len(group),
                            "full": 0, "mean_cov": float("nan")})
            continue
        lo = i - window
        idx = group["permno"].map(permno_pos).dropna().astype(int).to_numpy()
        have = cum[idx, i] - (cum[idx, lo] if lo >= 0 else 0)
        records.append({
            "rebal_dt": rebal_dt,
            "members": len(idx),
            "full": int((have >= window).sum()),
            "mean_cov": float(have.mean() / window),
        })
    return pd.DataFrame(records)


def interval_overlaps(members: pd.DataFrame) -> dict:
    """檢查資格區間有無重疊，並統計中斷後再入選的次數。

    只驗重疊，不驗缺口：成員跌出 universe 一段時間後再回來是正常的，區間之間
    本來就有空隙。把那種空隙當錯誤會誤報，所以分開報告——`overlaps` 是必須為 0
    的錯誤，`re_entries` 只是描述性統計。
    """
    overlaps = 0
    re_entries = 0
    for _, g in members.sort_values(["permno", "valid_from"]).groupby("permno"):
        vf = g["valid_from"].to_numpy()
        vt = g["valid_to"].to_numpy()
        for k in range(len(g) - 1):
            if pd.isna(vt[k]):
                continue
            if vt[k] > vf[k + 1]:
                overlaps += 1
            elif vt[k] < vf[k + 1]:
                re_entries += 1
    return {"overlaps": overlaps, "re_entries": re_entries}


def member_completeness(members, prices, calendar, date_pos, permno_pos, presence):
    """入選期間內的價格缺口，以及三種下市計數。

        成員在入選期間沒有某天的價格
                │
                ├─ 該天在成員最後交易日之前  → 缺口，是問題
                └─ 該天在成員最後交易日之後  → 提前結束（下市/併購），是預期行為

    下市同時用兩個來源計算：由最後觀測日推斷，以及 CRSP 的 `dlydelflg`。
    兩者不一致代表有資料中斷被誤判成下市（或反之）。
    """
    last_obs = prices.groupby("permno")["dlycaldt"].max().to_dict()
    final_dt = calendar[-1]

    gap_days = 0
    ended_days = 0
    ended_members = set()

    spans = members.groupby("permno").agg(
        first=("valid_from", "min"), last=("valid_to", "max")
    )
    for permno, row in spans.iterrows():
        pi = permno_pos.get(permno)
        if pi is None:
            continue
        start = date_pos.get(row["first"])
        end_dt = row["last"] if pd.notna(row["last"]) else final_dt
        end = date_pos.get(end_dt, len(calendar) - 1)
        if start is None:
            continue

        missing = np.flatnonzero(presence[pi, start:end] == 0) + start
        if not len(missing):
            continue

        cutoff = date_pos[last_obs[permno]]
        gap_days += int((missing < cutoff).sum())
        after = int((missing >= cutoff).sum())
        ended_days += after
        if after:
            ended_members.add(permno)

    ever_member_ended = sum(
        1 for p in spans.index if p in last_obs and last_obs[p] < final_dt
    )
    delisted = set(
        prices.loc[prices["dlydelflg"] == "Y", "permno"].unique()
    ) & set(spans.index)

    return {
        "gap_days": gap_days,
        "ended_days": ended_days,
        "ended_in_membership": len(ended_members),
        "ever_member_ended": ever_member_ended,
        "delflg_delisted": len(delisted),
        "sources_agree": len(delisted) == ever_member_ended,
    }


def concentration(members: pd.DataFrame) -> pd.Series:
    """每個 rebalance 日前 10 大市值佔該 universe 總市值的比重。"""
    def top10_share(g):
        cap = g.sort_values("dlycap", ascending=False)["dlycap"]
        return float(cap.head(10).sum() / cap.sum())
    return members.groupby("rebal_dt").apply(top10_share, include_groups=False)
