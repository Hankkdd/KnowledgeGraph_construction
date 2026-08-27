"""universe 選取與驗證邏輯的單元測試。

用小型合成資料，不連 WRDS。
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from kgc import universe, validation, wrds_io


def make_snapshots():
    """兩個 rebalance 日、五檔證券。第二期 permno 5 擠掉 permno 1。"""
    rows = []
    caps = {
        dt.date(2020, 1, 31): {1: 500, 2: 400, 3: 300, 4: 200, 5: 100},
        dt.date(2020, 2, 28): {1: 90, 2: 400, 3: 300, 4: 200, 5: 550},
    }
    for d, per in caps.items():
        for permno, cap in per.items():
            rows.append({"rebal_dt": pd.Timestamp(d), "permno": permno,
                         "dlycap": float(cap), "dlyprc": 10.0,
                         "ticker": f"T{permno}", "siccd": 1000,
                         "primaryexch": "N"})
    return pd.DataFrame(rows)


class TestSelectTopN:
    def test_exact_n_per_rebalance_date(self):
        members = universe.select_top_n(make_snapshots(), 3)
        assert (members.groupby("rebal_dt").size() == 3).all()

    def test_nesting_holds(self):
        snaps = make_snapshots()
        small = universe.select_top_n(snaps, 2)
        large = universe.select_top_n(snaps, 4)
        for d, g in small.groupby("rebal_dt"):
            big = set(large.loc[large["rebal_dt"] == d, "permno"])
            assert set(g["permno"]) <= big

    def test_ranked_by_descending_cap(self):
        members = universe.select_top_n(make_snapshots(), 3)
        first = members[members["rebal_dt"] == pd.Timestamp("2020-01-31")]
        assert list(first.sort_values("rank")["permno"]) == [1, 2, 3]

    def test_intervals_are_half_open_and_contiguous(self):
        members = universe.select_top_n(make_snapshots(), 3)
        held = members[members["permno"] == 2].sort_values("valid_from")
        assert len(held) == 2
        # 前一期的 valid_to 正好接上後一期的 valid_from，不重疊也不留缺口。
        assert held.iloc[0]["valid_to"] == held.iloc[1]["valid_from"]
        # 最後一期沒有下一個 rebalance 日，留 NaT 表示開放區間。
        assert pd.isna(held.iloc[1]["valid_to"])

    def test_ties_broken_by_permno_for_determinism(self):
        snaps = make_snapshots()
        snaps.loc[snaps["permno"].isin([3, 4]), "dlycap"] = 300.0
        a = universe.select_top_n(snaps, 4)["permno"].tolist()
        b = universe.select_top_n(snaps.sample(frac=1, random_state=7), 4)
        assert a == b["permno"].tolist()


class TestMonthlyTurnover:
    def test_counts_entrants_only(self):
        members = universe.select_top_n(make_snapshots(), 3)
        turnover = universe.monthly_turnover(members)
        # 第一期沒有前一期可比，只有一列。permno 5 進、permno 1 出。
        assert len(turnover) == 1
        assert turnover.iloc[0]["entered"] == 1
        assert turnover.iloc[0]["turnover"] == pytest.approx(1 / 3)


class TestIntervalOverlaps:
    def test_clean_universe_has_no_overlap(self):
        members = universe.select_top_n(make_snapshots(), 3)
        assert validation.interval_overlaps(members)["overlaps"] == 0

    def test_overlap_is_detected(self):
        members = universe.select_top_n(make_snapshots(), 3)
        # 把持有兩期的成員的第一段結束日往後推，製造重疊。
        idx = members[(members["permno"] == 2)].sort_values("valid_from").index[0]
        members.loc[idx, "valid_to"] = pd.Timestamp("2020-03-31")
        assert validation.interval_overlaps(members)["overlaps"] == 1

    def test_re_entry_counted_separately_not_as_error(self):
        members = universe.select_top_n(make_snapshots(), 3)
        idx = members[(members["permno"] == 2)].sort_values("valid_from").index[0]
        # 提早結束 → 兩段之間出現空隙，那是離開後再入選，不是錯誤。
        members.loc[idx, "valid_to"] = pd.Timestamp("2020-02-10")
        result = validation.interval_overlaps(members)
        assert result["overlaps"] == 0
        assert result["re_entries"] == 1


class TestCoercion:
    def test_dates_with_nulls_become_datetime64(self):
        df = pd.DataFrame({"d": [dt.date(2020, 1, 1), None, dt.date(2020, 1, 3)]})
        out = wrds_io.coerce_dates(df)
        assert pd.api.types.is_datetime64_any_dtype(out["d"])
        # 混著 NaN 的 object 欄位會讓 max() 直接炸，這是修這個的原因。
        assert out["d"].max() == pd.Timestamp("2020-01-03")

    def test_decimal_becomes_float_and_supports_ufuncs(self):
        df = pd.DataFrame({"x": [Decimal("1.5"), Decimal("2.5"), None]})
        out = wrds_io.coerce_numerics(df)
        assert out["x"].dtype == np.float64
        assert np.isfinite(np.log(out["x"].dropna().to_numpy())).all()

    def test_strings_are_left_alone(self):
        df = pd.DataFrame({"s": ["N", "Q", None]})
        out = wrds_io.coerce_numerics(wrds_io.coerce_dates(df.copy()))
        assert out["s"].tolist()[:2] == ["N", "Q"]


class TestWarmupCoverage:
    def test_counts_valid_returns_not_price_rows(self):
        """有價格列但 dlyret 是空的，不能算進覆蓋率。"""
        dates = pd.bdate_range("2020-01-01", periods=10)
        prices = pd.DataFrame({
            "permno": [1] * 10,
            "dlycaldt": dates,
            "dlyret": [0.01] * 7 + [None] * 3,   # 10 列價格，只有 7 個有效 return
            "dlydelflg": ["N"] * 10,
        })
        _, date_pos, permno_pos, _, cum = validation.build_presence(prices)
        members = pd.DataFrame({
            "rebal_dt": [dates[-1]], "permno": [1], "n_target": [1],
        })
        cov = validation.warmup_coverage(members, date_pos, permno_pos, cum, window=9)
        assert cov.iloc[0]["full"] == 0
        assert cov.iloc[0]["members"] == 1
