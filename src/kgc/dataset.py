"""把 Stage B 的落檔整理成逐日的橫斷面 panel，以及 walk-forward 切分。

所有取用都以決策日 `t` 為界：特徵是 `t` 當日與之前 60 個交易日的報酬，
target 從 `t+1` 起算。mask 沿用 Stage B 凍結的規則——視窗內有效 `dlyret`
不足 60 筆的成員從建圖與評估中排除，整個日期保留。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import wrds_io

WINDOW = 60
HORIZON = 20
PURGE_DAYS = 20


@dataclass(frozen=True)
class Fold:
    index: int
    test_year: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def __str__(self) -> str:
        return (f"fold{self.index:02d} test={self.test_year} "
                f"train={self.train_start.date()}..{self.train_end.date()} "
                f"test={self.test_start.date()}..{self.test_end.date()}")


class Panel:
    """一個 universe 的報酬矩陣、成員資格與 mask。

    報酬存成 (交易日 × permno) 的密集矩陣，缺值為 NaN。1,136 檔 × 4,130 日
    是 18.8 MB，整批放記憶體遠比每次查 DataFrame 快。
    """

    def __init__(self, prices: pd.DataFrame, members: pd.DataFrame):
        self.calendar = np.array(sorted(prices["dlycaldt"].unique()))
        self.permnos = np.array(sorted(prices["permno"].unique()))
        self._date_pos = {d: i for i, d in enumerate(self.calendar)}
        self._permno_pos = {p: i for i, p in enumerate(self.permnos)}

        self.returns = np.full((len(self.calendar), len(self.permnos)),
                               np.nan, dtype=np.float64)
        rows = prices["dlycaldt"].map(self._date_pos).to_numpy()
        cols = prices["permno"].map(self._permno_pos).to_numpy()
        self.returns[rows, cols] = prices["dlyret"].to_numpy(dtype=np.float64)

        self._members = members
        self._membership = self._build_membership(members)
        self._siccd_by_period = self._build_siccd(members)

    @classmethod
    def load(cls, n: int) -> "Panel":
        return cls(
            wrds_io.load("daily_prices", subdir="prices"),
            wrds_io.load(f"members_top{n}", subdir="universe"),
        )

    def _build_membership(self, members: pd.DataFrame) -> dict:
        """每個 rebalance 期間對應的成員索引。"""
        by_period = {}
        for rebal_dt, g in members.groupby("rebal_dt"):
            idx = np.array([self._permno_pos[p] for p in g["permno"]
                            if p in self._permno_pos], dtype=np.int64)
            by_period[rebal_dt] = np.sort(idx)
        return by_period

    def _build_siccd(self, members: pd.DataFrame) -> dict:
        """每個 rebalance 期間、每個節點當時的 SIC。

        取自成員表——`month_end_snapshots` 抓的就是該月末當日的 `siccd`，
        所以本來就是 point-in-time。用價格檔的最後一筆會把最新的產業分類
        回填到歷史，目前的資料剛好沒有跨期變動（三個 universe 都是 0 檔），
        但那是資料的巧合，不是設計上的保證。
        """
        by_period = {}
        for rebal_dt, g in members.groupby("rebal_dt"):
            lookup = np.zeros(len(self.permnos), dtype=np.int64)
            for permno, sic in zip(g["permno"], g["siccd"]):
                pos = self._permno_pos.get(permno)
                if pos is not None:
                    lookup[pos] = int(sic)
            by_period[rebal_dt] = lookup
        return by_period

    def rebalance_dates(self) -> np.ndarray:
        return np.array(sorted(self._membership))

    def _period_of(self, date):
        dates = self.rebalance_dates()
        pos = np.searchsorted(dates, date, side="right") - 1
        return None if pos < 0 else dates[pos]

    def siccd_at(self, date, nodes: np.ndarray) -> np.ndarray:
        """`date` 當時這些節點的 SIC，不使用之後才生效的分類。"""
        period = self._period_of(date)
        if period is None:
            return np.zeros(len(nodes), dtype=np.int64)
        return self._siccd_by_period[period][nodes]

    def nodes_at(self, date) -> np.ndarray:
        """在 `date` 具備成員資格的節點索引。

        資格區間是半開的 [valid_from, valid_to)，所以取最後一個
        valid_from <= date 的 rebalance 期間。
        """
        period = self._period_of(date)
        if period is None:
            return np.zeros(0, dtype=np.int64)
        return self._membership[period]

    def window_at(self, date, nodes: np.ndarray, window: int = WINDOW):
        """回傳 (報酬視窗 (window, len(nodes)), mask)。

        視窗是 `date` 當日往前數 `window` 個交易日，含當日。
        mask 標出視窗內沒有缺值的成員——有價格列但 return 是 NaN 的那些
        算不出相關係數，必須排除。
        """
        i = self._date_pos[date]
        lo = i - window + 1
        if lo < 0:
            return np.zeros((0, len(nodes))), np.zeros(len(nodes), dtype=bool)

        block = self.returns[lo:i + 1][:, nodes]
        mask = ~np.isnan(block).any(axis=0)
        return block, mask

    def target_at(self, date, nodes: np.ndarray, horizon: int = HORIZON):
        """未來 `horizon` 個交易日的累積報酬，橫斷面去均值。

        回傳 (target, 有效旗標)。去均值讓 target 成為相對表現，
        與 Rank IC 的橫斷面性質一致。
        """
        i = self._date_pos[date]
        hi = i + horizon
        if hi >= len(self.calendar):
            return np.zeros(len(nodes)), np.zeros(len(nodes), dtype=bool)

        block = self.returns[i + 1:hi + 1][:, nodes]
        valid = ~np.isnan(block).any(axis=0)
        cumulative = np.zeros(len(nodes))
        if valid.any():
            cumulative[valid] = np.prod(1.0 + block[:, valid], axis=0) - 1.0
            cumulative[valid] -= cumulative[valid].mean()
        return cumulative, valid

    def usable_mask(self, date, nodes: np.ndarray,
                    window: int = WINDOW, horizon: int = HORIZON) -> np.ndarray:
        """同時滿足特徵視窗與 target 都完整的成員。

        mask 只由資料決定，與圖無關——這是它能在 real 與所有 control 之間
        保持一致的原因，也是 Stage C 必須實際驗證的那個假設。
        """
        _, feature_ok = self.window_at(date, nodes, window)
        _, target_ok = self.target_at(date, nodes, horizon)
        return feature_ok & target_ok


def walk_forward(calendar: np.ndarray, first_test_year: int = 2013,
                 train_years: int = 3, purge_days: int = PURGE_DAYS) -> list[Fold]:
    """3 年 train / 1 年 test，每年滾動。

    train 結束與 test 開始之間空 `purge_days` 個交易日：最後一個訓練日的
    target 會延伸 20 個交易日，不留 gap 就會跟測試期重疊。
    """
    years = pd.DatetimeIndex(calendar).year
    last_year = int(years.max())

    folds = []
    for i, test_year in enumerate(range(first_test_year, last_year + 1)):
        train_mask = (years >= test_year - train_years) & (years < test_year)
        test_mask = years == test_year
        if not train_mask.any() or not test_mask.any():
            continue

        train_days = calendar[train_mask]
        test_days = calendar[test_mask]
        keep = len(train_days) - purge_days
        if keep <= 0:
            continue

        folds.append(Fold(
            index=i,
            test_year=test_year,
            train_start=pd.Timestamp(train_days[0]),
            train_end=pd.Timestamp(train_days[keep - 1]),
            test_start=pd.Timestamp(test_days[0]),
            test_end=pd.Timestamp(test_days[-1]),
        ))
    return folds
