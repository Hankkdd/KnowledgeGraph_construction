"""圖建構、shuffle controls 與 walk-forward 切分的單元測試。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kgc import dataset, graphs


@pytest.fixture
def rng():
    return np.random.default_rng(41)


@pytest.fixture
def corr_graph():
    gen = np.random.default_rng(7)
    # 12 檔、80 天，前 6 檔彼此相關較高，讓 top-k 有結構可抓。
    common = gen.normal(size=(80, 1))
    block_a = common + 0.3 * gen.normal(size=(80, 6))
    block_b = gen.normal(size=(80, 6))
    window = np.hstack([block_a, block_b])
    return graphs.corr_edges(window, k=3)


class TestCorrEdges:
    def test_in_degree_is_exactly_k(self, corr_graph):
        assert (corr_graph.in_degree() == 3).all()

    def test_in_degree_independent_of_universe_size(self):
        gen = np.random.default_rng(3)
        for n in (10, 50, 200):
            g = graphs.corr_edges(gen.normal(size=(80, n)), k=5)
            assert (g.in_degree() == 5).all()

    def test_no_self_loops(self, corr_graph):
        assert not (corr_graph.src == corr_graph.dst).any()

    def test_relation_encodes_sign_of_correlation(self):
        # 兩檔完全反向的序列，彼此必為 neg 邊。
        base = np.random.default_rng(1).normal(size=(80, 1))
        window = np.hstack([base, -base, np.random.default_rng(2).normal(size=(80, 3))])
        g = graphs.corr_edges(window, k=1)
        edge = (g.dst == 0)
        assert g.src[edge][0] == 1
        assert g.relation_names[g.relation[edge][0]] == "neg"


class TestSectorEdges:
    def test_connects_same_major_group_only(self):
        # 3571 與 3572 同屬 major group 35；4920 不同組。
        g = graphs.sector_edges(np.array([3571, 3572, 4920]), digits=2)
        pairs = set(zip(g.src.tolist(), g.dst.tolist()))
        assert pairs == {(0, 1), (1, 0)}

    def test_singleton_group_is_isolated(self):
        g = graphs.sector_edges(np.array([3571, 3572, 4920]), digits=2)
        assert g.isolated() == 1


class TestShuffleTopology:
    def test_preserves_both_degree_sequences_and_edge_count(self, corr_graph, rng):
        shuffled = graphs.shuffle_topology(corr_graph, rng)
        assert shuffled.n_edges == corr_graph.n_edges
        assert np.array_equal(np.sort(shuffled.in_degree()),
                              np.sort(corr_graph.in_degree()))
        assert np.array_equal(np.sort(shuffled.out_degree()),
                              np.sort(corr_graph.out_degree()))

    def test_preserves_relation_frequency(self, corr_graph, rng):
        shuffled = graphs.shuffle_topology(corr_graph, rng)
        assert np.array_equal(np.bincount(shuffled.relation, minlength=2),
                              np.bincount(corr_graph.relation, minlength=2))

    def test_introduces_no_self_loops(self, corr_graph, rng):
        shuffled = graphs.shuffle_topology(corr_graph, rng)
        assert not (shuffled.src == shuffled.dst).any()

    def test_actually_changes_topology(self, corr_graph, rng):
        shuffled = graphs.shuffle_topology(corr_graph, rng)
        assert not np.array_equal(shuffled.src, corr_graph.src)


class TestShuffleRelations:
    def test_keeps_topology_and_weights_untouched(self, corr_graph, rng):
        shuffled = graphs.shuffle_relations(corr_graph, rng)
        assert np.array_equal(shuffled.src, corr_graph.src)
        assert np.array_equal(shuffled.dst, corr_graph.dst)
        assert np.array_equal(shuffled.weight, corr_graph.weight)

    def test_preserves_label_frequency(self, corr_graph, rng):
        shuffled = graphs.shuffle_relations(corr_graph, rng)
        assert np.array_equal(np.bincount(shuffled.relation, minlength=2),
                              np.bincount(corr_graph.relation, minlength=2))

    def test_refuses_single_relation_graph(self, rng):
        """單一 relation 的圖必須明確拒絕，不能假造 typed control。"""
        sector = graphs.sector_edges(np.array([3571, 3572]), digits=2)
        with pytest.raises(ValueError, match="不適用"):
            graphs.shuffle_relations(sector, rng)


def make_panel(days: int = 400, n: int = 8, periods: int = 3) -> dataset.Panel:
    calendar = pd.bdate_range("2020-01-01", periods=days)
    gen = np.random.default_rng(11)
    sic = np.array([3571] * (n // 2) + [4920] * (n - n // 2))

    prices = pd.DataFrame({
        "dlycaldt": np.repeat(calendar, n),
        "permno": np.tile(np.arange(1, n + 1), days),
        "dlyret": gen.normal(scale=0.01, size=days * n),
        "siccd": np.tile(sic, days),
    })

    # 多個 rebalance 期間，才測得出 SIC 是不是按期取值。
    rebal_dates = [calendar[i] for i in np.linspace(0, days - 1, periods, dtype=int)]
    rows = []
    for k, rebal_dt in enumerate(rebal_dates):
        nxt = rebal_dates[k + 1] if k + 1 < len(rebal_dates) else pd.NaT
        for permno, s in zip(range(1, n + 1), sic):
            rows.append({"rebal_dt": rebal_dt, "permno": permno, "n_target": n,
                         "siccd": int(s), "valid_from": rebal_dt, "valid_to": nxt})
    return dataset.Panel(prices, pd.DataFrame(rows))


class TestPanelLeakage:
    def test_feature_window_stops_at_decision_date(self):
        """視窗只含決策日當日與之前——放在之後的異常值不得影響結果。"""
        panel = make_panel()
        date = panel.calendar[200]
        nodes = panel.nodes_at(date)
        before, _ = panel.window_at(date, nodes)

        i = panel._date_pos[date]
        panel.returns[i + 1:i + 30, :] = 999.0     # 未來塞入極端值
        after, _ = panel.window_at(date, nodes)

        assert np.array_equal(before, after)

    def test_target_starts_after_decision_date(self):
        panel = make_panel()
        date = panel.calendar[200]
        nodes = panel.nodes_at(date)
        i = panel._date_pos[date]

        baseline, _ = panel.target_at(date, nodes)
        panel.returns[i, :] = 999.0                # 改決策日當日
        assert np.allclose(baseline, panel.target_at(date, nodes)[0])

    def test_target_is_cross_sectionally_demeaned(self):
        panel = make_panel()
        date = panel.calendar[200]
        nodes = panel.nodes_at(date)
        target, valid = panel.target_at(date, nodes)
        assert abs(target[valid].mean()) < 1e-12


class TestWalkForward:
    def test_purge_gap_separates_train_and_test(self):
        calendar = pd.bdate_range("2010-01-01", "2025-12-31").to_numpy()
        folds = dataset.walk_forward(calendar, purge_days=20)
        assert folds, "應該至少產生一個 fold"
        for fold in folds:
            gap = np.sum((calendar > fold.train_end) & (calendar < fold.test_start))
            assert gap >= 20, f"{fold} 的 purge gap 只有 {gap} 天"

    def test_train_precedes_test_and_windows_do_not_overlap(self):
        calendar = pd.bdate_range("2010-01-01", "2025-12-31").to_numpy()
        for fold in dataset.walk_forward(calendar):
            assert fold.train_start < fold.train_end < fold.test_start <= fold.test_end

    def test_thirteen_folds_over_2010_2025(self):
        calendar = pd.bdate_range("2010-01-01", "2025-12-31").to_numpy()
        folds = dataset.walk_forward(calendar)
        assert [f.test_year for f in folds] == list(range(2013, 2026))


class TestShuffleProducesSimpleGraph:
    """real 圖沒有重複邊，shuffle 後也不能有——sum aggregation 會重複計入。"""

    @staticmethod
    def duplicates(g) -> int:
        keys = g.src.astype(np.int64) * g.n_nodes + g.dst.astype(np.int64)
        return int(g.n_edges - len(np.unique(keys)))

    def test_corr_shuffle_has_no_duplicates(self, corr_graph):
        for seed in range(41, 51):
            shuffled = graphs.shuffle_topology(
                corr_graph, np.random.default_rng(seed))
            assert self.duplicates(shuffled) == 0, f"seed {seed}"

    def test_sector_shuffle_has_no_duplicates(self):
        """sector graph 較密，單純重排會留下約 5% 重複邊。"""
        gen = np.random.default_rng(5)
        siccd = gen.choice([3571, 3572, 4920, 6021, 2834], size=60)
        sector = graphs.sector_edges(siccd, digits=2)
        assert self.duplicates(sector) == 0

        for seed in range(41, 51):
            shuffled = graphs.shuffle_topology(
                sector, np.random.default_rng(seed))
            assert self.duplicates(shuffled) == 0, f"seed {seed}"
            assert np.array_equal(np.sort(shuffled.in_degree()),
                                  np.sort(sector.in_degree()))
            assert np.array_equal(np.sort(shuffled.out_degree()),
                                  np.sort(sector.out_degree()))

    def test_no_self_loops_across_seeds(self, corr_graph):
        for seed in range(41, 51):
            shuffled = graphs.shuffle_topology(
                corr_graph, np.random.default_rng(seed))
            assert not (shuffled.src == shuffled.dst).any(), f"seed {seed}"


class TestPointInTimeSector:
    def test_siccd_comes_from_the_membership_period(self):
        """SIC 取自成員表的當期值，不是價格檔的最後一筆。"""
        panel = make_panel()
        date = panel.calendar[200]
        nodes = panel.nodes_at(date)
        sic = panel.siccd_at(date, nodes)
        assert len(sic) == len(nodes)
        assert set(sic.tolist()) <= {3571, 4920}

    def test_future_sic_change_does_not_leak_backwards(self):
        """把之後期別的 SIC 改掉，先前日期的分類不得跟著變。"""
        panel = make_panel()
        early = panel.calendar[100]
        before = panel.siccd_at(early, panel.nodes_at(early)).copy()

        period = panel._period_of(early)
        panel._siccd_by_period = dict(panel._siccd_by_period)
        for other in panel.rebalance_dates():
            if other != period:
                panel._siccd_by_period[other] = np.full_like(
                    panel._siccd_by_period[other], 9999)

        after = panel.siccd_at(early, panel.nodes_at(early))
        assert np.array_equal(before, after)
