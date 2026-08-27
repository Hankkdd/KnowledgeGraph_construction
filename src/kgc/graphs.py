"""圖的建構與 shuffle controls。

每個 variant 都由同一份 `real` 邊集衍生，因此邊數、度數序列與 relation 頻率
在 real 與 control 之間必然一致——差異只在語義。這是協定 §2 要求的公平性條件，
自己各建一份圖的話會混進容量差異。

圖一律以「給定日期即時建構」的方式使用，不預先整批建好再回填：
測試期的相關圖若用到未來資料就是 leakage，而那種錯誤不會有任何徵兆。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# 邊的方向是 src -> dst，訊息從 src 流向 dst。
# corr graph 每個節點固定取 k 個來源，因此 in-degree 恆為 k，不隨 universe 大小變動。


@dataclass(frozen=True)
class Graph:
    n_nodes: int
    src: np.ndarray
    dst: np.ndarray
    relation: np.ndarray
    weight: np.ndarray
    relation_names: tuple[str, ...]

    @property
    def n_edges(self) -> int:
        return int(len(self.src))

    @property
    def density(self) -> float:
        if self.n_nodes < 2:
            return 0.0
        return self.n_edges / (self.n_nodes * (self.n_nodes - 1))

    def in_degree(self) -> np.ndarray:
        return np.bincount(self.dst, minlength=self.n_nodes)

    def out_degree(self) -> np.ndarray:
        return np.bincount(self.src, minlength=self.n_nodes)

    def isolated(self) -> int:
        return int(((self.in_degree() + self.out_degree()) == 0).sum())


def empty_graph(n_nodes: int) -> Graph:
    """`no_graph` control：完全沒有邊。"""
    z_int = np.zeros(0, dtype=np.int64)
    return Graph(n_nodes, z_int, z_int.copy(), z_int.copy(),
                 np.zeros(0, dtype=np.float32), ("none",))


def self_loops(n_nodes: int) -> Graph:
    """`self` control：只有自環，沒有跨公司邊。"""
    idx = np.arange(n_nodes, dtype=np.int64)
    return Graph(n_nodes, idx, idx.copy(), np.zeros(n_nodes, dtype=np.int64),
                 np.ones(n_nodes, dtype=np.float32), ("self",))


def sector_edges(siccd: np.ndarray, digits: int = 2) -> Graph:
    """同 SIC 前 `digits` 位數的節點兩兩相連（雙向）。

    2 位數是 SIC major group，也是產業動能文獻的慣例。1 位數在 top-500 只有
    10 個群組、平均度 76.8；3 位數則留下 49 個孤立節點。
    """
    n = len(siccd)
    key = np.array([str(int(s)).zfill(4)[:digits] for s in siccd])

    src_list, dst_list = [], []
    for group in np.unique(key):
        members = np.flatnonzero(key == group)
        if len(members) < 2:
            continue
        a, b = np.meshgrid(members, members, indexing="ij")
        off_diagonal = a != b
        src_list.append(a[off_diagonal])
        dst_list.append(b[off_diagonal])

    if not src_list:
        return empty_graph(n)

    src = np.concatenate(src_list)
    dst = np.concatenate(dst_list)
    return Graph(n, src, dst,
                 np.zeros(len(src), dtype=np.int64),
                 np.ones(len(src), dtype=np.float32),
                 ("same_sector",))


def corr_edges(window: np.ndarray, k: int = 5) -> Graph:
    """由報酬視窗建 top-k 相關圖。

    `window` 形狀為 (T, N)，只能包含決策日當日與之前的報酬。
    每個節點取相關係數絕對值最大的 k 個其他節點作為訊息來源，
    因此 in-degree 恆為 k，與 N 無關——這讓 G_corr 成為密度受控的那一臂。

    relation 是相關係數的正負號，weight 是絕對值。
    """
    n = window.shape[1]
    if n < 2:
        return empty_graph(n)

    k = min(k, n - 1)
    corr = np.corrcoef(window, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)

    strength = np.abs(corr)
    # argpartition 取每列前 k 大，順序不重要因為之後不依賴排名。
    neighbours = np.argpartition(-strength, kth=k - 1, axis=1)[:, :k]

    dst = np.repeat(np.arange(n, dtype=np.int64), k)
    src = neighbours.reshape(-1).astype(np.int64)
    rho = corr[dst, src]

    return Graph(n, src, dst,
                 (rho < 0).astype(np.int64),
                 np.abs(rho).astype(np.float32),
                 ("pos", "neg"))


def _pair_key(src: np.ndarray, dst: np.ndarray, n_nodes: int) -> np.ndarray:
    return src.astype(np.int64) * n_nodes + dst.astype(np.int64)


def shuffle_topology(graph: Graph, rng: np.random.Generator,
                     max_passes: int = 200) -> Graph:
    """Degree-preserving rewiring，產出的圖必須是 simple graph。

    先把 `src` 整體重排——node i 作為來源出現的次數不變（out-degree 保留），
    `dst` 完全不動（in-degree 保留），relation 與 weight 的多重集合也不變。

    重排會製造自環與**重複邊**，兩者都必須修掉。重複邊特別要緊：real 圖沒有
    重複邊，而 sum aggregation 會把重複邊的訊息重複計入，等於 control 的有效
    容量比 real 大。單純重排一次的話，sector graph 會留下約 5% 重複邊。

    修法是 double-edge swap：把有問題那條邊的來源與另一條隨機邊互換，
    只在兩邊都不produce自環與重複時接受。這保住兩個度數序列。
    """
    if graph.n_edges == 0:
        return graph

    n = graph.n_nodes
    dst = graph.dst
    src = rng.permutation(graph.src)
    taken = set(_pair_key(src, dst, n).tolist())

    def offending() -> np.ndarray:
        keys = _pair_key(src, dst, n)
        _, first_idx, counts = np.unique(keys, return_index=True, return_counts=True)
        duplicated = np.isin(keys, keys[first_idx[counts > 1]])
        # 每組重複只留第一條，其餘視為待修。
        seen = set()
        flags = np.zeros(len(keys), dtype=bool)
        for i, (k, is_dup) in enumerate(zip(keys.tolist(), duplicated.tolist())):
            if src[i] == dst[i]:
                flags[i] = True
            elif is_dup:
                if k in seen:
                    flags[i] = True
                else:
                    seen.add(k)
        return np.flatnonzero(flags)

    for _ in range(max_passes):
        bad = offending()
        if not len(bad):
            break
        taken = set(_pair_key(src, dst, n).tolist())
        for i in bad:
            for _ in range(50):
                j = int(rng.integers(0, graph.n_edges))
                new_i = (int(src[j]), int(dst[i]))
                new_j = (int(src[i]), int(dst[j]))
                if new_i[0] == new_i[1] or new_j[0] == new_j[1]:
                    continue
                k_i = new_i[0] * n + new_i[1]
                k_j = new_j[0] * n + new_j[1]
                if k_i in taken or k_j in taken or k_i == k_j:
                    continue
                taken.discard(int(src[i]) * n + int(dst[i]))
                taken.discard(int(src[j]) * n + int(dst[j]))
                taken.add(k_i)
                taken.add(k_j)
                src[i], src[j] = src[j], src[i]
                break
    else:
        raise RuntimeError(
            f"topology shuffle 無法在 {max_passes} 輪內產生 simple graph"
            f"（{graph.n_edges} 邊 / {n} 節點，圖可能過密）"
        )

    return replace(graph, src=src)


def shuffle_relations(graph: Graph, rng: np.random.Generator) -> Graph:
    """只打亂 relation label，拓樸、度數、weight 全部不動。

    單一 relation 的圖套用這個等於沒做事，依協定應標記為不適用而非假造 control。
    """
    if len(graph.relation_names) < 2:
        raise ValueError(
            "單一 relation 的圖不適用 relation_shuffle，"
            "應在報告中標記為『不適用』，不可假造 typed control"
        )
    return replace(graph, relation=rng.permutation(graph.relation))
