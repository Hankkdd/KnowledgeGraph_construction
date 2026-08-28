"""Stage C 正控制：模型學得會一個「確定存在」的圖訊號嗎？

C1 smoke 顯示 loss 從隨機初始化下降，但下降到的位置約等於 target 變異數——
也就是模型收斂到「預測橫斷面均值」這個平凡解，五個 variant 因此得到幾乎相同的
最終 loss。loss 下降本身不足以證明 pipeline 能學。

這支腳本用真實的特徵與真實的 G_corr 圖，但把 target 換成一個由圖產生的合成訊號：

    y_i = beta * (i 的鄰居特徵平均) + noise

訊號是植入的，所以答案已知。判準有兩個，缺一不可：

    real 的 test Rank IC 必須高          → 模型學得會
    topology_shuffle 必須明顯較低        → 學到的是圖結構，不是節點自身特徵

兩者都成立，之後在真實 target 上得到的 null 才有意義；否則 null 只反映
模型學不動，跟 KG 有沒有資訊無關。

用法：
    PYTHONPATH=src .venv/bin/python scripts/19_positive_control.py
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy.stats import spearmanr

from kgc import dataset, gate, graphs, wrds_io
from kgc.model import GraphReturnModel, node_features

OUT_DIR = wrds_io.REPO_ROOT / "artifacts" / "stage_c_positive_control"

# 與 C2 相同的五個 variant。合成 target 只依賴鄰居拓樸、不依賴 relation label，
# 所以 relation_shuffle 應該接近 real，topology_shuffle 應該明顯掉下來——
# 這組對比確認我們量到的是拓樸訊號，不是 relation 或 edge weight 的意外效果。
VARIANTS = ("no_graph", "self", "real", "relation_shuffle", "topology_shuffle")


def planted_target(features: np.ndarray, graph: graphs.Graph,
                   beta: float, noise: float, rng) -> np.ndarray:
    """由鄰居特徵平均產生的合成 target，橫斷面標準化後加雜訊。

    只用 `graph` 的真實拓樸產生訊號。用 shuffled 圖訓練的模型看到的是錯的鄰居，
    因此理論上無法還原——這是判準的第二半。

    訊號必須先橫斷面標準化再加雜訊：鄰居的 60 日平均報酬量級約 5e-4，
    直接加上絕對量級的雜訊會讓 target 幾乎全是雜訊，正控制就測不到東西。
    `noise` 是相對於標準化訊號的比率，0.5 代表訊噪比 2:1。
    """
    signal = np.zeros(len(features), dtype=np.float64)
    if graph.n_edges:
        contribution = features[graph.src, 1]      # 鄰居的 60 日平均報酬
        np.add.at(signal, graph.dst, contribution)
        degree = np.bincount(graph.dst, minlength=len(features)).clip(min=1)
        signal /= degree

    spread = signal.std()
    signal = (signal - signal.mean()) / spread if spread > 0 else signal
    y = beta * signal + noise * rng.standard_normal(len(features))
    return y - y.mean()


def run(panel, train_dates, test_dates, variant, seed, epochs, device,
        beta, noise, lr):
    torch.manual_seed(seed)
    model = GraphReturnModel(input_dim=4, hidden_dim=64, n_relations=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    started = time.perf_counter()
    losses = []
    train_baseline: list[float] = []
    for epoch_i in range(epochs):
        model.train()
        epoch = []
        for date in train_dates:
            model_graph, kept = gate.graph_for(panel, date, variant, seed)
            true_graph, _ = gate.graph_for(panel, date, "real", seed)
            window, _ = panel.window_at(date, kept)
            x_np = node_features(window)

            # target 一律由真實圖產生；variant 只改模型看得到的圖。
            rng = np.random.default_rng(seed + int(np.datetime64(date, "D").astype(int)))
            y_np = planted_target(x_np, true_graph, beta, noise, rng)

            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np.astype(np.float32)).to(device)
            loss = torch.mean(torch.square(model(x, model_graph) - y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch.append(float(loss.detach().cpu()))
            if epoch_i == 0:
                # 訓練期自己的 mean-predictor 基準，不能拿測試期的變異數來比。
                train_baseline.append(float(np.mean(y_np ** 2)))
        losses.append(float(np.mean(epoch)))

    model.eval()
    ics, baseline, test_mse = [], [], []
    with torch.no_grad():
        for date in test_dates:
            model_graph, kept = gate.graph_for(panel, date, variant, seed)
            true_graph, _ = gate.graph_for(panel, date, "real", seed)
            window, _ = panel.window_at(date, kept)
            x_np = node_features(window)
            rng = np.random.default_rng(seed + int(np.datetime64(date, "D").astype(int)))
            y_np = planted_target(x_np, true_graph, beta, noise, rng)

            pred = model(torch.from_numpy(x_np).to(device), model_graph).cpu().numpy()
            if np.std(pred) == 0 or np.std(y_np) == 0:
                continue
            ics.append(float(spearmanr(pred, y_np).statistic))
            baseline.append(float(np.mean(np.square(y_np))))
            test_mse.append(float(np.mean(np.square(pred - y_np))))

    train_var = float(np.mean(train_baseline))
    test_var = float(np.mean(baseline))
    return {
        "variant": variant,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "train_target_variance": train_var,
        "beats_mean_on_train": bool(losses[-1] < train_var),
        "test_mse": float(np.mean(test_mse)),
        "test_target_variance": test_var,
        "beats_mean_on_test": bool(np.mean(test_mse) < test_var),
        "test_rank_ic": float(np.mean(ics)),
        "n_test_days": len(ics),
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", type=int, default=500)
    p.add_argument("--test-year", type=int, default=2025)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--noise", type=float, default=0.5,
                   help="相對於標準化訊號的雜訊比率；0.5 = 訊噪比 2:1")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-stride", type=int, default=1)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    panel = dataset.Panel.load(args.universe)
    fold = next(f for f in dataset.walk_forward(panel.calendar)
                if f.test_year == args.test_year)

    def window_dates(lo, hi, stride=1):
        sel = panel.calendar[(panel.calendar >= np.datetime64(lo)) &
                             (panel.calendar <= np.datetime64(hi))][::stride]
        return np.array([d for d in sel
                         if panel.usable_mask(d, panel.nodes_at(d)).sum() >= 2])

    train_dates = window_dates(fold.train_start, fold.train_end, args.train_stride)
    test_dates = window_dates(fold.test_start, fold.test_end)
    print(f"{fold}\ntrain={len(train_dates)} test={len(test_dates)} device={device}\n",
          flush=True)

    results = []
    for variant in VARIANTS:
        r = run(panel, train_dates, test_dates, variant, args.seed,
                args.epochs, device, args.beta, args.noise, args.lr)
        results.append(r)
        print(f"{r['variant']:18s} "
              f"train {r['train_loss_last']:.4f}/{r['train_target_variance']:.4f} "
              f"({'beats' if r['beats_mean_on_train'] else 'WORSE':>5s})  "
              f"test {r['test_mse']:.4f}/{r['test_target_variance']:.4f} "
              f"({'beats' if r['beats_mean_on_test'] else 'WORSE':>5s})  "
              f"IC {r['test_rank_ic']:+.4f}  {r['runtime_seconds']:.0f}s", flush=True)

    by = {r["variant"]: r for r in results}
    real_ic = by["real"]["test_rank_ic"]
    topo_ic = by["topology_shuffle"]["test_rank_ic"]
    rel_ic = by["relation_shuffle"]["test_rank_ic"]
    verdict = {
        "model_can_learn": bool(by["real"]["beats_mean_on_train"]
                                and by["real"]["beats_mean_on_test"]
                                and real_ic > 0.5),
        "model_uses_graph_structure": bool(real_ic - topo_ic > 0.2),
        # 合成 target 只依賴拓樸，所以 relation_shuffle 應該接近 real。
        # 若它也大幅下降，代表量到的不是純拓樸訊號。
        "signal_is_topological": bool(abs(real_ic - rel_ic) < abs(real_ic - topo_ic)),
        "real_ic": real_ic,
        "relation_shuffle_ic": rel_ic,
        "topology_shuffle_ic": topo_ic,
        "gap_vs_topology": real_ic - topo_ic,
        "gap_vs_relation": real_ic - rel_ic,
    }
    verdict["pass"] = (verdict["model_can_learn"]
                       and verdict["model_uses_graph_structure"]
                       and verdict["signal_is_topological"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"top{args.universe}_seed{args.seed}.json").write_text(
        json.dumps({"config": vars(args), "fold": str(fold),
                    "results": results, "verdict": verdict},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(f"\n模型學得會植入訊號   {verdict['model_can_learn']}")
    print(f"模型用到了圖結構     {verdict['model_uses_graph_structure']} "
          f"(real {real_ic:+.4f} vs topology {topo_ic:+.4f}, "
          f"gap {verdict['gap_vs_topology']:+.4f})")
    print(f"訊號來自拓樸而非標籤 {verdict['signal_is_topological']} "
          f"(relation {rel_ic:+.4f}, gap {verdict['gap_vs_relation']:+.4f})")
    print(f"正控制               {'PASS' if verdict['pass'] else 'FAIL'}")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
