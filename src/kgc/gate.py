"""Training and evaluation helpers for the Stage C supervised gate."""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.stats import spearmanr

from . import graphs
from .dataset import Panel
from .model import GraphReturnModel, node_features


def graph_for(panel: Panel, date, variant: str, seed: int) -> tuple[graphs.Graph, np.ndarray]:
    """Build one graph and return it with the usable global node indices."""
    nodes = panel.nodes_at(date)
    usable = panel.usable_mask(date, nodes)
    kept = nodes[usable]
    if len(kept) < 2:
        raise ValueError(f"fewer than two usable nodes at {date}")
    window, feature_ok = panel.window_at(date, kept)
    if not feature_ok.all():
        raise ValueError(f"usable mask disagrees with feature mask at {date}")
    if variant == "no_graph":
        graph = graphs.empty_graph(len(kept))
    elif variant == "self":
        graph = graphs.self_loops(len(kept))
    else:
        real = graphs.corr_edges(window, k=5)
        rng = np.random.default_rng(seed + int(np.datetime64(date, "D").astype(int)))
        if variant == "real":
            graph = real
        elif variant == "relation_shuffle":
            graph = graphs.shuffle_relations(real, rng)
        elif variant == "topology_shuffle":
            graph = graphs.shuffle_topology(real, rng)
        else:
            raise ValueError(f"unknown variant: {variant}")
    return graph, kept


def train_one(model: GraphReturnModel, panel: Panel, dates, variant: str,
              seed: int, epochs: int, device: str, lr: float) -> tuple[list[float], float]:
    """回傳 (每個 epoch 的平均 loss, 訓練期 target 變異數)。

    訓練期的變異數必須在訓練期算——拿測試期的變異數來比是不同資料區間，
    最終 loss 與它的比較就沒有意義。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    train_baseline: list[float] = []
    for epoch_i in range(epochs):
        epoch_losses = []
        for date in dates:
            graph, kept = graph_for(panel, date, variant, seed)
            window, _ = panel.window_at(date, kept)
            target, valid = panel.target_at(date, kept)
            if not valid.all():
                raise ValueError(f"target mask disagrees with usable mask at {date}")
            x = torch.from_numpy(node_features(window)).to(device)
            y = torch.from_numpy(target.astype(np.float32)).to(device)
            prediction = model(x, graph)
            loss = torch.mean(torch.square(prediction - y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            if epoch_i == 0:
                train_baseline.append(float(np.mean(np.square(target))))
        losses.append(float(np.mean(epoch_losses)))
    return losses, float(np.mean(train_baseline))


def evaluate(model: GraphReturnModel, panel: Panel, dates, variant: str,
             seed: int, device: str) -> dict[str, float | int | bool]:
    """評估並記錄退化診斷。

    C1 的教訓是 loss 下降不代表學到東西：模型可以收斂到「預測橫斷面均值」的常數解，
    此時 Rank IC 純粹是雜訊。因此每個 run 都要保存預測標準差與 mean-predictor 基準，
    讓 summarise 能標記退化的 run，而不是只看 IC。
    """
    model.eval()
    ics, mses, baselines = [], [], []
    pred_stds, target_stds = [], []
    degenerate_days = 0

    with torch.no_grad():
        for date in dates:
            graph, kept = graph_for(panel, date, variant, seed)
            window, _ = panel.window_at(date, kept)
            target, valid = panel.target_at(date, kept)
            if not valid.all():
                continue
            x = torch.from_numpy(node_features(window)).to(device)
            prediction = model(x, graph).cpu().numpy()

            pred_std = float(np.std(prediction))
            target_std = float(np.std(target))
            pred_stds.append(pred_std)
            target_stds.append(target_std)
            baselines.append(float(np.mean(np.square(target))))

            if pred_std == 0.0 or target_std == 0.0:
                degenerate_days += 1
                continue
            ics.append(float(spearmanr(prediction, target).statistic))
            mses.append(float(np.mean(np.square(prediction - target))))

    test_mse = float(np.mean(mses)) if mses else float("nan")
    baseline = float(np.mean(baselines)) if baselines else float("nan")
    pred_std = float(np.mean(pred_stds)) if pred_stds else float("nan")
    target_std = float(np.mean(target_stds)) if target_stds else float("nan")

    return {
        "test_rank_ic_mean": float(np.mean(ics)) if ics else float("nan"),
        "test_rank_ic_std": float(np.std(ics)) if ics else float("nan"),
        "test_mse": test_mse,
        "test_target_variance": baseline,
        "beats_mean_on_test": bool(test_mse < baseline) if mses else False,
        "pred_std_mean": pred_std,
        "target_std_mean": target_std,
        # 預測幾乎沒有橫斷面變異 = 常數預測器，此時的 Rank IC 不具意義。
        "constant_predictor": bool(pred_std < 1e-6 * max(target_std, 1e-12)),
        "degenerate_days": degenerate_days,
        "n_test_days": len(ics),
    }


def run_config(panel: Panel, train_dates, test_dates, variant: str, seed: int,
               epochs: int = 1, hidden_dim: int = 64, device: str = "cpu",
               lr: float = 1e-3) -> dict:
    torch.manual_seed(seed)
    model = GraphReturnModel(input_dim=4, hidden_dim=hidden_dim,
                             n_relations=2).to(device)
    started = time.perf_counter()
    losses, train_variance = train_one(
        model, panel, train_dates, variant, seed, epochs, device, lr)
    metrics = evaluate(model, panel, test_dates, variant, seed, device)
    metrics.update({
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "train_target_variance": train_variance,
        "beats_mean_on_train": bool(losses[-1] < train_variance),
        "n_train_dates": len(train_dates),
        "runtime_seconds": time.perf_counter() - started,
    })
    return metrics
