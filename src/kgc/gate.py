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
              seed: int, epochs: int, device: str, lr: float) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(epochs):
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
        losses.append(float(np.mean(epoch_losses)))
    return losses


def evaluate(model: GraphReturnModel, panel: Panel, dates, variant: str,
             seed: int, device: str) -> dict[str, float | int]:
    model.eval()
    ics = []
    mses = []
    with torch.no_grad():
        for date in dates:
            graph, kept = graph_for(panel, date, variant, seed)
            window, _ = panel.window_at(date, kept)
            target, valid = panel.target_at(date, kept)
            if not valid.all():
                continue
            x = torch.from_numpy(node_features(window)).to(device)
            prediction = model(x, graph).cpu().numpy()
            if np.std(prediction) == 0.0 or np.std(target) == 0.0:
                continue
            ics.append(float(spearmanr(prediction, target).statistic))
            mses.append(float(np.mean(np.square(prediction - target))))
    return {
        "test_rank_ic_mean": float(np.mean(ics)) if ics else float("nan"),
        "test_rank_ic_std": float(np.std(ics)) if ics else float("nan"),
        "test_mse": float(np.mean(mses)) if mses else float("nan"),
        "n_test_days": len(ics),
    }


def run_config(panel: Panel, train_dates, test_dates, variant: str, seed: int,
               epochs: int = 1, hidden_dim: int = 64, device: str = "cpu",
               lr: float = 1e-3) -> dict:
    torch.manual_seed(seed)
    model = GraphReturnModel(input_dim=4, hidden_dim=hidden_dim,
                             n_relations=2).to(device)
    started = time.perf_counter()
    losses = train_one(model, panel, train_dates, variant, seed, epochs, device, lr)
    metrics = evaluate(model, panel, test_dates, variant, seed, device)
    metrics.update({
        "variant": variant,
        "seed": seed,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "train_loss_first": losses[0],
        "train_loss_last": losses[-1],
        "runtime_seconds": time.perf_counter() - started,
    })
    return metrics
