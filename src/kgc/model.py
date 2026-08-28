"""Small fixed-capacity typed GNN used by the Stage C supervised gate.

The model deliberately has the same parameters for ``no_graph``, ``self`` and
all shuffled graphs.  A graph only changes the edge tensors passed to forward;
it never changes the number of layers or relation channels.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .graphs import Graph


class TypedGraphLayer(nn.Module):
    """One relation-aware message-passing layer."""

    def __init__(self, hidden_dim: int, n_relations: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.relation_linear = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=False)
             for _ in range(n_relations)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, states: torch.Tensor, graph: Graph) -> torch.Tensor:
        # The self path is present even for no_graph, so every variant has the
        # same parameter count and a well-defined price-only representation.
        output = self.self_linear(states)
        if graph.n_edges:
            src = torch.as_tensor(graph.src, dtype=torch.long, device=states.device)
            dst = torch.as_tensor(graph.dst, dtype=torch.long, device=states.device)
            relation = torch.as_tensor(graph.relation, dtype=torch.long, device=states.device)
            weight = torch.as_tensor(graph.weight, dtype=states.dtype, device=states.device)
            messages = torch.zeros_like(states)
            for rel_id, linear in enumerate(self.relation_linear):
                selected = relation == rel_id
                if not torch.any(selected):
                    continue
                transformed = linear(states[src[selected]])
                transformed = transformed * weight[selected].unsqueeze(-1)
                messages.index_add_(0, dst[selected], transformed)
            degree = torch.zeros(states.shape[0], dtype=states.dtype, device=states.device)
            degree.index_add_(0, dst, weight)
            output = output + messages / degree.clamp_min(1e-6).unsqueeze(-1)
        return torch.relu(self.norm(output))


class GraphReturnModel(nn.Module):
    """Two-layer typed GNN followed by a per-node linear return head."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64,
                 n_relations: int = 2) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            [TypedGraphLayer(hidden_dim, n_relations) for _ in range(2)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, graph: Graph) -> torch.Tensor:
        states = self.input(features)
        for layer in self.layers:
            states = layer(states, graph)
        return self.head(states).squeeze(-1)


def node_features(window: np.ndarray) -> np.ndarray:
    """Convert a (time, node) return window to fixed four-dimensional features."""
    if window.ndim != 2 or window.shape[0] == 0:
        raise ValueError("window must have shape (time, node) and be non-empty")
    cumulative = np.prod(1.0 + window, axis=0) - 1.0
    return np.column_stack([
        window[-1],
        np.mean(window, axis=0),
        np.std(window, axis=0),
        cumulative,
    ]).astype(np.float32)
