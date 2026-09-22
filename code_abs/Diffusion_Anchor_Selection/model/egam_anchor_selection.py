from __future__ import annotations

import math
import torch
from torch import nn
from .egam_layer import EGAMNodeActivationLayer, Normalization


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    frequency = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=timesteps.device) / (dim // 2))
    phase = timesteps.float()[:, None] * frequency[None, :]
    return torch.cat((phase.cos(), phase.sin()), dim=-1)


class ScalarEmbeddingSine(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, values):
        index = torch.arange(self.embed_dim, dtype=values.dtype, device=values.device)
        frequency = 10000 ** (2 * torch.div(index, 2, rounding_mode="trunc") / self.embed_dim)
        phase = values[..., None] / frequency
        return torch.stack((phase[..., 0::2].sin(), phase[..., 1::2].cos()), dim=-1).flatten(-2)


class EGAM_AnchorSelection(nn.Module):
    """Separate anchor/user embeddings, common dense graph blocks, anchor-only logits.

    Anchors: [x/scale, y/scale, K/N, noisy bit, sum(noisy bits)-K].
    Users: [x/scale, y/scale, K/N]. Graph order is anchors followed by users.
    """
    def __init__(self, embed_dim=128, n_encode_layers=4, n_heads=8,
                 feed_forward_hidden=None, normalization="layer"):
        super().__init__()
        if embed_dim < 2 or embed_dim % 2 or n_heads <= 0 or embed_dim % n_heads:
            raise ValueError("embed_dim must be positive/even and divisible by n_heads")
        if n_encode_layers <= 0:
            raise ValueError("n_encode_layers must be positive")
        self.embed_dim = embed_dim
        hidden = feed_forward_hidden if feed_forward_hidden is not None else 2 * embed_dim
        self.init_embed_anchor = nn.Linear(5, embed_dim)
        self.init_embed_user = nn.Linear(3, embed_dim)
        self.edge_pos_embed = ScalarEmbeddingSine(embed_dim)
        self.init_embed_edge = nn.Linear(embed_dim, embed_dim)
        self.time_embed = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.time_embed_layers = nn.ModuleList([
            nn.Sequential(nn.ReLU(), nn.Linear(embed_dim, embed_dim)) for _ in range(n_encode_layers)])
        self.layers = nn.ModuleList([
            EGAMNodeActivationLayer(n_heads, embed_dim, hidden, normalization) for _ in range(n_encode_layers)])
        self.anchor_out = nn.Sequential(Normalization(embed_dim, normalization), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, anchor_features, user_features, edge_features, timesteps):
        if anchor_features.ndim != 3 or anchor_features.shape[-1] != 5:
            raise ValueError("anchor_features must have shape [B,N,5]")
        if user_features.ndim != 3 or user_features.shape[-1] != 3:
            raise ValueError("user_features must have shape [B,M,3]")
        batch, n, _ = anchor_features.shape
        m = user_features.shape[1]
        if n == 0 or m == 0 or user_features.shape[0] != batch or edge_features.shape != (batch, n + m, n + m):
            raise ValueError("inconsistent anchor/user/edge graph sizes")
        nodes = torch.cat((self.init_embed_anchor(anchor_features), self.init_embed_user(user_features)), dim=1)
        edges = self.init_embed_edge(self.edge_pos_embed(edge_features))
        timesteps = torch.as_tensor(timesteps, device=nodes.device).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(batch)
        if timesteps.numel() != batch:
            raise ValueError("supply one timestep per graph")
        time = self.time_embed(timestep_embedding(timesteps, self.embed_dim).to(nodes.dtype))
        for layer, time_layer in zip(self.layers, self.time_embed_layers):
            nodes, edges = layer(nodes, edges)
            nodes = nodes + time_layer(time)[:, None, :]
        return self.anchor_out(nodes[:, :n]).squeeze(-1)
