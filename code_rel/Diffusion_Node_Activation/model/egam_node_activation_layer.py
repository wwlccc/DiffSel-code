from __future__ import annotations

import math

import torch
from torch import nn


class Normalization(nn.Module):
    def __init__(self, embed_dim: int, normalization: str = "layer") -> None:
        super().__init__()
        if normalization == "batch":
            self.normalizer = nn.BatchNorm1d(embed_dim, affine=True)
            self.batch = True
        elif normalization == "layer":
            self.normalizer = nn.LayerNorm(embed_dim, elementwise_affine=True)
            self.batch = False
        else:
            raise ValueError(f"Unsupported normalization type: {normalization}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x
        if self.batch:
            embed_dim = x.size(-1)
            return self.normalizer(x.reshape(-1, embed_dim)).view_as(x)
        return self.normalizer(x)


class BatchNodeToNodeAttention(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by n_heads={n_heads}")
        self.n_heads = int(n_heads)
        self.head_dim = int(embed_dim // n_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, embed_dim = query.shape
        q = self.q_proj(query).view(batch, num_nodes, self.n_heads, self.head_dim)
        k = self.k_proj(key_value).view(batch, num_nodes, self.n_heads, self.head_dim)
        v = self.v_proj(key_value).view(batch, num_nodes, self.n_heads, self.head_dim)
        scores = torch.einsum("bihd,bjhd->bhij", q, k) * self.scale
        weights = torch.softmax(scores, dim=-1)
        messages = torch.einsum("bhij,bjhd->bihd", weights, v)
        return self.out_proj(messages.reshape(batch, num_nodes, embed_dim))


class BatchEdgeFromNodes(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by n_heads={n_heads}")
        self.n_heads = int(n_heads)
        self.head_dim = int(embed_dim // n_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim + 1, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim + 1, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, edge_features: torch.Tensor, node_features: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, _, embed_dim = edge_features.shape
        q = self.q_proj(edge_features).view(batch, num_nodes, num_nodes, self.n_heads, 1, self.head_dim)
        src = node_features[:, :, None, :].expand(batch, num_nodes, num_nodes, embed_dim)
        dst = node_features[:, None, :, :].expand(batch, num_nodes, num_nodes, embed_dim)
        src = torch.cat([src, torch.full((*src.shape[:-1], 1), -1.0, device=src.device, dtype=src.dtype)], dim=-1)
        dst = torch.cat([dst, torch.full((*dst.shape[:-1], 1), 1.0, device=dst.device, dtype=dst.dtype)], dim=-1)
        pair = torch.stack([src, dst], dim=3)
        k = self.k_proj(pair).view(batch, num_nodes, num_nodes, 2, self.n_heads, self.head_dim).permute(0, 1, 2, 4, 3, 5)
        v = self.v_proj(pair).view(batch, num_nodes, num_nodes, 2, self.n_heads, self.head_dim).permute(0, 1, 2, 4, 3, 5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        out = torch.matmul(torch.softmax(attn, dim=-1), v)
        return self.out_proj(out.reshape(batch, num_nodes, num_nodes, embed_dim))


class BatchNodeFromEdgesAttention(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int) -> None:
        super().__init__()
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by n_heads={n_heads}")
        self.n_heads = int(n_heads)
        self.head_dim = int(embed_dim // n_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, embed_dim = node_features.shape
        q = self.q_proj(node_features).view(batch, num_nodes, self.n_heads, self.head_dim)
        k = self.k_proj(edge_features).view(batch, num_nodes, num_nodes, self.n_heads, self.head_dim)
        v = self.v_proj(edge_features).view(batch, num_nodes, num_nodes, self.n_heads, self.head_dim)
        scores = torch.einsum("bihd,bijhd->bhij", q, k) * self.scale
        weights = torch.softmax(scores, dim=-1)
        messages = torch.einsum("bhij,bijhd->bihd", weights, v)
        return self.out_proj(messages.reshape(batch, num_nodes, embed_dim))


class EGAMNodeActivationLayer(nn.Module):
    def __init__(self, n_heads: int, embed_dim: int, feed_forward_hidden: int, normalization: str = "layer") -> None:
        super().__init__()
        self.node_to_node = BatchNodeToNodeAttention(n_heads, embed_dim)
        self.edge_from_nodes = BatchEdgeFromNodes(n_heads, embed_dim)
        self.node_from_edges = BatchNodeFromEdgesAttention(n_heads, embed_dim)
        self.node_norm_1 = Normalization(embed_dim, normalization)
        self.node_norm_2 = Normalization(embed_dim, normalization)
        self.edge_norm_1 = Normalization(embed_dim, normalization)
        self.edge_norm_2 = Normalization(embed_dim, normalization)
        self.node_ffn = nn.Sequential(nn.Linear(embed_dim, feed_forward_hidden), nn.ReLU(), nn.Linear(feed_forward_hidden, embed_dim))
        self.edge_ffn = nn.Sequential(nn.Linear(embed_dim, feed_forward_hidden), nn.ReLU(), nn.Linear(feed_forward_hidden, embed_dim))

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = node_features + self.node_to_node(node_features, node_features)
        edge_features = edge_features + self.edge_from_nodes(edge_features, node_features)
        node_features = node_features + self.node_from_edges(node_features, edge_features)
        node_features = self.node_norm_1(node_features)
        node_features = self.node_norm_2(node_features + self.node_ffn(node_features))
        edge_features = self.edge_norm_1(edge_features)
        edge_features = self.edge_norm_2(edge_features + self.edge_ffn(edge_features))
        return node_features, edge_features
