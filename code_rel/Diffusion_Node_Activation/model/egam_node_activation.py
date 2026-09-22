from __future__ import annotations

import math

import torch
from torch import nn

from model.egam_node_activation_layer import EGAMNodeActivationLayer, Normalization


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ScalarEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int, temperature: int = 10000) -> None:
        super().__init__()
        if num_pos_feats <= 0 or num_pos_feats % 2 != 0:
            raise ValueError("num_pos_feats must be a positive even integer")
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = int(temperature)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.dim() >= 1 and values.size(-1) == 1:
            values = values.squeeze(-1)
        dim_t = torch.arange(self.num_pos_feats, dtype=values.dtype, device=values.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="trunc") / self.num_pos_feats)
        encoded = values.unsqueeze(-1) / dim_t
        return torch.stack((encoded[..., 0::2].sin(), encoded[..., 1::2].cos()), dim=-1).flatten(-2)


class EGAM_NodeActivation(nn.Module):
    """Dense batched EGAM model that predicts active-node logits."""

    def __init__(
        self,
        *,
        embed_dim: int = 128,
        n_encode_layers: int = 4,
        normalization: str = "layer",
        n_heads: int = 8,
        feed_forward_hidden: int | None = None,
        node_dim: int = 4,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if n_encode_layers <= 0:
            raise ValueError("n_encode_layers must be positive")
        self.embed_dim = int(embed_dim)
        self.out_channels = int(out_channels)
        if feed_forward_hidden is None:
            feed_forward_hidden = self.embed_dim * 2

        self.init_embed_node = nn.Linear(node_dim, self.embed_dim)
        self.edge_pos_embed = ScalarEmbeddingSine(self.embed_dim)
        self.init_embed_edge = nn.Linear(self.embed_dim, self.embed_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.time_embed_layers = nn.ModuleList(
            [nn.Sequential(nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim)) for _ in range(n_encode_layers)]
        )
        self.layers = nn.ModuleList(
            [
                EGAMNodeActivationLayer(
                    n_heads=n_heads,
                    embed_dim=self.embed_dim,
                    feed_forward_hidden=int(feed_forward_hidden),
                    normalization=normalization,
                )
                for _ in range(n_encode_layers)
            ]
        )
        self.node_out = nn.Sequential(
            Normalization(self.embed_dim, normalization),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.out_channels),
        )

    def preforward(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.init_embed_node(node_features), self.init_embed_edge(self.edge_pos_embed(edge_features))

    def _time_features(self, timesteps: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if timesteps is None:
            return None
        timesteps = timesteps.to(device=device)
        if timesteps.dim() == 0:
            timesteps = timesteps.view(1)
        features = self.time_embed(timestep_embedding(timesteps.float(), self.embed_dim))
        if features.size(0) == 1:
            return features.expand(batch_size, -1)
        if features.size(0) == batch_size:
            return features
        raise ValueError("timesteps must contain one value or one value per graph")

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, timesteps: torch.Tensor | None = None) -> torch.Tensor:
        node_embeddings, edge_embeddings = self.preforward(node_features, edge_features)
        time_features = self._time_features(timesteps, node_embeddings.size(0), node_embeddings.device)
        for layer, time_layer in zip(self.layers, self.time_embed_layers):
            node_embeddings, edge_embeddings = layer(node_embeddings, edge_embeddings)
            if time_features is not None:
                node_embeddings = node_embeddings + time_layer(time_features)[:, None, :]
        logits = self.node_out(node_embeddings)
        if self.out_channels == 1:
            return logits.squeeze(-1)
        return logits
