from __future__ import annotations

import os

import torch
from torch.utils.data import Dataset


class NodeActivationDataset(Dataset):
    """Node activation dataset backed by the NPZ files generated at repo root."""

    def __init__(
        self,
        filename: str | os.PathLike[str],
        *,
        num_instances: int | None = None,
        offset: int = 0,
    ) -> None:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if num_instances is not None and num_instances < 0:
            raise ValueError("num_instances must be non-negative")

        import numpy as np

        self.filename = os.fspath(filename)
        data = np.load(self.filename, allow_pickle=True)
        self.positions = data["positions"].astype("float32")
        self.selected_mask = data["selected_mask"].astype("float32")
        self.na = data["na"].astype("int64")
        self.side_length = data["side_length"].astype("float32")
        self.best_root_mean_crlb = data["best_root_mean_crlb"].astype("float32")
        self.instance_id = data["instance_id"].astype("int64")
        self.n = int(data["n"]) if "n" in data else int(self.positions.shape[1])

        if self.positions.ndim != 3 or self.positions.shape[-1] != 2:
            raise ValueError("positions must have shape [M, N, 2]")
        if self.selected_mask.shape != self.positions.shape[:2]:
            raise ValueError("selected_mask must have shape [M, N]")

        end = None if num_instances is None else offset + num_instances
        self.indices = list(range(self.positions.shape[0]))[offset:end]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        source_index = self.indices[index]
        positions = torch.as_tensor(self.positions[source_index], dtype=torch.float32)
        target = torch.as_tensor(self.selected_mask[source_index], dtype=torch.long)
        na = int(self.na[source_index])
        side_length = float(self.side_length[source_index])
        if side_length <= 0:
            raise ValueError("side_length must be positive")

        normalized_positions = positions / side_length
        distance = torch.linalg.norm(positions[:, None, :] - positions[None, :, :], dim=-1) / side_length
        na_ratio = torch.full((self.n, 1), float(na) / float(self.n), dtype=torch.float32)
        clean_selection = target.float().unsqueeze(-1)
        selection_count_bias = torch.zeros((self.n, 1), dtype=torch.float32)
        node_features_clean = torch.cat([normalized_positions, na_ratio, clean_selection, selection_count_bias], dim=-1)

        return {
            "dataset_index": int(source_index),
            "instance_id": int(self.instance_id[source_index]),
            "positions": positions,
            "edge_features": distance,
            "node_features_clean": node_features_clean,
            "target": target,
            "na": torch.tensor(na, dtype=torch.long),
            "na_ratio": torch.tensor(float(na) / float(self.n), dtype=torch.float32),
            "side_length": torch.tensor(side_length, dtype=torch.float32),
            "best_root_mean_crlb": torch.tensor(float(self.best_root_mean_crlb[source_index]), dtype=torch.float32),
        }


def load_node_activation_dataset(
    filename: str | os.PathLike[str],
    *,
    num_instances: int | None = None,
    offset: int = 0,
) -> NodeActivationDataset:
    return NodeActivationDataset(filename, num_instances=num_instances, offset=offset)
