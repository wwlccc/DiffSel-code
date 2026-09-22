from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class AnchorSelectionDataset(Dataset):
    def __init__(self, filename, *, num_instances=None, offset=0):
        if offset < 0 or (num_instances is not None and num_instances <= 0):
            raise ValueError("offset must be nonnegative and num_instances positive")
        with np.load(filename, allow_pickle=False) as archive:
            self.data = {key: archive[key] for key in archive.files}
        if str(self.data.get("schema", "")) != "absolute_toa_v1":
            raise ValueError("expected an absolute_toa_v1 dataset")
        self.n, self.m = int(self.data["n"]), int(self.data["m"])
        size = len(self.data["na"])
        if size == 0 or self.n < 3 or self.m < 1:
            raise ValueError("dataset must contain samples with N>=3 and M>=1")
        for key, shape in (("anchor_positions", (size, self.n, 2)), ("user_positions", (size, self.m, 2)),
                           ("selected_mask", (size, self.n))):
            if self.data[key].shape != shape or not np.isfinite(self.data[key]).all():
                raise ValueError(f"invalid {key} shape or nonfinite values")
        for key in ("side_length", "na", "geometry_id", "best_crlb", "sigma_d", "c", "rcond"):
            if self.data[key].shape != (size,):
                raise ValueError(f"invalid {key} shape")
        target, budget = self.data["selected_mask"], self.data["na"]
        if not np.all((target == 0) | (target == 1)) or not np.all(target.sum(-1) == budget):
            raise ValueError("labels must be binary and sum to Na")
        if not np.all((budget >= 3) & (budget <= self.n)) or not np.all(budget == budget.astype(np.int64)):
            raise ValueError("invalid anchor selection budgets")
        for key in ("side_length", "best_crlb", "sigma_d", "c", "rcond"):
            if not np.all(np.isfinite(self.data[key]) & (self.data[key] > 0)):
                raise ValueError(f"{key} must be finite and positive")
        self.indices = np.arange(size)[offset:None if num_instances is None else offset + num_instances]
        if len(self.indices) == 0:
            raise ValueError("empty dataset slice")
        self.geometry_ids = self.data["geometry_id"][self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = self.indices[index]
        return {
            "anchor_positions": torch.tensor(self.data["anchor_positions"][source], dtype=torch.float32),
            "user_positions": torch.tensor(self.data["user_positions"][source], dtype=torch.float32),
            "target": torch.tensor(self.data["selected_mask"][source], dtype=torch.float32),
            "na": torch.tensor(self.data["na"][source], dtype=torch.long),
            "side_length": torch.tensor(self.data["side_length"][source], dtype=torch.float32),
        }


def grouped_split(geometry_ids, validation_fraction=0.1, seed=42):
    """Keep every label for the same geometry on the same side of the split."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must lie in [0,1)")
    identities = np.asarray(geometry_ids)
    all_indices = np.arange(len(identities))
    if validation_fraction == 0:
        return all_indices, np.array([], dtype=np.int64)
    unique = np.unique(identities)
    if len(unique) < 2:
        raise ValueError("validation needs at least two geometries; use val_fraction=0 for a smoke test")
    shuffled = np.random.default_rng(seed).permutation(unique)
    count = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    is_validation = np.isin(identities, shuffled[:count])
    return all_indices[~is_validation], all_indices[is_validation]


def build_graph_features(anchors, users, na, selection, side_length):
    """One feature path for both training and reverse sampling (no user bits)."""
    if anchors.ndim != 3 or anchors.shape[-1] != 2 or users.ndim != 3 or users.shape[-1] != 2:
        raise ValueError("coordinates must have shape [B,count,2]")
    batch, n, _ = anchors.shape
    m = users.shape[1]
    if n == 0 or m == 0 or users.shape[0] != batch or selection.shape != (batch, n):
        raise ValueError("inconsistent geometry/selection sizes")
    budget = na.to(device=anchors.device, dtype=anchors.dtype).reshape(-1)
    if budget.numel() != batch or torch.any((budget < 0) | (budget > n)):
        raise ValueError("one budget in 0..N is required per graph")
    side = side_length.to(device=anchors.device, dtype=anchors.dtype).reshape(-1)
    if side.numel() == 1:
        side = side.expand(batch)
    if side.numel() != batch or not torch.all(torch.isfinite(side) & (side > 0)):
        raise ValueError("one finite positive scale is required per graph")
    ratio = (budget / n)[:, None, None]
    count_bias = (selection.sum(-1) - budget)[:, None, None]
    anchor_features = torch.cat((anchors / side[:, None, None], ratio.expand(-1, n, -1),
                                 selection[..., None].to(anchors.dtype), count_bias.expand(-1, n, -1)), dim=-1)
    user_features = torch.cat((users / side[:, None, None], ratio.expand(-1, m, -1)), dim=-1)
    positions = torch.cat((anchors, users), dim=1) / side[:, None, None]
    edges = torch.linalg.vector_norm(positions[:, :, None, :] - positions[:, None, :, :], dim=-1)
    return anchor_features, user_features, edges
