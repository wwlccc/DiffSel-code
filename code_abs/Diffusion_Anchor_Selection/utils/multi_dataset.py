"""Multiple NPZ sources, geometry-safe splits, and homogeneous graph batches."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, Subset

from .data_utils import AnchorSelectionDataset, grouped_split


def dataset_paths(filenames):
    """Accept legacy single paths and the multi-path CLI without implicit globbing."""
    if isinstance(filenames, (str, Path)):
        filenames = [filenames]
    if not filenames:
        raise ValueError("at least one dataset path is required")
    if any(not str(path).strip() for path in filenames):
        raise ValueError("dataset paths must not be empty")
    paths = [str(Path(path).expanduser().resolve()) for path in filenames]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate dataset paths would repeat training records")
    return paths


def shape_groups(graph_sizes):
    groups = {}
    for index, size in enumerate(graph_sizes):
        groups.setdefault(tuple(int(value) for value in size), []).append(index)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in sorted(groups.items())}


class MultiAnchorSelectionDataset(Dataset):
    """Index separate fixed-size files without stacking incompatible coordinates."""

    def __init__(self, filenames, *, num_instances=None, offset=0):
        self.paths = dataset_paths(filenames)
        self.datasets = [AnchorSelectionDataset(path, num_instances=num_instances, offset=offset)
                         for path in self.paths]
        self.boundaries = np.r_[0, np.cumsum([len(dataset) for dataset in self.datasets])]
        self.geometry_ids = np.concatenate([dataset.geometry_ids.astype(str) for dataset in self.datasets])
        self.graph_sizes = np.concatenate([
            np.tile((dataset.n, dataset.m), (len(dataset), 1)) for dataset in self.datasets])
        self.shape_groups = shape_groups(self.graph_sizes)
        # Existing geometry IDs are coordinate hashes shared by all K labels.
        # A reused ID with a different anchor/user layout is inconsistent metadata.
        check_geometry_sizes(self.geometry_ids, self.graph_sizes)

    def __len__(self):
        return int(self.boundaries[-1])

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        source = int(np.searchsorted(self.boundaries, index, side="right") - 1)
        return self.datasets[source][int(index - self.boundaries[source])]

    def source_manifest(self, selected_indices):
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        result = []
        for source, (path, dataset) in enumerate(zip(self.paths, self.datasets)):
            start, end = self.boundaries[source:source + 2]
            local = selected_indices[(selected_indices >= start) & (selected_indices < end)] - start
            result.append({
                "path": path, "n": dataset.n, "m": dataset.m,
                "loaded_records": len(dataset), "selected_records": len(local),
                "indices_in_file": dataset.indices[local].tolist(),
            })
        return result

    def run_label(self):
        shapes, budgets = set(), []
        for dataset in self.datasets:
            if "net_shape" in dataset.data:
                shapes.update(str(value) for value in dataset.data["net_shape"][dataset.indices])
            else:
                shapes.add("unknown")
            budgets.extend(dataset.data["na"][dataset.indices].tolist())
        shape = next(iter(shapes)) if len(shapes) == 1 else "mixed"
        ns = "-".join(str(n) for n in sorted({n for n, _ in self.shape_groups}))
        ms = "-".join(str(m) for m in sorted({m for _, m in self.shape_groups}))
        return f"{shape}-N{ns}-M{ms}-Na{min(budgets)}-{max(budgets)}"


def check_geometry_sizes(geometry_ids, graph_sizes):
    seen = {}
    for identity, size in zip(geometry_ids, graph_sizes):
        size = tuple(int(value) for value in size)
        if seen.setdefault(str(identity), size) != size:
            raise ValueError("one geometry_id has inconsistent N/M across dataset sources")


def stratified_grouped_split(dataset, validation_fraction=0.1, seed=42):
    """Split jointly across files, stratified by (N,M), grouping all K labels."""
    train, validation = [], []
    for size, indices in dataset.shape_groups.items():
        try:
            local_train, local_val = grouped_split(dataset.geometry_ids[indices], validation_fraction, seed)
        except ValueError as error:
            raise ValueError(f"N={size[0]}, M={size[1]}: {error}") from error
        train.extend(indices[local_train])
        validation.extend(indices[local_val])
    return np.sort(np.asarray(train, dtype=np.int64)), np.sort(np.asarray(validation, dtype=np.int64))


class ShapeSubset(Subset):
    """Subset with graph-size metadata and optional zero-weight validation padding."""

    def __init__(self, dataset, indices):
        super().__init__(dataset, np.asarray(indices, dtype=np.int64).tolist())
        self.graph_sizes = dataset.graph_sizes[self.indices]
        self.geometry_ids = dataset.geometry_ids[self.indices]
        self.shape_groups = shape_groups(self.graph_sizes)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, weight = index
            item = dict(self.dataset[self.indices[index]])
            item["validation_weight"] = torch.tensor(weight, dtype=torch.float32)
            return item
        return self.dataset[self.indices[index]]

    def __getitems__(self, indices):
        # Subset's batched implementation bypasses __getitem__; preserve padding weights.
        return [self[index] for index in indices]


class GraphSizeBatchSampler(Sampler):
    """Shuffle inside each size bucket and then shuffle the resulting batch order.

    batch_size is per rank, before geometry augmentation. All ranks visit the
    same shape at each step. A global tail is padded only to a multiple of the
    world size, so every rank has the same nonzero local batch size. Validation
    padding is flagged with weight zero and must be excluded from statistics.
    """

    def __init__(self, groups, batch_size, *, shuffle, seed=42, rank=0, world_size=1,
                 validation=False):
        if batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid batch size or distributed rank/world size")
        self.groups = {size: np.asarray(indices, dtype=np.int64)
                       for size, indices in sorted(groups.items()) if len(indices)}
        self.batch_size, self.drop_last = int(batch_size), False
        self.shuffle, self.seed, self.epoch = bool(shuffle), int(seed), 0
        self.rank, self.world_size = int(rank), int(world_size)
        self.validation = bool(validation)

    @property
    def sampler(self):
        # Lightning calls dataloader.batch_sampler.sampler.set_epoch(epoch).
        return self

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @property
    def padding_records(self):
        return sum((-len(indices)) % self.world_size for indices in self.groups.values())

    def __len__(self):
        width = self.batch_size * self.world_size
        return sum((len(indices) + width - 1) // width for indices in self.groups.values())

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        width = self.batch_size * self.world_size
        batches = []
        for indices in self.groups.values():
            indices = rng.permutation(indices) if self.shuffle else indices
            for start in range(0, len(indices), width):
                chunk = indices[start:start + width].tolist()
                real_count = len(chunk)
                padding = (-real_count) % self.world_size
                chunk.extend(chunk[i % real_count] for i in range(padding))
                if self.validation:
                    batch = [(chunk[i], float(i < real_count))
                             for i in range(self.rank, len(chunk), self.world_size)]
                else:
                    batch = chunk[self.rank::self.world_size]
                batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches
