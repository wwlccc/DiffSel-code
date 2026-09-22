from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, MODULE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from relative_localization.crlb import root_mean_crlb
from relative_localization.geometry import node_generation
from relative_localization.params import NetPara
from train_node_activation_diffusion import NodeActivationDiffusionPL


@dataclass
class EvaluationStats:
    values: list[float]

    def summary(self) -> dict[str, float | int]:
        array = np.asarray(self.values, dtype=np.float64)
        return {
            "count": int(array.size),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if array.size > 1 else float("nan"),
            "min": float(array.min()),
            "p25": float(np.percentile(array, 25)),
            "median": float(np.percentile(array, 50)),
            "p75": float(np.percentile(array, 75)),
            "max": float(array.max()),
        }


class GeneratedLocalizationDataset(Dataset):
    def __init__(
        self,
        *,
        num_instances: int,
        n: int,
        na_min: int,
        na_max: int,
        shape: str,
        side_length: float,
        seed: int | None,
    ) -> None:
        if num_instances <= 0:
            raise ValueError("num_instances must be positive")
        if not 1 <= na_min <= na_max <= n:
            raise ValueError("active-node range must satisfy 1 <= na_min <= na_max <= n")
        self.instances_per_na = int(num_instances)
        self.n = int(n)
        self.na_values = list(range(int(na_min), int(na_max) + 1))
        self.records = [(na, local_index) for na in self.na_values for local_index in range(self.instances_per_na)]
        self.net_para = NetPara(shape=shape, side_length=float(side_length))
        rng = np.random.default_rng(seed)
        self.instance_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=len(self.records), dtype=np.uint32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        na, local_index = self.records[index]
        rng = np.random.default_rng(int(self.instance_seeds[index]))
        positions = node_generation(self.n, rng=rng, net_para=self.net_para).astype("float32")
        return {
            "instance_index": torch.tensor(index, dtype=torch.long),
            "na_local_index": torch.tensor(local_index, dtype=torch.long),
            "positions": torch.as_tensor(positions, dtype=torch.float32),
            "na": torch.tensor(na, dtype=torch.long),
            "side_length": torch.tensor(float(self.net_para.side_length), dtype=torch.float32),
        }


def parse_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def parse_active_range(args: argparse.Namespace) -> tuple[int, int]:
    if args.na is not None:
        return int(args.na), int(args.na)
    return int(args.na_min), int(args.na_max)


def load_model(args: argparse.Namespace, device: torch.device) -> NodeActivationDiffusionPL:
    model = NodeActivationDiffusionPL.load_from_checkpoint(
        str(args.ckpt_path),
        map_location=device,
        train_dataset="",
        num_nodes=int(args.n),
        inference_diffusion_steps=args.inference_diffusion_steps,
        inference_schedule=args.inference_schedule,
    )
    model.to(device)
    model.eval()
    return model


def model_active_indices(scores: torch.Tensor, na: torch.Tensor, stochastic: bool) -> list[np.ndarray]:
    selections: list[np.ndarray] = []
    for row, active_count in enumerate(na.detach().cpu().tolist()):
        k = int(active_count)
        if stochastic:
            weights = scores[row].detach().to(dtype=torch.float32).clamp_min(0.0)
            if float(weights.sum().item()) <= 0.0:
                weights = torch.ones_like(weights)
            indices = torch.multinomial(weights + 1.0e-12, num_samples=k, replacement=False)
        else:
            indices = torch.topk(scores[row], k=k, largest=True).indices
        selections.append(indices.detach().cpu().numpy().astype(np.int64) + 1)
    return selections


def print_summary(name: str, stats: EvaluationStats) -> None:
    summary = stats.summary()
    print(
        f"{name}: count={summary['count']}, mean={summary['mean']:.6f}, std={summary['std']:.6f}, "
        f"min={summary['min']:.6f}, p25={summary['p25']:.6f}, "
        f"median={summary['median']:.6f}, p75={summary['p75']:.6f}, max={summary['max']:.6f}"
    )


def print_grouped_summary(name: str, values: list[float], na_values: list[int]) -> None:
    print_summary(name, EvaluationStats(values))
    print(f"{name}_by_Na:")
    for active_count in sorted(set(na_values)):
        grouped = [value for value, na in zip(values, na_values) if na == active_count]
        print_summary(f"  Na={active_count}", EvaluationStats(grouped))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained node-activation checkpoint on generated localization instances.")
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--num_instances", type=int, default=1000, help="number of generated test instances per Na")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--inference_diffusion_steps", type=int, default=50)
    parser.add_argument("--inference_schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--na", type=int, default=None)
    parser.add_argument("--na_min", type=int, default=4)
    parser.add_argument("--na_max", type=int, default=8)
    parser.add_argument("--shape", choices=["square", "circle", "rectangle", "l_shape"], default="square")
    parser.add_argument("--side_length", type=float, default=100.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_instances <= 0:
        raise ValueError("num_instances must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if args.inference_diffusion_steps <= 0:
        raise ValueError("inference_diffusion_steps must be positive")
    if args.n <= 0:
        raise ValueError("n must be positive")
    if args.side_length <= 0:
        raise ValueError("side_length must be positive")
    na_min, na_max = parse_active_range(args)
    if not 1 <= na_min <= na_max <= args.n:
        raise ValueError("active-node range must satisfy 1 <= Na <= N")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    if args.seed is not None:
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    device = parse_device(args.device)
    na_min, na_max = parse_active_range(args)
    dataset = GeneratedLocalizationDataset(
        num_instances=int(args.num_instances),
        n=int(args.n),
        na_min=na_min,
        na_max=na_max,
        shape=args.shape,
        side_length=float(args.side_length),
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=int(args.num_workers) > 0,
    )
    model = load_model(args, device)

    best_model_crlb = [np.inf] * len(dataset)
    instance_na = [0] * len(dataset)
    with torch.no_grad():
        for _ in range(int(args.num_samples)):
            for batch in dataloader:
                na = batch["na"].to(device=device, dtype=torch.long)
                model_batch = {key: value.to(device=device) if torch.is_tensor(value) else value for key, value in batch.items()}
                scores = model.run_diffusion(model_batch)

                selected = model_active_indices(scores, na, stochastic=int(args.num_samples) > 1)
                positions_np = batch["positions"].numpy()
                instance_indices = batch["instance_index"].numpy()
                na_np = batch["na"].numpy().astype(int)
                for local_row, active_indices in enumerate(selected):
                    value = root_mean_crlb(int(args.n), positions_np[local_row], active_indices)
                    dataset_index = int(instance_indices[local_row])
                    instance_na[dataset_index] = int(na_np[local_row])
                    if value < best_model_crlb[dataset_index]:
                        best_model_crlb[dataset_index] = value

    model_values = [float(value) for value in best_model_crlb if np.isfinite(value)]
    if len(model_values) != len(dataset):
        raise RuntimeError("not all generated instances were evaluated")

    print_grouped_summary("model_root_mean_crlb", model_values, instance_na)


if __name__ == "__main__":
    main()
