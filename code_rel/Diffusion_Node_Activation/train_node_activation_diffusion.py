from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
except ImportError:  # pragma: no cover
    import lightning.pytorch as pl
    from lightning.pytorch import Trainer
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger

from model.egam_node_activation import EGAM_NodeActivation
from utils.data_utils import load_node_activation_dataset
from utils.augmentation import augment_positions


class CategoricalDiffusion:
    def __init__(self, steps: int, schedule: str) -> None:
        self.T = int(steps)
        if self.T <= 0:
            raise ValueError("diffusion steps must be positive")
        if schedule == "linear":
            beta = np.linspace(1e-4, 2e-2, self.T)
        elif schedule == "cosine":
            alphabar = self._cos_noise(np.arange(0, self.T + 1)) / self._cos_noise(0)
            beta = np.clip(1 - (alphabar[1:] / alphabar[:-1]), None, 0.999)
        else:
            raise ValueError(f"Unsupported diffusion schedule: {schedule}")
        self.beta = beta.astype(np.float64)
        self._q_bar_cache: dict[tuple[int, int], np.ndarray] = {}

    def _cos_noise(self, t: np.ndarray | int) -> np.ndarray | float:
        offset = 0.008
        return np.cos(math.pi * 0.5 * (np.asarray(t) / self.T + offset) / (1 + offset)) ** 2

    def _q_bar_for(self, na: int, num_nodes: int) -> np.ndarray:
        key = (int(na), int(num_nodes))
        cached = self._q_bar_cache.get(key)
        if cached is not None:
            return cached

        active_ratio = float(na) / float(num_nodes)
        inactive_ratio = 1.0 - active_ratio
        beta = self.beta.reshape((-1, 1, 1))
        eye = np.eye(2, dtype=np.float64).reshape((1, 2, 2))
        stationary = np.array(
            [
                [inactive_ratio, active_ratio],
                [inactive_ratio, active_ratio],
            ],
            dtype=np.float64,
        ).reshape((1, 2, 2))
        qs = (1 - beta) * eye + beta * stationary
        q_bar = [np.eye(2, dtype=np.float64)]
        for q in qs:
            q_bar.append(q_bar[-1] @ q)
        self._q_bar_cache[key] = np.stack(q_bar, axis=0)
        return self._q_bar_cache[key]

    def q_bar_for_batch(self, na: torch.Tensor, num_nodes: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        matrices = [self._q_bar_for(int(value), num_nodes) for value in na.detach().cpu().reshape(-1).tolist()]
        return torch.as_tensor(np.stack(matrices, axis=0), dtype=dtype, device=device)

    def sample(self, x0_onehot: torch.Tensor, t: torch.Tensor, na: torch.Tensor, num_nodes: int) -> torch.Tensor:
        q_bar = self.q_bar_for_batch(na, num_nodes, dtype=x0_onehot.dtype, device=x0_onehot.device)
        q_bar_t = q_bar[torch.arange(q_bar.size(0), device=x0_onehot.device), t.long()]
        probs = torch.bmm(x0_onehot.unsqueeze(1), q_bar_t).squeeze(1)
        return torch.bernoulli(probs[:, 1].clamp(0, 1))


class InferenceSchedule:
    def __init__(self, inference_schedule: str, T: int, inference_T: int) -> None:
        self.inference_schedule = inference_schedule
        self.T = int(T)
        self.inference_T = int(inference_T)
        if self.inference_T <= 0:
            raise ValueError("inference_diffusion_steps must be positive")

    def __call__(self, i: int) -> tuple[int, int]:
        if not 0 <= i < self.inference_T:
            raise ValueError(f"inference step {i} outside [0, {self.inference_T})")

        if self.inference_schedule == "linear":
            t1 = self.T - int((float(i) / self.inference_T) * self.T)
            t2 = self.T - int((float(i + 1) / self.inference_T) * self.T)
        elif self.inference_schedule == "cosine":
            t1 = self.T - int(np.sin((float(i) / self.inference_T) * np.pi / 2) * self.T)
            t2 = self.T - int(np.sin((float(i + 1) / self.inference_T) * np.pi / 2) * self.T)
        else:
            raise ValueError(f"Unsupported inference_schedule: {self.inference_schedule}")

        return int(np.clip(t1, 1, self.T)), int(np.clip(t2, 0, self.T - 1))


class NodeActivationDiffusionPL(pl.LightningModule):
    def __init__(
        self,
        *,
        train_dataset: str,
        num_nodes: int = 20,
        diffusion_schedule: str = "cosine",
        diffusion_steps: int = 1000,
        inference_diffusion_steps: int | None = None,
        inference_schedule: str = "linear",
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        feed_forward_hidden: int | None = None,
        normalization: str = "layer",
        batch_size: int = 16,
        num_workers: int = 0,
        learning_rate: float = 1e-4,
        final_learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
        lr_scheduler: str = "constant",
        train_num_instances: int | None = None,
        train_offset: int = 0,
        geometry_augmentation: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.diffusion = CategoricalDiffusion(diffusion_steps, diffusion_schedule)
        self.model = EGAM_NodeActivation(
            embed_dim=embed_dim,
            n_encode_layers=n_layers,
            normalization=normalization,
            n_heads=n_heads,
            feed_forward_hidden=feed_forward_hidden,
            node_dim=5,
            out_channels=1,
        )

    def train_dataloader(self) -> DataLoader:
        dataset = load_node_activation_dataset(
            self.hparams.train_dataset,
            num_instances=self.hparams.train_num_instances,
            offset=int(self.hparams.train_offset),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.hparams.batch_size),
            # Shuffle the full training dataset at the beginning of each epoch.
            shuffle=True,
            num_workers=int(self.hparams.num_workers),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=int(self.hparams.num_workers) > 0,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.hparams.learning_rate),
            weight_decay=float(self.hparams.weight_decay),
        )
        if self.hparams.lr_scheduler == "constant":
            return optimizer
        if self.hparams.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(self.trainer.estimated_stepping_batches)),
                eta_min=float(self.hparams.final_learning_rate),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        raise ValueError(f"Unsupported lr_scheduler: {self.hparams.lr_scheduler}")

    def _sample_graph_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(1, int(self.diffusion.T) + 1, size=(batch_size,), device=self.device)

    def _build_inference_features(
        self,
        batch: dict[str, torch.Tensor],
        selection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = batch["positions"].to(device=self.device, dtype=torch.float32)
        na = batch["na"].to(device=self.device, dtype=torch.long)
        side_length = batch["side_length"].to(device=self.device, dtype=torch.float32).reshape(-1)
        if positions.dim() != 3 or positions.size(-1) != 2:
            raise ValueError("positions must have shape [B, N, 2]")
        batch_size, num_nodes, _ = positions.shape
        if side_length.numel() == 1:
            side_length = side_length.expand(batch_size)
        if torch.any(side_length <= 0):
            raise ValueError("side_length must be positive")
        if selection.shape != (batch_size, num_nodes):
            raise ValueError("selection must have shape [B, N]")

        normalized_positions = positions / side_length[:, None, None]
        edge_features = torch.linalg.norm(positions[:, :, None, :] - positions[:, None, :, :], dim=-1)
        edge_features = edge_features / side_length[:, None, None]
        na_ratio = (na.to(dtype=torch.float32) / float(num_nodes)).reshape(batch_size, 1, 1).expand(-1, num_nodes, -1)
        selection_count_bias = (selection.sum(dim=1) - na.to(dtype=selection.dtype)).reshape(batch_size, 1, 1).expand(-1, num_nodes, -1)
        node_features = torch.cat([normalized_positions, na_ratio, selection.float().unsqueeze(-1), selection_count_bias], dim=-1)
        return node_features, edge_features, na

    def _categorical_posterior(
        self,
        *,
        target_t: int,
        t: int,
        x0_pred_prob: torch.Tensor,
        xt: torch.Tensor,
        na: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_nodes = xt.shape
        q_bar = self.diffusion.q_bar_for_batch(na, num_nodes, dtype=x0_pred_prob.dtype, device=x0_pred_prob.device)
        q_bar_source = q_bar[:, t]
        q_bar_target = q_bar[:, target_t]
        q_t = torch.linalg.solve(q_bar_target, q_bar_source)

        xt_onehot = F.one_hot(xt.long().clamp(0, 1), num_classes=2).to(dtype=x0_pred_prob.dtype)
        target_part = torch.bmm(xt_onehot, q_t.transpose(1, 2).contiguous())

        denom_0 = (q_bar_source[:, 0, :].unsqueeze(1) * xt_onehot).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        target_prob_0 = (target_part * q_bar_target[:, 0, :].unsqueeze(1)) / denom_0
        active_prob = target_prob_0[..., 1] * x0_pred_prob[..., 0]

        denom_1 = (q_bar_source[:, 1, :].unsqueeze(1) * xt_onehot).sum(dim=-1, keepdim=True).clamp_min(1e-12)
        target_prob_1 = (target_part * q_bar_target[:, 1, :].unsqueeze(1)) / denom_1
        active_prob = active_prob + target_prob_1[..., 1] * x0_pred_prob[..., 1]

        if target_t > 0:
            return torch.bernoulli(active_prob.clamp(0, 1))
        return active_prob.clamp(0, 1)

    @torch.no_grad()
    def run_diffusion(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_history: bool = False,
        include_initial_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        positions = batch["positions"].to(device=self.device, dtype=torch.float32)
        na = batch["na"].to(device=self.device, dtype=torch.long)
        batch_size, num_nodes, _ = positions.shape
        # Match the stationary diffusion prior; the initial count equals Na in expectation.
        init_prob = (na.to(dtype=positions.dtype) / float(num_nodes)).reshape(batch_size, 1).expand(-1, num_nodes)
        xt = torch.bernoulli(init_prob)
        history: list[torch.Tensor] = []
        if return_history and include_initial_state:
            history.append(xt.detach().clone())

        steps = self.hparams.inference_diffusion_steps
        steps = int(self.diffusion.T if steps is None else steps)
        schedule = InferenceSchedule(
            inference_schedule=self.hparams.inference_schedule,
            T=int(self.diffusion.T),
            inference_T=steps,
        )

        for step_idx in range(steps):
            t1, t2 = schedule(step_idx)
            node_features, edge_features, na = self._build_inference_features(batch, xt)
            graph_timesteps = torch.full((batch_size,), float(t1), dtype=torch.float32, device=self.device)
            logits = self.model(node_features, edge_features, timesteps=graph_timesteps)
            active_prior = torch.softmax(logits, dim=-1) * na.to(dtype=logits.dtype).unsqueeze(-1)
            active_prior = active_prior.clamp(0, 1)
            x0_pred_prob = torch.stack([1.0 - active_prior, active_prior], dim=-1)
            xt = self._categorical_posterior(
                target_t=t2,
                t=t1,
                x0_pred_prob=x0_pred_prob,
                xt=xt,
                na=na,
            )
            if return_history:
                history.append(xt.float().detach().clone())

        if return_history:
            return xt.float(), history
        return xt.float()

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        target = batch["target"].to(device=self.device, dtype=torch.long)
        positions = batch["positions"].to(device=self.device, dtype=torch.float32)
        na = batch["na"].to(device=self.device, dtype=torch.long)
        side = batch["side_length"].to(device=self.device, dtype=torch.float32)
        if self.hparams.geometry_augmentation:
            positions = augment_positions(positions, side)
            side = side.reshape(-1).expand(len(target)).repeat(2)
            target = torch.cat((target, target), dim=0)
            na = torch.cat((na, na), dim=0)
        batch_size, num_nodes = target.shape
        if int(self.hparams.num_nodes) > 0 and num_nodes != int(self.hparams.num_nodes):
            raise ValueError("batch num_nodes does not match configured num_nodes")

        graph_timesteps = self._sample_graph_timesteps(batch_size)
        clean_onehot = F.one_hot(target.reshape(-1), num_classes=2).float()
        node_na = torch.repeat_interleave(na, num_nodes, dim=0)
        node_timesteps = torch.repeat_interleave(graph_timesteps, num_nodes, dim=0)
        noisy_selection = self.diffusion.sample(clean_onehot, node_timesteps, node_na, num_nodes).reshape(batch_size, num_nodes)
        # Rebuild from the current geometry; cached clean features describe the original view only.
        node_features, edge_features, na = self._build_inference_features(
            {"positions": positions, "na": na, "side_length": side}, noisy_selection)

        logits = self.model(node_features, edge_features, timesteps=graph_timesteps.float())
        selection_scores = torch.softmax(logits, dim=-1) * na.to(dtype=logits.dtype).unsqueeze(-1)
        loss = F.mse_loss(selection_scores, target.to(dtype=selection_scores.dtype), reduction="mean")
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train categorical diffusion for active-node selection.")
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--train_num_instances", type=int, default=None)
    parser.add_argument("--train_offset", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--diffusion_schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_diffusion_steps", type=int, default=None)
    parser.add_argument("--inference_schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--feed_forward_hidden", type=int, default=None)
    parser.add_argument("--normalization", choices=["batch", "layer"], default="layer")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_geometry_augmentation", dest="geometry_augmentation", action="store_false",
                        help="Disable default supervised B-to-2B rotation/reflection augmentation")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--final_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32")
    parser.add_argument("--default_root_dir", default="Diffusion_Node_Activation/outputs")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    return parser


def parse_devices(raw_devices: str) -> str | int | list[int]:
    if raw_devices == "auto":
        return "auto"
    if "," in raw_devices:
        return [int(item.strip()) for item in raw_devices.split(",") if item.strip()]
    try:
        return int(raw_devices)
    except ValueError:
        return raw_devices


def parse_precision(raw_precision: str) -> int | str:
    if raw_precision in {"16", "32", "64"}:
        return int(raw_precision)
    return raw_precision


def resolve_strategy(parsed_devices: str | int | list[int]) -> str:
    if isinstance(parsed_devices, list) and len(parsed_devices) > 1:
        return "ddp_find_unused_parameters_true"
    if isinstance(parsed_devices, int) and parsed_devices > 1:
        return "ddp_find_unused_parameters_true"
    if parsed_devices == "auto" and torch.cuda.device_count() > 1:
        return "ddp_find_unused_parameters_true"
    return "auto"


def dataset_run_name(dataset_path: str, num_nodes: int) -> str:
    with np.load(dataset_path, allow_pickle=True) as data:
        shape = "unknown"
        if "net_shape" in data and data["net_shape"].size > 0:
            shape = str(data["net_shape"][0])
        n = int(data["n"]) if "n" in data else int(num_nodes)
        na_values = data["na"].astype(int)
        na_min = int(na_values.min())
        na_max = int(na_values.max())
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{shape}-N{n}-Na{na_min}-{na_max}-{timestamp}"


def main() -> None:
    args = build_arg_parser().parse_args()
    model_arg_names = {
        "train_dataset",
        "train_num_instances",
        "train_offset",
        "num_nodes",
        "diffusion_schedule",
        "diffusion_steps",
        "inference_diffusion_steps",
        "inference_schedule",
        "embed_dim",
        "n_layers",
        "n_heads",
        "feed_forward_hidden",
        "normalization",
        "batch_size",
        "geometry_augmentation",
        "num_workers",
        "learning_rate",
        "final_learning_rate",
        "weight_decay",
        "lr_scheduler",
    }
    model = NodeActivationDiffusionPL(**{name: getattr(args, name) for name in model_arg_names})
    run_name = dataset_run_name(args.train_dataset, int(args.num_nodes))
    run_dir = Path(args.default_root_dir) / run_name
    checkpoint_kwargs: dict[str, Any] = {"save_last": True, "save_top_k": -1, "every_n_epochs": 1}
    checkpoint_kwargs["dirpath"] = args.checkpoint_dir if args.checkpoint_dir is not None else str(run_dir / "checkpoints")
    logger = CSVLogger(save_dir=str(args.default_root_dir), name=run_name, version="")
    print(f"Run output directory: {run_dir}")
    devices = parse_devices(args.devices)
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=devices,
        strategy=resolve_strategy(devices),
        precision=parse_precision(args.precision),
        max_epochs=int(args.num_epochs),
        default_root_dir=str(run_dir),
        logger=logger,
        callbacks=[LearningRateMonitor(logging_interval="step"), ModelCheckpoint(**checkpoint_kwargs)],
        log_every_n_steps=int(args.log_every_n_steps),
    )
    trainer.fit(model)


if __name__ == "__main__":
    main()
