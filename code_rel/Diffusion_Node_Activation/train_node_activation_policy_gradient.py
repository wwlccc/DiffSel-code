from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, MODULE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from model.egam_node_activation import EGAM_NodeActivation
from relative_localization.geometry import node_generation
from relative_localization.params import MeasurePara, NetPara
from train_node_activation_diffusion import CategoricalDiffusion, InferenceSchedule


class RandomLocalizationDataset(Dataset):
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
        if side_length <= 0:
            raise ValueError("side_length must be positive")

        self.num_instances = int(num_instances)
        self.n = int(n)
        self.na_min = int(na_min)
        self.na_max = int(na_max)
        self.shape = shape
        self.side_length = float(side_length)
        rng = np.random.default_rng(seed)
        self.instance_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=self.num_instances, dtype=np.uint32)

    def __len__(self) -> int:
        return self.num_instances

    def _generate_positions(self, rng: np.random.Generator) -> np.ndarray:
        if self.shape in ("rectangle", "l_shape"):
            return node_generation(
                self.n, rng=rng, net_para=NetPara(shape=self.shape, side_length=self.side_length)
            ).astype("float32")
        if self.shape == "square":
            return np.column_stack(
                (rng.random(self.n) * self.side_length, rng.random(self.n) * self.side_length)
            ).astype("float32")
        if self.shape == "circle":
            center = np.array([self.side_length / 2.0, self.side_length / 2.0], dtype=np.float32)
            radius = np.sqrt(rng.random(self.n)).astype("float32") * (self.side_length / 2.0)
            theta = rng.random(self.n).astype("float32") * (2.0 * np.pi)
            offset = np.column_stack((radius * np.cos(theta), radius * np.sin(theta))).astype("float32")
            return center + offset
        raise ValueError(f"Unsupported shape: {self.shape!r}")

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(int(self.instance_seeds[index]))
        positions = self._generate_positions(rng)
        na = int(rng.integers(self.na_min, self.na_max + 1))
        return {
            "positions": torch.as_tensor(positions, dtype=torch.float32),
            "na": torch.tensor(na, dtype=torch.long),
            "side_length": torch.tensor(self.side_length, dtype=torch.float32),
        }


def build_node_and_edge_features(
    *,
    positions: torch.Tensor,
    na: torch.Tensor,
    side_length: torch.Tensor,
    selection: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.dim() != 3 or positions.size(-1) != 2:
        raise ValueError("positions must have shape [B, N, 2]")
    batch_size, num_nodes, _ = positions.shape
    side_length = side_length.to(device=positions.device, dtype=positions.dtype).reshape(-1)
    if side_length.numel() == 1:
        side_length = side_length.expand(batch_size)
    if torch.any(side_length <= 0):
        raise ValueError("side_length must be positive")
    if selection.shape != (batch_size, num_nodes):
        raise ValueError("selection must have shape [B, N]")

    normalized_positions = positions / side_length[:, None, None]
    edge_features = torch.linalg.norm(positions[:, :, None, :] - positions[:, None, :, :], dim=-1)
    edge_features = edge_features / side_length[:, None, None]
    na_ratio = (na.to(dtype=positions.dtype) / float(num_nodes)).reshape(batch_size, 1, 1).expand(-1, num_nodes, -1)
    count_bias = (selection.sum(dim=1) - na.to(dtype=selection.dtype)).reshape(batch_size, 1, 1).expand(-1, num_nodes, -1)
    node_features = torch.cat([normalized_positions, na_ratio, selection.float().unsqueeze(-1), count_bias], dim=-1)
    return node_features, edge_features


def sym_extend_batch(batch: dict[str, torch.Tensor], num_sym: int) -> dict[str, torch.Tensor]:
    positions = batch["positions"]
    side_length = batch["side_length"].to(device=positions.device, dtype=positions.dtype).reshape(-1)
    if side_length.numel() == 1:
        side_length = side_length.expand(positions.size(0))

    batch_size, num_nodes, _ = positions.shape
    device = positions.device
    dtype = positions.dtype

    transform_index = torch.arange(int(num_sym), device=device)
    rotation_index = transform_index % 4
    theta = rotation_index.to(dtype=dtype) * (math.pi / 2.0)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rotation = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta], dim=-1),
            torch.stack([sin_theta, cos_theta], dim=-1),
        ],
        dim=-2,
    )
    reflection = torch.eye(2, dtype=dtype, device=device).reshape(1, 2, 2).repeat(int(num_sym), 1, 1)
    reflection[transform_index >= 4, 0, 0] = -1.0
    transform = torch.matmul(rotation, reflection)

    center = side_length[:, None, None] * 0.5
    centered = positions - center
    transformed = torch.matmul(transform.reshape(1, int(num_sym), 1, 2, 2), centered[:, None, :, :, None]).squeeze(-1)
    transformed = transformed + center[:, None, :, :]

    extended: dict[str, torch.Tensor] = {
        "positions": transformed.reshape(batch_size * int(num_sym), num_nodes, 2),
        "na": batch["na"].repeat_interleave(int(num_sym), dim=0),
        "side_length": batch["side_length"].repeat_interleave(int(num_sym), dim=0),
    }
    return extended


def bernoulli_sample_and_logp(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = probs.clamp(1.0e-6, 1.0 - 1.0e-6)
    sample = torch.bernoulli(probs)
    return sample, bernoulli_logp(probs, sample)


def bernoulli_logp(probs: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1.0e-6, 1.0 - 1.0e-6)
    logp = sample * torch.log(probs) + (1.0 - sample) * torch.log1p(-probs)
    return logp.sum(dim=1)


def downsample_if_too_many(selection: torch.Tensor, weights: torch.Tensor, na: torch.Tensor) -> torch.Tensor:
    batch_size, num_nodes = selection.shape
    fixed = torch.zeros_like(selection)
    for row in range(batch_size):
        active = torch.nonzero(selection[row] > 0.5, as_tuple=False).flatten()
        active_count = int(active.numel())
        target_count = int(na[row].item())
        if target_count <= 0:
            continue
        if active_count < target_count:
            if active_count > 0:
                fixed[row, active] = 1.0
            inactive = torch.nonzero(fixed[row] < 0.5, as_tuple=False).flatten()
            missing_count = min(target_count - active_count, int(inactive.numel()))
            if missing_count <= 0:
                continue
            inactive_weights = weights[row, inactive].detach().to(dtype=torch.float32).clamp_min(0.0)
            if float(inactive_weights.sum().item()) <= 0.0:
                inactive_weights = torch.ones_like(inactive_weights)
            chosen_local = torch.multinomial(inactive_weights + 1.0e-12, num_samples=missing_count, replacement=False)
            fixed[row, inactive[chosen_local]] = 1.0
            continue
        if active_count == target_count:
            fixed[row, active] = 1.0
            continue
        active_weights = weights[row, active].detach().to(dtype=torch.float32).clamp_min(0.0)
        if float(active_weights.sum().item()) <= 0.0:
            active_weights = torch.ones_like(active_weights)
        chosen_local = torch.multinomial(active_weights + 1.0e-12, num_samples=target_count, replacement=False)
        fixed[row, active[chosen_local]] = 1.0
    return fixed


def batched_root_mean_crlb(
    positions: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    sigma_d: float,
    empty_selection_penalty: float,
) -> torch.Tensor:
    positions = positions.to(dtype=torch.float32)
    batch_size, num_nodes, _ = positions.shape
    dtype = torch.float32
    device = positions.device
    coord_dim = 2 * num_nodes
    state_dim = 3 * num_nodes
    selector = torch.zeros((2 * num_nodes, 3 * num_nodes), dtype=dtype, device=device)
    selector[:, : 2 * num_nodes] = torch.eye(2 * num_nodes, dtype=dtype, device=device)

    src = torch.arange(num_nodes, device=device).repeat_interleave(num_nodes)
    dst = torch.arange(num_nodes, device=device).repeat(num_nodes)
    pair_count = int(src.numel())
    valid_pair = src != dst

    delta = positions[:, src, :] - positions[:, dst, :]
    distance = torch.linalg.norm(delta, dim=-1).clamp_min(1.0e-12)
    unit = delta / distance.unsqueeze(-1)
    scale = 1.0 / (float(sigma_d) * distance)

    cols = torch.zeros((batch_size, pair_count, state_dim), dtype=dtype, device=device)
    batch_idx = torch.arange(batch_size, device=device)[:, None].expand(-1, pair_count)
    pair_idx = torch.arange(pair_count, device=device)[None, :].expand(batch_size, -1)

    cols[batch_idx, pair_idx, 2 * src] = unit[..., 0]
    cols[batch_idx, pair_idx, 2 * src + 1] = unit[..., 1]
    cols[batch_idx, pair_idx, 2 * dst] = -unit[..., 0]
    cols[batch_idx, pair_idx, 2 * dst + 1] = -unit[..., 1]
    cols[batch_idx, pair_idx, coord_dim + dst] = -1.0
    cols[batch_idx, pair_idx, coord_dim + src] = 1.0
    cols = cols * scale.unsqueeze(-1)

    pair_weight = (active_mask[:, src] > 0.5).to(dtype=dtype)
    pair_weight = pair_weight * valid_pair.to(dtype=dtype).reshape(1, pair_count)
    fim = torch.einsum("bpc,bpd,bp->bcd", cols, cols, pair_weight)

    u_nc = torch.zeros((batch_size, state_dim, 4), dtype=dtype, device=device)
    u_nc[:, :coord_dim:2, 0] = 1.0
    u_nc[:, 1:coord_dim:2, 1] = 1.0
    u_nc[:, :coord_dim:2, 2] = -positions[:, :, 1]
    u_nc[:, 1:coord_dim:2, 2] = positions[:, :, 0]
    u_nc[:, coord_dim:, 3] = 1.0
    vh = torch.linalg.svd(u_nc.transpose(1, 2), full_matrices=True).Vh
    u_c = vh[:, 4:, :].transpose(1, 2)

    e = torch.matmul(selector.unsqueeze(0), u_c)
    constrained_fim = torch.matmul(u_c.transpose(1, 2), torch.matmul(fim, u_c))
    constrained_inv, info = torch.linalg.inv_ex(constrained_fim)

    crlb_matrix = torch.matmul(e, torch.matmul(constrained_inv, e.transpose(1, 2)))
    crlb_trace = torch.diagonal(crlb_matrix, dim1=-2, dim2=-1).sum(dim=-1)
    cost = torch.sqrt((crlb_trace / float(num_nodes)).clamp_min(0.0))
    penalty = torch.full_like(cost, float(empty_selection_penalty))
    valid = (info == 0) & torch.isfinite(crlb_trace) & (crlb_trace > 0.0)
    return torch.where(valid, cost, penalty)


class NodeActivationPolicyGradientPL(pl.LightningModule):
    def __init__(
        self,
        *,
        num_nodes: int = 20,
        na_min: int = 4,
        na_max: int = 8,
        shape: str = "square",
        side_length: float = 100.0,
        diffusion_schedule: str = "cosine",
        diffusion_steps: int = 1000,
        inference_diffusion_steps: int | None = 50,
        inference_schedule: str = "linear",
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        feed_forward_hidden: int | None = None,
        normalization: str = "layer",
        batch_size: int = 16,
        num_workers: int = 0,
        train_num_instances: int = 12800,
        learning_rate: float = 1e-4,
        final_learning_rate: float = 1e-5,
        weight_decay: float = 0.0,
        lr_scheduler: str = "cosine",
        seed: int | None = None,
        num_sym: int = 8,
        trajectory_update_epochs: int = 1,
        clip_range: float = 1.0e-4,
        empty_selection_penalty: float = 1.0e3,
        cost_clip: float | None = 10.0,
        sigma_d: float = MeasurePara().sigma_d,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
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
        dataset = RandomLocalizationDataset(
            num_instances=int(self.hparams.train_num_instances),
            n=int(self.hparams.num_nodes),
            na_min=int(self.hparams.na_min),
            na_max=int(self.hparams.na_max),
            shape=self.hparams.shape,
            side_length=float(self.hparams.side_length),
            seed=self.hparams.seed,
        )
        return DataLoader(
            dataset,
            batch_size=int(self.hparams.batch_size),
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
            total_optimizer_steps = int(self.trainer.estimated_stepping_batches) * int(self.hparams.trajectory_update_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_optimizer_steps),
                eta_min=float(self.hparams.final_learning_rate),
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        raise ValueError(f"Unsupported lr_scheduler: {self.hparams.lr_scheduler}")

    def _categorical_posterior_active_prob(
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
        return active_prob.clamp(1.0e-6, 1.0 - 1.0e-6)

    def _step_active_prob(
        self,
        *,
        positions: torch.Tensor,
        na: torch.Tensor,
        side_length: torch.Tensor,
        selection: torch.Tensor,
        t: int,
        target_t: int,
    ) -> torch.Tensor:
        batch_size = positions.shape[0]
        node_features, edge_features = build_node_and_edge_features(
            positions=positions,
            na=na,
            side_length=side_length,
            selection=selection,
        )
        graph_timesteps = torch.full((batch_size,), float(t), dtype=torch.float32, device=self.device)
        logits = self.model(node_features, edge_features, timesteps=graph_timesteps)
        active_prior = torch.softmax(logits, dim=-1) * na.to(dtype=logits.dtype).unsqueeze(-1)
        active_prior = active_prior.clamp(1.0e-6, 1.0 - 1.0e-6)
        x0_pred_prob = torch.stack([1.0 - active_prior, active_prior], dim=-1)
        return self._categorical_posterior_active_prob(
            target_t=target_t,
            t=t,
            x0_pred_prob=x0_pred_prob,
            xt=selection,
            na=na,
        )

    def rollout(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        positions = batch["positions"].to(device=self.device, dtype=torch.float32)
        na = batch["na"].to(device=self.device, dtype=torch.long)
        side_length = batch["side_length"].to(device=self.device, dtype=torch.float32)
        batch_size, num_nodes, _ = positions.shape

        init_prob = (na.to(dtype=positions.dtype) / float(num_nodes)).reshape(batch_size, 1).expand(-1, num_nodes)
        xt = torch.bernoulli(init_prob)
        xt_steps: list[torch.Tensor] = []
        next_xt_steps: list[torch.Tensor] = []
        step_logps: list[torch.Tensor] = []
        step_t: list[int] = []
        step_target_t: list[int] = []

        steps = self.hparams.inference_diffusion_steps
        steps = int(self.diffusion.T if steps is None else steps)
        schedule = InferenceSchedule(
            inference_schedule=self.hparams.inference_schedule,
            T=int(self.diffusion.T),
            inference_T=steps,
        )
        last_active_prob = init_prob

        with torch.no_grad():
            for step_idx in range(steps):
                t1, t2 = schedule(step_idx)
                xt_steps.append(xt)
                step_t.append(int(t1))
                step_target_t.append(int(t2))
                last_active_prob = self._step_active_prob(
                    positions=positions,
                    na=na,
                    side_length=side_length,
                    selection=xt,
                    t=int(t1),
                    target_t=int(t2),
                )
                xt, step_logp = bernoulli_sample_and_logp(last_active_prob)
                next_xt_steps.append(xt)
                step_logps.append(step_logp)

        final_selection = downsample_if_too_many(xt, last_active_prob, na)
        cost = batched_root_mean_crlb(
            positions,
            final_selection,
            sigma_d=float(self.hparams.sigma_d),
            empty_selection_penalty=float(self.hparams.empty_selection_penalty),
        )
        raw_cost = cost
        if self.hparams.cost_clip is not None:
            cost = cost.clamp(max=float(self.hparams.cost_clip))
        return {
            "cost": cost,
            "raw_cost": raw_cost,
            "xt_steps": torch.stack(xt_steps, dim=1),
            "next_xt_steps": torch.stack(next_xt_steps, dim=1),
            "step_t": torch.as_tensor(step_t, dtype=torch.long, device=self.device),
            "step_target_t": torch.as_tensor(step_target_t, dtype=torch.long, device=self.device),
            "logp_steps": torch.stack(step_logps, dim=1),
            "raw_selection": xt,
            "final_selection": final_selection,
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        num_sym = int(self.hparams.num_sym)
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        scheduler = self.lr_schedulers()
        if isinstance(scheduler, list):
            scheduler = scheduler[0] if scheduler else None

        sym_batch = sym_extend_batch(batch, num_sym)
        rollout = self.rollout(sym_batch)
        cost = rollout["cost"]
        raw_cost = rollout["raw_cost"]
        sample_logp_steps = rollout["logp_steps"]
        xt_steps = rollout["xt_steps"]
        next_xt_steps = rollout["next_xt_steps"]
        step_t = rollout["step_t"]
        step_target_t = rollout["step_target_t"]
        final_selection = rollout["final_selection"]
        raw_selection = rollout["raw_selection"]
        na = batch["na"].to(device=self.device, dtype=torch.float32)
        sym_positions = sym_batch["positions"].to(device=self.device, dtype=torch.float32)
        sym_na = sym_batch["na"].to(device=self.device, dtype=torch.long)
        sym_side_length = sym_batch["side_length"].to(device=self.device, dtype=torch.float32)

        cost_group = cost.reshape(-1, num_sym)
        group_mean = cost_group.mean(dim=1, keepdim=True)
        group_std = cost_group.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        advantage = ((cost_group - group_mean) / group_std).detach()

        step_losses: list[torch.Tensor] = []
        ratio_values: list[torch.Tensor] = []
        clipfrac_values: list[torch.Tensor] = []
        approx_kl_values: list[torch.Tensor] = []
        num_steps = int(xt_steps.shape[1])
        for _ in range(int(self.hparams.trajectory_update_epochs)):
            optimizer.zero_grad()
            for step_idx in range(num_steps):
                active_prob = self._step_active_prob(
                    positions=sym_positions,
                    na=sym_na,
                    side_length=sym_side_length,
                    selection=xt_steps[:, step_idx],
                    t=int(step_t[step_idx].item()),
                    target_t=int(step_target_t[step_idx].item()),
                )
                step_logp = bernoulli_logp(active_prob, next_xt_steps[:, step_idx])
                old_step_logp = sample_logp_steps[:, step_idx].detach()
                ratio = torch.exp(step_logp - old_step_logp)
                ratio_group = ratio.reshape(-1, num_sym)
                clipped_ratio_group = torch.clamp(
                    ratio_group,
                    1.0 - float(self.hparams.clip_range),
                    1.0 + float(self.hparams.clip_range),
                )
                unclipped_loss = advantage * ratio_group
                clipped_loss = advantage * clipped_ratio_group
                step_loss = torch.maximum(unclipped_loss, clipped_loss).mean()
                self.manual_backward(step_loss / float(num_steps))
                step_losses.append(step_loss.detach())
                ratio_values.append(ratio.detach().mean())
                clipfrac_values.append((torch.abs(ratio.detach() - 1.0) > float(self.hparams.clip_range)).to(dtype=torch.float32).mean())
                approx_kl_values.append((0.5 * (step_logp.detach() - old_step_logp) ** 2).mean())
            if self.hparams.gradient_clip_val is not None:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=float(self.hparams.gradient_clip_val),
                    gradient_clip_algorithm=self.hparams.gradient_clip_algorithm or "norm",
                )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        loss = torch.stack(step_losses).mean()
        batch_size = int(cost.numel())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/root_mean_crlb", cost.mean(), on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train/raw_root_mean_crlb", raw_cost.mean(), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/group_root_mean_crlb_std", group_std.mean(), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/advantage_std", advantage.std(unbiased=False), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/logp_step", sample_logp_steps.mean(), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/logp_sum", sample_logp_steps.sum(dim=1).mean(), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/ratio", torch.stack(ratio_values).mean(), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/clipfrac", torch.stack(clipfrac_values).mean(), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/approx_kl", torch.stack(approx_kl_values).mean(), on_step=True, on_epoch=True, batch_size=batch_size)
        self.log(
            "train/final_active_count_bias",
            (final_selection.sum(dim=1) - na.repeat_interleave(num_sym, dim=0)).mean(),
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/empty_selection_rate",
            (raw_selection.sum(dim=1) <= 0.0).to(dtype=torch.float32).mean(),
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train node-activation diffusion with policy-gradient CRLB cost.")
    parser.add_argument("--num_nodes", "--n", dest="num_nodes", type=int, default=20)
    parser.add_argument("--na_min", type=int, default=4)
    parser.add_argument("--na_max", type=int, default=8)
    parser.add_argument("--shape", choices=["square", "circle", "rectangle", "l_shape"], default="square")
    parser.add_argument("--side_length", type=float, default=100.0)
    parser.add_argument("--diffusion_schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--inference_diffusion_steps", type=int, default=50)
    parser.add_argument("--inference_schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--feed_forward_hidden", type=int, default=None)
    parser.add_argument("--normalization", choices=["batch", "layer"], default="layer")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_num_instances", type=int, default=12800)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--final_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", choices=["constant", "cosine"], default="cosine")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_sym", type=int, default=8)
    parser.add_argument("--trajectory_update_epochs", type=int, default=1)
    parser.add_argument("--clip_range", type=float, default=1.0e-4)
    parser.add_argument("--empty_selection_penalty", type=float, default=1.0e3)
    parser.add_argument("--cost_clip", type=parse_optional_float, default=10.0)
    parser.add_argument("--sigma_d", type=float, default=MeasurePara().sigma_d)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32")
    parser.add_argument("--gradient_clip_val", type=parse_optional_float, default=None)
    parser.add_argument("--gradient_clip_algorithm", choices=["norm", "value"], default=None)
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


def parse_optional_float(raw_value: str | float | None) -> float | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, float):
        return raw_value
    if raw_value.lower() in {"none", "null", "no"}:
        return None
    return float(raw_value)


def resolve_strategy(parsed_devices: str | int | list[int]) -> str:
    if isinstance(parsed_devices, list) and len(parsed_devices) > 1:
        return "ddp_find_unused_parameters_true"
    if isinstance(parsed_devices, int) and parsed_devices > 1:
        return "ddp_find_unused_parameters_true"
    if parsed_devices == "auto" and torch.cuda.device_count() > 1:
        return "ddp_find_unused_parameters_true"
    return "auto"


def validate_args(args: argparse.Namespace) -> None:
    if args.num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if not 1 <= args.na_min <= args.na_max <= args.num_nodes:
        raise ValueError("active-node range must satisfy 1 <= na_min <= na_max <= num_nodes")
    if args.side_length <= 0:
        raise ValueError("side_length must be positive")
    if args.diffusion_steps <= 0:
        raise ValueError("diffusion_steps must be positive")
    if args.inference_diffusion_steps is not None and args.inference_diffusion_steps <= 0:
        raise ValueError("inference_diffusion_steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.train_num_instances <= 0:
        raise ValueError("train_num_instances must be positive")
    if args.num_epochs <= 0:
        raise ValueError("num_epochs must be positive")
    if args.num_sym < 2:
        raise ValueError("num_sym must be at least 2 for grouped advantage normalization")
    if args.trajectory_update_epochs <= 0:
        raise ValueError("trajectory_update_epochs must be positive")
    if args.clip_range <= 0:
        raise ValueError("clip_range must be positive")
    if args.empty_selection_penalty <= 0:
        raise ValueError("empty_selection_penalty must be positive")
    if args.cost_clip is not None and args.cost_clip <= 0:
        raise ValueError("cost_clip must be positive or None")
    if args.sigma_d <= 0:
        raise ValueError("sigma_d must be positive")
    if args.gradient_clip_val is not None and args.gradient_clip_val <= 0:
        raise ValueError("gradient_clip_val must be positive or None")


def run_name(args: argparse.Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"policy-{args.shape}-N{args.num_nodes}-Na{args.na_min}-{args.na_max}-{timestamp}"


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)
    if args.seed is not None:
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

    model_arg_names = {
        "num_nodes",
        "na_min",
        "na_max",
        "shape",
        "side_length",
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
        "num_workers",
        "train_num_instances",
        "learning_rate",
        "final_learning_rate",
        "weight_decay",
        "lr_scheduler",
        "seed",
        "num_sym",
        "trajectory_update_epochs",
        "clip_range",
        "empty_selection_penalty",
        "cost_clip",
        "sigma_d",
        "gradient_clip_val",
        "gradient_clip_algorithm",
    }
    model = NodeActivationPolicyGradientPL(**{name: getattr(args, name) for name in model_arg_names})
    name = run_name(args)
    run_dir = Path(args.default_root_dir) / name
    checkpoint_kwargs: dict[str, Any] = {"save_last": True, "save_top_k": -1, "every_n_epochs": 1}
    checkpoint_kwargs["dirpath"] = args.checkpoint_dir if args.checkpoint_dir is not None else str(run_dir / "checkpoints")
    logger = CSVLogger(save_dir=str(args.default_root_dir), name=name, version="")
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
