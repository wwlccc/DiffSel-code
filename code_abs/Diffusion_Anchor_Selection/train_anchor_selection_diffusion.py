from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

if __package__:
    from .model import EGAM_AnchorSelection
    from .utils.data_utils import build_graph_features
    from .utils.multi_dataset import MultiAnchorSelectionDataset, ShapeSubset, GraphSizeBatchSampler, stratified_grouped_split
    from .utils.diffusion import CategoricalDiffusion, inference_schedule
    from .utils.augmentation import augment_positions
else:
    from model import EGAM_AnchorSelection
    from utils.data_utils import build_graph_features
    from utils.multi_dataset import MultiAnchorSelectionDataset, ShapeSubset, GraphSizeBatchSampler, stratified_grouped_split
    from utils.diffusion import CategoricalDiffusion, inference_schedule
    from utils.augmentation import augment_positions


class AnchorSelectionDiffusionPL(pl.LightningModule):
    def __init__(self, *, train_dataset="", validation_dataset=None, num_anchors=None, num_users=None,
                 diffusion_steps=1000, diffusion_schedule="cosine", inference_diffusion_steps=50,
                 inference_schedule="linear", embed_dim=128, n_layers=4, n_heads=8,
                 feed_forward_hidden=None, normalization="layer", batch_size=16, num_workers=0,
                 learning_rate=1e-4, final_learning_rate=1e-5, weight_decay=0.0,
                 lr_scheduler="cosine", train_num_instances=None, train_offset=0,
                 val_fraction=0.1, seed=42, geometry_augmentation=True):
        super().__init__()
        self.save_hyperparameters()
        if batch_size <= 0 or num_workers < 0 or learning_rate <= 0 or not 0 <= val_fraction < 1:
            raise ValueError("invalid batch/worker/learning-rate/validation settings")
        self.model = EGAM_AnchorSelection(embed_dim, n_layers, n_heads, feed_forward_hidden, normalization)
        self.diffusion = CategoricalDiffusion(diffusion_steps, diffusion_schedule)
        self.train_data = self.validation_data = None
        self.split_manifest = None
        self.run_label = None

    def _check_configured_sizes(self, dataset):
        for n, m in dataset.shape_groups:
            if self.hparams.num_anchors is not None and n != self.hparams.num_anchors:
                raise ValueError(f"dataset N={n} differs from explicit num_anchors={self.hparams.num_anchors}; "
                                 "omit num_anchors for mixed-N training")
            if self.hparams.num_users is not None and m != self.hparams.num_users:
                raise ValueError(f"dataset M={m} differs from explicit num_users={self.hparams.num_users}; "
                                 "omit num_users for mixed-M training")

    def setup(self, stage=None):
        if stage not in (None, "fit", "validate") or self.train_data is not None:
            return
        dataset = MultiAnchorSelectionDataset(self.hparams.train_dataset,
            num_instances=self.hparams.train_num_instances, offset=self.hparams.train_offset)
        self._check_configured_sizes(dataset)
        self.run_label = dataset.run_label()
        if self.hparams.validation_dataset:
            validation = MultiAnchorSelectionDataset(self.hparams.validation_dataset)
            self._check_configured_sizes(validation)
            if set(dataset.geometry_ids) & set(validation.geometry_ids):
                raise ValueError("training and validation share geometry instances")
            train_indices, val_indices = np.arange(len(dataset)), np.arange(len(validation))
        else:
            validation = dataset
            train_indices, val_indices = stratified_grouped_split(dataset, self.hparams.val_fraction, self.hparams.seed)
        self.train_data = ShapeSubset(dataset, train_indices)
        self.validation_data = ShapeSubset(validation, val_indices) if len(val_indices) else None
        self.split_manifest = {
            "schema": "absolute_multi_dataset_split_v1",
            "train_datasets": dataset.paths,
            "validation_datasets": validation.paths,
            "validation_mode": "external" if self.hparams.validation_dataset else "grouped_split",
            "seed": self.hparams.seed, "train_slice_offset": self.hparams.train_offset,
            "train_num_instances_per_file": self.hparams.train_num_instances,
            "train_sources": dataset.source_manifest(train_indices),
            "validation_sources": validation.source_manifest(val_indices),
            "train_geometry_ids": sorted(set(dataset.geometry_ids[train_indices].tolist())),
            "validation_geometry_ids": sorted(set(validation.geometry_ids[val_indices].tolist())),
            "train_size_counts": self._size_counts(self.train_data),
            "validation_size_counts": self._size_counts(self.validation_data),
            "batch_size_before_augmentation_per_rank": self.hparams.batch_size,
            "geometry_augmentation": self.hparams.geometry_augmentation,
            "sampling": "each record once before minimal distributed tail padding; no size balancing",
        }

    @staticmethod
    def _size_counts(dataset):
        if dataset is None:
            return []
        return [{"n": n, "m": m, "records": len(indices),
                 "geometries": len(set(dataset.geometry_ids[indices]))}
                for (n, m), indices in dataset.shape_groups.items()]

    def _loader(self, dataset, shuffle):
        trainer = self._trainer
        rank, world_size = (trainer.global_rank, trainer.world_size) if trainer is not None else (0, 1)
        sampler = GraphSizeBatchSampler(dataset.shape_groups, self.hparams.batch_size,
            shuffle=shuffle, seed=self.hparams.seed, rank=rank, world_size=world_size, validation=not shuffle)
        sampler.set_epoch(self.current_epoch)
        return DataLoader(dataset, batch_sampler=sampler,
            num_workers=self.hparams.num_workers, persistent_workers=self.hparams.num_workers > 0,
            pin_memory=self.device.type == "cuda")

    def train_dataloader(self):
        return self._loader(self.train_data, True)

    def val_dataloader(self):
        return [] if self.validation_data is None else self._loader(self.validation_data, False)

    def on_train_start(self):
        world_size = self.trainer.world_size
        self.split_manifest["world_size"] = world_size
        for stage, dataset in (("train", self.train_data), ("validation", self.validation_data)):
            self.split_manifest[f"{stage}_padding_records_per_pass"] = (
                sum((-len(indices)) % world_size for indices in dataset.shape_groups.values())
                if dataset is not None else 0)
        if self.trainer.is_global_zero and isinstance(self.logger, CSVLogger):
            (Path(self.logger.log_dir) / "split.json").write_text(
                json.dumps(self.split_manifest, indent=2) + "\n", encoding="utf-8")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)
        if self.hparams.lr_scheduler == "constant":
            return optimizer
        if self.hparams.lr_scheduler != "cosine":
            raise ValueError("lr_scheduler must be constant or cosine")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
            T_max=max(1, int(self.trainer.estimated_stepping_batches)), eta_min=self.hparams.final_learning_rate)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _geometry_tensors(self, batch):
        return (
            batch["anchor_positions"].to(device=self.device, dtype=torch.float32),
            batch["user_positions"].to(device=self.device, dtype=torch.float32),
            batch["na"].to(device=self.device, dtype=torch.long),
            batch["side_length"].to(device=self.device, dtype=torch.float32),
        )

    def prediction_loss(self, batch, *, augment=False, per_graph=False):
        anchors, users, na, side = self._geometry_tensors(batch)
        clean = batch["target"].to(device=self.device, dtype=torch.float32)
        if clean.shape != anchors.shape[:2]:
            raise ValueError("target must contain one bit per candidate anchor")
        if augment:
            batch_size, n = anchors.shape[:2]
            positions = augment_positions(torch.cat((anchors, users), dim=1), side)
            anchors, users = positions[:, :n], positions[:, n:]
            clean = torch.cat((clean, clean), dim=0)
            na = torch.cat((na, na), dim=0)
            side = side.reshape(-1).expand(batch_size).repeat(2)
        timesteps = torch.randint(1, self.diffusion.T + 1, (len(anchors),), device=self.device)
        noisy = self.diffusion.sample(clean, timesteps, na)
        features = build_graph_features(anchors, users, na, noisy, side)
        logits = self.model(*features, timesteps=timesteps)
        selection_scores = torch.softmax(logits, dim=-1) * na.to(logits.dtype)[:, None]
        if per_graph:
            return F.mse_loss(selection_scores, clean.to(selection_scores.dtype), reduction="none").mean(-1)
        return F.mse_loss(selection_scores, clean.to(selection_scores.dtype), reduction="mean")

    def training_step(self, batch, batch_idx):
        augment = bool(self.hparams.geometry_augmentation)
        loss = self.prediction_loss(batch, augment=augment)
        effective_batch_size = len(batch["na"]) * (2 if augment else 1)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=effective_batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        losses = self.prediction_loss(batch, per_graph=True)
        weight = batch.get("validation_weight", torch.ones_like(losses)).to(losses)
        count = int(weight.sum().item())
        loss = (losses * weight).sum() / max(count, 1)
        # Zero-weight DDP padding does not contribute to either numerator or count.
        # All ranks visit identical sizes in identical order, including tiny tails.
        n, m = batch["anchor_positions"].shape[1], batch["user_positions"].shape[1]
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=count, sync_dist=True)
        self.log(f"val/loss_N{n}_M{m}", loss, on_step=False, on_epoch=True, batch_size=count, sync_dist=True)

    @torch.no_grad()
    def run_diffusion(self, batch, *, reverse_steps=None):
        anchors, users, na, side = self._geometry_tensors(batch)
        n = anchors.shape[1]
        xt = self.diffusion.sample_prior(na, n)
        steps = self.hparams.inference_diffusion_steps if reverse_steps is None else reverse_steps
        for t, target_t in inference_schedule(self.diffusion.T, int(steps), self.hparams.inference_schedule):
            features = build_graph_features(anchors, users, na, xt, side)
            timesteps = torch.full((len(anchors),), t, dtype=torch.long, device=self.device)
            logits = self.model(*features, timesteps=timesteps)
            clean_probability = (torch.softmax(logits, dim=-1) * na.to(logits.dtype)[:, None]).clamp(0, 1)
            probability = self.diffusion.reverse_prob(xt, clean_probability.float(), na, t=t, target_t=target_t)
            xt = torch.bernoulli(probability) if target_t > 0 else probability
        return xt

    @staticmethod
    def decode(scores, na, *, stochastic=False):
        """Return 0/1 [B,N] anchor masks with exactly the prescribed counts."""
        masks = torch.zeros_like(scores)
        for row, budget in enumerate(na.detach().cpu().tolist()):
            k = int(budget)
            if not 0 <= k <= scores.shape[1]:
                raise ValueError("invalid decoding budget")
            if k == 0:
                continue
            indices = (torch.multinomial(scores[row].clamp_min(0) + 1e-12, k, replacement=False)
                       if stochastic else scores[row].topk(k).indices)
            masks[row, indices] = 1
        return masks


def parse_devices(value):
    if value == "auto":
        return value
    return [int(x) for x in value.split(",")] if "," in value else int(value)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Supervised diffusion for absolute-localization anchor selection.")
    parser.add_argument("--train_dataset", nargs="+", required=True, help="One or more absolute_toa_v1 NPZ files")
    parser.add_argument("--validation_dataset", nargs="+", help="Optional independent validation NPZ files")
    parser.add_argument("--num_anchors", type=int, help="Optional fixed N assertion; omit for mixed N")
    parser.add_argument("--num_users", type=int, help="Optional fixed M assertion; omit for mixed M")
    parser.add_argument("--train_num_instances", type=int, help="Maximum records read from EACH training file")
    parser.add_argument("--train_offset", type=int, default=0, help="Record offset within EACH training file")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--diffusion_schedule", choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--inference_diffusion_steps", type=int, default=50)
    parser.add_argument("--inference_schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--feed_forward_hidden", type=int)
    parser.add_argument("--normalization", choices=["layer", "batch"], default="layer")
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
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--default_root_dir", type=Path, default=Path("Diffusion_Anchor_Selection/outputs"))
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--resume_from")
    return parser


def main():
    args = build_arg_parser().parse_args()
    pl.seed_everything(args.seed, workers=True)
    excluded = {"num_epochs", "accelerator", "devices", "precision", "default_root_dir", "log_every_n_steps", "resume_from"}
    model = AnchorSelectionDiffusionPL(**{k: v for k, v in vars(args).items() if k not in excluded})
    model.setup("fit")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    # Lightning's DDP children inherit this name when re-executing the script.
    name = os.environ.setdefault("DIFFSEL_ABS_RUN_NAME",
        f"{model.run_label}-{timestamp}")
    run_dir = args.default_root_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=str(args.default_root_dir), name=name, version="")
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0:
        # Initialize the logger before writing our manifest into its directory.
        _ = logger.experiment
        (run_dir / "split.json").write_text(json.dumps(model.split_manifest, indent=2) + "\n", encoding="utf-8")
    checkpoint = ModelCheckpoint(dirpath=str(run_dir / "checkpoints"), save_last=True, save_top_k=-1, every_n_epochs=1)
    devices = parse_devices(args.devices)
    multi = ((isinstance(devices, list) and len(devices) > 1)
             or (isinstance(devices, int) and devices > 1)
             or (devices == "auto" and args.accelerator != "cpu" and torch.cuda.device_count() > 1))
    precision = {"16": "16-mixed", "32": "32-true", "64": "64-true"}.get(args.precision, args.precision)
    trainer = Trainer(accelerator=args.accelerator, devices=devices, precision=precision,
        strategy="ddp_find_unused_parameters_true" if multi else "auto", max_epochs=args.num_epochs,
        use_distributed_sampler=False,
        default_root_dir=str(run_dir), logger=logger, callbacks=[LearningRateMonitor(logging_interval="step"), checkpoint],
        log_every_n_steps=args.log_every_n_steps, limit_val_batches=1.0 if model.validation_data is not None else 0)
    print(f"Run directory: {run_dir}; train={len(model.train_data)}, val={len(model.validation_data) if model.validation_data else 0}")
    trainer.fit(model, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
