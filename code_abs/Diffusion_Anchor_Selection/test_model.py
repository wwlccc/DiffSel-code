"""Basic generated-geometry evaluation of an absolute-localization checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from absolute_localization import AbsoluteGeometry, generate_geometry
from absolute_localization.params import NetPara, MeasurePara
if __package__:
    from .train_anchor_selection_diffusion import AnchorSelectionDiffusionPL
else:
    from train_anchor_selection_diffusion import AnchorSelectionDiffusionPL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--na_min", type=int, default=4)
    parser.add_argument("--na_max", type=int, default=7)
    parser.add_argument("--num_instances", type=int, default=100, help="instances per Na")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--inference_diffusion_steps", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shape", choices=["square", "circle", "rectangle", "l_shape"], default="square")
    parser.add_argument("--side_length", type=float, default=100.0)
    parser.add_argument("--user_side_ratio", "--user-side-ratio", type=float, default=NetPara().user_side_ratio,
                        help="Centered user-square / anchor-square side ratio in (0,1]; square only (default: 0.8)")
    parser.add_argument("--sigma_d", type=float, default=0.015)
    parser.add_argument("--c", type=float, default=299792458.0)
    parser.add_argument("--rcond", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 3 <= args.na_min <= args.na_max <= args.n or min(args.m, args.num_instances, args.batch_size, args.num_samples) <= 0:
        parser.error("invalid counts or selection budget")
    if not np.isfinite(args.user_side_ratio) or not 0 < args.user_side_ratio <= 1:
        parser.error("user_side_ratio must be finite and lie in (0,1]")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    model = AnchorSelectionDiffusionPL.load_from_checkpoint(args.ckpt_path, map_location=device,
        train_dataset="", validation_dataset=None, inference_diffusion_steps=args.inference_diffusion_steps).to(device).eval()
    records = []
    measure = MeasurePara(args.sigma_d, args.c, args.rcond)
    for na in range(args.na_min, args.na_max + 1):
        for start in range(0, args.num_instances, args.batch_size):
            ids = list(range(start, min(args.num_instances, start + args.batch_size)))
            # Same geometry for each Na, controlled by an evaluation seed distinct from training.
            geometries = [generate_geometry(args.n, args.m, np.random.default_rng(args.seed + i),
                                             NetPara(args.shape, args.side_length, args.user_side_ratio)) for i in ids]
            evaluators = [AbsoluteGeometry(a, u, measure) for a, u in geometries]
            batch = {"anchor_positions": torch.tensor(np.stack([a for a, u in geometries]), dtype=torch.float32),
                     "user_positions": torch.tensor(np.stack([u for a, u in geometries]), dtype=torch.float32),
                     "na": torch.full((len(ids),), na, dtype=torch.long),
                     "side_length": torch.full((len(ids),), args.side_length)}
            best = np.full(len(ids), np.inf)
            for _ in range(args.num_samples):
                scores = model.run_diffusion(batch)
                masks = model.decode(scores, batch["na"], stochastic=args.num_samples > 1).cpu().numpy()
                values = np.array([evaluator.objective(np.flatnonzero(mask) + 1) for evaluator, mask in zip(evaluators, masks)])
                best = np.minimum(best, values)
            for i, objective in enumerate(best):
                row = {"instance_id": ids[i], "na": na,
                       "model_crlb": float(objective) if np.isfinite(objective) else None,
                       "model_rmcrlb": float(np.sqrt(objective)) if np.isfinite(objective) else None}
                records.append(row)
    summary = []
    for na in range(args.na_min, args.na_max + 1):
        rows = [r for r in records if r["na"] == na]
        values = [r["model_rmcrlb"] for r in rows if r["model_rmcrlb"] is not None]
        summary.append({"na": na, "metric": "model_rmcrlb", "count": len(rows),
                        "valid": len(values), "infeasible": len(rows)-len(values),
                        "mean": float(np.mean(values)) if values else None,
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    report = {"summary": summary, "records": records}
    print(json.dumps({"summary": summary}, indent=2, allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
