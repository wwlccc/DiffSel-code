"""Budget-conditioned binary refresh diffusion and exact skip-step posterior."""
from __future__ import annotations

import numpy as np
import torch


class CategoricalDiffusion:
    def __init__(self, steps=1000, schedule="cosine"):
        self.T = int(steps)
        if self.T <= 0:
            raise ValueError("diffusion steps must be positive")
        if schedule == "cosine":
            alpha = np.cos((np.arange(self.T + 1) / self.T + 0.008) / 1.008 * np.pi / 2) ** 2
            alpha /= alpha[0]
            self.beta = np.minimum(1 - alpha[1:] / alpha[:-1], 0.999)
        elif schedule == "linear":
            self.beta = np.linspace(1e-4, 0.02, self.T)
        else:
            raise ValueError(f"unknown schedule {schedule}")
        self.log_alpha_bar = np.r_[0.0, np.cumsum(np.log1p(-self.beta))]

    @staticmethod
    def _prior(na, n, dtype, device):
        if n <= 0:
            raise ValueError("candidate count must be positive")
        budget = na.to(device=device, dtype=dtype).reshape(-1)
        if not torch.all(torch.isfinite(budget) & (budget >= 0) & (budget <= n)):
            raise ValueError("invalid budgets")
        return budget / n

    def sample_prior(self, na, n, *, device=None, dtype=torch.float32):
        device = na.device if device is None else device
        rho = self._prior(na, n, dtype, device)
        return torch.bernoulli(rho[:, None].expand(-1, n))

    def sample(self, clean, timesteps, na):
        batch, n = clean.shape
        timesteps = timesteps.to(device=clean.device, dtype=torch.long).reshape(-1)
        if len(timesteps) != batch or torch.any((timesteps < 0) | (timesteps > self.T)):
            raise ValueError("one timestep in 0..T is required per graph")
        rho = self._prior(na, n, clean.dtype, clean.device)
        alpha = torch.as_tensor(np.exp(self.log_alpha_bar), device=clean.device, dtype=clean.dtype)[timesteps]
        probability = alpha[:, None] * clean + (1 - alpha[:, None]) * rho[:, None]
        return torch.bernoulli(probability.clamp(0, 1))

    @staticmethod
    def _transition(rho, alpha):
        prior = torch.stack((1 - rho, rho), dim=-1)[:, None, :].expand(-1, 2, -1)
        return alpha * torch.eye(2, dtype=rho.dtype, device=rho.device)[None, :, :] + (1 - alpha) * prior

    def reverse_prob(self, xt, clean_probability, na, *, t: int, target_t: int):
        """Mixture over clean bit 0/1; interval alpha avoids inverting near-rank-1 Qbar."""
        if not 0 <= target_t < t <= self.T:
            raise ValueError("reverse times must satisfy 0 <= target_t < t <= T")
        if xt.shape != clean_probability.shape or xt.ndim != 2:
            raise ValueError("xt/clean_probability must have identical [B,N] shape")
        rho = self._prior(na, xt.shape[1], clean_probability.dtype, xt.device)
        source = self._transition(rho, float(np.exp(self.log_alpha_bar[t])))
        target = self._transition(rho, float(np.exp(self.log_alpha_bar[target_t])))
        interval = self._transition(rho, float(np.exp(self.log_alpha_bar[t] - self.log_alpha_bar[target_t])))
        # For each possible previous bit k: Q_(r,t)[k, observed x_t].
        likelihood = (1 - xt[..., None]) * interval[:, None, :, 0] + xt[..., None] * interval[:, None, :, 1]
        probability = torch.zeros_like(clean_probability)
        for z, weight in ((0, 1 - clean_probability), (1, clean_probability)):
            denominator = (1 - xt) * source[:, z, 0][:, None] + xt * source[:, z, 1][:, None]
            posterior_active = likelihood[..., 1] * target[:, z, 1][:, None] / denominator.clamp_min(1e-12)
            probability = probability + weight * posterior_active
        return probability.clamp(0, 1)


def inference_schedule(total_steps: int, reverse_steps: int, kind="linear") -> list[tuple[int, int]]:
    if not 1 <= reverse_steps <= total_steps:
        raise ValueError("reverse_steps must be in 1..total_steps")
    grid = np.linspace(0, 1, reverse_steps + 1)
    if kind == "linear":
        raw = np.rint(total_steps * (1 - grid)).astype(int)
    elif kind == "cosine":
        raw = np.rint(total_steps * (1 - np.sin(grid * np.pi / 2))).astype(int)
    else:
        raise ValueError("inference schedule must be linear or cosine")
    times = [total_steps]
    for i in range(1, reverse_steps):
        times.append(int(np.clip(raw[i], reverse_steps - i, times[-1] - 1)))
    times.append(0)
    return list(zip(times[:-1], times[1:]))
