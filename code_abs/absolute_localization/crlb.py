"""Full 3x3 per-user TOA FIM inversion, with fixed reference link variances.

For theta_j=(p_x,p_y,delta_seconds), h_ij=(u_x/c,u_y/c,1).
We invert the equivalent FIM for eta_j=(p_x,p_y,c*delta_seconds).
This unit change leaves the position covariance block unchanged and avoids
testing numerical rank across meters and seconds. No EFIM or pseudoinverse
is used. Public selection indices are strictly 1-based.
"""
from __future__ import annotations

import numpy as np
from .params import MeasurePara


def _positions(value, name: str) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f"{name} must be a nonempty finite array of shape [count, 2]")
    return points


def selected_indices(indices, n: int) -> np.ndarray:
    raw = np.asarray(indices)
    if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("selected anchors must be a nonempty 1-D integer array (1-based)")
    if np.any(raw < 1) or np.any(raw > n) or np.unique(raw).size != raw.size:
        raise ValueError(f"selected anchors must be unique indices in 1..{n}")
    return raw.astype(np.int64) - 1


def position_blocks_from_range_fim(fim: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Return [...,2,2] position blocks; singular/non-positive FIMs yield inf."""
    matrices = np.asarray(fim, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError("FIM must be finite and end in [3,3]")
    shape = matrices.shape[:-2]
    flat = matrices.reshape(-1, 3, 3)
    eig = np.linalg.eigvalsh(flat)
    valid = (eig[:, 2] > 0) & (eig[:, 0] > rcond * eig[:, 2])
    blocks = np.full((len(flat), 2, 2), np.inf, dtype=np.float64)
    if np.any(valid):
        blocks[valid] = np.linalg.inv(flat[valid])[:, :2, :2]
    return blocks.reshape(*shape, 2, 2)


class AbsoluteGeometry:
    """Cache per-anchor/user information once, then evaluate many subsets."""

    def __init__(self, anchors, users, measure_para: MeasurePara | None = None):
        self.anchors = _positions(anchors, "anchors")
        self.users = _positions(users, "users")
        self.n, self.m = len(self.anchors), len(self.users)
        self.measure = MeasurePara() if measure_para is None else measure_para
        if not (np.isfinite(self.measure.sigma_d) and self.measure.sigma_d > 0
                and np.isfinite(self.measure.c) and self.measure.c > 0
                and 0 < self.measure.rcond < 1):
            raise ValueError("sigma_d/c must be positive and finite; rcond must lie in (0,1)")
        difference = self.users[None, :, :] - self.anchors[:, None, :]
        self.distances = np.linalg.norm(difference, axis=-1)
        if np.any(self.distances <= 0):
            raise ValueError("An anchor coincides with a user: TOA direction/variance is undefined")
        directions = difference / self.distances[..., None]
        h_range = np.concatenate((directions, np.ones((self.n, self.m, 1))), axis=-1)
        range_variance = (self.measure.sigma_d * self.distances) ** 2
        self.toa_variances = range_variance / self.measure.c ** 2
        self.link_information = np.einsum("nmi,nmj->nmij", h_range, h_range) / range_variance[..., None, None]

    def fim(self, indices, *, clock_units: str = "seconds") -> np.ndarray:
        result = self.link_information[selected_indices(indices, self.n)].sum(axis=0)
        if clock_units == "meters":
            return result
        if clock_units != "seconds":
            raise ValueError("clock_units must be 'seconds' or 'meters'")
        scale = np.array([1.0, 1.0, self.measure.c])
        return result * scale[:, None] * scale[None, :]

    def position_crlb(self, indices) -> np.ndarray:
        return position_blocks_from_range_fim(self.fim(indices, clock_units="meters"), self.measure.rcond)

    def objective(self, indices) -> float:
        """F_abs = mean_j trace(C_j_position), in square meters (already averaged)."""
        return float(np.trace(self.position_crlb(indices), axis1=-2, axis2=-1).mean())

def fim_of_toa(anchors, users, indices, measure_para=None, *, clock_units="seconds"):
    return AbsoluteGeometry(anchors, users, measure_para).fim(indices, clock_units=clock_units)


def fim_crlb_of_toa(anchors, users, indices, measure_para=None) -> float:
    return AbsoluteGeometry(anchors, users, measure_para).objective(indices)


def root_mean_crlb(anchors, users, indices, measure_para=None) -> float:
    """sqrt(mean user position-CRLB trace), in meters; no second division by M."""
    return float(np.sqrt(fim_crlb_of_toa(anchors, users, indices, measure_para)))
