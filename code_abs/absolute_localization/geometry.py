from __future__ import annotations

import numpy as np
from .params import NetPara


def node_generation(node_num: int, rng: np.random.Generator | None = None,
                    net_para: NetPara | None = None) -> np.ndarray:
    """Uniform points in square, circle, rectangle, or three-quadrant L region."""
    if node_num <= 0:
        raise ValueError("node_num must be positive")
    rng = np.random.default_rng() if rng is None else rng
    net_para = NetPara() if net_para is None else net_para
    side = float(net_para.side_length)
    if not np.isfinite(side) or side <= 0:
        raise ValueError("side_length must be finite and positive")
    if net_para.shape == "square":
        return rng.random((node_num, 2)) * side
    if net_para.shape == "rectangle":
        return rng.random((node_num, 2)) * [side / 2, side]
    if net_para.shape == "l_shape":
        offsets = np.array([[0, 0], [1, 0], [0, 1]])
        return (offsets[rng.integers(3, size=node_num)] + rng.random((node_num, 2))) * side / 2
    if net_para.shape == "circle":
        radius = np.sqrt(rng.random(node_num)) * side / 2
        angle = rng.random(node_num) * 2 * np.pi
        return side / 2 + np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    raise ValueError(f"Unsupported shape: {net_para.shape}")


def generate_geometry(n: int = 20, m: int = 5, rng: np.random.Generator | None = None,
                      net_para: NetPara | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample independent nodes; square users occupy a centered inner square.

    The default L=100 and user_side_ratio=0.8 give users in [10,90]^2.
    Other region shapes retain their original anchor/user distributions.
    """
    rng = np.random.default_rng() if rng is None else rng
    net_para = NetPara() if net_para is None else net_para
    anchors = node_generation(n, rng, net_para)
    users = node_generation(m, rng, net_para)
    if net_para.shape == "square" and net_para.user_side_ratio < 1:
        margin = net_para.side_length * (1 - net_para.user_side_ratio) / 2
        users = margin + net_para.user_side_ratio * users
    return anchors, users


def get_dis(position: np.ndarray) -> np.ndarray:
    points = np.asarray(position, dtype=np.float64)
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
