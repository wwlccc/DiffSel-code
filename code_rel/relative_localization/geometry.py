from __future__ import annotations

import numpy as np

from .params import NetPara


def get_dis(position: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix for an ``(N, 2)`` position array."""

    p = np.asarray(position, dtype=float)
    diff = p[:, None, :] - p[None, :, :]
    return np.linalg.norm(diff, axis=2)


def node_generation(node_num: int, rng: np.random.Generator | None = None, net_para: NetPara | None = None) -> np.ndarray:
    """Generate node positions in the configured deployment region.

    ``square`` samples uniformly in ``[0, side_length] x [0, side_length]``.
    ``circle`` samples uniformly inside the inscribed circle of that square.
    ``rectangle`` samples uniformly in ``[0, side_length/2] x [0, side_length]``.
    ``l_shape`` samples uniformly in the square where ``x < side_length/2``
    or ``y < side_length/2``.
    For the default ``side_length=100``, the circle is
    ``(x - 50)^2 + (y - 50)^2 <= 50^2``.
    """

    if rng is None:
        rng = np.random.default_rng()
    if net_para is None:
        net_para = NetPara()

    side_length = float(getattr(net_para, "side_length", 0.0))
    if side_length <= 0:
        raise ValueError(f"net_para.side_length must be positive, got {side_length}")

    if net_para.shape == "square":
        return np.column_stack((rng.random(node_num) * side_length, rng.random(node_num) * side_length))

    if net_para.shape == "rectangle":
        return rng.random((node_num, 2)) * np.array([side_length / 2.0, side_length])

    if net_para.shape == "l_shape":
        # Three equal-area squares, excluding the upper-right quadrant.
        offsets = np.array([[0, 0], [1, 0], [0, 1]])
        cells = rng.integers(0, 3, size=node_num)
        return (offsets[cells] + rng.random((node_num, 2))) * (side_length / 2.0)

    if net_para.shape == "circle":
        center = np.array([side_length / 2.0, side_length / 2.0], dtype=float)
        boundary_radius = side_length / 2.0
        radius = np.sqrt(rng.random(node_num)) * boundary_radius
        theta = rng.random(node_num) * 2.0 * np.pi
        offset = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
        return center + offset

    raise ValueError(f"Unsupported network shape: {net_para.shape!r}")
