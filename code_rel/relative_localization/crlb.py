from __future__ import annotations

import numpy as np

from .geometry import get_dis
from .params import MeasurePara


def _as_zero_based_indices(index_a: np.ndarray | list[int] | tuple[int, ...], n: int) -> np.ndarray:
    idx = np.asarray(index_a, dtype=int).reshape(-1)
    if idx.size == 0:
        raise ValueError("index_a must contain at least one active node")
    if np.any(idx < 0) or np.any(idx > n):
        raise ValueError(f"active node indices must be in 0..{n - 1} or 1..{n}")
    if np.any(idx == 0):
        zero_based = idx
    else:
        zero_based = idx - 1
    if np.any(zero_based < 0) or np.any(zero_based >= n):
        raise ValueError(f"active node indices out of range for n={n}: {idx.tolist()}")
    return np.unique(zero_based)


def _validate_position_and_measurement(
    n: int,
    p: np.ndarray,
    measure_para: MeasurePara | None,
    dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, MeasurePara]:
    position = np.asarray(p, dtype=dtype)
    if position.shape != (n, 2):
        raise ValueError(f"p must have shape ({n}, 2), got {position.shape}")
    if measure_para is None:
        measure_para = MeasurePara()
    return position, measure_para


def _fim_of_tdoa(n: int, position: np.ndarray, active: np.ndarray, measure_para: MeasurePara) -> np.ndarray:
    dtype = position.dtype
    active_mask = np.zeros(n, dtype=bool)
    active_mask[active] = True

    distance = get_dis(position)
    fim = np.zeros((3 * n, 3 * n), dtype=dtype)

    # Matches the MATLAB column weighting in L * diag(vec(Connet * diag(c))) * L.'
    # with the original i-outer, j-inner manual column order.
    for i in np.flatnonzero(active_mask):
        for j in range(n):
            if i == j:
                continue
            theta_ij = np.arctan2(position[i, 1] - position[j, 1], position[i, 0] - position[j, 0])
            col = np.zeros(3 * n, dtype=dtype)
            col[2 * i] = np.cos(theta_ij)
            col[2 * i + 1] = np.sin(theta_ij)
            col[2 * j] = np.cos(theta_ij - np.pi)
            col[2 * j + 1] = np.sin(theta_ij - np.pi)
            col[2 * n + j] = -1.0
            col[2 * n + i] = 1.0
            col /= measure_para.sigma_d * distance[i, j]
            fim += np.outer(col, col)
    return fim


def _position_selector(n: int) -> np.ndarray:
    selector = np.zeros((2 * n, 3 * n), dtype=np.float64)
    selector[:, : 2 * n] = np.eye(2 * n, dtype=np.float64)
    return selector


def _constraint_nullspace(position: np.ndarray) -> np.ndarray:
    n = position.shape[0]
    dtype = position.dtype
    u_nc = np.column_stack(
        (
            np.r_[np.tile(np.asarray([1.0, 0.0], dtype=dtype), n), np.zeros(n, dtype=dtype)],
            np.r_[np.tile(np.asarray([0.0, 1.0], dtype=dtype), n), np.zeros(n, dtype=dtype)],
            np.r_[np.column_stack((-position[:, 1], position[:, 0])).reshape(-1), np.zeros(n, dtype=dtype)],
            np.r_[np.zeros(2 * n, dtype=dtype), np.ones(n, dtype=dtype)],
        )
    ).astype(dtype, copy=False)
    _, singular_values, vh = np.linalg.svd(u_nc.T, full_matrices=True)
    tolerance = np.finfo(dtype).eps * max(u_nc.T.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    return vh[rank:].T


def fim_crlb_of_tdoa_constrained_inv(
    n: int,
    p: np.ndarray,
    index_a: np.ndarray | list[int] | tuple[int, ...],
    measure_para: MeasurePara | None = None,
    dtype: np.dtype | type = np.float64,
) -> float:
    """Compute constrained CRLB with ``inv(U_C.T @ FIM @ U_C)``."""

    position, measure_para = _validate_position_and_measurement(n, p, measure_para, dtype)
    active = _as_zero_based_indices(index_a, n)
    fim = _fim_of_tdoa(n, position, active, measure_para)
    u_c = _constraint_nullspace(position)
    e = _position_selector(n).astype(dtype, copy=False) @ u_c
    constrained_fim = u_c.T @ fim @ u_c
    return float(np.trace(e @ np.linalg.inv(constrained_fim) @ e.T))


def fim_crlb_of_tdoa(
    n: int,
    p: np.ndarray,
    index_a: np.ndarray | list[int] | tuple[int, ...],
    measure_para: MeasurePara | None = None,
    dtype: np.dtype | type = np.float64,
) -> float:
    """Compute the default CPU CRLB trace with constrained ``inv``."""

    return fim_crlb_of_tdoa_constrained_inv(n, p, index_a, measure_para, dtype)


def root_mean_crlb(
    n: int,
    p: np.ndarray,
    index_a: np.ndarray | list[int] | tuple[int, ...],
    measure_para: MeasurePara | None = None,
    dtype: np.dtype | type = np.float64,
) -> float:
    """Return ``sqrt(FIM_CRLB_of_TDoA(...) / N)``."""

    return float(np.sqrt(fim_crlb_of_tdoa(n, p, index_a, measure_para, dtype) / n))


def root_mean_crlb_constrained_inv(
    n: int,
    p: np.ndarray,
    index_a: np.ndarray | list[int] | tuple[int, ...],
    measure_para: MeasurePara | None = None,
    dtype: np.dtype | type = np.float64,
) -> float:
    """Return ``sqrt(FIM_CRLB_of_TDoA_constrained_inv(...) / N)``."""

    return float(np.sqrt(fim_crlb_of_tdoa_constrained_inv(n, p, index_a, measure_para, dtype) / n))
