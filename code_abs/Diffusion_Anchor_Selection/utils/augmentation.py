"""One original and one random D4 view per supervised geometry."""
import torch


def augment_positions(positions, side_length, *, transform_indices=None):
    """Return [original B; transformed B] without changing node order.

    D4 IDs match the PG convention: rotation by (id % 4)*90 degrees,
    preceded by x reflection for id >= 4, about (L/2,L/2). Sample one
    nonidentity ID (1..7) independently per graph. All nodes belonging to
    a geometry must be passed together (including both anchors and users).
    """
    if positions.ndim != 3 or positions.shape[-1] != 2 or not positions.is_floating_point():
        raise ValueError("positions must be floating-point [B,N,2]")
    batch_size = positions.shape[0]
    side = torch.as_tensor(side_length, device=positions.device, dtype=positions.dtype).reshape(-1)
    if side.numel() == 1:
        side = side.expand(batch_size)
    if side.numel() != batch_size or not torch.all(torch.isfinite(side) & (side > 0)):
        raise ValueError("one finite positive side length is required per graph")
    if transform_indices is None:
        ids = torch.randint(1, 8, (batch_size,), device=positions.device)
    else:
        ids = torch.as_tensor(transform_indices, device=positions.device)
        if (ids.shape != (batch_size,) or ids.is_floating_point() or ids.is_complex()
                or ids.dtype == torch.bool or torch.any((ids < 1) | (ids > 7))):
            raise ValueError("transform_indices must contain one integer in 1..7 per graph")
        ids = ids.long()
    # Exact signed permutations avoid trigonometric roundoff at 90-degree angles.
    rotations = positions.new_tensor([[[1, 0], [0, 1]], [[0, -1], [1, 0]],
                                      [[-1, 0], [0, -1]], [[0, 1], [-1, 0]]])
    transform = rotations[ids % 4].clone()
    transform[:, :, 0] *= torch.where(ids >= 4, -1, 1)[:, None]
    center = side[:, None, None] * 0.5
    transformed = torch.bmm(positions - center, transform.transpose(1, 2)) + center
    return torch.cat((positions, transformed), dim=0)
