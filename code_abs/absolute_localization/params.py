"""Defaults for synchronized-anchor, unknown-user-clock TOA localization."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class NetPara:
    shape: str = "square"
    side_length: float = 100.0
    user_side_ratio: float = 0.8  # Centered user square / anchor square side; square only.

    def __post_init__(self):
        if not math.isfinite(self.user_side_ratio) or not 0 < self.user_side_ratio <= 1:
            raise ValueError("user_side_ratio must be finite and lie in (0,1]")


@dataclass(frozen=True)
class MeasurePara:
    sigma_d: float = 0.015
    c: float = 299792458.0
    rcond: float = 1e-12  # numerical rank threshold in range-clock coordinates
