from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NetPara:
    shape: str = "square"
    side_length: float = 100.0


@dataclass(frozen=True)
class MeasurePara:
    sigma_d: float = 0.015
