from __future__ import annotations

from enum import Enum

__all__ = ["DrivingState"]


class DrivingState(Enum):
    LANE_KEEPING = "LANE KEEP"
    #: Neither boundary available, even from memory: the vehicle must halt.
    STOP = "STOP"
    #: The right boundary was inferred from the left.
    LEFT_ONLY = "LEFT ONLY"
    #: The left boundary was inferred from the right.
    RIGHT_ONLY = "RIGHT ONLY"
    #: At least one boundary is replayed from memory, not detected.
    COASTING = "COASTING"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value
