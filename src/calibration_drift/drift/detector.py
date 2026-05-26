"""Spatio-temporal drift detector — flags calibration drift over a sliding window."""

from __future__ import annotations

from collections import deque

import numpy as np


class DriftDetector:
    """Sliding-window linear-fit drift detector.

    Maintains a window of the most recent per-frame reprojection errors and
    flags drift when the linear-fit slope exceeds `slope_threshold`. A positive
    slope means errors are growing across the window — the calibration is
    getting worse over time.
    """

    def __init__(self, window_size: int = 30, slope_threshold: float = 0.1):
        if window_size < 2:
            raise ValueError("window_size must be at least 2 to fit a slope")
        self.window_size = window_size
        self.slope_threshold = slope_threshold
        self._errors: deque[float] = deque(maxlen=window_size)

    def update(self, frame_error: float) -> None:
        """Append a per-frame error. NaN values are silently dropped."""
        if not np.isnan(frame_error):
            self._errors.append(float(frame_error))

    def slope(self) -> float:
        """Linear-fit slope of errors over the current window (px per frame).

        Returns NaN when the window has fewer than 2 samples.
        """
        if len(self._errors) < 2:
            return float("nan")
        y = np.asarray(self._errors)
        x = np.arange(len(y), dtype=float)
        return float(np.polyfit(x, y, 1)[0])

    def is_drifting(self) -> bool:
        """True iff the window is full and the slope exceeds the threshold."""
        if len(self._errors) < self.window_size:
            return False
        return self.slope() > self.slope_threshold

    @property
    def errors(self) -> list[float]:
        """Snapshot of the current error window (oldest first)."""
        return list(self._errors)
