"""Spatio-temporal drift detector — flags calibration drift over a sliding window."""


class DriftDetector:
    def __init__(self, window_size, slope_threshold):
        raise NotImplementedError

    def update(self, frame_error):
        raise NotImplementedError

    def is_drifting(self):
        raise NotImplementedError
