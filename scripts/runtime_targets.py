"""NumPy reference for the HD-1910 target pipeline; calibration stays outside the repository."""
from pathlib import Path
import tomllib
import numpy as np

HEAD_JOINTS = {"neck_pitch", "head_pitch", "head_yaw", "head_roll"}


class TargetTransform:
    def __init__(self, names, home, calibration, lowpass=(.5, .7)):
        rows = {r["name"]: r for r in tomllib.loads(Path(calibration).read_text())["joints"]}
        self.home = np.asarray(home, dtype=float)
        self.lower = np.array([rows[n]["min_rad"] for n in names])
        self.upper = np.array([rows[n]["max_rad"] for n in names])
        self.zero = np.array([rows[n]["zero_raw"] for n in names])
        self.direction = np.array([rows[n]["direction"] for n in names])
        if not all(np.isfinite(v).all() for v in (self.home, self.lower, self.upper, self.zero, self.direction)):
            raise ValueError("Calibration and HOME must be finite")
        if np.any(self.lower > self.upper) or not np.isin(self.direction, [-1, 1]).all():
            raise ValueError("Invalid calibration bounds or direction")
        self.alpha = np.array([lowpass[0] if n in HEAD_JOINTS else lowpass[1] for n in names])
        if not ((self.alpha > 0) & (self.alpha <= 1)).all():
            raise ValueError("Lowpass alpha must be in (0, 1]")
        self.filtered, self.voltage, self.ticks = None, 0., 0

    def __call__(self, target, supply):
        if self.ticks % 50 == 0 and np.isfinite(supply) and supply >= 0:
            self.voltage = supply if self.voltage <= 0 else .1 * supply + .9 * self.voltage
        gain = 7.4 / np.clip(self.voltage, 6., 9.5) if self.voltage > 0 else 1.
        value = self.home + (np.asarray(target) - self.home) * gain
        self.filtered = value if self.filtered is None else self.alpha * value + (1 - self.alpha) * self.filtered
        clamped = self.filtered.clip(self.lower, self.upper)
        raw = self.zero + self.direction * clamped * (4096 / (2 * np.pi))
        raw = np.sign(raw) * np.floor(abs(raw) + .5)
        self.ticks += 1
        return (raw - self.zero) * self.direction * (2 * np.pi / 4096)
