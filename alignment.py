"""Image-space guidance for a wrist camera. This module cannot command a robot."""
from collections import deque
import math
import time

import numpy as np


def point(value):
    values = np.asarray(value, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Choose a point inside the image.")
    return values


def center(box):
    a = np.asarray(box["xyxy"], dtype=float)
    return (a[:2] + a[2:]) / 2


class Alignment:
    """Lock one visible detection; require reselection after loss or ambiguity.

    Coordinates are normalized, top-left origin. No depth, units in mm, tool
    axis angle, socket engagement, or motor direction is inferred from a box.
    """

    def __init__(self, aim=None, tolerance_px=5.0, stable_frames=5, max_gap_s=.75):
        self.aim = point(aim) if aim is not None else None
        if not math.isfinite(tolerance_px) or not 1 <= tolerance_px <= 50:
            raise ValueError("Image tolerance must be 1 to 50 pixels.")
        if not isinstance(stable_frames, int) or stable_frames < 2:
            raise ValueError("At least two fresh frames are required.")
        if not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("Frame timeout must be positive.")
        self.tolerance = tolerance_px
        self.max_gap = max_gap_s
        self.history = deque(maxlen=stable_frames)
        self.target = None
        self.last_time = None
        self.shape = None
        self.status = "select_target"

    def select(self, detections, location):
        p = point(location)
        matches = [d for d in detections if d["xyxy"][0] <= p[0] <= d["xyxy"][2]
                   and d["xyxy"][1] <= p[1] <= d["xyxy"][3]]
        if len(matches) != 1:
            raise ValueError("Click inside one unambiguous lug-nut detection.")
        self.target = matches[0]
        self.history.clear()
        self.status = "acquiring"

    def clear(self, status="select_target"):
        self.target = None
        self.history.clear()
        self.status = status

    def update(self, detections, width, height, now=None):
        now = time.monotonic() if now is None else now
        if not math.isfinite(now) or width <= 0 or height <= 0:
            raise ValueError("Invalid frame metadata.")
        if self.shape is not None and self.shape != (width, height):
            self.aim = None
            self.clear("camera_changed")
        if self.last_time is not None and (now <= self.last_time or now - self.last_time > self.max_gap):
            self.clear("stale")
        self.shape = (width, height)
        self.last_time = now
        base = {"state": self.status, "image_aligned": False, "robot_motion_enabled": False,
                "error_px": None, "target": None, "stable_frames": 0,
                "aim": self.aim.tolist() if self.aim is not None else None}
        if self.target is None:
            return base
        scale = np.array([width, height])
        old = self.target
        size = (np.asarray(old["xyxy"])[2:] - np.asarray(old["xyxy"])[:2]) * scale
        radius = max(8., float(np.min(size)) * .6)
        candidates = []
        for d in detections:
            new_size = (np.asarray(d["xyxy"])[2:] - np.asarray(d["xyxy"])[:2]) * scale
            ratio = new_size / size
            if (ratio >= .65).all() and (ratio <= 1.55).all():
                distance = np.linalg.norm((center(d) - center(old)) * scale)
                if distance <= radius:
                    candidates.append(d)
        if len(candidates) != 1:
            self.clear("ambiguous" if len(candidates) > 1 else "target_lost")
            return {**base, "state": self.status}
        self.target = candidates[0]
        base["target"] = self.target
        if self.aim is None:
            return {**base, "state": "set_aim"}
        error = (center(self.target) - self.aim) * scale
        self.history.append(error)
        aligned = (len(self.history) == self.history.maxlen
                   and all(np.linalg.norm(e) <= self.tolerance for e in self.history))
        self.status = "image_aligned" if aligned else "offset"
        return {**base, "state": self.status, "image_aligned": bool(aligned),
                "error_px": error.round(2).tolist(), "stable_frames": len(self.history)}
