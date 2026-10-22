"""Optional local YOLO detector. No generic weights masquerade as lug-nut weights."""
from pathlib import Path
import threading

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "models" / "lugnut.pt"


class LugnutDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS):
        self.weights = Path(weights)
        self.model = None
        self.lock = threading.Lock()
        self.class_id = None
        self.signature = None

    def status(self):
        return {"weights_available": self.weights.is_file(), "loaded": self.model is not None,
                "model": self.weights.name, "robot_motion_enabled": False}

    def predict(self, frame, confidence=.5):
        if not np.isfinite(confidence) or not .05 <= confidence <= .95:
            raise ValueError("Confidence threshold must be between 0.05 and 0.95.")
        if not self.weights.is_file():
            raise ValueError("No trained lug-nut model yet. Capture and label examples, then train.")
        stat = self.weights.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        with self.lock:
            if self.model is None or self.signature != signature:
                try:
                    from ultralytics import YOLO
                except ImportError as exc:
                    raise ValueError("Start the alignment launcher to install the vision dependencies.") from exc
                model = YOLO(str(self.weights), task="detect")
                ids = [int(k) for k, name in model.names.items() if name == "lug_nut"]
                if len(ids) != 1:
                    raise ValueError("Model must contain exactly one class named lug_nut. Generic COCO weights cannot recognize lug nuts.")
                self.model, self.class_id, self.signature = model, ids[0], signature
            result = self.model.predict(frame, conf=confidence, iou=.45, imgsz=640,
                                        classes=[self.class_id], max_det=20, verbose=False)[0]
            if result.boxes is None:
                return []
            boxes = result.boxes.xyxyn.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            found = []
            for box, score in zip(boxes, scores):
                box = np.clip(box, 0, 1)
                if np.isfinite(box).all() and np.isfinite(score) and box[2] > box[0] and box[3] > box[1]:
                    found.append({"xyxy": box.tolist(), "confidence": float(score), "label": "lug_nut"})
            return found
