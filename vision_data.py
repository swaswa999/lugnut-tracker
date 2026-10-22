"""Human-reviewed examples; recording sessions stay within one dataset split."""
import hashlib
import json
from pathlib import Path
import re
import threading
import uuid

import cv2
import numpy as np

DATA_ROOT = Path(__file__).resolve().parent / "data" / "lugnut"


class Dataset:
    def __init__(self, root=DATA_ROOT):
        self.root = Path(root)
        self.lock = threading.Lock()

    def status(self):
        counts = {}
        for split in ("train", "val", "test"):
            counts[split] = len(list((self.root / "images" / split).glob("*.jpg")))
        return counts

    def save(self, frame, boxes, session, split, reviewed=False):
        if reviewed is not True:
            raise ValueError("Confirm that every visible lug nut is labeled, or that this is an empty scene.")
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown dataset split.")
        if not isinstance(session, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", session):
            raise ValueError("Session name must use 1–48 letters, numbers, hyphens or underscores.")
        if not isinstance(boxes, list) or len(boxes) > 20:
            raise ValueError("A frame can have 0 to 20 lug-nut labels.")
        labels = []
        h, w = frame.shape[:2]
        for box in boxes:
            a = np.asarray(box, dtype=float)
            if a.shape != (4,) or not np.isfinite(a).all():
                raise ValueError("Invalid label box.")
            x1, y1, x2, y2 = a
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError("Label boxes must be inside the image.")
            if min((x2 - x1) * w, (y2 - y1) * h) < 6:
                raise ValueError("Lug nuts must be at least 6 pixels across. Move the camera closer.")
            labels.append(f"0 {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}")
        digest = hashlib.sha256(frame.tobytes() + str(frame.shape).encode()).hexdigest()
        with self.lock:
            manifest_path = self.root / "manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sessions": {}, "frames": []}
            previous = manifest["sessions"].get(session)
            if previous is not None and previous != split:
                raise ValueError("This recording session already belongs to another split. Use a separate recording for validation/test.")
            if any(item["sha256"] == digest for item in manifest["frames"]):
                raise ValueError("This exact frame is already saved. Capture a different view.")
            name = uuid.uuid4().hex
            for folder in ("images", "labels"):
                (self.root / folder / split).mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(self.root / "images" / split / f"{name}.jpg"), frame):
                raise ValueError("Could not save the image.")
            (self.root / "labels" / split / f"{name}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
            manifest["sessions"][session] = split
            manifest["frames"].append({"id": name, "session": session, "split": split,
                                       "sha256": digest, "objects": len(labels)})
            temporary = self.root / "manifest.tmp"
            temporary.write_text(json.dumps(manifest, indent=2))
            temporary.replace(manifest_path)
            # JSON strings are valid YAML scalars; paths with spaces remain valid.
            (self.root / "dataset.yaml").write_text(
                f"path: {json.dumps(str(self.root.resolve()))}\ntrain: images/train\nval: images/val\n"
                + ("test: images/test\n" if self.status()["test"] else "") + "names:\n  0: lug_nut\n")
        return {"saved": name, "counts": self.status()}
