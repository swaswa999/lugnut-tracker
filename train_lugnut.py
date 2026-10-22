"""Train and evaluate the local lug-nut model using reviewed camera images."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from vision_data import DATA_ROOT, Dataset


def check_dataset(root):
    root = Path(root)
    counts = Dataset(root).status()
    if counts["train"] < 10 or counts["val"] < 5:
        raise ValueError("Capture at least 10 training and 5 validation images from separate recording sessions before a smoke-training run. See VISION-README.md for the larger recommended dataset.")
    seen = set()
    for split in ("train", "val", "test"):
        positives = 0
        for image in (root / "images" / split).glob("*.jpg"):
            label = root / "labels" / split / f"{image.stem}.txt"
            if not label.is_file():
                raise ValueError(f"Missing reviewed labels for {image.name}.")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen:
                raise ValueError("Duplicate images found. Remove duplicates before training.")
            seen.add(digest)
            for line in label.read_text().splitlines():
                fields = line.split()
                if len(fields) != 5 or fields[0] != "0":
                    raise ValueError(f"Invalid label in {label.name}.")
                x, y, w, h = map(float, fields[1:])
                if not (0 < w <= 1 and 0 < h <= 1 and w/2-1e-7 <= x <= 1-w/2+1e-7 and h/2-1e-7 <= y <= 1-h/2+1e-7):
                    raise ValueError(f"Label coordinates outside image in {label.name}.")
                positives += 1
        if counts[split] and not positives:
            raise ValueError(f"{split} needs some labeled lug nuts as well as negative images.")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cpu", help="cpu, mps (Apple GPU), or CUDA index")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--base", default="yolo11n.pt", help="Local pretrained weights or official Ultralytics model name")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        counts = check_dataset(args.data)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Dataset not ready: {exc}\n")
    if args.epochs < 1 or args.batch < 1:
        parser.error("epochs and batch must be positive")
    print(f"Reviewed images: {counts}")
    if args.check_only:
        return
    from ultralytics import YOLO
    root = Path(__file__).resolve().parent
    model = YOLO(args.base)
    model.train(data=str(args.data.resolve() / "dataset.yaml"), epochs=args.epochs,
                imgsz=640, batch=args.batch, device=args.device, workers=0,
                project=str(root / "runs"), name="lugnut", seed=42,
                patience=15, fliplr=.5, flipud=0, degrees=15, translate=.1,
                scale=.3, mosaic=.5, close_mosaic=10)
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError("Training did not produce best.pt; the active model was not replaced.")
    trained = YOLO(str(best))
    validation = trained.val(data=str(args.data.resolve() / "dataset.yaml"), split="val",
                             device=args.device, workers=0, plots=True)
    report = {"model": "YOLO11n" if args.base == "yolo11n.pt" else args.base,
              "counts": counts, "epochs_requested": args.epochs,
              "validation": {k: float(v) for k, v in validation.results_dict.items()},
              "hardware_validated": False, "precision_alignment_validated": False,
              "notes": "Detection boxes estimate visual centers. Validate center error and socket alignment on separate physical trials."}
    if counts["test"]:
        test = trained.val(data=str(args.data.resolve() / "dataset.yaml"), split="test",
                           device=args.device, workers=0, plots=True)
        report["held_out_test"] = {k: float(v) for k, v in test.results_dict.items()}
    output = root / "models"
    output.mkdir(exist_ok=True)
    temporary = output / "lugnut.tmp.pt"
    shutil.copy2(best, temporary)
    temporary.replace(output / "lugnut.pt")
    (output / "training-report.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {output / 'lugnut.pt'}\nRestart live detection to use the model. Physical alignment is not yet validated.")


if __name__ == "__main__":
    main()
