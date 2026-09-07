"""Train a YOLO classification model on prepared captcha character crops."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/characters"))
    parser.add_argument("--model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=96,
                        help="Classification input size; 96 preserves more character detail than 64")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="0", help="GPU index, cpu, or auto")
    parser.add_argument("--project", type=Path, default=Path("runs/captcha"))
    parser.add_argument("--name", default="character_classifier")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not (args.data / "train").exists() or not (args.data / "val").exists():
        raise SystemExit("Prepared dataset not found. Run: python prepare_dataset.py --clear")
    device = args.device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=device, project=str(args.project), name=args.name, workers=args.workers,
        pretrained=True, patience=20, exist_ok=True,
        # Mirroring a glyph changes its identity and hurts CAPTCHA recognition.
        fliplr=0.0, flipud=0.0,
        degrees=3.0, translate=0.05, scale=0.2,
        erasing=0.1,
    )


if __name__ == "__main__":
    main()
