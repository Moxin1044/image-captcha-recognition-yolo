"""Recognize a fixed-width captcha with a trained YOLO classification model."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def recognize(model: YOLO, image_path: Path, positions: int = 5, imgsz: int = 64) -> tuple[str, float]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        if width % positions:
            raise ValueError(f"Image width {width} is not divisible by {positions}")
        char_width = width // positions
        results = []
        for position in range(positions):
            crop = image.crop((position * char_width, 0, (position + 1) * char_width, height))
            result = model.predict(source=crop, imgsz=imgsz, verbose=False)[0]
            index = int(result.probs.top1)
            results.append((result.names[index], float(result.probs.top1conf)))
    text = "".join(character for character, _ in results)
    confidence = sum(score for _, score in results) / positions
    return text, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("runs/captcha/character_classifier/weights/best.pt"))
    parser.add_argument("--positions", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=64)
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    text, confidence = recognize(YOLO(str(args.model)), args.image, args.positions, args.imgsz)
    print(f"{text}\tconfidence={confidence:.4f}")


if __name__ == "__main__":
    main()
