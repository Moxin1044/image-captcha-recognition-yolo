"""Recognize a fixed-width captcha with a trained YOLO classification model."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


def recognize(model: YOLO, image_path: Path, positions: int = 5, imgsz: int = 96) -> tuple[str, float]:
    if positions < 1:
        raise ValueError("positions must be at least 1")
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
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to best.pt; when omitted, use the latest training result")
    parser.add_argument("--positions", type=int, default=5,
                        help="Number of characters in the image (use 4 for a four-character CAPTCHA)")
    parser.add_argument("--imgsz", type=int, default=96)
    args = parser.parse_args()
    model_path = args.model
    if model_path is None:
        candidates = [
            Path("runs/captcha/character_classifier/weights/best.pt"),
            Path("runs/classify/runs/captcha/character_classifier/weights/best.pt"),
        ]
        candidates.extend(sorted(Path("runs").glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True))
        model_path = next((path for path in candidates if path.exists()), None)
    if model_path is None or not model_path.exists():
        raise SystemExit("Model not found. Train first or pass --model path/to/best.pt")
    text, confidence = recognize(YOLO(str(model_path)), args.image, args.positions, args.imgsz)
    print(f"{text}\tconfidence={confidence:.4f}")


if __name__ == "__main__":
    main()
