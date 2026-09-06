"""Prepare fixed-width captcha character crops for Ultralytics classification."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image


def read_manifest(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected '<image> <label>'")
        rows.append((fields[0], fields[1]))
    return rows


def prepare(source: Path, output: Path, clear: bool = False) -> tuple[int, int]:
    if clear and output.exists():
        shutil.rmtree(output)
    total = 0
    for split in ("train", "val"):
        manifest = source / f"{split}.txt"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest}")
        for image_name, text in read_manifest(manifest):
            image_path = source / image_name
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                if width < len(text) or width % len(text):
                    raise ValueError(f"{image_name}: width {width} is not divisible by label length {len(text)}")
                char_width = width // len(text)
                for position, character in enumerate(text):
                    target = output / split / character
                    target.mkdir(parents=True, exist_ok=True)
                    crop = image.crop((position * char_width, 0, (position + 1) * char_width, height))
                    crop.save(target / f"{Path(image_name).stem}_{position}{Path(image_name).suffix.lower()}")
                    total += 1
    return total, len(read_manifest(source / "train.txt")) + len(read_manifest(source / "val.txt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Verification_code"))
    parser.add_argument("--output", type=Path, default=Path("data/characters"))
    parser.add_argument("--clear", action="store_true", help="Remove output before preparing it")
    args = parser.parse_args()
    crops, samples = prepare(args.source, args.output, args.clear)
    print(f"Prepared {crops} character crops from {samples} captcha images in {args.output}")


if __name__ == "__main__":
    main()
