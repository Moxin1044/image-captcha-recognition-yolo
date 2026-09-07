"""Prepare fixed-width captcha character crops for Ultralytics classification."""

from __future__ import annotations

import argparse
import hashlib
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


def discover_images(source: Path, val_fraction: float = 0.2) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Use image stems as labels and make a deterministic train/val split.

    This is useful when a source directory contains images but no manifests.
    Labels are upper-cased so ``a`` and ``A`` share one class.
    """
    items = []
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            label = path.stem.upper()
            if not label:
                continue
            items.append((path.name, label))
    if not items:
        raise ValueError(f"No image files found in {source}")
    train, val = [], []
    for item in items:
        digest = hashlib.sha1(item[0].encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") / 2**32
        (val if bucket < val_fraction else train).append(item)
    return train, val


def prepare(source: Path, output: Path, clear: bool = False, discover: bool = False,
            val_fraction: float = 0.2) -> tuple[int, int]:
    if clear and output.exists():
        shutil.rmtree(output)
    total = 0
    if discover:
        splits = dict(zip(("train", "val"), discover_images(source, val_fraction)))
    else:
        splits = {}
        for split in ("train", "val"):
            manifest = source / f"{split}.txt"
            if not manifest.exists():
                raise FileNotFoundError(f"Missing manifest: {manifest}")
            splits[split] = read_manifest(manifest)
    for split in ("train", "val"):
        for image_name, text in splits[split]:
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
    return total, sum(len(rows) for rows in splits.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("Verification_code"))
    parser.add_argument("--output", type=Path, default=Path("data/characters"))
    parser.add_argument("--clear", action="store_true", help="Remove output before preparing it")
    parser.add_argument("--discover", action="store_true",
                        help="Use image filename stems as labels and split images deterministically; supports 4-digit files")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    crops, samples = prepare(args.source, args.output, args.clear, args.discover, args.val_fraction)
    print(f"Prepared {crops} character crops from {samples} captcha images in {args.output}")


if __name__ == "__main__":
    main()
