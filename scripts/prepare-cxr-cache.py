"""Build a resized JPEG cache for the ``cxr_first`` competition dataset.

Renting a GPU by the hour makes JPEG decoding of 2–4k px originals expensive.
This script writes a mirror tree with the short side resized to ``--size`` and
copies the organizer CSVs, so ``cxr-competition train/predict`` can use
``--data-root <cache>`` directly. The cache is small enough (~4–6 GB) to upload
to a rented instance in minutes.

Usage::

    python scripts/prepare-cxr-cache.py \
        --source cxr_first/cxr_first --output cxr_first-640 --size 640 --workers 6
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SPLITS = ("train", "valid", "test")


def _one_image(source: Path, destination: Path, size: int, quality: int) -> tuple[bool, int]:
    """Resize one image to ``short side <= size`` and save as grayscale JPEG."""

    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - dependency is declared for ml extra
        raise RuntimeError("Pillow is required to build the image cache") from None
    if destination.is_file() and destination.stat().st_mtime >= source.stat().st_mtime:
        return False, destination.stat().st_size
    with Image.open(source) as image:
        gray = image.convert("L")
        width, height = gray.size
        short_side = min(width, height)
        if short_side > size:
            scale = size / short_side
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            gray = gray.resize(new_size, Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        gray.save(destination, format="JPEG", quality=quality, optimize=True)
    return True, destination.stat().st_size


def _worker(job: tuple[str, str, int, int]) -> int:
    from PIL import Image  # noqa: F401  (import inside worker for process safety)

    source, destination, size, quality = job
    return _one_image(Path(source), Path(destination), size, quality)[1]


def build_cache(
    source_root: str | Path,
    output_root: str | Path,
    *,
    size: int = 640,
    quality: int = 92,
    workers: int = 6,
) -> dict[str, int]:
    source = Path(source_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    if size <= 0:
        raise ValueError("size must be positive")
    for split in SPLITS:
        csv_path = source / f"{split}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"missing split CSV: {csv_path}")
        shutil.copy2(csv_path, output / f"{split}.csv")

    jobs: list[tuple[str, str, int, int]] = []
    for split in SPLITS:
        split_root = source / split
        if not split_root.is_dir():
            raise FileNotFoundError(f"missing split directory: {split_root}")
        for path in split_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(source)
            jobs.append(
                (
                    str(path),
                    str(output / relative.with_suffix(".jpg")),
                    size,
                    quality,
                )
            )
    if not jobs:
        raise ValueError(f"no images found under {source}")
    print(f"caching {len(jobs)} images to {output} (short side {size}, quality {quality})")
    started = time.perf_counter()
    total_bytes = 0
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for written_bytes in pool.map(_worker, jobs, chunksize=16):
            total_bytes += written_bytes
            done += 1
            if done % 2000 == 0:
                elapsed = time.perf_counter() - started
                print(f"  {done}/{len(jobs)} images, {elapsed:.0f}s, {total_bytes / 1e9:.2f} GB", flush=True)
    elapsed = time.perf_counter() - started
    summary = {
        "images": len(jobs),
        "bytes": total_bytes,
        "seconds": round(elapsed, 1),
    }
    print(f"done: {summary}")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the cxr_first resized JPEG cache")
    parser.add_argument("--source", type=Path, default=Path("cxr_first") / "cxr_first")
    parser.add_argument("--output", type=Path, default=Path("cxr_first-640"))
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    summary = build_cache(
        args.source,
        args.output,
        size=args.size,
        quality=args.quality,
        workers=args.workers,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
