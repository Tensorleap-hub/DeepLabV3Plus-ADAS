#!/usr/bin/env python3
"""Regenerate the unlabeled_subset recipe. Cityscapes frames (with fine GT) already live
in the datasets-reteai bucket, so they are emitted as cloud paths; the raw KITTI Karlsruhe
frames do not, so they are uploaded to KITTI/raw_karlsruhe/ and then emitted as cloud
paths. Selection still reads the local raw mirrors (vegetation scoring, drive walk)."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from domain_gap.utils.gcs_utils import _connect_to_gcs_and_return_bucket
from domain_gap.utils.config import CONFIG


VEGETATION_ID = 21
KITTI_RAW_PREFIX = "KITTI/raw_karlsruhe"
DEFAULT_CITYSCAPES_ROOT = Path(os.environ.get("CITYSCAPES_ROOT", "/Users/orram/Tensorleap/data/cityscapes"))
DEFAULT_KITTI_ROOT = Path(os.environ.get("KITTI_ROOT", "/Users/orram/Tensorleap/data/kitti"))
DEFAULT_EXCLUDE_SPLITS_ROOT = Path("domain_gap/data/splits/original_csv_subset")
DEFAULT_CITYSCAPES_CITIES = ("tubingen", "munster", "bremen")


def remove_suffix(value, suffix):
    if value.endswith(suffix):
        return value[:-len(suffix)]
    return value


def non_negative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def csv_list(value):
    return tuple(item.strip() for item in value.split(",") if item.strip())


def load_excluded_stems(root):
    excluded = set()
    for split in ("train", "val"):
        path = root / f"{split}.json"
        if not path.exists():
            continue
        with path.open("r") as f:
            data = json.load(f)
        excluded.update(data.get("file_names", []))
    return excluded


def select_cityscapes(root, bucket, threshold, limit, cities, excluded_stems):
    selected = []
    allowed_cities = set(cities) if cities else None

    image_roots = [
        root / "leftImg8bit_trainvaltest" / "leftImg8bit",
        root / "leftImg8bit_trainextra" / "leftImg8bit",
    ]

    for image_root in image_roots:
        if not image_root.exists():
            continue

        for image_path in image_root.glob("*/*/*_leftImg8bit.png"):
            split, city = image_path.relative_to(image_root).parts[:2]
            if allowed_cities is not None and city not in allowed_cities:
                continue

            stem = remove_suffix(image_path.name, "_leftImg8bit.png")
            if stem in excluded_stems:
                continue

            possible_masks = [
                root / "gtFine_trainvaltest" / "gtFine" /
                split / city / f"{stem}_gtFine_labelIds.png",

                root / "gtCoarse" / "gtCoarse" /
                split / city / f"{stem}_gtCoarse_labelIds.png",
            ]

            mask_path = next(
                (path for path in possible_masks if path.exists()),
                None,
            )

            if mask_path is None:
                continue

            mask = np.asarray(Image.open(mask_path))
            vegetation_fraction = float(np.mean(mask == VEGETATION_ID))

            if vegetation_fraction >= threshold:
                selected.append({
                    "source": image_path,
                    "dataset": "cityscapes",
                    "city": city,
                    "split": split,
                    "vegetation_percent": vegetation_fraction * 100,
                    "selection_reason": "high_vegetation",
                })

    selected.sort(
        key=lambda sample: sample["vegetation_percent"],
        reverse=True,
    )

    if limit is not None:
        selected = selected[:limit]

    manifest = []
    for sample in selected:
        source = sample.pop("source")
        cloud_path = f"Cityscapes/{source.relative_to(root).as_posix()}"
        if not bucket.blob(cloud_path).exists():
            print(f"Skipping cityscapes frame absent from bucket: {cloud_path}")
            continue

        manifest.append({
            **sample,
            "path": cloud_path,
            "filename": source.name,
        })

    return manifest


def select_kitti(root, bucket, stride, limit):
    # Each drive reuses frame names such as 0000000000.png, so the drive ID is added to
    # the uploaded filename. Raw KITTI frames are not in the bucket, so upload each one.
    paths = sorted(root.glob(
        "**/*_drive_*_sync/image_02/data/*.png"
    ))

    selected = paths[::stride]
    if limit is not None:
        selected = selected[:limit]

    manifest = []
    for source in selected:
        drive_dir = next(
            parent for parent in source.parents
            if "_drive_" in parent.name and parent.name.endswith("_sync")
        )
        drive = remove_suffix(drive_dir.name, "_sync")

        filename = f"{drive}_{source.name}"
        cloud_path = f"{KITTI_RAW_PREFIX}/{filename}"
        blob = bucket.blob(cloud_path)
        if not blob.exists():
            blob.upload_from_filename(str(source))

        manifest.append({
            "dataset": "kitti_raw",
            "city": "Karlsruhe",
            "drive": drive,
            "frame": source.stem,
            "vegetation_percent": None,
            "selection_reason": "karlsruhe",
            "path": cloud_path,
            "filename": filename,
        })

    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cityscapes-root",
        type=Path,
        default=DEFAULT_CITYSCAPES_ROOT,
    )
    parser.add_argument(
        "--kitti-root",
        type=Path,
        default=DEFAULT_KITTI_ROOT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("domain_gap/data/splits/unlabeled_subset"),
    )
    parser.add_argument(
        "--vegetation-threshold",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--cityscapes-cities",
        type=csv_list,
        default=DEFAULT_CITYSCAPES_CITIES,
    )
    parser.add_argument(
        "--exclude-splits-root",
        type=Path,
        default=DEFAULT_EXCLUDE_SPLITS_ROOT,
    )
    parser.add_argument("--max-cityscapes", type=non_negative_int, default=500)
    parser.add_argument("--max-kitti", type=non_negative_int, default=500)
    parser.add_argument("--kitti-stride", type=positive_int, default=10)
    args = parser.parse_args()
    excluded_stems = load_excluded_stems(args.exclude_splits_root)
    bucket = _connect_to_gcs_and_return_bucket(CONFIG['BUCKET_NAME'])

    cityscapes = select_cityscapes(
        root=args.cityscapes_root,
        bucket=bucket,
        threshold=args.vegetation_threshold,
        limit=args.max_cityscapes,
        cities=args.cityscapes_cities,
        excluded_stems=excluded_stems,
    )

    kitti = select_kitti(
        root=args.kitti_root,
        bucket=bucket,
        stride=args.kitti_stride,
        limit=args.max_kitti,
    )

    manifest = cityscapes + kitti
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Cityscapes selected: {len(cityscapes)}")
    print(f"KITTI selected:      {len(kitti)}")
    print(f"Excluded stems:      {len(excluded_stems)}")
    print(f"Cityscapes cities:   {', '.join(args.cityscapes_cities)}")
    print(f"Output:              {args.output.resolve()}")
    print(f"Manifest:            {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
