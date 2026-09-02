#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path


DEFAULT_CSV_PATHS = [
    Path("domain_gap_samples.csv"),
    Path("domian_gap_samples.csv"),
]
DEFAULT_CITYSCAPES_ROOT = Path("/Users/orram/Tensorleap/data/cityscapes")
DEFAULT_KITTI_ROOT = Path("/Users/orram/Tensorleap/data/kitti")
DEFAULT_OUTPUT = Path("domain_gap/data/datasets/local_csv_subset")


def remove_suffix(value, suffix):
    if value.endswith(suffix):
        return value[:-len(suffix)]
    return value


def existing_csv_path():
    for path in DEFAULT_CSV_PATHS:
        if path.exists():
            return path
    return DEFAULT_CSV_PATHS[0]


def build_cityscapes_index(root):
    image_root = root / "leftImg8bit_trainvaltest" / "leftImg8bit"
    gt_root = root / "gtFine_trainvaltest" / "gtFine"
    metadata_root = root / "vehicle_trainvaltest" / "vehicle"

    images = {
        remove_suffix(path.name, "_leftImg8bit.png"): path
        for path in image_root.glob("*/*/*_leftImg8bit.png")
    }
    masks = {
        remove_suffix(path.name, "_gtFine_labelIds.png"): path
        for path in gt_root.glob("*/*/*_gtFine_labelIds.png")
    }
    gt_images = {
        remove_suffix(path.name, "_gtFine_color.png"): path
        for path in gt_root.glob("*/*/*_gtFine_color.png")
    }
    metadata = {
        remove_suffix(path.name, "_vehicle.json"): path
        for path in metadata_root.glob("*/*/*_vehicle.json")
    }
    return images, masks, gt_images, metadata


def build_kitti_index(root):
    dataset_root = root / "data_semantics" / "training"
    images = {
        str(path.relative_to(root)): path
        for path in (dataset_root / "image_2").glob("*.png")
    }
    masks = {
        "KITTI/data_semantics/training/image_2/" + path.name: path
        for path in (dataset_root / "semantic").glob("*.png")
    }
    gt_images = {
        "KITTI/data_semantics/training/image_2/" + path.name: path
        for path in (dataset_root / "semantic_rgb").glob("*.png")
    }
    return images, masks, gt_images


def cityscapes_sample(row, indexes):
    images, masks, gt_images, metadata = indexes
    filename = row["Filename"]
    image_path = images.get(filename)
    mask_path = masks.get(filename)
    if image_path is None or mask_path is None:
        return None

    return {
        "image_path": str(image_path),
        "gt_path": str(mask_path),
        "gt_image_path": str(gt_images.get(filename, "")),
        "file_name": filename,
        "city": row["City"],
        "dataset": "cityscapes",
        "metadata": str(metadata.get(filename, "")),
    }


def kitti_sample(row, indexes):
    images, masks, gt_images = indexes
    filename = row["Filename"]
    image_path = images.get(filename)
    mask_path = masks.get(filename)
    if image_path is None or mask_path is None:
        return None

    return {
        "image_path": str(image_path),
        "gt_path": str(mask_path),
        "gt_image_path": str(gt_images.get(filename, "")),
        "file_name": filename,
        "city": row["City"],
        "dataset": "kitti",
        "metadata": "",
    }


def response_data(samples, subset_name):
    return {
        "subset_name": subset_name,
        "image_path": [sample["image_path"] for sample in samples],
        "gt_path": [sample["gt_path"] for sample in samples],
        "gt_image_path": [sample["gt_image_path"] for sample in samples],
        "file_names": [sample["file_name"] for sample in samples],
        "cities": [sample["city"] for sample in samples],
        "dataset": [sample["dataset"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
        "real_size": len(samples),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=existing_csv_path())
    parser.add_argument("--cityscapes-root", type=Path, default=DEFAULT_CITYSCAPES_ROOT)
    parser.add_argument("--kitti-root", type=Path, default=DEFAULT_KITTI_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-percent", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv.open(newline="")))
    cityscapes_indexes = build_cityscapes_index(args.cityscapes_root)
    kitti_indexes = build_kitti_index(args.kitti_root)

    samples = []
    missing = []
    for row in rows:
        dataset = row["Dataset"]
        if dataset == "cityscapes":
            sample = cityscapes_sample(row, cityscapes_indexes)
        elif dataset == "kitti":
            sample = kitti_sample(row, kitti_indexes)
        else:
            sample = None

        if sample is None:
            missing.append({
                "dataset": dataset,
                "filename": row["Filename"],
                "city": row["City"],
                "reason": "missing local image or mask",
            })
        else:
            samples.append(sample)

    random.Random(args.seed).shuffle(samples)
    train_size = int(len(samples) * args.train_percent)
    train_samples = samples[:train_size]
    val_samples = samples[train_size:]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train.json").write_text(json.dumps(response_data(train_samples, "train"), indent=2))
    (args.output / "val.json").write_text(json.dumps(response_data(val_samples, "val"), indent=2))
    (args.output / "missing.json").write_text(json.dumps(missing, indent=2))

    print(f"CSV rows:         {len(rows)}")
    print(f"Matched samples:  {len(samples)}")
    print(f"Train samples:    {len(train_samples)}")
    print(f"Val samples:      {len(val_samples)}")
    print(f"Missing samples:  {len(missing)}")
    print(f"Output:           {args.output.resolve()}")


if __name__ == "__main__":
    main()
