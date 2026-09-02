#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from domain_gap.utils.gcs_utils import _download


DEFAULT_CSV_PATHS = [
    Path("domain_gap_samples.csv"),
    Path("domian_gap_samples.csv"),
]
DEFAULT_OUTPUT = Path("domain_gap/data/datasets/original_csv_subset")
DEFAULT_INSIGHT_CSV = Path("insight-2-deeplab.csv")
DATASETS = ("cityscapes", "kitti")


def existing_csv_path():
    for path in DEFAULT_CSV_PATHS:
        if path.exists():
            return path
    return DEFAULT_CSV_PATHS[0]


def cityscapes_paths(row):
    city = row["City"]
    filename = row["Filename"]
    return {
        "image_path": f"Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/{city}/{filename}_leftImg8bit.png",
        "gt_path": f"Cityscapes/gtFine_trainvaltest/gtFine/train/{city}/{filename}_gtFine_labelIds.png",
        "gt_image_path": f"Cityscapes/gtFine_trainvaltest/gtFine/train/{city}/{filename}_gtFine_color.png",
        "metadata": f"Cityscapes/vehicle_trainvaltest/vehicle/train/{city}/{filename}_vehicle.json",
        "file_name": filename,
        "city": city,
        "dataset": "cityscapes",
    }


def kitti_paths(row):
    image_path = row["Filename"]
    semantic_path = image_path.replace("/image_2/", "/semantic/")
    semantic_rgb_path = image_path.replace("/image_2/", "/semantic_rgb/")
    return {
        "image_path": image_path,
        "gt_path": semantic_path,
        "gt_image_path": semantic_rgb_path,
        "metadata": "",
        "file_name": image_path,
        "city": row["City"],
        "dataset": "kitti",
    }


def local_path(output, cloud_path):
    return output / "files" / cloud_path


def download_required_files(sample, output):
    downloaded = {}
    missing = []
    for key in ("image_path", "gt_path", "gt_image_path", "metadata"):
        cloud_path = sample[key]
        if not cloud_path:
            downloaded[key] = ""
            continue

        target = local_path(output, cloud_path)
        try:
            downloaded[key] = _download(cloud_path, local_file_path=str(target))
        except Exception as error:
            downloaded[key] = ""
            missing.append({
                "path_type": key,
                "cloud_path": cloud_path,
                "error": str(error),
            })

    required_missing = [item for item in missing if item["path_type"] in ("image_path", "gt_path")]
    if required_missing:
        return None, missing

    return {
        **sample,
        "image_path": downloaded["image_path"],
        "gt_path": downloaded["gt_path"],
        "gt_image_path": downloaded["gt_image_path"],
        "metadata": downloaded["metadata"],
    }, missing


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


def load_low_perf_file_names(path):
    """File names of the insight's low performance root members, which validation must hold."""
    if not path.exists():
        print(f"Insight csv not found, no low performance priority applied: {path}")
        return set()
    rows = csv.DictReader(path.open(newline=""))
    return {row["metadata.filename"] for row in rows if row["is_low_perf_root_member"] == "True"}


def assign_splits(samples, val_percent, kitti_val_share, seed, low_perf, low_perf_val_share):
    """Split stratified by dataset so kitti makes up `kitti_val_share` of validation, while
    validation holds at least `low_perf_val_share` of the low performance samples.
    The csv repeats some file names, so samples are grouped per image and every copy
    of an image stays on the same side of the split."""
    groups = {dataset: {} for dataset in DATASETS}
    for sample in samples:
        groups[sample["dataset"]].setdefault(sample["file_name"], []).append(sample)

    rng = random.Random(seed)
    shuffled = {}
    for dataset in DATASETS:
        shuffled[dataset] = list(groups[dataset].values())
        rng.shuffle(shuffled[dataset])

    sizes = {dataset: sum(len(group) for group in shuffled[dataset]) for dataset in DATASETS}
    val_total = round(len(samples) * val_percent)
    val_sizes = {"kitti": min(round(val_total * kitti_val_share), sizes["kitti"])}
    val_sizes["cityscapes"] = min(max(val_total - val_sizes["kitti"], 0), sizes["cityscapes"])

    train_samples, val_samples = [], []
    for dataset in DATASETS:
        priority = [group for group in shuffled[dataset] if group[0]["file_name"] in low_perf]
        other = [group for group in shuffled[dataset] if group[0]["file_name"] not in low_perf]
        required = math.ceil(sum(len(group) for group in priority) * low_perf_val_share)
        remaining = max(val_sizes[dataset], required)
        if remaining > val_sizes[dataset]:
            print(f"Grew {dataset} validation to {remaining} to fit {required} low performance samples")

        held_back = []
        for group in priority:
            if required > 0 and len(group) <= remaining:
                val_samples += group
                remaining -= len(group)
                required -= len(group)
            else:
                held_back.append(group)

        for group in other + held_back:
            if len(group) <= remaining:
                val_samples += group
                remaining -= len(group)
            else:
                train_samples += group
    return train_samples, val_samples


def describe(samples, low_perf):
    counts = Counter(sample["dataset"] for sample in samples)
    breakdown = ", ".join(f"{dataset} {counts[dataset]}" for dataset in DATASETS)
    low_perf_count = sum(1 for sample in samples if sample["file_name"] in low_perf)
    return f"{len(samples)} ({breakdown}, low perf {low_perf_count})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=existing_csv_path())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--insight-csv", type=Path, default=DEFAULT_INSIGHT_CSV)
    parser.add_argument("--val-percent", type=float, default=0.2)
    parser.add_argument("--kitti-val-share", type=float, default=0.6)
    parser.add_argument("--low-perf-val-share", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.csv.open(newline="")))
    low_perf = load_low_perf_file_names(args.insight_csv)
    samples = []
    missing = []

    for row in rows:
        if row["Dataset"] == "cityscapes":
            sample = cityscapes_paths(row)
        elif row["Dataset"] == "kitti":
            sample = kitti_paths(row)
        else:
            missing.append({
                "dataset": row["Dataset"],
                "filename": row["Filename"],
                "reason": "unsupported dataset",
            })
            continue

        downloaded_sample, sample_missing = download_required_files(sample, args.output)
        if downloaded_sample is None:
            missing.append({
                "dataset": sample["dataset"],
                "filename": sample["file_name"],
                "reason": "missing required image or mask",
                "missing": sample_missing,
            })
            continue

        for item in sample_missing:
            missing.append({
                "dataset": sample["dataset"],
                "filename": sample["file_name"],
                "reason": "missing optional file",
                "missing": [item],
            })

        samples.append(downloaded_sample)

    train_samples, val_samples = assign_splits(
        samples,
        val_percent=args.val_percent,
        kitti_val_share=args.kitti_val_share,
        seed=args.seed,
        low_perf=low_perf,
        low_perf_val_share=args.low_perf_val_share,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "train.json").write_text(json.dumps(response_data(train_samples, "train"), indent=2))
    (args.output / "val.json").write_text(json.dumps(response_data(val_samples, "val"), indent=2))
    (args.output / "missing.json").write_text(json.dumps(missing, indent=2))

    matched_low_perf = {sample["file_name"] for sample in samples} & low_perf
    val_low_perf = sum(1 for sample in val_samples if sample["file_name"] in low_perf)
    print(f"CSV rows:        {len(rows)}")
    print(f"Matched samples: {describe(samples, low_perf)}")
    print(f"Train samples:   {describe(train_samples, low_perf)}")
    print(f"Val samples:     {describe(val_samples, low_perf)}")
    print(f"Low perf in val: {val_low_perf}/{len(matched_low_perf)}")
    print(f"Missing records: {len(missing)}")
    print(f"Output:          {args.output.resolve()}")


if __name__ == "__main__":
    main()
