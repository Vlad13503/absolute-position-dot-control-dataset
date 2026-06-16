from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EXPECTED_SPLITS = {
    "position_train": 2304,
    "position_test_iid": 576,
    "position_test_checker": 576,
    "shape_control_train": 2304,
    "shape_control_test": 576,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the generated Absolute Position Dot dataset.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    return parser.parse_args()


def validate_split(path: Path, expected_size: int) -> list[str]:
    errors: list[str] = []
    data = np.load(path)
    required = {"images", "xy", "coords", "position_bin", "shape_id"}
    missing = required - set(data.files)
    if missing:
        return [f"{path.name}: missing arrays {sorted(missing)}"]

    n = data["images"].shape[0]
    if n != expected_size:
        errors.append(f"{path.name}: expected {expected_size} samples, found {n}")
    if data["images"].shape[1:] != (32, 32):
        errors.append(f"{path.name}: expected 32 x 32 images, found {data['images'].shape[1:]}")
    if data["images"].dtype != np.uint8:
        errors.append(f"{path.name}: expected uint8 images, found {data['images'].dtype}")

    position_counts = np.bincount(data["position_bin"], minlength=16)
    if position_counts.min() != position_counts.max():
        errors.append(f"{path.name}: position bins are not balanced: {position_counts.tolist()}")

    shape_counts = np.bincount(data["shape_id"], minlength=2)
    if path.stem.startswith("shape_control") and shape_counts[0] != shape_counts[1]:
        errors.append(f"{path.name}: shape labels are not balanced: {shape_counts.tolist()}")
    if path.stem.startswith("position") and shape_counts[1] != 0:
        errors.append(f"{path.name}: position task should only use square markers: {shape_counts.tolist()}")

    return errors


def main() -> None:
    args = parse_args()
    errors: list[str] = []

    metadata_path = args.data / "metadata.json"
    annotations_path = args.data / "annotations.csv"
    summary_path = args.data / "split_summary.csv"
    for path in (metadata_path, annotations_path, summary_path):
        if not path.exists():
            errors.append(f"Missing required file: {path}")

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("position_classes") != 16:
            errors.append("metadata.json: expected 16 position classes")

    for split_name, expected_size in EXPECTED_SPLITS.items():
        split_path = args.data / f"{split_name}.npz"
        if not split_path.exists():
            errors.append(f"Missing split: {split_path}")
            continue
        errors.extend(validate_split(split_path, expected_size))

    if annotations_path.exists():
        with annotations_path.open(newline="", encoding="utf-8") as f:
            annotation_count = sum(1 for _ in csv.DictReader(f))
        expected_total = sum(EXPECTED_SPLITS.values())
        if annotation_count != expected_total:
            errors.append(f"annotations.csv: expected {expected_total} rows, found {annotation_count}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
