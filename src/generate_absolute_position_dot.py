from __future__ import annotations

import argparse
import csv
import json
import struct
import zlib
from pathlib import Path

import numpy as np


IMAGE_SIZE = 32
GRID_SIZE = 4
MARKER_RADIUS = 1
MARGIN = 2
DEFAULT_SEED = 20260616


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Output directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--samples-per-bin", type=int, default=180)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--noise-prob", type=float, default=0.01)
    return parser.parse_args()


def marker_positions() -> list[tuple[int, int, int]]:
    positions: list[tuple[int, int, int]] = []
    bin_size = IMAGE_SIZE // GRID_SIZE
    for y in range(MARGIN, IMAGE_SIZE - MARGIN):
        for x in range(MARGIN, IMAGE_SIZE - MARGIN):
            bx = min(x // bin_size, GRID_SIZE - 1)
            by = min(y // bin_size, GRID_SIZE - 1)
            positions.append((x, y, by * GRID_SIZE + bx))
    return positions


def draw_marker(shape_id: int, x: int, y: int, rng: np.random.Generator, noise_prob: float) -> np.ndarray:
    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    if shape_id == 0:
        image[y - MARKER_RADIUS : y + MARKER_RADIUS + 1, x - MARKER_RADIUS : x + MARKER_RADIUS + 1] = 255
    elif shape_id == 1:
        image[y, x - MARKER_RADIUS : x + MARKER_RADIUS + 1] = 255
        image[y - MARKER_RADIUS : y + MARKER_RADIUS + 1, x] = 255
    else:
        raise ValueError(f"Unknown shape_id: {shape_id}")

    noise = rng.random((IMAGE_SIZE, IMAGE_SIZE)) < noise_prob
    image[noise] = np.maximum(image[noise], 80)
    return image


def coords_to_normalized(x: int, y: int) -> tuple[float, float]:
    return (2.0 * x / (IMAGE_SIZE - 1) - 1.0, 2.0 * y / (IMAGE_SIZE - 1) - 1.0)


def build_position_split(
    rng: np.random.Generator,
    samples_per_bin: int,
    test_fraction: float,
    noise_prob: float,
) -> dict[str, dict[str, np.ndarray]]:
    positions = marker_positions()
    by_bin = {i: [] for i in range(GRID_SIZE * GRID_SIZE)}
    for x, y, bin_id in positions:
        by_bin[bin_id].append((x, y, bin_id))

    train_rows = []
    iid_test_rows = []
    checker_test_rows = []
    test_count = int(round(samples_per_bin * test_fraction))
    train_count = samples_per_bin - test_count

    for bin_id, choices in by_bin.items():
        choices_arr = np.array(choices, dtype=np.int64)
        iid_idx = rng.choice(len(choices_arr), size=samples_per_bin, replace=True)
        iid_rows = choices_arr[iid_idx]
        rng.shuffle(iid_rows)
        train_rows.extend(iid_rows[:train_count].tolist())
        iid_test_rows.extend(iid_rows[train_count:].tolist())

        checker_choices = [row for row in choices if (row[0] + row[1]) % 2 == 1]
        checker_idx = rng.choice(len(checker_choices), size=test_count, replace=True)
        checker_test_rows.extend(np.array(checker_choices, dtype=np.int64)[checker_idx].tolist())

    return {
        "position_train": rows_to_arrays(train_rows, rng, noise_prob, fixed_shape=0),
        "position_test_iid": rows_to_arrays(iid_test_rows, rng, noise_prob, fixed_shape=0),
        "position_test_checker": rows_to_arrays(checker_test_rows, rng, noise_prob, fixed_shape=0),
    }


def build_shape_control(
    rng: np.random.Generator,
    samples_per_bin: int,
    test_fraction: float,
    noise_prob: float,
) -> dict[str, dict[str, np.ndarray]]:
    positions = marker_positions()
    by_bin = {i: [] for i in range(GRID_SIZE * GRID_SIZE)}
    for x, y, bin_id in positions:
        by_bin[bin_id].append((x, y, bin_id))

    test_count = int(round(samples_per_bin * test_fraction))
    train_rows = []
    test_rows = []
    for bin_id, choices in by_bin.items():
        choices_arr = np.array(choices, dtype=np.int64)
        for shape_id in (0, 1):
            count = samples_per_bin // 2
            idx = rng.choice(len(choices_arr), size=count, replace=True)
            rows = [(int(x), int(y), int(b), shape_id) for x, y, b in choices_arr[idx]]
            rng.shuffle(rows)
            shape_test_count = test_count // 2
            train_rows.extend(rows[:-shape_test_count])
            test_rows.extend(rows[-shape_test_count:])
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return {
        "shape_control_train": rows_to_arrays(train_rows, rng, noise_prob),
        "shape_control_test": rows_to_arrays(test_rows, rng, noise_prob),
    }


def rows_to_arrays(
    rows: list[tuple[int, int, int]] | list[tuple[int, int, int, int]],
    rng: np.random.Generator,
    noise_prob: float,
    fixed_shape: int | None = None,
) -> dict[str, np.ndarray]:
    images = []
    xy = []
    coords = []
    position_bin = []
    shape_id = []
    for row in rows:
        x, y, bin_id = int(row[0]), int(row[1]), int(row[2])
        sid = int(fixed_shape if fixed_shape is not None else row[3])
        images.append(draw_marker(sid, x, y, rng, noise_prob))
        xy.append((x, y))
        coords.append(coords_to_normalized(x, y))
        position_bin.append(bin_id)
        shape_id.append(sid)
    return {
        "images": np.stack(images).astype(np.uint8),
        "xy": np.array(xy, dtype=np.int64),
        "coords": np.array(coords, dtype=np.float32),
        "position_bin": np.array(position_bin, dtype=np.int64),
        "shape_id": np.array(shape_id, dtype=np.int64),
    }


def save_npz(output: Path, name: str, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(output / f"{name}.npz", **arrays)


def label_for_split(split_name: str, arrays: dict[str, np.ndarray]) -> tuple[str, np.ndarray, list[str]]:
    if split_name.startswith("position_"):
        labels = arrays["position_bin"]
        names = [f"position_bin_{int(label)}" for label in labels]
        return "absolute_position", labels, names
    labels = arrays["shape_id"]
    names = ["square" if int(label) == 0 else "plus" for label in labels]
    return "shape_control", labels, names


def write_annotations_csv(output: Path, splits: dict[str, dict[str, np.ndarray]]) -> None:
    path = output / "annotations.csv"
    fieldnames = [
        "sample_id",
        "split",
        "npz_file",
        "npz_index",
        "task",
        "label",
        "label_name",
        "x",
        "y",
        "x_norm",
        "y_norm",
        "position_bin",
        "shape_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, arrays in splits.items():
            task, labels, label_names = label_for_split(split_name, arrays)
            for idx in range(arrays["images"].shape[0]):
                x, y = arrays["xy"][idx]
                x_norm, y_norm = arrays["coords"][idx]
                writer.writerow(
                    {
                        "sample_id": f"{split_name}_{idx:05d}",
                        "split": split_name,
                        "npz_file": f"{split_name}.npz",
                        "npz_index": idx,
                        "task": task,
                        "label": int(labels[idx]),
                        "label_name": label_names[idx],
                        "x": int(x),
                        "y": int(y),
                        "x_norm": f"{float(x_norm):.6f}",
                        "y_norm": f"{float(y_norm):.6f}",
                        "position_bin": int(arrays["position_bin"][idx]),
                        "shape_id": int(arrays["shape_id"][idx]),
                    }
                )


def write_summary_csv(output: Path, splits: dict[str, dict[str, np.ndarray]]) -> None:
    path = output / "split_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "samples",
                "task",
                "position_classes",
                "shape_classes",
                "min_position_count",
                "max_position_count",
                "min_shape_count",
                "max_shape_count",
            ],
        )
        writer.writeheader()
        for split_name, arrays in splits.items():
            task, _, _ = label_for_split(split_name, arrays)
            position_counts = np.bincount(arrays["position_bin"], minlength=GRID_SIZE * GRID_SIZE)
            shape_counts = np.bincount(arrays["shape_id"], minlength=2)
            writer.writerow(
                {
                    "split": split_name,
                    "samples": int(arrays["images"].shape[0]),
                    "task": task,
                    "position_classes": int(np.count_nonzero(position_counts)),
                    "shape_classes": int(np.count_nonzero(shape_counts)),
                    "min_position_count": int(position_counts.min()),
                    "max_position_count": int(position_counts.max()),
                    "min_shape_count": int(shape_counts.min()),
                    "max_shape_count": int(shape_counts.max()),
                }
            )


def write_png(path: Path, image: np.ndarray) -> None:
    """Write a grayscale uint8 PNG using only the standard library."""
    h, w = image.shape
    raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def write_preview_grid(output: Path, split_name: str, arrays: dict[str, np.ndarray], n: int = 16) -> None:
    examples = arrays["images"][:n]
    cols = 4
    pad = 2
    tile = IMAGE_SIZE
    rows = int(np.ceil(n / cols))
    canvas = np.full((rows * tile + (rows - 1) * pad, cols * tile + (cols - 1) * pad), 30, dtype=np.uint8)
    for idx, image in enumerate(examples):
        r, c = divmod(idx, cols)
        y0 = r * (tile + pad)
        x0 = c * (tile + pad)
        canvas[y0 : y0 + tile, x0 : x0 + tile] = image
    write_png(output / "examples" / f"{split_name}_preview.png", canvas)


def write_metadata(output: Path, config: dict[str, object], split_sizes: dict[str, int]) -> None:
    metadata = {
        "name": "Absolute Position Dot",
        "paper": {
            "title": "An intriguing failing of convolutional neural networks and the CoordConv solution",
            "arxiv": "https://arxiv.org/abs/1807.03247",
        },
        "hypothesis": (
            "Models built only from ordinary convolutions struggle when the correct answer "
            "depends on absolute spatial position; explicit coordinate channels should help."
        ),
        "image_size": IMAGE_SIZE,
        "grid_size": GRID_SIZE,
        "position_classes": GRID_SIZE * GRID_SIZE,
        "splits": split_sizes,
        "config": config,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main(default_output: Path | None = None) -> None:
    args = parse_args()
    output = args.output or default_output or Path("data")
    output.mkdir(parents=True, exist_ok=True)
    (output / "examples").mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    config = {
        "seed": args.seed,
        "samples_per_bin": args.samples_per_bin,
        "test_fraction": args.test_fraction,
        "noise_prob": args.noise_prob,
    }
    splits = {}
    splits.update(build_position_split(rng, args.samples_per_bin, args.test_fraction, args.noise_prob))
    splits.update(build_shape_control(rng, args.samples_per_bin, args.test_fraction, args.noise_prob))

    split_sizes = {}
    for name, arrays in splits.items():
        save_npz(output, name, arrays)
        write_preview_grid(output, name, arrays)
        split_sizes[name] = int(arrays["images"].shape[0])

    write_annotations_csv(output, splits)
    write_summary_csv(output, splits)
    write_metadata(output, config, split_sizes)
    print(f"Wrote {len(splits)} splits to {output.resolve()}")


if __name__ == "__main__":
    main()
