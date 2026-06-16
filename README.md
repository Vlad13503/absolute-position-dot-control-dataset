# Absolute Position Dot

Controlled dataset for testing whether image models can use **absolute spatial position**.

This dataset is motivated by:

Rosanne Liu et al., **An intriguing failing of convolutional neural networks and the CoordConv solution**, NeurIPS 2018.  
https://arxiv.org/abs/1807.03247

## Hypothesis

Ordinary convolutional networks can struggle when the correct label depends on absolute spatial position. CoordConv should help because it adds explicit coordinate channels.

## Dataset

Each sample is a 32 x 32 grayscale image with one small marker.

- Main task: predict the marker's absolute 4 x 4 position bin.
- Control task: predict marker shape, while position is balanced and irrelevant.

The generated dataset is in `data/`:

- `*.npz`: image arrays and labels for ML use.
- `annotations.csv`: one row per sample for inspection.
- `split_summary.csv`: class-balance summary.
- `metadata.json`: dataset configuration and hypothesis.
- `examples/*.png`: preview images.

## Generate

```bash
python main.py
```

## Validate

```bash
python validate.py
```

Expected output:

```text
Dataset validation passed.
```

## Code

- `src/generate_absolute_position_dot.py`: dataset generator.
- `src/validate_dataset.py`: validation checks.
