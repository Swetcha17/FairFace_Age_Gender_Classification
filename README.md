# FairFace Multi-Task Age & Gender Classification

This project implements a **multi-task deep learning framework** for simultaneous **Age Group Classification** and **Gender Classification** using the **FairFace dataset** in PyTorch.

It supports multiple modern backbones, class imbalance handling, stratified splitting, mixed precision training, and automatic benchmark logging.

---

## Features

- Multi-task learning (shared backbone + dual classification heads)
- Class-weighted loss for imbalance handling
- Stratified validation/test split (age + gender)
- Mixed precision training (AMP)
- Cosine annealing learning rate scheduler
- Confusion matrix export (CSV)
- JSON metrics report
- Cross-backbone benchmark logging

---

## Supported Backbones

- `resnet50`
- `convnext_tiny`
- `efficientnet_v2_s`
- `vit_b_16`

All models use ImageNet pretrained weights by default.

---

## Project Structure

```
project_root/
│
├── add fairface-dataset
├── fairface_label_train.csv
├── fairface_label_val.csv
└── train.py

```

The script automatically searches for images inside:

- `root/`
- `root/train/`
- `root/val/`
- `root/images/`

---

## Dataset Format

CSV must contain:

| Column | Description |
|--------|------------|
| file (or image_path / img / path / filename) | Relative image path |
| age | Age group label |
| gender | Gender label |

### Age Classes

```

0-2
3-9
10-19
20-29
30-39
40-49
50-59
60-69
more than 70

```

### Gender Classes

```

Male
Female

````

Common variants like `m`, `f`, `70+` are automatically normalized.

---

## Installation

Create environment:

```bash
conda create -n fairface python=3.10
conda activate fairface
````

Install dependencies:

```bash
pip install torch torchvision pandas numpy scikit-learn pillow
```

---

## Training

Basic training:

```bash
python train.py \
    --root /path/to/fairface \
    --backbone resnet50 \
    --epochs 20 \
    --batch_size 128
```

Full example:

```bash
python train.py \
    --root /data/fairface \
    --train_csv fairface_label_train.csv \
    --val_csv fairface_label_val.csv \
    --backbone efficientnet_v2_s \
    --epochs 30 \
    --batch_size 64 \
    --lr 3e-4 \
    --weight_decay 1e-4 \
    --out_dir runs/efficientnet_exp \
    --results_csv runs/benchmark_results.csv
```

---

## Automatic Data Split

The provided validation CSV is split into:

* 80% → validation
* 20% → test

Stratified by:

```
age + gender
```

---

## Loss Function

Final training loss:

```
L = 0.5 * AgeLoss + 0.5 * GenderLoss
```

Both losses use class-weighted CrossEntropy.

---

## Outputs

Inside `out_dir`:

```
best.pt
best_<backbone>_seed<seed>.pt
confusion_matrix_age.csv
confusion_matrix_gender.csv
report.json
```

### Example report.json

```json
{
  "age_labels": [...],
  "gender_labels": [...],
  "test_metrics": {
    "loss": 0.84,
    "age_acc": 0.71,
    "age_f1m": 0.68,
    "gen_acc": 0.93,
    "gen_f1m": 0.92
  }
}
```

---

## Benchmark Logging

Each run appends results to:

```
runs/benchmark_results.csv
```

This enables comparison across:

* Backbones
* Hyperparameters
* Random seeds

---

## Metrics Reported

For both tasks:

* Accuracy
* Macro F1-score
* Confusion Matrix

Model selection is based on:

```
(val_age_f1 + val_gender_f1) / 2
```

---

## Mixed Precision

Enabled automatically if CUDA is available.

Disable with:

```bash
--no_amp
```

---

## Reproducibility

* Fixed random seed
* Deterministic CuDNN
* Stratified splitting

Set seed with:

```bash
--seed 42
```

---

## Notes

* EfficientNetV2-S uses 384×384 input resolution
* Other backbones use 224×224
* Class imbalance handled via inverse-frequency weighting
* Confusion matrices exported as CSV

---
