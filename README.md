# Cell Division Pressure Pair Ranking

| | |
| --- | --- |
| Final rank | not ranked |
| Domain | Computer Vision |
| Difficulty | Medium |
| Scoring | ↓ Lower is better |
| Compute | A10G |
| Challenge status | Accepted / closed |
| Solutions submitted | 4 |
| Last submission | 2026-06-27 |

## Problem statement

### Overview

Time-lapse microscopy lets researchers study which local cell neighborhoods are likely to remodel or divide over the next day. In this challenge, you are given pairs of cropped 0 h microscopy images from plant-cell fields. For each pair, predict whether the left crop has higher hidden 24 h division pressure than the right crop.

The label is not a simple visible cell count. It is derived from temporal lineage annotations that track mother-cell contours at 0 h to daughter-cell contours at 24 h, combined with local crowding and annotated cell area. The test set is grouped by held-out plant fields, so robust solutions need to learn image morphology cues that transfer across fields rather than memorize crop-specific texture or acquisition artifacts.

This task is practical for plant-cell microscopy because division-prone regions are often reviewed manually before researchers decide where to focus follow-up annotation, imaging, or phenotyping effort. A model that can rank local 0 h neighborhoods by likely next-day division pressure would help prioritize time-lapse review, flag active growth zones, and compare strain behavior without requiring full lineage annotation for every new field.

### Dataset

### File descriptions

- `train.csv` -- Labeled image-pair rows. Each row contains two crop image paths and the binary target `left_higher_division_pressure`.
- `test.csv` -- Unlabeled image-pair rows with the same image path columns but without the target.
- `sample_submission.csv` -- Template submission with random probability values for the required target column.
- `train/` -- Hashed grayscale PNG crop images used by `train.csv`.
- `test/` -- Hashed grayscale PNG crop images used by `test.csv`.

### Column descriptions

- `id` (string) -- Unique 12-character hex identifier for each image pair.
- `left_image_path` (string) -- Relative path to the left crop image inside `dataset/public/`.
- `right_image_path` (string) -- Relative path to the right crop image inside `dataset/public/`.
- `left_higher_division_pressure` (integer) -- Target in `train.csv` only. `1` means the left crop has higher hidden division pressure than the right crop; `0` means the right crop is higher.
- `pair_weight` (float) -- Training row weight derived from how separated the hidden pressure scores are. Larger weights identify clearer pairwise comparisons.
- `prob_left_higher_division_pressure` (float) -- Submission probability that the left crop has higher hidden division pressure.

### Evaluation

Submissions are scored using weighted log loss. Lower is better. The returned score is `100 * weighted_log_loss`, clipped to the range `[0, 100]`.

```
import numpy as np

def evaluate(y_true, y_pred, pair_weight):

    clipped = np.clip(y_pred, 1e-6, 1.0 - 1e-6)

    loss = -(y_true  *np.log(clipped) + (1.0 - y_true)*  np.log1p(-clipped))

    return min(100.0, 100.0  *float(np.sum(pair_weight*  loss) / np.sum(pair_weight)))
```

### Submission

Submit a CSV file with one probability for every row in `test.csv`.

- `id` (string) -- The pair identifier from `test.csv`.
- `prob_left_higher_division_pressure` (float) -- Probability from 0 to 1 inclusive.

Example:

```
id,prob_left_higher_division_pressure

0f6c9d2f0c34,0.638

3a918c7b12d0,0.421

8d7702b9fa1e,0.714
```

### Requirements

- The file must contain exactly the same number of rows as `test.csv`, plus the header.
- Every `id` from `test.csv` must be present exactly once.
- Probabilities must be finite numeric values between 0 and 1 inclusive.
- File format must be `.csv` with exact columns `id,prob_left_higher_division_pressure`.

### What Not To Use

- Do not use manually written lookup tables or cached answers for the public test crop pairs.
- Do not attempt to recover original source filenames, plant folder names, or time-point labels from the hashed public crop images.
- Do not use reverse image search, web search, or external copies of matching time-lapse annotation files to recover hidden lineage labels.
- Do not build a purely rule-based contour-counting shortcut as the main prediction method; the intended task is learned visual ranking under held-out plant-field shift.
