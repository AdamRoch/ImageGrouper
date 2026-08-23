"""
AutoHDR Challenge — Submission

Reads test images from /input/images/ and writes predictions.csv to /output/.

Algorithm (validated config, see grouper.py):
  SIFT on gamma-normalized grayscale (per-image median luminance -> 128),
  all-pairs knnMatch + Lowe ratio test + RANSAC homography inliers, score
  normalized by sqrt(kp_i * kp_j), then verify-then-merge clustering where a
  candidate joins a group if it verifies (score >= T) against >= 75% of the
  group's current members (T = 0.018).

Local scores: 0.7391 on sample-500 test sim, 0.7513 on medium spot-check.

Contract:
    Input:  /input/images/  — JPEG images from a single photoshoot (read-only)
    Output: /output/predictions.csv — your grouping predictions

predictions.csv format:
    filename,group_id
    IMG_001.jpg,0
    IMG_002.jpg,0
    IMG_003.jpg,1
"""

import csv
import os
import time
from pathlib import Path

import grouper

INPUT_DIR = Path("/input/images")
OUTPUT_DIR = Path("/output")
SUPPORTED = {".jpg", ".jpeg", ".png"}

# Validated on sample-500 (0.7391) and medium spot-check (0.7513)
CONFIG = dict(preprocess="gamma", normalize="norm_sqrt", fraction=0.75, threshold=0.018)


def group_images(image_paths: list[str]) -> list[list[str]]:
    """
    Group images by camera angle.

    Args:
        image_paths: List of absolute file paths to images.

    Returns:
        List of groups. Each group is a list of filenames (basenames only).
    """
    t0 = time.time()
    groups = grouper.group_images(image_paths, workers=os.cpu_count(), **CONFIG)
    print(f"grouping done in {time.time() - t0:.1f}s", flush=True)
    return groups


def main():
    t_start = time.time()

    # Load images
    images = sorted([
        str(p) for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED
    ])
    print(f"Loaded {len(images)} images from {INPUT_DIR}", flush=True)

    # Run grouping
    groups = group_images(images)
    print(f"Predicted {len(groups)} groups", flush=True)

    # Write predictions.csv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "predictions.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "group_id"])
        for group_id, group in enumerate(groups):
            for filename in sorted(group):
                writer.writerow([os.path.basename(filename), group_id])

    print(f"Wrote {sum(len(g) for g in groups)} predictions to {out_path}", flush=True)
    print(f"total runtime {time.time() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
