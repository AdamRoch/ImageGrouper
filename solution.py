"""
AutoHDR Challenge — Submission

Reads test images from /input/images/ and writes predictions.csv to /output/.

Algorithm (validated config, see grouper.py):
  SIFT on gamma-normalized grayscale (per-image median luminance -> 128),
  all-pairs knnMatch + Lowe ratio test + RANSAC homography inliers, score
  normalized by sqrt(kp_i * kp_j); then a pair-directed exposure
  equalization re-verify pass (borderline band 0.3-1.2 x 0.018 with raw
  luminance gap >= 1.3: histogram-match the darker image onto the brighter
  one's tonal distribution, re-detect, max-upgrade the pair score); then a
  camera-translation guard (pairs re-matched + pose-decomposed down to
  0.3xT, cheirality ratio >= 0.5 -> edge blocked); then AVERAGE-OVER-
  MEASURABLE clustering (candidate joins iff mean score over links with a
  valid measurement >= T_avg = 0.025, >= 2 measurable links, failed
  measurements excluded from the mean); finally a fused-stack orphan rescue
  (MergeMertens composites nominate, cheirality guard gates,
  T_fuse = 0.022).

Local scores: 0.9275 sample-500 test sim / 0.8808 medium spot / 0.9159
holdout.

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

# v5 config: 0.9275 sample / 0.8808 spot / 0.9159 holdout (validated)
CONFIG = dict(preprocess="gamma", normalize="norm_sqrt", reverify="histeq",
              guard="cheirality", fuse_threshold=0.022, policy="avg_meas",
              threshold=0.025)


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
