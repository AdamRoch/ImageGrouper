"""
Lever 1 — normalized pair score (no re-matching).

Re-runs feature extraction only (~9s) to get per-image keypoint counts, then
normalizes the cached raw inlier matrix two ways:
  (a) norm_min  = inliers / min(kp_i, kp_j)
  (b) norm_sqrt = inliers / sqrt(kp_i * kp_j)
Each normalized matrix is saved as its own cache variant (tags norm_min /
norm_sqrt) so sweep_thresholds.py --variant can grade threshold sweeps.

Usage:
    .venv/bin/python scripts/lever1_normalized.py --images data/sample500/test_sim
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grouper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--preprocess", default="none", choices=grouper.PREPROCESSORS,
                        help="preprocessing used for the keypoint-count extraction")
    parser.add_argument("--base-variant", default="",
                        help="cache tag of the inlier matrix to normalize (default: raw)")
    args = parser.parse_args()

    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    names = [p.name for p in paths]

    raw = grouper.load_cached_matrix(grouper.cache_path_for(args.images, args.base_variant), names)
    if raw is None:
        sys.exit(f"no cache found for variant {args.base_variant!r} — run grouper.py first")

    base_key = grouper.cache_path_for(args.images, args.base_variant).stem[7:]
    counts_path = grouper.CACHE_DIR / f"kpcounts_{base_key}{'_' + args.preprocess if args.preprocess != 'none' else ''}.npy"
    if counts_path.exists():
        counts = np.load(counts_path)
        print(f"  loaded cached keypoint counts from {counts_path}")
    else:
        _, kps, _ = grouper.extract_features([str(p) for p in paths], preprocess=args.preprocess)
        counts = np.array([len(k) for k in kps], dtype=np.int32)
        np.save(counts_path, counts)
        print(f"  saved keypoint counts to {counts_path}")

    print(f"keypoints per image: min {counts.min()}, median {int(np.median(counts))}, max {counts.max()}")

    prefix = f"{args.base_variant}_" if args.base_variant else ""
    safe = counts.astype(np.float32).clip(1)
    variants = {
        f"{prefix}norm_min": raw / np.minimum.outer(safe, safe),
        f"{prefix}norm_sqrt": raw / np.sqrt(np.outer(safe, safe)),
    }
    for tag, matrix in variants.items():
        matrix = matrix.astype(np.float32)
        np.fill_diagonal(matrix, 0)
        off = matrix[np.triu_indices(len(names), 1)]
        nonzero = off[off > 0]
        print(f"{tag}: nonzero pairs {len(nonzero)}/{len(off)}, "
              f"p50 {np.median(nonzero):.4f}, p90 {np.percentile(nonzero, 90):.4f}, "
              f"p99 {np.percentile(nonzero, 99):.4f}, max {nonzero.max():.4f}")
        grouper.save_cached_matrix(grouper.cache_path_for(args.images, tag), names, matrix)


if __name__ == "__main__":
    main()
