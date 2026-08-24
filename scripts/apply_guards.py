"""
Apply merge-guard rules to a histeq-upgraded matrix: zero out candidate edges
(score >= T) whose guard metric exceeds a cutoff, save as a new cache variant,
then grade via the normal sweep.

Usage:
    .venv/bin/python scripts/apply_guards.py --images data/sample500/test_sim \
        --blob-cut 0.35            # lever A
    .venv/bin/python scripts/apply_guards.py --images data/sample500/test_sim \
        --cheir-cut 0.5            # lever B
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grouper

T = 0.022


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--blob-cut", type=float, default=None)
    parser.add_argument("--cheir-cut", type=float, default=None)
    args = parser.parse_args()
    if args.blob_cut is None and args.cheir_cut is None:
        sys.exit("give at least one cutoff")

    paths = sorted(Path(args.images).glob("*.jpg"))
    names = [p.name for p in paths]
    tag_in = f"{Path(args.images).parent.name}_{Path(args.images).name}"
    d = np.load(Path("output/reverify") / f"guards_{tag_in}.npz")
    ii, jj = d["i"], d["j"]

    blocked = np.zeros(len(ii), dtype=bool)
    if args.blob_cut is not None:
        blocked |= d["blob"] >= args.blob_cut
    if args.cheir_cut is not None:
        blocked |= d["cheir"] >= args.cheir_cut
    blocked &= ~np.isnan(d["blob"]) & ~np.isnan(d["cheir"])  # no evidence -> keep

    matrix = grouper.load_cached_matrix(
        grouper.cache_path_for(args.images, "gamma_norm_sqrt_histeq"), names).astype(np.float32)
    out = matrix.copy()
    for k in np.flatnonzero(blocked):
        out[ii[k], jj[k]] = out[jj[k], ii[k]] = 0.0

    tag = "gamma_norm_sqrt_histeq_guard"
    if args.blob_cut is not None:
        tag += f"_b{args.blob_cut:g}"
    if args.cheir_cut is not None:
        tag += f"_c{args.cheir_cut:g}"
    grouper.save_cached_matrix(grouper.cache_path_for(args.images, tag), names, out)
    print(f"blocked {blocked.sum()} candidate edges "
          f"(true {int((blocked & d['in_group']).sum())}, cross {int((blocked & ~d['in_group']).sum())})")
    print(f"grade with: --variant {tag}")


if __name__ == "__main__":
    main()
