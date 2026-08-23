"""
Sweep the inlier threshold T on the cached pairwise score matrix and grade
each value with the local scorer. No re-matching — clustering reads the
cache written by grouper.py.

Usage:
    .venv/bin/python scripts/sweep_thresholds.py \
        --images data/sample500/test_sim \
        --manifest data/sample500/test_sim/manifest.csv \
        --thresholds 10,20,30,40,60,100 [--policy verify_all]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate
import grouper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--thresholds", required=True, help="comma-separated, e.g. 10,20,30,40,60,100")
    parser.add_argument("--policy", choices=sorted(grouper.POLICIES), default="verify_all")
    parser.add_argument("--out-dir", default="output/sweep")
    args = parser.parse_args()

    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    names = [p.name for p in paths]
    matrix = grouper.load_cached_matrix(grouper.cache_path_for(args.images), names)
    if matrix is None:
        sys.exit("no cache found — run grouper.py on this image dir first")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"policy={args.policy}  ({len(names)} images)")
    print(f"{'T':>6} {'score':>7} {'exact':>10} {'pred_groups':>11} {'false_merge':>11} {'false_split':>11}")
    for t in [float(x) for x in args.thresholds.split(",")]:
        clusters = grouper.POLICIES[args.policy](matrix, t)
        groups = [[names[i] for i in c] for c in clusters]
        pred_path = out_dir / f"predictions_{args.policy}_T{t:g}.csv"
        grouper.write_predictions(groups, str(pred_path))
        r = evaluate.evaluate(args.manifest, str(pred_path))
        print(f"{t:>6g} {r['score']:>7.4f} "
              f"{r['exact_matches']:>5}/{r['total_reference_groups']:<4} "
              f"{r['total_predicted_groups']:>11} "
              f"{len(r['false_merges']):>11} {len(r['false_splits']):>11}")


if __name__ == "__main__":
    main()
