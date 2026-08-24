"""
Sweep the score threshold T on a cached pairwise score matrix and grade
each value with the local scorer. No re-matching — clustering reads the
cache written by grouper.py (or a derived matrix saved under a variant tag).

Usage:
    .venv/bin/python scripts/sweep_thresholds.py \
        --images data/sample500/test_sim \
        --manifest data/sample500/test_sim/manifest.csv \
        --thresholds 10,20,30,40,60,100 \
        [--variant ""] [--policy verify_all] [--fraction 1.0] [--label NAME]
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
    parser.add_argument("--variant", default="", help="cache tag: '', gamma, clahe, norm_min, ...")
    parser.add_argument("--policy", choices=sorted(grouper.POLICIES), default="verify_all")
    parser.add_argument("--fraction", "-f", type=float, default=1.0,
                        help="verify-fraction for verify_all joins")
    parser.add_argument("--label", default=None, help="label for output CSVs (default: derived)")
    parser.add_argument("--out-dir", default="output/sweep")
    args = parser.parse_args()

    paths = sorted(p for p in Path(args.images).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    names = [p.name for p in paths]
    matrix = grouper.load_cached_matrix(grouper.cache_path_for(args.images, args.variant), names)
    if matrix is None:
        sys.exit(f"no cache found for variant {args.variant!r} — build it first")

    luminance = None
    if args.policy in ("chain", "rescue", "rescue_guarded"):
        luminance = grouper.load_or_compute_luminance(args.images, names)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = args.label or f"{args.variant or 'raw'}_{args.policy}" + \
        (f"_f{args.fraction:g}" if args.fraction != 1.0 else "")
    print(f"variant={args.variant or 'raw'}  policy={args.policy}  f={args.fraction:g}  ({len(names)} images)")
    print(f"{'T':>8} {'score':>7} {'exact':>10} {'pred_groups':>11} {'false_merge':>11} {'false_split':>11}")
    for t in [float(x) for x in args.thresholds.split(",")]:
        if args.policy in ("chain", "rescue", "rescue_guarded"):
            clusters = grouper.POLICIES[args.policy](matrix, t, luminance, args.fraction)
        elif args.policy == "verify_all" and args.fraction != 1.0:
            clusters = grouper.cluster_verify_all(matrix, t, args.fraction)
        else:
            clusters = grouper.POLICIES[args.policy](matrix, t)
        groups = [[names[i] for i in c] for c in clusters]
        pred_path = out_dir / f"predictions_{label}_T{t:g}.csv"
        grouper.write_predictions(groups, str(pred_path))
        r = evaluate.evaluate(args.manifest, str(pred_path))
        print(f"{t:>8g} {r['score']:>7.4f} "
              f"{r['exact_matches']:>5}/{r['total_reference_groups']:<4} "
              f"{r['total_predicted_groups']:>11} "
              f"{len(r['false_merges']):>11} {len(r['false_splits']):>11}")


if __name__ == "__main__":
    main()
