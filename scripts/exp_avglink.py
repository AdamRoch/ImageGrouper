"""
Experiment 2 — average-linkage clustering variants vs f=0.75 (v3 stack).

Same guarded matrix as v3; only the join rule changes:
  avg_all   — candidate joins iff mean(matrix[idx, members]) >= T
  avg_meas  — mean over members with VALID guard metrics (not NaN) >= T,
              requiring >= 2 measurable links (failed measurement excluded)
  f0.75     — reference: verify-fraction 0.75 (current config)
Screening runs WITHOUT the fuse stage (isolates the clustering variable);
the winner (if any) then gets fuse_rescue applied for the full-stack gate.

Usage:
    .venv/bin/python scripts/exp_avglink.py --images data/sample500/test_sim \
        --manifest data/sample500/test_sim/manifest.csv
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate
import grouper

T = float(__import__("os").environ.get("T_AVG", "0.025"))


def cluster_custom(matrix, threshold, rule, valid=None):
    """verify-then-merge with a pluggable join predicate (merge rule mirrors it)."""
    n = matrix.shape[0]
    groups, membership = [], {}

    def verifies(idx, members):
        scores = matrix[idx, members]
        if rule == "avg_all":
            return scores.mean() >= threshold
        if rule == "avg_meas":
            ok = ~np.isnan(valid[idx, members])
            return ok.sum() >= 2 and scores[ok].mean() >= threshold
        raise ValueError(rule)

    for _, i, j in grouper.candidate_edges(matrix, threshold):
        gi, gj = membership.get(i), membership.get(j)
        if gi is None and gj is None:
            membership[i] = membership[j] = len(groups)
            groups.append([i, j])
        elif gi is None:
            if verifies(i, groups[gj]):
                groups[gj].append(i); membership[i] = gj
        elif gj is None:
            if verifies(j, groups[gi]):
                groups[gi].append(j); membership[j] = gi
        elif gi != gj:
            if all(verifies(a, groups[gj]) for a in groups[gi]):
                groups[gi].extend(groups[gj])
                for m in groups[gj]:
                    membership[m] = gi
                groups[gj] = None
    result = [g for g in groups if g]
    assigned = set(membership)
    result.extend([[i] for i in range(n) if i not in assigned])
    return result


def grade(manifest, names, clusters, out):
    grouper.write_predictions([[names[i] for i in c] for c in clusters], out)
    r = evaluate.evaluate(manifest, out)
    return r


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fuse", action="store_true",
                        help="apply fuse_rescue after clustering (full v3-stack parity)")
    args = parser.parse_args()

    names = sorted(p.name for p in Path(args.images).glob("*.jpg"))
    n = len(names)
    tag = f"{Path(args.images).parent.name}_{Path(args.images).name}"
    matrix = grouper.load_cached_matrix(
        grouper.cache_path_for(args.images, "gamma_norm_sqrt_histeq_guard_c0.5"), names).astype(np.float32)
    d = np.load(Path("output/reverify") / f"guards_{tag}.npz")
    valid = np.full((n, n), np.nan, dtype=np.float32)
    ok = ~np.isnan(d["blob"])
    valid[d["i"][ok], d["j"][ok]] = 1.0
    valid[d["j"][ok], d["i"][ok]] = 1.0

    out_dir = Path("output/avglink")
    out_dir.mkdir(parents=True, exist_ok=True)

    fuse_ctx = None
    if args.fuse:
        import cv2
        paths = [str(Path(args.images) / nm) for nm in names]
        luminance = grouper.load_or_compute_luminance(args.images, names)
        images = [grouper.load_gray(p, preprocess="gamma") for p in paths]
        sift = cv2.SIFT_create()
        feats = [grouper.detect(sift, grouper.load_gray(p, preprocess="gamma")) for p in paths]
        fuse_ctx = (images, luminance, feats)

    print(f"{args.images} ({n} images) — fuse stage {'ON' if args.fuse else 'OFF (screening)'}")
    for rule in ("f0.75", "avg_all", "avg_meas"):
        if rule == "f0.75":
            clusters = grouper.cluster_verify_all(matrix, T, 0.75)
        else:
            clusters = cluster_custom(matrix, T, rule, valid)
        if args.fuse:
            images, luminance, feats = fuse_ctx
            kps = [f[0] for f in feats]
            descs = [f[1] for f in feats]
            clusters = grouper.fuse_rescue(matrix, clusters, images, luminance, kps, descs,
                                           0.022, None)
        r = grade(args.manifest, names, clusters, str(out_dir / f"{tag}_{rule}{'_fuse' if args.fuse else ''}.csv"))
        print(f"  {rule:8s}: {r['score']:.4f} ({r['exact_matches']}/{r['total_reference_groups']}), "
              f"groups {r['total_predicted_groups']}, merges {len(r['false_merges'])}, "
              f"splits {len(r['false_splits'])}")


if __name__ == "__main__":
    main()
