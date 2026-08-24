"""
Score-level re-verify levers on the borderline band of the cached
gamma_norm_sqrt matrix. One lever per run.

Band: pairs scoring [lo*T, hi*T] on the current matrix (score-only — no
manifest knowledge, so the apply step transfers to unlabeled data).

Levers:
  histeq    — pair-directed exposure equalization: for band pairs with a raw
              luminance gap >= --min-gap, histogram-match the darker image to
              the brighter one's tonal distribution (beyond the global
              median->128 gamma), re-detect SIFT on the adjusted darker side,
              re-match against the brighter side's base features.
  relaxsift — relaxed-SIFT re-verify: re-detect every image once with a lower
              contrastThreshold (more keypoints in low-contrast regions),
              re-match band pairs with those features.

In both cases the pair score is RANSAC inliers / sqrt(kp_i*kp_j) from the
lever's own detections, and the upgraded matrix keeps max(old, new) per pair.
Separability (true in-group links vs cross-group links in the band, labels
from the manifest) is measured on the NEW scores and saved to output/reverify/.

Usage:
    .venv/bin/python scripts/reverify_band.py --images data/sample500/test_sim \
        --manifest data/sample500/test_sim/manifest.csv --lever relaxsift
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grouper
from grouper import detect, hist_match, match_features

T = 0.018  # current config threshold on the normalized matrix
RATIO = grouper.RATIO_TEST


def auc(pos, neg, seed=0, draws=200_000):
    """P(pos > neg) + 0.5*P(tie), Monte Carlo (tie-safe)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    p = pos[rng.integers(0, len(pos), draws)]
    n = neg[rng.integers(0, len(neg), draws)]
    return float((p > n).mean() + 0.5 * (p == n).mean())


def separability_report(label, old, new, in_group, T):
    print(f"\nseparability [{label}] — within-group band pairs {in_group.sum()}, "
          f"cross-group {(~in_group).sum()}")
    for name, vals in (("old", old), ("new", new)):
        pos, neg = vals[in_group], vals[~in_group]
        print(f"  {name}: within-group >=T {100*(pos >= T).mean():5.1f}% (median {np.median(pos):.4f}) | "
              f"cross-group >=T {100*(neg >= T).mean():5.1f}% (median {np.median(neg):.4f}) | "
              f"AUC {auc(pos, neg):.3f}")
    lift_pos = float((new[in_group] - old[in_group]).mean()) if in_group.sum() else float("nan")
    lift_neg = float((new[~in_group] - old[~in_group]).mean())
    print(f"  mean uplift: within-group {lift_pos:+.4f}, cross-group {lift_neg:+.4f} "
          f"(want within >> cross)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lever", choices=("histeq", "relaxsift"), required=True)
    parser.add_argument("--band-lo", type=float, default=0.3, help="band lower edge, xT")
    parser.add_argument("--band-hi", type=float, default=1.2, help="band upper edge, xT")
    parser.add_argument("--min-gap", type=float, default=1.3, help="histeq: min luminance ratio")
    parser.add_argument("--contrast", type=float, default=0.02, help="relaxsift: contrastThreshold")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    workers = args.workers or __import__("os").cpu_count() or 4

    names = sorted(p.name for p in Path(args.images).glob("*.jpg"))
    idx = {nm: i for i, nm in enumerate(names)}
    n = len(names)
    matrix = grouper.load_cached_matrix(
        grouper.cache_path_for(args.images, "gamma_norm_sqrt"), names).astype(np.float32)
    lum = grouper.load_or_compute_luminance(args.images, names)

    f2g = {r["filename"]: r["group_id"] for r in csv.DictReader(open(args.manifest))}
    gids = np.array([f2g[nm] for nm in names])

    iu = np.triu_indices(n, 1)
    vals = matrix[iu]
    band_mask = (vals >= args.band_lo * T) & (vals <= args.band_hi * T)
    band = list(zip(iu[0][band_mask].tolist(), iu[1][band_mask].tolist()))
    print(f"{n} images, band [{args.band_lo}*T, {args.band_hi}*T] = {len(band):,} pairs")

    t0 = time.time()
    print(f"loading gamma-normalized images...", flush=True)
    images = [grouper.load_gray(str(Path(args.images) / nm), preprocess="gamma") for nm in names]
    print(f"  loaded {n} images ({time.time()-t0:.1f}s)")

    # --- lever setup + per-image detection pass ---
    if args.lever == "relaxsift":
        sift = cv2.SIFT_create(contrastThreshold=args.contrast)
        t0 = time.time()
        feats = []
        for k, img in enumerate(images, 1):
            feats.append(detect(sift, img))
            if k % 200 == 0 or k == n:
                print(f"  relaxed detect {k}/{n} ({time.time()-t0:.1f}s)", flush=True)
    else:  # histeq: base features for the BRIGHTER side; darker side is per-pair
        sift = cv2.SIFT_create()
        t0 = time.time()
        feats = []
        for k, img in enumerate(images, 1):
            feats.append(detect(sift, img))
            if k % 200 == 0 or k == n:
                print(f"  base detect {k}/{n} ({time.time()-t0:.1f}s)", flush=True)

    # --- band re-match ---
    if args.lever == "relaxsift":
        work = band
    else:
        work = [(i, j) for i, j in band
                if max(lum[i], lum[j]) / max(min(lum[i], lum[j]), 1e-6) >= args.min_gap]
        print(f"histeq: {len(work):,} of {len(band):,} band pairs have luminance gap >= {args.min_gap}")

    results = [None] * len(work)

    if args.lever == "relaxsift":
        def run_batch(batch):
            bf = cv2.BFMatcher(cv2.NORM_L2)
            out = []
            for k, (i, j) in batch:
                pts_i, d_i = feats[i]
                pts_j, d_j = feats[j]
                inl = match_features(pts_i, d_i, pts_j, d_j, bf)
                ki, kj = max(len(pts_i), 1), max(len(pts_j), 1)
                out.append((k, inl / float(np.sqrt(ki * kj))))
            return out
    else:
        def run_batch(batch):
            bf = cv2.BFMatcher(cv2.NORM_L2)
            s = cv2.SIFT_create()
            out = []
            for k, (i, j) in batch:
                dark, bright = (i, j) if lum[i] <= lum[j] else (j, i)
                adj = hist_match(images[dark], images[bright])
                pts_d, d_d = detect(s, adj)
                pts_b, d_b = feats[bright]
                inl = match_features(pts_d, d_d, pts_b, d_b, bf)
                kd, kb = max(len(pts_d), 1), max(len(pts_b), 1)
                out.append((k, inl / float(np.sqrt(kd * kb))))
            return out

    batches = [[(k, work[k]) for k in range(b, min(b + 500, len(work)))]
               for b in range(0, len(work), 500)]
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for out in pool.map(run_batch, batches):
            for k, score in out:
                results[k] = score
            done += len(out)
            if done % 10000 < 500 or done == len(work):
                print(f"  re-match {done}/{len(work)} ({time.time()-t0:.1f}s)", flush=True)

    # --- separability measurement ---
    rematched = np.array([k for k, (i, j) in enumerate(work)], dtype=int)
    ii = np.array([work[k][0] for k in rematched])
    jj = np.array([work[k][1] for k in rematched])
    old = matrix[ii, jj]
    new = np.array([results[k] for k in rematched], dtype=np.float32)
    in_group = gids[ii] == gids[jj]
    separability_report(f"{args.lever}", old, new, in_group, T)

    out_dir = Path("output/reverify")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"measure_{args.lever}_{Path(args.images).parent.name}_{Path(args.images).name}.npz",
             i=ii, j=jj, old=old, new=new, in_group=in_group)

    # --- upgrade matrix: max(old, new) on re-matched pairs ---
    upgraded = matrix.copy()
    upgraded[ii, jj] = np.maximum(old, new)
    upgraded[jj, ii] = upgraded[ii, jj]
    tag = f"gamma_norm_sqrt_{args.lever}"
    grouper.save_cached_matrix(grouper.cache_path_for(args.images, tag), names, upgraded)
    print(f"upgraded matrix saved with tag {tag!r} — grade with:\n"
          f"  .venv/bin/python scripts/sweep_thresholds.py --images {args.images} "
          f"--manifest {args.manifest} --variant {tag} -f 0.75 --thresholds 0.018")


if __name__ == "__main__":
    main()
