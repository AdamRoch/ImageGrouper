"""
Merge-guard measurements on candidate pairs (score >= T on the histeq-upgraded
matrix). One shared re-match pass feeds both levers:

  Lever A (door-check residual): align via the estimated homography, tone-match,
  absdiff, largest coherent blob fraction of the residual map. Hypothesis:
  cross-group links (repositioned camera) show large coherent/diffuse residual;
  true same-angle links show only JPEG noise.

  Lever B (camera-translation test): essential-matrix fit on the matched points
  (assumed K: f = max image dim, pp = center). Same-position brackets are
  rotation-dominant (degenerate essential fit -> low inlier ratio); repositioned
  pairs fit well. Also records recoverPose cheirality ratio.

Pairs with raw-luminance gap >= 1.3 are re-matched on the histeq-adjusted
darker side (mirroring the reverify pass); others use base gamma features.

Labels (in_group) come from the manifest — measurement only. Saves
output/reverify/guards_<tag>.npz and prints separability reports.

Usage:
    .venv/bin/python scripts/merge_guards.py --images data/sample500/test_sim \
        --manifest data/sample500/test_sim/manifest.csv
"""

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grouper
from grouper import detect, hist_match, match_features

T = 0.022            # clustering threshold on the upgraded matrix
RESID_NOISE = 25     # gray levels — above JPEG re-encode noise
MIN_GAP = grouper.REVERIFY_MIN_GAP

# per-pair feature store for the worker (indices into shared lists)
_IMGS = None
_FEATS = None
_LUM = None


def pair_metrics(i, j, bf, sift):
    dark, bright = (i, j) if _LUM[i] <= _LUM[j] else (j, i)
    lo, hi = _LUM[dark], _LUM[bright]
    if hi / max(lo, 1e-6) >= MIN_GAP:
        img_d = hist_match(_IMGS[dark], _IMGS[bright])
        pts_d, d_d = detect(sift, img_d)
    else:
        img_d = _IMGS[dark]
        pts_d, d_d = _FEATS[dark]
    img_b = _IMGS[bright]
    pts_b, d_b = _FEATS[bright]

    nan = (np.nan, np.nan, np.nan)
    if len(d_d) < 2 or len(d_b) < 2:
        return nan
    good = []
    for pair in bf.knnMatch(d_d, d_b, k=2):
        if len(pair) == 2:
            m, n = pair
            if m.distance < grouper.RATIO_TEST * n.distance:
                good.append(m)
    if len(good) < grouper.MIN_GOOD_MATCHES:
        return nan
    src = pts_d[[m.queryIdx for m in good]]
    dst = pts_b[[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, grouper.RANSAC_THRESH)
    if H is None or mask is None or mask.sum() < grouper.MIN_GOOD_MATCHES:
        return nan
    inl = mask.ravel().astype(bool)
    src_i, dst_i = src[inl], dst[inl]

    # Lever A: residual coherent-blob fraction after alignment
    warped = cv2.warpPerspective(img_b, H, (img_d.shape[1], img_d.shape[0]))
    warped_tm = hist_match(warped, img_d)  # tone safety
    diff = cv2.absdiff(img_d, warped_tm)
    above = (diff > RESID_NOISE).astype(np.uint8)
    ncomp, _, stats, _ = cv2.connectedComponentsWithStats(above, connectivity=8)
    blob_frac = (stats[1:, cv2.CC_STAT_AREA].max() / diff.size) if ncomp > 1 else 0.0

    # Lever B: essential-matrix signatures on the homography inliers
    h, w = img_d.shape
    K = np.array([[max(w, h), 0, w / 2], [0, max(w, h), h / 2], [0, 0, 1]], dtype=np.float64)
    E, mask_e = cv2.findEssentialMat(src_i, dst_i, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask_e is None:
        ess_ratio, cheir = np.nan, np.nan
    else:
        ess_ratio = float(mask_e.sum()) / len(src_i)
        try:
            _, _, _, mask_rp = cv2.recoverPose(E, src_i, dst_i, K, mask=mask_e)
            cheir = float((mask_rp > 0).sum()) / len(src_i)
        except cv2.error:
            cheir = np.nan

    return blob_frac, ess_ratio, cheir


def _run_batch(batch):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    sift = cv2.SIFT_create()
    return [(i, j, *pair_metrics(i, j, bf, sift)) for i, j in batch]


def report(name, values, in_group, higher_is_cross=True):
    ok = ~np.isnan(values)
    pos, neg = values[ok & in_group], values[ok & ~in_group]
    print(f"  {name}: valid on {ok.sum()}/{len(values)} pairs")
    if len(pos) and len(neg):
        for lbl, v in (("true ", pos), ("cross", neg)):
            print(f"    {lbl}: median {np.median(v):.3f} p25 {np.percentile(v,25):.3f} "
                  f"p75 {np.percentile(v,75):.3f} p90 {np.percentile(v,90):.3f}")
        rng = np.random.default_rng(0)
        draws = 200_000
        cross_s = neg[rng.integers(0, len(neg), draws)]
        true_s = pos[rng.integers(0, len(pos), draws)]
        auc = (cross_s > true_s).mean() + 0.5 * (cross_s == true_s).mean()
        print(f"    AUC P(cross > true): {auc:.3f} (higher = better separation)")


def main():
    global _IMGS, _FEATS, _LUM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--min-score", type=float, default=T,
                        help=f"measure pairs with score >= this (default {T}; "
                             f"0.3xT covers the rescue band)")
    args = parser.parse_args()

    paths = sorted(Path(args.images).glob("*.jpg"))
    names = [p.name for p in paths]
    n = len(names)
    matrix = grouper.load_cached_matrix(
        grouper.cache_path_for(args.images, "gamma_norm_sqrt_histeq"), names).astype(np.float32)
    _LUM = grouper.load_or_compute_luminance(args.images, names)

    f2g = {r["filename"]: r["group_id"] for r in csv.DictReader(open(args.manifest))}
    gids = np.array([f2g[nm] for nm in names])

    iu = np.triu_indices(n, 1)
    cand = matrix[iu] >= args.min_score
    pairs = list(zip(iu[0][cand].tolist(), iu[1][cand].tolist()))
    print(f"{n} images, {len(pairs):,} candidate pairs >= {args.min_score}")

    t0 = time.time()
    _IMGS = [grouper.load_gray(str(p), preprocess="gamma") for p in paths]
    print(f"  loaded {n} gamma images ({time.time()-t0:.1f}s)", flush=True)
    t0 = time.time()
    sift = cv2.SIFT_create()
    _FEATS = [detect(sift, img) for img in _IMGS]
    print(f"  base detections ({time.time()-t0:.1f}s)", flush=True)

    batches = [pairs[k:k + 200] for k in range(0, len(pairs), 200)]
    t0 = time.time()
    done, results = 0, []
    with ThreadPoolExecutor() as pool:
        for out in pool.map(_run_batch, batches):
            results.extend(out)
            done += len(out)
            if done % 2000 < 200 or done == len(pairs):
                print(f"  pairs {done}/{len(pairs)} ({time.time()-t0:.1f}s)", flush=True)

    ii = np.array([r[0] for r in results])
    jj = np.array([r[1] for r in results])
    blob = np.array([r[2] for r in results])
    essr = np.array([r[3] for r in results])
    cheir = np.array([r[4] for r in results])
    in_group = gids[ii] == gids[jj]

    out_dir = Path("output/reverify")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.images).parent.name}_{Path(args.images).name}"
    np.savez(out_dir / f"guards_{tag}.npz", i=ii, j=jj, blob=blob, ess=essr,
             cheir=cheir, in_group=in_group)

    print(f"\n=== separability on {len(pairs)} candidates (true {in_group.sum()}, cross {(~in_group).sum()}) ===")
    report("A: residual blob fraction", blob, in_group, higher_is_cross=True)
    report("B: essential inlier ratio", essr, in_group, higher_is_cross=True)
    report("B: cheirality ratio", cheir, in_group, higher_is_cross=True)


if __name__ == "__main__":
    main()
