"""
Lever F — fused-stack orphan rescue.

After the final-config clustering (guarded matrix, verify_all f=0.75), fuse
each >=2-member group into an all-detail exposure composite (MergeMertens on
the gamma-preprocessed images; AlignMTB is broken in cv2 5.0.0 — segfault —
and brackets are same-viewpoint with ~px jitter, so fusion runs unaligned).
Orphans (singleton clusters) are matched against composites:
histeq-equalized when the luminance gap demands, RANSAC inliers normalized
norm_sqrt-style by (orphan kp x composite kp).

Composite score only NOMINATES: a join requires the cheirality guard (<0.5)
against at least one real member of the group (merge_guards machinery).

Modes:
  default: measurement only — saves output/reverify/fuse_<tag>.npz and prints
           orphan->true-group vs orphan->wrong-group separability.
  --apply T_FUSE: also join orphans with composite score >= T_FUSE that pass
           the guard, write predictions, grade.

Usage:
    .venv/bin/python scripts/fuse_rescue.py --images data/sample500/test_sim \
        --manifest data/sample500/test_sim/manifest.csv [--apply 0.022]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate
import grouper
from grouper import detect, hist_match, match_features
import merge_guards

T_CLUSTER = 0.025
MIN_GAP = grouper.REVERIFY_MIN_GAP


def fuse_group(member_imgs):
    """MergeMertens exposure fusion of a list of grayscale uint8 images.

    All inputs must share dimensions — resize to the group's modal size
    (a dimension-mixed group is already partially suspect; the composite is
    only a nomination signal, mild resize distortion is acceptable).
    """
    from collections import Counter
    (w, h), _ = Counter((im.shape[1], im.shape[0]) for im in member_imgs).most_common(1)[0]
    member_imgs = [im if (im.shape[1], im.shape[0]) == (w, h)
                   else cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
                   for im in member_imgs]
    fusion = cv2.createMergeMertens().process(member_imgs)
    return (np.clip(fusion, 0, 1) * 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--threshold", type=float, default=T_CLUSTER)
    parser.add_argument("--apply", type=float, default=None, metavar="T_FUSE")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    T = args.threshold

    paths = sorted(Path(args.images).glob("*.jpg"))
    names = [p.name for p in paths]
    n = len(names)
    tag = f"{Path(args.images).parent.name}_{Path(args.images).name}"

    matrix = grouper.load_cached_matrix(
        grouper.cache_path_for(args.images, "gamma_norm_sqrt_histeq_guard_c0.5"), names)
    lum = grouper.load_or_compute_luminance(args.images, names)
    f2g = {r["filename"]: r["group_id"] for r in csv.DictReader(open(args.manifest))}

    clusters = grouper.cluster_verify_all(matrix, T, 0.75)
    groups = [c for c in clusters if len(c) >= 2]
    orphans = [c[0] for c in clusters if len(c) == 1]
    print(f"{n} images -> {len(groups)} multi-member groups, {len(orphans)} orphans "
          f"(final-config clustering @ T={T})")

    t0 = time.time()
    images = [grouper.load_gray(str(p), preprocess="gamma") for p in paths]
    sift = cv2.SIFT_create()
    feats = [detect(sift, img) for img in images]
    print(f"  loaded + detected {n} images ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    composites, comp_feats, comp_lum = [], [], []
    for k, g in enumerate(groups, 1):
        comp = fuse_group([images[m] for m in g])
        pts, desc = detect(sift, comp)
        composites.append(comp)
        comp_feats.append((pts, desc))
        comp_lum.append(float(comp.mean()))
        if k % 25 == 0 or k == len(groups):
            print(f"  fused {k}/{len(groups)} composites ({time.time()-t0:.1f}s)", flush=True)

    # --- orphan x composite matching ---
    def pair_score(oi, gi, bf, s):
        pts_o, d_o = feats[oi]
        pts_c, d_c = comp_feats[gi]
        lo, hi = sorted((lum[oi], comp_lum[gi]))
        if hi / max(lo, 1e-6) >= MIN_GAP:
            # equalize the darker side onto the brighter
            if lum[oi] < comp_lum[gi]:
                adj = hist_match(images[oi], composites[gi])
                pts_o, d_o = detect(s, adj)
            else:
                adj = hist_match(composites[gi], images[oi])
                pts_c, d_c = detect(s, adj)
        inl = match_features(pts_o, d_o, pts_c, d_c, bf)
        return inl / float(np.sqrt(max(len(pts_o), 1) * max(len(pts_c), 1)))

    work = [(oi, gi) for oi in orphans for gi in range(len(groups))]
    batches = [work[k:k + 500] for k in range(0, len(work), 500)]
    t0 = time.time()
    results = {}

    def run_batch(batch):
        bf = cv2.BFMatcher(cv2.NORM_L2)
        s = cv2.SIFT_create()
        return [(oi, gi, pair_score(oi, gi, bf, s)) for oi, gi in batch]

    done = 0
    with ThreadPoolExecutor() as pool:
        for out in pool.map(run_batch, batches):
            for oi, gi, sc in out:
                results[(oi, gi)] = sc
            done += len(out)
            if done % 5000 < 500 or done == len(work):
                print(f"  orphan x composite {done}/{len(work)} ({time.time()-t0:.1f}s)", flush=True)

    # --- labels: positive if the composite's group contains a member of the orphan's true group ---
    orphan_true = np.array([f2g[names[oi]] for oi in orphans])
    group_gids = [set(f2g[names[m]] for m in g) for g in groups]
    pos_mask = np.array([[f2g[names[oi]] in group_gids[gi] for gi in range(len(groups))]
                         for oi in orphans])

    scores = np.array([[results[(oi, gi)] for gi in range(len(groups))] for oi in orphans])
    pos, neg = scores[pos_mask], scores[~pos_mask]
    print(f"\norphan x composite: {scores.size} tests, positives {pos.size}, negatives {neg.size}")
    if pos.size:
        for lbl, v in (("true-group ", pos), ("wrong-group", neg)):
            print(f"  {lbl}: median {np.median(v):.4f} p75 {np.percentile(v,75):.4f} "
                  f"p90 {np.percentile(v,90):.4f} max {v.max():.4f}")
        rng = np.random.default_rng(0)
        d200 = 200_000
        a = pos[rng.integers(0, len(pos), d200)]
        b = neg[rng.integers(0, len(neg), d200)]
        print(f"  AUC P(true > wrong): {(a > b).mean() + 0.5*(a == b).mean():.3f}")
        for tf in (0.012, 0.018, 0.022, 0.025, 0.03):
            print(f"  T_fuse={tf:g}: true >= {int((pos >= tf).sum())}/{pos.size}, "
                  f"wrong >= {int((neg >= tf).sum())}/{neg.size}")

    out_dir = Path("output/reverify")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"fuse_{tag}.npz",
             orphans=np.array(orphans), scores=scores, pos_mask=pos_mask,
             groups=np.array([np.array(g) for g in groups], dtype=object))

    if args.apply is None:
        return

    # --- apply: nominations above T_fuse, cheirality guard vs a real member ---
    t_fuse = args.apply
    merge_guards._IMGS = images
    merge_guards._FEATS = feats
    merge_guards._LUM = lum
    bf = cv2.BFMatcher(cv2.NORM_L2)

    nominations = []
    for k, oi in enumerate(orphans):
        for gi in range(len(groups)):
            if scores[k, gi] >= t_fuse:
                nominations.append((scores[k, gi], oi, gi))
    nominations.sort(reverse=True)
    print(f"\napply T_fuse={t_fuse:g}: {len(nominations)} nominations", flush=True)

    joined = {}
    for sc, oi, gi in nominations:
        if oi in joined:
            continue
        # guard: cheirality vs members, strongest-matrix-score members first
        members = sorted(groups[gi], key=lambda m: -matrix[oi, m])
        ok = False
        for m in members[:5]:
            _, _, cheir = merge_guards.pair_metrics(oi, m, bf, sift)
            if not np.isnan(cheir) and cheir < 0.5:
                ok = True
                break
        if ok:
            joined[oi] = gi
    print(f"  passed guard: {len(joined)} orphans join", flush=True)

    new_clusters = [list(c) for c in clusters]
    gid_of = {}
    for k, c in enumerate(new_clusters):
        for m in c:
            gid_of[m] = k
    # map composite index -> cluster index (composite gi <-> groups[gi] subset of one cluster)
    for oi, gi in joined.items():
        target_cluster = gid_of[groups[gi][0]]
        new_clusters[gid_of[oi]] = [m for m in new_clusters[gid_of[oi]] if m != oi]
        new_clusters[target_cluster].append(oi)
    new_clusters = [c for c in new_clusters if c]

    out_path = args.out or f"output/guards/predictions_{tag}_fuse{args.apply:g}.csv"
    grouper.write_predictions([[names[m] for m in c] for c in new_clusters], out_path)
    r = evaluate.evaluate(args.manifest, out_path)
    print(f"score {r['score']:.4f} ({r['exact_matches']}/{r['total_reference_groups']}), "
          f"merges {len(r['false_merges'])}, splits {len(r['false_splits'])}")


if __name__ == "__main__":
    main()
