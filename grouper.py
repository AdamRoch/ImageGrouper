"""
Classical baseline for the AutoHDR image-grouping challenge.

Pipeline:
  1. Per image: grayscale at <=1024px, SIFT detectAndCompute.
  2. All-pairs matching: knnMatch(k=2) + Lowe ratio test, then RANSAC
     homography; the pairwise same-angle score is the RANSAC inlier count.
  3. Clustering on the cached score matrix (default: verify-then-merge —
     a candidate joins a group only if it verifies, inliers >= T, against
     ALL current members). Unassigned images become singletons.

Public API (used by solution.py later):
    group_images(image_paths) -> list of groups of basenames

CLI:
    python grouper.py --images data/sample500/test_sim --out output/predictions.csv
                      [--threshold 40] [--policy verify_all|single_link]
                      [--workers N] [--no-cache]

The pairwise score matrix is cached to .grouper_cache/ (keyed by the image
dir) so threshold/clustering experiments don't recompute matching.
"""

import argparse
import bisect
import csv
import hashlib
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

# --- matching parameters (calibrated on the labeled sample set) ---
MATCH_MAX_DIM = 1024      # images are already downscaled by prep; this is a safety cap
RATIO_TEST = 0.75         # Lowe ratio
RANSAC_THRESH = 5.0       # px reprojection threshold for findHomography
MIN_GOOD_MATCHES = 8      # below this, don't bother with RANSAC
DEFAULT_THRESHOLD = 40    # inlier count separating "same angle" from "different angle"

# Re-verify band anchor (pair-directed exposure equalization, the "histeq"
# lever): the band is defined as [0.3, 1.2] x this reference T on the
# normalized matrix. Calibrated on the workshop sets; deliberately NOT the
# clustering threshold (0.022) so the band reproduces the validated runs.
BAND_REF_T = 0.018
REVERIFY_MIN_GAP = 1.3    # min raw-luminance ratio for a pair to be re-verified
RESID_NOISE = 25          # residual-map gray level counted as "above JPEG noise"
CHEIRALITY_CUT = 0.5      # cheirality ratio above which an edge reads as camera translation

CACHE_DIR = Path(".grouper_cache")

# exposure-invariant preprocessing variants (experiment levers; "none" is the baseline)
PREPROCESSORS = ("none", "gamma", "clahe")
GAMMA_TARGET_MEDIAN = 128.0

# --- matching state (set by compute_score_matrix before the thread pool starts) ---
_KP = None      # list of float32 keypoint coordinate arrays, one per image
_DESC = None    # list of float32 descriptor arrays
_RATIO = RATIO_TEST


def _gamma_normalize(img: np.ndarray) -> np.ndarray:
    """Stretch each image's median luminance to a canonical target via a gamma LUT."""
    median = min(max(float(np.median(img)), 1.0), 254.0)
    gamma = math.log(GAMMA_TARGET_MEDIAN / 255.0) / math.log(median / 255.0)
    lut = np.clip((np.arange(256) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def load_gray(path: str, max_dim: int = MATCH_MAX_DIM, preprocess: str = "none") -> np.ndarray:
    if preprocess not in PREPROCESSORS:
        raise ValueError(f"unknown preprocess {preprocess!r}, expected one of {PREPROCESSORS}")
    img = cv2.imread(path, cv2.IMREAD_COLOR if preprocess == "clahe" else cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    if preprocess == "gamma":
        return _gamma_normalize(img)
    if preprocess == "clahe":
        luminance = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luminance)
    return img


def extract_features(paths: list[str], max_dim: int = MATCH_MAX_DIM, preprocess: str = "none"):
    """Return (names, keypoint arrays, descriptor arrays). Sequential — fast enough (~0.1s/img)."""
    sift = cv2.SIFT_create()
    names, kps, descs = [], [], []
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        img = load_gray(p, max_dim, preprocess)
        kp, desc = sift.detectAndCompute(img, None)
        pts = np.array([k.pt for k in kp], dtype=np.float32) if kp else np.zeros((0, 2), np.float32)
        names.append(os.path.basename(p))
        kps.append(pts)
        descs.append(desc if desc is not None else np.zeros((0, 128), np.float32))
        if i % 50 == 0 or i == len(paths):
            print(f"  features: {i}/{len(paths)} images ({time.time() - t0:.1f}s)", flush=True)
    return names, kps, descs


def pair_inliers(i: int, j: int, bf: cv2.BFMatcher) -> int:
    """RANSAC homography inlier count between image i and j (0 if too few matches)."""
    d1, d2 = _DESC[i], _DESC[j]
    if len(d1) < 2 or len(d2) < 2:
        return 0
    good = []
    for pair in bf.knnMatch(d1, d2, k=2):
        if len(pair) == 2:
            m, n = pair
            if m.distance < _RATIO * n.distance:
                good.append(m)
    if len(good) < MIN_GOOD_MATCHES:
        return 0
    src = _KP[i][[m.queryIdx for m in good]]
    dst = _KP[j][[m.trainIdx for m in good]]
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
    return int(mask.sum()) if mask is not None else 0


def _match_batch(batch):
    bf = cv2.BFMatcher(cv2.NORM_L2)  # one matcher per batch — not thread-safe to share
    return [(i, j, pair_inliers(i, j, bf)) for i, j in batch]


def compute_score_matrix(kps, descs, ratio: float = RATIO_TEST, workers: int | None = None) -> np.ndarray:
    """
    Symmetric NxN matrix of pairwise RANSAC inlier counts, parallelized over pairs.

    Uses threads, not processes: cv2 matching/homography calls release the GIL,
    and threads share the feature arrays with no copying. (A fork-based process
    pool was tried first — it deadlocks on macOS with OpenCV loaded.)
    """
    global _KP, _DESC, _RATIO
    _KP, _DESC, _RATIO = kps, descs, ratio

    n = len(kps)
    matrix = np.zeros((n, n), dtype=np.int32)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    batches = [pairs[k:k + 500] for k in range(0, len(pairs), 500)]
    workers = workers or os.cpu_count() or 4

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(workers) as pool:
        for result in pool.map(_match_batch, batches):
            for i, j, score in result:
                matrix[i, j] = matrix[j, i] = score
            done += len(result)
            if done % 5000 < 500 or done == len(pairs):
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  matching: {done}/{len(pairs)} pairs "
                      f"({rate:.0f} pairs/s, {time.time() - t0:.1f}s)", flush=True)
    return matrix


# --- pair-directed exposure equalization re-verify ("histeq" lever) ---
#
# Band pairs (borderline scores with a big raw-luminance gap) are re-matched
# after histogram-matching the darker image onto the brighter one's tonal
# distribution. Reference: scripts/reverify_band.py, which now imports these
# helpers. Validated: sample 0.7391 -> 0.8116, spot 0.7513 -> 0.7927,
# holdout 0.7449 -> 0.8145 (T=0.022, f=0.75, max-upgrade semantics).

def hist_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Map src's tonal distribution onto ref's (CDF matching, uint8)."""
    s = np.bincount(src.ravel(), minlength=256).astype(np.float64)
    r = np.bincount(ref.ravel(), minlength=256).astype(np.float64)
    scdf = np.cumsum(s) / s.sum()
    rcdf = np.cumsum(r) / r.sum()
    lut = np.interp(scdf, rcdf, np.arange(256))
    return cv2.LUT(src, np.clip(lut, 0, 255).astype(np.uint8))


def detect(sift, img):
    """detectAndCompute -> (float32 Nx2 points, float32 Nx128 descriptors)."""
    kp, desc = sift.detectAndCompute(img, None)
    pts = np.array([k.pt for k in kp], dtype=np.float32) if kp else np.zeros((0, 2), np.float32)
    return pts, (desc if desc is not None else np.zeros((0, 128), np.float32))


def match_features(pts_i, d_i, pts_j, d_j, bf, ratio: float = RATIO_TEST) -> int:
    """RANSAC homography inlier count for pre-computed features."""
    if len(d_i) < 2 or len(d_j) < 2:
        return 0
    good = []
    for pair in bf.knnMatch(d_i, d_j, k=2):
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
    if len(good) < MIN_GOOD_MATCHES:
        return 0
    src = pts_i[[m.queryIdx for m in good]]
    dst = pts_j[[m.trainIdx for m in good]]
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
    return int(mask.sum()) if mask is not None else 0


def reverify_histeq(matrix: np.ndarray, image_paths: list[str], luminance,
                    kps, descs, band_ref_t: float = BAND_REF_T,
                    band_lo: float = 0.3, band_hi: float = 1.2,
                    min_gap: float = REVERIFY_MIN_GAP,
                    workers: int | None = None) -> np.ndarray:
    """
    Max-upgrade re-verify of the borderline band of a NORMALIZED matrix:
    pairs scoring [band_lo, band_hi] x band_ref_t with raw-luminance gap
    >= min_gap are re-matched after histogram equalization of the darker
    side; the pair's score becomes max(old, new). Score-only band — no
    manifest knowledge. kps/descs are the base (gamma) detections, reused as
    the brighter side's features.
    """
    luminance = np.asarray(luminance, dtype=np.float64)
    n = len(image_paths)
    iu = np.triu_indices(n, 1)
    vals = matrix[iu]
    in_band = (vals >= band_lo * band_ref_t) & (vals <= band_hi * band_ref_t)
    pairs = []
    for i, j in zip(iu[0][in_band].tolist(), iu[1][in_band].tolist()):
        lo, hi = min(luminance[i], luminance[j]), max(luminance[i], luminance[j])
        if hi / max(lo, 1e-6) >= min_gap:
            pairs.append((int(i), int(j)))
    print(f"  reverify: {len(pairs):,} band pairs (of {len(vals):,}) qualify", flush=True)
    if not pairs:
        return matrix

    t0 = time.time()
    images = [load_gray(p, preprocess="gamma") for p in image_paths]
    print(f"  reverify: loaded {n} gamma images ({time.time() - t0:.1f}s)", flush=True)

    def run_batch(batch):
        bf = cv2.BFMatcher(cv2.NORM_L2)
        sift = cv2.SIFT_create()
        out = []
        for i, j in batch:
            dark, bright = (i, j) if luminance[i] <= luminance[j] else (j, i)
            pts_d, d_d = detect(sift, hist_match(images[dark], images[bright]))
            inl = match_features(pts_d, d_d, kps[bright], descs[bright], bf)
            kd, kb = max(len(pts_d), 1), max(len(kps[bright]), 1)
            out.append((i, j, inl / float(np.sqrt(kd * kb))))
        return out

    batches = [pairs[k:k + 500] for k in range(0, len(pairs), 500)]
    workers = workers or os.cpu_count() or 4
    t0 = time.time()
    done = 0
    upgraded = matrix.copy()
    with ThreadPoolExecutor(workers) as pool:
        for result in pool.map(run_batch, batches):
            for i, j, score in result:
                if score > upgraded[i, j]:
                    upgraded[i, j] = upgraded[j, i] = score
            done += len(result)
            if done % 10000 < 500 or done == len(pairs):
                print(f"  reverify: {done}/{len(pairs)} pairs ({time.time() - t0:.1f}s)",
                      flush=True)
    print(f"  reverify: done in {time.time() - t0:.1f}s", flush=True)
    return upgraded


# --- camera-translation guard ("cheirality" guard) ---
#
# Candidate edges (score >= T) are re-matched and pose-decomposed; an edge
# whose matched points triangulate in front of both cameras (cheirality ratio
# >= CHEIRALITY_CUT) reads as a repositioned camera and is blocked (zeroed).
# Same-viewpoint brackets are rotation-dominant: their cheirality is ~0.
# Validated: holdout merges 38 -> 9 with only 10/25,267 true edges blocked.

def guard_pair_metrics(i, j, images, feats, luminance, bf, sift):
    """
    (blob_frac, ess_inlier_ratio, cheirality_ratio) for one pair, re-matched
    with homography + essential-matrix pose. Pairs with raw-luminance gap
    >= REVERIFY_MIN_GAP are re-matched on the histeq-adjusted darker side
    (mirrors the reverify pass). NaN = no evidence (never blocks).
    """
    dark, bright = (i, j) if luminance[i] <= luminance[j] else (j, i)
    lo, hi = luminance[dark], luminance[bright]
    if hi / max(lo, 1e-6) >= REVERIFY_MIN_GAP:
        pts_d, d_d = detect(sift, hist_match(images[dark], images[bright]))
    else:
        pts_d, d_d = feats[dark]
    pts_b, d_b = feats[bright]

    nan = (np.nan, np.nan, np.nan)
    if len(d_d) < 2 or len(d_b) < 2:
        return nan
    good = []
    for pair in bf.knnMatch(d_d, d_b, k=2):
        if len(pair) == 2:
            m, n_ = pair
            if m.distance < RATIO_TEST * n_.distance:
                good.append(m)
    if len(good) < MIN_GOOD_MATCHES:
        return nan
    src = pts_d[[m.queryIdx for m in good]]
    dst = pts_b[[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
    if H is None or mask is None or mask.sum() < MIN_GOOD_MATCHES:
        return nan
    inl = mask.ravel().astype(bool)
    src_i, dst_i = src[inl], dst[inl]

    img_d = images[dark]
    warped = cv2.warpPerspective(images[bright], H, (img_d.shape[1], img_d.shape[0]))
    diff = cv2.absdiff(img_d, hist_match(warped, img_d))  # tone safety
    above = (diff > RESID_NOISE).astype(np.uint8)
    ncomp, _, stats, _ = cv2.connectedComponentsWithStats(above, connectivity=8)
    blob_frac = (stats[1:, cv2.CC_STAT_AREA].max() / diff.size) if ncomp > 1 else 0.0

    h, w = img_d.shape
    K = np.array([[max(w, h), 0, w / 2], [0, max(w, h), h / 2], [0, 0, 1]], dtype=np.float64)
    E, mask_e = cv2.findEssentialMat(src_i, dst_i, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None or mask_e is None:
        return blob_frac, np.nan, np.nan
    ess_ratio = float(mask_e.sum()) / len(src_i)
    try:
        _, _, _, mask_rp = cv2.recoverPose(E, src_i, dst_i, K, mask=mask_e)
        cheir = float((mask_rp > 0).sum()) / len(src_i)
    except cv2.error:
        cheir = np.nan
    return blob_frac, ess_ratio, cheir


def guards_cache_path_for(images_dir: str, min_score: float) -> Path:
    key = hashlib.md5(os.path.abspath(images_dir).encode()).hexdigest()[:12]
    return CACHE_DIR / f"guards_{key}_s{min_score:g}.npz"


def compute_guard_metrics(matrix: np.ndarray, image_paths: list[str], images, luminance,
                          kps, descs, min_score: float, workers: int | None = None,
                          images_dir: str | None = None):
    """
    Guard metrics (blob, cheirality) for every pair scoring >= min_score.
    Returns (ii, jj, blob, cheir) index/value arrays. Caches to
    .grouper_cache/ when images_dir is given (keyed like the score matrices).
    """
    n = len(image_paths)
    names = [os.path.basename(p) for p in image_paths]
    if images_dir is not None:
        cache = guards_cache_path_for(images_dir, min_score)
        if cache.exists():
            d = np.load(cache)
            if d["names"].tolist() == names:
                print(f"  guard: loaded cached metrics from {cache}", flush=True)
                return d["i"], d["j"], d["blob"], d["cheir"]

    iu = np.triu_indices(n, 1)
    sel = matrix[iu] >= min_score
    pairs = list(zip(iu[0][sel].tolist(), iu[1][sel].tolist()))
    print(f"  guard: {len(pairs):,} candidate pairs >= {min_score:g}", flush=True)
    if not pairs:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([]), np.array([])

    feats = list(zip(kps, descs))

    def run_batch(batch):
        bf = cv2.BFMatcher(cv2.NORM_L2)
        sift = cv2.SIFT_create()
        return [(i, j, *guard_pair_metrics(i, j, images, feats, luminance, bf, sift))
                for i, j in batch]

    batches = [pairs[k:k + 200] for k in range(0, len(pairs), 200)]
    workers = workers or os.cpu_count() or 4
    t0 = time.time()
    done = 0
    results = []
    with ThreadPoolExecutor(workers) as pool:
        for out in pool.map(run_batch, batches):
            results.extend(out)
            done += len(out)
            if done % 2000 < 200 or done == len(pairs):
                print(f"  guard: {done}/{len(pairs)} pairs ({time.time() - t0:.1f}s)", flush=True)

    ii = np.array([r[0] for r in results], dtype=np.int64)
    jj = np.array([r[1] for r in results], dtype=np.int64)
    blob = np.array([r[2] for r in results], dtype=np.float32)
    cheir = np.array([r[4] for r in results], dtype=np.float32)
    if images_dir is not None:
        np.savez(guards_cache_path_for(images_dir, min_score),
                 names=np.array(names), i=ii, j=jj, blob=blob, cheir=cheir)
    return ii, jj, blob, cheir


def apply_cheirality_guard(matrix: np.ndarray, ii, jj, cheir,
                           cut: float = CHEIRALITY_CUT,
                           block_floor: float | None = None) -> np.ndarray:
    """
    Zero candidate edges whose cheirality ratio >= cut (camera moved).
    Only edges scoring >= block_floor are zeroed (default: block everything
    measured). v3 used block_floor = clustering threshold.
    """
    guarded = matrix.copy()
    ok = ~np.isnan(cheir)
    if block_floor is not None:
        ok &= matrix[ii, jj] >= block_floor
    n_blocked = int((ok & (cheir >= cut)).sum())
    for k in np.flatnonzero(ok & (cheir >= cut)):
        guarded[ii[k], jj[k]] = guarded[jj[k], ii[k]] = 0.0
    print(f"  guard: blocked {n_blocked:,} candidate edges (cheirality >= {cut:g})", flush=True)
    return guarded


# --- fused-stack orphan rescue (post-clustering stage) ---
#
# Each confident (>= 2-member) group is fused into one exposure composite
# (MergeMertens, UNALIGNED — AlignMTB segfaults in opencv-headless 5.0.0, and
# same-viewpoint brackets carry ~px jitter). Orphans (singleton clusters)
# match against composites; a composite score only NOMINATES — the join
# requires the cheirality guard against a real member. Validated:
# sample 0.8551 -> 0.8696, spot 0.8187 -> 0.8446, holdout 0.8870 -> 0.8957.

def fuse_composites(groups: list[list[int]], images) -> tuple[list, list, list]:
    """One unaligned MergeMertens composite per group + SIFT features + mean luminance."""
    sift = cv2.SIFT_create()
    composites, feats, lums = [], [], []
    t0 = time.time()
    for k, g in enumerate(groups, 1):
        member_imgs = [images[m] for m in g]
        # MergeMertens requires uniform dimensions: resize to the modal size
        from collections import Counter
        (w, h), _ = Counter((im.shape[1], im.shape[0]) for im in member_imgs).most_common(1)[0]
        member_imgs = [im if (im.shape[1], im.shape[0]) == (w, h)
                       else cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
                       for im in member_imgs]
        fusion = cv2.createMergeMertens().process(member_imgs)
        comp = (np.clip(fusion, 0, 1) * 255).astype(np.uint8)
        composites.append(comp)
        feats.append(detect(sift, comp))
        lums.append(float(comp.mean()))
        if k % 50 == 0 or k == len(groups):
            print(f"  fuse: {k}/{len(groups)} composites ({time.time() - t0:.1f}s)", flush=True)
    return composites, feats, lums


def fuse_rescue(matrix: np.ndarray, clusters: list[list[int]], images, luminance,
                kps, descs, t_fuse: float, workers: int | None = None) -> list[list[int]]:
    """
    Rescue stage: match each orphan (singleton cluster) against every fused
    group composite; nominations with norm_sqrt score >= t_fuse join their
    strongest group only if the cheirality guard (< CHEIRALITY_CUT) passes
    against at least one of the group's 5 strongest-linked real members.
    One join per orphan; nominations processed strongest-first.
    """
    groups = [c for c in clusters if len(c) >= 2]
    orphans = [c[0] for c in clusters if len(c) == 1]
    print(f"  fuse: {len(groups)} multi-member groups, {len(orphans)} orphans", flush=True)
    if not groups or not orphans:
        return clusters

    composites, comp_feats, comp_lum = fuse_composites(groups, images)
    feats = list(zip(kps, descs))
    luminance = np.asarray(luminance, dtype=np.float64)

    def pair_score(oi, gi, bf, sift):
        pts_o, d_o = feats[oi]
        pts_c, d_c = comp_feats[gi]
        lo, hi = sorted((luminance[oi], comp_lum[gi]))
        if hi / max(lo, 1e-6) >= REVERIFY_MIN_GAP:
            if luminance[oi] < comp_lum[gi]:
                pts_o, d_o = detect(sift, hist_match(images[oi], composites[gi]))
            else:
                pts_c, d_c = detect(sift, hist_match(composites[gi], images[oi]))
        inl = match_features(pts_o, d_o, pts_c, d_c, bf)
        return inl / float(np.sqrt(max(len(pts_o), 1) * max(len(pts_c), 1)))

    work = [(oi, gi) for oi in orphans for gi in range(len(groups))]
    batches = [work[k:k + 500] for k in range(0, len(work), 500)]
    workers = workers or os.cpu_count() or 4

    def run_batch(batch):
        bf = cv2.BFMatcher(cv2.NORM_L2)
        sift = cv2.SIFT_create()
        return [(oi, gi, pair_score(oi, gi, bf, sift)) for oi, gi in batch]

    t0 = time.time()
    done = 0
    results = {}
    with ThreadPoolExecutor(workers) as pool:
        for out in pool.map(run_batch, batches):
            for oi, gi, sc in out:
                results[(oi, gi)] = sc
            done += len(out)
            if done % 5000 < 500 or done == len(work):
                print(f"  fuse: {done}/{len(work)} orphan x composite tests "
                      f"({time.time() - t0:.1f}s)", flush=True)

    nominations = sorted(((sc, oi, gi) for (oi, gi), sc in results.items() if sc >= t_fuse),
                         reverse=True)
    print(f"  fuse: {len(nominations)} nominations >= {t_fuse:g}", flush=True)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    sift = cv2.SIFT_create()
    joined = {}
    for sc, oi, gi in nominations:
        if oi in joined:
            continue
        members = sorted(groups[gi], key=lambda m: -matrix[oi, m])
        for m in members[:5]:
            _, _, cheir = guard_pair_metrics(oi, m, images, feats, luminance, bf, sift)
            if not np.isnan(cheir) and cheir < CHEIRALITY_CUT:
                joined[oi] = gi
                break
    print(f"  fuse: {len(joined)} orphans pass the cheirality guard", flush=True)

    member_of = {}
    for k, c in enumerate(clusters):
        for m in c:
            member_of[m] = k
    new_clusters = [list(c) for c in clusters]
    for oi, gi in joined.items():
        target = member_of[groups[gi][0]]
        new_clusters[member_of[oi]].remove(oi)
        new_clusters[target].append(oi)
    return [c for c in new_clusters if c]


def cache_path_for(images_dir: str, tag: str = "") -> Path:
    key = hashlib.md5(os.path.abspath(images_dir).encode()).hexdigest()[:12]
    suffix = f"_{tag}" if tag else ""
    return CACHE_DIR / f"scores_{key}{suffix}.npz"


def load_cached_matrix(path: Path, names: list[str]) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.load(path)
    cached_names = data["names"].tolist()
    if cached_names != names:
        print(f"  cache {path} is stale (file list changed), ignoring")
        return None
    print(f"  loaded cached score matrix from {path}")
    return data["matrix"]


def save_cached_matrix(path: Path, names: list[str], matrix: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, names=np.array(names), matrix=matrix)
    print(f"  saved score matrix cache to {path}")


def luminance_path_for(images_dir: str) -> Path:
    key = hashlib.md5(os.path.abspath(images_dir).encode()).hexdigest()[:12]
    return CACHE_DIR / f"luminance_{key}.npy"


def load_or_compute_luminance(images_dir: str, names: list[str]) -> np.ndarray:
    """
    Raw mean grayscale luminance per image (no exposure preprocessing — the
    chain policy needs the true exposure ordering). Cached per image dir;
    array aligned with `names` (sorted dir listing, same as score matrices).
    """
    path = luminance_path_for(images_dir)
    if path.exists():
        data = np.load(path)
        if len(data) == len(names):
            print(f"  loaded cached luminance from {path}")
            return data
    values = np.zeros(len(names), dtype=np.float32)
    for i, name in enumerate(names):
        values[i] = float(load_gray(os.path.join(images_dir, name)).mean())
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values)
    print(f"  computed + cached luminance to {path}")
    return values


# --- clustering (operates on the score matrix only, so policies are swappable) ---

def candidate_edges(matrix: np.ndarray, threshold: float):
    """All (weight, i, j) with weight >= threshold, strongest first."""
    n = matrix.shape[0]
    edges = [(matrix[i, j], i, j) for i in range(n) for j in range(i + 1, n)
             if matrix[i, j] >= threshold]
    edges.sort(reverse=True)
    return edges


def _fraction_verifier(matrix: np.ndarray, threshold: float, fraction: float, members):
    """Predicate: idx verifies against >= ceil(fraction * len(members)) of members."""
    need = max(1, math.ceil(fraction * len(members)))
    if need >= len(members):
        return lambda idx: all(matrix[idx, m] >= threshold for m in members)
    return lambda idx: sum(matrix[idx, m] >= threshold for m in members) >= need


def cluster_verify_all(matrix: np.ndarray, threshold: float, fraction: float = 1.0) -> list[list[int]]:
    """
    Verify-then-merge: process candidate edges strongest-first. A candidate
    image joins a group only if it verifies (score >= threshold) against at
    least ceil(fraction * size) of the group's current members; fraction=1.0
    is the strict ALL-members baseline. Two groups merge only if every member
    of one verifies against the other. Unassigned images become singletons.
    """

    def make_verifier(members):
        return _fraction_verifier(matrix, threshold, fraction, members)

    n = matrix.shape[0]
    groups: list[list[int] | None] = []
    membership: dict[int, int] = {}

    for _, i, j in candidate_edges(matrix, threshold):
        gi, gj = membership.get(i), membership.get(j)
        if gi is None and gj is None:
            membership[i] = membership[j] = len(groups)
            groups.append([i, j])
        elif gi is None:
            if make_verifier(groups[gj])(i):
                groups[gj].append(i)
                membership[i] = gj
        elif gj is None:
            if make_verifier(groups[gi])(j):
                groups[gi].append(j)
                membership[j] = gi
        elif gi != gj:
            if all(make_verifier(groups[gj])(a) for a in groups[gi]):
                groups[gi].extend(groups[gj])
                for m in groups[gj]:
                    membership[m] = gi
                groups[gj] = None

    result = [g for g in groups if g]
    assigned = set(membership)
    result.extend([[i] for i in range(n) if i not in assigned])
    return result


def cluster_avg_meas(matrix: np.ndarray, threshold: float, valid: np.ndarray,
                     min_measurable: int = 2) -> list[list[int]]:
    """
    Average-over-measurable join rule (v5; from scripts/exp_avglink.py,
    semantics line-for-line). A candidate joins a group iff the mean score
    over its MEASURABLE links to members is >= threshold, requiring at least
    `min_measurable` measurable links. `valid` marks measurability: 1.0 where
    the pair produced a valid measurement (homography in the guard pass),
    NaN where the measurement failed (forgiven — excluded from the mean) or
    was never attempted. Guard-blocked (contradicted) links are measurable
    and carry score 0 in the guarded matrix — a veto via the mean. The merge
    rule mirrors the join rule. Unassigned images become singletons.

    Validated (v3 stack + fuse): 0.9275 sample / 0.8808 spot / 0.9159 holdout.
    """

    def verifies(idx, members):
        scores = matrix[idx, members]
        ok = ~np.isnan(valid[idx, members])
        return ok.sum() >= min_measurable and scores[ok].mean() >= threshold

    n = matrix.shape[0]
    groups: list[list[int] | None] = []
    membership: dict[int, int] = {}

    for _, i, j in candidate_edges(matrix, threshold):
        gi, gj = membership.get(i), membership.get(j)
        if gi is None and gj is None:
            membership[i] = membership[j] = len(groups)
            groups.append([i, j])
        elif gi is None:
            if verifies(i, groups[gj]):
                groups[gj].append(i)
                membership[i] = gj
        elif gj is None:
            if verifies(j, groups[gi]):
                groups[gi].append(j)
                membership[j] = gi
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


def guard_valid_matrix(n: int, ii, jj, blob) -> np.ndarray:
    """Measurability matrix for avg_meas: 1.0 where the guard pass produced a
    valid measurement (homography succeeded), NaN elsewhere."""
    valid = np.full((n, n), np.nan, dtype=np.float32)
    ok = ~np.isnan(blob)
    valid[ii[ok], jj[ok]] = 1.0
    valid[jj[ok], ii[ok]] = 1.0
    return valid


def cluster_verify_rescue(matrix: np.ndarray, threshold: float, luminance,
                          fraction: float = 0.75,
                          global_guard: bool = False) -> list[list[int]]:
    """
    verify_all (fraction join rule, unchanged merge discipline) PLUS an
    exposure-extreme rescue path, designed from the n=5 failure diagnosis:
    the dominant split pattern is a bracket's extreme-exposure member
    (usually the brightest) that verifies only against its 1-2 adjacent
    neighbors and so fails the ceil(f*m) join check.

    Rescue rule: an unassigned candidate whose raw luminance lies STRICTLY
    OUTSIDE the group's current luminance range (it extends the exposure
    chain at an end) may join a >= 2-member group when
      (a) the geometric entry fee holds against the exposure-adjacent end
          member (score >= threshold), and
      (b) coherence: that adjacent end member is the candidate's STRONGEST
          link into the group (ties allowed), and
      (c) with global_guard=True: the adjacent link also beats the
          candidate's best link to ANY non-member (belongs-here-most test).
    Interior candidates (luminance inside the range) get no rescue — they
    must pass the normal fraction rule. Merges use the fraction rule only.
    """
    luminance = np.asarray(luminance, dtype=np.float64)
    n = matrix.shape[0]
    groups: list[list[int] | None] = []
    membership: dict[int, int] = {}

    def joinable(idx, members):
        return _fraction_verifier(matrix, threshold, fraction, members)(idx)

    def rescuable(idx, members):
        if len(members) < 2:
            return False
        li = luminance[idx]
        lums = luminance[members]
        if lums.min() <= li <= lums.max():
            return False  # interior — normal rule applies
        nb = members[int(np.argmin(lums))] if li < lums.min() \
            else members[int(np.argmax(lums))]
        if matrix[idx, nb] < threshold:
            return False
        if not all(matrix[idx, m] <= matrix[idx, nb] for m in members):
            return False
        if global_guard:
            member_set = set(members) | {idx}
            best_outside = max((matrix[idx, w] for w in range(n) if w not in member_set),
                               default=0.0)
            if matrix[idx, nb] < best_outside:
                return False
        return True

    for _, i, j in candidate_edges(matrix, threshold):
        gi, gj = membership.get(i), membership.get(j)
        if gi is None and gj is None:
            membership[i] = membership[j] = len(groups)
            groups.append([i, j])
        elif gi is None:
            if joinable(i, groups[gj]) or rescuable(i, groups[gj]):
                groups[gj].append(i)
                membership[i] = gj
        elif gj is None:
            if joinable(j, groups[gi]) or rescuable(j, groups[gi]):
                groups[gi].append(j)
                membership[j] = gi
        elif gi != gj:
            if all(joinable(a, groups[gj]) for a in groups[gi]):
                groups[gi].extend(groups[gj])
                for m in groups[gj]:
                    membership[m] = gi
                groups[gj] = None

    result = [g for g in groups if g]
    assigned = set(membership)
    result.extend([[i] for i in range(n) if i not in assigned])
    return result


def guards_to_matrices(guards_npz: str, n: int):
    """Assemble symmetric (blob, cheir) matrices from a merge_guards npz."""
    d = np.load(guards_npz)
    blob = np.full((n, n), np.nan, dtype=np.float32)
    cheir = np.full((n, n), np.nan, dtype=np.float32)
    blob[d["i"], d["j"]] = d["blob"]
    blob[d["j"], d["i"]] = d["blob"]
    cheir[d["i"], d["j"]] = d["cheir"]
    cheir[d["j"], d["i"]] = d["cheir"]
    return blob, cheir


def cluster_guard_rescue(matrix: np.ndarray, threshold: float,
                         blob_m: np.ndarray, cheir_m: np.ndarray,
                         fraction: float = 0.75, blob_cut: float = 0.35,
                         cheir_cut: float = 0.5, rescue_lo: float = 0.3) -> list[list[int]]:
    """
    Lever D: guarded verify-then-merge + guard-gated weak-link rescue.

    Stage 1 = lever B exactly: candidate edges with cheirality >= cheir_cut
    (camera moved) are blocked; verify-then-merge (fraction rule) runs on the
    rest. Stage 2 (rescue): an image left unassigned may join a >= 2-member
    group when it has a weak link in [rescue_lo*T, T) to some member that
    physically verifies — cheirality < cheir_cut AND aligned-residual blob
    < blob_cut (camera still, scene unchanged). Rescue candidates are
    processed strongest-link-first; joins extend groups and may enable
    further rescues (single pass over candidates, repeated until stable).
    NaN guard metrics = no evidence = never blocks, never rescues.
    """
    n = matrix.shape[0]
    blocked = np.zeros((n, n), dtype=bool)
    ok = ~np.isnan(cheir_m)
    blocked[ok & (cheir_m >= cheir_cut)] = True

    eff = matrix.copy()
    eff[blocked] = 0.0
    clusters = cluster_verify_all(eff, threshold, fraction)

    groups: list[list[int]] = [list(c) for c in clusters]
    membership: dict[int, int] = {}
    for gi, g in enumerate(groups):
        for m in g:
            membership[m] = gi

    # sparse per-image rescue-band links (on the ORIGINAL matrix)
    rescue_ok = ~np.isnan(blob_m) & ~np.isnan(cheir_m) & (cheir_m < cheir_cut) & (blob_m < blob_cut)
    band_links: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in np.flatnonzero((matrix[i] >= rescue_lo * threshold) & (matrix[i] < threshold) & rescue_ok[i]):
            band_links[i].append((int(j), float(matrix[i, j])))
    for links in band_links:
        links.sort(key=lambda t: -t[1])

    changed = True
    while changed:
        changed = False
        for i in range(n):
            gi = membership[i]
            if len(groups[gi]) > 1 or not band_links[i]:
                continue  # only singletons (unassigned images) are rescue candidates
            best_g, best_s = None, rescue_lo * threshold
            for m, s in band_links[i]:
                g = membership[m]
                if g != gi and len(groups[g]) >= 2 and s >= best_s:
                    best_g, best_s = g, s
            if best_g is not None:
                groups[best_g].append(i)
                groups[gi] = []
                membership[i] = best_g
                changed = True
    return [g for g in groups if g]


def cluster_verify_chain(matrix: np.ndarray, threshold: float, luminance,
                         merge_fraction: float = 0.75) -> list[list[int]]:
    """
    Exposure-adjacent chain verification (lever 2): like verify-then-merge,
    but the JOIN rule is exposure-local. Groups are treated as exposure chains:
    members are kept sorted by raw mean luminance, and a candidate joins if it
    verifies (score >= threshold — the geometric entry fee) against ALL of its
    exposure-adjacent neighbors in the augmented ordering (1 neighbor at a
    chain end, 2 in the middle). This lets extreme exposures ride dark->mid->
    bright links instead of requiring a direct dark<->bright match.

    Merge discipline is unchanged from verify_all: two groups merge only if
    every member of one verifies against >= ceil(merge_fraction * size) of the
    other. Unassigned images become singletons.

    `luminance` is a per-image array aligned with the matrix indices.
    """
    luminance = np.asarray(luminance, dtype=np.float64)
    n = matrix.shape[0]
    groups: list[list[int] | None] = []
    membership: dict[int, int] = {}

    def chain_verifies(idx, members):
        ordered = sorted(members, key=lambda m: (luminance[m], m))
        keys = [(luminance[m], m) for m in ordered]
        pos = bisect.bisect_left(keys, (luminance[idx], idx))
        neighbors = []
        if pos > 0:
            neighbors.append(ordered[pos - 1])
        if pos < len(ordered):
            neighbors.append(ordered[pos])
        return all(matrix[idx, nb] >= threshold for nb in neighbors)

    for _, i, j in candidate_edges(matrix, threshold):
        gi, gj = membership.get(i), membership.get(j)
        if gi is None and gj is None:
            membership[i] = membership[j] = len(groups)
            groups.append([i, j])
        elif gi is None:
            if chain_verifies(i, groups[gj]):
                groups[gj].append(i)
                membership[i] = gj
        elif gj is None:
            if chain_verifies(j, groups[gi]):
                groups[gi].append(j)
                membership[j] = gi
        elif gi != gj:
            if all(_fraction_verifier(matrix, threshold, merge_fraction, groups[gj])(a)
                   for a in groups[gi]):
                groups[gi].extend(groups[gj])
                for m in groups[gj]:
                    membership[m] = gi
                groups[gj] = None

    result = [g for g in groups if g]
    assigned = set(membership)
    result.extend([[i] for i in range(n) if i not in assigned])
    return result


def cluster_single_link(matrix: np.ndarray, threshold: float) -> list[list[int]]:
    """Union-find over candidate edges ("either member vouches" policy)."""
    n = matrix.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, i, j in candidate_edges(matrix, threshold):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    return list(buckets.values())


POLICIES = {
    "verify_all": cluster_verify_all,
    "single_link": cluster_single_link,
    "chain": cluster_verify_chain,    # needs luminance — callers pass it via partial/kwargs
    "rescue": cluster_verify_rescue,  # needs luminance too
    "rescue_guarded": lambda matrix, threshold, luminance, fraction=0.75:
        cluster_verify_rescue(matrix, threshold, luminance, fraction, global_guard=True),
    "guard_rescue": cluster_guard_rescue,  # needs guards npz — sweep script supplies matrices
}


# --- public API ---

def normalize_score_matrix(matrix: np.ndarray, kps, mode: str) -> np.ndarray:
    """
    Normalize raw inlier counts by keypoint counts (lever 1):
    "norm_sqrt" divides by sqrt(kp_i * kp_j), "norm_min" by min(kp_i, kp_j).
    Mirrors scripts/lever1_normalized.py exactly.
    """
    counts = np.array([len(k) for k in kps], dtype=np.int32)
    safe = counts.astype(np.float32).clip(1)
    if mode == "norm_sqrt":
        out = matrix / np.sqrt(np.outer(safe, safe))
    elif mode == "norm_min":
        out = matrix / np.minimum.outer(safe, safe)
    else:
        raise ValueError(f"unknown normalize mode {mode!r}")
    out = out.astype(np.float32)
    np.fill_diagonal(out, 0)
    return out


def cluster_with_policy(matrix: np.ndarray, policy: str, threshold: float,
                        fraction: float, valid: np.ndarray | None = None) -> list[list[int]]:
    """Shared clustering dispatch (used by group_images and server.py)."""
    if policy == "verify_all":
        return cluster_verify_all(matrix, threshold, fraction)
    if policy == "avg_meas":
        if valid is None:
            raise ValueError("avg_meas requires guard measurability (valid matrix)")
        return cluster_avg_meas(matrix, threshold, valid)
    return POLICIES[policy](matrix, threshold)


def group_images(image_paths: list[str], threshold: float = DEFAULT_THRESHOLD,
                 ratio: float = RATIO_TEST, policy: str = "verify_all",
                 preprocess: str = "none", normalize: str | None = None,
                 fraction: float = 1.0, workers: int | None = None,
                 reverify: str | None = None, guard: str | None = None,
                 fuse_threshold: float | None = None) -> list[list[str]]:
    """
    Group images by camera angle. Returns groups of basenames.

    Defaults reproduce the raw baseline. The validated v5 submission config:
    preprocess="gamma", normalize="norm_sqrt", reverify="histeq",
    guard="cheirality", fuse_threshold=0.022, policy="avg_meas",
    threshold=0.025.
    """
    paths = [str(p) for p in image_paths]
    names, kps, descs = extract_features(paths, preprocess=preprocess)
    matrix = compute_score_matrix(kps, descs, ratio, workers)
    if normalize is not None:
        matrix = normalize_score_matrix(matrix, kps, normalize)
    luminance = None
    if reverify is not None or guard is not None or fuse_threshold is not None:
        t0 = time.time()
        luminance = np.array([load_gray(p).mean() for p in paths], dtype=np.float32)
        print(f"  stages: luminance pass ({time.time() - t0:.1f}s)", flush=True)
    if reverify is not None:
        if reverify != "histeq":
            raise ValueError(f"unknown reverify mode {reverify!r}")
        matrix = reverify_histeq(matrix, paths, luminance, kps, descs, workers=workers)

    valid = None
    if guard is not None or fuse_threshold is not None or policy == "avg_meas":
        t0 = time.time()
        images = [load_gray(p, preprocess=preprocess) for p in paths]
        print(f"  stages: loaded {len(paths)} {preprocess} images ({time.time() - t0:.1f}s)",
              flush=True)
        # avg_meas needs measurability down to 0.3xT (workshop coverage);
        # the same measurement drives edge blocking at >= threshold.
        min_score = 0.3 * threshold if policy == "avg_meas" else threshold
        ii, jj, blob, cheir = compute_guard_metrics(matrix, paths, images, luminance,
                                                    kps, descs, min_score, workers)
        if policy == "avg_meas":
            valid = guard_valid_matrix(len(paths), ii, jj, blob)
    if guard is not None:
        if guard != "cheirality":
            raise ValueError(f"unknown guard mode {guard!r}")
        matrix = apply_cheirality_guard(matrix, ii, jj, cheir, block_floor=threshold)
    clusters = cluster_with_policy(matrix, policy, threshold, fraction, valid)
    if fuse_threshold is not None:
        clusters = fuse_rescue(matrix, clusters, images, luminance, kps, descs,
                               fuse_threshold, workers)
    return [[names[i] for i in cluster] for cluster in clusters]


def write_predictions(groups: list[list[str]], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "group_id"])
        for gid, group in enumerate(groups):
            for name in sorted(group):
                writer.writerow([name, f"g{gid:04d}"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, help="directory of images to group")
    parser.add_argument("--out", required=True, help="predictions CSV (filename,group_id)")
    parser.add_argument("--threshold", "-T", type=float, default=DEFAULT_THRESHOLD,
                        help=f"inlier threshold for same-angle edges (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--ratio", type=float, default=RATIO_TEST, help="Lowe ratio (default 0.75)")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="verify_all")
    parser.add_argument("--fraction", "-f", type=float, default=1.0,
                        help="verify-fraction for verify_all joins (default 1.0 = all members)")
    parser.add_argument("--preprocess", choices=PREPROCESSORS, default="none",
                        help="exposure-invariant preprocessing before SIFT (own cache file)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--no-cache", action="store_true", help="ignore/refresh the score cache")
    args = parser.parse_args()

    t_start = time.time()
    paths = sorted(str(p) for p in Path(args.images).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        sys.exit(f"no images found in {args.images}")
    print(f"{len(paths)} images in {args.images} (preprocess={args.preprocess})")

    names = [os.path.basename(p) for p in paths]
    cache_tag = "" if args.preprocess == "none" else args.preprocess
    cache_path = cache_path_for(args.images, cache_tag)
    matrix = None if args.no_cache else load_cached_matrix(cache_path, names)

    if matrix is None:
        t0 = time.time()
        names, kps, descs = extract_features(paths, preprocess=args.preprocess)
        t_feat = time.time() - t0
        t0 = time.time()
        matrix = compute_score_matrix(kps, descs, args.ratio, args.workers)
        t_match = time.time() - t0
        save_cached_matrix(cache_path, names, matrix)
        print(f"timing: features {t_feat:.1f}s, pairwise matching {t_match:.1f}s")

    t0 = time.time()
    if args.policy == "verify_all":
        clusters = cluster_verify_all(matrix, args.threshold, args.fraction)
    elif args.policy in ("chain", "rescue"):
        lum = load_or_compute_luminance(args.images, names)
        clusters = POLICIES[args.policy](matrix, args.threshold, lum, args.fraction)
    else:
        clusters = POLICIES[args.policy](matrix, args.threshold)
    groups = [[names[i] for i in cluster] for cluster in clusters]
    write_predictions(groups, args.out)
    print(f"clustering ({args.policy}, T={args.threshold:g}, f={args.fraction:g}): "
          f"{len(groups)} groups in {time.time() - t0:.2f}s")
    print(f"wrote {args.out}  (total {time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
