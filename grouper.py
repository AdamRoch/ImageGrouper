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


# --- cache (matrix only; features are cheap to recompute) ---

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


def group_images(image_paths: list[str], threshold: float = DEFAULT_THRESHOLD,
                 ratio: float = RATIO_TEST, policy: str = "verify_all",
                 preprocess: str = "none", normalize: str | None = None,
                 fraction: float = 1.0, workers: int | None = None) -> list[list[str]]:
    """
    Group images by camera angle. Returns groups of basenames.

    Defaults reproduce the raw baseline; the validated submission config is
    preprocess="gamma", normalize="norm_sqrt", fraction=0.75, threshold=0.018.
    """
    names, kps, descs = extract_features([str(p) for p in image_paths], preprocess=preprocess)
    matrix = compute_score_matrix(kps, descs, ratio, workers)
    if normalize is not None:
        matrix = normalize_score_matrix(matrix, kps, normalize)
    if policy == "verify_all":
        clusters = cluster_verify_all(matrix, threshold, fraction)
    else:
        clusters = POLICIES[policy](matrix, threshold)
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
