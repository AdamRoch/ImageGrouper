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
import csv
import hashlib
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

# --- matching state (set by compute_score_matrix before the thread pool starts) ---
_KP = None      # list of float32 keypoint coordinate arrays, one per image
_DESC = None    # list of float32 descriptor arrays
_RATIO = RATIO_TEST


def load_gray(path: str, max_dim: int = MATCH_MAX_DIM) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    h, w = img.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def extract_features(paths: list[str], max_dim: int = MATCH_MAX_DIM):
    """Return (names, keypoint arrays, descriptor arrays). Sequential — fast enough (~0.1s/img)."""
    sift = cv2.SIFT_create()
    names, kps, descs = [], [], []
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        img = load_gray(p, max_dim)
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

def cache_path_for(images_dir: str) -> Path:
    key = hashlib.md5(os.path.abspath(images_dir).encode()).hexdigest()[:12]
    return CACHE_DIR / f"scores_{key}.npz"


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


# --- clustering (operates on the score matrix only, so policies are swappable) ---

def candidate_edges(matrix: np.ndarray, threshold: float):
    """All (weight, i, j) with weight >= threshold, strongest first."""
    n = matrix.shape[0]
    edges = [(matrix[i, j], i, j) for i in range(n) for j in range(i + 1, n)
             if matrix[i, j] >= threshold]
    edges.sort(reverse=True)
    return edges


def cluster_verify_all(matrix: np.ndarray, threshold: float) -> list[list[int]]:
    """
    Verify-then-merge: process candidate edges strongest-first. A candidate
    image joins a group only if it verifies (inliers >= threshold) against
    ALL current members; two groups merge only if every cross pair verifies.
    Unassigned images become singletons.
    """
    n = matrix.shape[0]
    groups: list[list[int] | None] = []
    membership: dict[int, int] = {}

    def verifies(idx, members):
        return all(matrix[idx, m] >= threshold for m in members)

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
}


# --- public API ---

def group_images(image_paths: list[str], threshold: float = DEFAULT_THRESHOLD,
                 ratio: float = RATIO_TEST, policy: str = "verify_all") -> list[list[str]]:
    """Group images by camera angle. Returns groups of basenames."""
    names, kps, descs = extract_features([str(p) for p in image_paths])
    matrix = compute_score_matrix(kps, descs, ratio)
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
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--no-cache", action="store_true", help="ignore/refresh the score cache")
    args = parser.parse_args()

    t_start = time.time()
    paths = sorted(str(p) for p in Path(args.images).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not paths:
        sys.exit(f"no images found in {args.images}")
    print(f"{len(paths)} images in {args.images}")

    names = [os.path.basename(p) for p in paths]
    cache_path = cache_path_for(args.images)
    matrix = None if args.no_cache else load_cached_matrix(cache_path, names)

    if matrix is None:
        t0 = time.time()
        names, kps, descs = extract_features(paths)
        t_feat = time.time() - t0
        t0 = time.time()
        matrix = compute_score_matrix(kps, descs, args.ratio, args.workers)
        t_match = time.time() - t0
        save_cached_matrix(cache_path, names, matrix)
        print(f"timing: features {t_feat:.1f}s, pairwise matching {t_match:.1f}s")

    t0 = time.time()
    clusters = POLICIES[args.policy](matrix, args.threshold)
    groups = [[names[i] for i in cluster] for cluster in clusters]
    write_predictions(groups, args.out)
    print(f"clustering ({args.policy}, T={args.threshold:g}): "
          f"{len(groups)} groups in {time.time() - t0:.2f}s")
    print(f"wrote {args.out}  (total {time.time() - t_start:.1f}s)")


if __name__ == "__main__":
    main()
