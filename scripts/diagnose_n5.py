"""
Diagnose n=5 bracket failures for the current config (gamma_norm_sqrt,
verify_all f=0.75, T=0.018) on a cached dataset.

For each failing n=5 reference group, classify the within-group score
structure at T:
  - single_weak_link: one member has all within-group links < T while the
    other four form a >= T clique
  - two_cluster: members split into two >= T cliques (2+3) with all cross
    links < T
  - global_depression: diffuse / everything weak (none of the above)
Also records the isolated/weak members' max link strength (rescue-ability)
and luminance range of failing vs passing n=5 groups.

Usage:
    .venv/bin/python scripts/diagnose_n5.py --images <dir> --manifest <csv> \
        --variant gamma_norm_sqrt --predictions <current-config predictions.csv>
"""

import argparse
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate
import grouper


def classify(sub: np.ndarray, T: float):
    """Classify a 5x5 within-group score matrix at threshold T."""
    adj = sub >= T
    np.fill_diagonal(adj, False)
    n = len(sub)

    # connected components at T
    seen = set()
    comps = []
    for s in range(n):
        if s in seen:
            continue
        comp, stack = {s}, [s]
        while stack:
            u = stack.pop()
            for v in np.flatnonzero(adj[u]):
                v = int(v)
                if v not in comp:
                    comp.add(v)
                    stack.append(v)
        seen |= comp
        comps.append(sorted(comp))

    def is_clique(members):
        return all(sub[a, b] >= T for a, b in itertools.combinations(members, 2))

    # single weak link: isolated singleton + 4-clique
    if len(comps) == 2:
        sizes = sorted(len(c) for c in comps)
        if sizes == [1, 4]:
            solo = comps[0] if len(comps[0]) == 1 else comps[1]
            rest = comps[0] if len(comps[0]) == 4 else comps[1]
            if is_clique(rest):
                return "single_weak_link", comps
        if sizes == [2, 3]:
            a, b = (comps[0], comps[1]) if len(comps[0]) == 2 else (comps[1], comps[0])
            if is_clique(a) and is_clique(b):
                return "two_cluster", comps
    if len(comps) == 1:
        return "connected_not_clique", comps
    return "global_depression", comps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--variant", default="gamma_norm_sqrt")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, default=0.018)
    args = parser.parse_args()
    T = args.threshold

    names = sorted(p.name for p in Path(args.images).glob("*.jpg"))
    idx = {nm: i for i, nm in enumerate(names)}
    matrix = grouper.load_cached_matrix(grouper.cache_path_for(args.images, args.variant), names)
    lum = grouper.load_or_compute_luminance(args.images, names)

    result = evaluate.evaluate(args.manifest, args.predictions)
    failing = {tuple(sorted(g)): kind
               for kind, lst in (("split", result["false_splits"]), ("merged", result["false_merges"]))
               for g in lst}
    n5 = [g for g in failing if len(g) == 5]
    passing_n5 = [g for g in evaluate.load_groups(args.manifest)
                  if len(g) == 5 and tuple(sorted(g)) not in failing]

    print(f"{args.images}: {len(n5)} failing n=5 groups, {len(passing_n5)} passing")
    counts = defaultdict(int)
    weak_maxlinks = []
    fail_ranges, pass_ranges = [], []
    for g in n5:
        ii = [idx[f] for f in g]
        sub = matrix[np.ix_(ii, ii)].astype(np.float64)
        kind, comps = classify(sub, T)
        kind = f"{kind}({'merged' if failing[tuple(sorted(g))] == 'merged' else 'split'})"
        counts[kind] += 1
        fail_ranges.append(float(lum[ii].max() - lum[ii].min()))
        # rescue-ability: for members outside the largest component, best link to it
        largest = max(comps, key=len)
        for c in comps:
            if c is largest:
                continue
            for m in c:
                weak_maxlinks.append(sub[m, largest].max() / T)
    for g in passing_n5:
        ii = [idx[f] for f in g]
        pass_ranges.append(float(lum[ii].max() - lum[ii].min()))

    print("pattern counts:", dict(sorted(counts.items())))
    if weak_maxlinks:
        w = np.array(weak_maxlinks)
        print(f"weak-member max link to main cluster, as fraction of T: "
              f"min {w.min():.2f}, median {np.median(w):.2f}, max {w.max():.2f}")
        print(f"  >=1.0xT: {(w >= 1).sum()}, >=0.8xT: {(w >= 0.8).sum()}, "
              f">=0.5xT: {(w >= 0.5).sum()}, <0.5xT: {(w < 0.5).sum()} (of {len(w)})")
    fr, pr = np.array(fail_ranges), np.array(pass_ranges)
    print(f"luminance range  failing n=5: median {np.median(fr):.1f}  "
          f"passing n=5: median {np.median(pr):.1f}")

    # concrete examples: first 3 failing groups with their 5x5 matrices
    print("\nexamples:")
    for g in n5[:3]:
        ii = [idx[f] for f in g]
        sub = matrix[np.ix_(ii, ii)]
        order = np.argsort(lum[ii])
        kind, _ = classify(sub.astype(np.float64), T)
        print(f"  {failing[tuple(sorted(g))]} group ({kind}), members sorted by luminance "
              f"(lum={[f'{lum[ii[o]]:.0f}' for o in order]}), scores x1000:")
        sq = (sub[np.ix_(order, order)] * 1000).astype(int)
        for row in sq:
            print("   ", " ".join(f"{v:5d}" for v in row))


if __name__ == "__main__":
    main()
