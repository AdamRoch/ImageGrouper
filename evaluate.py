"""
Local evaluation harness for the AutoHDR grouping challenge.

Replicates the official scorer from SCORING.md exactly (frozenset
intersection of predicted vs reference groups), then adds diagnostics:
false merges and false splits reported separately, since they have
different costs under exact-match scoring.

Usage:
    python evaluate.py --manifest data/sample500/public_manifest.csv \
                       --predictions output/predictions.csv
"""

import argparse
import csv
from collections import defaultdict


def load_groups(path: str) -> list[frozenset]:
    """
    Load a CSV with group_id,filename columns into a list of frozensets.

    Faithful to SCORING.md: "if a file appears in multiple groups, only
    the last occurrence counts" — rows are processed in order and a file
    reassigned to a new group is removed from its previous group.
    """
    buckets = defaultdict(set)
    file_to_group = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            filename, group_id = row["filename"], row["group_id"]
            previous = file_to_group.get(filename)
            if previous is not None and previous != group_id:
                buckets[previous].discard(filename)
            buckets[group_id].add(filename)
            file_to_group[filename] = group_id
    return [frozenset(v) for v in buckets.values() if v]


def evaluate(manifest_path: str, predictions_path: str) -> dict:
    reference = set(load_groups(manifest_path))
    predicted = set(load_groups(predictions_path))

    # Official score: exact frozenset intersection / number of reference groups
    exact_matches = reference & predicted
    score = len(exact_matches) / len(reference) if reference else 0.0

    # Diagnostics: classify each reference group that failed
    splits = []   # ref group spread across multiple predicted groups
    merges = []   # ref group intact inside a larger predicted group
    for ref in sorted(reference - exact_matches, key=len, reverse=True):
        touched = {p for p in predicted if p & ref}
        if len(touched) == 1:
            merges.append(ref)
        else:
            splits.append(ref)

    return {
        "score": score,
        "exact_matches": len(exact_matches),
        "total_reference_groups": len(reference),
        "total_predicted_groups": len(predicted),
        "false_merges": merges,
        "false_splits": splits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Reference CSV (group_id,filename)")
    parser.add_argument("--predictions", required=True, help="Predicted CSV (filename,group_id)")
    args = parser.parse_args()

    result = evaluate(args.manifest, args.predictions)

    print(f"Score:            {result['score']:.4f}  "
          f"({result['exact_matches']}/{result['total_reference_groups']} exact matches)")
    print(f"Predicted groups: {result['total_predicted_groups']}")
    print(f"False merges:     {len(result['false_merges'])} reference groups damaged")
    for g in result["false_merges"]:
        print(f"  merged group of {len(g)}: {sorted(g)[:3]}{'...' if len(g) > 3 else ''}")
    print(f"False splits:     {len(result['false_splits'])} reference groups broken")
    for g in result["false_splits"]:
        print(f"  split group of {len(g)}: {sorted(g)[:3]}{'...' if len(g) > 3 else ''}")


if __name__ == "__main__":
    main()
