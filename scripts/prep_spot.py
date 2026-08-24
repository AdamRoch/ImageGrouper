"""
Build a spot-check subset of the medium-5000 package under test conditions.

Selection: ALL groups with n >= --min-large (the dominant failure mode on the
sample set) plus whole smaller groups sampled with a seeded RNG until the
image target is reached. Groups are kept whole — a partially included group
would distort exact-match scoring.

Output (mirrors prep_test_sim.py conventions):
    <out-dir>/images/<uuid>.jpg   (1024px max, JPEG q90, randomized names)
    <out-dir>/manifest.csv        (group_id,filename)

Usage:
    .venv/bin/python scripts/prep_spot.py \
        --src-dir data/medium5000/images \
        --manifest data/medium5000/public_manifest.csv \
        --out-dir data/medium5000/spot --target 900 --seed 1234
"""

import argparse
import csv
import random
import uuid
from collections import defaultdict
from pathlib import Path

from PIL import Image

MAX_DIM = 1024
JPEG_QUALITY = 90


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target", type=int, default=900)
    parser.add_argument("--min-large", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--exclude-manifest", default=None,
                        help="CSV whose group_ids are excluded (e.g. the spot manifest)")
    parser.add_argument("--all-groups", action="store_true",
                        help="select every remaining group (ignore target/min-large sampling)")
    args = parser.parse_args()

    src_dir, out_dir = Path(args.src_dir), Path(args.out_dir)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    with open(args.manifest) as f:
        for row in csv.DictReader(f):
            groups[row["group_id"]].append(row["filename"])

    if args.exclude_manifest:
        with open(args.exclude_manifest) as f:
            excluded = {row["group_id"] for row in csv.DictReader(f)}
        groups = defaultdict(list, {g: fs for g, fs in groups.items() if g not in excluded})
        print(f"excluded {len(excluded)} group_ids from {args.exclude_manifest}")

    if args.all_groups:
        selected = sorted(groups)
        large = [g for g in selected if len(groups[g]) >= args.min_large]
    else:
        large = sorted([g for g, fs in groups.items() if len(fs) >= args.min_large])
        small = [g for g, fs in groups.items() if len(fs) < args.min_large]
        random.Random(args.seed).shuffle(small)

        selected = list(large)
        total = sum(len(groups[g]) for g in selected)
        for g in small:
            if total >= args.target:
                break
            selected.append(g)
            total += len(groups[g])

    rows = []
    for i, g in enumerate(selected, 1):
        for filename in groups[g]:
            src = src_dir / filename
            if not src.exists():
                raise FileNotFoundError(f"manifest row without image: {src}")
            new_name = f"{uuid.uuid4()}.jpg"
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
                im.save(img_out / new_name, "JPEG", quality=JPEG_QUALITY)
            rows.append((g, new_name))
        if i % 25 == 0 or i == len(selected):
            print(f"  processed {i}/{len(selected)} groups", flush=True)

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "filename"])
        writer.writerows(rows)

    print(f"selected {len(selected)} groups ({len(large)} large), {len(rows)} images -> {out_dir}")


if __name__ == "__main__":
    main()
