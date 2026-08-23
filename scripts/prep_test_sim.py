"""
Build a simulated test set from the labeled sample-500 package.

The real test set is resized to 1024px max dimension, JPEG re-encoded,
and renamed with randomized UUIDs (no filename leakage of the group id).
This script reproduces those conditions locally:

    data/sample500/images/            (full-res originals)
    data/sample500/public_manifest.csv (answer key: group_id,filename)
        ->
    data/sample500/test_sim/<uuid>.jpg (1024px max, JPEG q90, random name)
    data/sample500/test_sim/manifest.csv (group_id,filename with uuids)

Usage:
    .venv/bin/python scripts/prep_test_sim.py
"""

import csv
import uuid
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "sample500" / "images"
SRC_MANIFEST = ROOT / "data" / "sample500" / "public_manifest.csv"
OUT_DIR = ROOT / "data" / "sample500" / "test_sim"

MAX_DIM = 1024
JPEG_QUALITY = 90


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(SRC_MANIFEST) as f:
        for row in csv.DictReader(f):
            rows.append((row["group_id"], row["filename"]))

    out_rows = []
    for i, (group_id, filename) in enumerate(rows, 1):
        src = SRC_DIR / filename
        if not src.exists():
            raise FileNotFoundError(f"manifest row without image: {src}")

        new_name = f"{uuid.uuid4()}.jpg"
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)
            im.save(OUT_DIR / new_name, "JPEG", quality=JPEG_QUALITY)
        out_rows.append((group_id, new_name))

        if i % 50 == 0 or i == len(rows):
            print(f"  processed {i}/{len(rows)}", flush=True)

    with open(OUT_DIR / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "filename"])
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} images + manifest to {OUT_DIR}")


if __name__ == "__main__":
    main()
