# AutoHDR Image Grouping — Solution & Eval Harness

A solution to AutoHDR's image-grouping task, developed as a hiring project.

> This repo started from AutoHDR's public challenge starter kit. The public competition
> has ended; this is now an independent project built against the same container contract
> and evaluation flow. Prize/registration framing in the original challenge docs
> (`project-brief.md`, `SUBMISSION_GUIDE.md`) no longer applies.

## The task

Real-estate photographers shoot multiple exposures (HDR brackets) from the same camera
angle and upload the whole shoot unsorted. Given a folder of JPEGs (resized to 1024px
max, randomized filenames), group the images by camera angle and write `predictions.csv`.

- Group sizes vary: 1, 3, 5, 7, even 25 images — mixed within a single shoot.
- Upload order is meaningless; filenames carry no information.
- Same-group images can differ radically in brightness, while near-identical images can
  be different groups (door open vs. closed, tripod nudged a few centimeters).

**Scoring:** exact-match only — `score = exact_matches / total_reference_groups`.
No partial credit: one wrong filename fails the whole group. A false merge kills two
groups; a false split kills one.

Background videos from AutoHDR:
[Challenge overview](https://youtu.be/NSEMhzPd_bw) ·
[The task explained](https://youtu.be/zHP4wDuIYPU)

## Container contract

| Path | Description |
|------|-------------|
| **Input** `/input/images/` | Test images (mounted read-only) |
| **Output** `/output/predictions.csv` | `filename,group_id` — every image exactly once |

Runtime: CPU-only (`cpu-large` 8 vCPU/16 GB or `cpu-xlarge` 16 vCPU/32 GB), no internet
access, per-run time limit.

## Repo layout

- `solution.py` — the grouping algorithm (container entrypoint)
- `evaluate.py` — local grader: replicates the official exact-match scorer exactly, plus
  false-merge / false-split diagnostics
- `Dockerfile`, `submission.yaml` — build & submission config
- `docs/project-context/` — notes from AutoHDR's explainer video + edge-case screenshots
- `research/` — feasibility research on in-browser grouping (stretch goal)
- `AGENTS.md` — running log of settled design decisions
- `data/` — local datasets (gitignored)

## Local development

Download a labeled package (images + `public_manifest.csv` answer key) into `data/`:

| Package | Images | Size | Download |
|---|---|---|---|
| Sample | 366 | ~1.4 GB | [autohdr_sample_500.zip](https://grouping-dataset-solution.s3.amazonaws.com/downloads/autohdr_sample_500.zip) |
| Medium | 5,000 | ~21 GB | [autohdr_medium_5000.zip](https://grouping-dataset-solution.s3.amazonaws.com/downloads/autohdr_medium_5000.zip) |
| Large | 10,000 | ~42 GB | [autohdr_large_10000.zip](https://grouping-dataset-solution.s3.amazonaws.com/downloads/autohdr_large_10000.zip) |

Grade a run of the algorithm:

```bash
python evaluate.py --manifest data/sample500/public_manifest.csv \
                   --predictions output/predictions.csv
```

Reference points on the sample: perfect grouping = 1.0; every-image-alone = 0.0725.

## Approach

1. **Pairwise test:** SIFT/AKAZE keypoints → descriptor match → RANSAC homography fit →
   inlier count is the same-angle score. A post-alignment residual check catches the
   door-open/closed case; parallax misfit catches small camera repositions.
2. **Clustering:** verify-then-merge — a candidate joins a group only if it verifies
   against all current members. Exact-match scoring punishes a false merge twice as hard
   as a false split, so ambiguous cases split.
3. **Iteration:** thresholds calibrated locally via `evaluate.py` (one variable changed
   at a time); private-test submissions spent strategically — they are rate-limited per
   day and are the only grade that counts.

Full decision log in `AGENTS.md`.

## Roadmap

- [ ] Classical SIFT+RANSAC baseline with first real local score
- [ ] Bake-offs: matcher (SIFT / AKAZE / LightGlue), inlier threshold, clustering policy
- [ ] Demo web UI served from the same container (AWS App Runner, `*.adamroch.com`)
- [ ] Stretch: true in-browser grouping (see `research/`)
