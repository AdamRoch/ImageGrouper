# AutoHDR Image Grouping — Solution & Eval Harness

A solution to AutoHDR's image-grouping task, developed as a hiring project.

> This repo started from AutoHDR's public challenge starter kit. The public competition
> has ended and its submission channel is gone; this is now an independent project built
> against the same container contract and evaluation flow. All scores below are local,
> computed against labeled public packages with the official exact-match metric.

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
groups; a false split kills one. Details in `SCORING.md`.

Background videos from AutoHDR:
[Challenge overview](https://youtu.be/NSEMhzPd_bw) ·
[The task explained](https://youtu.be/zHP4wDuIYPU)

## Approach (v5 pipeline)

Seven stages, all classical CV (OpenCV SIFT) plus geometric verification — no learned
models, no internet, CPU-only:

1. **Gamma normalization** — each image's median luminance mapped to a canonical target
   (kills most false matches that were really exposure noise).
2. **All-pairs matching** — SIFT → Lowe ratio test → RANSAC homography; inlier count is
   the same-angle score. Parallelized over threads.
3. **Score normalization** — inliers / √(kpᵢ·kpⱼ), so texture-rich images don't dominate.
4. **Exposure re-verify** — borderline pairs with a large raw-luminance gap are re-matched
   after histogram-matching the darker image onto the brighter one's tonal distribution;
   scores upgrade via max(). This is what assembles extreme-exposure brackets.
5. **Camera-translation guard** — candidate pairs are pose-decomposed (essential matrix →
   recoverPose); edges whose matches triangulate in front of both cameras (cheirality
   ratio ≥ 0.5) read as a repositioned camera and are blocked. This is what stops
   same-room-different-angle merges.
6. **Average-over-measurable clustering** — a candidate joins a group iff its mean score
   over links with a *valid measurement* ≥ T (≥2 measurable links required; failed
   measurements are forgiven, not counted as negative evidence).
7. **Fused-stack orphan rescue** — confident groups are fused into one exposure composite
   (Mertens fusion); leftover singletons match against composites, but a match only
   nominates — the join still has to pass the cheirality guard against a real member.

The full experiment log (what worked, what failed, and why) is in `AGENTS.md`.

## Validated scores

Three labeled packages, test-conditioned (1024px max, randomized UUID filenames —
no filename leakage). Holdout discipline: tuning on the sample + spot sets only;
the holdout was opened exactly once per accepted change, at pre-registered settings.
The sample + spot + holdout numbers below are therefore honest generalization
estimates, not tuned maxima.

| Set | Images / groups | Score | Merges / splits |
|---|---|---|---|
| Sample test-sim | 366 / 69 | **0.9275** | 0 / 5 |
| Medium spot-check | 902 / 193 | **0.8808** | 2 / 21 |
| Medium holdout (sealed) | 1,224 / 345 | **0.9159** | 15 / 14 |

The exact-match metric is binary per group, so scores move in group-sized steps
(±1 group ≈ ±0.015 on the sample, ±0.003 on the holdout).

Container-vs-local fidelity: the image below reproduces the local pipeline
group-for-group (76/76 identical) in an emulated linux/amd64 run.

## Container contract

| Path | Description |
|------|-------------|
| **Input** `/input/images/` | Test images (mounted read-only) |
| **Output** `/output/predictions.csv` | `filename,group_id` — every image exactly once |

Runtime: CPU-only, no internet access, per-run time limit.

Run it (linux/amd64 image; on Apple Silicon add `--platform linux/amd64`):

```bash
docker run --rm \
  -v /path/to/test/images:/input/images:ro \
  -v /path/to/outdir:/output \
  adamm13/autohdr-solution:v5
```

**Capacity honesty:** the pipeline is all-pairs O(n²). Measured throughput puts the
practical ceiling at roughly **450–550 images per 45-minute run** on a 16-vCPU box.
Above that, the documented headroom item is a shortlist pre-filter (cheap global
descriptor retrieval before all-pairs matching) — designed, not built, since every
labeled package we have fits under the ceiling.

## Demo

The same image carries a browser demo (drag-drop upload → labeled thumbnail clusters):

```bash
docker run --rm -p 8080:8080 adamm13/autohdr-solution:v5 \
  uvicorn server:app --host 0.0.0.0 --port 8080
```

(The default `CMD` is unchanged: `python solution.py`, the batch contract.)
`deploy/` contains App Runner scaffolding (ECR push + service definition + README with
cost notes and teardown) for hosting the demo; no cloud resources are created until
you run it.

## Repo layout

- `solution.py` — container entrypoint + validated config (the algorithm is in `grouper.py`)
- `grouper.py` — the full pipeline: matching, re-verify, guard, clustering policies, fuse-rescue
- `server.py`, `static/` — the demo server + page
- `evaluate.py` — local grader: replicates the `SCORING.md` scorer exactly, plus
  false-merge / false-split diagnostics
- `scripts/` — the experiment harnesses (prep, sweeps, per-lever measurements)
- `deploy/` — App Runner deploy scaffolding (not executed by default)
- `docs/project-context/` — notes from AutoHDR's explainer video + edge-case screenshots
- `research/` — feasibility research on in-browser grouping (stretch goal)
- `Dockerfile`, `.dockerignore` — image build
- `AGENTS.md` — running log of every experiment round and design decision
- `data/` — local datasets (gitignored)

## Local development

Download a labeled package (images + `public_manifest.csv` answer key) into `data/`:

| Package | Labeled images | Size | Download |
|---|---|---|---|
| Sample | 366 | ~1.4 GB | [autohdr_sample_500.zip](https://grouping-dataset-solution.s3.amazonaws.com/downloads/autohdr_sample_500.zip) |
| Medium | 2,126 | ~9 GB | [autohdr_medium_5000.zip](https://grouping-dataset-solution.s3.amazonaws.com/downloads/autohdr_medium_5000.zip) |
| Large | 10,000 | ~42 GB | [autohdr_large_10000.zip](https://grouping-dataset-solution.s3.amazonaws.com/downloads/autohdr_large_10000.zip) |

Environment: `.venv` on Python 3.11 (`opencv-python-headless`, `numpy`, `pillow`;
`fastapi`/`uvicorn`/`python-multipart` for the demo). See `Dockerfile` for pinned versions.

Grade a run:

```bash
python evaluate.py --manifest data/sample500/public_manifest.csv \
                   --predictions output/predictions.csv
```

Reference points on the sample: perfect grouping = 1.0; every-image-alone = 0.0725;
the current pipeline = 0.9275.

## Status: done vs backlog

Done: validated pipeline at the scores above; container image reproducible
group-for-group; local eval harness with merge/split diagnostics; demo server;
deploy scaffolding; systematic experiment log with sealed-holdout protocol.

Backlog (documented, deliberately not built): shortlist pre-filter for >~500-image
batches (throughput headroom); true in-browser grouping (see `research/` — orthogonal
to the scored pipeline).
