# AGENTS.md — AutoHDR Image Grouping (hiring project)

## Project status

The docs in this repo (`README.md`, `project-brief.md`, `SUBMISSION_GUIDE.md`, `SCORING.md`)
describe a cash-prize competition. **That competition is over — there is no prize.** This is
now a hiring/interview project for the repo owner. Treat all prize/leaderboard framing in
those files as outdated.

Believed still true (unconfirmed — verify before relying on):

- Evaluation still runs through the same container contract + Codabench submission flow.
- ~3 test submissions per day against their private test set.
- Container constraints: no internet, CPU-only (`cpu-large` 8 vCPU/16 GB, `cpu-xlarge`
  16 vCPU/32 GB). Timeout docs conflict: `README.md` says 60 min, `SUBMISSION_GUIDE.md`
  says 30/45 min — design for the conservative number.

## Scoring

Exact-match only: `score = exact_matches / total_reference_groups`. No partial credit —
one wrong filename fails the whole group. A false merge kills two groups; a false split
kills one — bias toward splitting when ambiguous. Details in `SCORING.md`.

## Data

- Labeled public packages (include `public_manifest.csv` answer keys): sample 500 (~2 GB),
  medium 5,000 (~21 GB), large 10,000 (~42 GB). Download links in `README.md`.
- Full bucket `s3://grouping-dataset-solution/images/` (~276K images) self-labels:
  the `g<N>_` filename prefix is the group id.
- Test images are resized to 1024px max with randomized UUID filenames. Training package
  images may be full-res — match test conditions when evaluating.
- The private test set never leaves their infra; the only grade on it comes via submission.

## Settled design decisions

Priority order: **(1) algorithm accuracy in the container, (2) browser-UI demo around the
container, (3) true client-side grouping — stretch/backlog only.**

Algorithm (`solution.py`):

- Pairwise test: classical local features (SIFT/AKAZE via OpenCV) → descriptor match →
  RANSAC homography/affine fit → inlier count is the same-angle score; plus a
  post-alignment residual check for the door-open/closed edge case.
- LightGlue (ALIKED or DISK extractor — **not** SuperPoint, restrictive license) as an
  early benchmarked harness variant, not the foundation.
- Clustering: verify-then-merge (a candidate joins a group only if it verifies against
  ALL current members) as the default; single-link agglomerative ("either") kept as a
  harness variable.
- Assume handheld jitter and pipeline re-encoding: nothing is pixel-identical. The inlier
  threshold separating "wobble" from "intentional reposition" is calibrated on labeled data.
- Performance: measure-first. All-pairs parallelized across vCPUs; match at ~512px,
  re-verify borderline pairs at 1024px; dormant shortlist escape hatch if n is huge
  (~>1,000 images per run).

Eval harness (to build):

- `evaluate.py` replicating the `SCORING.md` scorer exactly (frozenset intersection), plus
  false-merge / false-split diagnostics reported separately.
- Bake-off variables, changed one at a time: feature/matcher (SIFT / AKAZE / LightGlue),
  inlier threshold, clustering policy.

Workflow:

- Heavy local iteration against the sample-500 (spot-check medium-5,000 to avoid
  overfitting); submissions spent strategically — cadence is the owner's call.
- One early unmodified-starter submission to de-risk Docker/Codabench logistics.
- Docker builds on this Mac require `--platform linux/amd64`; the image must be public
  on Docker Hub.

## Open questions

- Demo deployment: public AWS (App Runner = simple lean; ECS/ALB = more standard talking
  points) vs local-only during interviews — undecided.
- Whether the 3/day submission limit and Codabench flow still operate post-competition.

## Backlog (stretch only)

- True in-browser grouping (research: `research/browser-viewpoint-grouping.md`):
  DINOv2-small/CLIP embeddings in a Web Worker + ORB/LightGlue WASM verification, with
  runtime WebGPU capability detection. Does not improve the scored algorithm.

## Current state (2026-08-23)

Baseline implemented and scored locally:

- `grouper.py` — SIFT + Lowe ratio test (0.75) + RANSAC-homography inlier counts over all
  pairs; parallelism via **threads, not fork-based multiprocessing** (fork + OpenCV
  deadlocks/segfaults on macOS — one crash dialog already observed). Score matrix cached
  in `.grouper_cache/`. Clustering policies in `POLICIES`: `verify_all` (default) and
  `single_link`. `grouper.group_images()` matches `solution.py`'s contract;
  `solution.py` remains the untouched starter stub until submission wiring.
- `scripts/prep_test_sim.py` — builds `data/sample500/test_sim/` (1024px max, uuid4
  filenames, own `manifest.csv`) to simulate test conditions.
- `scripts/sweep_thresholds.py` — re-clusters from the cached matrix at multiple
  thresholds and grades each (experiments cost seconds, not a re-match).
- Local env: `.venv` on Python 3.11 (`/Users/adam/.local/bin/python3.11`) to match the
  Docker image (`python:3.11-slim`); cv2 5.0.0, numpy, pillow. Avoid the Homebrew
  Python 3.14 for this work.

Baseline results (366 images, 69 reference groups, verify_all): best T=20 → **0.4203**
(29/69), 5 false merges / 35 false splits. single_link is worse (best 0.3913 at T=100,
chain-merges at low T). All-alone baseline 0.0725. Full-run runtime ~5.5 min locally
(~207 pairs/s, 10 threads).

Failure profile: splits dominate; the large groups (n=12–25) never assemble because
extreme-exposure pairs lose texture and match weakly. Cross-group leakage (same room,
different angle shares structure) blocks simply lowering T (false merges double at T=10).

Next levers, one variable at a time from the T=20 / verify_all baseline:

1. Normalized pair score (e.g. inliers / min(keypoint counts)) so texture-rich images
   don't dominate.
2. Exposure-invariant preprocessing (gamma / CLAHE) before matching — targets the
   dominant split failure.
3. Softer clustering (join if verified against a fraction of members) so large groups
   with a few dead pairs survive.
