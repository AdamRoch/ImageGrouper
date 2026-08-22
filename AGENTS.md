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
