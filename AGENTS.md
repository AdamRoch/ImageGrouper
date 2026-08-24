# AGENTS.md — AutoHDR Image Grouping (hiring project)

## Project status

This repo started from AutoHDR's public challenge starter kit. The competition is over —
no prize — and the original challenge docs (`project-brief.md`, `SUBMISSION_GUIDE.md`,
`submission.yaml`) were removed 2026-08-23 for being outdated. This is a hiring/interview
project for the repo owner.

Submission channel: **dead** (verified 2026-08-23). codabench.org/competitions/15365
404s, bounty.autohdr.com is gone, Codabench signup confirmation emails never arrive, and
an external contestant confirms no access. No private-test grading exists unless the
hiring partner provides a channel. Local evaluation via `evaluate.py` + labeled packages
is the only score that exists.

Container contract (still honored — it's the artifact spec): no internet, CPU-only
(`cpu-large` 8 vCPU/16 GB, `cpu-xlarge` 16 vCPU/32 GB), ~30–45 min timeout. Built image
verified end-to-end: `adamm13/autohdr-solution:v1` (public on Docker Hub) reproduces the
local score group-for-group (0.7391 on the sample test-sim).

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

- Heavy local iteration: tune on sample-500 + medium spot subset; confirm on the holdout
  before believing any number (see "Accuracy push" below).
- Docker builds on this Mac require `--platform linux/amd64`; the image must be public
  on Docker Hub (verified working: `adamm13/autohdr-solution:v1`).

## Open questions

- Demo deployment: decided — public, AWS App Runner at a subdomain of adamroch.com
  (same image, server entrypoint). Not yet built.

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

## Experiment round 1 results (2026-08-23)

All levers beat baseline individually (baseline 0.4203 @ T=20 raw/verify_all):

- Lever 1: geometric-mean normalization (inliers / sqrt(kp_i*kp_j)) → 0.4783 @ T=0.02.
  min(kp) normalization HURTS (0.33) — unstable on low-texture images.
- Lever 2 (biggest win): gamma preprocessing (median luminance → 128 LUT) → 0.6957 @ T=35,
  **zero false merges** — much apparent cross-room leakage was exposure noise, not
  geometry. Keypoints/image rose ~753 → 1156. CLAHE weaker (0.5362) + 3x slower matching.
- Lever 3: verify-fraction f=0.5 on raw → 0.5942 (but doubles merges on dirty scores);
  f=0.75 on gamma → 0.7101 with merges at 2–3.
- Combination: gamma + norm_sqrt + f=0.75 @ T=0.018 → **0.7391 (51/69), 4 merges /
  14 splits**. Predictions: `output/combo/predictions_combo_f0.75_T0.018.csv`.
  Cached matrices in `.grouper_cache/` per variant (raw, gamma, clahe, norm_*, combo) —
  sweeps reproducible in seconds via `scripts/sweep_thresholds.py --variant ...`.

Residual failures: splits of the large groups (n=12–25) still dominate; merges are now
small look-alike rooms. Winner's-curse caveat: thresholds were swept on the same 69-group
key — confirm ordering on medium-5,000 spot-check before calibrating submission settings.

## Medium-set spot-check (2026-08-23)

`data/medium5000/` = 2,126 labeled images (538 groups, mostly n=3/n=5 brackets, thin
large-group tail). Spot subset `data/medium5000/spot/` via `scripts/prep_spot.py`:
902 images / 193 groups (all 19 groups n≥8 + sampled smaller), test-conditioned.

- **gamma > raw: holds emphatically** (0.689 vs 0.383; wider gap than the sample).
- **f=0.75 ≥ f=1.0 on gamma: holds** (0.767 vs 0.689; assembles 9/19 large groups vs 5/19).
- **norm_sqrt's extra gain did NOT replicate** (combo 0.7513 < gamma f=0.75 alone 0.7668
  on spot) — treat lever 1 as neutral, kept only because the combo config is exactly
  optimal on both datasets and normalization is safer under resolution/texture shift.
- **Combo T=0.018 transfers exactly** (spot peak too). Raw-inlier optima shift between
  datasets but sit on flat plateaus (±0.02), so exact T barely matters.
- Submission config: **gamma + norm_sqrt + f=0.75 @ T=0.018** (0.7391 sample / 0.7513
  spot — best-on-both). Alternative: gamma + f=0.75 raw @ T=14–17 (spot-best 0.7668).
- Medium-specific failures: n=5 brackets are now the top split source; big groups
  (n≥12) still split, and big groups attract cross-group members at low T.

**Throughput constraint**: gamma matching ≈ 113 pairs/s locally → all-pairs covers only
~850–1,000 images within a 30–45 min container limit at 8–16 vCPU. If the private test
serves > ~1,000 images per run, the shortlist pre-filter escape hatch is REQUIRED before
submission. Build it before the first submission unless test-set size is confirmed small.


## Accuracy push plan (2026-08-24)

Goal: maximize the honest local score. (An external "98%" claim was a local,
unverifiable number — do not chase it at the cost of overfitting.)

- **Holdout discipline**: the medium package has 1,224 labeled images never used in
  tuning (2,126 − 902 spot). Build a test-condition holdout from them. Tune on
  sample + spot; confirm on the holdout before accepting any change. More validation
  data can be minted from the 276K-image bucket (self-labeling `g<N>_` prefixes).
- **Error mass to attack** (measured): large-group splits (n≥12), then n=5 bracket
  splits; merges are small look-alike rooms.
- **Lever queue**: (1) door-check residual after alignment (designed, unimplemented);
  (2) exposure-bridge matching (dark→mid→bright chains instead of requiring
  dark↔bright pairs); (3) LightGlue matcher variant (ALIKED/DISK, not SuperPoint);
  (4) large-group-aware clustering.

## Holdout baseline + exposure-bridge result (2026-08-24)

`data/medium5000/holdout/` = 1,224 untouched medium images, 345 groups, test-conditioned
(**zero groups n≥8** — spot took them all; the holdout validates bracket handling, not
large-group assembly). Sealed-exam reference: current config (gamma + norm_sqrt + f=0.75
@ T=0.018) → **0.7449 (257/345), 38 merges / 50 splits**. The three evaluations cluster
tightly (sample 0.7391 / spot 0.7513 / holdout 0.7449): the config generalizes, no
tuning-set overfit. Residual error mass on the holdout: **n=5 brackets are the dominant
failure** (62.2% exact vs 84.0% for n=3); n=3s next. Attack 5-bracket splits next
(e.g. asymmetric small-group thresholds, higher-res re-verify), not the large-group tail.

Exposure-bridge lever (`POLICIES["chain"]`, luminance-adjacent join rule): **FAILED the
workshop gate** — tied sample at best (0.7391 @ T=0.04, above the config's T), lost spot,
merge-exploded at low T. Root cause: the binding constraint is the entry fee (truly
failed pairs have NO strong links to bridge with), not the join rule; brightness-adjacent
relaxation admits look-alike same-room images. Negative result recorded; holdout
confirmation not run (gate not met), holdout stays sealed.

## 5-bracket diagnosis + rescue levers (2026-08-24)

Diagnosis (`scripts/diagnose_n5.py`, cached matrices): ~100% of n=5 failure mass is
**exposure-extreme orphans** — almost always the BRIGHTEST member, with links at
0.5–0.9×T to its own group (passing 1–2 adjacent neighbors, failing the 75% join).
Failing groups have much wider exposure ranges (median luminance range ~217–232 vs
~165–169 passing). No two-cluster or global-depression cases.

Rescue levers (`POLICIES["rescue"]` / `["rescue_guarded"]` — exterior-luminance join
paths): **both FAILED the workshop gate.** Unguarded fixes real orphans but
merge-explodes on spot (9→18 merges). Guarded blocks true orphans too — a bright
extreme member often matches some OTHER bright image better than its own darker
neighbor. **The orphans' links are not separable from cross-group links on this score
matrix: clustering-rule surgery is exhausted; the problem is score-level.**

Next levers (score-level, test-valid — remember test images are 1024px max, so any
"re-verify at 2048px" idea is INVALID for the real artifact):
1. Pair-directed exposure equalization (extra-gamma the darker side when matching
   across big exposure gaps).
2. Borderline-pair re-verify with relaxed SIFT params (lower contrast threshold →
   more keypoints in low-contrast regions) at the same 1024px.
3. LightGlue matcher variant (the remaining planned matcher lever).
If none move the holdout: ~0.74–0.75 is this pipeline's honest ceiling — report as such.

## Score-level round: pair-directed equalization — BREAKTHROUGH (2026-08-24)

Lever 1 (`scripts/reverify_band.py`, band = 0.3–1.2×T pairs with raw-luminance gap ≥1.3,
histogram-match the darker side onto the brighter side's tonal distribution, re-detect
+ re-match, max-upgrade semantics): **PASSED all gates.** Separability is the target
signature — true orphan links ≥T: 22.7% → 81.6% (sample), 23.3% → 75.9% (spot),
27.4% → 96.8% (holdout); impostors flat-to-down. AUC 0.72 → 0.95 / 0.69 → 0.92 /
0.73 → 0.98.

Workshop: 0.7391 → **0.8116** (sample), 0.7513 → **0.7927** (spot) @ T=0.022.
Holdout confirmation (one run, pre-registered T=0.022): **0.7449 → 0.8145**
(+24 exact groups; merges 38 → 32 AND splits 50 → 32 simultaneously). n=5 brackets
62.2% → 74.8% exact — fixed the failure class it targeted. (Secondary, NOT
pre-registered: T=0.025 scored 0.8290 on holdout but spot prefers 0.022 — 0.022 stands.)

Lever 2 (relaxed SIFT): FAILED — absolute scores fall (norm_sqrt denominator inflates);
dominated by lever 1. LightGlue: parked — classical CV delivered; poor next-dollar
allocation vs integration risk.

**Container-capacity impact**: re-verify adds ~40–60% on top of base matching →
capacity ~550–700 images per 45-min cpu-xlarge run (was ~900–1,100). Shortlist
pre-filter becomes REQUIRED above that batch size if this config is wired into
`solution.py`.

## Merge-side round: camera-translation guard — BREAKTHROUGH (2026-08-24)

Levers (`scripts/merge_guards.py` + `apply_guards.py`; guards zero blocked candidate
edges before clustering):

- Lever A (door-check residual, coherent-blob after alignment): separability was real
  (P(cross>true) 0.93–0.99) but it blocks ~16% of true load-bearing links → splits
  rise, net never beats current. **FAILED the gate.**
- Lever B (camera-translation / cheirality guard): candidate pairs ≥T get re-matched
  with homography + essential-matrix pose decomposition; the discriminating signature
  is the **cheirality ratio** (fraction of matched points triangulating in front of
  both cameras under best pose): true same-position pairs median ~0.001–0.004,
  cross-group ~0.63–0.9. Block candidate edges with cheirality ≥ 0.5. (Essential
  inlier ratio was a dud — rotation-only pairs still admit high-inlier fits.)
  **PASSED all gates.** Holdout confirmation (pre-registered T=0.025): **0.8145 →
  0.8870** (+25 exact groups; merges 32 → 9, splits 32 → 30). Blocked 25,267 holdout
  candidate edges of which only 10 were true links — surgical.
- Rider C (widen histeq band below 0.3×T): NEGATIVE — ~220 impostor edges per ~4 true
  links rescued. Deep orphans are unrecoverable at affordable impostor cost with this
  matcher.

Score trajectory (holdout): 0.7449 → 0.8145 → **0.8870**. Recommended v3 container
config: histeq + cheirality guard @ T=0.025 (best combined three-set score, fewest
merges); T=0.022 the close alternative (0.8116/0.8290/0.8841). Guard cost: re-match +
pose for candidate pairs ≥T (~7% of pairs) ≈ 1.2× v2 cost.

Incident note: a rider-C run clobbered the sample histeq cache mid-round; restored
bitwise from a refactor-check backup and the script now suffixes non-default band
outputs. Fully recovered, all grades re-verified.

## Guard-gated rescue round — NEGATIVE, config final (2026-08-24)

Lever D (rescue joins for sub-T links gated on cheirality <0.5 AND residual blob
<0.35): **FAILED the workshop gate.** Relative separation is excellent (~100–175:1),
but the cross-group population is ~650× larger, so absolute pass-through is ~4
impostors per true link — the base-rate problem. Recovered splits but reintroduced
the merges the guard had just eliminated (spot merges 2→7). No T wins both sets.
Half of band pairs yield no homography at all (no evidence either way).

Lever E (LightGlue for dead pairs): skipped per precondition — rider C showed the
sub-0.3×T population is 98%+ impostor after histeq; recorded as standing contingency
with a pessimistic prior.

**Final config: gamma + norm_sqrt + histeq re-verify + cheirality guard (≥0.5 blocks)
+ f=0.75 @ T=0.025 → 0.8551 sample / 0.8187 spot / 0.8870 holdout** (T=0.022 the
robust alternative: 0.8116/0.8290/0.8841). Residual error (9 merges / 30 splits) is
dominated by deep orphans unrecoverable at affordable impostor cost with this matcher
— the pipeline has converged; ~0.89 is the honest ceiling. Holdout stays sealed after
the lever-B confirmation.
