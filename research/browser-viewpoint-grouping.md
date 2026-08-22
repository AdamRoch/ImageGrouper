# Browser-side viewpoint grouping of real-estate photos — research notes

Research date: 2026-08-20. All claims link to primary sources (official docs, GitHub repos, papers, model cards, official pricing pages). Where sources are silent or conflict, that is stated explicitly.

Problem restated: users upload 20–100 interior/exterior property photos in the browser; group them so that photos taken from the same camera position/angle land in one group even when exposure differs drastically (flash vs. ambient, exposure brackets), while different rooms/angles form separate groups. Question: what is feasible in-browser before upload, what belongs on a backend, and which models fit.

---

## 1. transformers.js / ONNX Runtime Web — current state

**Backends.** Transformers.js runs models via ONNX Runtime; in the browser the default is CPU via WASM, and WebGPU is enabled per-model with `device: 'webgpu'`. Quantized variants are selected with `dtype` (typical options: `"fp32"` default for WebGPU, `"fp16"`, `"q8"` default for WASM, `"q4"`). Source: https://huggingface.co/docs/transformers.js/en/index

**v3 release (Oct 2024).** Transformers.js v3 added WebGPU support ("up to 100x faster than WASM" — a marketing claim from the release blog, not a per-model benchmark), 120 supported architectures, and "over 1200 pre-converted models on the Hugging Face Hub". Source: https://huggingface.co/blog/transformersjs-v3

**ONNX Runtime Web WebGPU EP.** Official tutorial confirms a `webgpu` execution provider for the browser build, with features like IO binding (keep tensors on GPU) and graph capture. Note: this page's browser-availability statement is stale — it lists "Firefox behind a flag and Safari Technology Preview", which predates both engines shipping WebGPU by default in 2025 (see §8). Source: https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html

**Officially hosted / converted vision models.** The transformers.js source tree has first-class model implementations for CLIP, SigLIP, DINOv2, DINOv2-with-registers, JinaCLIP, Chinese-CLIP, and CLIPSeg (https://github.com/huggingface/transformers.js/tree/main/packages/transformers/src/models). The docs list `image-feature-extraction` as a supported pipeline task. Source: https://huggingface.co/docs/transformers.js/en/index

**Model sizes in MB (from the HF Hub file listings — measured via the Hub API on 2026-08-20):**

- `Xenova/clip-vit-base-patch32` — vision encoder only: fp32 `vision_model.onnx` 351.7 MB, fp16 176.1 MB, quantized (int8) 89.1 MB. (Full dual-encoder `model.onnx` fp32 is 605.8 MB.) https://huggingface.co/Xenova/clip-vit-base-patch32/tree/main/onnx
- `onnx-community/dinov2-small` — fp32 88.5 MB, fp16 44.4 MB, int8 24.4 MB. https://huggingface.co/onnx-community/dinov2-small
- `onnx-community/dinov2-base` — fp32 346.6 MB, fp16 173.5 MB, int8 91.0 MB. https://huggingface.co/onnx-community/dinov2-base
- `onnx-community/siglip2-base-patch16-224-ONNX` — vision encoder fp16 186.0 MB, int8 94.6 MB. https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX
- `Xenova/siglip-base-patch16-384` — vision encoder fp16 186.7 MB, int8 ~95–100 MB. https://huggingface.co/Xenova/siglip-base-patch16-384

**Official performance numbers for in-browser image encoding.** There are no official per-image latency numbers for CLIP/DINOv2 embedding extraction specifically. The closest official benchmark: ONNX Runtime Web with WebGPU accelerated the Segment Anything ViT encoder **19×** vs the WASM EP (RTX 3060 laptop) — a reasonable proxy since SAM's encoder is a ViT like CLIP/DINOv2. Source: https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/ . Chrome's WebGPU launch post claims ">3x improvements in machine learning model inferences" vs WebGL. Source: https://developer.chrome.com/blog/webgpu-release

## 2. Embedding robustness to illumination (CLIP / DINOv2 / SigLIP)

**The primary sources are largely silent on exposure-bracket stability.** The CLIP paper's robustness analysis concerns natural distribution shift and zero-shot transfer, not exposure/brightness perturbation of the same scene (https://arxiv.org/abs/2103.00020). The DINOv2 paper claims "all-purpose visual features" that beat OpenCLIP on most benchmarks and evaluates instance-level recognition and copy detection (near-duplicate retrieval under edits), but the paper text contains no illumination-robustness claim — grep of the full text finds no mention of "illumination" (https://arxiv.org/abs/2304.07193). The SigLIP paper is about the sigmoid loss and training efficiency, with no illumination analysis (https://arxiv.org/abs/2303.15343). Whether cosine similarity of these embeddings stays above a usable threshold across flash-vs-ambient brackets of an interior is, therefore, **not answered by the primary sources**; it must be measured empirically for the chosen model.

**Where illumination robustness *is* documented**: the local-feature / place-recognition literature. The HPatches benchmark (CVPR 2017) consists of 116 sequences — 57 with photometric (illumination) changes and 59 with viewpoint changes — with homography ground truth, and is the standard benchmark for matching under exactly this nuisance factor (https://arxiv.org/abs/1704.05939, full text §4). LightGlue's paper states training used "strong photometric augmentations, including blur, hue, saturation, sharpness, illumination, gamma and noise", and LightGlue is evaluated on HPatches (https://arxiv.org/abs/2306.13643, full text). Visual place recognition models are explicitly trained to be invariant to both illumination and viewpoint (see §5).

## 3. Learned local-feature matchers in the browser (LoFTR, LightGlue, SuperGlue)

**LightGlue.** Official repo (cvg/LightGlue): takes keypoints+descriptors per image and returns correspondence indices; pretrained with SuperPoint, DISK, ALIKED, and SIFT features. Benchmarks from the README: 150 FPS @ 1024 keypoints on RTX 3080 (GPU, with compile+adaptivity); 20 FPS @ 512 keypoints on Intel i7-10700K CPU. Licenses: LightGlue code+weights Apache-2.0; DISK Apache-2.0; ALIKED BSD-3-Clause; **SuperPoint weights/inference code have a restrictive non-Apache license**. Source: https://github.com/cvg/LightGlue

**LightGlue-ONNX (community, fabio-sim)** is the de-facto ONNX port: exports full extractor→matcher pipelines, prebuilt ONNX assets on GitHub Releases, and **ships an in-repo browser WebGPU demo** (`uv run python web/serve.py`, served at localhost). Model sizes from Releases: `superpoint_lightglue_pipeline.onnx` 51.2 MB (v2.0), fp16 variants ~26 MB (v1.0.0, `*_fp16.onnx` 23–27 MB), latest `raco_aliked_lightglue_pipeline_k*.onnx` ~64–66 MB (v3.0). Standalone SuperPoint ONNX is ~5 MB. Source: https://github.com/fabio-sim/LightGlue-ONNX

**Browser caveat (known bug).** An onnxruntime GitHub issue (June 2025, ORT Web 1.22.0) reports SuperPoint+LightGlue-ONNX works correctly on the WASM EP and WebNN in onnxruntime-web, but the **WebGPU EP returns keypoints with no matches** — i.e., plan to run the matcher on WASM (or verify the bug is fixed) before assuming GPU acceleration. Source: https://github.com/microsoft/onnxruntime/issues/25227

**LoFTR.** Detector-free dense matcher whose Transformer self/cross-attention gives a global receptive field that "produce[s] dense matches in low-texture areas, where feature detectors usually struggle" (https://arxiv.org/abs/2104.00680) — directly relevant to interiors with blank walls. Official code is PyTorch-only (https://github.com/zju3dv/LoFTR). **No official ONNX or browser/WebGPU port exists**; third-party conversion repos exist but are unofficial and have reported correctness problems. Treat browser-LoFTR as a DIY ONNX export project, not an off-the-shelf option.

**SuperGlue.** Superseded by LightGlue for practical purposes (LightGlue is 4–10× faster per the cvg README benchmark). No official ONNX/web port.

## 4. Classical CV in the browser — OpenCV.js

**What the default JS build exposes** (from the official build whitelist, `platforms/js/opencv_js.config.py` in opencv/opencv 4.x — this file is the authoritative list of what gets compiled into opencv.js): https://github.com/opencv/opencv/blob/4.x/platforms/js/opencv_js.config.py

- features2d: `Feature2D.detect/compute/detectAndCompute`, **ORB, BRISK, KAZE, AKAZE**, MSER, FastFeatureDetector, AgastFeatureDetector, GFTTDetector, `DescriptorMatcher.match/knnMatch`, `BFMatcher`. **SIFT is not in the whitelist** — it is not available in the default opencv.js build (it would require a custom Emscripten build).
- calib3d: **`findHomography`**, `estimateAffine2D`, `solvePnPRansac`, `UsacParams` (modern USAC-family RANSAC variants).
- photo: `createAlignMTB` (exposure-bracket alignment), `createMergeMertens`/`createMergeDebevec`/`createMergeRobertson` (exposure fusion / HDR merge) — useful if brackets should also be fused after grouping.

**Patents.** SIFT's patent expired in March 2020 and OpenCV moved SIFT into the main repository (tracking issue: https://github.com/opencv/opencv/issues/16736). ORB was created at OpenCV Labs explicitly as a patent-free alternative to SIFT/SURF ("Yes, SIFT and SURF are patented... But ORB is not !!!") per the official OpenCV tutorial: https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html . So SIFT is legally fine but missing from the stock JS build; ORB/AKAZE/KAZE/BRISK are patent-free and present.

**Feasibility of 100-image pairwise matching in-browser.** 100 images ⇒ 4,950 pairs. No official OpenCV.js performance numbers exist; the only primary-source performance claim is qualitative — ORB "is a good choice in low-power devices" (same tutorial). Whether ORB+RANSAC on 4,950 pairs is "fast" in WASM depends on device and image resolution and must be benchmarked on target hardware. ORB descriptors are binary and match with Hamming distance, which is the cheapest option among the exposed detectors.

## 5. Place recognition / image retrieval models — wrong invariance direction

This family is optimized for the **opposite** of the task: recognizing the same *place* despite viewpoint and illumination changes, i.e., making different viewpoints of one place look *similar*. That blurs exactly the distinction the app needs (different angles of the same room should be *separated*).

- NetVLAD: CNN for large-scale place recognition; robustness to lighting and viewpoint change is a design goal (https://arxiv.org/abs/1511.07247).
- CosPlace: casts visual geo-localization as classification over large-scale data (https://arxiv.org/abs/2204.02287).
- EigenPlaces: explicitly trains "images from different point of views, which embeds viewpoint robustness into the learned global descriptors" (https://arxiv.org/abs/2308.10832).
- AnyLoc: builds universal VPR descriptors from frozen DINOv2 features + VLAD aggregation, targeting deployment "anywhere, anytime, and across anyview", up to 4× over prior VPR approaches (https://arxiv.org/abs/2308.00688). This is also the strongest primary evidence that plain DINOv2 embeddings are *place*-sensitive and illumination-robust — good for "same room/building?", but not designed to separate viewpoints within a room.

**No primary source found describing a model explicitly trained for "same camera viewpoint" clustering.** The task decomposes into known pieces instead: (a) near-duplicate detection across exposure changes (embeddings or local features), (b) geometric verification of overlap (matchers/homography), (c) appearance-sensitive embeddings to split rooms/angles.

## 6. VLM APIs for grouping — cost and latency

All numbers below are from official pricing/docs pages on 2026-08-20; the cost figures for 100 images are arithmetic from those published rates.

**OpenAI.** Up to 500 images per request, 50 MB total payload (https://platform.openai.com/docs/guides/images). Token accounting: GPT-4o/4.1/o-series (except o4-mini): base 85 + 170 per 512px tile in `high` detail (a 1024×1024 image = 765 tokens); `low` detail = flat 85 tokens. Newer models (GPT-4.1-mini/nano, GPT-5-mini/nano) use 32px-patch counting with a model multiplier (e.g., 4.1-mini: 1024×1024 → ⌈1024×1.62⌉ = 1,659 tokens ≈ $0.00066 at $0.40/1M input). GPT-4o input is $2.50/1M tokens (https://platform.openai.com/docs/pricing). **Estimate: 100 images ≈ $0.007 (4.1-mini high detail) to ~$0.19 (GPT-4o high detail) in input tokens, plus output.**

**Anthropic.** Images are tokenized as ⌈w/28⌉×⌈h/28⌉ visual tokens; the standard tier caps at 1,568 tokens/image (long edge 1,568 px). Up to 100 images per API request on 200k-context models, 600 on others; many-image requests have stricter per-image pixel limits (https://docs.anthropic.com/en/docs/build-with-claude/vision). Claude Haiku 4.5 input is $1/1M tokens; the docs' own example: a 1000×1000 image (1,296 tokens) costs ≈$1.30 per thousand images on Haiku 4.5 (https://docs.anthropic.com/en/docs/about-claude/pricing). **Estimate: 100 images ≈ $0.13–0.16 input on Haiku 4.5.**

**Google Gemini.** 258 tokens per image if both dimensions ≤384px; larger images are tiled into 768×768 crops at 258 tokens/tile; up to 3,600 images per request (https://ai.google.dev/gemini-api/docs/image-understanding). Gemini 2.5 Flash-Lite input is $0.10/1M tokens (text/image/video), Gemini 2.5 Flash $0.30/1M (https://ai.google.dev/gemini-api/docs/pricing). **Estimate: 100 images ≈ $0.0026 (Flash-Lite, downscaled) to ~$0.05 (Flash, tiled full-res) input.**

**Practicality.** At these rates, a server-side "send 100 thumbnails, ask for viewpoint/room groups" call is trivially cheap (cents or sub-cent on Gemini/mini tiers) and fits in one request within all three providers' per-request image limits. Latency is not published officially; expect a multi-second single round trip. Caveats: providers publish no accuracy numbers for viewpoint-grouping prompts, and usage policies/limits apply; this is an inference about practicality from pricing and request limits, not a benchmarked result.

## 7. Clustering libraries in JS

- `mljs/hclust` — hierarchical (agglomerative, AGNES) clustering in JavaScript, MIT, maintained (ml-hclust 4.0.0, npm modified 2025-11): https://github.com/mljs/hclust
- `ml-dbscan` — DBSCAN from the same ml.js family: https://mljs.github.io/dbscan/
- `hdbscan-ts` — TypeScript HDBSCAN implementation on npm (1.0.17, modified 2026-02): https://www.npmjs.com/package/hdbscan-ts

For ~100 512–768-dim embeddings, any of these is computationally trivial; the choice is API/maintenance preference, not performance. Threshold-based connected components over the cosine-distance matrix is also adequate at this scale and needs no dependency.

## 8. Practical browser constraints

**Model caching.** Transformers.js caches downloaded model files in the browser's **Cache API** by default when available (`env.useBrowserCache`, default `true` if the Cache API exists; `env.cacheKey` default `'transformers-cache'`; Node uses a filesystem cache via `useFSCache`; a custom cache implementing `match`/`put` is supported). Source (source code, definitive): https://github.com/huggingface/transformers.js/blob/main/packages/transformers/src/env.js . Custom/local model hosting is configurable via `env.localModelPath` / `env.allowRemoteModels`: https://huggingface.co/docs/transformers.js/custom_usage

**Web Workers.** The official transformers.js vanilla-JS tutorial explicitly warns that running models on the main thread freezes the UI and recommends a Web Worker: https://github.com/huggingface/transformers.js/blob/main/packages/transformers/docs/source/tutorials/vanilla-js.md

**WebGPU support status (engine-official sources):**
- Chrome/Edge: default since Chrome 113 (Apr 2023) on ChromeOS/macOS/Windows; Android since Chrome 121 (https://developer.chrome.com/blog/webgpu-release , https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/).
- Safari: WebGPU shipped in Safari 26.0 (Sept 2025) for macOS, iOS, iPadOS, visionOS (https://webkit.org/blog/17333/webkit-features-in-safari-26-0/).
- Firefox: default on Windows since Firefox 141 (July 2025), other platforms rolling out later (https://mozillagfx.wordpress.com/2025/07/15/shipping-webgpu-on-windows-in-firefox-141/).
- Aggregate support table: https://caniuse.com/webgpu . Transformers.js v3's release blog cited ~70% global support as of Oct 2024 — a figure that has since improved with Safari 26 and Firefox 141, but WASM remains the required fallback path.

---

## Bottom line for the design question

**(a) Best in-browser approach for same-viewpoint grouping under brightness variation.** The evidence supports a two-stage local pipeline: (1) cheap global embeddings to shortlist candidate pairs, and (2) **geometric verification with local-feature matching**, which is the only approach whose primary literature explicitly targets illumination change (HPatches benchmark; LightGlue's photometric-augmentation training). Concretely: compute per-image embeddings in a Web Worker with transformers.js (e.g., DINOv2-small int8, 24 MB, or CLIP ViT-B/32 vision q8, 89 MB — sizes in §1) on WebGPU with WASM fallback; cluster/shortlist by cosine distance; then verify candidate pairs with ORB+`knnMatch`+`findHomography`/`estimateAffine2D`+USAC in OpenCV.js (patent-free, in the stock JS build) or SuperPoint/ALIKED+LightGlue-ONNX (~26–66 MB; browser demo exists, but run it on the WASM EP until the WebGPU-EP matcher bug in onnxruntime#25227 is confirmed fixed). Match/inlier count per pair is a direct "same viewpoint?" score, and OpenCV.js even ships exposure-bracket alignment (`AlignMTB`) and fusion (`MergeMertens`) for the HDR use case. Caveat to verify empirically: no primary source quantifies CLIP/DINOv2 embedding drift across exposure brackets, so embedding thresholds must be calibrated on real bracketed interior photos — which is exactly why the matcher verification stage should be authoritative.

**(b) What belongs server-side.** Nothing in this pipeline *requires* a server at 20–100 images. Server-side is justified for: heavyweight matching if in-browser WASM matching benchmarks poorly on low-end devices (LoFTR is GPU-oriented and has no browser port — its low-texture robustness is attractive for blank-wall interiors, so it is a server-side option); higher-accuracy re-verification of ambiguous pairs; and any VLM call (API keys must not live in the browser anyway). Place-recognition models (NetVLAD/CosPlace/EigenPlaces/AnyLoc) are the wrong tool — they are trained to erase viewpoint differences, not detect them.

**(c) Is a VLM needed at all?** Not for the core task — no primary source shows frontier VLMs benchmarked on same-viewpoint grouping, while the geometric approach directly computes the required property. A VLM is a reasonable *optional* server-side fallback or labeling pass (e.g., room naming, sanity-checking odd groups): all three providers accept ~100 images in a single request, at an estimated input cost between ~$0.003 (Gemini 2.5 Flash-Lite) and ~$0.19 (GPT-4o high detail) per 100-image batch. That is cheap enough to be a fallback, not a dependency.

**Open gaps (sources silent):** official in-browser latency numbers for CLIP/DINOv2 embedding extraction; embedding cosine stability across exposure brackets for CLIP/DINOv2/SigLIP; OpenCV.js matching throughput numbers; current status of the LightGlue-on-WebGPU bug (check onnxruntime#25227 before shipping).
