# AutoHDR Challenge — Project Brief

## Task

Given a folder of real estate photos from photoshoots, identify which images were taken from the same camera angle. Each angle may have multiple exposures (HDR brackets), but your job is to figure out which images belong together.

- **Input:** A folder of JPEG images (resized to 1024px max, randomized filenames).
- **Output:** A CSV file grouping images by camera angle.

## How It Works

You build a Docker container that reads images and writes a predictions CSV. We run your container against our private test set, score the output, and update the leaderboard.

## Your Container's Contract

| Path | Description |
|------|-------------|
| **Input** `/input/images/` | Test images (mounted read-only) |
| **Output** `/output/predictions.csv` | Your grouping predictions |

## predictions.csv Format

```csv
filename,group_id
a7f3b2c1.jpg,0
d4e5f6a7.jpg,0
b8c9d0e1.jpg,1
f2a3b4c5.jpg,2
```

- Every image must appear exactly once
- Images in the same group share a `group_id`
- `group_id` can be any string or number
- Use filenames only (no paths)

## Quick Start

### 1. Get the Starter Kit

```bash
git clone https://github.com/AutoHDRHackathon/autohdr-challenge-starter.git
```

Contains: `solution.py`, `Dockerfile`, `submission.yaml`

### 2. Implement Your Algorithm

Edit `solution.py`:

```python
def group_images(image_paths: list[str]) -> list[list[str]]:
    return [
        ["a7f3b2c1.jpg", "d4e5f6a7.jpg"],   # same camera angle
        ["b8c9d0e1.jpg"],                    # different angle
        ["f2a3b4c5.jpg", "e6f7a8b9.jpg"],    # another angle
    ]
```

### 3. Build and Test Locally

```bash
docker build --platform linux/amd64 -t my-solution:v1 .
docker run -v /path/to/images:/input/images:ro -v /tmp/output:/output my-solution:v1
```

**Mac users:** `--platform linux/amd64` is required or your container will crash.

### 4. Push to Docker Hub

```bash
docker login
docker tag my-solution:v1 yourusername/autohdr-solution:v1
docker push yourusername/autohdr-solution:v1
```

Make sure your Docker Hub repo is **PUBLIC**.

### 5. Submit on Codabench

Create `submission.yaml` (template in starter kit):

```yaml
docker_image: yourusername/autohdr-solution:v1
machine_type: cpu-xlarge
email: your-registered-email@example.com
```

- `machine_type`: `cpu-large` (8 vCPU, 16 GB) or `cpu-xlarge` (16 vCPU, 32 GB)
- `email`: must match your registration at bounty.autohdr.com

```bash
zip submission.zip submission.yaml
```

Upload `submission.zip` on the **My Submissions** tab.

It will take a while to run — go get a coffee and come back in an hour or so. We'll email you if something goes wrong.

## Scoring

```
score = exact_matches / total_groups
```

An **exact match** means your predicted group contains exactly the same set of filenames as the labeled group. **No partial credit.**

| Scenario | Score |
|----------|-------|
| All groups predicted perfectly | 1.0 |
| Baseline (each image alone) | ~0.09 |
| All images in one group | 0.0 |

## Training Data

Full dataset (266K images):

```bash
aws s3 sync s3://grouping-dataset-solution/images/ ./images/ --no-sign-request
```

## Rules

- US-based contestants only
- Your container runs with **no internet access**
- Time limit: **60 minutes** per submission
- Machine types: `cpu-large` (8 vCPU, 16 GB RAM) or `cpu-xlarge` (16 vCPU, 32 GB RAM)
- Maximum **3 submissions per day**
- Print progress to stdout (shows up in your submission logs)

## Tips

- Images from the same angle are typically exposure brackets (dark, mid, bright) of the same scene
- Filenames are randomized UUIDs
- Group sizes vary (singles, 3s, 5s, 7+ brackets)
- Test locally before submitting to save your daily limit
