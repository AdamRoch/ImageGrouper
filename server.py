"""
AutoHDR demo server — browser demo around the validated grouper (v3 config).

Alternate entrypoint alongside the solution.py CLI (which is unchanged and
remains the submission contract). Run:

    uvicorn server:app --host 0.0.0.0 --port 8080

Endpoints:
    GET  /                        static demo page
    GET  /health                  liveness
    POST /api/group               multipart JPG/PNG upload (<=100 files, <=8MB each)
                                  -> {"job_id"}; 429 if a job is already running
    GET  /api/status/{job_id}     {state, progress}
    GET  /api/results/{job_id}    {groups: [[{name, url}, ...], ...]}
    GET  /api/images/{job_id}/{name}  uploaded image (thumbnails)

One job at a time (demo box), in-memory registry, artifacts under
/tmp/autohdr_jobs with a 30-minute TTL.
"""

import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import grouper
from solution import CONFIG  # the validated v3 config — single source of truth

JOB_ROOT = Path("/tmp/autohdr_jobs")
JOB_TTL_S = 30 * 60
MAX_FILES = 100
MAX_FILE_BYTES = 8 * 1024 * 1024
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}

app = FastAPI(title="AutoHDR Grouping Demo")

_jobs: dict[str, dict] = {}
_busy = threading.Lock()
_registry_lock = threading.Lock()


def _sanitize(name: str) -> str:
    base = os.path.basename(name)
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in base) or "image"


def _cleanup_loop():
    while True:
        time.sleep(60)
        now = time.time()
        with _registry_lock:
            stale = [jid for jid, j in _jobs.items()
                     if now - j["created"] > JOB_TTL_S and j["state"] != "running"]
            for jid in stale:
                _jobs.pop(jid, None)
                shutil.rmtree(JOB_ROOT / jid, ignore_errors=True)


threading.Thread(target=_cleanup_loop, daemon=True).start()


def _run_job(job_id: str, paths: list[str]):
    """Mirror grouper.group_images(**CONFIG) stage by stage, with progress notes."""
    job = _jobs[job_id]
    try:
        workers = os.cpu_count()
        names = [os.path.basename(p) for p in paths]

        job["progress"] = f"extracting SIFT features from {len(paths)} images"
        kps, descs = [], []
        sift = cv2.SIFT_create()
        for k, p in enumerate(paths, 1):
            img = grouper.load_gray(p, preprocess=CONFIG["preprocess"])
            pts, desc = grouper.detect(sift, img)
            kps.append(pts)
            descs.append(desc)
            if k % 10 == 0 or k == len(paths):
                job["progress"] = f"features {k}/{len(paths)}"

        job["progress"] = "all-pairs matching"
        matrix = grouper.compute_score_matrix(kps, descs, workers=workers)
        matrix = grouper.normalize_score_matrix(matrix, kps, CONFIG["normalize"])

        luminance = np.array([grouper.load_gray(p).mean() for p in paths], dtype=np.float32)
        images = [grouper.load_gray(p, preprocess=CONFIG["preprocess"]) for p in paths]

        job["progress"] = "re-verifying exposure-extreme pairs (histogram equalization)"
        matrix = grouper.reverify_histeq(matrix, paths, luminance, kps, descs, workers=workers)

        job["progress"] = "camera-motion guard (pose decomposition)"
        ii, jj, _, cheir = grouper.compute_guard_metrics(
            matrix, paths, images, luminance, kps, descs, CONFIG["threshold"], workers)
        matrix = grouper.apply_cheirality_guard(matrix, ii, jj, cheir)

        job["progress"] = "clustering"
        clusters = grouper.cluster_verify_all(matrix, CONFIG["threshold"], CONFIG["fraction"])

        job["progress"] = "fused-stack orphan rescue"
        clusters = grouper.fuse_rescue(matrix, clusters, images, luminance, kps, descs,
                                       CONFIG["fuse_threshold"], workers)

        groups = [[names[i] for i in c] for c in clusters]
        job["groups"] = groups
        job["state"] = "done"
        job["progress"] = f"done — {len(groups)} groups"
    except Exception as exc:  # noqa: BLE001 — demo server, report anything
        job["state"] = "error"
        job["progress"] = f"{type(exc).__name__}: {exc}"
    finally:
        _busy.release()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/group", status_code=202)
async def start_group(files: list[UploadFile] = File(...)):
    if not _busy.acquire(blocking=False):
        raise HTTPException(429, "a grouping job is already running — try again shortly")
    try:
        if not files:
            raise HTTPException(400, "no files uploaded")
        if len(files) > MAX_FILES:
            raise HTTPException(413, f"too many files (max {MAX_FILES})")

        job_id = uuid.uuid4().hex[:12]
        img_dir = JOB_ROOT / job_id / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for k, f in enumerate(files):
            name = _sanitize(f.filename or f"image_{k}")
            ext = Path(name).suffix.lower()
            if ext not in ALLOWED_EXT:
                raise HTTPException(415, f"{name}: only JPG/PNG supported")
            data = await f.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(413, f"{name}: exceeds {MAX_FILE_BYTES // 2**20} MB limit")
            dest = img_dir / f"{k:04d}_{name}"
            dest.write_bytes(data)
            paths.append(str(dest))

        with _registry_lock:
            _jobs[job_id] = {"state": "running", "progress": "queued",
                             "created": time.time(), "groups": None}
        threading.Thread(target=_run_job, args=(job_id, paths), daemon=True).start()
        return {"job_id": job_id}
    except Exception:
        _busy.release()
        raise


@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown or expired job")
    return {"state": job["state"], "progress": job["progress"]}


@app.get("/api/results/{job_id}")
def job_results(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown or expired job")
    if job["state"] != "done":
        raise HTTPException(409, f"job not done (state={job['state']})")
    return {
        "groups": [
            [{"name": name, "url": f"/api/images/{job_id}/{name}"} for name in sorted(g)]
            for g in job["groups"]
        ]
    }


@app.get("/api/images/{job_id}/{name}")
def job_image(job_id: str, name: str):
    if job_id not in _jobs:
        raise HTTPException(404, "unknown or expired job")
    path = (JOB_ROOT / job_id / "images" / _sanitize(name)).resolve()
    if not str(path).startswith(str((JOB_ROOT / job_id).resolve())) or not path.exists():
        raise HTTPException(404, "no such image")
    return FileResponse(path)


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
