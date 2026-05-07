"""
core/camera.py — Main camera capture loop.

Opens the video source, runs YOLO person detection every N frames,
detects faces within person ROIs, tracks them, and pushes stable
frontal faces to Redis for background recognition.
"""

import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import redis
from ultralytics import YOLO

import db
import app_state
from config import (
    CAMERA_ID,
    CAMERA_DIRECTION,
    DETECTION_WIDTH,
    JPEG_QUALITY,
    LOG_FILE,
    PROCESS_EVERY_N_FRAMES,
    REDIS_URL,
    REDIS_QUEUE_KEY,
)
from core.detector import detect_faces_in_roi, encode_jpeg, encode_thumbnail, load_cascades
from core.tracker import FaceTracker

log = logging.getLogger(__name__)


# ── Redis queue helper ────────────────────────────────────────────────────────

def _push_to_queue(r: redis.Redis, sighting_id: int, face_jpeg: bytes):
    job = {"sighting_id": sighting_id, "face_image_hex": face_jpeg.hex()}
    r.rpush(REDIS_QUEUE_KEY, json.dumps(job))


import platform as _platform

# Windows: Docker runs as a GUI app (Docker Desktop)
_DOCKER_DESKTOP_EXE = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Daemon-not-running error phrases (same on Windows and Linux)
_DAEMON_NOT_RUNNING_PHRASES = (
    "error during connect",
    "is the docker daemon running",
    "cannot connect to the docker daemon",
    "open //./pipe/docker_engine",   # Windows named pipe
    "/var/run/docker.sock",          # Linux socket missing
)


def _is_docker_daemon_running() -> bool:
    """Return True if the Docker daemon is reachable (docker info succeeds)."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _start_docker_daemon() -> bool:
    """
    Start the Docker daemon in a platform-appropriate way and wait up to 60 s.

    - Windows : launch Docker Desktop.exe (GUI app)
    - Linux   : `sudo systemctl start docker`  (system service, no GUI)

    Returns True if the daemon becomes reachable within the timeout.
    """
    os_name = _platform.system()

    if os_name == "Windows":
        import os as _os
        if not _os.path.exists(_DOCKER_DESKTOP_EXE):
            log.warning(
                f"Docker Desktop not found at {_DOCKER_DESKTOP_EXE!r}. Cannot auto-start."
            )
            return False
        log.warning("Docker daemon not running. Launching Docker Desktop…")
        try:
            subprocess.Popen(
                [_DOCKER_DESKTOP_EXE],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            log.warning(f"Could not launch Docker Desktop: {exc}")
            return False

    elif os_name == "Linux":
        log.warning("Docker daemon not running. Attempting: sudo systemctl start docker…")
        try:
            result = subprocess.run(
                ["sudo", "systemctl", "start", "docker"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode != 0:
                log.warning(f"systemctl start docker failed: {result.stderr.strip()}")
                # Fallback: try service command (older distros / non-systemd)
                log.warning("Falling back to: sudo service docker start…")
                subprocess.run(
                    ["sudo", "service", "docker", "start"],
                    capture_output=True, text=True, timeout=15,
                )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            log.warning(f"Could not start Docker service: {exc}")
            return False

    else:
        log.warning(f"Unsupported OS '{os_name}' — cannot auto-start Docker.")
        return False

    # Poll until the daemon responds (Docker Desktop on Windows can take 30–60 s)
    log.info("Waiting for Docker daemon to be ready (up to 60 s)…")
    for tick in range(30):
        time.sleep(2)
        if _is_docker_daemon_running():
            log.info(f"Docker daemon ready after ~{(tick + 1) * 2} s")
            return True
        log.info(f"Docker not ready yet… ({(tick + 1) * 2}/60 s)")

    log.warning("Docker daemon did not become ready in time.")
    return False


def _ensure_redis(retries: int = 5, delay: float = 2.0) -> redis.Redis:
    """
    Ensure Redis is reachable. Works on Windows and Linux.

    Recovery order:
      1. Ping Redis — if reachable, return immediately.
      2. Try `docker start my-redis` (covers: daemon running, container stopped).
      3. If Docker daemon itself is not running:
           Windows → launch Docker Desktop.exe and wait
           Linux   → sudo systemctl start docker (or service docker start)
         Then retry `docker start my-redis`.
      4. Retry the Redis ping up to `retries` times.
    """
    # ── Step 1: quick probe ───────────────────────────────────────────────────
    try:
        r = redis.from_url(REDIS_URL, socket_connect_timeout=2)
        r.ping()
        log.info("Connected to Redis")
        return r
    except redis.RedisError:
        pass  # not running — try to start it

    # ── Step 2: attempt docker start (fast path — daemon already running) ─────
    log.warning("Redis not reachable. Attempting to start Docker container 'my-redis'…")
    container_started = False
    try:
        result = subprocess.run(
            ["docker", "start", "my-redis"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            log.info("'my-redis' container started.")
            container_started = True
        else:
            err = result.stderr.strip()
            log.warning(f"docker start failed: {err}")
            daemon_not_running = any(p in err.lower() for p in _DAEMON_NOT_RUNNING_PHRASES)
            if daemon_not_running:
                # ── Step 3: start Docker daemon (platform-aware), retry ───────
                if _start_docker_daemon():
                    retry = subprocess.run(
                        ["docker", "start", "my-redis"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if retry.returncode == 0:
                        log.info("'my-redis' container started after daemon launch.")
                        container_started = True
                    else:
                        log.warning(f"docker start still failed: {retry.stderr.strip()}")
    except FileNotFoundError:
        log.warning("'docker' CLI not found — cannot auto-start Redis.")
    except subprocess.TimeoutExpired:
        log.warning("docker start timed out.")

    # ── Step 4: retry Redis ping ──────────────────────────────────────────────
    for attempt in range(1, retries + 1):
        time.sleep(delay)
        try:
            r = redis.from_url(REDIS_URL, socket_connect_timeout=2)
            r.ping()
            log.info(f"Connected to Redis (attempt {attempt}/{retries})")
            return r
        except redis.RedisError as exc:
            log.info(f"Redis not ready yet ({exc}). Retrying… ({attempt}/{retries})")

    log.error("Could not connect to Redis after all retries. Exiting.")
    sys.exit(1)


# ── Main camera loop ──────────────────────────────────────────────────────────

def run_camera(source, show_display: bool = True, camera_id: str = CAMERA_ID,
               direction: str = CAMERA_DIRECTION):
    """
    Open video source, detect/track faces, push to Redis queue.

    source: int (webcam index), str (file path or RTSP URL)
    """

    # Connect Redis (auto-starts Docker container if needed)
    r = _ensure_redis()

    # Load models
    log.info("Loading YOLO model…")
    model = YOLO("yolov8n.pt")
    face_cascade, eye_cascade = load_cascades()
    log.info("Models loaded")

    # Open source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open source: {source}")
        sys.exit(1)

    is_webcam = isinstance(source, int)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect = orig_w / orig_h
    det_w = DETECTION_WIDTH
    det_h = int(det_w / aspect)
    scale_x = orig_w / det_w
    scale_y = orig_h / det_h

    log.info(f"Source={source}  orig={orig_w}x{orig_h}  det={det_w}x{det_h}")

    win = "FaceLog - press Q to quit"
    if show_display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1280, 720)
        cv2.waitKey(1)  # flush window creation event on Windows

    tracker = FaceTracker()
    frame_no = 0
    queued_total = 0   # session counter shown on display
    last_boxes: list = []
    last_tracked: list = []
    fps_timer = time.time()
    fps_val = 0.0
    fps_counter = 0
    track_colors: dict = {}      # face_id → (B, G, R) random colour
    centroid_history: dict = {}  # face_id → list of (cx_d, cy_d) in display coords

    log.info("Camera loop started")

    while app_state.running():
        ret, frame = cap.read()
        if not ret:
            if is_webcam:
                log.warning("Webcam read failed")
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_no = 0
            continue

        frame_no += 1
        fps_counter += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps_val = fps_counter / elapsed
            fps_counter = 0
            fps_timer = time.time()

        small = cv2.resize(frame, (det_w, det_h))
        run_det = (frame_no % PROCESS_EVERY_N_FRAMES == 0) or frame_no == 1

        if run_det:
            yolo_results = model(small, verbose=False, classes=[0], conf=0.5)
            frame_detections = []
            person_boxes = []

            for box in yolo_results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                x1o = int(x1 * scale_x)
                y1o = int(y1 * scale_y)
                x2o = int(x2 * scale_x)
                y2o = int(y2 * scale_y)
                person_boxes.append((int(x1), int(y1), int(x2), int(y2)))

                roi = frame[y1o:y2o, x1o:x2o]
                faces = detect_faces_in_roi(
                    roi, face_cascade, eye_cascade,
                    person_box_orig=(x1o, y1o, x2o, y2o),
                )
                for f in faces:
                    f["conf"] = conf  # inject YOLO confidence
                frame_detections.extend(faces)

            last_boxes = person_boxes
            last_tracked = tracker.update(frame_detections, frame_no)

            # ── Assign colours + record centroid history ──────────────────
            active_ids = {fid for fid, _ in last_tracked}
            for face_id, det in last_tracked:
                if face_id not in track_colors:
                    track_colors[face_id] = (
                        random.randint(50, 255),
                        random.randint(50, 255),
                        random.randint(50, 255),
                    )
                cx_o, cy_o = det["center"]
                cx_d = int(cx_o / scale_x)
                cy_d = int(cy_o / scale_y)
                centroid_history.setdefault(face_id, []).append((cx_d, cy_d))
                if len(centroid_history[face_id]) > 60:
                    centroid_history[face_id] = centroid_history[face_id][-60:]
            # Prune history for expired tracks
            for fid in list(centroid_history):
                if fid not in active_ids:
                    del centroid_history[fid]

            # ── Queue eligible faces ──────────────────────────────────────
            for face_id, det in last_tracked:
                if not det["is_frontal"]:
                    continue
                if not tracker.ready_to_queue(face_id):
                    continue
                face_img = det["face_image"]
                if face_img is None or face_img.size == 0:
                    continue

                face_jpeg = encode_jpeg(face_img)
                thumb_jpeg = encode_thumbnail(frame)
                detected_at = datetime.now(tz=timezone.utc)
                raw_meta = {
                    "tracker_id": face_id,
                    "n_eyes": det["n_eyes"],
                    "face_ratio": round(det["face_ratio"], 3),
                    "yolo_conf": round(det["conf"], 3),
                    "bbox_orig": list(det["bbox_orig"]),
                    "person_bbox_orig": list(det["person_bbox_orig"]),
                }

                try:
                    sighting_id = db.insert_sighting(
                        face_image=face_jpeg,
                        detected_at=detected_at,
                        camera_id=camera_id,
                        raw_meta=raw_meta,
                        full_frame_image=thumb_jpeg,
                        direction=direction,
                    )
                    db.record_queued(sighting_id)
                    _push_to_queue(r, sighting_id, face_jpeg)
                    tracker.mark_queued(face_id)
                    queued_total += 1
                    log.info(f"Queued face: tracker={face_id} sighting={sighting_id}")
                except Exception as exc:
                    log.error(f"Failed to queue face: {exc}", exc_info=True)

        # ── Draw UI ───────────────────────────────────────────────────────
        if show_display:
            display = small.copy()

            for px1, py1, px2, py2 in last_boxes:
                cv2.rectangle(display, (px1, py1), (px2, py2), (100, 100, 100), 1)

            for face_id, det in last_tracked:
                fx_o, fy_o, fw, fh = det["bbox_orig"]
                fx_d = int(fx_o / scale_x)
                fy_d = int(fy_o / scale_y)
                fw_d = int(fw / scale_x)
                fh_d = int(fh / scale_y)

                count = tracker.get_detect_count(face_id)
                color = track_colors.get(face_id, (0, 255, 0))
                label = f"#{face_id} ({count})" if det["is_frontal"] else f"#{face_id} angled"

                # Draw centroid trail
                history = centroid_history.get(face_id, [])
                for i in range(1, len(history)):
                    cv2.line(display, history[i - 1], history[i], color, 2)
                if history:
                    cv2.circle(display, history[-1], 5, color, -1)

                cv2.rectangle(display, (fx_d, fy_d), (fx_d + fw_d, fy_d + fh_d), color, 2)
                cv2.putText(display, label, (fx_d, fy_d - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            cv2.putText(display, f"FPS: {fps_val:.1f}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Queued: {queued_total}", (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

            cv2.imshow(win, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("Q pressed — quitting")
                break

    cap.release()
    if show_display:
        cv2.destroyAllWindows()
    log.info("Camera loop stopped")
