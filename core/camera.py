"""
core/camera.py — Main camera capture loop.

Opens the video source, runs YOLO person detection every N frames,
detects faces within person ROIs, tracks them, and pushes stable
frontal faces to Redis for background recognition.
"""

import json
import logging
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


def _ensure_redis(retries: int = 5, delay: float = 2.0) -> redis.Redis:
    """
    Ensure Redis is reachable. If not, start the 'my-redis' Docker container
    first, then retry the connection up to `retries` times.
    """
    # ── Step 1: quick probe ───────────────────────────────────────────────────
    try:
        r = redis.from_url(REDIS_URL, socket_connect_timeout=2)
        r.ping()
        log.info("Connected to Redis")
        return r
    except redis.RedisError:
        pass  # not running — try to start it

    # ── Step 2: start Docker container ───────────────────────────────────────
    log.warning("Redis not reachable. Attempting to start Docker container 'my-redis'…")
    try:
        result = subprocess.run(
            ["docker", "start", "my-redis"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            log.info("'my-redis' container started. Waiting for Redis to be ready…")
        else:
            log.warning(f"docker start failed: {result.stderr.strip()}")
    except FileNotFoundError:
        log.warning("Docker not found — cannot auto-start Redis.")
    except subprocess.TimeoutExpired:
        log.warning("docker start timed out.")

    # ── Step 3: retry until Redis responds ───────────────────────────────────
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

def run_camera(source, show_display: bool = True, camera_id: str = CAMERA_ID):
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
                        direction="unknown",
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
                if det["is_frontal"]:
                    color, label = (0, 255, 0), f"#{face_id} ({count})"
                else:
                    color, label = (0, 165, 255), f"#{face_id} angled"

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
