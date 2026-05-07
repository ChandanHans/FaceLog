"""
config.py — Central configuration.

All settings come from the .env file in the project root.
Copy .env.example → .env and fill in your values.

Never commit .env to version control.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Project root = the directory that contains this file.
_ROOT = Path(__file__).parent

# Load .env  (override=False: won't overwrite variables already set in the OS
# environment, so real env vars always win over the .env file)
load_dotenv(_ROOT / ".env", override=False)


# ─── Database ─────────────────────────────────────────────────────────────────
DB_URL: str = os.getenv("DB_URL")

# ─── Redis ────────────────────────────────────────────────────────────────────
REDIS_URL: str = os.getenv("REDIS_URL")
REDIS_QUEUE_KEY: str = "facelog:face_queue"

# ─── Camera ───────────────────────────────────────────────────────────────────
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")  # default: first webcam
# Convert to int if it looks like a device index
if CAMERA_SOURCE.isdigit():
    CAMERA_SOURCE = int(CAMERA_SOURCE)

CAMERA_ID: str = os.getenv("CAMERA_ID", "cam_entry")
CAMERA_DIRECTION: str = os.getenv("CAMERA_DIRECTION", "entry")  # 'entry' or 'exit'

# Second camera (optional — leave CAMERA_SOURCE2 empty for single-camera mode)
CAMERA_SOURCE2 = os.getenv("CAMERA_SOURCE2", "")   # e.g. "1" or rtsp://...
if CAMERA_SOURCE2.isdigit():
    CAMERA_SOURCE2 = int(CAMERA_SOURCE2)
CAMERA_ID2: str = os.getenv("CAMERA_ID2", "cam_exit")
CAMERA_DIRECTION2: str = os.getenv("CAMERA_DIRECTION2", "exit")

# ─── Detection ────────────────────────────────────────────────────────────────
DETECTION_WIDTH: int = 1000
DETECTION_HEIGHT: int = 800
PROCESS_EVERY_N_FRAMES: int = 2
MAX_FACE_ROI_HEIGHT: int = 400

# Face angle / quality thresholds
MIN_EYES_FOR_FRONTAL: int = 1       # 1 = lenient (CCTV-friendly)
FACE_ASPECT_RATIO_MIN: float = 0.7
FACE_ASPECT_RATIO_MAX: float = 1.3
MIN_FACE_WIDTH: int = 50
MIN_FACE_QUEUE_WIDTH: int = 50
MIN_FACE_QUEUE_HEIGHT: int = 50

# Tracker
MIN_CONSISTENT_DETECTIONS: int = 5
FACE_TRACKER_DISTANCE: int = 80     # pixels (original resolution)
FACE_TRACKER_TIMEOUT: int = 15      # frames before track is dropped

# ─── Recognition ──────────────────────────────────────────────────────────────
RECOGNITION_TOLERANCE: float = float(os.getenv("RECOGNITION_TOLERANCE", "0.48"))  # see .env
# Seconds before the same person can be re-queued (avoids spam)
FACE_QUEUE_COOLDOWN: int = int(os.getenv("FACE_QUEUE_COOLDOWN", "60"))  # see .env

# ─── Workers ──────────────────────────────────────────────────────────────────
WORKER_PROCESSES: int = int(os.getenv("WORKER_PROCESSES", "2"))  # see .env

# ─── Storage ──────────────────────────────────────────────────────────────────
JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", "75"))  # see .env

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE: str = os.getenv("LOG_FILE", "logs/facelog.log")  # see .env
