# FaceLog

FaceLog automates visual identity logging at entry/exit points, replacing manual monitoring with a real-time camera-based system that stores evidence and flags unknowns for review.

---

## Features

- **Real-time detection** — YOLOv8 person detection + OpenCV Haar cascade face detection
- **Background recognition** — Redis job queue (`facelog:face_queue`) + multiprocessing workers run face recognition without blocking the camera loop
- **PostgreSQL logging** — every sighting stored with face image (BYTEA), timestamp, camera ID, direction, and match confidence
- **Crash recovery** — `queue_audit` table mirrors Redis so unprocessed jobs can be re-queued after a restart
- **Web dashboard** — Streamlit UI to review unknowns, label faces, and browse all persons and sightings
- **Webcam capture** — multi-angle face capture CLI to enrol known persons
- **Dual-camera support** — optional second camera for simultaneous entry + exit tracking
- **Auto Redis restart** — automatically starts the `my-redis` Docker container if Redis is down

---

## Architecture

```
Camera loop (main process)
  │  YOLOv8 → Haar cascade → FaceTracker
  │
  ├─► INSERT sightings (PostgreSQL)
  ├─► INSERT queue_audit (PostgreSQL)
  └─► RPUSH facelog:face_queue (Redis)
              │
        Worker pool (2 processes)
              │  BLPOP → face_recognition
              └─► UPDATE sightings (matched / unmatched)
                  UPDATE queue_audit (processed=TRUE)

Admin
  streamlit run dashboard.py    ← review unknowns, browse sightings
  python main.py capture <name> ← enrol new person via webcam
```

---

## Project Structure

```
entry-exit/
├── main.py              # Single CLI entry point
├── dashboard.py         # Streamlit web dashboard
├── config.py            # All settings (loaded from .env)
├── app_state.py         # Graceful shutdown flag
│
├── core/
│   ├── camera.py        # Main camera loop
│   ├── detector.py      # Haar cascade face detection helpers
│   └── tracker.py       # Centroid-based FaceTracker
│
├── worker/
│   ├── recognizer.py    # Face recognition logic
│   └── runner.py        # Redis consumer worker processes
│
├── db/
│   ├── connection.py    # psycopg2 connection context manager
│   ├── setup.py         # DDL — creates tables
│   ├── persons.py       # persons table CRUD
│   ├── sightings.py     # sightings table CRUD
│   └── queue_audit.py   # queue_audit table CRUD
│
├── admin/
│   ├── label.py         # CLI: list / label sightings, import face/ dir
│   └── capture.py       # Webcam multi-angle face capture → DB
│
├── components/
│   └── image_selector/  # Custom Streamlit drag-to-select image grid
│       └── index.html
│
├── face/                # Person image folders (gitignored)
│   └── <Name>/
│       └── image_1.jpg …
├── logs/                # Log files (gitignored)
├── .env                 # Secrets (gitignored — copy from .env.example)
└── .env.example         # Template with comments
```

---

## Requirements

- Python 3.10+
- PostgreSQL (existing instance — not Docker)
- Redis (Docker container `my-redis` or any Redis server)
- Docker Desktop (for auto-starting Redis)

---

## Setup

### 1. Install dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> `dlib` (required by `face-recognition`) needs CMake and a C++ compiler.  
> On Windows: install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or use a pre-built wheel.

### 2. Configure environment

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```env
DB_URL=postgresql://postgres:yourpassword@localhost:5432/facelog
REDIS_URL=redis://localhost:6379/0

# Primary camera (entry)
CAMERA_SOURCE=0
CAMERA_ID=cam_entry
CAMERA_DIRECTION=entry

# Second camera (optional — leave blank for single-camera mode)
# CAMERA_SOURCE2=1
# CAMERA_ID2=cam_exit
# CAMERA_DIRECTION2=exit

WORKER_PROCESSES=2
RECOGNITION_TOLERANCE=0.48
FACE_QUEUE_COOLDOWN=60
JPEG_QUALITY=75
```

### 3. Create database tables

```powershell
python main.py setup-db
```

### 4. Enrol known persons (optional)

**From webcam** (recommended — captures 5 angles automatically):
```powershell
python main.py capture Alice
python main.py capture Alice --photos 7
```

**From existing face/ folder images:**
```powershell
python main.py admin add-known
```

---

## Running

### Interactive menu (no arguments)

```powershell
python main.py
```

```
╔══════════════════════════════════════════════╗
║              FaceLog                         ║
╠══════════════════════════════════════════════╣
║  1. Start camera (run)                       ║
║  2. Setup database                           ║
║  3. Capture face from webcam                 ║
║  4. Admin — list unmatched sightings         ║
║  5. Admin — label a sighting                 ║
║  6. Admin — import face/ directory into DB   ║
║  7. Open web dashboard (Streamlit)           ║
║  q. Quit                                     ║
╚══════════════════════════════════════════════╝
```

### CLI commands

```powershell
# Start camera + recognition workers
python main.py run
python main.py run --source 0              # webcam
python main.py run --source video.mp4      # video file
python main.py run --source rtsp://...     # RTSP stream
python main.py run --no-display            # headless (server)
python main.py run --workers 4
python main.py run --direction entry       # mark sightings as 'entry'

# Two-camera setup (entry + exit in one command)
python main.py run --source 0 --camera-id cam_entry --direction entry `
               --source2 1 --camera-id2 cam_exit --direction2 exit

# Database
python main.py setup-db

# Webcam enrolment
python main.py capture Alice
python main.py capture Alice --photos 7
python main.py capture Alice --photos 7 --min-angle 2

# Admin CLI
python main.py admin list
python main.py admin label 42
python main.py admin add-known

# Web dashboard
streamlit run dashboard.py
```

---

## Web Dashboard

```powershell
streamlit run dashboard.py
# opens http://localhost:8501
```

| Page | Description |
|------|-------------|
| 📋 Recent Sightings | Table of all detections with status, person name, confidence, timestamp |
| 🔍 Review Unknowns | Card grid of unmatched faces — select existing person or create new, saves encoding |
| 👥 People Directory | All persons with reference photo, label filter |

When labelling an unknown as **known**:
- Select an existing person from the dropdown → encoding is **merged** (averaged) into their existing record, improving future recognition
- Select "➕ New person…" → creates a new person row with encoding

---

## Database Schema

### `persons`
| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| name | TEXT | NULL until labelled |
| label | TEXT | `known` / `unknown` / `suspicious` |
| face_encoding | BYTEA | Pickled 128-D numpy array |
| reference_image | BYTEA | JPEG face crop |
| added_at | TIMESTAMPTZ | |
| notes | TEXT | Admin notes |

### `sightings`
| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| person_id | INT FK | → persons, NULL until matched |
| detected_at | TIMESTAMPTZ | When camera detected the face |
| resolved_at | TIMESTAMPTZ | When worker finished matching |
| direction | TEXT | `entry` / `exit` / `unknown` |
| camera_id | TEXT | e.g. `cam_entry` |
| face_image | BYTEA | Cropped face JPEG |
| full_frame_image | BYTEA | Thumbnail of full frame |
| confidence | FLOAT | Match score 0–1 |
| status | TEXT | `pending` → `matched` / `unmatched` / `flagged` |
| raw_meta | JSONB | bbox, eyes, YOLO conf, tracker ID, etc. |

### `queue_audit`
| Column | Type | Notes |
|--------|------|-------|
| id | SERIAL PK | |
| sighting_id | INT FK | → sightings |
| queued_at | TIMESTAMPTZ | |
| processed | BOOLEAN | FALSE until worker claims it |
| worker_pid | INT | PID of the worker that processed it |

---

## Configuration Reference

All values can be set in `.env`. Defaults shown below.

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_URL` | — | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CAMERA_SOURCE` | `0` | Primary camera — webcam index, video path, or RTSP URL |
| `CAMERA_ID` | `cam_entry` | Logical name for primary camera stored in DB |
| `CAMERA_DIRECTION` | `entry` | Direction tag for primary camera (`entry` or `exit`) |
| `CAMERA_SOURCE2` | `` | Second camera source (blank = disabled) |
| `CAMERA_ID2` | `cam_exit` | Logical name for second camera stored in DB |
| `CAMERA_DIRECTION2` | `exit` | Direction tag for second camera (`entry` or `exit`) |
| `WORKER_PROCESSES` | `2` | Number of recognition worker processes |
| `RECOGNITION_TOLERANCE` | `0.48` | Face match threshold (lower = stricter) |
| `FACE_QUEUE_COOLDOWN` | `60` | Seconds before same person can be re-queued |
| `JPEG_QUALITY` | `75` | Quality of stored face images |
| `LOG_FILE` | `logs/facelog.log` | Log output path |

### Detection tuning (in `config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `PROCESS_EVERY_N_FRAMES` | `2` | Run YOLO every N frames (1 = every frame) |
| `MIN_CONSISTENT_DETECTIONS` | `3` | Frames face must appear before queuing |
| `FACE_TRACKER_DISTANCE` | `80` | Max pixel distance to link same face across frames |
| `FACE_TRACKER_TIMEOUT` | `15` | Frames before a track is dropped |
| `MIN_EYES_FOR_FRONTAL` | `1` | Eye count threshold for frontal check (1 = CCTV-lenient) |
| `MIN_FACE_WIDTH` | `50` | Minimum face bounding-box width (px) to process |
| `MIN_FACE_QUEUE_WIDTH` | `50` | Minimum face crop width (px) to push to queue |
| `MIN_FACE_QUEUE_HEIGHT` | `50` | Minimum face crop height (px) to push to queue |
| `FACE_ASPECT_RATIO_MIN` | `0.7` | Minimum face width/height ratio (filters non-face blobs) |
| `FACE_ASPECT_RATIO_MAX` | `1.3` | Maximum face width/height ratio |
