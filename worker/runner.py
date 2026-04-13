"""
worker/runner.py — Background recognition worker process.

Each worker:
  1. BLPOPs jobs from Redis
  2. Runs face recognition against known persons in PostgreSQL
  3. Writes match/unmatch result back to the sightings table

Used by main.py via start_workers() / stop_workers().
Can also be started standalone:  python -m worker.runner
"""

import json
import logging
import os
import signal
import sys

import redis

import db
from config import (
    LOG_FILE,
    RECOGNITION_TOLERANCE,
    REDIS_QUEUE_KEY,
    REDIS_URL,
)
from worker.recognizer import recognize

log = logging.getLogger(__name__)

_running = True


def _handle_signal(signum, frame):
    global _running
    log.info("Worker received shutdown signal")
    _running = False


def run_worker():
    """Main loop: pull jobs from Redis and resolve sightings."""
    # Register signals here, inside the worker process/thread (main thread only)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    pid = os.getpid()
    log.info(f"Worker started (pid={pid})")

    r = redis.from_url(REDIS_URL, decode_responses=False)
    known_persons = db.load_known_encodings()
    log.info(f"Loaded {len(known_persons)} known persons")

    reload_every = 50
    jobs_processed = 0

    while _running:
        item = r.blpop(REDIS_QUEUE_KEY, timeout=2)
        if item is None:
            continue

        _, raw = item
        try:
            job = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            log.warning(f"Could not decode job: {exc}")
            continue

        sighting_id = job.get("sighting_id")
        face_jpeg_hex = job.get("face_image_hex")
        if not sighting_id or not face_jpeg_hex:
            log.warning(f"Malformed job keys={list(job.keys())}")
            continue

        face_jpeg = bytes.fromhex(face_jpeg_hex)

        try:
            person_id, name, confidence, query_enc = recognize(
                face_jpeg, known_persons, RECOGNITION_TOLERANCE
            )

            if person_id is not None:
                status = "matched"
                log.info(f"Sighting {sighting_id}: matched → {name} (conf={confidence:.3f})")
            else:
                status = "unmatched"
                person_id = db.insert_person(label="unknown", reference_image=face_jpeg)
                log.info(f"Sighting {sighting_id}: unmatched → new unknown person id={person_id}")

            db.resolve_sighting(sighting_id, status, person_id=person_id, confidence=confidence)
            # Cache the 128-D encoding so future re-matching is fast (no image decode needed)
            if query_enc is not None:
                db.save_sighting_encoding(sighting_id, query_enc)
            db.mark_queue_processed(sighting_id, worker_pid=pid)

        except Exception as exc:
            log.error(f"Error on sighting {sighting_id}: {exc}", exc_info=True)

        jobs_processed += 1
        if jobs_processed % reload_every == 0:
            known_persons = db.load_known_encodings()
            log.info(f"Reloaded known persons ({len(known_persons)} entries)")

    log.info(f"Worker {pid} exiting cleanly")


# ── Process management (called from main.py) ─────────────────────────────────

def _worker_process_target():
    """Entry point for each subprocess — sets up logging then runs."""
    import multiprocessing
    os.makedirs(
        os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else ".", exist_ok=True
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [worker-%(process)d] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )
    run_worker()


def start_workers(n: int) -> list:
    """Spawn n worker subprocesses. Returns list of Process objects."""
    import multiprocessing
    procs = []
    for _ in range(n):
        p = multiprocessing.Process(target=_worker_process_target, daemon=True)
        p.start()
        log.info(f"Started worker pid={p.pid}")
        procs.append(p)
    return procs


def stop_workers(procs: list, timeout: float = 5.0):
    """Terminate workers gracefully, then kill if still alive."""
    for p in procs:
        p.terminate()
    for p in procs:
        p.join(timeout=timeout)
        if p.is_alive():
            p.kill()
    log.info("All workers stopped")


if __name__ == "__main__":
    os.makedirs(
        os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else ".", exist_ok=True
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [worker-%(process)d] %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_worker()
