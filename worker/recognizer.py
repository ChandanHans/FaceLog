"""
worker/recognizer.py — Face recognition logic.

Compares a face JPEG against known persons loaded from the DB.
"""

import cv2
import numpy as np
import face_recognition

from typing import Optional


def decode_jpeg(jpeg_bytes: bytes) -> Optional[np.ndarray]:
    """JPEG bytes → BGR numpy array."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def recognize(
    face_jpeg: bytes,
    known_persons: list,
    tolerance: float,
) -> tuple:
    """
    Compare face_jpeg against known_persons.

    known_persons: list of {person_id, name, encoding}
    Returns: (person_id, name, confidence) — or (None, 'unknown', None) if no match.
    """
    if not known_persons:
        return None, "unknown", None

    img_bgr = decode_jpeg(face_jpeg)
    if img_bgr is None:
        return None, "unknown", None

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")
    if not locs:
        return None, "unknown", None

    encs = face_recognition.face_encodings(rgb, locs)
    if not encs:
        return None, "unknown", None

    query_enc = encs[0]
    known_encs = [p["encoding"] for p in known_persons]
    distances = face_recognition.face_distance(known_encs, query_enc)

    best_idx = int(np.argmin(distances))
    best_dist = float(distances[best_idx])

    if best_dist <= tolerance:
        p = known_persons[best_idx]
        return p["person_id"], p["name"], round(1.0 - best_dist, 4)

    return None, "unknown", None
