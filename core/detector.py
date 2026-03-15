"""
core/detector.py — Face detection helpers.

Provides:
  - encode_jpeg / encode_thumbnail   : image → bytes
  - detect_faces_in_roi              : run Haar cascade on a person ROI,
                                       return list of face detection dicts
"""

import cv2
import numpy as np

from config import (
    FACE_ASPECT_RATIO_MAX,
    FACE_ASPECT_RATIO_MIN,
    JPEG_QUALITY,
    MAX_FACE_ROI_HEIGHT,
    MIN_EYES_FOR_FRONTAL,
    MIN_FACE_QUEUE_HEIGHT,
    MIN_FACE_QUEUE_WIDTH,
    MIN_FACE_WIDTH,
)


# ── Image encoding ─────────────────────────────────────────────────────────────

def encode_jpeg(img: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """BGR numpy array → JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf) if ok else b""


def encode_thumbnail(img: np.ndarray, max_side: int = 480) -> bytes:
    """Downscale to thumbnail and encode as JPEG."""
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return encode_jpeg(img, quality=60)


# ── Face detection ─────────────────────────────────────────────────────────────

def load_cascades():
    """Load OpenCV Haar cascades. Call once at startup."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )
    return face_cascade, eye_cascade


def detect_faces_in_roi(
    roi: np.ndarray,
    face_cascade,
    eye_cascade,
    person_box_orig: tuple,
) -> list:
    """
    Run Haar cascade on a single person ROI (original-resolution crop).

    person_box_orig: (x1, y1, x2, y2) in original frame coords.

    Returns list of face-detection dicts ready for FaceTracker.update().
    Each dict has:
        center, bbox, bbox_orig, person_bbox_orig,
        is_frontal, n_eyes, face_ratio, conf (0 — passed through),
        face_image
    """
    if roi.size == 0:
        return []

    x1o, y1o, x2o, y2o = person_box_orig
    roi_h, roi_w = roi.shape[:2]

    # Cap ROI height for Haar cascade speed
    if roi_h > MAX_FACE_ROI_HEIGHT:
        rs = MAX_FACE_ROI_HEIGHT / roi_h
        roi_r = cv2.resize(roi, (int(roi_w * rs), MAX_FACE_ROI_HEIGHT))
    else:
        roi_r, rs = roi, 1.0

    gray = cv2.cvtColor(roi_r, cv2.COLOR_BGR2GRAY)
    faces_raw = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )

    detections = []
    for rfx, rfy, rfw, rfh in faces_raw if len(faces_raw) else []:
        fx = int(rfx / rs)
        fy = int(rfy / rs)
        fw = int(rfw / rs)
        fh = int(rfh / rs)

        # Eye detection on original face crop
        face_crop = roi[fy: fy + fh, fx: fx + fw]
        if face_crop.size == 0:
            continue
        crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(crop_gray, scaleFactor=1.1, minNeighbors=3)
        n_eyes = len(eyes)

        ratio = fw / fh if fh else 0
        too_small = fw < MIN_FACE_QUEUE_WIDTH or fh < MIN_FACE_QUEUE_HEIGHT

        is_frontal = False
        if n_eyes >= MIN_EYES_FOR_FRONTAL:
            is_frontal = not too_small
        elif FACE_ASPECT_RATIO_MIN <= ratio <= FACE_ASPECT_RATIO_MAX and fw >= MIN_FACE_WIDTH:
            is_frontal = not too_small

        face_cx = x1o + fx + fw // 2
        face_cy = y1o + fy + fh // 2

        detections.append(
            {
                "center": (face_cx, face_cy),
                "bbox": (fx, fy, fw, fh),
                "bbox_orig": (x1o + fx, y1o + fy, fw, fh),
                "person_bbox_orig": (x1o, y1o, x2o, y2o),
                "is_frontal": is_frontal,
                "n_eyes": n_eyes,
                "face_ratio": ratio,
                "conf": 0.0,          # filled in by camera.py with YOLO conf
                "face_image": face_crop.copy(),
            }
        )

    return detections
