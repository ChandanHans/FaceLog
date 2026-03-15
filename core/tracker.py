"""
core/tracker.py — Centroid-based face tracker.

Assigns stable IDs to detected faces across frames.
Prevents duplicate queuing via per-face cooldown timer.
"""

import time

from config import (
    FACE_QUEUE_COOLDOWN,
    FACE_TRACKER_DISTANCE,
    FACE_TRACKER_TIMEOUT,
    MIN_CONSISTENT_DETECTIONS,
)


class FaceTracker:
    def __init__(self):
        self._next_id = 1
        self._tracks: dict = {}
        self._last_queued: dict = {}  # face_id → epoch-seconds of last queue push

    # ── public ────────────────────────────────────────────────────────────────

    def update(self, detections: list, frame_no: int) -> list:
        """
        Match detections to existing tracks by centroid distance.

        detections: list of dicts — must contain keys:
            center, bbox, bbox_orig, person_bbox_orig,
            is_frontal, n_eyes, face_ratio, conf, face_image

        Returns: list of (face_id, detection_dict)
        """
        results = []
        matched_ids: set = set()
        unmatched = []

        for det in detections:
            cx, cy = det["center"]
            best_id, best_dist = None, float("inf")

            for fid, track in self._tracks.items():
                if fid in matched_ids:
                    continue
                dx = cx - track["center"][0]
                dy = cy - track["center"][1]
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < best_dist and dist < FACE_TRACKER_DISTANCE:
                    best_dist, best_id = dist, fid

            if best_id is not None:
                matched_ids.add(best_id)
                t = self._tracks[best_id]
                t["center"] = det["center"]
                t["bbox"] = det["bbox"]
                t["last_seen"] = frame_no
                t["detect_count"] += 1
                area = det["bbox"][2] * det["bbox"][3]
                if area > t.get("best_area", 0):
                    t["best_area"] = area
                    t["best_face_image"] = det.get("face_image")
                results.append((best_id, det))
            else:
                unmatched.append(det)

        for det in unmatched:
            fid = self._next_id
            self._next_id += 1
            area = det["bbox"][2] * det["bbox"][3]
            self._tracks[fid] = {
                "center": det["center"],
                "bbox": det["bbox"],
                "last_seen": frame_no,
                "detect_count": 1,
                "best_area": area,
                "best_face_image": det.get("face_image"),
            }
            results.append((fid, det))

        # Expire stale tracks
        stale = [
            fid for fid, t in self._tracks.items()
            if frame_no - t["last_seen"] > FACE_TRACKER_TIMEOUT
        ]
        for fid in stale:
            del self._tracks[fid]
            self._last_queued.pop(fid, None)

        return results

    def ready_to_queue(self, face_id: int) -> bool:
        """True if stable enough + cooldown has passed since last queue push."""
        t = self._tracks.get(face_id)
        if t is None or t["detect_count"] < MIN_CONSISTENT_DETECTIONS:
            return False
        return (time.time() - self._last_queued.get(face_id, 0)) >= FACE_QUEUE_COOLDOWN

    def mark_queued(self, face_id: int):
        self._last_queued[face_id] = time.time()

    def get_detect_count(self, face_id: int) -> int:
        return self._tracks.get(face_id, {}).get("detect_count", 0)
