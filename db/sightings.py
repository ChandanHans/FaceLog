"""db/sightings.py — sightings table queries."""

import json
import pickle
from datetime import datetime
from typing import Optional

import psycopg2
import psycopg2.extras

from .connection import get_conn


def insert_sighting(
    face_image: bytes,
    detected_at: datetime,
    camera_id: str,
    raw_meta: dict,
    full_frame_image: Optional[bytes] = None,
    direction: str = "entry",
) -> int:
    """Insert a new pending sighting. Returns new sighting id."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO sightings
                (detected_at, direction, camera_id, face_image, full_frame_image, status, raw_meta)
            VALUES (%s, %s, %s, %s, %s, 'pending', %s)
            RETURNING id
            """,
            (
                detected_at,
                direction,
                camera_id,
                psycopg2.Binary(face_image),
                psycopg2.Binary(full_frame_image) if full_frame_image else None,
                json.dumps(raw_meta),
            ),
        )
        return cur.fetchone()[0]


def resolve_sighting(
    sighting_id: int,
    status: str,
    person_id: Optional[int] = None,
    confidence: Optional[float] = None,
):
    """Update sighting with match result: 'matched' | 'unmatched' | 'flagged'."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE sightings
               SET status      = %s,
                   person_id   = %s,
                   confidence  = %s,
                   resolved_at = NOW()
             WHERE id = %s
            """,
            (status, person_id, confidence, sighting_id),
        )


def get_unresolved_sightings(limit: int = 50) -> list:
    """Fetch unmatched sightings for admin review."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            """
            SELECT s.id, s.detected_at, s.camera_id, s.direction,
                   s.confidence, s.status, s.raw_meta, s.face_image
              FROM sightings s
             WHERE s.status = 'unmatched'
             ORDER BY s.detected_at DESC
             LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()


def save_sighting_encoding(sighting_id: int, encoding) -> None:
    """Persist a 128-D face encoding for a sighting (avoids re-decoding later)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE sightings SET face_encoding = %s WHERE id = %s",
            (psycopg2.Binary(pickle.dumps(encoding)), sighting_id),
        )


def get_all_unmatched_face_images() -> list:
    """Return all unmatched sightings as (id, face_image, face_encoding) for bulk re-matching."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            """
            SELECT id, face_image, face_encoding
              FROM sightings
             WHERE status = 'unmatched'
             ORDER BY detected_at DESC
            """
        )
        return cur.fetchall()


def delete_sightings(ids: list) -> int:
    """Permanently delete sightings by ID list. Returns count deleted."""
    if not ids:
        return 0
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM sightings WHERE id = ANY(%s)", (list(ids),))
        return cur.rowcount


def get_sighting_date_range() -> tuple:
    """Return (oldest_date, newest_date) from sightings, or (None, None) if table is empty."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT MIN(detected_at)::date, MAX(detected_at)::date FROM sightings")
        row = cur.fetchone()
        return (row[0], row[1]) if row else (None, None)


def get_recent_sightings(
    limit: int = 100,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
    direction: Optional[str] = None,
) -> list:
    """Fetch recent sightings joined with person name, including face and full-frame images."""
    filters = []
    params: list = []
    if date_from:
        filters.append("s.detected_at >= %s")
        params.append(date_from)
    if date_to:
        filters.append("s.detected_at < %s")
        params.append(date_to)
    if direction:
        filters.append("s.direction = %s")
        params.append(direction)
    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    params.append(limit)
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            f"""
            SELECT s.id, s.detected_at, s.direction, s.camera_id,
                   s.status, s.confidence,
                   s.face_image, s.full_frame_image,
                   p.name, p.label
              FROM sightings s
              LEFT JOIN persons p ON p.id = s.person_id
             {where}
             ORDER BY s.detected_at DESC
             LIMIT %s
            """,
            params,
        )
        return cur.fetchall()
