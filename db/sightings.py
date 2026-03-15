"""db/sightings.py — sightings table queries."""

import json
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
    direction: str = "unknown",
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


def get_recent_sightings(limit: int = 100) -> list:
    """Fetch recent sightings joined with person name."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            """
            SELECT s.id, s.detected_at, s.direction, s.camera_id,
                   s.status, s.confidence,
                   p.name, p.label
              FROM sightings s
              LEFT JOIN persons p ON p.id = s.person_id
             ORDER BY s.detected_at DESC
             LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()
