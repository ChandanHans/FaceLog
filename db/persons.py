"""db/persons.py — persons table queries."""

import pickle
from typing import Optional

import psycopg2
import psycopg2.extras

from .connection import get_conn


def insert_person(
    name: Optional[str] = None,
    label: str = "unknown",
    face_encoding=None,
    reference_image: Optional[bytes] = None,
    notes: Optional[str] = None,
) -> int:
    """Insert a person record, return new id."""
    enc_bytes = pickle.dumps(face_encoding) if face_encoding is not None else None
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO persons (name, label, face_encoding, reference_image, notes)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (name, label, enc_bytes, reference_image, notes),
        )
        return cur.fetchone()[0]


def update_person_label(
    person_id: int,
    label: str,
    name: Optional[str] = None,
    notes: Optional[str] = None,
):
    """Admin: re-label an existing person (known / suspicious)."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE persons
               SET label = %s,
                   name  = COALESCE(%s, name),
                   notes = COALESCE(%s, notes)
             WHERE id = %s
            """,
            (label, name, notes, person_id),
        )


def load_known_encodings() -> list:
    """
    Load all known persons' face encodings from the DB.
    Returns list of {person_id, name, encoding}.
    """
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            """
            SELECT id, name, face_encoding
              FROM persons
             WHERE label = 'known' AND face_encoding IS NOT NULL
            """
        )
        rows = cur.fetchall()

    return [
        {
            "person_id": row["id"],
            "name": row["name"],
            "encoding": pickle.loads(bytes(row["face_encoding"])),
        }
        for row in rows
    ]


def get_all_persons() -> list:
    """Return all persons ordered by most recently added."""
    with get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute(
            """
            SELECT id, name, label, added_at, notes, reference_image
              FROM persons
             ORDER BY added_at DESC
            """
        )
        return cur.fetchall()


def get_person_names() -> list[tuple[int, str]]:
    """Return [(id, name)] for all persons that have a name, for dropdowns."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, name FROM persons
             WHERE name IS NOT NULL
             ORDER BY name ASC
            """
        )
        return cur.fetchall()


def merge_person_encoding(person_id: int, new_encoding) -> None:
    """
    Average `new_encoding` with the person's existing stored encoding.
    If no encoding exists yet, just stores the new one.
    Also updates reference_image if provided via update_person_label.
    """
    import numpy as np
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT face_encoding FROM persons WHERE id = %s",
            (person_id,),
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            existing = pickle.loads(bytes(row[0]))
            merged = np.mean([existing, new_encoding], axis=0)
        else:
            merged = new_encoding
        cur.execute(
            "UPDATE persons SET face_encoding = %s WHERE id = %s",
            (psycopg2.Binary(pickle.dumps(merged)), person_id),
        )
        conn.commit()
