"""
db/setup.py — DDL and table creation.
Run once:  python -m db.setup   OR   python main.py --setup-db
"""

import sys
import psycopg2
from config import DB_URL

DDL = """
-- ─────────────────────────────────────────────────────────────────────────────
-- persons: identity registry
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS persons (
    id               SERIAL PRIMARY KEY,
    name             TEXT,                          -- NULL until admin labels
    label            TEXT NOT NULL DEFAULT 'unknown'
                         CHECK (label IN ('known', 'unknown', 'suspicious')),
    face_encoding    BYTEA,                         -- pickled 128-D numpy array
    reference_image  BYTEA,                         -- JPEG bytes of best face photo
    added_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes            TEXT
);

CREATE INDEX IF NOT EXISTS idx_persons_label ON persons(label);

-- ─────────────────────────────────────────────────────────────────────────────
-- sightings: one row per face detection event
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sightings (
    id               SERIAL PRIMARY KEY,
    person_id        INT REFERENCES persons(id) ON DELETE SET NULL,
    detected_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at      TIMESTAMPTZ,
    direction        TEXT DEFAULT 'unknown'
                         CHECK (direction IN ('entry', 'exit', 'unknown')),
    camera_id        TEXT NOT NULL DEFAULT 'cam_01',
    face_image       BYTEA NOT NULL,                -- cropped face JPEG bytes
    face_encoding    BYTEA,                         -- pickled 128-D numpy array (cached)
    full_frame_image BYTEA,                         -- thumbnail of full frame
    confidence       FLOAT,
    status           TEXT NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending', 'matched', 'unmatched', 'flagged')),
    raw_meta         JSONB
);

CREATE INDEX IF NOT EXISTS idx_sightings_detected_at ON sightings(detected_at);
CREATE INDEX IF NOT EXISTS idx_sightings_person_id   ON sightings(person_id);
CREATE INDEX IF NOT EXISTS idx_sightings_status      ON sightings(status);
CREATE INDEX IF NOT EXISTS idx_sightings_camera_id   ON sightings(camera_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- queue_audit: persistent mirror of Redis queue for crash recovery
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS queue_audit (
    id           SERIAL PRIMARY KEY,
    sighting_id  INT NOT NULL REFERENCES sightings(id) ON DELETE CASCADE,
    queued_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed    BOOLEAN NOT NULL DEFAULT FALSE,
    worker_pid   INT
);

CREATE INDEX IF NOT EXISTS idx_queue_audit_processed ON queue_audit(processed);
"""


def create_tables():
    """Create all tables. Safe to run multiple times (IF NOT EXISTS)."""
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(DDL)
        # Migration: add face_encoding column if it doesn't exist yet
        cur.execute(
            """
            ALTER TABLE sightings
              ADD COLUMN IF NOT EXISTS face_encoding BYTEA;
            """
        )
        print("✅ All tables created (or already exist).")
        cur.close()
        conn.close()
    except psycopg2.OperationalError as exc:
        print(f"❌ Cannot connect to PostgreSQL: {exc}")
        print("   Check DB_URL in your .env file.")
        sys.exit(1)


if __name__ == "__main__":
    create_tables()
