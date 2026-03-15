"""db/queue_audit.py — queue_audit table queries."""

from .connection import get_conn


def record_queued(sighting_id: int):
    """Record that a sighting was pushed onto the Redis queue."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO queue_audit (sighting_id) VALUES (%s)",
            (sighting_id,),
        )


def mark_queue_processed(sighting_id: int, worker_pid: int):
    """Mark the queue_audit row processed after a worker finishes it."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE queue_audit
               SET processed = TRUE, worker_pid = %s
             WHERE sighting_id = %s AND processed = FALSE
            """,
            (worker_pid, sighting_id),
        )
