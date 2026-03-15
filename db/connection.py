"""
db/connection.py — Single psycopg2 connection context manager.
All other db modules import get_conn() from here.
"""

import psycopg2
import psycopg2.extras
from contextlib import contextmanager

from config import DB_URL


@contextmanager
def get_conn():
    """
    Yield a psycopg2 connection, commit on clean exit, rollback on exception.
    Usage:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(...)
    """
    conn = psycopg2.connect(DB_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
