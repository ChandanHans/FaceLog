"""
db/__init__.py — Public API for the db package.
Any other module can do: from db import insert_sighting, resolve_sighting, ...
"""

from .connection import get_conn
from .persons import insert_person, update_person_label, load_known_encodings, get_all_persons, get_person_names, merge_person_encoding, delete_person, reset_person_encoding, delete_orphaned_unknowns
from .sightings import (
    insert_sighting,
    resolve_sighting,
    get_unresolved_sightings,
    get_recent_sightings,
    get_sighting_date_range,
    delete_sightings,
    get_all_unmatched_face_images,
    save_sighting_encoding,
)
from .queue_audit import record_queued, mark_queue_processed

__all__ = [
    "get_conn",
    "insert_person",
    "update_person_label",
    "load_known_encodings",
    "get_all_persons",
    "get_person_names",
    "merge_person_encoding",
    "delete_person",
    "reset_person_encoding",
    "delete_orphaned_unknowns",
    "insert_sighting",
    "resolve_sighting",
    "get_unresolved_sightings",
    "get_recent_sightings",
    "get_sighting_date_range",
    "delete_sightings",
    "get_all_unmatched_face_images",
    "save_sighting_encoding",
    "record_queued",
    "mark_queue_processed",
]
