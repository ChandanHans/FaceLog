"""
admin/label.py — CLI tool for reviewing and labelling unknown sightings.

Usage:
    python main.py admin list                     # show unmatched sightings
    python main.py admin label <id>               # interactively label one
    python main.py admin add-known [--face-dir face]  # import face/ directory
"""

import os
import sys
import textwrap

import cv2
import face_recognition
import numpy as np

import db
from config import JPEG_QUALITY


# ── Display helper ────────────────────────────────────────────────────────────

def _show_face(jpeg_bytes: bytes, title: str = "Face"):
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("  (could not decode image)")
        return
    h, w = img.shape[:2]
    if max(h, w) < 200:
        scale = 200 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _encode_face(jpeg_bytes: bytes):
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model="hog")
    if not locs:
        return None
    encs = face_recognition.face_encodings(rgb, locs)
    return encs[0] if encs else None


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list(args):
    rows = db.get_unresolved_sightings(limit=args.limit)
    if not rows:
        print("No unmatched sightings.")
        return
    print(f"\n{'ID':>6}  {'Detected At':<26}  {'Camera':<10}  {'Direction':<10}")
    print("─" * 62)
    for r in rows:
        print(
            f"{r['id']:>6}  {str(r['detected_at']):<26}  "
            f"{r['camera_id']:<10}  {r['direction']:<10}"
        )
    print()


def cmd_label(args):
    sighting_id = args.id

    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, detected_at, camera_id, person_id, face_image "
            "FROM sightings WHERE id = %s",
            (sighting_id,),
        )
        row = cur.fetchone()

    if row is None:
        print(f"Sighting {sighting_id} not found.")
        sys.exit(1)

    sid, detected_at, camera_id, current_person_id, face_jpeg = row
    face_jpeg = bytes(face_jpeg)

    print(f"\nSighting #{sid}  |  {detected_at}  |  {camera_id}")
    print("Showing face image — press any key to close…")
    _show_face(face_jpeg, f"Sighting #{sid}")

    print("\n  [1] Known person (enter name)")
    print("  [2] Suspicious")
    print("  [3] Skip")
    choice = input("Choice: ").strip()

    if choice == "1":
        name = input("Name: ").strip()
        if not name:
            print("Name cannot be empty. Skipping.")
            return
        enc = _encode_face(face_jpeg)
        person_id = db.insert_person(
            name=name, label="known", face_encoding=enc, reference_image=face_jpeg
        )
        db.resolve_sighting(sighting_id, status="matched", person_id=person_id)
        print(f"Saved as KNOWN: {name} (person_id={person_id})")

    elif choice == "2":
        notes = input("Notes (optional): ").strip() or None
        person_id = current_person_id
        if person_id is None:
            person_id = db.insert_person(
                label="suspicious", reference_image=face_jpeg, notes=notes
            )
        else:
            db.update_person_label(person_id, label="suspicious", notes=notes)
        db.resolve_sighting(sighting_id, status="flagged", person_id=person_id)
        print(f"Marked as SUSPICIOUS (person_id={person_id})")

    else:
        print("Skipped.")


def cmd_add_known(args):
    """Import all persons from face/<name>/ directories into the DB."""
    face_dir = args.face_dir
    if not os.path.isdir(face_dir):
        print(f"Directory not found: {face_dir}")
        sys.exit(1)

    for person_name in sorted(os.listdir(face_dir)):
        person_path = os.path.join(face_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        image_files = [
            f for f in os.listdir(person_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not image_files:
            print(f"  No images found for {person_name}")
            continue

        encodings = []
        ref_jpeg = None

        for img_file in image_files:
            img = cv2.imread(os.path.join(person_path, img_file))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model="hog")
            if not locs:
                print(f"    No face in {img_file}")
                continue
            encs = face_recognition.face_encodings(rgb, locs)
            if encs:
                encodings.append(encs[0])
                if ref_jpeg is None:
                    ok, buf = cv2.imencode(
                        ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    )
                    ref_jpeg = bytes(buf) if ok else None

        if not encodings:
            print(f"  Could not encode any face for {person_name}")
            continue

        avg_enc = np.mean(encodings, axis=0)
        person_id = db.insert_person(
            name=person_name,
            label="known",
            face_encoding=avg_enc,
            reference_image=ref_jpeg,
        )
        print(f"  Added: {person_name} (person_id={person_id}, {len(encodings)} photos)")


# ── Argument parser (registered by main.py) ───────────────────────────────────

def register_subparser(sub):
    """Attach 'admin' subcommand to the main argparse subparsers object."""
    p = sub.add_parser(
        "admin",
        help="Admin tools: list/label unknowns, add known persons",
        formatter_class=__import__("argparse").RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py admin list
              python main.py admin label 42
              python main.py admin add-known
              python main.py admin add-known --face-dir face
        """),
    )
    asub = p.add_subparsers(dest="admin_command")

    pl = asub.add_parser("list", help="List unmatched sightings")
    pl.add_argument("--limit", type=int, default=20)

    plb = asub.add_parser("label", help="Label a sighting")
    plb.add_argument("id", type=int)

    pa = asub.add_parser("add-known", help="Import face/ directory")
    pa.add_argument("--face-dir", default="face")

    return p


def run(args):
    if args.admin_command == "list":
        cmd_list(args)
    elif args.admin_command == "label":
        cmd_label(args)
    elif args.admin_command == "add-known":
        cmd_add_known(args)
    else:
        print("Use: python main.py admin list|label <id>|add-known")
