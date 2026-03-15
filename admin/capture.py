"""
admin/capture.py — Multi-angle webcam face capture.

Captures N photos of a person at different head angles,
saves them to face/<name>/ and imports the averaged encoding
directly into the PostgreSQL persons table.

Usage:
    python main.py capture <name>
    python main.py capture <name> --photos 5 --min-angle 2
"""

import os

import cv2
import face_recognition
import numpy as np

import db
from config import JPEG_QUALITY

FACES_DIR = "face"

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_PHOTOS = 5
DEFAULT_MIN_ANGLE_DELTA = 2  # degrees

_STEPS = [
    "Look STRAIGHT at the camera",
    "Turn head slightly LEFT  (~20°)",
    "Turn head slightly RIGHT (~20°)",
    "Tilt head DOWN  slightly (~15°)",
    "Tilt head UP    slightly (~15°)",
]


# ── Main capture function ─────────────────────────────────────────────────────

def capture_person(name: str, num_photos: int = DEFAULT_PHOTOS,
                   min_angle_delta: float = DEFAULT_MIN_ANGLE_DELTA) -> bool:
    """
    Capture `num_photos` images of `name` from the default webcam.
    Auto-captures whenever head angle changes enough from the previous shot.

    Saves images to face/<name>/image_N.jpg, computes an averaged
    128-D face encoding, and upserts into the persons table as 'known'.

    Returns True on success.
    """
    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    print(f"\n📹 Multi-angle capture for '{name}'")
    print(f"   Target : {num_photos} photos  |  Min angle change: {min_angle_delta}°")
    print(f"   Saving : {person_dir}/image_1.jpg … image_{num_photos}.jpg")
    print(f"   Controls: SPACE = force capture  |  Q = cancel\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return False

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    captured_count = 0
    captured_angles: list[tuple[float, float]] = []
    stable_frames = 0
    STABLE_NEEDED = 15
    cooldown = 0
    COOLDOWN_MAX = 40
    flash = 0
    FLASH_MAX = 7

    window_name = f"Capture — {name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while captured_count < num_photos:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from webcam")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        display = frame.copy()

        # ── Detect face + eyes ────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        cur_yaw: float | None = None
        cur_pitch: float | None = None
        cur_roll = 0.0

        if len(faces) > 0:
            stable_frames += 1
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])

            face_gray = gray[fy : fy + fh, fx : fx + fw]
            eyes = eye_cascade.detectMultiScale(
                face_gray, scaleFactor=1.1, minNeighbors=3
            )

            face_cx = fx + fw // 2
            cur_yaw = ((face_cx - w // 2) / (w // 2)) * 45.0

            face_cy = fy + fh // 2
            cur_pitch = ((face_cy - h // 3) / (h // 3)) * 30.0

            if len(eyes) >= 2:
                pts = sorted(eyes, key=lambda e: e[0])
                lx = pts[0][0] + pts[0][2] // 2
                ly = pts[0][1] + pts[0][3] // 2
                rx = pts[1][0] + pts[1][2] // 2
                ry = pts[1][1] + pts[1][3] // 2
                cur_roll = float(np.degrees(np.arctan2(ry - ly, rx - lx)))

            ready = stable_frames >= STABLE_NEEDED
            box_color = (0, 255, 0) if ready else (0, 200, 200)
            cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), box_color, 2)
            cv2.putText(
                display,
                f"Yaw:{cur_yaw:+.0f}°  Pitch:{cur_pitch:+.0f}°  Roll:{cur_roll:+.0f}°",
                (fx, fy + fh + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1,
            )
        else:
            stable_frames = 0

        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            print("❌ Cancelled by user")
            break

        # ── Capture decision ──────────────────────────────────────────────
        auto_capture = False
        manual_capture = key == ord(" ")

        if cur_yaw is not None and cooldown == 0:
            if not captured_angles:
                auto_capture = stable_frames >= STABLE_NEEDED
            else:
                min_dist = min(
                    max(abs(cur_yaw - ay), abs(cur_pitch - ap))
                    for ay, ap in captured_angles
                )
                auto_capture = min_dist >= min_angle_delta

        if (auto_capture or manual_capture) and cur_yaw is not None and cooldown == 0:
            captured_count += 1
            filename = f"image_{captured_count}.jpg"
            filepath = os.path.join(person_dir, filename)
            cv2.imwrite(filepath, frame)
            captured_angles.append((cur_yaw, cur_pitch))  # type: ignore[arg-type]
            cooldown = COOLDOWN_MAX
            flash = FLASH_MAX
            next_hint = _STEPS[captured_count] if captured_count < len(_STEPS) else "Done!"
            print(
                f"   📸 [{captured_count}/{num_photos}] {filename}"
                f"  Yaw:{cur_yaw:+.0f}°  Pitch:{cur_pitch:+.0f}°  Roll:{cur_roll:+.0f}°"
            )
            if captured_count < num_photos:
                print(f"        → Next: {next_hint}")

        if cooldown > 0:
            cooldown -= 1

        # ── Flash overlay ────────────────────────────────────────────────
        if flash > 0:
            alpha = (flash / FLASH_MAX) * 0.65
            white = np.full_like(display, 255)
            display = cv2.addWeighted(display, 1 - alpha, white, alpha, 0)
            cv2.putText(
                display, "CAPTURED!", (w // 2 - 90, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 180, 0), 3,
            )
            flash -= 1

        # ── HUD ──────────────────────────────────────────────────────────
        cv2.rectangle(display, (0, 0), (w, 50), (0, 0, 0), -1)
        step_idx = min(captured_count, num_photos - 1)
        step_label = _STEPS[step_idx] if captured_count < num_photos else "All done!"
        cv2.putText(
            display,
            f"Step {min(captured_count + 1, num_photos)}/{num_photos}: {step_label}",
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        cv2.putText(
            display, f"Person: {name}",
            (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1,
        )

        for i in range(num_photos):
            dot_col = (0, 255, 0) if i < captured_count else (70, 70, 70)
            cv2.circle(display, (w - 25 - i * 24, 25), 9, dot_col, -1)

        if len(faces) == 0:
            cv2.putText(
                display, "No face detected — move closer",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2,
            )
        elif not captured_angles and stable_frames < STABLE_NEEDED:
            bar = int((stable_frames / STABLE_NEEDED) * 200)
            cv2.rectangle(display, (10, h - 32), (210, h - 18), (50, 50, 50), -1)
            cv2.rectangle(display, (10, h - 32), (10 + bar, h - 18), (0, 220, 100), -1)
            cv2.putText(
                display, f"Hold still… {stable_frames}/{STABLE_NEEDED}",
                (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )
        elif captured_angles and cooldown == 0 and cur_yaw is not None:
            diff = min(
                max(abs(cur_yaw - ay), abs(cur_pitch - ap))
                for ay, ap in captured_angles
            )
            bar = int(min(diff / min_angle_delta, 1.0) * 200)
            cv2.rectangle(display, (10, h - 32), (210, h - 18), (50, 50, 50), -1)
            cv2.rectangle(display, (10, h - 32), (10 + bar, h - 18), (0, 200, 255), -1)
            cv2.putText(
                display, f"Angle change: {diff:.0f}/{min_angle_delta}°",
                (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )
        elif cooldown > 0:
            bar = int((1 - cooldown / COOLDOWN_MAX) * 200)
            cv2.rectangle(display, (10, h - 32), (210, h - 18), (50, 50, 50), -1)
            cv2.rectangle(display, (10, h - 32), (10 + bar, h - 18), (80, 255, 160), -1)
            cv2.putText(
                display, "Cooldown…",
                (10, h - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
            )

        cv2.putText(
            display, "SPACE = force capture   Q = cancel",
            (w - 290, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1,
        )
        cv2.imshow(window_name, display)

    cap.release()
    cv2.destroyAllWindows()

    if captured_count == 0:
        print("❌ No images captured")
        return False

    # ── Generate encodings from saved images ─────────────────────────────────
    print(f"\n🔄 Processing {captured_count} captured image(s)…")
    encodings: list[np.ndarray] = []
    reference_jpeg: bytes | None = None

    for i in range(1, captured_count + 1):
        fp = os.path.join(person_dir, f"image_{i}.jpg")
        img = cv2.imread(fp)
        if img is None:
            print(f"   ⚠️  image_{i}.jpg: could not read — skipped")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model="hog")
        if not locs:
            print(f"   ⚠️  image_{i}.jpg: no face detected — skipped")
            continue
        encs = face_recognition.face_encodings(rgb, [locs[0]])
        if not encs:
            print(f"   ⚠️  image_{i}.jpg: encoding failed — skipped")
            continue
        encodings.append(encs[0])
        # Use the first successful image as the reference photo
        if reference_jpeg is None:
            top, right, bottom, left = locs[0]
            face_crop = img[top:bottom, left:right]
            _, buf = cv2.imencode(".jpg", face_crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            reference_jpeg = buf.tobytes()
        print(f"   ✅ image_{i}.jpg")

    if not encodings:
        print("❌ Could not generate any encodings. Check images in:", person_dir)
        return False

    import pickle

    avg_encoding = np.mean(encodings, axis=0)
    encoding_bytes = pickle.dumps(avg_encoding)
    print(f"\n✅ Averaged {len(encodings)}/{captured_count} encodings → 128-D vector")

    # ── Upsert into DB ────────────────────────────────────────────────────────
    person_id = db.insert_person(
        name=name,
        label="known",
        face_encoding=encoding_bytes,
        reference_image=reference_jpeg,
    )
    print(f"✅ Saved to DB as person #{person_id}  (label=known)")
    print(f"📁 Images saved to: {person_dir}/")
    return True


# ── CLI wiring ────────────────────────────────────────────────────────────────

def register_subparser(sub) -> None:
    p = sub.add_parser("capture", help="Webcam multi-angle capture → save to DB")
    p.add_argument("name", help="Person's name")
    p.add_argument(
        "--photos", type=int, default=DEFAULT_PHOTOS,
        help=f"Number of photos to capture (default: {DEFAULT_PHOTOS})"
    )
    p.add_argument(
        "--min-angle", type=float, default=DEFAULT_MIN_ANGLE_DELTA,
        help=f"Min head-angle change between shots in degrees (default: {DEFAULT_MIN_ANGLE_DELTA})"
    )


def run(args) -> None:
    capture_person(
        name=args.name,
        num_photos=args.photos,
        min_angle_delta=args.min_angle,
    )
