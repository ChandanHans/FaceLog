"""
main.py — Single entry point for the FaceLog Security System.

Commands
────────
  python main.py run                        # start camera + workers (uses .env)
  python main.py run --source 0             # explicit webcam index
  python main.py run --source /path/to.mp4  # video file
  python main.py run --source rtsp://...    # RTSP stream
  python main.py run --no-display           # headless mode (server / CCTV)
  python main.py run --workers 4            # override worker count

  # Two-camera setup (entry + exit in one command):
  python main.py run --source 0 --camera-id cam_entry --direction entry \
                     --source2 1 --camera-id2 cam_exit --direction2 exit

  python main.py setup-db                   # create PostgreSQL tables

  python main.py admin list                 # show unmatched sightings
  python main.py admin label 42             # label sighting #42
  python main.py admin add-known            # import face/ directory into DB

  python main.py capture Alice              # webcam capture → save to DB
  python main.py capture Alice --photos 7   # capture 7 angles
"""

import argparse
import logging
import multiprocessing
import os
import signal
import sys

import app_state
from config import (
    CAMERA_ID,
    CAMERA_DIRECTION,
    CAMERA_SOURCE,
    CAMERA_ID2,
    CAMERA_DIRECTION2,
    CAMERA_SOURCE2,
    LOG_FILE,
    WORKER_PROCESSES,
)


# ── Logging (main process) ────────────────────────────────────────────────────

def _setup_logging():
    log_dir = os.path.dirname(LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [main] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )


log = logging.getLogger(__name__)


# ── Signal handlers ───────────────────────────────────────────────────────────

def _handle_signal(signum, frame):
    log.info("Shutdown signal received")
    app_state.shutdown()


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_run(args):
    import multiprocessing
    from db.setup import create_tables
    from core.camera import run_camera
    from worker.runner import start_workers, stop_workers

    # ── Camera 1 (primary) ────────────────────────────────────────────────
    source = args.source
    if source is None:
        source = CAMERA_SOURCE
    elif source.isdigit():
        source = int(source)

    camera_id = args.camera_id or CAMERA_ID
    direction = args.direction or CAMERA_DIRECTION
    n_workers = args.workers or WORKER_PROCESSES

    # ── Camera 2 (optional exit camera) ───────────────────────────────────
    source2 = args.source2 or (CAMERA_SOURCE2 if CAMERA_SOURCE2 else None)
    if source2 is not None and isinstance(source2, str) and source2.isdigit():
        source2 = int(source2)
    camera_id2 = args.camera_id2 or CAMERA_ID2
    direction2 = args.direction2 or CAMERA_DIRECTION2

    log.info("Ensuring database schema exists...")
    create_tables()

    log.info(f"Starting {n_workers} recognition worker(s)...")
    workers = start_workers(n_workers)

    # Spawn second camera process if --source2 provided
    cam2_proc = None
    if source2 is not None:
        log.info(f"Starting exit camera: source={source2} id={camera_id2}")
        cam2_proc = multiprocessing.Process(
            target=run_camera,
            kwargs=dict(
                source=source2,
                show_display=not args.no_display,
                camera_id=camera_id2,
                direction=direction2,
            ),
            daemon=True,
        )
        cam2_proc.start()

    try:
        run_camera(
            source=source,
            show_display=not args.no_display,
            camera_id=camera_id,
            direction=direction,
        )
    finally:
        if cam2_proc and cam2_proc.is_alive():
            cam2_proc.terminate()
            cam2_proc.join(timeout=5)
        stop_workers(workers)
        log.info("Shutdown complete")


def cmd_setup_db(args):
    from db.setup import create_tables
    create_tables()


def cmd_admin(args):
    from admin.label import run
    run(args)


def cmd_capture(args):
    from admin.capture import run
    run(args)


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="FaceLog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ──────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Start camera loop + recognition workers")
    p_run.add_argument(
        "--source", default=None,
        help="Camera index (0), video file, or RTSP URL. Defaults to CAMERA_SOURCE in .env"
    )
    p_run.add_argument(
        "--no-display", action="store_true",
        help="Headless mode — no OpenCV window (use on servers)"
    )
    p_run.add_argument(
        "--camera-id", default=None,
        help="Logical camera name stored in DB (default: CAMERA_ID from .env)"
    )
    p_run.add_argument(
        "--direction", default=None, choices=["entry", "exit"],
        help="Mark all sightings from this camera as 'entry' or 'exit'. Defaults to CAMERA_DIRECTION in .env (default: entry)"
    )
    p_run.add_argument(
        "--source2", default=None,
        help="(Optional) Second camera source for exit recording. Camera index (1), file, or RTSP URL."
    )
    p_run.add_argument(
        "--camera-id2", default=None,
        help="Logical name for second camera stored in DB (default: cam_exit)"
    )
    p_run.add_argument(
        "--direction2", default=None, choices=["entry", "exit"],
        help="Direction for second camera (default: exit)"
    )
    p_run.add_argument(
        "--workers", type=int, default=None,
        help="Number of recognition workers (default: WORKER_PROCESSES from .env)"
    )

    # ── setup-db ─────────────────────────────────────────────────────────────
    sub.add_parser("setup-db", help="Create/migrate PostgreSQL tables")

    # ── admin ─────────────────────────────────────────────────────────────────
    from admin.label import register_subparser
    register_subparser(sub)

    # ── capture ──────────────────────────────────────────────────────────────
    from admin.capture import register_subparser as register_capture
    register_capture(sub)

    return parser


# ── Interactive menu (no-argument mode) ──────────────────────────────────────

def _interactive_menu():
    """Prompt the user interactively when no CLI arguments are given."""
    menu = """
╔══════════════════════════════════════════════╗
║              FaceLog                         ║
╠══════════════════════════════════════════════╣
║  1. Start camera (run)                       ║
║  2. Setup database                           ║
║  3. Capture face from webcam                 ║
║  4. Admin — list unmatched sightings         ║
║  5. Admin — label a sighting                 ║
║  6. Admin — import face/ directory into DB   ║
║  7. Open web dashboard (Streamlit)           ║
║  q. Quit                                     ║
╚══════════════════════════════════════════════╝"""
    print(menu)

    choice = input("Choose an option: ").strip().lower()

    if choice == "1":
        source = input("  Camera source [0]: ").strip() or "0"
        no_display = input("  Headless mode? (y/N): ").strip().lower() == "y"
        workers_in = input(f"  Workers [{WORKER_PROCESSES}]: ").strip()
        workers = int(workers_in) if workers_in.isdigit() else None

        import argparse as _ap
        args = _ap.Namespace(
            source=source,
            no_display=no_display,
            camera_id=None,
            direction=None,
            workers=workers,
            source2=None,
            camera_id2=None,
            direction2=None,
        )
        cmd_run(args)

    elif choice == "2":
        cmd_setup_db(None)

    elif choice == "3":
        name = input("  Person's name: ").strip()
        if not name:
            print("Name cannot be empty.")
            return
        photos_in = input("  Number of photos [5]: ").strip()
        angle_in = input("  Min angle delta [2]: ").strip()
        import argparse as _ap
        args = _ap.Namespace(
            name=name,
            photos=int(photos_in) if photos_in.isdigit() else 5,
            min_angle=float(angle_in) if angle_in else 2.0,
        )
        cmd_capture(args)

    elif choice == "4":
        import argparse as _ap
        args = _ap.Namespace(admin_command="list", limit=20)
        cmd_admin(args)

    elif choice == "5":
        sid = input("  Sighting ID to label: ").strip()
        if not sid.isdigit():
            print("Invalid ID.")
            return
        import argparse as _ap
        args = _ap.Namespace(admin_command="label", sighting_id=int(sid))
        cmd_admin(args)

    elif choice == "6":
        face_dir = input("  Face directory [face]: ").strip() or "face"
        import argparse as _ap
        args = _ap.Namespace(admin_command="add-known", face_dir=face_dir)
        cmd_admin(args)

    elif choice == "7":
        import subprocess as _sp
        import sys as _sys
        print("  Starting Streamlit dashboard… (Ctrl+C to stop)")
        _sp.run([_sys.executable, "-m", "streamlit", "run", "dashboard.py"])

    elif choice == "q":
        print("Bye.")
    else:
        print(f"Unknown option: {choice!r}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "setup-db":
        cmd_setup_db(args)
    elif args.command == "admin":
        cmd_admin(args)
    elif args.command == "capture":
        cmd_capture(args)
    else:
        _interactive_menu()


if __name__ == "__main__":
    # Required on Windows for multiprocessing
    multiprocessing.freeze_support()
    main()
