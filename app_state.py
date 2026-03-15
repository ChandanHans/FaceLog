"""
app_state.py — Shared process-level shutdown flag.

core/camera.py calls running() each frame to know when to stop.
main.py installs SIGINT/SIGTERM handlers that call shutdown().
"""

_running = True


def running() -> bool:
    return _running


def shutdown():
    global _running
    _running = False
