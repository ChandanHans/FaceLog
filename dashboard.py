"""
dashboard.py — Streamlit web UI for FaceLog.

Run with:
    streamlit run dashboard.py
or from the interactive menu (option 7).
"""

import base64
import os
import pickle

import cv2
import face_recognition
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components
from io import BytesIO

import db

# ── Custom component: drag-to-select image grid ───────────────────────────────
_image_selector = st_components.declare_component(
    "image_selector",
    path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "components", "image_selector"),
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FaceLog",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🔐 FaceLog")
page = st.sidebar.radio(
    "Navigate",
    ["📋 Recent Sightings", "🔍 Review Unknowns", "👥 People Directory"],
)
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh"):
    st.rerun()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _img(raw) -> bytes | None:
    return bytes(raw) if raw is not None else None


def _compute_encoding(jpeg_bytes: bytes):
    """Decode a face JPEG and return a 128-D encoding, or None if no face found."""
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


def _rematch_person(person_id: int, tolerance: float = 0.48) -> int:
    """
    Compare every existing unmatched sighting against a single person's encoding.
    Uses the cached per-sighting encoding when available; falls back to decoding
    the JPEG for old sightings that pre-date the encoding cache.
    Returns count of newly matched sightings.
    """
    import pickle
    import face_recognition as _fr

    with db.get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT face_encoding FROM persons WHERE id = %s", (person_id,))
        row = cur.fetchone()
    if not row or row[0] is None:
        return 0
    known_enc = pickle.loads(bytes(row[0]))

    unmatched = db.get_all_unmatched_face_images()
    matched_count = 0
    for s in unmatched:
        sid = s["id"]

        # Fast path: use cached encoding
        if s["face_encoding"] is not None:
            sighting_enc = pickle.loads(bytes(s["face_encoding"]))
        else:
            # Slow fallback: decode JPEG and compute encoding (old sightings)
            from worker.recognizer import decode_jpeg
            img_bgr = decode_jpeg(bytes(s["face_image"]))
            if img_bgr is None:
                continue
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            locs = _fr.face_locations(rgb, model="hog")
            if not locs:
                # Treat whole crop as face region (same fallback as recognizer.py)
                h, w = rgb.shape[:2]
                locs = [(0, w, h, 0)]
            encs = _fr.face_encodings(rgb, locs)
            if not encs:
                continue
            sighting_enc = encs[0]
            # Cache it now so next rematch is fast
            db.save_sighting_encoding(sid, sighting_enc)

        dist = float(_fr.face_distance([known_enc], sighting_enc)[0])
        if dist <= tolerance:
            db.resolve_sighting(sid, status="matched", person_id=person_id,
                                confidence=round(1.0 - dist, 4))
            matched_count += 1

    # Remove orphaned unknown person rows left behind after re-assignment
    db.delete_orphaned_unknowns()
    return matched_count


STATUS_ICON = {
    "matched":   "🟢",
    "unmatched": "🔴",
    "pending":   "🟡",
    "flagged":   "🟠",
}
LABEL_ICON = {
    "known":      "🟢",
    "unknown":    "🟡",
    "suspicious": "🔴",
}

# ── Page: Recent Sightings ────────────────────────────────────────────────────

if page == "📋 Recent Sightings":
    st.title("📋 Recent Sightings")

    limit = st.sidebar.number_input(
        "Show last N sightings", min_value=10, max_value=500, value=50, step=10
    )
    rows = db.get_recent_sightings(limit=int(limit))

    if not rows:
        st.info("No sightings recorded yet.")
    else:
        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", len(rows))
        m2.metric("Matched 🟢",   sum(1 for r in rows if r["status"] == "matched"))
        m3.metric("Unmatched 🔴", sum(1 for r in rows if r["status"] == "unmatched"))
        m4.metric("Pending 🟡",   sum(1 for r in rows if r["status"] == "pending"))

        st.markdown("---")

        data = [
            {
                "ID": r["id"],
                "Status": f"{STATUS_ICON.get(r['status'], '⚪')} {r['status']}",
                "Person": r["name"] or "—",
                "Label":  r["label"] or "—",
                "Camera": r["camera_id"],
                "Direction": r["direction"],
                "Confidence": f"{r['confidence']:.0%}" if r["confidence"] else "—",
                "Detected At": (
                    r["detected_at"].strftime("%Y-%m-%d %H:%M:%S")
                    if r["detected_at"] else "—"
                ),
            }
            for r in rows
        ]
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

# ── Page: Review Unknowns ─────────────────────────────────────────────────────

elif page == "🔍 Review Unknowns":
    st.title("🔍 Review Unmatched Sightings")

    _NEW = "➕ New person…"

    limit = st.sidebar.number_input(
        "Load per page", min_value=24, max_value=200, value=24, step=24
    )
    rows = db.get_unresolved_sightings(limit=int(limit))

    # Filter out sightings the user skipped this session
    _skipped = st.session_state.get("_skipped_ids", set())
    rows = [r for r in rows if r["id"] not in _skipped]

    if not rows:
        st.success("✅ All sightings have been reviewed!")
    else:
        # ── Action panel (always at top) ──────────────────────────────────────
        existing_persons: list[tuple[int, str]] = db.get_person_names()
        person_options = [_NEW] + [name for _, name in existing_persons]
        person_id_by_name = {name: pid for pid, name in existing_persons}

        # Read selection returned by the component from the PREVIOUS render
        component_value = st.session_state.get("_img_sel_value", [])
        selected_ids: list[int] = [int(i) for i in (component_value or [])]
        selected_rows = [r for r in rows if r["id"] in set(selected_ids)]

        st.markdown(
            f"**{len(rows)}** loaded &nbsp;·&nbsp; "
            f"**{len(selected_ids)}** selected — "
            "*drag or click images below to select*",
            unsafe_allow_html=True,
        )

        submitted = False
        with st.form("bulk_action"):
            fc1, fc2, fc3, fc4 = st.columns([1.2, 2.5, 2.5, 1.4])
            with fc1:
                action = st.radio("Label as", ["known", "suspicious", "skip", "🗑️ delete"])
            with fc2:
                selected_option = st.selectbox(
                    "Person",
                    options=person_options,
                    help="Choose existing to merge encoding, or '➕ New person…' to create.",
                )
            with fc3:
                new_name_in = st.text_input("New person name", placeholder="Enter name…")
            with fc4:
                st.markdown("<br><br>", unsafe_allow_html=True)
                btn_label = (
                    f"🗑️ Delete {len(selected_ids)}" if action == "🗑️ delete" and selected_ids
                    else f"✅ Apply to {len(selected_ids)}" if selected_ids
                    else "✅ Apply"
                )
                submitted = st.form_submit_button(
                    btn_label,
                    use_container_width=True,
                    disabled=len(selected_ids) == 0,
                )

        if submitted and selected_ids:
            if action == "🗑️ delete":
                n = db.delete_sightings(selected_ids)
                skipped = st.session_state.setdefault("_skipped_ids", set())
                skipped.update(selected_ids)
                st.session_state["_img_sel_value"] = []
                st.warning(f"🗑️ Permanently deleted {n} sighting(s).")
                st.rerun()
            elif action == "skip":
                skipped = st.session_state.setdefault("_skipped_ids", set())
                skipped.update(selected_ids)
                st.session_state["_img_sel_value"] = []
                st.toast(f"Skipped {len(selected_ids)} sighting(s).")
                st.rerun()
            else:
                is_new = selected_option == _NEW
                final_name = new_name_in.strip() if is_new else selected_option
                if not final_name:
                    st.error("Please enter a name.")
                else:
                    pid = None
                    count = 0
                    for row in selected_rows:
                        face = _img(row["face_image"])
                        encoding = _compute_encoding(face) if face else None
                        if pid is None:
                            if is_new:
                                pid = db.insert_person(
                                    name=final_name,
                                    label="known" if action == "known" else "suspicious",
                                    face_encoding=encoding,
                                    reference_image=face,
                                )
                            else:
                                pid = person_id_by_name[final_name]
                                if action == "suspicious":
                                    db.update_person_label(pid, label="suspicious")
                                if encoding is not None:
                                    db.merge_person_encoding(pid, encoding)
                        else:
                            if encoding is not None:
                                db.merge_person_encoding(pid, encoding)
                        status = "matched" if action == "known" else "flagged"
                        db.resolve_sighting(row["id"], status=status, person_id=pid)
                        count += 1

                    # Clean up unknown placeholder rows that are now orphaned
                    db.delete_orphaned_unknowns()
                    st.session_state["_img_sel_value"] = []
                    if action == "known":
                        st.success(f"✅ Labelled **{count}** sighting(s) as **{final_name}**.")
                    else:
                        st.warning(
                            f"⚠️ Flagged **{count}** sighting(s) as suspicious"
                            f"{(' (' + final_name + ')') if final_name else ''}."
                        )
                    st.rerun()

        st.markdown("---")

        # ── Drag-to-select image grid (custom component) ──────────────────────
        items = []
        for row in rows:
            face = _img(row["face_image"])
            b64 = base64.b64encode(face).decode() if face else ""
            items.append({
                "id": row["id"],
                "img_b64": b64,
                "caption": row["detected_at"].strftime("%b%d %H:%M"),
            })

        # Estimate height: toolbar ~35px + rows * (cell ~120px + gap)
        n_rows = max(1, -(-len(items) // 6))  # ceiling division
        est_h = 35 + n_rows * 125 + 20

        raw_value = _image_selector(items=items, key="_img_sel", default=[], height=est_h)
        # Persist the returned selection so the action panel above can read it.
        # Trigger a rerun when the value changes so the count at the top updates immediately.
        if raw_value is not None and raw_value != st.session_state.get("_img_sel_value"):
            st.session_state["_img_sel_value"] = raw_value
            st.rerun()

# ── Page: People Directory ────────────────────────────────────────────────────

elif page == "👥 People Directory":
    st.title("👥 People Directory")

    label_filter = st.sidebar.multiselect(
        "Filter by label",
        ["known", "unknown", "suspicious"],
        default=["known", "unknown", "suspicious"],
    )
    rows = db.get_all_persons()
    rows = [r for r in rows if r["label"] in label_filter]

    if not rows:
        st.info("No persons in the database yet.")
    else:
        st.markdown(f"**{len(rows)}** person(s) found.")
        st.markdown("---")

        N_COLS = 4
        for batch_start in range(0, len(rows), N_COLS):
            cols = st.columns(N_COLS)
            for col_idx, row in enumerate(rows[batch_start : batch_start + N_COLS]):
                with cols[col_idx]:
                    ref = _img(row["reference_image"])
                    if ref:
                        st.image(BytesIO(ref), use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='height:120px;background:#2a2a2a;"
                            "border-radius:8px;display:flex;align-items:center;"
                            "justify-content:center;color:#666'>no photo</div>",
                            unsafe_allow_html=True,
                        )
                    icon = LABEL_ICON.get(row["label"], "⚪")
                    st.markdown(f"**{row['name'] or '— unnamed —'}**")
                    st.caption(
                        f"{icon} {row['label']}  ·  #{row['id']}\n"
                        + (row["added_at"].strftime("%Y-%m-%d") if row["added_at"] else "")
                    )

                    pid = row["id"]
                    confirm_key = f"_confirm_del_person_{pid}"
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        if st.button("🔄 Reset faces", key=f"reset_{pid}",
                                     help="Clear all stored face encodings so "
                                          "this person can be re-enrolled from scratch",
                                     use_container_width=True):
                            db.reset_person_encoding(pid)
                            st.toast(f"Encoding cleared for {row['name'] or f'#{pid}'}.")
                            st.rerun()
                    with bc2:
                        if not st.session_state.get(confirm_key):
                            if st.button("🗑️ Delete", key=f"del_{pid}",
                                         use_container_width=True):
                                st.session_state[confirm_key] = True
                                st.rerun()
                        else:
                            if st.button("⚠️ Confirm delete", key=f"confirm_{pid}",
                                         use_container_width=True, type="primary"):
                                db.delete_person(pid)
                                st.session_state.pop(confirm_key, None)
                                st.toast(f"Deleted {row['name'] or f'#{pid}'}.")
                                st.rerun()

                    # Re-match past unmatched sightings against this person
                    if row["label"] == "known" and st.button(
                        "🔍 Re-match history",
                        key=f"rematch_{pid}",
                        help="Scan all old unmatched sightings and link any that belong to this person",
                        use_container_width=True,
                    ):
                        with st.spinner("Scanning past sightings…"):
                            n = _rematch_person(pid)
                        if n:
                            st.success(f"✅ Linked **{n}** past sighting(s) to **{row['name'] or f'#{pid}'}**.")
                        else:
                            st.info("No additional matches found in past sightings.")
