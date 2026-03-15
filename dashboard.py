"""
dashboard.py — Streamlit web UI for the Entry/Exit Security System.

Run with:
    streamlit run dashboard.py
or from the interactive menu (option 7).
"""

import pickle

import cv2
import face_recognition
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO

import db

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Entry/Exit Security",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🔐 Entry/Exit Security")
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

    limit = st.sidebar.number_input(
        "Load per page", min_value=6, max_value=100, value=18, step=6
    )
    rows = db.get_unresolved_sightings(limit=int(limit))

    if not rows:
        st.success("✅ All sightings have been reviewed!")
    else:
        st.info(f"**{len(rows)}** unmatched sighting(s) awaiting review.")
        st.markdown("---")

        # Build person name options for the dropdown (loaded once per page render)
        _NEW = "➕ New person…"
        existing_persons: list[tuple[int, str]] = db.get_person_names()
        person_options = [_NEW] + [name for _, name in existing_persons]
        person_id_by_name = {name: pid for pid, name in existing_persons}

        N_COLS = 3
        for batch_start in range(0, len(rows), N_COLS):
            cols = st.columns(N_COLS)
            for col_idx, row in enumerate(rows[batch_start : batch_start + N_COLS]):
                with cols[col_idx]:
                    face = _img(row["face_image"])
                    if face:
                        st.image(BytesIO(face), use_container_width=True)
                    else:
                        st.write("*(no image)*")

                    st.caption(
                        f"**#{row['id']}** &nbsp;·&nbsp; "
                        f"{row['detected_at'].strftime('%b %d %H:%M:%S')} &nbsp;·&nbsp; "
                        f"{row['camera_id']}"
                    )

                    with st.form(key=f"review_{row['id']}"):
                        action = st.radio(
                            "Label as",
                            ["known", "suspicious", "skip"],
                            horizontal=True,
                            key=f"action_{row['id']}",
                        )

                        # Searchable person dropdown (only visible for known/suspicious)
                        selected_option = st.selectbox(
                            "Person",
                            options=person_options,
                            index=0,
                            key=f"person_sel_{row['id']}",
                            help="Start typing to search. Choose an existing person to merge, or '➕ New person…' to create one.",
                        )
                        # New name input — only needed when creating a new person
                        new_name_in = st.text_input(
                            "New person name",
                            placeholder="Enter name…",
                            key=f"newname_{row['id']}",
                        )
                        submitted = st.form_submit_button("✅ Save")

                    if submitted:
                        sid = row["id"]
                        if action == "skip":
                            st.toast(f"Sighting #{sid} skipped.")

                        elif action == "known":
                            is_new = (selected_option == _NEW)
                            final_name = new_name_in.strip() if is_new else selected_option

                            if not final_name:
                                st.error("Name is required.")
                            else:
                                encoding = _compute_encoding(face) if face else None
                                if is_new:
                                    # Create brand-new person row
                                    if encoding is None:
                                        st.warning("⚠️ No face encoding extracted — person saved but won't be auto-recognised.")
                                    pid = db.insert_person(
                                        name=final_name,
                                        label="known",
                                        face_encoding=encoding,
                                        reference_image=face,
                                    )
                                    st.success(f"✅ New person **{final_name}** created.")
                                else:
                                    # Merge encoding into existing person
                                    pid = person_id_by_name[selected_option]
                                    if encoding is not None:
                                        db.merge_person_encoding(pid, encoding)
                                    st.success(f"✅ Encoding merged into **{final_name}**.")

                                db.resolve_sighting(sid, status="matched", person_id=pid)
                                st.rerun()

                        elif action == "suspicious":
                            is_new = (selected_option == _NEW)
                            final_name = new_name_in.strip() if is_new else selected_option
                            encoding = _compute_encoding(face) if face else None

                            if is_new:
                                pid = db.insert_person(
                                    name=final_name or None,
                                    label="suspicious",
                                    face_encoding=encoding,
                                    reference_image=face,
                                )
                            else:
                                pid = person_id_by_name[selected_option]
                                db.update_person_label(pid, label="suspicious")
                                if encoding is not None:
                                    db.merge_person_encoding(pid, encoding)

                            db.resolve_sighting(sid, status="flagged", person_id=pid)
                            st.warning(f"⚠️ Marked as suspicious{(' — ' + final_name) if final_name else ''}.")
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
