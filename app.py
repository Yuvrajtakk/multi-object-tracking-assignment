
import streamlit as st
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import tempfile, os, time, threading

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Object Tracker",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Multi-Object Tracking — Sports Video")
st.markdown(
    "Upload a sports video and the pipeline will detect & track every player "
    "using **YOLOv8m + ByteTrack**. Each player gets a persistent color-coded ID."
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def filter_by_area(det, min_area):
    areas = (det.xyxy[:, 2] - det.xyxy[:, 0]) * (det.xyxy[:, 3] - det.xyxy[:, 1])
    return det[areas >= min_area]


def id_color(track_id):
    np.random.seed(int(track_id) * 7)
    return tuple(int(c) for c in np.random.randint(80, 230, 3))


@st.cache_resource(show_spinner="Loading YOLOv8m model…")
def load_model():
    return YOLO("yolov8m.pt")


# ── Sidebar — parameters ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    st.subheader("Detection")
    conf_thresh  = st.slider("Confidence threshold", 0.1, 0.9, 0.30, 0.05,
                             help="YOLO detection confidence cut-off")
    min_box_area = st.slider("Min box area (px²)", 100, 5000, 1000, 100,
                             help="Drops tiny partial-body detections")

    st.subheader("ByteTrack")
    track_thresh = st.slider("Track activation threshold", 0.1, 0.9, 0.50, 0.05,
                             help="Minimum confidence to start a new track")
    match_thresh = st.slider("IoU match threshold", 0.1, 0.99, 0.75, 0.05,
                             help="IoU cut-off for matching detections to tracks")
    buffer_size  = st.slider("Lost-track buffer (frames)", 10, 120, 60, 5,
                             help="How long a lost track stays alive")
    min_hits     = st.slider("Min consecutive frames for ID", 1, 10, 3, 1,
                             help="Frames before a detection gets a permanent ID")

    st.subheader("Stationary filter")
    enable_stationary = st.toggle("Suppress stationary viewers", value=True)
    stationary_px     = st.slider("Movement threshold (px)", 5, 100, 20, 5,
                                  help="Centroids that move less than this are greyed out",
                                  disabled=not enable_stationary)
    stationary_after  = st.slider("Apply after N frames", 50, 600, 300, 50,
                                  disabled=not enable_stationary)

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your video (MP4, AVI, MOV, MKV)",
    type=["mp4", "avi", "mov", "mkv"],
)

if not uploaded:
    st.info("👆 Upload a video to get started.")
    st.stop()

# Save upload to a temp file
tmp_in  = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
tmp_in.write(uploaded.read())
tmp_in.flush()
VIDEO_PATH = tmp_in.name

# Read meta
cap   = cv2.VideoCapture(VIDEO_PATH)
FPS   = cap.get(cv2.CAP_PROP_FPS) or 30
W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Resolution", f"{W}×{H}")
col2.metric("FPS", f"{FPS:.0f}")
col3.metric("Duration", f"{TOTAL/FPS:.1f}s")
col4.metric("Frames", TOTAL)

# ── Preview mid-frame detection ───────────────────────────────────────────────
with st.expander("🔍 Detection preview (mid-video frame)", expanded=False):
    if st.button("Run detection check"):
        model = load_model()
        cap2  = cv2.VideoCapture(VIDEO_PATH)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, TOTAL // 2)
        _, test_frame = cap2.read()
        cap2.release()

        results    = model(test_frame, conf=conf_thresh, classes=[0], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = filter_by_area(detections, min_box_area)

        viz = test_frame.copy()
        for box, conf in zip(detections.xyxy, detections.confidence):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 200, 80), 2)
            cv2.putText(viz, f"{conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 80), 1)

        st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB),
                 caption=f"{len(detections)} persons detected (conf ≥ {conf_thresh})")

# ── Main run ──────────────────────────────────────────────────────────────────
st.divider()
run_btn = st.button("🚀 Run Tracking Pipeline", type="primary", use_container_width=True)

if run_btn:
    model = load_model()

    tracker = sv.ByteTrack(
        track_activation_threshold = track_thresh,
        lost_track_buffer          = buffer_size,
        minimum_matching_threshold = match_thresh,
        minimum_consecutive_frames = min_hits,
        frame_rate                 = int(FPS),
    )

    tmp_out = tempfile.NamedTemporaryFile(suffix="_tracked.mp4", delete=False)
    OUTPUT_PATH = tmp_out.name
    tmp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (W, H))

    cap = cv2.VideoCapture(VIDEO_PATH)

    progress_bar   = st.progress(0, text="Starting…")
    stats_cols     = st.columns(4)
    frame_metric   = stats_cols[0].empty()
    active_metric  = stats_cols[1].empty()
    total_metric   = stats_cols[2].empty()
    fps_metric     = stats_cols[3].empty()

    # centroid history for stationary filter
    centroid_history: dict[int, list] = {}

    frame_idx = 0
    all_ids   = set()
    grey_ids  = set()
    start     = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results    = model(frame, conf=conf_thresh, classes=[0], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = filter_by_area(detections, min_box_area)
        tracked    = tracker.update_with_detections(detections)

        if tracked.tracker_id is not None:
            all_ids.update(tracked.tracker_id.tolist())

            # Stationary filter — track centroid history
            if enable_stationary:
                for box, tid in zip(tracked.xyxy, tracked.tracker_id):
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    centroid_history.setdefault(int(tid), []).append((cx, cy))

                if frame_idx >= stationary_after:
                    for tid, pts in centroid_history.items():
                        if len(pts) >= 10:
                            xs = [p[0] for p in pts]
                            ys = [p[1] for p in pts]
                            total_mv = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
                            if total_mv < stationary_px:
                                grey_ids.add(tid)

        annotated = frame.copy()
        if tracked.tracker_id is not None:
            for box, tid in zip(tracked.xyxy, tracked.tracker_id):
                x1, y1, x2, y2 = map(int, box)
                is_grey = enable_stationary and int(tid) in grey_ids
                color   = (160, 160, 160) if is_grey else id_color(int(tid))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label   = f"ID {tid}" + (" (viewer)" if is_grey else "")
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 7), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        active = len(tracked.tracker_id) if tracked.tracker_id is not None else 0
        cv2.putText(annotated,
                    f"Frame {frame_idx}   Active: {active}   Total IDs: {len(all_ids)}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 230), 2)

        writer.write(annotated)
        frame_idx += 1

        # Update UI every 10 frames
        if frame_idx % 10 == 0 or frame_idx == TOTAL:
            pct     = frame_idx / max(TOTAL, 1)
            elapsed = time.time() - start
            fps_now = frame_idx / max(elapsed, 0.001)
            eta_s   = (TOTAL - frame_idx) / max(fps_now, 0.001)
            progress_bar.progress(
                pct,
                text=f"Processing frame {frame_idx}/{TOTAL} — ETA {eta_s:.0f}s"
            )
            frame_metric.metric("Frame", f"{frame_idx}/{TOTAL}")
            active_metric.metric("Active IDs", active)
            total_metric.metric("Unique IDs", len(all_ids))
            fps_metric.metric("Speed (FPS)", f"{fps_now:.1f}")

    cap.release()
    writer.release()

    progress_bar.progress(1.0, text="✅ Done!")

    elapsed = time.time() - start
    st.success(
        f"Processed **{frame_idx} frames** in **{elapsed:.0f}s** "
        f"({frame_idx/elapsed:.1f} FPS) — **{len(all_ids)} unique IDs** found"
    )

    # ── Results ───────────────────────────────────────────────────────────────
    st.subheader("📊 Results")

    res_cols = st.columns(3)
    res_cols[0].metric("Frames processed", frame_idx)
    res_cols[1].metric("Unique player IDs", len(all_ids - grey_ids))
    res_cols[2].metric("Viewer IDs suppressed", len(grey_ids))

    # Sample frame from output
    cap_out = cv2.VideoCapture(OUTPUT_PATH)
    cap_out.set(cv2.CAP_PROP_POS_FRAMES, frame_idx // 2)
    _, sample = cap_out.read()
    cap_out.release()
    if sample is not None:
        st.image(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB),
                 caption="Sample frame from tracked output (mid-video)")

    # Download button
    with open(OUTPUT_PATH, "rb") as f:
        st.download_button(
            label="⬇️ Download tracked video",
            data=f.read(),
            file_name=f"{uploaded.name.rsplit('.', 1)[0]}_tracked.mp4",
            mime="video/mp4",
            use_container_width=True,
            type="primary",
        )

    # Clean up temp files
    try:
        os.unlink(VIDEO_PATH)
        os.unlink(OUTPUT_PATH)
    except Exception:
        pass
