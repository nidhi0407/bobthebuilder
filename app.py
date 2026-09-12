"""
ForeSite AI - live object detection/tracking demo.

Streamlit app that takes an uploaded video, runs YOLOv8 detection + tracking
frame by frame, overlays boxes/labels, and logs "worker approaching forklift"
events to a terminal-style command log.

This is the MVP detection layer for the fuller ForeSite pipeline described in
README.md (trajectory prediction / risk engine / counterfactual simulation
are not implemented here yet).
"""

import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model configuration
#
# Using the stock COCO-pretrained checkpoint so the app runs out of the box.
# COCO has no "worker"/"forklift" classes, so we stand in with the closest
# COCO classes for now:
#   - "person" -> worker
#   - "truck"  -> forklift
#
# To swap in a fine-tuned checkpoint with real worker/forklift classes later,
# just point MODEL_PATH at it and update WORKER_CLASS_NAME / FORKLIFT_CLASS_NAME
# (and WORKER_LABEL / FORKLIFT_LABEL if you want the log wording to change too).
# The rest of the pipeline (tracking, drawing, proximity logging) is written
# against these names/ids generically and does not need to change.
# ---------------------------------------------------------------------------
MODEL_PATH = "yolov8n.pt"

WORKER_CLASS_NAME = "person"
FORKLIFT_CLASS_NAME = "truck"
WORKER_LABEL = "Worker"
FORKLIFT_LABEL = "Forklift"

# COCO class ids:
# 0: person -> Worker
# 2: car, 5: bus, 6: train, 7: truck -> Forklift / Heavy Machinery
WORKER_CLASS_ID = 0
FORKLIFT_CLASS_IDS = {2, 5, 6, 7}

DETECT_CLASSES = [WORKER_CLASS_ID] + list(FORKLIFT_CLASS_IDS)

DEFAULT_PROXIMITY_PX = 150
RETREAT_DISTANCE_M = 25

CLASS_COLORS = {
    WORKER_CLASS_NAME: (60, 200, 60),      # green, BGR
    FORKLIFT_CLASS_NAME: (40, 40, 230),    # red, BGR
}
DEFAULT_COLOR = (200, 200, 0)


st.set_page_config(page_title="ForeSite AI", layout="wide")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def init_session_state():
    defaults = {
        "log": [],
        "video_path": None,
        "cap": None,
        "fps": 25.0,
        "frame_idx": 0,
        "playing": False,
        "last_annotated": None,
        "active_pairs": set(),
        "proximity_px": DEFAULT_PROXIMITY_PX,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_timestamp(frame_idx: int, fps: float) -> str:
    seconds_total = frame_idx / fps if fps > 0 else 0
    minutes = int(seconds_total // 60)
    seconds = seconds_total % 60
    return f"{minutes:02d}:{seconds:04.1f}"


def init_video(path: str, display_name: str):
    old_cap = st.session_state.get("cap")
    if old_cap is not None:
        old_cap.release()

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:  # NaN guard
        fps = 25.0

    st.session_state.video_path = path
    st.session_state.cap = cap
    st.session_state.fps = fps
    st.session_state.frame_idx = 0
    st.session_state.playing = True
    st.session_state.last_annotated = None
    st.session_state.active_pairs = set()
    st.session_state.log.append(f"--- Loaded video: {display_name} ---")


def reset_video():
    cap = st.session_state.get("cap")
    if cap is not None:
        cap.release()
    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
        try:
            os.remove(st.session_state.video_path)
        except OSError:
            pass

    st.session_state.video_path = None
    st.session_state.cap = None
    st.session_state.frame_idx = 0
    st.session_state.playing = False
    st.session_state.last_annotated = None
    st.session_state.active_pairs = set()


def process_frame(model, frame: np.ndarray, frame_idx: int, fps: float, proximity_px: int) -> np.ndarray:
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=DETECT_CLASSES,
        conf=0.15,
        imgsz=640,
        verbose=False,
    )
    result = results[0]
    annotated = frame.copy()

    workers = []    # list of (track_id, center_xy)
    forklifts = []  # list of (track_id, center_xy)

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.arange(1, len(xyxy) + 1)
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            cls_id = int(cls_ids[i])
            track_id = int(track_ids[i])
            conf = float(confs[i])

            if cls_id == WORKER_CLASS_ID:
                label_name = WORKER_LABEL
                color = CLASS_COLORS[WORKER_CLASS_NAME]
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                workers.append((track_id, center))
            elif cls_id in FORKLIFT_CLASS_IDS:
                label_name = FORKLIFT_LABEL
                color = CLASS_COLORS[FORKLIFT_CLASS_NAME]
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                forklifts.append((track_id, center))
            else:
                continue

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            display_str = f"{label_name} #{track_id} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(display_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated, display_str, (x1 + 3, max(14, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
            )

    current_close_pairs = set()
    for w_id, w_center in workers:
        for f_id, f_center in forklifts:
            dist = float(np.hypot(w_center[0] - f_center[0], w_center[1] - f_center[1]))
            if dist <= proximity_px:
                current_close_pairs.add((w_id, f_id))
                # Draw hazard line between worker and forklift
                p1 = (int(w_center[0]), int(w_center[1]))
                p2 = (int(f_center[0]), int(f_center[1]))
                cv2.line(annotated, p1, p2, (0, 0, 255), 2, cv2.LINE_AA)
                mid_p = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(
                    annotated, f"⚠️ DANGER {int(dist)}px", mid_p,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
                )

    newly_close = current_close_pairs - st.session_state.active_pairs
    if newly_close:
        ts = format_timestamp(frame_idx, fps)
        for w_id, f_id in newly_close:
            st.session_state.log.append(
                f"[{ts}] RISK: {WORKER_LABEL} #{w_id} + {FORKLIFT_LABEL} #{f_id} "
                f"converging → Move forklift back by {RETREAT_DISTANCE_M}m"
            )
    st.session_state.active_pairs = current_close_pairs

    return annotated


def render_command_log(placeholder, log_lines):
    log_text = "\n".join(log_lines) if log_lines else "Waiting for events..."
    with placeholder.container():
        with st.container(height=520):
            st.code(log_text, language="text")


def main():
    init_session_state()
    model = load_model()

    st.title("\U0001F6A7 ForeSite AI")
    st.caption("Live worker/forklift proximity detection (MVP demo using stock YOLOv8 COCO classes)")

    col_left, col_right = st.columns([0.65, 0.35])

    with col_right:
        st.subheader("Command Log")
        log_placeholder = st.empty()
        render_command_log(log_placeholder, st.session_state.log)

    with col_left:
        st.subheader("Live Feed")

        if st.session_state.video_path is None:
            uploaded = st.file_uploader(
                "Upload a video", type=["mp4", "mov", "avi"], key="uploader"
            )
            if uploaded is not None:
                suffix = os.path.splitext(uploaded.name)[1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded.read())
                tmp.flush()
                tmp.close()
                init_video(tmp.name, uploaded.name)
                st.rerun()
        else:
            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
            with ctrl_col1:
                if st.session_state.playing:
                    if st.button("⏸ Pause"):
                        st.session_state.playing = False
                        st.rerun()
                else:
                    if st.button("▶ Play"):
                        st.session_state.playing = True
                        st.rerun()
            with ctrl_col2:
                if st.button("\U0001F504 New video"):
                    reset_video()
                    st.rerun()
            with ctrl_col3:
                st.session_state.proximity_px = st.slider(
                    "Proximity threshold (px)", 50, 400, st.session_state.proximity_px, 10
                )

            frame_placeholder = st.empty()

            cap = st.session_state.cap
            if cap is not None and st.session_state.playing:
                last_log_len = len(st.session_state.log)
                while st.session_state.playing and cap is not None:
                    t_start = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        # loop back to the start of the video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        st.session_state.active_pairs = set()
                        ret, frame = cap.read()
                        if not ret:
                            break

                    st.session_state.frame_idx += 1
                    annotated = process_frame(
                        model, frame, st.session_state.frame_idx,
                        st.session_state.fps, st.session_state.proximity_px,
                    )
                    st.session_state.last_annotated = annotated
                    frame_placeholder.image(annotated, channels="BGR", use_column_width=True)

                    if len(st.session_state.log) != last_log_len:
                        render_command_log(log_placeholder, st.session_state.log)
                        last_log_len = len(st.session_state.log)

                    t_elapsed = time.time() - t_start
                    target_delay = 1.0 / st.session_state.fps
                    time.sleep(max(0.001, target_delay - t_elapsed))
            elif st.session_state.last_annotated is not None:
                frame_placeholder.image(st.session_state.last_annotated, channels="BGR", use_column_width=True)
            else:
                frame_placeholder.info("Loading video...")


if __name__ == "__main__":
    main()
