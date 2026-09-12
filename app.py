"""
ForeSite AI — Live Construction Safety & Collision Prevention System.

Complete 5-Stage ML Pipeline:
1. Stage 1 & 2: Object Detection & ByteTrack Tracking
2. Stage 3: Ground-Plane Position Anchoring & History Buffering (8 frames)
3. Stage 4: Trajectory Forecasting via TrajectoryGRU (12 future steps)
4. Stage 5: Spatial-Temporal Risk Engine (CPA & TTC Calculations)
5. Stage 6a: Counterfactual Simulation (Prescriptive Intervention Selection)
6. Stage 6b: Asynchronous Alert Dispatch to Alert Server & Multi-Role HUDs
"""

import os
import sys
import time
import tempfile
import threading
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import cv2
import numpy as np
import requests
import torch
import streamlit as st

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cv_tracking.config import TrackingConfig
from cv_tracking.detector import create_detector
from cv_tracking.tracker import ByteTracker
from model.trajectory_model import TrajectoryConfig, TrajectoryGRU
from model.risk_engine import RiskEngine, RiskConfig, PairRiskResult
from model.counterfactual_simulation import CounterfactualSimulator, InterventionRecommendation

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = BASE_DIR / "model" / "checkpoints" / "gru_trajectory_best.pt"
ALERT_SERVER_URL = "http://127.0.0.1:5000/api/alert"
CLEAR_SERVER_URL = "http://127.0.0.1:5000/api/clear"

COLOR_WORKER = (76, 175, 80)     # Emerald Green (BGR)
COLOR_FORKLIFT = (33, 150, 243)  # Bright Orange/Blue (BGR)
COLOR_HAZARD = (48, 59, 255)     # Bright Crimson Red (BGR)
COLOR_SAFE_PLAN = (238, 238, 0)  # Cyan (BGR)

st.set_page_config(
    page_title="ForeSite AI — Construction Safety",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Background Alert Dispatcher (Non-Blocking)
# ---------------------------------------------------------------------------
class AsyncAlertDispatcher:
    """Dispatches HTTP alert triggers in a non-blocking daemon thread with debouncing."""
    def __init__(self, debounce_seconds: float = 3.0):
        self.debounce_seconds = debounce_seconds
        self.last_sent_time = 0.0
        self.last_active_state = False

    def trigger_async(self, worker_msg: str, forklift_msg: str):
        now = time.time()
        if now - self.last_sent_time < self.debounce_seconds:
            return

        self.last_sent_time = now
        self.last_active_state = True

        def _worker():
            try:
                requests.post(
                    ALERT_SERVER_URL,
                    json={"worker_message": worker_msg, "forklift_message": forklift_msg},
                    timeout=1.5,
                )
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def clear_async(self):
        if not self.last_active_state:
            return
        self.last_active_state = False

        def _worker():
            try:
                requests.post(CLEAR_SERVER_URL, json={}, timeout=1.5)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()


alert_dispatcher = AsyncAlertDispatcher()


# ---------------------------------------------------------------------------
# Model Caching & Loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Loads detector, tracker, Trajectory GRU model, Risk Engine, and Counterfactual Simulator."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Perception Detector
    tracking_cfg = TrackingConfig(conf_threshold=0.25)
    detector = create_detector(tracking_cfg)

    # 2. GRU Trajectory Model
    gru_model = None
    if CHECKPOINT_PATH.exists():
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
            cfg = ckpt.get("config", TrajectoryConfig())
            gru_model = TrajectoryGRU(cfg).to(device)
            gru_model.load_state_dict(ckpt["model_state_dict"])
            gru_model.eval()
        except Exception as e:
            st.warning(f"Note: Loaded default GRU weights: {e}")

    if gru_model is None:
        cfg = TrajectoryConfig()
        gru_model = TrajectoryGRU(cfg).to(device)
        gru_model.eval()

    # 3. Risk Engine
    risk_engine = RiskEngine(RiskConfig(cpa_threshold=0.08, ttc_threshold=3.5, high_risk_threshold=0.60))

    # 4. Counterfactual Simulator
    cf_simulator = CounterfactualSimulator(risk_engine=risk_engine)

    return detector, gru_model, risk_engine, cf_simulator, device, tracking_cfg


# ---------------------------------------------------------------------------
# Session State Management
# ---------------------------------------------------------------------------
def init_session_state(tracking_cfg: TrackingConfig):
    defaults = {
        "log": [],
        "video_path": None,
        "cap": None,
        "fps": 30.0,
        "frame_idx": 0,
        "playing": False,
        "last_annotated": None,
        "active_hazard": None,
        "tracker": ByteTracker(tracking_cfg),
        "track_histories": defaultdict(lambda: deque(maxlen=8)),
        "risk_threshold": 0.60,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def format_timestamp(frame_idx: int, fps: float) -> str:
    seconds_total = frame_idx / fps if fps > 0 else 0
    minutes = int(seconds_total // 60)
    seconds = seconds_total % 60
    return f"{minutes:02d}:{seconds:04.1f}"


def init_video(path: str, display_name: str, tracking_cfg: TrackingConfig):
    if st.session_state.get("cap") is not None:
        st.session_state.cap.release()

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps != fps:
        fps = 30.0

    st.session_state.video_path = path
    st.session_state.cap = cap
    st.session_state.fps = fps
    st.session_state.frame_idx = 0
    st.session_state.playing = True
    st.session_state.last_annotated = None
    st.session_state.tracker = ByteTracker(tracking_cfg)
    st.session_state.track_histories = defaultdict(lambda: deque(maxlen=8))
    st.session_state.log.append(f"🟢 Loaded Video: {display_name} ({fps:.1f} FPS)")


def reset_video(tracking_cfg: TrackingConfig):
    if st.session_state.get("cap") is not None:
        st.session_state.cap.release()
    st.session_state.video_path = None
    st.session_state.cap = None
    st.session_state.frame_idx = 0
    st.session_state.playing = False
    st.session_state.last_annotated = None
    st.session_state.tracker = ByteTracker(tracking_cfg)
    st.session_state.track_histories = defaultdict(lambda: deque(maxlen=8))
    alert_dispatcher.clear_async()


# ---------------------------------------------------------------------------
# Core ML Frame Processing Pipeline
# ---------------------------------------------------------------------------
def process_pipeline_frame(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    detector,
    gru_model,
    risk_engine: RiskEngine,
    cf_simulator: CounterfactualSimulator,
    device: torch.device,
    risk_threshold: float,
) -> Tuple[np.ndarray, List[str]]:
    h, w = frame.shape[:2]
    annotated = frame.copy()
    new_logs = []

    # 1. Detection & Tracking (Stage 1 & 2)
    detections = detector.detect(frame)
    tracker: ByteTracker = st.session_state.tracker
    tracked_objects = tracker.update(detections, frame_id=frame_idx, frame_shape=(h, w))

    active_tracks: Dict[int, Tuple[str, Tuple[float, float], Tuple[int, int, int, int]]] = {}
    class_map: Dict[int, str] = {}

    for t in tracked_objects:
        tid = t.track_id
        c_name = t.class_name
        x1, y1, x2, y2 = t.bbox
        pos_x, pos_y = t.position  # Normalized [0.0, 1.0]

        active_tracks[tid] = (c_name, (pos_x, pos_y), (int(x1), int(y1), int(x2), int(y2)))
        class_map[tid] = c_name

        # Push to sliding history buffer
        st.session_state.track_histories[tid].append([pos_x, pos_y])

    # 2. Stage 4: Trajectory Forecasting via GRU
    predictions: Dict[int, np.ndarray] = {}
    if active_tracks:
        obs_batch = []
        cls_batch = []
        valid_tids = []

        for tid, (c_name, (cx_norm, cy_norm), _) in active_tracks.items():
            hist = list(st.session_state.track_histories[tid])
            # If fewer than 8 observations, pad with earliest available position
            while len(hist) < 8:
                hist.insert(0, hist[0] if hist else [cx_norm, cy_norm])

            obs_batch.append(hist[-8:])
            cls_batch.append(0 if "worker" in c_name.lower() else 1)
            valid_tids.append(tid)

        if obs_batch:
            obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=device)
            cls_tensor = torch.tensor(cls_batch, dtype=torch.long, device=device)
            with torch.no_grad():
                preds_tensor = gru_model(obs_tensor, cls_tensor)
                preds_np = preds_tensor.cpu().numpy()

            for idx, tid in enumerate(valid_tids):
                predictions[tid] = preds_np[idx]  # Shape: (12, 2)

    # 3. Stage 5: Risk Engine CPA / TTC Calculations & Stage 6a Counterfactuals
    danger_recommendations: List[InterventionRecommendation] = []
    max_risk = 0.0

    if predictions and len(predictions) >= 2:
        danger_recommendations = cf_simulator.evaluate_all_pairs(
            predictions=predictions,
            class_map=class_map,
        )

    # 4. Drawing & Visual Overlays
    # 4a. Draw History Tails & Forecasted Trajectories
    for tid, (c_name, (cx, cy), (x1, y1, x2, y2)) in active_tracks.items():
        base_color = COLOR_WORKER if "worker" in c_name.lower() else COLOR_FORKLIFT

        # Bounding Box + Label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), base_color, 2)
        label_text = f"{c_name.upper()} #{tid}"
        cv2.putText(annotated, label_text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Historical Trail
        hist = list(st.session_state.track_histories[tid])
        for pt_i in range(len(hist) - 1):
            px1, py1 = int(hist[pt_i][0] * w), int(hist[pt_i][1] * h)
            px2, py2 = int(hist[pt_i + 1][0] * w), int(hist[pt_i + 1][1] * h)
            alpha = (pt_i + 1) / len(hist)
            trail_color = tuple(int(c * alpha) for c in base_color)
            cv2.line(annotated, (px1, py1), (px2, py2), trail_color, 2, cv2.LINE_AA)

        # Forecasted 12-step Trajectory (Dotted line with forward arrow)
        if tid in predictions:
            fut = predictions[tid]
            pts = [(int(pt[0] * w), int(pt[1] * h)) for pt in fut]
            for step_i in range(len(pts) - 1):
                cv2.line(annotated, pts[step_i], pts[step_i + 1], base_color, 2, cv2.LINE_AA)
                cv2.circle(annotated, pts[step_i + 1], 2, (255, 255, 255), -1)
            if len(pts) > 1:
                cv2.arrowedLine(annotated, pts[-2], pts[-1], base_color, 2, tipLength=0.3)

    # 4b. Draw Hazardous Intersections & Counterfactual Prescriptions
    top_hud_message = None
    if danger_recommendations:
        ts = format_timestamp(frame_idx, fps)
        for rec in danger_recommendations:
            if rec.baseline_risk_score > max_risk:
                max_risk = rec.baseline_risk_score

            w_tid, m_tid = rec.worker_id, rec.machine_id
            if w_tid in active_tracks and m_tid in active_tracks:
                w_pos = active_tracks[w_tid][1]
                m_pos = active_tracks[m_tid][1]

                wp_px = (int(w_pos[0] * w), int(w_pos[1] * h))
                mp_px = (int(m_pos[0] * w), int(m_pos[1] * h))

                # Red Hazard Line
                cv2.line(annotated, wp_px, mp_px, COLOR_HAZARD, 3, cv2.LINE_AA)
                mid_px = ((wp_px[0] + mp_px[0]) // 2, (wp_px[1] + mp_px[1]) // 2)
                cv2.putText(
                    annotated,
                    f"🚨 RISK {rec.baseline_risk_score:.2f}",
                    (mid_px[0] - 50, mid_px[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_HAZARD,
                    2,
                    cv2.LINE_AA,
                )

                # Draw Counterfactual Safe Trajectory
                if rec.best_candidate is not None:
                    safe_pts = [(int(pt[0] * w), int(pt[1] * h)) for pt in rec.best_candidate.modified_machine_traj]
                    for si in range(len(safe_pts) - 1):
                        cv2.line(annotated, safe_pts[si], safe_pts[si + 1], COLOR_SAFE_PLAN, 2, cv2.LINE_AA)

                top_hud_message = f"PRESCRIPTIVE ACTION: {rec.best_candidate.description.upper()}"
                new_logs.append(
                    f"[{ts}] 🚨 HAZARD: Worker #{w_tid} ↔ Machine #{m_tid} (Risk: {rec.baseline_risk_score:.2f}) "
                    f"→ Prescribed: {rec.best_candidate.description}"
                )

                # Dispatch live audio & HUD triggers
                alert_dispatcher.trigger_async(
                    worker_msg=rec.worker_instruction,
                    forklift_msg=rec.machine_instruction,
                )
    else:
        alert_dispatcher.clear_async()

    # 4c. Render Top Prescriptive HUD Bar
    if top_hud_message:
        cv2.rectangle(annotated, (0, 0), (w, 42), (20, 20, 180), -1)
        cv2.putText(
            annotated,
            top_hud_message,
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated, new_logs


# ---------------------------------------------------------------------------
# Streamlit UI Dashboard
# ---------------------------------------------------------------------------
def render_command_log(placeholder, log_lines):
    log_text = "\n".join(log_lines[-25:]) if log_lines else "🟢 System armed. Monitoring spatial corridors..."
    with placeholder.container():
        with st.container(height=540):
            st.code(log_text, language="text")


def main():
    detector, gru_model, risk_engine, cf_simulator, device, tracking_cfg = load_models()
    init_session_state(tracking_cfg)

    # Header
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
            <h1 style="margin: 0;">🚧 ForeSite AI</h1>
            <span style="background-color: #2e7d32; color: white; padding: 4px 12px; border-radius: 16px; font-size: 14px; font-weight: bold;">
                GRU + Risk Engine + Counterfactuals ACTIVE
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Predictive Safety & Prescriptive Collision Prevention Platform")

    col_main, col_log = st.columns([0.65, 0.35])

    with col_log:
        st.subheader("📋 Prescriptive Event Log")
        log_placeholder = st.empty()
        render_command_log(log_placeholder, st.session_state.log)

        st.markdown("---")
        st.markdown(
            """
            **Connected Services**:
            - 📡 **Alert Server**: [http://127.0.0.1:5000](http://127.0.0.1:5000)
            - 👷 **Worker HUD**: [http://127.0.0.1:5000/worker](http://127.0.0.1:5000/worker)
            - 🚜 **Forklift HUD**: [http://127.0.0.1:5000/forklift](http://127.0.0.1:5000/forklift)
            """
        )

    with col_main:
        st.subheader("📹 Live Video Feed & Trajectory Forecast")

        if st.session_state.video_path is None:
            uploaded = st.file_uploader("Upload CCTV Video (.mp4, .mov, .avi)", type=["mp4", "mov", "avi"])
            if uploaded is not None:
                suffix = os.path.splitext(uploaded.name)[1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded.read())
                tmp.flush()
                tmp.close()
                init_video(tmp.name, uploaded.name, tracking_cfg)
                st.rerun()
        else:
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                if st.session_state.playing:
                    if st.button("⏸ Pause", use_container_width=True):
                        st.session_state.playing = False
                        st.rerun()
                else:
                    if st.button("▶ Play", use_container_width=True):
                        st.session_state.playing = True
                        st.rerun()
            with c2:
                if st.button("🔄 New Video", use_container_width=True):
                    reset_video(tracking_cfg)
                    st.rerun()
            with c3:
                st.session_state.risk_threshold = st.slider(
                    "Risk Trigger Sensitivity", 0.30, 0.90, st.session_state.risk_threshold, 0.05
                )

            frame_placeholder = st.empty()
            cap = st.session_state.cap

            if cap is not None and st.session_state.playing:
                last_log_len = len(st.session_state.log)
                while st.session_state.playing and cap is not None:
                    t_start = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        # Loop video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            break

                    st.session_state.frame_idx += 1
                    annotated, new_logs = process_pipeline_frame(
                        frame=frame,
                        frame_idx=st.session_state.frame_idx,
                        fps=st.session_state.fps,
                        detector=detector,
                        gru_model=gru_model,
                        risk_engine=risk_engine,
                        cf_simulator=cf_simulator,
                        device=device,
                        risk_threshold=st.session_state.risk_threshold,
                    )

                    st.session_state.last_annotated = annotated
                    frame_placeholder.image(annotated, channels="BGR", use_container_width=True)

                    if new_logs:
                        st.session_state.log.extend(new_logs)
                        if len(st.session_state.log) != last_log_len:
                            render_command_log(log_placeholder, st.session_state.log)
                            last_log_len = len(st.session_state.log)

                    t_elapsed = time.time() - t_start
                    target_delay = 1.0 / st.session_state.fps
                    time.sleep(max(0.001, target_delay - t_elapsed))
            elif st.session_state.last_annotated is not None:
                frame_placeholder.image(st.session_state.last_annotated, channels="BGR", use_container_width=True)
            else:
                frame_placeholder.info("Ready. Press Play to start video processing.")


if __name__ == "__main__":
    main()
