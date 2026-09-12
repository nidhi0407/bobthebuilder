"""
ForeSite AI — Person 1: CV & Tracking Main Pipeline Runner
Executes: Video Ingestion -> Detection -> ByteTrack -> Position Extraction ->
          Trajectory Buffering -> Visualization -> Structured Export (JSON/CSV).

Usage:
    python -m cv_tracking.run_cv_pipeline --video data/demo_site.mp4
    python -m cv_tracking.run_cv_pipeline --generate-demo
"""

import os
import sys
import argparse
import time
import logging
from typing import Optional
import cv2
import numpy as np

from shared.schemas import FrameTracks
from .config import TrackingConfig
from .video_reader import VideoReader, VideoReaderError
from .detector import create_detector, BaseDetector
from .tracker import ByteTracker
from .trajectory_buffer import TrajectoryBuffer
from .visualize import Visualizer
from .exporter import TrajectoryExporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_demo_video(output_path: str = "data/demo_site.mp4", num_frames: int = 120) -> str:
    """
    Generate a high-quality simulated construction site video clip for standalone testing.
    Features:
    - Textured construction site ground with safety zone markings
    - Worker in hi-vis vest walking towards center
    - Forklift vehicle approaching on converging trajectory
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    w, h = 1280, 720
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    logger.info(f"Generating synthetic construction site demo clip: {output_path} ({num_frames} frames)...")

    for i in range(num_frames):
        # Progress t in [0.0, 1.0]
        t = i / float(num_frames)

        # 1. Base construction ground (dusty concrete / dirt tone)
        frame = np.full((h, w, 3), (60, 70, 75), dtype=np.uint8)

        # Construction grid lines / texture
        for gx in range(0, w, 80):
            cv2.line(frame, (gx, 0), (gx, h), (70, 80, 85), 1)
        for gy in range(0, h, 80):
            cv2.line(frame, (0, gy), (w, gy), (70, 80, 85), 1)

        # Safety hazard zone outline (yellow hatched boundary)
        cv2.rectangle(frame, (450, 250), (850, 520), (0, 180, 220), 2)
        cv2.putText(frame, "ACTIVE EQUIPMENT ZONE", (460, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 220), 1)

        # 2. Worker walking from left (x=200, y=500) towards center (x=620, y=390)
        wx = int(round(200 + 420 * t))
        wy = int(round(500 - 110 * t))
        # Draw worker: legs, hi-vis torso, hard hat
        cv2.line(frame, (wx - 8, wy), (wx - 4, wy - 30), (30, 30, 30), 4)  # Left leg
        cv2.line(frame, (wx + 8, wy), (wx + 4, wy - 30), (30, 30, 30), 4)  # Right leg
        cv2.rectangle(frame, (wx - 14, wy - 65), (wx + 14, wy - 30), (0, 215, 255), -1)  # Hi-vis vest
        cv2.circle(frame, (wx, wy - 75), 10, (180, 190, 200), -1)  # Head
        cv2.ellipse(frame, (wx, wy - 82), (13, 6), 0, 180, 360, (0, 255, 255), -1)  # Yellow hardhat

        # 3. Forklift vehicle approaching from right (x=1050, y=350) towards center (x=680, y=410)
        fx = int(round(1050 - 370 * t))
        fy = int(round(350 + 60 * t))
        # Forklift chassis (industrial yellow)
        cv2.rectangle(frame, (fx - 45, fy - 60), (fx + 45, fy - 15), (20, 180, 210), -1)
        # Cabin frame
        cv2.rectangle(frame, (fx - 25, fy - 110), (fx + 25, fy - 60), (40, 40, 40), 3)
        # Heavy wheels
        cv2.circle(frame, (fx - 30, fy - 12), 16, (20, 20, 20), -1)
        cv2.circle(frame, (fx + 30, fy - 12), 16, (20, 20, 20), -1)
        # Fork mast at front
        cv2.line(frame, (fx - 45, fy - 120), (fx - 45, fy), (50, 50, 50), 5)
        cv2.line(frame, (fx - 70, fy - 10), (fx - 45, fy - 10), (50, 50, 50), 5)  # Forks

        out.write(frame)

    out.release()
    logger.info(f"Synthetic demo clip created at: {output_path}")
    return output_path


def run_pipeline(
    video_path: Optional[str] = None,
    config: Optional[TrackingConfig] = None,
    save_video: bool = False,
    headless: bool = True,
    max_frames: Optional[int] = None,
    force_synthetic: bool = False
) -> dict:
    """
    Execute the ForeSite Person 1 CV & Tracking Pipeline.

    Returns:
        Summary dict containing statistics and paths to exported artifacts.
    """
    cfg = config or TrackingConfig()

    # If no video path provided, generate and use synthetic demo video
    if not video_path:
        demo_video_path = os.path.join("data", "demo_site.mp4")
        if not os.path.exists(demo_video_path):
            generate_synthetic_demo_video(demo_video_path)
        video_path = demo_video_path

    # Initialize Stage 1: Video Ingestion
    try:
        reader = VideoReader(video_path, resize_width=cfg.resize_width)
    except VideoReaderError as e:
        logger.error(f"Video Ingestion Error: {e}")
        sys.exit(1)

    logger.info(
        f"Video loaded: {video_path} | Resolution: {reader.frame_width}x{reader.frame_height} | "
        f"FPS: {reader.fps:.1f} | Total Frames: {reader.total_frames}"
    )

    # Initialize Modules
    detector: BaseDetector = create_detector(cfg, force_synthetic=force_synthetic)
    tracker = ByteTracker(
        track_thresh=cfg.track_thresh,
        match_thresh=cfg.match_thresh,
        max_time_lost=cfg.track_buffer
    )
    buffer = TrajectoryBuffer(
        history_length=cfg.history_length,
        expiry_seconds=cfg.track_expiry_seconds,
        enable_smoothing=cfg.enable_smoothing,
        smoothing_alpha=cfg.smoothing_alpha
    )
    visualizer = Visualizer(cfg)
    exporter = TrajectoryExporter(output_dir=cfg.export_dir)

    # Video Writer if save_video is enabled
    video_writer = None
    output_video_path = None
    if save_video:
        output_video_path = os.path.join(cfg.export_dir, "annotated_feed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_video_path, fourcc, reader.fps, (reader.frame_width, reader.frame_height)
        )

    processed_frames = 0
    unique_tracks_seen = set()
    start_time = time.time()

    try:
        for frame, frame_idx, timestamp in reader:
            if max_frames and processed_frames >= max_frames:
                break

            # Frame skipping if configured for extreme real-time speed
            if cfg.process_every_n_frames > 1 and (frame_idx % cfg.process_every_n_frames != 0):
                continue

            # Stage 2: Detection
            detections = detector.detect(frame)

            # Stage 3: ByteTrack Multi-Object Association
            active_tracks = tracker.update(
                detections=detections,
                frame_width=reader.frame_width,
                frame_height=reader.frame_height,
                timestamp=timestamp,
                anchor=cfg.worker_position_anchor
            )

            # Stage 4 & 5: Trajectory Buffering and Coordinate History
            for track in active_tracks:
                unique_tracks_seen.add(track.track_id)
                # Update buffer with normalized ground position
                buffer.update(
                    track_id=track.track_id,
                    class_name=track.class_name,
                    x_norm=track.position[0],
                    y_norm=track.position[1],
                    timestamp=track.timestamp
                )
                # Record to CSV exporter
                exporter.record_observation(
                    timestamp=track.timestamp,
                    track_id=track.track_id,
                    class_name=track.class_name,
                    x_norm=track.position[0],
                    y_norm=track.position[1]
                )

            # Remove expired tracks that haven't been observed for > expiry_seconds
            buffer.cleanup_expired(timestamp)

            # Record per-frame snapshot for Person 4 / dashboard
            frame_snapshot = FrameTracks(frame_index=frame_idx, timestamp=timestamp, tracks=active_tracks)
            exporter.record_frame(frame_snapshot)

            # Stage 8: Visualization
            annotated_frame = visualizer.annotate(
                frame=frame,
                tracks=active_tracks,
                buffer=buffer,
                frame_index=frame_idx,
                timestamp=timestamp,
                fps=reader.fps
            )

            if video_writer:
                video_writer.write(annotated_frame)

            if not headless:
                cv2.imshow("ForeSite AI — Person 1 CV Perception", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Pipeline stopped by user keypress.")
                    break

            processed_frames += 1

    finally:
        reader.release()
        if video_writer:
            video_writer.release()
        if not headless:
            cv2.destroyAllWindows()

    elapsed = max(0.001, time.time() - start_time)
    effective_fps = processed_frames / elapsed

    # Stage 7: File Exports
    csv_path = exporter.export_csv(cfg.default_csv_filename)
    all_trajectories = buffer.get_all_trajectories()
    json_path = exporter.export_trajectories_json(all_trajectories, cfg.default_json_filename)
    frame_json_path = exporter.export_frames_json("frame_tracks.json")

    # Also generate mock trajectories in data/ for Person 2
    mock_json_path = TrajectoryExporter.generate_mock_trajectories("data/mock_trajectories.json")

    summary = {
        "processed_frames": processed_frames,
        "elapsed_seconds": round(elapsed, 3),
        "effective_fps": round(effective_fps, 1),
        "unique_tracks_count": len(unique_tracks_seen),
        "csv_export": csv_path,
        "json_trajectories_export": json_path,
        "frame_tracks_export": frame_json_path,
        "mock_trajectories_export": mock_json_path,
        "annotated_video": output_video_path,
    }

    logger.info("=" * 60)
    logger.info("ForeSite AI CV Pipeline Completed Successfully!")
    logger.info(f"Processed {processed_frames} frames in {elapsed:.2f}s ({effective_fps:.1f} FPS)")
    logger.info(f"Unique tracks tracked: {len(unique_tracks_seen)}")
    logger.info(f"JSON trajectory output (for Person 2): {json_path}")
    logger.info(f"CSV trajectory output: {csv_path}")
    logger.info(f"Frame tracks output (for Person 4): {frame_json_path}")
    if output_video_path:
        logger.info(f"Annotated video output: {output_video_path}")
    logger.info("=" * 60)

    return summary


def main():
    parser = argparse.ArgumentParser(description="ForeSite AI — Person 1: CV & Tracking Pipeline")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file or RTSP stream")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model weights (.pt or .onnx)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--history-len", type=int, default=8, help="Trajectory history buffer length (must match GRU model's obs_len=8)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save exported results")
    parser.add_argument("--save-video", action="store_true", help="Save annotated output video")
    parser.add_argument("--headless", action="store_true", default=True, help="Run without UI window")
    parser.add_argument("--gui", dest="headless", action="store_false", help="Enable OpenCV GUI display window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--generate-demo", action="store_true", help="Generate synthetic demo video and run pipeline")
    parser.add_argument("--force-synthetic", action="store_true", help="Force synthetic fallback detector")

    args = parser.parse_args()

    config = TrackingConfig(
        model_name_or_path=args.model,
        conf_threshold=args.conf,
        history_length=args.history_len,
        export_dir=args.output_dir
    )

    if args.generate_demo:
        demo_path = os.path.join("data", "demo_site.mp4")
        generate_synthetic_demo_video(demo_path)
        args.video = demo_path

    run_pipeline(
        video_path=args.video,
        config=config,
        save_video=args.save_video,
        headless=args.headless,
        max_frames=args.max_frames,
        force_synthetic=args.force_synthetic
    )


if __name__ == "__main__":
    main()
