"""
Configuration for ForeSite CV & Tracking Module.
Centralizes all hyperparameters, model settings, class mappings, and path defaults.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Tuple


@dataclass
class TrackingConfig:
    """Master configuration for perception and tracking pipeline."""

    # Model parameters
    model_name_or_path: str = "yolov8n.pt"  # Lightweight pretrained YOLO
    device: Optional[str] = None  # Auto-selects mps/cuda/cpu if None
    conf_threshold: float = 0.35  # Detection confidence threshold
    iou_threshold: float = 0.45   # NMS IoU threshold

    # Class mappings (Source detector label -> Canonical ForeSite label)
    class_mapping: Dict[str, str] = field(default_factory=lambda: {
        "person": "worker",
        "forklift": "forklift",
        "truck": "truck",
        "bus": "truck",
        "excavator": "excavator",
        "loader": "loader",
    })

    # Allowed canonical target classes to pass downstream (filters out background clutter)
    target_classes: Set[str] = field(default_factory=lambda: {
        "worker",
        "forklift",
        "truck",
        "excavator",
        "loader",
    })

    # ByteTrack Tracker Settings
    tracker_type: str = "bytetrack.yaml"
    track_thresh: float = 0.35
    match_thresh: float = 0.40
    track_buffer: int = 30  # Frame buffer inside tracker

    # Trajectory Buffer & Timeouts
    history_length: int = 8        # Last N observations handed to Person 2 (must match GRU model's obs_len=8)
    track_expiry_seconds: float = 1.5  # Grace period before dropping temporarily lost track

    # Position Extraction
    # "bottom-center": x=(x1+x2)/2, y=y2 (optimal for ground-plane contact in image plane)
    # "center": x=(x1+x2)/2, y=(y1+y2)/2
    worker_position_anchor: str = "bottom-center"
    machinery_position_anchor: str = "bottom-center"

    # Smoothing (avoids bounding box jitter while maintaining responsive motion)
    enable_smoothing: bool = True
    smoothing_alpha: float = 0.6  # Exponential smoothing weight: x_t = alpha * new + (1-alpha) * prev

    # Performance / Latency optimizations
    process_every_n_frames: int = 1  # 1 = process all frames; 2 = process every second frame
    resize_width: Optional[int] = None   # Optional downscale width for faster inference (e.g. 1280 or 960)

    # Visualization styling
    trail_length: int = 12
    draw_boxes: bool = True
    draw_ids: bool = True
    draw_trails: bool = True
    draw_positions: bool = True

    # Color scheme (BGR for OpenCV)
    color_palette: Dict[str, Tuple[int, int, int]] = field(default_factory=lambda: {
        "worker": (0, 215, 255),    # Vibrant Amber/Yellow
        "forklift": (50, 205, 50),   # Lime Green
        "truck": (255, 140, 0),      # Deep Orange/Blue in BGR: (0, 140, 255) -> (255, 140, 0)
        "excavator": (255, 105, 180),# Hot Pink
        "loader": (180, 105, 255),   # Purple
        "default": (200, 200, 200),  # Light Grey
    })

    # Default export directories
    export_dir: str = "output"
    default_json_filename: str = "trajectories.json"
    default_csv_filename: str = "trajectories.csv"
