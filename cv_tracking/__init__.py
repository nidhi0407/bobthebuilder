"""
ForeSite AI — Person 1: Computer Vision & Tracking Module
Provides real-time video ingestion, YOLO detection, ByteTrack multi-object tracking,
coordinate normalization, trajectory-history buffering, visualization, and export.
"""

from .config import TrackingConfig
from .position import extract_representative_position, normalize_coordinates, denormalize_coordinates
from .trajectory_buffer import TrajectoryBuffer
from .exporter import TrajectoryExporter

__all__ = [
    "TrackingConfig",
    "extract_representative_position",
    "normalize_coordinates",
    "denormalize_coordinates",
    "TrajectoryBuffer",
    "TrajectoryExporter",
]
