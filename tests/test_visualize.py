"""
Tests for visualizer module.
Verifies overlay rendering, trails, badges, and HUD.
"""

import numpy as np
from cv_tracking.config import TrackingConfig
from cv_tracking.visualize import Visualizer
from cv_tracking.trajectory_buffer import TrajectoryBuffer
from shared.schemas import Track


def test_visualizer_renders():
    config = TrackingConfig()
    vis = Visualizer(config)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    buffer = TrajectoryBuffer(history_length=5)
    # Add history
    buffer.update(1, "worker", 0.3, 0.6, 0.0)
    buffer.update(1, "worker", 0.31, 0.61, 0.1)

    tracks = [
        Track(
            track_id=1,
            class_name="worker",
            bbox=[384, 432, 448, 576],
            position=[0.31, 0.61],
            timestamp=0.1,
            confidence=0.92
        )
    ]

    annotated = vis.annotate(
        frame=frame,
        tracks=tracks,
        buffer=buffer,
        frame_index=1,
        timestamp=0.1,
        fps=30.0
    )

    assert annotated.shape == (720, 1280, 3)
    assert annotated.dtype == np.uint8
    # Frame should no longer be purely zeros
    assert np.any(annotated > 0)
