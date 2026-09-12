"""
Tests for object detector and class label normalization.
"""

import numpy as np
from cv_tracking.config import TrackingConfig
from cv_tracking.detector import BaseDetector, SyntheticFallbackDetector, create_detector


def test_label_normalization():
    config = TrackingConfig()
    detector = SyntheticFallbackDetector(config)

    # Allowed & mapped
    assert detector.normalize_label("person") == "worker"
    assert detector.normalize_label("forklift") == "forklift"
    assert detector.normalize_label("truck") == "truck"
    assert detector.normalize_label("bus") == "truck"

    # Ignored clutter
    assert detector.normalize_label("chair") is None
    assert detector.normalize_label("bottle") is None
    assert detector.normalize_label("dining table") is None


def test_synthetic_detector_output_structure():
    config = TrackingConfig()
    detector = SyntheticFallbackDetector(config)

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)

    assert len(detections) == 2
    for det in detections:
        assert "class_name" in det
        assert "bbox" in det
        assert "confidence" in det
        assert len(det["bbox"]) == 4
        assert det["confidence"] > 0.5
        # Verify bounding box validity: x2 > x1, y2 > y1
        x1, y1, x2, y2 = det["bbox"]
        assert x2 > x1
        assert y2 > y1
