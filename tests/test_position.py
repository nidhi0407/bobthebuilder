"""
Tests for position extraction and coordinate normalization.
"""

import pytest
from cv_tracking.position import (
    extract_representative_position,
    normalize_coordinates,
    denormalize_coordinates,
)


def test_bottom_center_worker_position():
    # Bbox: x1=100, y1=200, x2=200, y2=400
    bbox = [100, 200, 200, 400]
    x, y = extract_representative_position(bbox, anchor="bottom-center")
    # x = (100 + 200) / 2 = 150
    # y = y2 = 400
    assert x == 150.0
    assert y == 400.0


def test_center_position():
    bbox = [100, 200, 200, 400]
    x, y = extract_representative_position(bbox, anchor="center")
    # x = 150, y = (200 + 400) / 2 = 300
    assert x == 150.0
    assert y == 300.0


def test_normalize_coordinates():
    # 1920x1080 frame
    frame_width = 1920
    frame_height = 1080

    x_norm, y_norm = normalize_coordinates(960, 540, frame_width, frame_height)
    assert pytest.approx(x_norm, 0.001) == 0.5
    assert pytest.approx(y_norm, 0.001) == 0.5


def test_normalize_clamping():
    # Points slightly outside bounds clamped to [0.0, 1.0]
    x_norm, y_norm = normalize_coordinates(-10, 1200, 1000, 1000, clamp=True)
    assert x_norm == 0.0
    assert y_norm == 1.0


def test_denormalize_coordinates():
    x_pix, y_pix = denormalize_coordinates(0.5, 0.5, 1920, 1080)
    assert x_pix == 960
    assert y_pix == 540


def test_invalid_arguments():
    with pytest.raises(ValueError):
        extract_representative_position([100, 200])  # Too short

    with pytest.raises(ValueError):
        extract_representative_position([100, 200, 300, 400], anchor="unknown")

    with pytest.raises(ValueError):
        normalize_coordinates(100, 100, 0, 1080)
