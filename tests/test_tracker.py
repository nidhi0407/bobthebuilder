"""
Tests for ByteTrack multi-object tracker.
Verifies ID persistence, IoU matching, and Track schema output.
"""

from cv_tracking.tracker import ByteTracker, compute_iou
from shared.schemas import Track


def test_compute_iou():
    # Identical boxes
    b1 = [0, 0, 10, 10]
    b2 = [0, 0, 10, 10]
    assert compute_iou(b1, b2) == 1.0

    # Non-overlapping
    b3 = [20, 20, 30, 30]
    assert compute_iou(b1, b3) == 0.0

    # 50% overlap horizontally
    b4 = [0, 0, 5, 10]
    assert compute_iou(b1, b4) == 0.5


def test_persistent_ids_across_frames():
    tracker = ByteTracker(track_thresh=0.35, match_thresh=0.4)
    tracker.reset()

    frame_w = 1000
    frame_h = 1000

    # Frame 1: Worker at [100, 200, 150, 300], Forklift at [600, 400, 700, 500]
    dets_frame1 = [
        {"bbox": [100, 200, 150, 300], "confidence": 0.92, "class_name": "worker"},
        {"bbox": [600, 400, 700, 500], "confidence": 0.88, "class_name": "forklift"},
    ]
    tracks1 = tracker.update(dets_frame1, frame_w, frame_h, timestamp=0.0)
    assert len(tracks1) == 2

    # Map class to ID
    id_map = {t.class_name: t.track_id for t in tracks1}
    assert "worker" in id_map
    assert "forklift" in id_map

    # Frame 2: Both moved slightly (10 pixels)
    dets_frame2 = [
        {"bbox": [110, 202, 160, 302], "confidence": 0.91, "class_name": "worker"},
        {"bbox": [590, 405, 690, 505], "confidence": 0.87, "class_name": "forklift"},
    ]
    tracks2 = tracker.update(dets_frame2, frame_w, frame_h, timestamp=0.05)
    assert len(tracks2) == 2

    # IDs MUST be persistent!
    for t in tracks2:
        assert t.track_id == id_map[t.class_name]
        # Coordinates must be normalized
        assert 0.0 <= t.position[0] <= 1.0
        assert 0.0 <= t.position[1] <= 1.0


def test_track_recovery_after_one_frame_miss():
    tracker = ByteTracker(track_thresh=0.35, match_thresh=0.4, max_time_lost=5)
    tracker.reset()

    frame_w = 1000
    frame_h = 1000

    # Frame 1: detect worker
    dets_1 = [{"bbox": [100, 100, 200, 200], "confidence": 0.9, "class_name": "worker"}]
    t1 = tracker.update(dets_1, frame_w, frame_h, 0.0)
    worker_id = t1[0].track_id

    # Frame 2: detector missed worker (empty detections)
    t2 = tracker.update([], frame_w, frame_h, 0.1)
    assert len(t2) == 0

    # Frame 3: worker re-detected close by
    dets_3 = [{"bbox": [105, 105, 205, 205], "confidence": 0.85, "class_name": "worker"}]
    t3 = tracker.update(dets_3, frame_w, frame_h, 0.2)
    assert len(t3) == 1
    # Should recover same ID!
    assert t3[0].track_id == worker_id
