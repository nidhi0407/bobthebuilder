"""
Tests for TrajectoryBuffer.
Verifies history length, grace periods, chronological consistency, and expiration.
"""

from cv_tracking.trajectory_buffer import TrajectoryBuffer


def test_buffer_history_length():
    buffer = TrajectoryBuffer(history_length=5, expiry_seconds=1.5, enable_smoothing=False)

    # Add 10 observations
    for i in range(10):
        buffer.update(track_id=1, class_name="worker", x_norm=0.1 * i, y_norm=0.2 * i, timestamp=0.1 * i)

    history = buffer.get_history(1)
    assert len(history) == 5  # capped at history_length=5
    # Oldest in the 5 should be observation 5 (timestamp 0.5)
    assert history[0][2] == 0.5
    # Newest should be observation 9 (timestamp 0.9)
    assert history[-1][2] == 0.9


def test_buffer_grace_period_and_expiry():
    buffer = TrajectoryBuffer(history_length=12, expiry_seconds=1.0)

    # Observe track 1 at t=0.0 and track 2 at t=0.0
    buffer.update(1, "worker", 0.2, 0.4, 0.0)
    buffer.update(2, "forklift", 0.7, 0.3, 0.0)

    # At t=0.5 (within 1.0s grace period), both should be retained
    expired = buffer.cleanup_expired(current_timestamp=0.5)
    assert len(expired) == 0
    assert buffer.get_history(1) is not None
    assert buffer.get_history(2) is not None

    # Track 1 updated at t=0.8, Track 2 not seen
    buffer.update(1, "worker", 0.21, 0.41, 0.8)

    # At t=1.2 (track 2 last seen at 0.0 -> 1.2s > 1.0s, should expire; track 1 last seen 0.8 -> 0.4s < 1.0s, kept)
    expired = buffer.cleanup_expired(current_timestamp=1.2)
    assert 2 in expired
    assert 1 not in expired
    assert buffer.get_history(2) is None
    assert buffer.get_history(1) is not None


def test_buffer_trajectory_export():
    buffer = TrajectoryBuffer(history_length=12, expiry_seconds=1.5)
    buffer.update(1, "worker", 0.1, 0.2, 0.0)
    buffer.update(1, "worker", 0.12, 0.22, 0.1)

    obj = buffer.get_trajectory_object(1)
    assert obj is not None
    assert obj.track_id == 1
    assert obj.class_name == "worker"
    assert len(obj.history) == 2

    all_objs = buffer.get_all_trajectories()
    assert len(all_objs) == 1
