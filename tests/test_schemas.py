"""
Tests for shared schemas across ForeSite pipeline.
Verifies serialization, deserialization, type safety, and Person 2 compatibility.
"""

import json
from shared.schemas import (
    Track,
    TrajectoryHistory,
    FrameTracks,
    Prediction,
    RiskResult,
    Recommendation,
    serialize_tracks_json,
    serialize_histories_json,
)


def test_track_serialization():
    track = Track(
        track_id=4,
        class_name="worker",
        bbox=[100.0, 200.0, 150.0, 320.0],
        position=[0.125, 0.444],
        timestamp=2.35,
        confidence=0.91
    )
    d = track.to_dict()
    assert d["track_id"] == 4
    assert d["class_name"] == "worker"
    assert d["position"] == [0.125, 0.444]
    assert d["timestamp"] == 2.35
    assert d["confidence"] == 0.91

    restored = Track.from_dict(d)
    assert restored.track_id == 4
    assert restored.class_name == "worker"
    assert restored.bbox == [100.0, 200.0, 150.0, 320.0]


def test_trajectory_history_serialization():
    history_data = [
        [0.20, 0.60, 0.0],
        [0.22, 0.59, 0.1],
        [0.24, 0.58, 0.2]
    ]
    th = TrajectoryHistory(
        track_id=1,
        class_name="worker",
        history=history_data
    )
    d = th.to_dict()
    assert d["track_id"] == 1
    assert len(d["history"]) == 3

    # Check JSON roundtrip
    json_str = serialize_histories_json([th])
    loaded = json.loads(json_str)
    assert len(loaded) == 1
    assert loaded[0]["track_id"] == 1
    assert loaded[0]["class_name"] == "worker"
    assert loaded[0]["history"] == history_data


def test_frame_tracks_roundtrip():
    t1 = Track(1, "worker", [10, 20, 30, 40], [0.1, 0.2], 1.0, 0.95)
    t2 = Track(2, "forklift", [50, 60, 100, 120], [0.5, 0.6], 1.0, 0.88)
    ft = FrameTracks(frame_index=15, timestamp=1.0, tracks=[t1, t2])

    d = ft.to_dict()
    assert d["frame_index"] == 15
    assert len(d["tracks"]) == 2

    restored = FrameTracks.from_dict(d)
    assert restored.frame_index == 15
    assert restored.tracks[0].track_id == 1
    assert restored.tracks[1].class_name == "forklift"


def test_downstream_schemas():
    pred = Prediction(track_id=1, future_positions=[[0.3, 0.4], [0.35, 0.45]])
    assert pred.to_dict()["future_positions"] == [[0.3, 0.4], [0.35, 0.45]]

    risk = RiskResult(
        worker_id=1,
        machine_id=2,
        risk_score=0.88,
        time_to_conflict=1.75,
        minimum_distance=0.03,
        collision_point=[0.45, 0.52]
    )
    assert risk.to_dict()["risk_score"] == 0.88

    rec = Recommendation(
        action="reduce_vehicle_speed",
        magnitude=0.38,
        time_window=1.7,
        resulting_risk=0.08
    )
    assert rec.to_dict()["magnitude"] == 0.38
