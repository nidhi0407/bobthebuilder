"""
ForeSite AI — Shared Schemas
Standardized data contracts for all 4 team roles:
- Person 1: CV & Tracking (produces Track, TrajectoryHistory, FrameTracks)
- Person 2: Trajectory Prediction & Risk Engine (consumes TrajectoryHistory, produces Prediction & RiskResult)
- Person 3: Intervention & Simulation Engine (consumes RiskResult & Prediction, produces Recommendation)
- Person 4: Dashboard & Visualization (consumes FrameTracks, Prediction, RiskResult, Recommendation)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import json


# Canonical class names
CANONICAL_CLASSES = {"worker", "forklift", "excavator", "truck", "loader"}


@dataclass
class Track:
    """Per-frame detected and tracked object representation."""
    track_id: int
    class_name: str
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates
    position: List[float]  # [x_norm, y_norm] normalized coordinates [0.0, 1.0]
    timestamp: float  # seconds from video start
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Track:
        return cls(
            track_id=int(data["track_id"]),
            class_name=str(data["class_name"]),
            bbox=[float(x) for x in data["bbox"]],
            position=[float(x) for x in data["position"]],
            timestamp=float(data["timestamp"]),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class TrajectoryHistory:
    """Historical movement sequence for a single tracked object (Handoff to Person 2)."""
    track_id: int
    class_name: str
    # List of observations: [x_norm, y_norm, timestamp_seconds]
    history: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrajectoryHistory:
        return cls(
            track_id=int(data["track_id"]),
            class_name=str(data["class_name"]),
            history=[[float(coord) for coord in obs] for obs in data.get("history", [])],
        )


@dataclass
class FrameTracks:
    """All tracks present in a single frame."""
    frame_index: int
    timestamp: float
    tracks: List[Track] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "tracks": [t.to_dict() for t in self.tracks],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FrameTracks:
        return cls(
            frame_index=int(data["frame_index"]),
            timestamp=float(data["timestamp"]),
            tracks=[Track.from_dict(t) for t in data.get("tracks", [])],
        )


@dataclass
class Prediction:
    """Future predicted trajectory for a track (Person 2 output)."""
    track_id: int
    # List of future predicted coordinates: [[x_norm, y_norm], ...] or [[x_norm, y_norm, t], ...]
    future_positions: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Prediction:
        return cls(
            track_id=int(data["track_id"]),
            future_positions=[[float(c) for c in pos] for pos in data.get("future_positions", [])],
        )


@dataclass
class RiskResult:
    """Collision risk assessment for a worker-machine pair (Person 2 output)."""
    worker_id: int
    machine_id: int
    risk_score: float  # [0.0 - 1.0]
    time_to_conflict: float  # seconds
    minimum_distance: float  # normalized coordinate distance
    collision_point: List[float]  # [x_norm, y_norm]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RiskResult:
        return cls(
            worker_id=int(data["worker_id"]),
            machine_id=int(data["machine_id"]),
            risk_score=float(data["risk_score"]),
            time_to_conflict=float(data["time_to_conflict"]),
            minimum_distance=float(data["minimum_distance"]),
            collision_point=[float(c) for c in data["collision_point"]],
        )


@dataclass
class Recommendation:
    """Prescriptive safety intervention (Person 3 output)."""
    action: str  # e.g., "reduce_vehicle_speed", "stop_worker", "stop_vehicle", "reroute"
    magnitude: float  # e.g., 0.38 for 38% reduction
    time_window: float  # seconds e.g. 1.7
    resulting_risk: float  # residual risk after intervention

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Recommendation:
        return cls(
            action=str(data["action"]),
            magnitude=float(data["magnitude"]),
            time_window=float(data["time_window"]),
            resulting_risk=float(data["resulting_risk"]),
        )


def serialize_tracks_json(tracks: List[Track]) -> str:
    """Serialize a list of Track objects to JSON string."""
    return json.dumps([t.to_dict() for t in tracks], indent=2)


def serialize_histories_json(histories: List[TrajectoryHistory]) -> str:
    """Serialize a list of TrajectoryHistory objects to JSON string."""
    return json.dumps([h.to_dict() for h in histories], indent=2)
