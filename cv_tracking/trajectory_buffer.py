"""
Trajectory history buffer for ForeSite.
Maintains recent movement history by track_id, handles track expiry / temporary occlusion grace periods,
and exports clean TrajectoryHistory structures for Person 2's GRU/LSTM model.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import logging

from shared.schemas import TrajectoryHistory

logger = logging.getLogger(__name__)


class TrackRecord:
    """Internal container for a single tracked object's state and recent history."""

    def __init__(self, track_id: int, class_name: str, max_length: int = 12):
        self.track_id: int = track_id
        self.class_name: str = class_name
        self.max_length: int = max_length
        # Deque containing [x_norm, y_norm, timestamp]
        self.history: deque = deque(maxlen=max_length)
        self.last_timestamp: float = 0.0
        self.last_position: Optional[Tuple[float, float]] = None

    def add_observation(
        self,
        x_norm: float,
        y_norm: float,
        timestamp: float,
        enable_smoothing: bool = False,
        smoothing_alpha: float = 0.6
    ):
        """Add an observation, optionally applying light exponential smoothing."""
        if enable_smoothing and self.last_position is not None:
            prev_x, prev_y = self.last_position
            smooth_x = smoothing_alpha * x_norm + (1.0 - smoothing_alpha) * prev_x
            smooth_y = smoothing_alpha * y_norm + (1.0 - smoothing_alpha) * prev_y
            x_norm = round(smooth_x, 5)
            y_norm = round(smooth_y, 5)

        self.last_position = (x_norm, y_norm)
        self.last_timestamp = timestamp
        self.history.append([x_norm, y_norm, round(timestamp, 3)])

    def get_history_list(self) -> List[List[float]]:
        """Return history as a regular Python list of [x_norm, y_norm, timestamp]."""
        return list(self.history)

    def to_schema(self) -> TrajectoryHistory:
        """Convert into Person 2 compatible TrajectoryHistory schema."""
        return TrajectoryHistory(
            track_id=self.track_id,
            class_name=self.class_name,
            history=self.get_history_list()
        )


class TrajectoryBuffer:
    """
    Thread-safe/modular buffer storing active trajectories over time.
    Keeps tracks alive during temporary occlusions (up to expiry_seconds).
    """

    def __init__(
        self,
        history_length: int = 12,
        expiry_seconds: float = 1.5,
        enable_smoothing: bool = True,
        smoothing_alpha: float = 0.6
    ):
        self.history_length: int = history_length
        self.expiry_seconds: float = expiry_seconds
        self.enable_smoothing: bool = enable_smoothing
        self.smoothing_alpha: float = smoothing_alpha
        self.tracks: Dict[int, TrackRecord] = {}

    def update(
        self,
        track_id: int,
        class_name: str,
        x_norm: float,
        y_norm: float,
        timestamp: float
    ) -> List[List[float]]:
        """
        Record an observation for a track and return its updated history.

        Args:
            track_id: ByteTrack integer ID.
            class_name: Canonical class label (e.g., 'worker', 'forklift').
            x_norm: Normalized horizontal coordinate [0.0, 1.0].
            y_norm: Normalized vertical coordinate [0.0, 1.0].
            timestamp: Time in seconds from video start.

        Returns:
            Current list of observations [[x, y, t], ...]
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackRecord(
                track_id=track_id,
                class_name=class_name,
                max_length=self.history_length
            )
        else:
            # Update class name if refined
            self.tracks[track_id].class_name = class_name

        record = self.tracks[track_id]
        record.add_observation(
            x_norm=x_norm,
            y_norm=y_norm,
            timestamp=timestamp,
            enable_smoothing=self.enable_smoothing,
            smoothing_alpha=self.smoothing_alpha
        )

        return record.get_history_list()

    def get_history(self, track_id: int) -> Optional[List[List[float]]]:
        """Return history for a specific track, or None if not found."""
        record = self.tracks.get(track_id)
        return record.get_history_list() if record else None

    def get_trajectory_object(self, track_id: int) -> Optional[TrajectoryHistory]:
        """Return TrajectoryHistory schema object for Person 2."""
        record = self.tracks.get(track_id)
        return record.to_schema() if record else None

    def get_all_trajectories(self) -> List[TrajectoryHistory]:
        """Return all current trajectories as TrajectoryHistory schemas."""
        return [record.to_schema() for record in self.tracks.values()]

    def get_all_trajectories_dict(self) -> List[Dict[str, Any]]:
        """Return all current trajectories as plain dictionaries for JSON export."""
        return [record.to_schema().to_dict() for record in self.tracks.values()]

    def cleanup_expired(self, current_timestamp: float) -> List[int]:
        """
        Remove tracks that have not been observed for longer than expiry_seconds.

        Args:
            current_timestamp: Current video timestamp in seconds.

        Returns:
            List of expired track IDs that were removed.
        """
        expired_ids = []
        for track_id, record in list(self.tracks.items()):
            time_since_seen = current_timestamp - record.last_timestamp
            if time_since_seen > self.expiry_seconds:
                expired_ids.append(track_id)
                del self.tracks[track_id]
                logger.debug(f"Expired track #{track_id} (inactive for {time_since_seen:.2f}s)")

        return expired_ids

    def clear(self):
        """Reset the buffer."""
        self.tracks.clear()
