"""
Exporters for ForeSite CV & Tracking Module.
Supports:
1. JSON export (per-frame tracks or complete trajectory histories for Person 2)
2. CSV export (chronological position records for training / debugging)
3. Mock trajectory generation for offline integration
"""

import csv
import json
import os
from typing import List, Dict, Any, Union

from shared.schemas import Track, TrajectoryHistory, FrameTracks


class TrajectoryExporter:
    """Handles structured file exports for downstream teammates and logging."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.csv_records: List[Dict[str, Any]] = []
        self.frame_records: List[Dict[str, Any]] = []

    def record_observation(
        self,
        timestamp: float,
        track_id: int,
        class_name: str,
        x_norm: float,
        y_norm: float
    ):
        """Append a single coordinate observation to the internal CSV buffer."""
        self.csv_records.append({
            "timestamp": round(float(timestamp), 3),
            "track_id": int(track_id),
            "class_name": str(class_name),
            "x": round(float(x_norm), 5),
            "y": round(float(y_norm), 5),
        })

    def record_frame(self, frame_tracks: Union[FrameTracks, Dict[str, Any]]):
        """Append a frame snapshot to the internal JSON frame buffer."""
        if isinstance(frame_tracks, FrameTracks):
            self.frame_records.append(frame_tracks.to_dict())
        else:
            self.frame_records.append(frame_tracks)

    def export_csv(self, filename: str = "trajectories.csv") -> str:
        """
        Export all recorded observations to a chronological CSV file.

        Format:
        timestamp,track_id,class_name,x,y
        """
        filepath = os.path.join(self.output_dir, filename)

        # Ensure chronological ordering by timestamp
        sorted_records = sorted(self.csv_records, key=lambda r: (r["timestamp"], r["track_id"]))

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            fieldnames = ["timestamp", "track_id", "class_name", "x", "y"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in sorted_records:
                writer.writerow(record)

        return filepath

    def export_trajectories_json(
        self,
        trajectories: List[Union[TrajectoryHistory, Dict[str, Any]]],
        filename: str = "trajectories.json"
    ) -> str:
        """
        Export Person 2 compatible trajectory history list to JSON.

        Format:
        [
            {
                "track_id": 4,
                "class_name": "worker",
                "history": [[x, y, t], ...]
            }
        ]
        """
        filepath = os.path.join(self.output_dir, filename)
        serializable = []
        for t in trajectories:
            if isinstance(t, TrajectoryHistory):
                serializable.append(t.to_dict())
            else:
                serializable.append(t)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        return filepath

    def export_frames_json(self, filename: str = "frame_tracks.json") -> str:
        """Export per-frame track history to JSON."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.frame_records, f, indent=2)
        return filepath

    @staticmethod
    def generate_mock_trajectories(
        filepath: str = "data/mock_trajectories.json"
    ) -> str:
        """
        Generate mock converging trajectory dataset for Person 2 and Person 3 testing.
        Scenario: Worker walking across path while a forklift advances, heading for intersection.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        mock_data = [
            {
                "track_id": 1,
                "class_name": "worker",
                "history": [
                    [0.200, 0.600, 0.0],
                    [0.225, 0.590, 0.1],
                    [0.250, 0.580, 0.2],
                    [0.275, 0.570, 0.3],
                    [0.300, 0.560, 0.4],
                    [0.325, 0.550, 0.5],
                    [0.350, 0.540, 0.6],
                    [0.375, 0.530, 0.7],
                    [0.400, 0.520, 0.8],
                    [0.425, 0.510, 0.9],
                    [0.450, 0.500, 1.0],
                    [0.475, 0.490, 1.1],
                ]
            },
            {
                "track_id": 2,
                "class_name": "forklift",
                "history": [
                    [0.800, 0.300, 0.0],
                    [0.770, 0.320, 0.1],
                    [0.740, 0.340, 0.2],
                    [0.710, 0.360, 0.3],
                    [0.680, 0.380, 0.4],
                    [0.650, 0.400, 0.5],
                    [0.620, 0.420, 0.6],
                    [0.590, 0.440, 0.7],
                    [0.560, 0.460, 0.8],
                    [0.530, 0.480, 0.9],
                    [0.500, 0.500, 1.0],
                    [0.470, 0.520, 1.1],
                ]
            }
        ]

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(mock_data, f, indent=2)

        return filepath
