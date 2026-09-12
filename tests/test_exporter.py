"""
Tests for TrajectoryExporter.
Verifies CSV and JSON formats, chronological consistency, and mock generation.
"""

import os
import csv
import json
import tempfile
from cv_tracking.exporter import TrajectoryExporter
from shared.schemas import TrajectoryHistory, Track, FrameTracks


def test_csv_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = TrajectoryExporter(output_dir=tmpdir)
        # Add out-of-order observations to test sorting
        exporter.record_observation(timestamp=1.2, track_id=1, class_name="worker", x_norm=0.3, y_norm=0.5)
        exporter.record_observation(timestamp=0.5, track_id=1, class_name="worker", x_norm=0.2, y_norm=0.4)
        exporter.record_observation(timestamp=0.5, track_id=2, class_name="forklift", x_norm=0.8, y_norm=0.2)

        csv_path = exporter.export_csv("test.csv")
        assert os.path.exists(csv_path)

        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 3
            # Chronological order
            assert float(reader[0]["timestamp"]) == 0.5
            assert float(reader[1]["timestamp"]) == 0.5
            assert float(reader[2]["timestamp"]) == 1.2


def test_json_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = TrajectoryExporter(output_dir=tmpdir)
        th = TrajectoryHistory(track_id=1, class_name="worker", history=[[0.2, 0.4, 1.0], [0.22, 0.41, 1.1]])

        json_path = exporter.export_trajectories_json([th], "test.json")
        assert os.path.exists(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["track_id"] == 1
            assert len(data[0]["history"]) == 2


def test_mock_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_path = os.path.join(tmpdir, "mock.json")
        TrajectoryExporter.generate_mock_trajectories(mock_path)
        assert os.path.exists(mock_path)

        with open(mock_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) >= 2
            assert data[0]["class_name"] == "worker"
            assert data[1]["class_name"] == "forklift"
