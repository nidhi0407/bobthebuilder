"""
End-to-end integration tests for ForeSite CV & Tracking pipeline.
"""

import os
import json
import tempfile
from cv_tracking.config import TrackingConfig
from cv_tracking.run_cv_pipeline import run_pipeline, generate_synthetic_demo_video


def test_end_to_end_pipeline_run():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Generate a mini 30-frame demo video
        video_path = os.path.join(tmpdir, "mini_site.mp4")
        generate_synthetic_demo_video(video_path, num_frames=30)
        assert os.path.exists(video_path)

        # 2. Configure pipeline with output to tmpdir
        config = TrackingConfig(
            export_dir=tmpdir,
            history_length=8,
            conf_threshold=0.3
        )

        # 3. Run pipeline
        summary = run_pipeline(
            video_path=video_path,
            config=config,
            save_video=True,
            headless=True,
            force_synthetic=True
        )

        # 4. Validate summary metrics
        assert summary["processed_frames"] == 30
        assert summary["effective_fps"] > 0
        assert summary["unique_tracks_count"] >= 2

        # 5. Validate Person 2 JSON output
        json_path = summary["json_trajectories_export"]
        assert os.path.exists(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            trajs = json.load(f)
            assert len(trajs) >= 2
            classes = {t["class_name"] for t in trajs}
            assert "worker" in classes
            assert "forklift" in classes
            for t in trajs:
                assert "track_id" in t
                assert "history" in t
                assert len(t["history"]) <= 8  # history_length capped
                for obs in t["history"]:
                    assert len(obs) == 3  # [x_norm, y_norm, timestamp]
                    x_norm, y_norm, timestamp = obs
                    assert 0.0 <= x_norm <= 1.0
                    assert 0.0 <= y_norm <= 1.0
                    assert timestamp >= 0.0

        # 6. Validate CSV output
        csv_path = summary["csv_export"]
        assert os.path.exists(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) > 1
            assert lines[0].strip() == "timestamp,track_id,class_name,x,y"

        # 7. Validate Person 4 Frame tracks output
        frame_json = summary["frame_tracks_export"]
        assert os.path.exists(frame_json)
        with open(frame_json, "r", encoding="utf-8") as f:
            frames = json.load(f)
            assert len(frames) == 30
            assert frames[0]["frame_index"] == 0
            assert "tracks" in frames[0]

        # 8. Validate annotated video output
        annotated_video = summary["annotated_video"]
        assert os.path.exists(annotated_video)
        assert os.path.getsize(annotated_video) > 0
