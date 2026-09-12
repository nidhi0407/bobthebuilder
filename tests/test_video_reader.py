"""
Tests for VideoReader module.
Verifies frame iteration, timestamps in seconds, metadata extraction, and error handling.
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from cv_tracking.video_reader import VideoReader, VideoReaderError


@pytest.fixture
def sample_video_path():
    """Create a temporary 10-frame video file for testing."""
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, "test_video.mp4")

    # Use mp4v fourcc
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (320, 240))

    for i in range(10):
        # Create distinct dummy frames
        frame = np.full((240, 320, 3), i * 20, dtype=np.uint8)
        out.write(frame)

    out.release()
    yield video_path

    if os.path.exists(video_path):
        os.remove(video_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)


def test_video_reader_metadata(sample_video_path):
    with VideoReader(sample_video_path) as reader:
        assert reader.frame_width == 320
        assert reader.frame_height == 240
        assert reader.fps == 10.0
        assert reader.total_frames == 10
        assert pytest.approx(reader.duration_seconds, 0.1) == 1.0


def test_video_reader_iteration(sample_video_path):
    with VideoReader(sample_video_path) as reader:
        frames_read = 0
        last_t = -1.0
        for frame, frame_idx, timestamp in reader:
            assert frame.shape == (240, 320, 3)
            assert frame_idx == frames_read
            assert timestamp >= last_t
            last_t = timestamp
            frames_read += 1

        assert frames_read == 10


def test_video_reader_resize(sample_video_path):
    with VideoReader(sample_video_path, resize_width=160) as reader:
        assert reader.frame_width == 160
        assert reader.frame_height == 120
        success, frame, idx, t = reader.read_frame()
        assert success
        assert frame.shape == (120, 160, 3)


def test_video_reader_nonexistent_file():
    with pytest.raises(VideoReaderError):
        VideoReader("non_existent_file_path.mp4")
