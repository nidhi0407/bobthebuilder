"""
Video ingestion module for ForeSite.
Provides a robust OpenCV-based video reader for recorded site feeds and live streams,
extracting frame index, accurate timestamps in seconds, FPS, and frame dimensions.
"""

import os
from typing import Generator, Tuple, Optional, Union
import cv2
import numpy as np


class VideoReaderError(Exception):
    """Raised when video cannot be found, opened, or read."""
    pass


class VideoReader:
    """
    Robust wrapper around cv2.VideoCapture for frame-by-frame processing.
    Ensures correct timing, metadata extraction, and clean resource release.
    """

    def __init__(self, video_source: Union[str, int], resize_width: Optional[int] = None):
        """
        Initialize the video reader.

        Args:
            video_source: Path to video file (str) or camera index (int, e.g. 0 for webcam).
            resize_width: Optional width to downscale frames for faster inference.
        """
        self.video_source = video_source
        self.resize_width = resize_width

        # Validate file existence if string path
        if isinstance(video_source, str) and not video_source.startswith(("rtsp://", "http://", "https://")):
            if not os.path.exists(video_source):
                raise VideoReaderError(f"Video file not found at path: {video_source}")

        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise VideoReaderError(f"Failed to open video source: {video_source}")

        # Read metadata
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0.0 or np.isnan(self.fps):
            self.fps = 30.0  # Safe default fallback

        self.orig_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.resize_width and self.resize_width < self.orig_width:
            scale = self.resize_width / float(self.orig_width)
            self.frame_width = int(self.resize_width)
            self.frame_height = int(round(self.orig_height * scale))
        else:
            self.frame_width = self.orig_width
            self.frame_height = self.orig_height

        self.current_frame_index = 0

    @property
    def duration_seconds(self) -> float:
        """Estimated total duration in seconds."""
        if self.fps > 0 and self.total_frames > 0:
            return self.total_frames / self.fps
        return 0.0

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """
        Read a single frame.

        Returns:
            (success, frame, frame_index, timestamp_seconds)
        """
        if not self.cap.isOpened():
            return False, None, self.current_frame_index, 0.0

        success, frame = self.cap.read()
        if not success or frame is None:
            return False, None, self.current_frame_index, 0.0

        # Optional resizing
        if (self.frame_width, self.frame_height) != (self.orig_width, self.orig_height):
            frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)

        # Precise timestamp computation
        # Try hardware timestamp first, fallback to frame_index / fps
        pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec > 0:
            timestamp = pos_msec / 1000.0
        else:
            timestamp = self.current_frame_index / self.fps

        frame_idx = self.current_frame_index
        self.current_frame_index += 1

        return True, frame, frame_idx, round(timestamp, 4)

    def frames(self) -> Generator[Tuple[np.ndarray, int, float], None, None]:
        """
        Generator yielding (frame, frame_index, timestamp_seconds) sequentially.
        """
        while True:
            success, frame, frame_idx, timestamp = self.read_frame()
            if not success:
                break
            yield frame, frame_idx, timestamp

    def release(self):
        """Release video capture device."""
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def __iter__(self):
        return self.frames()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
