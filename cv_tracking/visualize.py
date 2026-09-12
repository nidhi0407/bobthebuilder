"""
Visualization module for ForeSite CV & Tracking.
Draws bounding boxes, track IDs, class labels, representative bottom-center points,
and recent movement trails on video frames.
Does NOT draw future predicted trajectories (owned by Person 4 / dashboard).
"""

from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

from shared.schemas import Track
from .config import TrackingConfig
from .trajectory_buffer import TrajectoryBuffer
from .position import denormalize_coordinates


class Visualizer:
    """Renders real-time CV perception overlays on frames."""

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or TrackingConfig()

    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Return BGR color for class from config palette."""
        return self.config.color_palette.get(class_name, self.config.color_palette["default"])

    def draw_track_overlay(
        self,
        frame: np.ndarray,
        track: Track,
        buffer: Optional[TrajectoryBuffer] = None
    ) -> np.ndarray:
        """
        Draw bounding box, label badge, bottom-center dot, and historical trail for a single track.
        """
        h, w = frame.shape[:2]
        color = self.get_class_color(track.class_name)

        # 1. Bounding Box
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 2. Label Badge: "Worker #4 (91%)"
        label = f"{track.class_name.capitalize()} #{track.track_id}"
        if track.confidence > 0:
            label += f" {int(track.confidence * 100)}%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        badge_y1 = max(0, y1 - label_h - 8)
        badge_y2 = y1
        badge_x2 = min(w, x1 + label_w + 10)

        # Draw filled background rectangle for label
        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # Text in dark or white depending on contrast
        text_color = (0, 0, 0) if color in [(0, 215, 255), (50, 205, 50)] else (255, 255, 255)
        cv2.putText(frame, label, (x1 + 5, y1 - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # 3. Bottom-center representative point
        pos_x, pos_y = denormalize_coordinates(track.position[0], track.position[1], w, h)
        cv2.circle(frame, (pos_x, pos_y), 5, (0, 0, 255), -1)  # Red ground contact dot
        cv2.circle(frame, (pos_x, pos_y), 7, color, 1)

        # 4. Historical Trail
        if buffer and self.config.draw_trails:
            history = buffer.get_history(track.track_id)
            if history and len(history) > 1:
                trail_pts = []
                for obs in history:
                    # obs is [x_norm, y_norm, timestamp]
                    px, py = denormalize_coordinates(obs[0], obs[1], w, h)
                    trail_pts.append((px, py))

                # Draw trail segments with fading thickness
                num_pts = len(trail_pts)
                for i in range(1, num_pts):
                    alpha = i / float(num_pts)
                    pt1 = trail_pts[i - 1]
                    pt2 = trail_pts[i]
                    pt_thickness = max(1, int(round(alpha * 3)))
                    cv2.line(frame, pt1, pt2, color, pt_thickness, cv2.LINE_AA)
                    cv2.circle(frame, pt2, 3, color, -1)

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        frame_index: int,
        timestamp: float,
        active_tracks: int
    ) -> np.ndarray:
        """
        Draw semi-transparent telemetry HUD header at top-left.
        """
        hud_text = f"ForeSite AI | Frame {frame_index:04d} | Time {timestamp:.2f}s | FPS {fps:.1f} | Active Tracks: {active_tracks}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(hud_text, font, font_scale, thickness)

        # Dark pill banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 8), (20 + tw, 20 + th + 6), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, hud_text, (16, 14 + th), font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)
        return frame

    def annotate(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        buffer: Optional[TrajectoryBuffer] = None,
        frame_index: int = 0,
        timestamp: float = 0.0,
        fps: float = 30.0
    ) -> np.ndarray:
        """
        Full annotation pass for a frame.
        """
        annotated = frame.copy()

        # Render all track overlays
        for track in tracks:
            self.draw_track_overlay(annotated, track, buffer)

        # Render HUD
        self.draw_hud(annotated, fps, frame_index, timestamp, len(tracks))

        return annotated
