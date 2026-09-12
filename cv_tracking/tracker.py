"""
ByteTrack Multi-Object Tracking implementation for ForeSite.
Provides persistent track IDs across frames using ByteTrack's two-stage association
(matching high-confidence detections first, then low-confidence detections).
Ensures tracks persist through brief occlusions and detector misses.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np

from shared.schemas import Track
from .config import TrackingConfig
from .position import extract_representative_position, normalize_coordinates


def compute_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Compute Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, bbox1[2] - bbox1[0]) * max(0.0, bbox1[3] - bbox1[1])
    area2 = max(0.0, bbox2[2] - bbox2[0]) * max(0.0, bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class STrack:
    """Individual single-object track state."""

    _count = 0

    def __init__(self, bbox: List[float], score: float, class_name: str):
        STrack._count += 1
        self.track_id = STrack._count
        self.bbox = [float(c) for c in bbox]
        self.score = float(score)
        self.class_name = str(class_name)
        self.is_activated = True
        self.state = "tracked"  # 'tracked', 'lost', 'removed'
        self.frame_id = 0
        self.time_since_update = 0
        self.smooth_alpha = 0.7  # Spatial position smoothing across frames

    def update(self, new_bbox: List[float], new_score: float, new_class: str, frame_id: int):
        """Update track with newly matched detection."""
        # Light smoothing on bbox to suppress jitter
        smooth_box = [
            self.smooth_alpha * float(new_bbox[i]) + (1.0 - self.smooth_alpha) * self.bbox[i]
            for i in range(4)
        ]
        self.bbox = smooth_box
        self.score = float(new_score)
        self.class_name = str(new_class)
        self.frame_id = frame_id
        self.time_since_update = 0
        self.state = "tracked"

    def mark_lost(self):
        self.state = "lost"

    def mark_removed(self):
        self.state = "removed"


class ByteTracker:
    """
    ByteTrack Multi-Object Tracker.
    Maintains persistent IDs across frames with high tolerance for detector flicker.
    """

    def __init__(
        self,
        config: Optional[TrackingConfig] = None,
        track_thresh: float = 0.35,
        match_thresh: float = 0.40,
        max_time_lost: int = 30
    ):
        """
        Args:
            config: Optional TrackingConfig object.
            track_thresh: Threshold separating high-conf and low-conf detections.
            match_thresh: IoU threshold for associating detections to tracks.
            max_time_lost: Maximum frames a lost track is kept before being purged.
        """
        if isinstance(config, TrackingConfig):
            self.track_thresh = config.track_thresh
            self.match_thresh = config.match_thresh
            self.max_time_lost = config.track_buffer
        elif isinstance(config, (int, float)):
            self.track_thresh = float(config)
            self.match_thresh = match_thresh
            self.max_time_lost = max_time_lost
        else:
            self.track_thresh = track_thresh
            self.match_thresh = match_thresh
            self.max_time_lost = max_time_lost

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.frame_id = 0

    def reset(self):
        """Reset tracker state and ID counter."""
        STrack._count = 0
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.frame_id = 0

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        timestamp: float = 0.0,
        anchor: str = "bottom-center",
        frame_id: Optional[int] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Track]:
        """
        Update tracks with current frame detections.

        Args:
            detections: List of dicts [{"bbox": [x1, y1, x2, y2], "confidence": float, "class_name": str}, ...]
            frame_width: Width of current frame.
            frame_height: Height of current frame.
            timestamp: Current frame timestamp in seconds.
            anchor: Position extraction anchor ("bottom-center" or "center").
            frame_id: Optional frame index.
            frame_shape: Optional (height, width) tuple.

        Returns:
            List of Track schema objects for currently active tracks.
        """
        if frame_shape is not None:
            frame_height, frame_width = frame_shape
        if frame_width is None:
            frame_width = 1920
        if frame_height is None:
            frame_height = 1080

        if frame_id is not None:
            self.frame_id = frame_id
        else:
            self.frame_id += 1

        # Separate detections into high confidence (D_high) and low confidence (D_low)
        d_high = []
        d_low = []
        for det in detections:
            if det["confidence"] >= self.track_thresh:
                d_high.append(det)
            elif det["confidence"] >= 0.1:  # Filter out total noise
                d_low.append(det)

        # Candidates to match: tracked + lost
        candidate_tracks = self.tracked_stracks + self.lost_stracks

        # --- First Association: D_high with Candidate Tracks ---
        unmatched_tracks, unmatched_d_high, matches_first = self._associate(
            candidate_tracks, d_high, self.match_thresh
        )

        for track_idx, det_idx in matches_first:
            det = d_high[det_idx]
            candidate_tracks[track_idx].update(
                det["bbox"], det["confidence"], det["class_name"], self.frame_id
            )

        # --- Second Association: D_low with remaining unmatched tracks ---
        unmatched_tracks_second, _, matches_second = self._associate(
            [candidate_tracks[i] for i in unmatched_tracks], d_low, self.match_thresh
        )

        for trk_sub_idx, det_idx in matches_second:
            orig_trk_idx = unmatched_tracks[trk_sub_idx]
            det = d_low[det_idx]
            candidate_tracks[orig_trk_idx].update(
                det["bbox"], det["confidence"], det["class_name"], self.frame_id
            )

        # Tracks still unmatched after both stages
        still_unmatched = [unmatched_tracks[i] for i in unmatched_tracks_second]
        for trk_idx in still_unmatched:
            track = candidate_tracks[trk_idx]
            track.time_since_update += 1
            if track.time_since_update > self.max_time_lost:
                track.mark_removed()
            else:
                track.mark_lost()

        # --- Initialize new tracks from unmatched high-confidence detections ---
        for det_idx in unmatched_d_high:
            det = d_high[det_idx]
            new_track = STrack(det["bbox"], det["confidence"], det["class_name"])
            new_track.frame_id = self.frame_id
            self.tracked_stracks.append(new_track)

        # Update tracked and lost pools
        active_tracked = []
        active_lost = []
        for track in candidate_tracks + [t for t in self.tracked_stracks if t.frame_id == self.frame_id]:
            if track.state == "tracked":
                if track not in active_tracked:
                    active_tracked.append(track)
            elif track.state == "lost":
                if track not in active_lost:
                    active_lost.append(track)

        self.tracked_stracks = active_tracked
        self.lost_stracks = active_lost

        # Format output as clean Track schema objects
        output_tracks: List[Track] = []
        for strack in self.tracked_stracks:
            # Only output currently updated tracks in this frame
            if strack.frame_id == self.frame_id:
                # Extract representative point (bottom-center for worker/machine)
                px, py = extract_representative_position(strack.bbox, anchor=anchor)
                x_norm, y_norm = normalize_coordinates(px, py, frame_width, frame_height)

                output_tracks.append(
                    Track(
                        track_id=strack.track_id,
                        class_name=strack.class_name,
                        bbox=[round(v, 2) for v in strack.bbox],
                        position=[x_norm, y_norm],
                        timestamp=round(timestamp, 4),
                        confidence=round(strack.score, 4),
                    )
                )

        return output_tracks

    def _associate(
        self,
        tracks: List[STrack],
        detections: List[Dict[str, Any]],
        iou_threshold: float
    ) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
        """
        Greedy bipartite matching based on IoU overlap.
        Returns: (unmatched_track_indices, unmatched_detection_indices, list_of_(track_idx, det_idx))
        """
        if len(tracks) == 0 or len(detections) == 0:
            return list(range(len(tracks))), list(range(len(detections))), []

        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for t_idx, track in enumerate(tracks):
            for d_idx, det in enumerate(detections):
                # Prefer same class association
                iou = compute_iou(track.bbox, det["bbox"])
                if track.class_name == det["class_name"]:
                    iou_matrix[t_idx, d_idx] = iou
                else:
                    # Penalize class mismatch
                    iou_matrix[t_idx, d_idx] = iou * 0.5

        matched_tracks = set()
        matched_detections = set()
        matches = []

        # Greedy match from highest IoU down to threshold
        while True:
            max_val = np.max(iou_matrix)
            if max_val < iou_threshold:
                break
            t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            matches.append((int(t_idx), int(d_idx)))
            matched_tracks.add(int(t_idx))
            matched_detections.add(int(d_idx))
            iou_matrix[t_idx, :] = -1.0
            iou_matrix[:, d_idx] = -1.0

        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
        unmatched_dets = [j for j in range(len(detections)) if j not in matched_detections]

        return unmatched_tracks, unmatched_dets, matches
