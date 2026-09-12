"""
ForeSite AI — Risk Engine (Stage 5)

Computes collision risk between all worker-machine pairs using their
predicted future trajectories. Pure vectorized NumPy math — no ML model.

Metrics computed per pair:
  - CPA  (Closest Point of Approach): minimum Euclidean separation across
    the prediction horizon.
  - TTC  (Time-to-Conflict): the timestamp (in seconds) at which CPA occurs.
  - Risk Score: normalized [0.0 – 1.0] danger indicator derived from CPA
    relative to the safe-distance threshold.

Usage:
    from model.risk_engine import RiskEngine

    engine = RiskEngine(safe_distance=0.05, fps=30)
    results = engine.evaluate_all_pairs(predictions, class_map)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class RiskConfig:
    """Tuneable safety parameters."""
    safe_distance: float = 0.05      # Normalized coordinate threshold
    fps: float = 30.0                # Frames-per-second of the video feed
    risk_threshold: float = 0.60     # Score above which intervention triggers
    worker_classes: Tuple[str, ...] = ("worker",)
    machine_classes: Tuple[str, ...] = ("forklift", "excavator", "truck", "loader")


# ---------------------------------------------------------------------------
# Per-pair result
# ---------------------------------------------------------------------------
@dataclass
class PairRiskResult:
    """Risk assessment output for a single worker ↔ machine pair."""
    worker_id: int
    machine_id: int
    cpa_distance: float          # Closest predicted separation (normalized coords)
    time_to_conflict: float      # Seconds until CPA occurs
    collision_point: List[float] # [x, y] midpoint at CPA timestep
    risk_score: float            # 0.0 (safe) → 1.0 (imminent collision)
    is_dangerous: bool           # True when risk_score ≥ risk_threshold

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Risk Engine
# ---------------------------------------------------------------------------
class RiskEngine:
    """
    Vectorized risk evaluator for predicted trajectory pairs.

    All geometry is computed on normalized [0, 1] coordinate space so that
    the engine is resolution-independent and camera-agnostic.
    """

    def __init__(self, config: Optional[RiskConfig] = None):
        self.cfg = config or RiskConfig()

    # ------------------------------------------------------------------
    # Core math: single pair
    # ------------------------------------------------------------------
    def compute_pair_risk(
        self,
        worker_traj: np.ndarray,
        machine_traj: np.ndarray,
        worker_id: int = 0,
        machine_id: int = 1,
    ) -> PairRiskResult:
        """
        Compute CPA, TTC, and risk score for one worker–machine pair.

        Args:
            worker_traj:  np.ndarray of shape (T, 2) — predicted future (x, y).
            machine_traj: np.ndarray of shape (T, 2) — predicted future (x, y).
            worker_id:    Tracking ID of the worker.
            machine_id:   Tracking ID of the machine.

        Returns:
            PairRiskResult with all computed metrics.
        """
        # Euclidean distance at every future timestep  — shape (T,)
        distances = np.linalg.norm(worker_traj - machine_traj, axis=1)

        # CPA: minimum predicted distance
        min_idx = int(np.argmin(distances))
        cpa_distance = float(distances[min_idx])

        # TTC: convert frame index → seconds
        time_to_conflict = float(min_idx / self.cfg.fps)

        # Collision point: midpoint between the two agents at CPA timestep
        collision_point = (
            (worker_traj[min_idx] + machine_traj[min_idx]) / 2.0
        ).tolist()

        # Risk score: inversely proportional to CPA, clamped to [0, 1]
        if cpa_distance <= 0.0:
            risk_score = 1.0
        elif cpa_distance >= self.cfg.safe_distance:
            risk_score = 0.0
        else:
            risk_score = 1.0 - (cpa_distance / self.cfg.safe_distance)

        is_dangerous = risk_score >= self.cfg.risk_threshold

        return PairRiskResult(
            worker_id=worker_id,
            machine_id=machine_id,
            cpa_distance=round(cpa_distance, 6),
            time_to_conflict=round(time_to_conflict, 3),
            collision_point=[round(c, 6) for c in collision_point],
            risk_score=round(risk_score, 4),
            is_dangerous=is_dangerous,
        )

    # ------------------------------------------------------------------
    # Batch evaluation: all worker × machine pairs
    # ------------------------------------------------------------------
    def evaluate_all_pairs(
        self,
        predictions: Dict[int, np.ndarray],
        class_map: Dict[int, str],
    ) -> List[PairRiskResult]:
        """
        Evaluate risk for every worker–machine combination.

        Args:
            predictions: {track_id: np.ndarray (T, 2)} of predicted future coords.
            class_map:   {track_id: class_name} e.g. {1: "worker", 2: "forklift"}.

        Returns:
            List of PairRiskResult, one per worker–machine pair.
        """
        worker_ids = [
            tid for tid, cls in class_map.items()
            if cls in self.cfg.worker_classes
        ]
        machine_ids = [
            tid for tid, cls in class_map.items()
            if cls in self.cfg.machine_classes
        ]

        results: List[PairRiskResult] = []
        for wid in worker_ids:
            for mid in machine_ids:
                if wid in predictions and mid in predictions:
                    result = self.compute_pair_risk(
                        worker_traj=predictions[wid],
                        machine_traj=predictions[mid],
                        worker_id=wid,
                        machine_id=mid,
                    )
                    results.append(result)

        # Sort by risk score descending (most dangerous first)
        results.sort(key=lambda r: r.risk_score, reverse=True)
        return results

    def get_dangerous_pairs(
        self,
        predictions: Dict[int, np.ndarray],
        class_map: Dict[int, str],
    ) -> List[PairRiskResult]:
        """Return only pairs where risk_score ≥ threshold."""
        all_results = self.evaluate_all_pairs(predictions, class_map)
        return [r for r in all_results if r.is_dangerous]
