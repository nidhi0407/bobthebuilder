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
    safe_distance: float = 0.08      # Normalized coordinate threshold (approx 100px safety buffer)
    fps: float = 30.0                # Frames-per-second of the video feed
    risk_threshold: float = 0.50     # Score above which intervention triggers
    worker_classes: Tuple[str, ...] = ("worker", "person")
    machine_classes: Tuple[str, ...] = ("forklift", "excavator", "truck", "loader", "car", "bus")


# ---------------------------------------------------------------------------
# Per-pair result
# ---------------------------------------------------------------------------
@dataclass
class PairRiskResult:
    """Risk assessment output for a single entity pair."""
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
    Evaluates spatial-temporal collision risk between moving entities
    across predicted future trajectory horizons.
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
        Compute CPA distance, TTC, and composite risk score for a single pair.

        Args:
            worker_traj:  np.ndarray of shape (T, 2), normalized coords [0.0, 1.0].
            machine_traj: np.ndarray of shape (T, 2), normalized coords [0.0, 1.0].
            worker_id:    Tracking ID for first entity.
            machine_id:   Tracking ID for second entity.

        Returns:
            PairRiskResult with CPA, TTC, collision point, and danger flag.
        """
        assert worker_traj.ndim == 2 and worker_traj.shape[1] == 2, \
            f"Expected (T, 2) array for worker_traj, got {worker_traj.shape}"
        assert machine_traj.ndim == 2 and machine_traj.shape[1] == 2, \
            f"Expected (T, 2) array for machine_traj, got {machine_traj.shape}"

        # Align trajectory lengths (use the common prefix)
        horizon = min(len(worker_traj), len(machine_traj))
        w_seq = worker_traj[:horizon]  # (T, 2)
        m_seq = machine_traj[:horizon]  # (T, 2)

        # Vectorized Euclidean separation at each future timestep t = 1..T
        diffs = w_seq - m_seq                     # (T, 2)
        distances = np.linalg.norm(diffs, axis=1) # (T,)

        # CPA: minimum Euclidean distance across the prediction horizon
        cpa_idx = int(np.argmin(distances))
        cpa_distance = float(distances[cpa_idx])

        # TTC: seconds until closest approach occurs (1-indexed timestep)
        time_to_conflict = float((cpa_idx + 1) / self.cfg.fps)

        # Midpoint of the two entities at the moment of CPA
        collision_point = (
            (w_seq[cpa_idx] + m_seq[cpa_idx]) / 2.0
        ).tolist()

        # Risk score calculation:
        # Distance component: 1.0 at d=0, 0.0 at d >= 2 * safe_distance
        dist_factor = max(0.0, 1.0 - (cpa_distance / (2.0 * self.cfg.safe_distance)))

        # Time urgency factor: closer conflict horizons amplify risk
        max_horizon_sec = float(horizon / self.cfg.fps)
        time_factor = max(0.0, 1.0 - (time_to_conflict / (2.0 * max_horizon_sec)))

        # Composite score: weighted blend (70% distance proximity, 30% time urgency)
        raw_risk = 0.70 * dist_factor + 0.30 * time_factor
        risk_score = float(np.clip(raw_risk, 0.0, 1.0))

        # Binary danger trigger
        is_dangerous = (cpa_distance < self.cfg.safe_distance) or (risk_score >= self.cfg.risk_threshold)

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
    # Batch evaluation: all worker × machine & machine × machine pairs
    # ------------------------------------------------------------------
    def evaluate_all_pairs(
        self,
        predictions: Dict[int, np.ndarray],
        class_map: Dict[int, str],
    ) -> List[PairRiskResult]:
        """
        Evaluate risk for every relevant entity combination.

        Args:
            predictions: {track_id: np.ndarray (T, 2)} of predicted future coords.
            class_map:   {track_id: class_name} e.g. {1: "worker", 2: "forklift"}.

        Returns:
            List of PairRiskResult, one per evaluated entity pair.
        """
        all_tids = list(predictions.keys())
        results: List[PairRiskResult] = []
        seen_pairs = set()

        for i in range(len(all_tids)):
            for j in range(i + 1, len(all_tids)):
                id_a, id_b = all_tids[i], all_tids[j]
                cls_a = class_map.get(id_a, "").lower()
                cls_b = class_map.get(id_b, "").lower()

                # Determine worker vs machine roles or machine-machine
                is_a_worker = "worker" in cls_a or "person" in cls_a
                is_b_worker = "worker" in cls_b or "person" in cls_b

                if is_a_worker and not is_b_worker:
                    wid, mid = id_a, id_b
                elif is_b_worker and not is_a_worker:
                    wid, mid = id_b, id_a
                else:
                    wid, mid = id_a, id_b

                pair_key = (min(wid, mid), max(wid, mid))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

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
