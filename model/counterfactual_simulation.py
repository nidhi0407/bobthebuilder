"""
ForeSite AI — Counterfactual Simulation Layer (Stage 6a)

When the Risk Engine flags a dangerous worker–machine pair, this module
tests a menu of hypothetical interventions in software memory, re-evaluates
the risk under each one, and selects the **least disruptive action** that
restores the predicted separation above the safe-distance threshold.

Candidate interventions:
  1. Slow vehicle by 20%   (disruption cost: 2)
  2. Slow vehicle by 40%   (disruption cost: 4)
  3. Slow vehicle by 60%   (disruption cost: 6)
  4. Stop worker            (disruption cost: 5)
  5. Stop vehicle           (disruption cost: 8)

All trajectory perturbations are pure physics — scale displacement vectors
from the agent's current position.  No additional ML model is needed.

Usage:
    from model.counterfactual_simulation import CounterfactualSimulator

    simulator = CounterfactualSimulator()
    recommendation = simulator.find_best_intervention(
        worker_traj, machine_traj, worker_id=1, machine_id=2
    )
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from .risk_engine import RiskConfig, RiskEngine


# ---------------------------------------------------------------------------
# Intervention candidates
# ---------------------------------------------------------------------------
@dataclass
class InterventionCandidate:
    """A single hypothetical action that could be applied."""
    action: str              # Human-readable action description
    target: str              # "vehicle" or "worker"
    speed_factor: float      # Multiplicative factor (1.0 = no change, 0.0 = full stop)
    disruption_cost: int     # 1 (minimal) → 10 (maximum disruption)


DEFAULT_CANDIDATES: List[InterventionCandidate] = [
    InterventionCandidate(
        action="Slow vehicle by 20%",
        target="vehicle",
        speed_factor=0.80,
        disruption_cost=2,
    ),
    InterventionCandidate(
        action="Slow vehicle by 40%",
        target="vehicle",
        speed_factor=0.60,
        disruption_cost=4,
    ),
    InterventionCandidate(
        action="Slow vehicle by 60%",
        target="vehicle",
        speed_factor=0.40,
        disruption_cost=6,
    ),
    InterventionCandidate(
        action="Stop worker immediately",
        target="worker",
        speed_factor=0.0,
        disruption_cost=5,
    ),
    InterventionCandidate(
        action="Emergency stop vehicle",
        target="vehicle",
        speed_factor=0.0,
        disruption_cost=8,
    ),
]


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """Outcome of one counterfactual intervention test."""
    action: str
    target: str
    speed_factor: float
    disruption_cost: int
    resulting_cpa: float         # CPA under this intervention
    resulting_ttc: float         # TTC under this intervention
    resulting_risk_score: float  # Risk score under this intervention
    is_safe: bool                # True if resulting_risk_score < threshold

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class InterventionRecommendation:
    """
    The final prescriptive output of the simulation layer.

    Consumed by the Streamlit dashboard (Person 4) and displayed as:
      "Slow Forklift #2 by 40% within 1.7 seconds to avert collision."
    """
    status: str                          # "INTERVENTION_REQUIRED" | "SAFE" | "CRITICAL_DANGER"
    worker_id: int
    machine_id: int
    recommended_action: str              # Best intervention description
    speed_factor: float                  # e.g. 0.60 for 40% reduction
    deadline_seconds: float              # Time window to execute the action
    projected_safe_distance: float       # CPA after intervention
    projected_risk_score: float          # Residual risk after intervention
    all_tested: List[SimulationResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["all_tested"] = [s.to_dict() if isinstance(s, SimulationResult) else s for s in self.all_tested]
        return d

    def to_human_readable(self) -> str:
        """Generate a plain-language instruction for the dashboard."""
        if self.status == "SAFE":
            return "All clear — no intervention needed."
        pct = int((1.0 - self.speed_factor) * 100)
        return (
            f"{self.recommended_action} within {self.deadline_seconds:.1f}s "
            f"(projected safe distance: {self.projected_safe_distance:.4f})."
        )


# ---------------------------------------------------------------------------
# Trajectory perturbation helpers
# ---------------------------------------------------------------------------
def _perturb_trajectory(
    trajectory: np.ndarray,
    speed_factor: float,
) -> np.ndarray:
    """
    Scale displacement vectors from the starting position.

    If speed_factor == 1.0 → unchanged trajectory.
    If speed_factor == 0.0 → agent frozen at current position.
    If speed_factor == 0.6 → displacements shrunk by 40%.

    Args:
        trajectory: np.ndarray of shape (T, 2).
        speed_factor: Multiplicative scaling factor [0.0, 1.0].

    Returns:
        Modified trajectory of shape (T, 2).
    """
    origin = trajectory[0].copy()
    displacements = trajectory - origin
    return origin + (displacements * speed_factor)


# ---------------------------------------------------------------------------
# Counterfactual Simulator
# ---------------------------------------------------------------------------
class CounterfactualSimulator:
    """
    Tests hypothetical interventions on predicted trajectories and selects
    the least disruptive safe action.

    The simulator re-uses the RiskEngine to evaluate each perturbed scenario
    so that risk thresholds and CPA/TTC math are consistent throughout the
    pipeline.
    """

    def __init__(
        self,
        risk_config: Optional[RiskConfig] = None,
        candidates: Optional[List[InterventionCandidate]] = None,
    ):
        self.risk_config = risk_config or RiskConfig()
        self.risk_engine = RiskEngine(self.risk_config)
        self.candidates = candidates or DEFAULT_CANDIDATES

    def _simulate_candidate(
        self,
        candidate: InterventionCandidate,
        worker_traj: np.ndarray,
        machine_traj: np.ndarray,
        worker_id: int,
        machine_id: int,
    ) -> SimulationResult:
        """Run one counterfactual scenario and evaluate the resulting risk."""
        if candidate.target == "vehicle":
            sim_worker = worker_traj.copy()
            sim_machine = _perturb_trajectory(machine_traj, candidate.speed_factor)
        else:  # target == "worker"
            sim_worker = _perturb_trajectory(worker_traj, candidate.speed_factor)
            sim_machine = machine_traj.copy()

        pair_result = self.risk_engine.compute_pair_risk(
            worker_traj=sim_worker,
            machine_traj=sim_machine,
            worker_id=worker_id,
            machine_id=machine_id,
        )

        return SimulationResult(
            action=candidate.action,
            target=candidate.target,
            speed_factor=candidate.speed_factor,
            disruption_cost=candidate.disruption_cost,
            resulting_cpa=pair_result.cpa_distance,
            resulting_ttc=pair_result.time_to_conflict,
            resulting_risk_score=pair_result.risk_score,
            is_safe=not pair_result.is_dangerous,
        )

    def find_best_intervention(
        self,
        worker_traj: np.ndarray,
        machine_traj: np.ndarray,
        worker_id: int = 0,
        machine_id: int = 1,
    ) -> InterventionRecommendation:
        """
        Test all candidate interventions and return the least disruptive
        action that brings risk below the safety threshold.

        Args:
            worker_traj:  np.ndarray (T, 2) — worker's predicted future path.
            machine_traj: np.ndarray (T, 2) — machine's predicted future path.
            worker_id:    Tracking ID of the worker.
            machine_id:   Tracking ID of the machine.

        Returns:
            InterventionRecommendation with the best action, or a
            CRITICAL_DANGER fallback if no candidate is sufficient.
        """
        # First check: is intervention even needed?
        baseline = self.risk_engine.compute_pair_risk(
            worker_traj, machine_traj, worker_id, machine_id
        )
        if not baseline.is_dangerous:
            return InterventionRecommendation(
                status="SAFE",
                worker_id=worker_id,
                machine_id=machine_id,
                recommended_action="No intervention needed",
                speed_factor=1.0,
                deadline_seconds=0.0,
                projected_safe_distance=baseline.cpa_distance,
                projected_risk_score=baseline.risk_score,
                all_tested=[],
            )

        # Simulate every candidate
        all_results: List[SimulationResult] = []
        safe_results: List[SimulationResult] = []

        for candidate in self.candidates:
            result = self._simulate_candidate(
                candidate, worker_traj, machine_traj, worker_id, machine_id
            )
            all_results.append(result)
            if result.is_safe:
                safe_results.append(result)

        # Pick the least disruptive safe option
        if safe_results:
            best = min(safe_results, key=lambda r: r.disruption_cost)
            return InterventionRecommendation(
                status="INTERVENTION_REQUIRED",
                worker_id=worker_id,
                machine_id=machine_id,
                recommended_action=best.action,
                speed_factor=best.speed_factor,
                deadline_seconds=baseline.time_to_conflict,
                projected_safe_distance=best.resulting_cpa,
                projected_risk_score=best.resulting_risk_score,
                all_tested=all_results,
            )

        # No candidate restores safety — critical danger fallback
        least_risky = min(all_results, key=lambda r: r.resulting_risk_score)
        return InterventionRecommendation(
            status="CRITICAL_DANGER",
            worker_id=worker_id,
            machine_id=machine_id,
            recommended_action="Emergency stop vehicle immediately",
            speed_factor=0.0,
            deadline_seconds=max(baseline.time_to_conflict, 1.0),
            projected_safe_distance=least_risky.resulting_cpa,
            projected_risk_score=least_risky.resulting_risk_score,
            all_tested=all_results,
        )

    def evaluate_dangerous_pairs(
        self,
        predictions: Dict[int, np.ndarray],
        class_map: Dict[int, str],
    ) -> List[InterventionRecommendation]:
        """
        End-to-end: identify all dangerous pairs via the risk engine,
        then run counterfactual simulation on each.

        Args:
            predictions: {track_id: np.ndarray (T, 2)} predicted trajectories.
            class_map:   {track_id: class_name}.

        Returns:
            List of InterventionRecommendation, one per dangerous pair.
        """
        dangerous_pairs = self.risk_engine.get_dangerous_pairs(predictions, class_map)
        recommendations: List[InterventionRecommendation] = []

        for pair in dangerous_pairs:
            rec = self.find_best_intervention(
                worker_traj=predictions[pair.worker_id],
                machine_traj=predictions[pair.machine_id],
                worker_id=pair.worker_id,
                machine_id=pair.machine_id,
            )
            recommendations.append(rec)

        return recommendations
