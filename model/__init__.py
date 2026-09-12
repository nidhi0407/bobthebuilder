"""
ForeSite AI — model package.

Exposes the three core components of the ML/safety pipeline:
  1. TrajectoryGRU         — GRU trajectory prediction model (Stage 4)
  2. RiskEngine            — CPA/TTC risk evaluator (Stage 5)
  3. CounterfactualSimulator — Prescriptive intervention selector (Stage 6a)
"""

from .trajectory_model import TrajectoryConfig, TrajectoryGRU
from .risk_engine import RiskConfig, RiskEngine, PairRiskResult
from .counterfactual_simulation import (
    CounterfactualSimulator,
    InterventionCandidate,
    InterventionRecommendation,
    SimulationResult,
)

__all__ = [
    # Stage 4 — Trajectory Prediction
    "TrajectoryConfig",
    "TrajectoryGRU",
    # Stage 5 — Risk Engine
    "RiskConfig",
    "RiskEngine",
    "PairRiskResult",
    # Stage 6a — Counterfactual Simulation
    "CounterfactualSimulator",
    "InterventionCandidate",
    "InterventionRecommendation",
    "SimulationResult",
]
