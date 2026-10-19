"""Trajectory comparison metrics and the tolerance that makes them meaningful.

The tolerance here is not a guess. The Cosmos3-Edge repo ships the *same*
inverse-dynamics rollout computed two ways:

    assets/edge_action_id_av_0_output.json                  (bf16, reference impl)
    assets/diffusers_outputs/..._inverse_0_diffusers.json   (fp32, diffusers impl)

Both are correct. The disagreement between them is the noise floor of a passing
implementation:

    mean|delta| = 0.0117    max|delta| = 0.0796

Any equality check tighter than that will fail correct code. `DEFAULT_TOLERANCE`
is derived from those files by `calibrate_from_reference_pair`, with headroom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import ActionTrajectory


@dataclass(frozen=True)
class Tolerance:
    """Pass/fail thresholds for comparing two action trajectories."""

    mean_abs: float
    max_abs: float

    @classmethod
    def calibrate_from_reference_pair(
        cls,
        a: ActionTrajectory,
        b: ActionTrajectory,
        headroom: float = 1.5,
    ) -> Tolerance:
        """Derive thresholds from two known-good implementations of one rollout."""
        d = np.abs(a.data - b.data)
        return cls(
            mean_abs=float(d.mean()) * headroom,
            max_abs=float(d.max()) * headroom,
        )


# Calibrated from the shipped reference pair (1.5x headroom).
DEFAULT_TOLERANCE = Tolerance(mean_abs=0.0175, max_abs=0.1194)


@dataclass(frozen=True)
class TrajectoryComparison:
    mean_abs_error: float
    max_abs_error: float
    per_dim_max_abs: np.ndarray
    terminal_drift: float
    passed: bool
    tolerance: Tolerance

    def report(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        dims = " ".join(f"{v:.4f}" for v in self.per_dim_max_abs)
        return (
            f"[{status}] mean|err|={self.mean_abs_error:.5f} "
            f"(tol {self.tolerance.mean_abs:.5f})  "
            f"max|err|={self.max_abs_error:.5f} "
            f"(tol {self.tolerance.max_abs:.5f})\n"
            f"       terminal drift={self.terminal_drift:.5f}\n"
            f"       per-dim max: {dims}"
        )


def compare(
    predicted: ActionTrajectory,
    reference: ActionTrajectory,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
) -> TrajectoryComparison:
    """Compare a predicted trajectory against a reference.

    Raises ValueError on a shape or semantic mismatch — that is a bug in the
    caller, not a soft failure worth scoring.
    """
    if predicted.data.shape != reference.data.shape:
        raise ValueError(
            f"shape mismatch: {predicted.data.shape} vs {reference.data.shape}"
        )
    if predicted.action_mode != reference.action_mode:
        raise ValueError(
            f"action_mode mismatch: {predicted.action_mode} vs "
            f"{reference.action_mode}"
        )
    if predicted.domain_id != reference.domain_id:
        raise ValueError(
            f"domain_id mismatch: {predicted.domain_id} vs {reference.domain_id}"
        )

    d = np.abs(predicted.data - reference.data)
    mean_abs = float(d.mean())
    max_abs = float(d.max())

    # Integrated position error at the end of the rollout: small per-step errors
    # that all point the same way matter far more than large ones that cancel.
    drift = float(
        np.linalg.norm(
            predicted.data.cumsum(axis=0)[-1] - reference.data.cumsum(axis=0)[-1]
        )
    )

    return TrajectoryComparison(
        mean_abs_error=mean_abs,
        max_abs_error=max_abs,
        per_dim_max_abs=d.max(axis=0),
        terminal_drift=drift,
        passed=mean_abs <= tolerance.mean_abs and max_abs <= tolerance.max_abs,
        tolerance=tolerance,
    )
