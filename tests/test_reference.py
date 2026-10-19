"""Golden-file tests against the assets NVIDIA ships with Cosmos3-Edge.

These run on CPU in under a second. They pin down the data contract so that
when GPU time is finally rented, the only unknown left is the model itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from naggu import (
    ActionChunkSequence,
    ActionTrajectory,
    Tolerance,
    ValidationError,
    compare,
)
from naggu.metrics import DEFAULT_TOLERANCE

REF = Path(__file__).parent.parent / "data" / "reference"
AV_BF16 = REF / "edge_action_id_av_0_output.json"
AV_FP32 = REF / "edge_action_id_av_inverse_0_diffusers.json"
AV_OTHER = REF / "edge_action_id_av_1_output.json"
UMI_CHUNKS = REF / "example_action_fd_umi_action_chunks.json"


class TestInverseDynamicsLoading:
    def test_reference_matches_declared_shape(self):
        traj = ActionTrajectory.from_json(AV_BF16)
        assert traj.data.shape == (60, 9)
        assert traj.num_steps == 60
        assert traj.action_mode == "inverse_dynamics"
        assert traj.raw_action_dim == 9
        assert traj.domain_id == 1

    def test_both_precisions_load_to_same_shape(self):
        assert (
            ActionTrajectory.from_json(AV_BF16).data.shape
            == ActionTrajectory.from_json(AV_FP32).data.shape
        )

    def test_roundtrip_through_disk(self, tmp_path):
        original = ActionTrajectory.from_json(AV_BF16)
        out = tmp_path / "rt.json"
        original.to_json(out)
        assert np.array_equal(ActionTrajectory.from_json(out).data, original.data)


class TestValidation:
    def test_rejects_wrong_column_count(self):
        with pytest.raises(ValidationError, match="raw_action_dim"):
            ActionTrajectory(
                data=np.zeros((10, 7), dtype=np.float32),
                action_mode="inverse_dynamics",
                raw_action_dim=9,
                domain_id=1,
            )

    def test_rejects_domain_id_past_embodiment_count(self):
        with pytest.raises(ValidationError, match="outside"):
            ActionTrajectory(
                data=np.zeros((10, 9), dtype=np.float32),
                action_mode="inverse_dynamics",
                raw_action_dim=9,
                domain_id=32,
            )

    def test_rejects_nan(self):
        bad = np.zeros((10, 9), dtype=np.float32)
        bad[3, 4] = np.nan
        with pytest.raises(ValidationError, match="NaN"):
            ActionTrajectory(
                data=bad,
                action_mode="inverse_dynamics",
                raw_action_dim=9,
                domain_id=1,
            )

    def test_rejects_umi_chunks_with_wrong_width(self):
        with pytest.raises(ValidationError, match="10-D"):
            ActionChunkSequence(
                chunks=np.zeros((2, 16, 9), dtype=np.float32),
                prompt="x",
                domain_name="umi",
                view_point="ego_view",
                fps=20,
                image_size=256,
            )


class TestForwardDynamicsChunks:
    def test_umi_reference_shape(self):
        seq = ActionChunkSequence.from_json(UMI_CHUNKS)
        assert seq.chunks.shape == (2, 16, 10)
        assert seq.num_chunks == 2
        assert seq.chunk_size == 16
        assert seq.total_steps == 32
        assert seq.domain_name == "umi"
        assert seq.view_point == "ego_view"
        assert seq.fps == 20

    def test_duration_derives_from_fps(self):
        seq = ActionChunkSequence.from_json(UMI_CHUNKS)
        assert seq.duration_seconds == pytest.approx(32 / 20)

    def test_flatten_preserves_order(self):
        seq = ActionChunkSequence.from_json(UMI_CHUNKS)
        flat = seq.flatten()
        assert flat.shape == (32, 10)
        assert np.array_equal(flat[0], seq.chunks[0, 0])
        assert np.array_equal(flat[16], seq.chunks[1, 0])


class TestToleranceCalibration:
    """The reference pair defines what 'correct' numerically means."""

    def test_two_correct_implementations_agree_within_default_tolerance(self):
        result = compare(
            ActionTrajectory.from_json(AV_FP32),
            ActionTrajectory.from_json(AV_BF16),
        )
        assert result.passed, result.report()

    def test_default_tolerance_is_not_stricter_than_the_noise_floor(self):
        calibrated = Tolerance.calibrate_from_reference_pair(
            ActionTrajectory.from_json(AV_FP32),
            ActionTrajectory.from_json(AV_BF16),
        )
        assert DEFAULT_TOLERANCE.mean_abs >= calibrated.mean_abs * 0.99
        assert DEFAULT_TOLERANCE.max_abs >= calibrated.max_abs * 0.99

    def test_a_different_rollout_is_correctly_rejected(self):
        """Guards against a tolerance so loose it passes anything."""
        result = compare(
            ActionTrajectory.from_json(AV_OTHER),
            ActionTrajectory.from_json(AV_BF16),
        )
        assert not result.passed

    def test_mismatched_shapes_raise_rather_than_score(self):
        traj = ActionTrajectory.from_json(AV_BF16)
        truncated = ActionTrajectory(
            data=traj.data[:30],
            action_mode=traj.action_mode,
            raw_action_dim=traj.raw_action_dim,
            domain_id=traj.domain_id,
        )
        with pytest.raises(ValueError, match="shape mismatch"):
            compare(truncated, traj)
