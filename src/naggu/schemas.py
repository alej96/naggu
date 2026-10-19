"""Typed representations of Cosmos3 action data.

Field names and constraints are taken from the Cosmos3-Edge repo configs and
reference assets, not from the model card prose:

  transformer/config.json  ->  action_dim=64, num_embodiment_domains=32,
                               base_fps=24, temporal_compression_factor=4
  assets/*.json            ->  the two action modes and their real shapes
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

# From transformer/config.json.
NUM_EMBODIMENT_DOMAINS = 32
TEMPORAL_COMPRESSION_FACTOR = 4
BASE_FPS = 24

ActionMode = Literal["inverse_dynamics", "forward_dynamics"]

# Raw (pre-embedding) action widths, mirrored from diffusers'
# `_EMBODIMENT_TO_RAW_ACTION_DIM`. The model projects all of these into the
# shared action_dim=64 space, so raw width is a property of the embodiment.
# 15 domains are registered; transformer/config.json reserves 32 slots.
DOMAIN_ACTION_DIMS: dict[str, int] = {
    "agibot_gear_gripper": 29,
    "agibot_gear_gripper_ext": 29,
    "agibotworld": 29,
    "av": 9,
    "bridge_orig_lerobot": 10,
    "camera_pose": 9,
    "droid_lerobot": 10,
    "fractal": 10,
    "galbot": 30,
    "hand_pose": 57,
    "pusht": 2,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "umi": 10,
}

# Observed in the shipped reference outputs. Extend as you confirm more.
DOMAIN_IDS: dict[str, int] = {"av": 1}


class ValidationError(ValueError):
    """Raised when action data violates a constraint the model actually enforces."""


@dataclass(frozen=True)
class ActionTrajectory:
    """A flat (T, D) action sequence — the output of inverse dynamics."""

    data: np.ndarray  # (T, D) float32
    action_mode: ActionMode
    raw_action_dim: int
    domain_id: int
    dtype: str = "torch.float32"

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValidationError(f"expected (T, D), got shape {self.data.shape}")
        if self.data.shape[1] != self.raw_action_dim:
            raise ValidationError(
                f"raw_action_dim={self.raw_action_dim} but data has "
                f"{self.data.shape[1]} columns"
            )
        if not 0 <= self.domain_id < NUM_EMBODIMENT_DOMAINS:
            raise ValidationError(
                f"domain_id={self.domain_id} outside [0, {NUM_EMBODIMENT_DOMAINS})"
            )
        if not np.isfinite(self.data).all():
            raise ValidationError("trajectory contains NaN or Inf")

    @property
    def num_steps(self) -> int:
        return int(self.data.shape[0])

    @classmethod
    def from_json(cls, path: str | Path) -> ActionTrajectory:
        payload = json.loads(Path(path).read_text())
        data = np.asarray(payload["data"], dtype=np.float32)
        declared = tuple(payload["shape"])
        if data.shape != declared:
            raise ValidationError(
                f"{path}: declared shape {declared} != actual {data.shape}"
            )
        return cls(
            data=data,
            action_mode=payload["action_mode"],
            raw_action_dim=payload["raw_action_dim"],
            domain_id=payload["domain_id"],
            dtype=payload.get("dtype", "torch.float32"),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "data": self.data.tolist(),
                    "shape": list(self.data.shape),
                    "dtype": self.dtype,
                    "raw_action_dim": self.raw_action_dim,
                    "action_mode": self.action_mode,
                    "domain_id": self.domain_id,
                },
                indent=2,
            )
        )


@dataclass(frozen=True)
class ActionChunkSequence:
    """Chunked actions used to drive forward dynamics (actions -> video).

    The reference UMI example is 2 chunks x 16 steps x 10 dims. Each chunk is
    conditioned on the last generated frame of the previous chunk, so chunk
    count and chunk size are part of the contract, not a batching detail.
    """

    chunks: np.ndarray  # (num_chunks, chunk_size, D) float32
    prompt: str
    domain_name: str
    view_point: str
    fps: int
    image_size: int
    description: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.chunks.ndim != 3:
            raise ValidationError(
                f"expected (num_chunks, chunk_size, D), got {self.chunks.shape}"
            )
        expected = DOMAIN_ACTION_DIMS.get(self.domain_name)
        if expected is not None and self.chunks.shape[2] != expected:
            raise ValidationError(
                f"domain '{self.domain_name}' uses {expected}-D actions, "
                f"got {self.chunks.shape[2]}-D"
            )
        if not np.isfinite(self.chunks).all():
            raise ValidationError("action chunks contain NaN or Inf")

    @property
    def num_chunks(self) -> int:
        return int(self.chunks.shape[0])

    @property
    def chunk_size(self) -> int:
        return int(self.chunks.shape[1])

    @property
    def total_steps(self) -> int:
        return self.num_chunks * self.chunk_size

    @property
    def duration_seconds(self) -> float:
        return self.total_steps / self.fps

    def flatten(self) -> np.ndarray:
        """Concatenate chunks into one (T, D) trajectory."""
        return self.chunks.reshape(-1, self.chunks.shape[2])

    @classmethod
    def from_json(cls, path: str | Path) -> ActionChunkSequence:
        payload = json.loads(Path(path).read_text())
        chunks = np.asarray(payload["action_chunks"], dtype=np.float32)
        declared = tuple(payload["action_shape_per_chunk"])
        if chunks.shape[1:] != declared:
            raise ValidationError(
                f"{path}: action_shape_per_chunk {declared} != actual "
                f"{chunks.shape[1:]}"
            )
        if chunks.shape[0] != payload["num_chunks"]:
            raise ValidationError(
                f"{path}: num_chunks={payload['num_chunks']} != actual "
                f"{chunks.shape[0]}"
            )
        return cls(
            chunks=chunks,
            prompt=payload["prompt"],
            domain_name=payload["domain_name"],
            view_point=payload["view_point"],
            fps=payload["fps"],
            image_size=payload["image_size"],
            description=payload.get("description", ""),
            metadata={
                k: v
                for k, v in payload.items()
                if k
                not in {
                    "action_chunks",
                    "prompt",
                    "domain_name",
                    "view_point",
                    "fps",
                    "image_size",
                    "description",
                }
            },
        )
