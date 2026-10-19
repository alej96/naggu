"""Inverse dynamics (video -> actions) on the shipped AV reference clip.

Device-agnostic: picks cuda if present, else mps, else cpu. The same script is
what you run on a rented GPU — only the device changes.

Scored against NVIDIA's own output with the tolerance calibrated in naggu.metrics.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from naggu import ActionTrajectory, compare  # noqa: E402

MODEL = Path("models/Cosmos3-Edge")
REF_VIDEO = Path("data/reference/example_action_id_av_0_input.mp4")
REF_ACTIONS = Path("data/reference/edge_action_id_av_0_output.json")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_frames(path: Path, n: int) -> list:
    import av
    from PIL import Image

    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_image())
            if len(frames) >= n:
                break
    if len(frames) < n:
        raise RuntimeError(f"{path}: needed {n} frames, got {len(frames)}")
    return frames


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--chunk-size", type=int, default=60)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--resolution-tier", type=int, default=480,
                    choices=[256, 480, 704, 720])
    ap.add_argument("--out", default="outputs/inverse_dynamics_av_0.json")
    args = ap.parse_args()

    device = args.device or pick_device()
    dtype = getattr(torch, args.dtype)
    print(
        f"device={device} dtype={args.dtype} steps={args.steps} "
        f"chunk={args.chunk_size} tier={args.resolution_tier}"
    )

    from diffusers import Cosmos3OmniPipeline
    from diffusers.pipelines.cosmos import CosmosActionCondition

    t0 = time.time()
    pipe = Cosmos3OmniPipeline.from_pretrained(str(MODEL), torch_dtype=dtype)
    pipe = pipe.to(device)
    print(f"pipeline loaded in {time.time() - t0:.1f}s")

    # A chunk of N transitions is conditioned on N+1 frames.
    frames = load_frames(REF_VIDEO, args.chunk_size + 1)
    print(f"loaded {len(frames)} frames @ {frames[0].size}")

    condition = CosmosActionCondition(
        mode="inverse_dynamics",
        chunk_size=args.chunk_size,
        domain_name="av",
        resolution_tier=args.resolution_tier,
        video=frames,
    )

    t0 = time.time()
    result = pipe(
        prompt="",
        action=condition,
        num_inference_steps=args.steps,
        generator=torch.Generator(device="cpu").manual_seed(0),
        output_type="np",
    )
    elapsed = time.time() - t0
    print(f"inference took {elapsed:.1f}s ({elapsed / args.steps:.2f}s/step)")
    if device == "mps":
        print(f"peak MPS memory: {torch.mps.driver_allocated_memory() / 1e9:.2f} GB")

    actions = np.asarray(result.action[0], dtype=np.float32)
    print(f"predicted actions: {actions.shape}")

    predicted = ActionTrajectory(
        data=actions,
        action_mode="inverse_dynamics",
        raw_action_dim=actions.shape[1],
        domain_id=1,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    predicted.to_json(args.out)

    reference = ActionTrajectory.from_json(REF_ACTIONS)
    print()
    if predicted.data.shape != reference.data.shape:
        print(
            f"[SKIP] shape {predicted.data.shape} != reference "
            f"{reference.data.shape}; scoring needs --chunk-size "
            f"{reference.num_steps}. Wrote {args.out}."
        )
        return 0
    print(compare(predicted, reference).report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
