"""Does the 4B diffusion tower actually load and step on Apple Silicon?"""
import time, torch
from diffusers import AutoModel

MODEL = "models/Cosmos3-Edge/transformer"
dev = "mps" if torch.backends.mps.is_available() else "cpu"

t0 = time.time()
m = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
n = sum(p.numel() for p in m.parameters())
print(f"loaded  {n/1e9:.2f}B params  ({time.time()-t0:.1f}s)  class={type(m).__name__}")

t0 = time.time()
try:
    m = m.to(dev).eval()
    print(f"moved to {dev}  ({time.time()-t0:.1f}s)")
    print(f"mps allocated: {torch.mps.current_allocated_memory()/1e9:.2f} GB")
except Exception as e:
    print(f"to({dev}) FAILED: {type(e).__name__}: {e}")
    raise SystemExit(1)

import inspect
sig = inspect.signature(m.forward)
print(f"\nforward() params: {list(sig.parameters)}")
