"""Probe which Cosmos3-Edge components actually load and run on this machine.

Prints a component-by-component verdict so the split between 'local work' and
'needs a rented GPU' is measured, not assumed.
"""
import time, traceback, torch
from pathlib import Path

MODEL = Path("models/Cosmos3-Edge")
dev = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"torch {torch.__version__} | device={dev}\n")

def probe(name, fn):
    t0 = time.time()
    try:
        detail = fn()
        print(f"  PASS  {name:<22} {time.time()-t0:6.1f}s  {detail}")
        return True
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"  FAIL  {name:<22} {time.time()-t0:6.1f}s  {msg[:150]}")
        return False

def vae():
    from diffusers import AutoencoderKLWan
    m = AutoencoderKLWan.from_pretrained(MODEL/"vae", torch_dtype=torch.float32).to(dev).eval()
    n = sum(p.numel() for p in m.parameters())
    # 1 sample, 12 latent-in channels per config, 17 frames, 64x64
    x = torch.randn(1, 3, 17, 64, 64, device=dev)
    with torch.no_grad():
        lat = m.encode(x).latent_dist.sample()
        rec = m.decode(lat).sample
    return f"{n/1e6:.0f}M params | {tuple(x.shape)} -> lat {tuple(lat.shape)} -> {tuple(rec.shape)}"

def tokenizer():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL/"text_tokenizer")
    ids = tk("put the flower into the red bottle").input_ids
    return f"vocab={tk.vocab_size} | 7 words -> {len(ids)} tokens"

def transformer_meta():
    """Load the 4B diffusion tower's weight index without materializing it."""
    import json
    idx = json.loads((MODEL/"transformer"/"diffusion_pytorch_model.safetensors.index.json").read_text())
    total = idx["metadata"]["total_size"]
    return f"{total/1e9:.2f} GB on disk, {len(idx['weight_map'])} tensors"

def transformer_load():
    from diffusers import AutoModel
    m = AutoModel.from_pretrained(MODEL/"transformer", torch_dtype=torch.bfloat16)
    n = sum(p.numel() for p in m.parameters())
    return f"{n/1e9:.2f}B params loaded to CPU"

print("Components:")
probe("tokenizer (cpu)", tokenizer)
probe("vae encode+decode", vae)
probe("transformer index", transformer_meta)
probe("transformer weights", transformer_load)
