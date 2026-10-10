"""Compare PyTorch logits with llama.cpp logits for the same token IDs.
Usage: python tools/compare_logits.py <llama_logits_probe> <ckpt.pt> <model.gguf>
"""
import subprocess, sys, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from mobile_lm import MobileLM, MobileLMConfig
from mobile_lm.qat import enable_qat
state = torch.load(sys.argv[2], map_location="cpu")
cfg = MobileLMConfig(**state["config"])
m = MobileLM(cfg)
if state.get("qat"):
    enable_qat(m)
m.load_state_dict(state["model"]); m.eval()
ids = [1, 5, 17, 42, 99, 123, 7]
with torch.no_grad():
    ref = m(torch.tensor([ids])).logits[0].numpy()
out = subprocess.run([sys.argv[1], sys.argv[3]] + [str(i) for i in ids],
                     capture_output=True, text=True, check=True).stdout
import numpy as np
got = np.array([[float(x) for x in line.split()] for line in out.strip().splitlines()])
print("shape ref", ref.shape, "got", got.shape)
print("max abs diff", np.abs(ref - got).max(), "mean abs diff", np.abs(ref - got).mean())
print("argmax agree:", (ref.argmax(-1) == got.argmax(-1)).mean())
