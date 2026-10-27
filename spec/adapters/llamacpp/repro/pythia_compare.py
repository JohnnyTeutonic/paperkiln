# Compare llama.cpp's logits for Pythia (dump_logits output) with transformers in float32.
# usage: python pythia_compare.py MODEL_DIR ids    -> writes ids.txt
#        python pythia_compare.py MODEL_DIR cmp LOGITS.f32 [LOGITS.f32 ...]
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

d, mode = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(d)
text = ("The history of the city goes back to a small settlement on the river, where traders met to exchange "
        "grain, salt and timber. Over the centuries the market grew into a town, and the town into a city.")
ids = tok(text)["input_ids"]
if mode == "ids":
    open("ids.txt", "w").write(" ".join(map(str, ids)) + "\n")
    print(len(ids), "ids")
    sys.exit(0)
torch.set_num_threads(2)
for act in ("gelu", "gelu_pytorch_tanh"):
    m = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32, hidden_act=act).eval()
    with torch.no_grad():
        want = m(torch.tensor([ids])).logits[0].numpy()
    for f in sys.argv[3:]:
        got = np.fromfile(f, dtype=np.float32).reshape(len(ids), -1)[:, :want.shape[1]]
        err = np.abs(got - want).max() / np.abs(want).max()
        flips = int((got.argmax(-1) != want.argmax(-1)).sum())
        print(f"transformers hidden_act={act:18s} vs {f}: max |diff| / max |logit| {err:.2e}, "
              f"argmax differs at {flips} of {len(ids)}")
