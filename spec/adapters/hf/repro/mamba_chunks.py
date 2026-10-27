import torch
import transformers
from transformers import MambaConfig, MambaForCausalLM

torch.manual_seed(0)
torch.set_num_threads(2)
cfg = MambaConfig(vocab_size=64, hidden_size=16, state_size=8, num_hidden_layers=2, conv_kernel=4, expand=2,
                  initializer_range=0.5)
model = MambaForCausalLM(cfg).eval()
ids = torch.randint(1, 64, (1, 16))


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()      # max |a - b| / max |b|


with torch.no_grad():
    full = model(ids).logits                                  # one forward over 16 tokens

    out = model(ids[:, :8], use_cache=True)                   # 8 tokens, then 8 more over the cache
    chunk = model(ids[:, 8:], cache_params=out.cache_params, use_cache=True).logits

    out = model(ids[:, :8], use_cache=True)                   # 8 tokens, then 1 token at a time
    steps = []
    for t in range(8, 16):
        out = model(ids[:, t:t + 1], cache_params=out.cache_params, use_cache=True)
        steps.append(out.logits)
    steps = torch.cat(steps, dim=1)

print("transformers", transformers.__version__, "torch", torch.__version__)
print("8 + 8 tokens over the cache vs one forward:  ", f"{rel(chunk, full[:, 8:]):.2e}")
print("8 + 1 x 8 tokens over the cache vs one forward:", f"{rel(steps, full[:, 8:]):.2e}")
