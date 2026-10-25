import json, subprocess, sys, time, urllib.request, signal
sys.path.insert(0, "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/experiments/compat_2026-10-26")
sys.path.insert(0, "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch/tools")
import llamacpp_gate as G, chat_template as CT, llama3_bpe
m = G.conversations()[24]
mine = CT.render(m, generation_prompt=True)
tok = llama3_bpe.from_gguf(G.SFT)
B = "/home/jonat/llamacpp_b6000_build/build/bin/llama-server"
p = subprocess.Popen([B, "-m", G.SFT, "--port", "8133", "-t", "1", "-c", "512", "-np", "1", "-ctk", "f32", "-ctv", "f32", "--no-webui"], stdout=open("/home/jonat/cp_c24.log", "w"), stderr=subprocess.STDOUT)
def post(path, body):
    r = urllib.request.Request("http://127.0.0.1:8133" + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))
try:
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:8133/health", timeout=2); break
        except Exception: time.sleep(0.5)
    r = post("/completion", {"prompt": mine, "n_predict": 24, "temperature": 0, "top_k": 1, "samplers": ["top_k", "temperature"], "cache_prompt": False, "return_tokens": True, "logit_bias": [[0, False]]})
    print("llama.cpp:", repr(r["content"]))
    ids = r["tokens"]
    # margins along llama.cpp's own greedy path
    pre = tok.encode(mine, add_special_tokens=False).ids
    for k in range(len(ids)):
        q = post("/completion", {"prompt": pre + ids[:k], "n_predict": 1, "temperature": 0, "top_k": 2, "n_probs": 2, "samplers": ["top_k"], "post_sampling_probs": False, "cache_prompt": False, "logit_bias": [[0, False]]})
        pr = q["completion_probabilities"][0]["top_logprobs"]
        if len(pr) > 1 and pr[0]["logprob"] - pr[1]["logprob"] < 0.05:
            print("step", k, "near tie:", [(x["token"], round(x["logprob"], 5)) for x in pr])
finally:
    p.send_signal(signal.SIGINT); p.wait(20)
