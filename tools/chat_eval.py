#!/usr/bin/env python3
"""Side-by-side evaluation of chat checkpoints on the frozen held-out prompts.

    python tools/chat_eval.py --mtstudio build/mtstudio \\
        --model base=runs/base/spec.json --model sft=runs/sft/spec.json [--model po=...] \\
        --prompts corpus/eval/tinychat_hf_prompts_v1.jsonl --out runs/chat_eval [--n 2 --temp 0.7]

Every model gets the same prompts in the served chat template
(<|user|>P<|assistant|>) and the same seeds. Per model it reports:
  clean_stop   reply ended at <|endoftext|> within the budget, with no role
               marker inside (the turn protocol: what SFT exists to teach)
  distinct2    distinct bigrams / bigrams in the reply (1 = no repetition)
  repeat_frac  share of replies repeating a word trigram or a word twice
               running (the gate for preference stages)
  words        mean reply length in words
  prompt_overlap  share of the prompt's content words the reply takes up
  ref_f1       content-word F1 between the reply and the dataset's reply
               (both crude lexical proxies for topic and relevance)
  ref_bpb      bits per byte of the REFERENCE reply under the model, via a
               masked-SFT evaluation pass (how well it predicts a good answer).

When the prompts carry more than one "source", every metric is also
reported per source (by_source; one extra ref_bpb pass each).
It writes side_by_side.md: each prompt, the reference, and every model's
reply, for reading. Metrics are descriptive; a pre-registered comparison
reads them from summary.json.

With --ember-bin, replies are sampled by an ember.cpp server started per
model on its GGUF (out_dir/name.gguf, which must carry the BPE tokenizer)
through tools/chat_sample_ember.py: pure temperature, the same seeds per
prompt for every model, minutes instead of mtstudio's uncached sampler.
ember also stops at a generated <|user|>/<|assistant|> (stop "turn"),
which counts as not clean. ref_bpb is always mtstudio's scoring pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpe  # noqa: E402


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def distinct2(text):
    w = text.split()
    bg = list(zip(w, w[1:]))
    return len(set(bg)) / len(bg) if bg else 1.0


def repeats(text):
    """True if the reply repeats itself: a word trigram twice, the same word
    twice running ("today today"), or a bigram of two content words twice
    ("dark clouds and dark clouds"; function-word bigrams such as "I think"
    recur in normal replies). The gate for preference stages: an arm that
    repeats more than its SFT start has failed, whatever its pair accuracy."""
    w = [t.strip(".,!?;:'\"()").lower() for t in text.split()]
    w = [t for t in w if t]
    if any(a == b for a, b in zip(w, w[1:])):
        return True
    tri = list(zip(w, w[1:], w[2:]))
    if len(set(tri)) < len(tri):
        return True
    bi = [(a, b) for a, b in zip(w, w[1:]) if a not in STOP and b not in STOP]
    return len(set(bi)) < len(bi)


# Function words of BASIC English chat: dropped before the overlap measures
# so they count shared content, not "I", "the", "you".
STOP = set("""a an the and or but if so of to in on at by for with from as is am are was were be been being
do does did have has had i you he she it we they me him her us them my your his its our their this that these
those there here what which who whom how why when where can could will would shall should may might must not no
yes very just too also really about up out some any all more most much many such than then now well okay oh
feel feels think know like get got go going make sure maybe sometimes always often""".split())


def content(text):
    return {w for w in (t.strip(".,!?;:'\"()").lower() for t in text.split()) if w and w not in STOP}


def prompt_overlap(reply, prompt):
    """Share of the prompt's content words the reply takes up: a crude topic signal."""
    p = content(prompt)
    return len(p & content(reply)) / len(p) if p else 0.0


def ref_f1(reply, ref):
    """Content-word F1 between the reply and the dataset's reply: a crude relevance signal."""
    a, b = content(reply), content(ref)
    if not a or not b:
        return 0.0
    k = len(a & b)
    return 0.0 if k == 0 else 2 * k / (len(a) + len(b))


def ref_bpb(mtstudio, spec_path, prompts, work, tag):
    """Bits per byte of the reference replies: one 1-step SFT run at lr 0 whose
    held-out evaluation covers every example (masked CE, nats/token), turned
    into bits per byte with the reference's own byte and token counts."""
    spec = json.load(open(spec_path, encoding="utf-8"))
    ck = os.path.join(spec["out_dir"], spec["name"] + ".safetensors")
    vocab = spec["data"]["vocab"]
    tok = bpe.read_gguf(vocab)
    # mtstudio refuses an example longer than data.T + 1 tokens (prompt +
    # reference + end-of-text). Long references (chat_prompts_v1's UltraChat
    # half) are cut to the longest token prefix that fits and still decodes to
    # whole characters; a prompt that alone leaves no room is left out. The
    # summary counts both.
    room = spec["data"]["T"] + 1
    ex, cut = [], 0
    for p in prompts:
        pr = f"<|user|>{p['prompt']}<|assistant|>"
        ref = p["reference"]
        n = room - len(tok.encode(pr)) - 1
        if n < 1:
            continue
        ids = tok.encode(ref)
        if len(ids) > n:
            cut += 1
            while n > 0:
                ref = tok.decode(ids[:n]).rstrip("�")
                if ref and len(tok.encode(ref)) <= room - len(tok.encode(pr)) - 1:
                    break
                n -= 1
            if not ref:
                continue
        ex.append({"prompt": pr, "completion": ref})
    ref_bpb.last = {"truncated": cut, "dropped": len(prompts) - len(ex)}
    # 19 copies of the first example per real one, then the real ones: the
    # held-out tail mtstudio scores (1/20 of the file) is exactly the real set.
    path = os.path.join(work, f"{tag}_ref.jsonl")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for e in [ex[0]] * (19 * len(ex)) + ex:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    ev = {"name": f"{tag}_refeval", "arch": spec["arch"],
          "data": {"vocab": vocab, "vocab_cap": spec["data"].get("vocab_cap", 8192), "T": spec["data"]["T"],
                   "sft_examples": path},
          "train": {"steps": 1, "lr": 0.0, "batch": 1, "eval_every": 1, "checkpoint_every": 1000,
                    "gradmap_every": 0, "seed": 1, "objective": "sft", "init_from": ck},
          "export": {"formats": []}, "out_dir": os.path.join(work, f"{tag}_refeval")}
    sp = os.path.join(work, f"{tag}_refeval.json")
    json.dump(ev, open(sp, "w"))
    subprocess.run([mtstudio, "run", sp], check=True, capture_output=True)
    evs = [json.loads(x) for x in open(os.path.join(ev["out_dir"], "events.jsonl"), encoding="utf-8")]
    nats = next(e["val_loss"] for e in evs if e["event"] == "eval")
    held = ex
    toks = sum(len(tok.encode(e["completion"])) + 1 for e in held)
    byts = sum(len(e["completion"].encode()) + 1 for e in held)
    return nats * toks / byts / math.log(2), len(held)


def check_rope_heads(spec_path):
    """A spec over bare weights (no events.jsonl in out_dir) with no
    arch.rope_heads loads as legacy "first" (mtstudio resolve_rope_heads),
    which samples a model trained with "all" under the wrong architecture
    and corrupts every metric but ref_bpb. Refuse it."""
    spec = json.load(open(spec_path, encoding="utf-8"))
    if spec.get("arch", {}).get("rope_heads"):
        return
    if not os.path.exists(os.path.join(spec["out_dir"], "events.jsonl")):
        sys.exit(f"{spec_path}: out_dir has no events.jsonl and arch.rope_heads is unset, so mtstudio would load "
                 "the weights as rope_heads=\"first\"; state arch.rope_heads or copy the run's events.jsonl")


def sample_ember(ember_bin, port, spec_path, prompts, n, temp, max_new, out, greedy=False, repeat=(1.0, 64)):
    """Serve the model's GGUF with ember and sample n replies per prompt."""
    import time
    import urllib.request
    spec = json.load(open(spec_path, encoding="utf-8"))
    gguf = os.path.join(spec["out_dir"], spec["name"] + ".gguf")
    bpe.read_gguf(gguf)  # raises unless the GGUF carries the BPE tokenizer
    www = os.path.join(os.path.dirname(out), "www")
    os.makedirs(www, exist_ok=True)
    url = f"http://127.0.0.1:{port}"
    # Another server already on the port would answer for ours (18 Oct 2026:
    # a shared machine's ember instance held the port and served the first
    # samples of the wrong model), so the port must be free, and /health must
    # name our GGUF before any sample is drawn.
    try:
        urllib.request.urlopen(url + "/health", timeout=2).read()
        sys.exit(f"port {port} already answers /health: another server is there; pick a free --port")
    except OSError:
        pass
    # ember's last argument is the GPU layer count: -1 (all) for a CUDA build, 0 for a CPU-only
    # build, which otherwise refuses to generate and returns an empty reply (22 Oct 2026).
    srv = subprocess.Popen([ember_bin, gguf, str(port), "127.0.0.1", www, os.environ.get("EMBER_GPU_LAYERS", "-1")],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
    try:
        health = None
        for _ in range(120):
            try:
                health = json.loads(urllib.request.urlopen(url + "/health", timeout=2).read())
                break
            except OSError:
                if srv.poll() is not None:
                    sys.exit(f"ember exited before serving {gguf}")
                time.sleep(1)
        if not health or os.path.realpath(health.get("model", "")) != os.path.realpath(gguf):
            sys.exit(f"/health on port {port} names {health and health.get('model')}, not {gguf}")
        ex = os.path.join(os.path.dirname(out), os.path.basename(out) + ".examples.jsonl")
        with open(ex, "w", encoding="utf-8", newline="\n") as f:
            for _ in range(n):
                for p in prompts:
                    f.write(json.dumps({"prompt": f"<|user|>{p['prompt']}<|assistant|>", "completion": p["reference"]},
                                       ensure_ascii=False) + "\n")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_sample_ember.py"),
                        ex, "--url", url, "--vocab", spec["data"]["vocab"], "--out", out, "--temp", repr(temp),
                        "--max-new", str(max_new), "--seed", "7"] + (["--greedy"] if greedy else []) +
                       ["--repeat-penalty", repr(repeat[0]), "--repeat-last-n", str(repeat[1])],
                       check=True, capture_output=True)
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    recs = read_jsonl(out)
    for r in recs:  # back to an index into prompts (the examples repeat them n times)
        r["prompt_index"] %= len(prompts)
    return recs


def reply_metrics(recs, clean, prompts):
    n = len(recs)
    return {"clean_stop": sum(clean) / n,
            "distinct2": sum(distinct2(r["completion"]) for r in recs) / n,
            "repeat_frac": sum(repeats(r["completion"]) for r in recs) / n,
            "prompt_overlap": sum(prompt_overlap(r["completion"], prompts[r["prompt_index"]]["prompt"])
                                  for r in recs) / n,
            "ref_f1": sum(ref_f1(r["completion"], prompts[r["prompt_index"]]["reference"]) for r in recs) / n,
            "words": sum(len(r["completion"].split()) for r in recs) / n,
            "samples": n}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mtstudio", required=True)
    ap.add_argument("--model", action="append", required=True, help="tag=spec.json")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--ember-bin", default="", help="sample with this ember.cpp tinyllama_server instead")
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--repeat-penalty", type=float, default=1.0, help="contract repetition penalty (ember only; 1.0 = off)")
    ap.add_argument("--repeat-last-n", type=int, default=64)
    a = ap.parse_args(argv)
    for m in a.model:
        check_rope_heads(m.split("=", 1)[1])
    os.makedirs(a.out, exist_ok=True)
    prompts = read_jsonl(a.prompts)[:a.limit]
    pf = os.path.join(a.out, "prompts_templated.jsonl")
    with open(pf, "w", encoding="utf-8", newline="\n") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": f"<|user|>{p['prompt']}<|assistant|>", "uri": p["uri"]}) + "\n")
    summary, replies = {}, {}
    for m in a.model:
        tag, spec = m.split("=", 1)
        out = os.path.join(a.out, f"{tag}_samples.jsonl")
        if a.ember_bin:
            recs = sample_ember(a.ember_bin, a.port, spec, prompts, a.n, a.temp, a.max_new, out,
                                repeat=(a.repeat_penalty, a.repeat_last_n))
        else:
            subprocess.run([a.mtstudio, "sample-set", spec, "--prompts", pf, "--out", out, "--n", str(a.n),
                            "--temp", repr(a.temp), "--seed", "7", "--max-new", str(a.max_new)], check=True,
                           capture_output=True)
            recs = read_jsonl(out)
        clean = [r["stop"] == "eos" and not any(t in r["completion"] for t in ("<|user|>", "<|assistant|>"))
                 for r in recs]
        bpb, nref = ref_bpb(a.mtstudio, spec, prompts, a.out, tag)
        summary[tag] = {**reply_metrics(recs, clean, prompts), "ref_bpb": bpb, "ref_examples": nref,
                        "ref_truncated": ref_bpb.last["truncated"], "ref_dropped": ref_bpb.last["dropped"]}
        # Per source: chat_prompts_v1's everyday half is greetings whose
        # reference reply ("Hello! How can I help you today?") is in the
        # everyday training data thousands of times, below decontam's
        # thresholds, so a gate reads the other half.
        sources = sorted({p.get("source", "") for p in prompts})
        if len(sources) > 1:
            summary[tag]["by_source"] = {}
            for src in sources:
                idx = [i for i, p in enumerate(prompts) if p.get("source", "") == src]
                keep = [k for k, r in enumerate(recs) if prompts[r["prompt_index"]].get("source", "") == src]
                sb, sn = ref_bpb(a.mtstudio, spec, [prompts[i] for i in idx], a.out, f"{tag}_{src}")
                summary[tag]["by_source"][src] = {**reply_metrics([recs[k] for k in keep], [clean[k] for k in keep],
                                                                  prompts), "ref_bpb": sb, "ref_examples": sn,
                                                 "ref_truncated": ref_bpb.last["truncated"],
                                                 "ref_dropped": ref_bpb.last["dropped"]}
        replies[tag] = {}
        for r in recs:
            replies[tag].setdefault(r["prompt_index"], r["completion"])
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, indent=1)
    L = ["# Chat side by side", "",
         "| model | clean stop | distinct-2 | repeats | words | prompt overlap | ref F1 | ref bits/byte |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t, s in summary.items():
        rows = [(t, s)] + [(f"{t} ({src})", v) for src, v in s.get("by_source", {}).items()]
        for name, v in rows:
            L.append(f"| {name} | {v['clean_stop']:.2f} | {v['distinct2']:.2f} | {v['repeat_frac']:.2f} | "
                     f"{v['words']:.1f} | {v['prompt_overlap']:.3f} | {v['ref_f1']:.3f} | {v['ref_bpb']:.3f} |")
    L += ["", "Prompt overlap and ref F1 are content-word measures, crude proxies for topic and relevance, not judges."]
    L.append("")
    for i, p in enumerate(prompts):
        L += [f"### {i + 1}. {p['prompt']}", f"- reference: {p['reference']}"]
        for t in summary:
            L.append(f"- **{t}**: {replies[t].get(i, '')}")
        L.append("")
    open(os.path.join(a.out, "side_by_side.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
