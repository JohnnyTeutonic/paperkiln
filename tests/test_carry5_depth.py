"""tools/carry5_depth.py, H-MoR2's depth trace (experiments/tiny_chat_2026-10/PREREGISTRATION.md amendment 4,
A4.5), and its place in tiny_chat_run.py post.

Without a binary: the positions on every context-family item of Carry-5 (answer positions predict exactly
the gold value's tokens in the target reply; other assistant positions predict every other token of the
canned replies and the target reply, each closing <|endoftext|> included, and nothing else), and the
statistics on hand-made traces (Delta pooled over positions, the OLS slope over items, the normalised
entropy).

With MTSTUDIO_BIN (or ./build/mtstudio): a tiny causal token-choice MoR model (1 + [2] x {1, 2, 3} + 1,
d 32) trained on CPU on rendered Carry-5 conversations; the tool scores a subset of the context families
through `mtstudio score`, and its depths at the registered positions are checked against the score's own
per-position trace; then tiny_chat_run.py post on that model (Carry-5 itself through a stand-in, the depth
step for real) records pt_mor (m_s) and mor_depth in summary.json, and its numbers agree with the tool's.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import bpe  # noqa: E402
import carry5_depth as cd  # noqa: E402
import carry5_eval as ce  # noqa: E402
from test_mtstudio_chat import BIN  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
SET = ROOT / "corpus" / "eval" / "carry5_v1.jsonl"
TOOL = [sys.executable, str(ROOT / "tools" / "carry5_depth.py")]
RUN = [sys.executable, str(ROOT / "tools" / "colab_chat" / "tiny_chat_run.py")]


def context_items():
    return [it for it in ce.read_jsonl(SET) if it["family"] in cd.CONTEXT_FAMILIES]


def conversations(items):
    """The gold conversations rendered in the served template (the model's training text)."""
    return "".join(ce.template([(t["user"], t["gold"]) for t in it["turns"]], "")[:-len(ce.ASSISTANT) - len(ce.USER)]
                   + "\n" for it in items)


class BpeTok:
    """carry5_eval.Tok's interface over a tools/bpe.py tokeniser (no GGUF round trip needed here)."""

    def __init__(self, t):
        self.encode, self.decode = t.encode, t.decode
        self.eot, self.user, self.assistant = 1, 2, 3


@pytest.fixture(scope="module")
def tok():
    return bpe.train(conversations(context_items()[::4]), 1200)


def test_positions_on_every_context_item(tok):
    t = BpeTok(tok)
    items = context_items()
    assert len(items) == 828  # A4.5: recall, correction, topic return
    for it in items:
        for mode in cd.MODES:
            r = cd.render(t, it, mode)
            hist, user = cd.history_of(it, mode)
            assert r["prompt_ids"] == t.encode(ce.template(hist, user))  # carry5_eval's scored prompt
            reply = it["frame"].format(cd.gold_value(it))
            assert r["continuation_ids"] == t.encode(reply) + [t.eot]
            full = r["prompt_ids"] + r["continuation_ids"]
            nxt = [full[i + 1] for i in r["answer_pos"]]
            assert t.decode(nxt).strip() == cd.gold_value(it)  # exactly the value's tokens
            assert r["answer_pos"] == list(range(r["answer_pos"][0], r["answer_pos"][-1] + 1))
            assert r["answer_pos"][0] >= len(r["prompt_ids"]) - 1  # inside the target reply
            # The other assistant positions predict the canned replies, the target reply's frame and every
            # closing <|endoftext|>: nothing of a user turn or a role marker.
            other = "".join(t.decode([full[i + 1]]) for i in r["other_pos"])
            want = "".join(a + ce.EOT for _, a in hist) + reply.replace(cd.gold_value(it), "", 1) + ce.EOT
            assert other.replace(" ", "") == want.replace(" ", "")
            assert not set(r["answer_pos"]) & set(r["other_pos"])
            assert all(full[i + 1] not in (t.user, t.assistant) for i in r["other_pos"] + r["answer_pos"])
        assert len(cd.render(t, it, "twin")["other_pos"]) < len(cd.render(t, it, "gold")["other_pos"])


def test_statistics_on_hand_made_traces():
    def rec(fam, dist, ans, oth, hist):
        return {"id": f"{fam}{dist}", "family": fam, "distance": dist,
                "modes": {"gold": {"answer": ans, "other_n": len(oth), "other_sum": sum(oth), "hist": hist}}}
    recs = [rec("recall", 1, [2, 2], [1, 1, 2], [3, 3, 0]), rec("recall", 2, [3], [1], [1, 1, 2]),
            rec("topic", 3, [3, 3, 3], [2, 2], [0, 3, 4])]
    s = cd.analyse(recs, 3)["modes"]["gold"]
    ans, oth = [2, 2, 3, 3, 3, 3], [1, 1, 2, 1, 2, 2]  # pooled over positions, not over items
    assert s["answer"]["n"] == 6 and abs(s["answer"]["mean"] - sum(ans) / 6) < 1e-12
    assert abs(s["delta"] - (sum(ans) / 6 - sum(oth) / 6)) < 1e-12
    # Item means 2, 3, 3 on distances 1, 2, 3: slope 0.5.
    assert abs(s["slope_answer_on_distance"]["slope"] - 0.5) < 1e-12
    hist = [4, 7, 6]
    h = -sum(c / 17 * math.log(c / 17) for c in hist) / math.log(3)
    assert s["all_positions"]["hist"] == hist and abs(s["all_positions"]["entropy_norm"] - h) < 1e-12
    assert s["answer"]["hist"] == [0, 2, 4] and set(s["family"]) == {"recall", "topic"}
    assert cd.entropy_norm([10, 0, 0]) == 0.0 and abs(cd.entropy_norm([5, 5, 5]) - 1.0) < 1e-12


def mor_model(tmp, tok, steps=60):
    """A tiny causal token-choice MoR model on rendered Carry-5 conversations."""
    bpe.write_gguf(str(tmp / "v.gguf"), tok)
    (tmp / "c.txt").write_text(conversations(context_items()) * 2, encoding="utf-8")
    spec = {"name": "mor_tiny", "arch": {"preset": "llama-tiny", "rope_heads": "all",
                                         "custom": {"d": 32, "heads": 2, "layers": 4, "d_ff": 64,
                                                    "recursion": {"prelude": 1, "block": 2, "repeats": 3, "coda": 1,
                                                                  "schedule": "mor", "inject": "none"}}},
            "data": {"vocab": str(tmp / "v.gguf"), "vocab_cap": 4096, "corpus": str(tmp / "c.txt"), "T": 192,
                     "max_tokens": 0},
            "train": {"steps": steps, "lr": 0.003, "batch": 2, "accum": 1, "eval_every": 30, "eval_windows": 4,
                      "checkpoint_every": 30, "gradmap_every": 0, "seed": 1, "optimizer": "adamw"},
            "export": {"formats": ["safetensors", "gguf"]}, "out_dir": str(tmp / "pt")}
    p = tmp / "mor_tiny.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    r = subprocess.run([BIN, "run", str(p)], capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return p, spec


def subset(tmp, every=9):
    items = context_items()[::every]
    p = tmp / "items.jsonl"
    p.write_text("".join(json.dumps(it) + "\n" for it in items), encoding="utf-8")
    return p, items


@needs_bin
def test_tool_end_to_end_on_a_tiny_mor_model(tmp_path, tok):
    spec, _ = mor_model(tmp_path, tok)
    items_p, items = subset(tmp_path)
    out = tmp_path / "depth"
    argv = TOOL + ["run", "--model", str(spec), "--mtstudio", BIN, "--out", str(out), "--set", str(items_p)]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=1800, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = json.loads((out / "depth.json").read_text())
    reg, g = d["registered"], d["modes"]["gold"]
    assert d["repeats"] == 3 and d["items"] == len(items) and set(d["modes"]) == {"gold", "twin", "noctx"}
    assert g["items"] + d["over_ctx"]["gold"] == len(items) and g["items"] >= 0.8 * len(items)
    assert 1 <= g["answer"]["mean"] <= 3 and 1 <= g["other_assistant"]["mean"] <= 3
    assert reg["delta"] == g["delta"] and reg["slope_answer_on_distance"] is not None
    assert 0 <= reg["entropy_norm"] <= 1 and abs(sum(reg["frac_all_positions"]) - 1) < 1e-9
    assert sum(g["answer"]["hist"]) == g["answer"]["n"] > 0
    # The registered positions' depths are mtstudio score's own trace at those positions.
    t = ce.Tok(str(tmp_path / "v.gguf"))
    recs = {x["id"]: x for x in ce.read_jsonl(out / "items.jsonl")}
    it = next(i for i in items if not recs[i["id"]]["modes"]["gold"].get("over_ctx"))
    x = cd.render(t, it, "gold")
    q = tmp_path / "q.jsonl"
    q.write_text(json.dumps({"prompt_ids": x["prompt_ids"], "continuation_ids": x["continuation_ids"]}) + "\n")
    subprocess.run([BIN, "score", str(spec), "--in", str(q), "--out", str(tmp_path / "a.jsonl")], check=True,
                   capture_output=True)
    trace = ce.read_jsonl(tmp_path / "a.jsonl")[0]["depth"]
    m = recs[it["id"]]["modes"]["gold"]
    assert m["answer"] == [trace[i] for i in x["answer_pos"]]
    assert m["other_sum"] == sum(trace[i] for i in x["other_pos"]) and m["hist"] == cd.hist_of(trace, 3)
    # Recomputed from the item records, Delta and the slope are the summary's.
    rows = [v for v in recs.values() if not v["modes"]["gold"].get("over_ctx")]
    a = [z for v in rows for z in v["modes"]["gold"]["answer"]]
    o = sum(v["modes"]["gold"]["other_sum"] for v in rows) / sum(v["modes"]["gold"]["other_n"] for v in rows)
    assert abs(reg["delta"] - (sum(a) / len(a) - o)) < 1e-12
    b, _ = cd.ols([v["distance"] for v in rows], [sum(v["modes"]["gold"]["answer"]) / len(v["modes"]["gold"]["answer"])
                                                  for v in rows])
    assert abs(reg["slope_answer_on_distance"] - b) < 1e-12
    print(f"tiny MoR model, {g['items']} items (gold history): answer depth {g['answer']['mean']:.3f} "
          f"(n {g['answer']['n']}), other assistant {g['other_assistant']['mean']:.3f} (n {g['other_assistant']['n']}), "
          f"Delta {reg['delta']:+.4f}, slope {reg['slope_answer_on_distance']:+.4f} per turn, entropy "
          f"{reg['entropy_norm']:.3f}, fractions {[round(f, 3) for f in reg['frac_all_positions']]}; twin Delta "
          f"{d['modes']['twin']['delta']:+.4f}, question-alone Delta {d['modes']['noctx']['delta']:+.4f}; "
          f"smallest routing margin {d['depth_margin_min']:.3e}")
    # Rerun: the chunks are reused and the result is the same.
    r = subprocess.run(argv, capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert r.returncode == 0 and "depth-scored" not in r.stdout
    assert json.loads((out / "depth.json").read_text())["registered"] == reg
    # A model without schedule mor is refused.
    plain = json.loads(spec.read_text())
    plain["arch"]["custom"].pop("recursion")
    (tmp_path / "plain.json").write_text(json.dumps(plain))
    r = subprocess.run(TOOL + ["run", "--model", str(tmp_path / "plain.json"), "--mtstudio", BIN, "--out",
                               str(tmp_path / "d2"), "--set", str(items_p)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "not mor" in r.stderr


FAKE_CARRY5 = ("import json, os, sys\n"
               "arg = lambda k: sys.argv[sys.argv.index(k) + 1]\n"
               "acc = {'gold': 0.6, 'self': 0.5, 'twin': 0.7, 'noctx': 0.25}\n"
               "modes = [k for k in acc if k != 'self' or arg('--gen') != 'none']\n"
               "fc = {k: {m: {'all': {'mean': acc[m], 'n': 4}} for m in acc} for k in ('acc', 'acc_norm')}\n"
               "os.makedirs(os.path.join(arg('--out'), 'm'), exist_ok=True)\n"
               "with open(os.path.join(arg('--out'), 'm', 'items.jsonl'), 'w') as f:\n"
               "    for i in range(20):\n"
               "        f.write(json.dumps({'family': 'recall', 'fc': {m: {'acc': float(i < round(20 * acc[m])),"
               " 'acc_norm': 0.0} for m in modes}, 'gen': {}}) + chr(10))\n"
               "json.dump({'meta': {'gen': arg('--gen')}, 'models': {'m': {'fc': fc}}},"
               " open(os.path.join(arg('--out'), 'summary.json'), 'w'))\n")


@needs_bin
def test_post_runs_the_depth_step_for_a_mor_spec(tmp_path, tok):
    spec, pt = mor_model(tmp_path, tok, steps=40)
    items_p, items = subset(tmp_path, every=12)
    shape = {k: v for k, v in pt.items() if k != "out_dir"}
    (tmp_path / "shape.json").write_text(json.dumps(shape))
    ex = [{"prompt": f"<|user|>{t['user']}<|assistant|>", "completion": t["gold"]} for it in items for t in it["turns"]]
    (tmp_path / "sft.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ex), encoding="utf-8")
    sft = {"data": {"vocab": pt["data"]["vocab"], "vocab_cap": 4096, "T": 192, "sft_examples": str(tmp_path / "sft.jsonl")},
           "train": {"objective": "sft", "steps": 6, "lr": 0.001, "batch": 4, "eval_every": 3, "eval_batch": 4,
                     "checkpoint_every": 3, "gradmap_every": 0, "seed": 1, "optimizer": "adamw"},
           "export": {"formats": ["safetensors", "gguf"]}}
    (tmp_path / "sft.json").write_text(json.dumps(sft))
    fake = tmp_path / "carry5_fake.py"
    fake.write_text(FAKE_CARRY5)
    env = {**os.environ, "JOBS_MTSTUDIO": BIN, "TINY_CHAT_CARRY5": str(fake)}
    post = tmp_path / "post"
    argv = RUN + ["post", "--out", str(post), "--spec", str(tmp_path / "shape.json"), "--pt-weights",
                  str(tmp_path / "pt" / "mor_tiny.safetensors"), "--pt-dir", str(tmp_path / "pt"), "--sft-template",
                  str(tmp_path / "sft.json"), "--seed", "1", "--cell", "MoR3", "--share", "35", "--carry5-items",
                  str(items_p)]
    r = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=ROOT, timeout=1800)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    s = json.loads((post / "summary.json").read_text())
    assert abs(s["primary"] - (0.5 - 0.7)) < 1e-9
    pm = s["pt_mor"]  # m_s from the pretraining events: one record per step, depth fractions over 1..3
    ev = [json.loads(x) for x in (tmp_path / "pt" / "events.jsonl").read_text().splitlines() if x.strip()]
    steps = [e for e in ev if e.get("event") == "step"]
    assert pm["steps_logged"] == len(steps) == 40 and 1 <= pm["m"] <= 3
    assert abs(pm["m"] - sum(e["mor_depth"] for e in steps) / len(steps)) < 1e-4  # mtstudio's own mean depth
    assert abs(pm["active"][0] - 1) < 1e-4
    md = s["mor_depth"]
    assert "error" not in md and md["model"] == "tc_sft_MoR3_r1" and md["families"] == list(cd.CONTEXT_FAMILIES)
    assert md == json.loads((post / "mor_depth" / "depth.json").read_text())
    assert (post / "mor_depth" / "items.jsonl").exists() and md["items"] == len(items)
    print(f"post: m_s {pm['m']:.4f} over {pm['steps_logged']} steps; SFT model Delta {md['registered']['delta']:+.4f}, "
          f"slope {md['registered']['slope_answer_on_distance']:+.4f}, entropy {md['registered']['entropy_norm']:.3f}")
    # Done: a rerun is a no-op. A summary whose depth step failed is redone (the depth step only).
    r = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0 and "present" in r.stdout
    s["mor_depth"] = {"error": "carry5_depth failed (rc 1)"}
    (post / "summary.json").write_text(json.dumps(s))
    r = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=ROOT, timeout=1800)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads((post / "summary.json").read_text())["mor_depth"]["registered"] == md["registered"]
    # A failing depth tool: the summary (with Y) is written, the error recorded, and the job exits 1.
    env_bad = {**env, "TINY_CHAT_MOR_DEPTH": str(tmp_path / "missing_tool.py")}
    post2 = tmp_path / "post2"
    r = subprocess.run([*argv[:argv.index("--out") + 1], str(post2), *argv[argv.index("--out") + 2:]],
                       capture_output=True, text=True, env=env_bad, cwd=ROOT, timeout=1800)
    s2 = json.loads((post2 / "summary.json").read_text())
    assert r.returncode == 1 and "error" in s2["mor_depth"] and s2["primary"] is not None
