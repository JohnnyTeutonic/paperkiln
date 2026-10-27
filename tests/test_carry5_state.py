"""tools/carry5_eval.py on the S cells' template (experiments/tiny_chat_long_2026-10/PREREGISTRATION_DRAFT.md
section 11 item 5, the main session's ruling of 27 Oct 2026): carry5_eval inserts the state block after every
<|endoftext|> that closes an exchange only for a model whose GGUF declares state tokens
(paperkiln.state_tokens.k, the family llama_state), as carry12_eval's prompt_ids does.

- Every other model's prompts are byte-identical to carry5_eval's output before the change: the whole Carry-5
  set's log-prob requests (gold, twin, noctx, self and the parity-gate rows, from fixed stand-in replies that
  cover every stop kind) hash to the digests the pre-change code produced (paperkiln 437db1f2), on E1's
  pick_e1_L16 GGUF (the 16k Llama 3-scheme vocabulary; skipped when the file is not on this machine) and on a
  toy byte-level BPE vocabulary built here.
- On a toy state-token model (k 4, turn_state, trained by mtstudio; POSIX and MTSTUDIO_BIN), carry5_eval's
  pre-tokenised gold request scores what `mtstudio score` scores on the same prompt as text (which inserts the
  blocks itself), and its prompt ids hold the state ids.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import bpe  # noqa: E402
import carry5_eval as ce  # noqa: E402

BIN = os.environ.get("MTSTUDIO_BIN")
PICK = os.environ.get("CARRY5_STATE_TEST_PICK", "C:/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf")
if not os.path.exists(PICK) and os.path.exists("/mnt/c/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf"):
    PICK = "/mnt/c/ml_artifacts/tiny_chat/e1/pick_e1_L16/model.gguf"
# sha256 of the request list (json.dumps of [key, prompt_ids, continuation_ids] rows) that carry5_eval at
# paperkiln 437db1f2, before the state-block change, writes for the whole Carry-5 set with fake_gens' replies.
DIGEST_PICK_E1_L16 = "cef4837a6b1186497eb09e565a520990421b53edf2220e45ebddc5322e1b789a"
DIGEST_TOY = "dc96ef6d8e1fe33c5685e8016ff6b72a3e0e6e294e0db4af3373aa37b7c60157"


def items():
    return ce.read_jsonl(ce.SET)


def toy_vocab(path):
    """A byte-level BPE vocabulary trained on Carry-5's own text (deterministic)."""
    text = " ".join(t["user"] + " " + t["gold"] for it in items()[::20] for t in it["turns"])
    bpe.write_gguf(str(path), bpe.train(text, 400))


def fake_gens(its, tok):
    """Fixed stand-in generations: five self replies per item with token ids, cycling the stop kinds the
    servers return (eos, length with stop_id -1, turn with a role-marker id, a dropped history), so the
    self prompts and every branch of the gate rows are exercised."""
    out = {}
    for n, it in enumerate(its):
        recs = []
        for k, t in enumerate(it["turns"]):
            reply = f"okay {t['gold'] if (n + k) % 3 else 'sure thing'}"
            kind = (n + k) % 4
            r = {"reply": reply, "ids": tok.encode(reply), "history_used": k, "dropped": kind == 3 and k > 0}
            r.update({0: {"stop": "eos", "stop_id": tok.eot}, 1: {"stop": "length", "stop_id": -1},
                      2: {"stop": "turn", "stop_id": tok.user}, 3: {"stop": "eos", "stop_id": tok.eot}}[kind])
            recs.append(r)
        out[it["id"]] = {"id": it["id"], "self": recs}
    return out


def request_digest(tok, its=None):
    its = its if its is not None else items()
    reqs = ce.requests(its, tok, fake_gens(its, tok))
    return hashlib.sha256(json.dumps([list(r) for r in reqs]).encode()).hexdigest(), len(reqs)


@pytest.mark.skipif(not os.path.exists(PICK), reason=f"{PICK} not on this machine")
def test_non_state_pick_prompts_byte_identical():
    """E1's pick_e1_L16 (no state tokens): carry5_eval leaves its prompts as they were."""
    assert ce.state_template(PICK) is None
    tok = ce.Tok(PICK)
    tok.state = ce.state_template(PICK)
    digest, n = request_digest(tok)
    assert n == 1032 * 3 * 4 + 1032 * 4 + sum(1 for r in fake_gens(items(), tok).values()
                                               for x in r["self"] if not x["dropped"])
    assert digest == DIGEST_PICK_E1_L16


def test_non_state_toy_prompts_byte_identical(tmp_path):
    toy_vocab(tmp_path / "v.gguf")
    assert ce.state_template(str(tmp_path / "v.gguf")) is None
    tok = ce.Tok(str(tmp_path / "v.gguf"))
    tok.state = None
    assert request_digest(tok)[0] == DIGEST_TOY
    # prompt_ids is tok.encode for a model without state tokens, on every rendering
    it = items()[0]
    text = ce.template([(t["user"], t["gold"]) for t in it["turns"][:3]], it["turns"][3]["user"])
    assert ce.prompt_ids(tok, text) == tok.encode(text)


def test_prompt_ids_state_layout():
    class T:
        eot, user, assistant = 1, 2, 3

        @staticmethod
        def encode(text):
            m = {"<|endoftext|>": 1, "<|user|>": 2, "<|assistant|>": 3}
            return [m.get(w, 10 + len(w)) for w in text.split()]
    tok = T()
    text = "<|user|> hi <|assistant|> hello <|endoftext|> <|user|> again <|assistant|>"
    assert ce.prompt_ids(tok, text) == T.encode(text)
    tok.state = (100, 4)
    assert ce.prompt_ids(tok, text) == [2, 12, 3, 15, 1, 100, 101, 102, 103, 2, 15, 3]


# ---------------------------------------------------------------- a toy state-token model, end to end
e2e = pytest.mark.skipif(os.name == "nt" or not BIN, reason="needs POSIX and MTSTUDIO_BIN")


@pytest.fixture(scope="module")
def toy_state(tmp_path_factory):
    """A toy llama with per-turn state tokens (cell S-gist's template: k 4, turn_state), SFT from scratch on
    Carry-5 target turns; mtstudio inserts the state blocks itself in training, `score` and `chat`."""
    t = tmp_path_factory.mktemp("c5state")
    toy_vocab(t / "v.gguf")
    its = items()[::40]
    ex = [{"prompt": ce.template([(x["user"], x["gold"]) for x in it["turns"][:it["target_turn"] - 1]],
                                 it["turns"][it["target_turn"] - 1]["user"]),
           "completion": it["turns"][it["target_turn"] - 1]["gold"]} for it in its]
    (t / "ex.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ex), encoding="utf-8")
    spec = {"name": "c5state", "arch": {"preset": "llama-tiny", "rope_heads": "all",
                                        "custom": {"d": 32, "layers": 2, "heads": 2, "d_ff": 64},
                                        "state_tokens": {"k": 4, "mask": "turn_state"}},
            "data": {"vocab": str(t / "v.gguf"), "vocab_cap": 4096, "T": 256, "sft_examples": str(t / "ex.jsonl")},
            "train": {"steps": 60, "lr": 0.003, "batch": 2, "eval_every": 60, "checkpoint_every": 1000,
                      "gradmap_every": 0, "seed": 1, "objective": "sft"},
            "export": {"formats": ["safetensors", "gguf"]}, "out_dir": str(t / "run")}
    sp = t / "spec.json"
    sp.write_text(json.dumps(spec), encoding="utf-8")
    p = subprocess.run([BIN, "run", str(sp)], capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return t, sp


@e2e
def test_state_model_prompts_match_mtstudio_score(toy_state, tmp_path):
    t, sp = toy_state
    gguf = str(t / "run" / "c5state.gguf")
    V, k = ce.state_template(gguf)
    assert k == 4
    out = tmp_path / "e"
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "carry5_eval.py"), "run", "--model", f"m={sp}",
                        "--out", str(out), "--mtstudio", BIN, "--gen", "mtstudio", "--every", "52", "--max-new", "6",
                        "--port", "8231"], capture_output=True, text=True, cwd=ROOT, timeout=3600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())["models"]["m"]
    assert s["items"] == 20 and all(s["fc"]["acc"][m]["all"]["n"] == 20 for m in ce.MODES)
    got = {}
    for f in sorted((out / "m" / "score").glob("chunk*.json")):
        got.update(json.loads(f.read_text())["results"])
    tok = ce.Tok(gguf)
    tok.state = (V, k)
    its = items()[::52]
    rows = []
    for it in its[:4]:
        tt = it["target_turn"]
        for mode, text in (("gold", ce.template([(x["user"], x["gold"]) for x in it["turns"][:tt - 1]],
                                                it["turns"][tt - 1]["user"])),
                           ("twin", ce.template([], it["twin"]))):
            pids = ce.prompt_ids(tok, text)
            if mode == "gold":
                assert max(pids) >= V and pids.count(V) == tt - 1  # one block per closed exchange
            else:
                assert max(pids) < V  # nothing closed before a one-turn prompt
            c = tok.encode(it["frame"].format(it["candidates"][0])) + [tok.eot]
            rows.append((f"{it['id']}|{mode}|0", text, c))
    inp, outp = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    inp.write_text("".join(json.dumps({"prompt": text, "continuation_ids": c}) + "\n" for _, text, c in rows),
                   encoding="utf-8")
    p = subprocess.run([BIN, "score", str(sp), "--in", str(inp), "--out", str(outp)], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    want = [json.loads(x)["sum"] for x in outp.read_text().splitlines()]
    for (key, _, _), w in zip(rows, want):
        assert abs(got[key]["sum"] - w) < 1e-4, (key, got[key]["sum"], w)
    # the self history went through `mtstudio chat`, which inserts the blocks itself: its prompt ids (when
    # returned) matched carry5_eval's layout, or carry5_eval would have exited
    assert (out / "m" / "gen.jsonl").exists()
