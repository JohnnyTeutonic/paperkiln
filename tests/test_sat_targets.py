"""train.symbol_targets (tools/mtstudio.cpp), end to end on CPU with the real mtstudio: SAT's trainer change
(AI_ML/chat_screens_claude/base2/token_ladder/symbols_as_targets/REGISTRATION.md in the author's research tree).
Claude Opus 5.5, 26 Nov 2026.

The claims: labels never enter the forward (the registered implementation check, section 7); w0 = 0 is the plain
spec bit for bit; from cut_step a SAT run continues as the plain spec does; a resume inside the scaffold restores the
heads and their optimiser; the logged w is the registered schedule; twin rows are exactly g % every == 0 before
cut_step and every other window is the plain run's; labels land on the tokens they were made from through the
state-token insertion, and a span whose hash differs refuses the run; the exported tensors are the plain run's.
Small synthetic .ids corpora with turn markers, so state ids are inserted. Needs MTSTUDIO_BIN; skipped otherwise.
MTSTUDIO_BASE_BIN (the trainer built from the ladder's launch commit 672501bb) adds a plain-spec comparison against it.
"""
import json
import os
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
sys.path.insert(0, str(ROOT / "tools"))
from test_mtstudio_chat import BIN, write_vocab_gguf  # noqa: E402
from test_checkpoint_txn import WORDS  # noqa: E402
import sat_labels as SL  # noqa: E402
import state_tokens  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
BASE_BIN = os.environ.get("MTSTUDIO_BASE_BIN")

VOCAB = ["<unk>", "<|endoftext|>", "<|user|>", "<|assistant|>"] + WORDS
NV = len(VOCAB)
EOT, USER, ASST = 1, 2, 3
K = 2
T, BATCH, ACCUM = 32, 2, 2
STEPS, CUT, EVERY, HALF = 20, 12, 3, 5.0
TEXT = list(range(4, NV))


# ---------------------------------------------------------------- data

def main_raw(n_docs, seed):
    """Plain sentences and dialogue turns, some replies cut short (a <|user|> while a reply is open)."""
    r = random.Random(seed)
    ids = []
    for _ in range(n_docs):
        kind = r.random()
        words = [r.choice(TEXT) for _ in range(r.randint(4, 12))]
        if kind < 0.4:
            ids += words + [EOT]
        elif kind < 0.85:
            ids += [USER] + words[: len(words) // 2] + [ASST] + words[len(words) // 2:] + [EOT]
        else:
            ids += [USER] + words[:3] + [ASST] + words[3:]          # no <|endoftext|>: closed early by the next turn
    return ids


def twin_raw(n, seed):
    r = random.Random(seed)
    return [r.choice(TEXT) for _ in range(n)]


def write_ids(path, ids):
    with open(path, "wb") as f:
        f.write(b"PKIDS001" + struct.pack("<IIQ", 2, NV, len(ids)))
        f.write(struct.pack(f"<{len(ids)}H", *ids))


def make_spans(raw, seed, length=20, stride=37, start=3, end=None):
    """Dense random labels: roles, mentions and pointers to earlier mentions in the span."""
    r = random.Random(seed)
    end = len(raw) if end is None else end
    spans = []
    o = start
    while o + length <= end:
        codes = [SL.code(r.randrange(8), labelled=r.random() < 0.5, mention=r.random() < 0.35) for _ in range(length)]
        ptrs = []
        for pos in range(1, length):
            cands = [j for j in range(pos) if codes[j] & SL.MENTION]
            if cands and r.random() < 0.3:
                ptrs.append((pos, pos - r.choice(cands)))
        spans.append({"raw_offset": o, "codes": bytes(codes), "pointers": ptrs})
        o += stride
    return spans


def scramble(spans):
    """Labels B: every role moved to the next, every pointer moved to another earlier mention (dropped if none)."""
    out = []
    for s in spans:
        codes = bytearray(s["codes"])
        for i, c in enumerate(codes):
            if c & SL.LABELLED:
                codes[i] = (c & ~7) | (((c & 7) + 1) % 8)
        ptrs = []
        for pos, back in s["pointers"]:
            others = [j for j in range(pos - 1, -1, -1) if codes[j] & SL.MENTION and j != pos - back]
            if others:
                ptrs.append((pos, pos - others[0]))
        out.append({"raw_offset": s["raw_offset"], "codes": bytes(codes), "pointers": ptrs})
    return out


def post_positions(raw):
    """Each raw id's post-insertion position, by the greedy match over tools/state_tokens.py's output."""
    post, _, _ = state_tokens.insert(raw, NV, K, EOT, USER, ASST)
    pos, j = [], 0
    for i, t in enumerate(post):
        if j < len(raw) and t == raw[j]:
            pos.append(i)
            j += 1
    assert j == len(raw)
    return post, pos


def setup(tmp):
    """The corpora, the vocabulary and labels A (main), B (scrambled), none (no spans), eval and twin labels."""
    if (tmp / "ready").exists():
        return json.loads((tmp / "ready").read_text())
    write_vocab_gguf(tmp / "v.gguf", VOCAB)
    raw = main_raw(900, 0)
    write_ids(tmp / "main.ids", raw)
    write_ids(tmp / "tail.ids", main_raw(200, 1))
    tw = twin_raw(1500, 2)
    write_ids(tmp / "twin.ids", tw)
    post, pos = post_positions(raw)
    n = len(post)
    val_start = n - n // 20
    # Main labels over the training part, eval labels: three held-out windows of T + 1 raw tokens.
    first_val_raw = next(r for r, p in enumerate(pos) if p >= val_start)
    spans_a = make_spans(raw, 10, end=first_val_raw)
    SL.write(tmp / "A.satl", spans_a, raw, "synthetic-main")
    SL.write(tmp / "B.satl", scramble(spans_a), raw, "synthetic-main")
    SL.write(tmp / "none.satl", [], raw, "synthetic-main")
    SL.write(tmp / "nopointers.satl", [{**s, "pointers": []} for s in spans_a], raw, "synthetic-main")
    ev_spans = []
    for o in (first_val_raw + 2, first_val_raw + 40, first_val_raw + 80):
        if pos[o] + T + 1 <= n and o + T + 1 <= len(raw):
            ev_spans += make_spans(raw, 20 + o, length=T + 1, stride=10 ** 9, start=o, end=o + T + 1)
    assert len(ev_spans) >= 2
    SL.write(tmp / "eval.satl", ev_spans, raw, "synthetic-main")
    SL.write(tmp / "twin.satl", make_spans(tw, 30, length=16, stride=23), tw, "synthetic-twin")
    bad = [dict(s) for s in spans_a]
    bad[5] = {**bad[5], "sha256": b"\0" * 32, "force_sha256": True}
    SL.write(tmp / "bad.satl", bad, raw, "synthetic-main")
    info = {"n_raw": len(raw), "n_post": n, "val_start": val_start}
    (tmp / "ready").write_text(json.dumps(info))
    return info


def sat_cfg(tmp, labels="A", w0=1.0, twins=True, cut=CUT, half=HALF, eval_labels=True):
    d = {"labels": str(tmp / f"{labels}.satl"), "w0": w0, "half_life": half, "cut_step": cut, "role_classes": 8,
         "pointer_dim": 8}
    if eval_labels:
        d["eval_labels"] = str(tmp / "eval.satl")
    if twins:
        d["twins"] = {"corpus": str(tmp / "twin.ids"), "labels": str(tmp / "twin.satl"), "every": EVERY}
    return d


def spec(tmp, out, sat=None, steps=STEPS, ckpt_every=4, fused=False, anneal=0.75):
    s = {"name": "m",
         "arch": {"preset": "llama-tiny", "custom": {"d": 32, "layers": 2, "heads": 2},
                  "state_tokens": {"k": K, "mask": "causal"}},
         "data": {"corpus": str(tmp / "main.ids"), "vocab": str(tmp / "v.gguf"), "T": T, "max_tokens": 0,
                  "anneal": {"corpus": str(tmp / "tail.ids"), "from_fraction": anneal}},
         "train": {"steps": steps, "lr": 0.003, "batch": BATCH, "accum": ACCUM, "clip": 1.0, "eval_every": 4,
                   "eval_windows": 2, "checkpoint_every": ckpt_every, "seed": 3, "optimizer": "adamw",
                   "gradmap_every": 0, "lr_schedule": {"warmup_steps": 4, "decay": "cosine", "min_lr_frac": 0.1}},
         "export": {"formats": ["safetensors"]}, "out_dir": str(out)}
    if sat is not None:
        s["train"]["symbol_targets"] = sat
    if fused:
        s["train"]["fused_head"] = True
    p = Path(str(out) + ".spec.json")
    p.write_text(json.dumps(s), encoding="utf-8")
    return p


def run(p, binary=None, **env):
    e = {**os.environ, "OMP_NUM_THREADS": "1", "MICROTORCH_LOG_WINDOWS": "1", "CUDA_VISIBLE_DEVICES": "",
         "MICROTORCH_CHECKPOINT_TXN": "1", **env}
    return subprocess.run([binary or BIN, "run", str(p)], env=e, capture_output=True, text=True, timeout=600)


def evs(out, kind):
    return [e for e in map(json.loads, (out / "events.jsonl").read_text().splitlines()) if e.get("event") == kind]


def last_by_step(out, kind):
    d = {}
    for e in evs(out, kind):
        d[(e["step"], e.get("micro"))] = e        # a restarted run's events come last and win
    return d


def stitched_events(before_text, after_text):
    """The event stream a reader collects across a restart: everything the ended run logged, then what the restarted
    run logged past the lines it kept (the trainer drops lines past the checkpoint on resume, truncate_events_after)."""
    a, b = before_text.splitlines(), after_text.splitlines()
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return [json.loads(x) for x in a + b[n:]]


def st_keys(path):
    b = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", b[:8])
    return sorted(k for k in json.loads(b[8:8 + n]) if k != "__metadata__")


def full(tmp, name, sat, **kw):
    out = tmp / name
    if not (out / "result.json").exists():
        r = run(spec(tmp, out, sat, **kw))
        assert r.returncode == 0, r.stderr[-1500:]
    return out


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sat")
    info = setup(tmp)
    return tmp, info


# ---------------------------------------------------------------- SATL (no binary needed)

def test_satl_round_trip(tmp_path):
    raw = main_raw(200, 5)
    spans = make_spans(raw, 1)
    h = SL.write(tmp_path / "x.satl", spans, raw, "abc")
    h2, back = SL.read(tmp_path / "x.satl")
    assert h == h2 and h["spans"] == len(spans) and h["pointer_targets"] == sum(len(s["pointers"]) for s in spans)
    for a, b in zip(spans, back):
        assert (a["raw_offset"], a["codes"], a["pointers"]) == (b["raw_offset"], b["codes"], b["pointers"])
    assert SL.verify(back, raw) == []
    changed = list(raw)
    changed[3] = 4 if raw[3] != 4 else 5
    assert SL.verify(back, changed) == [3]
    # The layout, byte for byte, of a one-span file.
    SL.write(tmp_path / "one.satl", [{"raw_offset": 7, "codes": bytes([SL.code("SUBJ", True, True), 0, 0x0E]),
                                      "pointers": [(2, 2)]}], raw, "s")
    b = (tmp_path / "one.satl").read_bytes()
    (hl,) = struct.unpack_from("<I", b, 8)
    body = b[12 + hl:]
    assert body[:12] == struct.pack("<QI", 7, 3) and body[12:44] == SL.ids_sha256(raw[7:10])
    assert body[44:47] == bytes([1 | 8 | 16, 0, 6 | 8]) and body[47:] == struct.pack("<III", 1, 2, 2)


@pytest.mark.parametrize("bad,why", [
    ([{"raw_offset": 5, "codes": b"\x20", "pointers": []}], "bits 5-7"),
    ([{"raw_offset": 5, "codes": b"\x00\x00", "pointers": []}, {"raw_offset": 6, "codes": b"\x00", "pointers": []}],
     "overlapping"),
    ([{"raw_offset": 5, "codes": b"\x00\x00", "pointers": [(1, 1)]}], "not a mention"),
    ([{"raw_offset": 5, "codes": b"\x10\x00", "pointers": [(1, 2)]}], "outside its span"),
])
def test_satl_writer_refuses(tmp_path, bad, why):
    with pytest.raises(ValueError, match=why):
        SL.write(tmp_path / "x.satl", bad, list(range(4, 14)) * 3, "s")


# ---------------------------------------------------------------- the trainer

def check_result_can_fail(res):
    """The conditions that make the implementation check one that could fail (Explore 10, GPT 4)."""
    assert res["a_role_targets"] > 0 and res["a_pointer_targets"] > 0
    assert sum(res["b_roles_moved"].values()) > 0 and sum(res["b_pointers_moved"].values()) > 0
    assert res["aux_finite"] is True and res["aux_differ"] == {"role": True, "pointer": True}
    assert res["after_aux_identical"] is True
    sums = res["aux_loss_sums"]
    assert sums["A"]["role"] != sums["B"]["role"] and sums["A"]["pointer"] != sums["B"]["pointer"]


@needs_bin
@pytest.mark.parametrize("labels,why", [("nopointers", "a_pointer_targets"), ("none", "a_role_targets")])
def test_implementation_check_refuses_when_it_cannot_fail(data, labels, why):
    """Labels without pointers, or none at all (twins off): the logits agree, but the check could not have failed on
    the pointer (or any) term, so it does not pass and the trainer exits non-zero."""
    tmp, _ = data
    out = tmp / f"check_cannot_{labels}"
    shutil.rmtree(out, ignore_errors=True)
    r = run(spec(tmp, out, sat_cfg(tmp, labels, twins=False)), MICROTORCH_SAT_CHECK="8")
    assert r.returncode != 0 and "could not fail" in r.stderr, r.stderr[-800:]
    res = evs(out, "sat_check_result")[-1]
    assert res["equal"] is True and res["pass"] is False and res[why] == 0


@needs_bin
def test_implementation_check_at_init_and_after_five_steps(data):
    """REGISTRATION section 7. Fixed weights, labels A, B (roles moved, pointers moved) and none: every LM logit of
    the scaffold step's forward agrees bit for bit, with each other and with the plain step's forward (forward(x) ==
    head(forward_hidden(x))), at initialisation and at the weights of a checkpoint after 5 training steps."""
    tmp, _ = data
    hashes = {}
    for lab in ("A", "B", "none"):
        out = tmp / f"check0_{lab}"
        shutil.rmtree(out, ignore_errors=True)
        r = run(spec(tmp, out, sat_cfg(tmp, lab)), MICROTORCH_SAT_CHECK="8")
        assert r.returncode == 0, r.stderr[-1500:]
        res = evs(out, "sat_check_result")
        assert len(res) == 1 and res[0]["equal"] is True and res[0]["pass"] is True and res[0]["from_step"] == 1
        check_result_can_fail(res[0])
        ch = evs(out, "sat_check")
        assert all(c["after_aux_identical"] is True for c in ch)
        assert {c["variant"] for c in ch} == {"A", "B", "none", "plain"} and len(ch) == 4 * 4
        hashes[lab] = [(c["variant"], c["step"], c["micro"], c["logits_sha256"]) for c in ch]
        if lab == "A":
            assert sum(c.get("role_n", 0) for c in ch if c["variant"] == "A") > 0
            assert sum(c.get("ptr_n", 0) for c in ch if c["variant"] == "A") > 0
            assert sum(c.get("ptr_n", 0) for c in ch if c["variant"] == "none") == 0
        assert not (out / "result.json").exists()           # the check trains nothing
    assert hashes["A"] == hashes["B"] == hashes["none"]
    assert len({h[3] for h in hashes["A"]}) == 4                   # four different micro-batches, one hash each

    # Five training steps under A (published at step 5), then the check at those weights with A, B and none.
    base = tmp / "five_A"
    p = spec(tmp, base, sat_cfg(tmp, "A"), ckpt_every=5)
    r = run(p, MICROTORCH_CKPT_CRASH="prune:5")
    assert r.returncode != 0 and "crash test" in r.stderr
    after = {}
    for lab in ("A", "B", "none"):
        out = tmp / f"check5_{lab}"
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(base, out, symlinks=True)
        r = run(spec(tmp, out, sat_cfg(tmp, lab), ckpt_every=5), MICROTORCH_SAT_CHECK="8")
        assert r.returncode == 0, r.stderr[-1500:]
        res = evs(out, "sat_check_result")[-1]
        assert (res["pass"], res["equal"], res["windows"], res["from_step"]) == (True, True, 8, 6)
        check_result_can_fail(res)
        after[lab] = [(c["variant"], c["step"], c["logits_sha256"]) for c in evs(out, "sat_check") if c["step"] >= 6]
    assert after["A"] == after["B"] == after["none"] and after["A"]
    assert {h[2] for h in after["A"]}.isdisjoint({h[3] for h in hashes["A"]})   # other weights: other logits
    # The intervention is real: five steps under B leave other trunk weights than under A.
    other = tmp / "five_B"
    r = run(spec(tmp, other, sat_cfg(tmp, "B"), ckpt_every=5), MICROTORCH_CKPT_CRASH="prune:5")
    assert r.returncode != 0
    assert (base / "model.safetensors").read_bytes() != (other / "model.safetensors").read_bytes()
    # ...and the first step's LM loss is the same under A, B and none (labels never reach the forward).
    l1 = []
    for lab in ("A", "B", "none"):
        o = tmp / f"first_{lab}"
        shutil.rmtree(o, ignore_errors=True)
        run(spec(tmp, o, sat_cfg(tmp, lab), ckpt_every=1), MICROTORCH_CKPT_CRASH="prune:1")
        l1.append(evs(o, "step")[0]["loss"])
    assert l1[0] == l1[1] == l1[2]


@needs_bin
def test_w0_zero_is_the_plain_spec(data):
    """w0 = 0 (twins off): the plain spec's loss trace, evaluations and weights, bit for bit."""
    tmp, _ = data
    plain = full(tmp, "plain", None)
    zero = full(tmp, "zero", sat_cfg(tmp, w0=0.0, twins=False))
    for kind in ("step", "eval"):
        a = [{k: v for k, v in e.items() if k != "symbol_targets"} for e in evs(zero, kind)]
        b = evs(plain, kind)
        assert [x.get("loss", x.get("val_loss")) for x in a] == [x.get("loss", x.get("val_loss")) for x in b]
        assert a == b
    assert (zero / "model.safetensors").read_bytes() == (plain / "model.safetensors").read_bytes()
    assert (zero / "m.safetensors").read_bytes() == (plain / "m.safetensors").read_bytes()
    assert not evs(zero, "symbol_targets_eval")


@needs_bin
@pytest.mark.skipif(not BASE_BIN, reason="set MTSTUDIO_BASE_BIN to the trainer built from 672501bb")
def test_plain_spec_equals_the_launch_commits_trainer(data):
    tmp, _ = data
    plain = full(tmp, "plain", None)
    out = tmp / "plain_base"
    if not (out / "result.json").exists():
        e = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "MICROTORCH_CHECKPOINT_TXN": "1"}
        r = subprocess.run([BASE_BIN, "run", str(spec(tmp, out, None))], env=e, capture_output=True, text=True,
                           timeout=600)
        assert r.returncode == 0, r.stderr[-1500:]
    assert [e["loss"] for e in evs(out, "step")] == [e["loss"] for e in evs(plain, "step")]
    assert evs(out, "eval") == evs(plain, "eval")
    assert (out / "model.safetensors").read_bytes() == (plain / "model.safetensors").read_bytes()


@needs_bin
def test_from_the_cut_sat_and_plain_continue_identically(data):
    """A SAT run published at cut_step, resumed once under SAT's spec and once under the plain spec: identical steps,
    evaluations, windows and weights from cut_step + 1 to the end (the auxiliary branch is gone, twins too)."""
    tmp, _ = data
    base = tmp / "to_cut"
    r = run(spec(tmp, base, sat_cfg(tmp)), MICROTORCH_CKPT_CRASH=f"prune:{CUT}")
    assert r.returncode != 0 and "crash test" in r.stderr
    outs = {}
    for name, sat in (("sat", sat_cfg(tmp)), ("plain", None)):
        out = tmp / f"cont_{name}"
        shutil.rmtree(out, ignore_errors=True)
        shutil.copytree(base, out, symlinks=True)
        r = run(spec(tmp, out, sat))
        assert r.returncode == 0, r.stderr[-1500:]
        outs[name] = out
    a, b = outs["sat"], outs["plain"]
    sa = {k: v for k, v in last_by_step(a, "step").items() if k[0] > CUT}
    sb = {k: v for k, v in last_by_step(b, "step").items() if k[0] > CUT}
    assert sorted(sa) == sorted(sb) and len(sa) == STEPS - CUT and sa == sb
    assert {k: v for k, v in last_by_step(a, "eval").items() if k[0] > CUT} == \
           {k: v for k, v in last_by_step(b, "eval").items() if k[0] > CUT}
    # The window log names the twin corpus's limit (0 without one): every other field is compared, and no row is a twin.
    wa = {k: {f: x for f, x in v.items() if f != "twin_lim"} for k, v in last_by_step(a, "windows").items() if k[0] > CUT}
    wb = {k: {f: x for f, x in v.items() if f != "twin_lim"} for k, v in last_by_step(b, "windows").items() if k[0] > CUT}
    assert wa == wb and len(wa) == (STEPS - CUT) * ACCUM
    assert all(v["twin"] == [0] * BATCH for v in wa.values())
    assert (a / "model.safetensors").read_bytes() == (b / "model.safetensors").read_bytes()
    assert (a / "m.safetensors").read_bytes() == (b / "m.safetensors").read_bytes()


@needs_bin
@pytest.mark.parametrize("stage,at,crash_at", [("prune", 4, 4), ("prune", 8, 8), ("mkdir", 8, 4)])
def test_resume_inside_the_scaffold(data, stage, at, crash_at):
    """A run ended inside the scaffold and restarted equals the uninterrupted run: steps (w, auxiliary losses and
    counts included), held-out accuracies, windows and the final weights. Ended after the checkpoint at 4 or 8 was
    published, or while the one at 8 was being written (so it resumes from 4 and steps 5 to 8 are logged twice)."""
    tmp, _ = data
    whole = full(tmp, "sat_full", sat_cfg(tmp))
    cut = tmp / f"sat_resume_{stage}_{at}"
    shutil.rmtree(cut, ignore_errors=True)
    p = spec(tmp, cut, sat_cfg(tmp))
    r1 = run(p, MICROTORCH_CKPT_CRASH=f"{stage}:{at}")
    assert r1.returncode != 0
    before = (cut / "events.jsonl").read_text()
    r2 = run(p)
    assert r2.returncode == 0, r2.stderr[-1500:]
    stitched = [e for e in stitched_events(before, (cut / "events.jsonl").read_text()) if e.get("event") == "step"]
    logged = [e["step"] for e in stitched]
    assert (len(logged) > len(set(logged))) == (at != crash_at)       # steps logged twice when the checkpoint was lost
    dedup = {e["step"]: e for e in stitched}
    for k in ("natural_role_targets", "natural_pointer_targets", "twin_role_targets", "twin_pointer_targets", "twin_rows"):
        assert sum(v["symbol_targets"][k] for s_, v in dedup.items() if s_ < CUT) == \
               sum(e["symbol_targets"][k] for e in evs(whole, "step") if e["step"] < CUT), k
    res = evs(cut, "symbol_targets_resume")
    assert len(res) == 1 and res[0]["step"] == crash_at and res[0]["adamw_t"] == crash_at
    for kind in ("step", "eval", "windows"):
        a = {k: v for k, v in last_by_step(whole, kind).items() if k[0] > crash_at}
        b = {k: v for k, v in last_by_step(cut, kind).items() if k[0] > crash_at}
        assert a == b and a, kind
    # The held-out accuracies of steps 4 and 8 (the last before the cut), before and after the restart.
    assert last_by_step(whole, "symbol_targets_eval") == last_by_step(cut, "symbol_targets_eval")
    assert len(last_by_step(cut, "symbol_targets_eval")) == 2
    assert (whole / "model.safetensors").read_bytes() == (cut / "model.safetensors").read_bytes()
    # The observed counts were carried through the checkpoint: the end event is the uninterrupted run's, field for field,
    # and the per-step counts de-duplicated by step sum to the same totals.
    end_a, end_b = evs(whole, "symbol_targets_end"), evs(cut, "symbol_targets_end")
    assert len(end_a) == len(end_b) == 1 and end_a[0] == end_b[0]
    cnt = evs(cut, "symbol_targets_counts")
    assert len(cnt) == 1 and cnt[0]["step"] == crash_at
    by_a, by_b = last_by_step(whole, "step"), last_by_step(cut, "step")
    for k in ("natural_role_targets", "natural_pointer_targets", "twin_role_targets", "twin_pointer_targets", "twin_rows"):
        sa = sum(v["symbol_targets"][k] for (s_, _), v in by_a.items() if s_ < CUT)
        sb = sum(v["symbol_targets"][k] for (s_, _), v in by_b.items() if s_ < CUT)
        assert sa == sb == end_a[0][k], k


@needs_bin
def test_heads_restored_not_refreshed(data):
    """A checkpoint inside the scaffold without the heads (optim.safetensors stripped of them) refuses to resume."""
    tmp, _ = data
    out = tmp / "noheads"
    shutil.rmtree(out, ignore_errors=True)
    p = spec(tmp, out, sat_cfg(tmp))
    assert run(p, MICROTORCH_CKPT_CRASH="prune:4").returncode != 0
    keys = st_keys(out / "optim.safetensors")
    assert {"sat_heads.role", "sat_heads.wq", "sat_heads.wk"} <= set(keys)
    assert any(k.startswith("sat_heads_optim.") for k in keys)
    # The same checkpoint with the heads removed (the generation's manifest rewritten to match): SAT refuses.
    import hashlib
    live = out / "ckpt_gen" / os.readlink(out / "ckpt_gen" / "live")
    f = live / "optim.safetensors"
    b = f.read_bytes()
    (n,) = struct.unpack("<Q", b[:8])
    head = json.loads(b[8:8 + n])
    data0 = 8 + n
    keep, blob = {}, b""
    for k, v in sorted(head.items(), key=lambda kv: kv[1]["data_offsets"][0] if kv[0] != "__metadata__" else -1):
        if k == "__metadata__":
            keep[k] = v
            continue
        if k.startswith("sat_heads"):
            continue
        s0, s1 = v["data_offsets"]
        keep[k] = {**v, "data_offsets": [len(blob), len(blob) + s1 - s0]}
        blob += b[data0 + s0:data0 + s1]
    h = json.dumps(keep).encode()
    h += b" " * (-len(h) % 8)
    f.unlink()
    f.write_bytes(struct.pack("<Q", len(h)) + h + blob)
    nb = f.read_bytes()
    manifest = json.loads((live / "MANIFEST.json").read_text())
    manifest["files"]["optim.safetensors"] = {"bytes": len(nb), "sha256": hashlib.sha256(nb).hexdigest()}
    (live / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    r = run(p)
    assert r.returncode != 0 and "has no sat_heads.role" in r.stderr, r.stderr[-800:]


@needs_bin
def test_logged_w_is_the_schedule(data):
    tmp, _ = data
    out = full(tmp, "sat_full", sat_cfg(tmp))
    steps = evs(out, "step")
    for e in steps:
        if e["step"] < CUT:
            assert e["symbol_targets"]["w"] == 1.0 * 2.0 ** (-e["step"] / HALF)
            assert e["symbol_targets"]["role_targets"] > 0
        else:
            assert "symbol_targets" not in e
    end = evs(out, "symbol_targets_end")
    assert len(end) == 1 and end[0]["step"] == CUT and end[0]["w"] == 0.0 and end[0]["through_step"] == CUT - 1
    tw = sum(e["symbol_targets"]["twin_rows"] for e in steps if e["step"] < CUT)
    assert end[0]["twin_rows"] == tw == end[0]["twin_rows_scheduled"] == -(-(CUT - 1) * BATCH * ACCUM // EVERY)
    assert end[0]["twin_positions"] == tw * T
    assert end[0]["natural_rows"] == (CUT - 1) * BATCH * ACCUM - tw
    assert end[0]["natural_positions"] == ((CUT - 1) * BATCH * ACCUM - tw) * T
    # The end's observed totals are the per-step counts summed, by source.
    for k in ("natural_role_targets", "natural_pointer_targets", "natural_pointers_dropped", "twin_role_targets",
              "twin_pointer_targets", "twin_pointers_dropped"):
        assert end[0][k] == sum(e["symbol_targets"][k] for e in steps if e["step"] < CUT), k
    for e in steps:
        if e["step"] < CUT:
            st = e["symbol_targets"]
            assert st["role_targets"] == st["natural_role_targets"] + st["twin_role_targets"]
            assert st["pointer_targets"] == st["natural_pointer_targets"] + st["twin_pointer_targets"]
            assert st["natural_rows"] + st["twin_rows"] == BATCH * ACCUM
    assert end[0]["twin_pointer_targets"] > 0 and end[0]["natural_pointer_targets"] > 0
    assert sum(e["symbol_targets"]["pointer_targets"] for e in steps if e["step"] < CUT) > 0
    assert sum(e["symbol_targets"]["pointer_targets_dropped"] for e in steps if e["step"] < CUT) > 0
    ev = evs(out, "symbol_targets_eval")
    assert [e["step"] for e in ev] == [s for s in range(4, CUT, 4)]
    for e in ev:
        assert e["windows"] >= 2 and e["role_targets"] > 0 and 0 <= e["role_acc"] <= 1 and 0 <= e["recency_acc"] <= 1
    st = evs(out, "symbol_targets")
    assert len(st) == 1
    for k in ("labels", "eval_labels", "twin_labels"):
        assert st[0][k]["hash_checks_passed"] == st[0][k]["spans"] > 0


@needs_bin
def test_twin_rows_and_windows(data):
    """Twin rows are exactly g % every == 0 before cut_step and read the twin corpus at draw % its limit; every other
    window, and every window from cut_step on, is the plain run's; the generator is consumed identically."""
    tmp, _ = data
    sat = full(tmp, "sat_full", sat_cfg(tmp))
    plain = full(tmp, "plain", None)
    wa, wb = evs(sat, "windows"), evs(plain, "windows")
    assert len(wa) == len(wb) == STEPS * ACCUM
    twin_lim = evs(sat, "symbol_targets")[0]["twins"]["limit"]
    n_twin = 0
    for x, y in zip(wa, wb):
        assert (x["step"], x["micro"]) == (y["step"], y["micro"]) and x["raw"] == y["raw"]
        assert x["lim"] == y["lim"] and x["src"] == y["src"]
        for b in range(BATCH):
            g = (x["step"] - 1) * BATCH * ACCUM + x["micro"] * BATCH + b
            want = x["step"] < CUT and g % EVERY == 0
            assert x["twin"][b] == int(want)
            if want:
                n_twin += 1
                assert x["at"][b] == x["raw"][b] % twin_lim
            else:
                assert x["at"][b] == y["at"][b]
    assert n_twin == -(-(CUT - 1) * BATCH * ACCUM // EVERY)


def expected_window_labels(spans, pos, at, b):
    """The labels of one window from the label file and tools/state_tokens.py's insertion (independent of the
    trainer's map): {index: [index, role, labelled, mention]}, [[query, gold]], dropped."""
    lo = next((r for r, p in enumerate(pos) if p >= at), len(pos))
    hi = next((r for r, p in enumerate(pos) if p >= at + T), len(pos))
    lab, ptr, dropped = {}, [], 0
    for s in spans:
        off, codes = s["raw_offset"], s["codes"]
        if off + len(codes) <= lo or off >= hi:
            continue
        for k, c in enumerate(codes):
            r = off + k
            if lo <= r < hi and c & (SL.LABELLED | SL.MENTION):
                i = b * T + pos[r] - at
                lab[i] = [i, (c & 7) if c & SL.LABELLED else 0, int(bool(c & SL.LABELLED)), int(bool(c & SL.MENTION))]
        for p_, back in s["pointers"]:
            rt, rg = off + p_, off + p_ - back
            if not lo <= rt < hi:
                continue
            if rg < lo:
                dropped += 1
                continue
            ptr.append([b * T + pos[rt] - at, b * T + pos[rg] - at])
    return lab, ptr, dropped


@needs_bin
def test_labels_land_on_their_tokens(data):
    """Every logged target, main and twin rows, is the label file's, at the position state-token insertion moved its
    token to (computed here from tools/state_tokens.py): roles, mentions, pointers and the pointers dropped because
    their gold lies before the window. No label sits on a state id."""
    tmp, _ = data
    out = full(tmp, "sat_full", sat_cfg(tmp))
    raw = list(SL_read_ids(tmp / "main.ids"))
    tw = list(SL_read_ids(tmp / "twin.ids"))
    post, pos = post_positions(raw)
    tpost, tpos = post_positions(tw)
    _, spans = SL.read(tmp / "A.satl")
    _, tspans = SL.read(tmp / "twin.satl")
    checked = straddle = 0
    for w in evs(out, "windows"):
        if w["step"] >= CUT:
            assert "labels" not in w
            continue
        lab, ptr, dropped = {}, [], 0
        for b in range(BATCH):
            twin = w["twin"][b]
            l_, p_, d_ = expected_window_labels(tspans if twin else spans, tpos if twin else pos, w["at"][b], b)
            lab.update(l_)
            ptr += p_
            dropped += d_
            ids = tpost if twin else post
            for i, row in l_.items():
                assert ids[w["at"][b] + i - b * T] < NV          # never a state id
            window = ids[w["at"][b]:w["at"][b] + T]
            straddle += any(t >= NV for t in window) and any(l_)
        assert sorted(w["labels"]) == sorted(lab.values())
        assert sorted(w["pointers"]) == sorted(ptr)
        assert w["dropped"] == dropped
        checked += len(lab)
    assert checked > 100 and straddle > 0


def SL_read_ids(path):
    b = Path(path).read_bytes()
    (n,) = struct.unpack_from("<Q", b, 16)
    return struct.unpack_from(f"<{n}H", b, 24)


@needs_bin
def test_a_span_whose_hash_differs_is_refused(data):
    tmp, _ = data
    sat = sat_cfg(tmp)
    sat["labels"] = str(tmp / "bad.satl")
    out = tmp / "bad"
    r = run(spec(tmp, out, sat))
    assert r.returncode != 0 and "differs from the corpus (sha256); refusing the run" in r.stderr
    assert not (out / "model.safetensors").exists()


@needs_bin
def test_exported_tensor_names_are_the_plain_runs(data):
    tmp, _ = data
    sat = full(tmp, "sat_full", sat_cfg(tmp))
    plain = full(tmp, "plain", None)
    assert st_keys(sat / "m.safetensors") == st_keys(plain / "m.safetensors")
    assert st_keys(sat / "model.safetensors") == st_keys(plain / "model.safetensors")
    assert st_keys(sat / "optim.safetensors") == st_keys(plain / "optim.safetensors")   # the end: no heads
    assert not any("sat" in k for k in st_keys(sat / "m.safetensors"))


@needs_bin
@pytest.mark.parametrize("change,why", [
    (lambda s, t: t.update(fused_head=True), "needs train.fused_head off"),
    (lambda s, t: t["symbol_targets"].update(cut_step=16), "must come before data.anneal"),
    (lambda s, t: t["symbol_targets"].update(cut_step=0), "1 <= cut_step"),
    (lambda s, t: t["symbol_targets"].update(half_life=0), "half_life > 0"),
    (lambda s, t: t["symbol_targets"].update(w0=-1), "w0 >= 0"),
    (lambda s, t: t["symbol_targets"]["twins"].update(every=1), "every >= 2"),
    (lambda s, t: t["symbol_targets"].update(labels="/nonexistent.satl"), "cannot read labels"),
    (lambda s, t: t["symbol_targets"].update(role_classes=4), "role_classes must be 8"),
    (lambda s, t: t["symbol_targets"].update(colour="teal"), "unknown key colour"),
])
def test_refused(data, change, why):
    tmp, _ = data
    out = tmp / "refused"
    p = spec(tmp, out, sat_cfg(tmp))
    s = json.loads(p.read_text())
    change(s, s["train"])
    p.write_text(json.dumps(s))
    r = run(p)
    assert r.returncode != 0 and why in (r.stderr + r.stdout), (r.stderr + r.stdout)[-800:]
