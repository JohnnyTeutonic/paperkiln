"""The tiny-chat Colab pipeline end to end on CPU at toy size
(experiments/tiny_chat_2026-10/PREREGISTRATION.md): tools/colab_chat/tiny_chat_data.py builds the
mixes, panels and SFT set from a toy FineWeb directory and toy human dialogues; tiny_chat_run.py
probes, screens, post-processes (SFT, zero-rate panels, Carry-5 through a stand-in tool), picks,
gates and chooses. Checks the files the jobs files rely on and the registered rules (dialogue
upsampling, the screen's tie rule, the gate's refusals). The data part needs no binary; the rest
needs MTSTUDIO_BIN (or ./build/mtstudio).
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import bpe  # noqa: E402
from test_mtstudio_chat import BIN  # noqa: E402

needs_bin = pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN to a built mtstudio")
DATA = [sys.executable, str(ROOT / "tools" / "colab_chat" / "tiny_chat_data.py")]
RUN = [sys.executable, str(ROOT / "tools" / "colab_chat" / "tiny_chat_run.py")]
EOT = "<|endoftext|>"
WORDS = "the cat sat on a mat and then it ran to the old red barn where a dog slept in hay".split()


def fineweb(d, n):
    d.mkdir(parents=True)
    docs = [" ".join(WORDS[(i + j) % len(WORDS)] for j in range(40)) for i in range(n)]
    # Bytes, not text: the mixer splits on "<|endoftext|>" + LF, which Windows text mode would turn into CRLF.
    (d / "pretrain_train.txt").write_bytes("".join(x + "\n" + EOT + "\n" for x in docs).encode("utf-8"))
    (d / "val.txt").write_bytes("".join(x + "\n" + EOT + "\n" for x in docs[:60]).encode("utf-8"))


def dialogues(d, n):
    d.mkdir(parents=True)
    rows = []
    for i in range(n):
        a, b = WORDS[i % len(WORDS)], WORDS[(3 * i + 1) % len(WORDS)]
        rows.append({"messages": [{"role": "user", "content": f"where is the {a} {i}"},
                                  {"role": "assistant", "content": f"the {a} is in the {b}"},
                                  {"role": "user", "content": f"and the {b}"},
                                  {"role": "assistant", "content": f"the {b} sat on a mat"}], "uri": f"toy/{i}"})
    rows.append({"messages": [{"role": "assistant", "content": "broken"}], "uri": "toy/bad"})
    # A user turn equal to a Carry-5 item's (4+ words): the data job's decontamination drops it.
    rows.append({"messages": [{"role": "user", "content": CARRY5_TURN},
                              {"role": "assistant", "content": "the cat sat on a mat"}], "uri": "toy/carry5"})
    (d / "dialogues.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (d / "text.txt").write_text("".join(json.dumps(r) + EOT + "\n" for r in rows), encoding="utf-8")


def _carry5_turn():
    with open(ROOT / "corpus" / "eval" / "carry5_v1.jsonl", encoding="utf-8") as f:
        for line in f:
            for t in json.loads(line)["turns"]:
                if len(t["user"].split()) >= 6:
                    return t["user"]


CARRY5_TURN = _carry5_turn()


def vocab(path):
    text = " ".join(WORDS) * 30 + " where is in and 0 1 2 3 4 5 6 7 8 9"
    bpe.write_gguf(str(path), bpe.train(text, 300))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    t = tmp_path_factory.mktemp("tc")
    fineweb(t / "fw", 1000)
    dialogues(t / "human" / "toy", 1000)
    vocab(t / "v.gguf")
    out = t / "data"
    argv = DATA + ["--out", str(out), "--vocab", str(t / "v.gguf"), "--fineweb", str(t / "fw"), "--shares", "0,50,100",
                   "--mix-bytes", "150000", "--max-epochs", "4", "--T", "64", "--seed", "1",
                   "--human-dir", str(t / "human"), "--human-fetch", ""]
    p = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return t, out


def test_data_job_outputs_and_upsampling(built):
    t, out = built
    led = json.loads((out / "data.ledger.json").read_text())
    for f in ("mix_s0.ids", "mix_s50.ids", "mix_s100.ids", "panel_fineweb.ids", "panel_dialogue.ids",
              "sft_human.jsonl"):
        assert (out / f).exists(), f
    assert led["human"]["dropped_not_alternating"] == 1
    assert sum(led["human"]["split"].values()) == 1000
    # Carry-5 is a decontamination target (amendment 1): its user turn's dialogue is dropped.
    assert led["steps"]["decontam"]["rows_dropped"] == 1 and led["human"]["carry5_v1_sha256"]
    m = led["mixes"]
    assert m["s0"]["dialogue_copies"] == 0 and m["s0"]["share_effective"] == 0
    # ~60 kB of pretraining dialogue for 75 kB (s50) and 150 kB (s100): copies cover the need, FineWeb unrepeated.
    assert m["s50"]["dialogue_copies"] >= 1 and m["s100"]["dialogue_copies"] > m["s50"]["dialogue_copies"]
    src = m["s50"]["mix_ledger"]["parts"]["main"]["sources"]
    assert abs(src["dialogue"]["bytes"] / (src["dialogue"]["bytes"] + src["fineweb"]["bytes"]) - 0.5) < 0.05
    sft = [json.loads(x) for x in (out / "sft_human.jsonl").read_text().splitlines()]
    assert sft and all(r["segments"][0]["loss"] is False and r["segments"][1]["loss"] is True for r in sft)


def test_measure_only_equals_the_data_job_and_pins_are_enforced(built, tmp_path):
    """P measured at home (--measure-only) is the data job's human_pt_bytes; --sources picks the
    sources; --require-pinned refuses bytes without a pin."""
    t, out = built
    led = json.loads((out / "data.ledger.json").read_text())
    base = DATA + ["--human-dir", str(t / "human"), "--human-fetch", ""]
    p = subprocess.run(base + ["--out", str(tmp_path / "m"), "--measure-only", "--sources", "toy"],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-2000:]
    assert json.loads(p.stdout.strip().splitlines()[-1])["human_pt_bytes"] == led["human_pt_bytes"]
    p = subprocess.run(base + ["--out", str(tmp_path / "m2"), "--measure-only", "--sources", "toy,absent"],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode != 0 and "absent" in p.stderr
    p = subprocess.run(base + ["--out", str(tmp_path / "m3"), "--measure-only", "--require-pinned"],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode != 0 and "not the pinned bytes" in p.stderr


def test_asha_promotion_rule(tmp_path):
    """asha_promote: the top `keep` by base-model acc_gold, ties by the dialogue panel; diverged and
    failed runs never promoted; results from this VM or from the uploaded state file; a candidate
    with no result anywhere stops the step; the next round reads the previous decisions."""
    chat = tmp_path / "chat"
    plan = {"configs": [{"id": f"c0{i}"} for i in range(5)],
            "rounds": [{"id": "r0", "budget": 10, "keep": 5}, {"id": "r1", "budget": 30, "keep": 2},
                       {"id": "r2", "budget": 100, "keep": 1}]}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    rows = {"c00": {"acc_gold": 0.50, "panels": {"dialogue": 3.0}},
            "c01": {"acc_gold": 0.60, "panels": {"dialogue": 3.1}},
            "c02": {"acc_gold": None, "diverged": True}}
    for c, r in rows.items():
        (chat / f"asha_r0_{c}").mkdir(parents=True)
        (chat / f"asha_r0_{c}" / "rung.json").write_text(json.dumps({"config": c, **r}))
    (chat / "STATUS_asha_r0_c04").write_text("FAILED\n")
    state = tmp_path / "asha_state.json"
    env = {**os.environ, "JOBS_CHAT_DIR": str(chat)}
    cmd = RUN + ["asha_promote", "--plan", str(tmp_path / "plan.json"), "--state", str(state)]
    r = subprocess.run(cmd + ["--out", str(chat / "asha_promote_r1"), "--round", "r1"], capture_output=True,
                       text=True, env=env, cwd=ROOT)
    assert r.returncode == 1 and "c03" in r.stderr  # c03's result is nowhere yet
    # c03 came home before the VM was replaced: the state file carries it, tied with c01 on acc_gold.
    state.write_text(json.dumps({"rungs": {"r0": {"c03": {"config": "c03", "acc_gold": 0.60,
                                                         "panels": {"dialogue": 2.9}}}}}))
    r = subprocess.run(cmd + ["--out", str(chat / "asha_promote_r1"), "--round", "r1"], capture_output=True,
                       text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    d = json.loads((chat / "asha_promote_r1" / "decisions.json").read_text())
    assert d["decisions"]["r1"] == {"c00": False, "c01": True, "c02": False, "c03": True, "c04": False}
    assert [x["config"] for x in d["ranking"]][:2] == ["c03", "c01"]  # tie on acc_gold: lower dialogue loss first
    # Round r2 on a fresh VM: the r1 decisions and results come from the state file only.
    import shutil
    shutil.rmtree(chat)
    chat.mkdir()
    state.write_text(json.dumps({"decisions": d["decisions"],
                                 "rungs": {"r1": {"c01": {"config": "c01", "acc_gold": 0.7},
                                                  "c03": {"config": "c03", "acc_gold": 0.65}}}}))
    r = subprocess.run(cmd + ["--out", str(chat / "asha_promote_r2"), "--round", "r2"], capture_output=True,
                       text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    d2 = json.loads((chat / "asha_promote_r2" / "decisions.json").read_text())
    assert [c for c, v in d2["decisions"]["r2"].items() if v] == ["c01"]


def test_short_dialogue_runs_at_the_largest_covered_share(tmp_path):
    fineweb(tmp_path / "fw", 400)
    dialogues(tmp_path / "human" / "toy", 20)
    vocab(tmp_path / "v.gguf")
    out = tmp_path / "data"
    p = subprocess.run(DATA + ["--out", str(out), "--vocab", str(tmp_path / "v.gguf"), "--fineweb", str(tmp_path / "fw"),
                               "--shares", "100", "--mix-bytes", "60000", "--max-epochs", "4", "--T", "64",
                               "--human-dir", str(tmp_path / "human"), "--human-fetch", ""],
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-2000:]
    m = json.loads((out / "data.ledger.json").read_text())["mixes"]["s100"]
    assert m["dialogue_copies"] == 4 and m["share_effective"] < 100 and "note" in m


def _carry12_only_turn():
    """A Carry-12 user turn of 4+ words that no Carry-5 item has."""
    c5 = {t["user"] for line in open(ROOT / "corpus" / "eval" / "carry5_v1.jsonl", encoding="utf-8")
          for t in json.loads(line)["turns"]}
    for line in open(ROOT / "corpus" / "eval" / "carry12_v1.jsonl", encoding="utf-8"):
        for t in json.loads(line)["turns"]:
            if len(t["user"].split()) >= 6 and t["user"] not in c5:
                return t["user"]


def test_long_carry_data_job_carry12_decontam_and_long_sft(tmp_path):
    """The long programme's data job (PREREGISTRATION_DRAFT.md sections 2 and 6): --carry12 drops a dialogue
    holding a Carry-12 user turn (Carry-5 still drops its own), counted, with the set's pinned sha256 in the
    ledger; --long-sft 3:4 writes cell D's SFT set, the same rows with each 3+-turn row four times."""
    fineweb(tmp_path / "fw", 400)
    d = tmp_path / "human" / "toy"
    dialogues(d, 200)
    rows = [json.loads(x) for x in (d / "dialogues.jsonl").read_text(encoding="utf-8").splitlines()]
    turn12 = _carry12_only_turn()
    rows.append({"messages": [{"role": "user", "content": turn12},
                              {"role": "assistant", "content": "the dog slept in hay"}], "uri": "toy/carry12"})
    for i in range(300):  # long dialogues: 4 exchanges each
        a = WORDS[i % len(WORDS)]
        ms = []
        for k in range(4):
            ms += [{"role": "user", "content": f"tell me about the {a} {i} {k}"},
                   {"role": "assistant", "content": f"the {a} sat on a mat {k}"}]
        rows.append({"messages": ms, "uri": f"toy/long/{i}"})
    (d / "dialogues.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    vocab(tmp_path / "v.gguf")

    def run(out, *extra):
        p = subprocess.run(DATA + ["--out", str(out), "--vocab", str(tmp_path / "v.gguf"), "--fineweb",
                                   str(tmp_path / "fw"), "--shares", "35", "--mix-bytes", "60000", "--T", "128",
                                   "--human-dir", str(tmp_path / "human"), "--human-fetch", "", *extra],
                           capture_output=True, text=True, cwd=ROOT)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
        return json.loads((out / "data.ledger.json").read_text())
    base = run(tmp_path / "plain")
    led = run(tmp_path / "long", "--carry12", "--long-sft", "3:4")
    assert base["steps"]["decontam"]["rows_dropped"] == 1 and "carry12_v1_sha256" not in base["human"]
    assert led["steps"]["decontam"]["rows_dropped"] == 2
    assert led["human"]["carry12_v1_sha256"] == "de24f2c726a1006e8e61542dd3b2dfbf8677b9e6fe689277cca035305bd32797"
    sft = [json.loads(x) for x in (tmp_path / "long" / "sft_human.jsonl").read_text().splitlines()]
    long4 = [json.loads(x) for x in (tmp_path / "long" / "sft_human_long4.jsonl").read_text().splitlines()]
    n_long = sum(1 for r in sft if sum(s["loss"] for s in r["segments"]) >= 3)
    ls = led["long_sft"]
    assert n_long > 0 and ls["rows"] == len(sft) and ls["rows_long"] == n_long
    assert len(long4) == ls["rows_out"] == len(sft) + 3 * n_long
    key = lambda r: json.dumps(r, sort_keys=True)  # noqa: E731
    from collections import Counter
    c = Counter(key(r) for r in long4)
    assert all(c[key(r)] == (4 if sum(s["loss"] for s in r["segments"]) >= 3 else 1) for r in sft)
    assert "sft_human_long4.jsonl" in led["outputs"]
    # The plain job's outputs are unchanged by the new options' code path when they are not given.
    assert "long_sft" not in base and not (tmp_path / "plain" / "sft_human_long4.jsonl").exists()


# A stand-in for tools/carry5_eval.py's `run` interface and summary schema.
FAKE_CARRY5 = ("import json, os, sys\n"
               "arg = lambda k: sys.argv[sys.argv.index(k) + 1]\n"
               "assert sys.argv[1] == 'run' and arg('--scorer') == 'mtstudio'\n"
               "spec = json.load(open(arg('--model').split('=', 1)[1]))\n"
               "assert os.path.exists(os.path.join(spec['out_dir'], spec['name'] + '.safetensors'))\n"
               "m = 0.5 if spec['name'].endswith('r1') else 0.6\n"
               "acc = {'gold': 0.65, 'self': m, 'twin': 0.7, 'noctx': 0.25}\n"
               "modes = [k for k in acc if k != 'self' or arg('--gen') != 'none']\n"
               # carry5_eval's summary lists every mode; an unscored one has mean None.
               "fc = {k: {mode: {'all': {'mean': acc[mode] + 0.05 if mode in modes else None,"
               " 'n': 100 if mode in modes else 0}} for mode in acc} for k in ('acc', 'acc_norm')}\n"
               "os.makedirs(os.path.join(arg('--out'), 'm'), exist_ok=True)\n"
               # Per-item records (amendment 2 pools from them): 20 recall items carrying the means
               # above, and 4 instruction items right in every mode, which the primary pool leaves out.
               "with open(os.path.join(arg('--out'), 'm', 'items.jsonl'), 'w') as f:\n"
               "    for i in range(20):\n"
               "        r = {'family': 'recall', 'fc': {mode: {'acc': float(i < round(20 * acc[mode])),"
               " 'acc_norm': 0.0} for mode in modes}, 'gen': {}}\n"
               "        f.write(json.dumps(r) + '\\n')\n"
               "    for i in range(4):\n"
               "        g = {mode: {'complies': i < 1} for mode in modes} if arg('--gen') != 'none' else {}\n"
               "        r = {'family': 'instruction', 'fc': {mode: {'acc': 1.0, 'acc_norm': 1.0} for mode in modes},"
               " 'gen': g}\n"
               "        f.write(json.dumps(r) + '\\n')\n"
               "json.dump({'meta': {'gen': arg('--gen'), 'port': arg('--port')}, 'models': {'m': {'fc': fc}}},"
               " open(os.path.join(arg('--out'), 'summary.json'), 'w'))\n")


def shape_spec(t, out, steps=20):
    spec = {"name": "tc_toy", "arch": {"preset": "llama-tiny", "rope_heads": "all",
                                       "custom": {"d": 32, "layers": 2, "heads": 2, "d_ff": 64}},
            "data": {"corpus": str(out / "mix_s50.ids"), "vocab": str(t / "v.gguf"), "vocab_cap": 300, "T": 64,
                     "max_tokens": 0},
            "train": {"steps": steps, "lr": 0.003, "batch": 2, "accum": 1, "eval_every": 10, "eval_windows": 4,
                      "checkpoint_every": 10, "gradmap_every": 0, "seed": 1, "optimizer": "adamw"},
            "export": {"formats": ["safetensors"]}, "_shape": {"size": "toy", "nonemb": 1, "emb": 1, "total": 2}}
    p = t / "shape_toy.json"
    p.write_text(json.dumps(spec))
    return p


@needs_bin
def test_probe_screen_post_pick(built, tmp_path):
    t, out = built
    env = {**os.environ, "JOBS_MTSTUDIO": BIN}
    spec = shape_spec(t, out)
    r = subprocess.run(RUN + ["probe", "--out", str(tmp_path / "probe"), "--spec", str(spec), "--corpus",
                              str(out / "mix_s50.ids"), "--steps", "60"], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    pr = json.loads((tmp_path / "probe" / "summary.json").read_text())
    assert pr["tok_per_s_steady"] and pr["tokens_per_step"] == 2 * 64
    r = subprocess.run(RUN + ["screen", "--out", str(tmp_path / "screen"), "--spec", str(spec), "--corpus",
                              str(out / "mix_s50.ids"), "--lr", "0.003", "--steps", "40", "--seed", "101"],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    sc = json.loads((tmp_path / "screen" / "summary.json").read_text())
    assert sc["final_val"] is not None and sc["diverged"] is False
    # A pretrained model, then the post job with a stand-in Carry-5 tool.
    pt = json.loads(spec.read_text())
    pt["out_dir"] = str(tmp_path / "pt")
    (tmp_path / "pt.json").write_text(json.dumps(pt))
    assert subprocess.run([BIN, "run", str(tmp_path / "pt.json")], capture_output=True).returncode == 0
    fake = tmp_path / "carry5_fake.py"
    # The stand-in speaks tools/carry5_eval.py's `run` interface and summary schema.
    fake.write_text(FAKE_CARRY5)
    items = tmp_path / "items.jsonl"
    items.write_text("{}\n")
    sft = {"data": {"vocab": str(t / "v.gguf"), "vocab_cap": 300, "T": 64, "sft_examples": str(out / "sft_human.jsonl")},
           "train": {"objective": "sft", "steps": 6, "lr": 0.001, "batch": 32, "eval_every": 3, "eval_batch": 4,
                     "checkpoint_every": 3, "gradmap_every": 0, "seed": 1, "optimizer": "adamw"},
           "export": {"formats": ["safetensors", "gguf"]}}
    (tmp_path / "sft.json").write_text(json.dumps(sft))
    env5 = {**env, "TINY_CHAT_CARRY5": str(fake)}
    for seed in (1, 2):
        post = tmp_path / f"post_r{seed}"
        r = subprocess.run(RUN + ["post", "--out", str(post), "--spec", str(spec), "--pt-weights",
                                  str(tmp_path / "pt" / "tc_toy.safetensors"), "--pt-dir", str(tmp_path / "pt"),
                                  "--sft-template", str(tmp_path / "sft.json"), "--seed", str(seed), "--cell", "s50",
                                  "--share", "50", "--panel", f"fineweb={out / 'panel_fineweb.ids'}",
                                  "--panel", f"dialogue={out / 'panel_dialogue.ids'}", "--carry5-items", str(items)],
                           capture_output=True, text=True, env=env5, cwd=ROOT)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        s = json.loads((post / "summary.json").read_text())
        assert set(s["panels"]) == {"fineweb", "dialogue"} and all(v is not None for v in s["panels"].values())
        assert s["sft_evals"] and s["carry5"]["meta"]["gen"] == "mtstudio" and s["carry5_base"]["meta"]["gen"] == "none"
        assert s["carry5"]["meta"]["port"] != s["carry5_base"]["meta"]["port"]
        assert abs(s["primary"] - ((0.5 if seed == 1 else 0.6) - 0.7)) < 1e-9
        # Amendment 2: the pool leaves the instruction items out; the all-items values stay beside it.
        assert s["carry5"]["pools"]["acc"]["primary"]["twin"]["n"] == 20 and s["carry5"]["ctx"]["noctx"] == 0.25
        assert s["carry5"]["acc"]["twin"] == 0.75 and s["primary_all_items"] is not None
        assert s["carry5"]["pools"]["instruction_complies_gen"]["self"] == {"mean": 0.25, "n": 4}
        assert "self" not in s["carry5_base"]["pool"] and abs(s["final_rung"]["acc_gold_all"] - 0.70) < 1e-9
    r = subprocess.run(RUN + ["pick", "--out", str(tmp_path / "pick"), "--run", str(tmp_path / "post_r1"),
                              "--run", str(tmp_path / "post_r2")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-1000:]
    best = json.loads((tmp_path / "pick" / "best.json").read_text())
    assert best["best"]["seed"] == 2 and (tmp_path / "pick" / "model.gguf").exists()
    s = json.loads((tmp_path / "post_r1" / "summary.json").read_text())
    assert abs(s["final_rung"]["Y_base"] - (0.65 - 0.7)) < 1e-9 and s["final_rung"]["acc_gold"] == 0.65
    # A rung: the checkpoint at step 10 kept aside (jobs_vm.py's snapshot), scored as the post job scores
    # the base model, plus the run's validation loss at that step.
    import shutil
    snap = tmp_path / "pt" / "rung_10"
    snap.mkdir()
    shutil.copyfile(tmp_path / "pt" / "model.safetensors", snap / "model.safetensors")
    (snap / "state.txt").write_text("10\n{}\n")
    rung = RUN + ["rung", "--out", str(tmp_path / "rung"), "--spec", str(spec), "--weights",
                  str(snap / "model.safetensors"), "--pt-dir", str(tmp_path / "pt"), "--job", "pt", "--cell", "s50",
                  "--seed", "1", "--share", "50", "--panel", f"fineweb={out / 'panel_fineweb.ids'}",
                  "--panel", f"dialogue={out / 'panel_dialogue.ids'}", "--carry5-items", str(items)]
    r = subprocess.run(rung + ["--step", "10"], capture_output=True, text=True, env=env5, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    g = json.loads((tmp_path / "rung" / "rung.json").read_text())
    assert g["step"] == 10 and g["acc_gold"] == 0.65 and g["val_loss"] is not None
    assert set(g["panels"]) == {"fineweb", "dialogue"} and g["carry5_base"]["meta"]["gen"] == "none"
    r = subprocess.run(rung[:4] + [str(tmp_path / "rung2")] + rung[5:] + ["--step", "20"], capture_output=True,
                       text=True, env=env5, cwd=ROOT)
    assert r.returncode != 0 and "holds step 10" in r.stderr  # a snapshot of another step is refused
    # A successive-halving candidate: trained from scratch to its budget, then the same rung metrics.
    r = subprocess.run(RUN + ["asha_train", "--out", str(tmp_path / "asha"), "--spec", str(spec), "--corpus",
                              str(out / "mix_s50.ids"), "--budget", "20", "--lr", "0.003", "--warmup-frac", "0.1",
                              "--seed", "1001", "--share", "50", "--config", "c00", "--round", "r0",
                              "--panel", f"dialogue={out / 'panel_dialogue.ids'}", "--carry5-items", str(items)],
                       capture_output=True, text=True, env=env5, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    a = json.loads((tmp_path / "asha" / "rung.json").read_text())
    assert a["budget"] == 20 and not a["diverged"] and a["acc_gold"] == 0.65 and len(a["evals"]) >= 2
    assert (tmp_path / "asha" / "run" / "events.jsonl").exists()


def test_choose_rule_and_gate(tmp_path):
    art = tmp_path / "art"
    rows = {"20m_L2": [(5e-4, 3.50, False), (1e-3, 3.495, False), (2e-3, 3.60, False)],
            "20m_L8": [(5e-4, 3.40, False), (1e-3, 3.30, False), (2e-3, 3.10, True)]}
    for sid, ss in rows.items():
        for k, (lr, v, dv) in enumerate(ss):
            d = art / f"screen_{sid}_lr{k}"
            d.mkdir(parents=True)
            (d / "summary.json").write_text(json.dumps({"lr": lr, "final_val": v, "diverged": dv, "rc": 0,
                                                        "tok_per_s_steady": 1.0}))
    choice = tmp_path / "lr_choice.json"
    r = subprocess.run(RUN + ["choose", "--art", str(art), "--out", str(choice)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lr = json.loads(choice.read_text())["lr"]
    assert lr == {"20m_L2": 5e-4, "20m_L8": 1e-3}  # L2: tie within 0.01 -> smaller; L8: diverged screen excluded
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"name": "pt_a", "spec": {"name": "tc_20m_L8", "train": {"lr": 0.001}}},
                                         {"name": "post_a", "cmd": ["x"]}]}))
    items = tmp_path / "items.jsonl"
    items.write_text("{}\n")
    fake = tmp_path / "c5.py"
    fake.write_text("")
    env = {**os.environ, "TINY_CHAT_CARRY5": str(fake)}
    design = tmp_path / "design.json"
    design.write_text(json.dumps({"levels": [0, 13, 26, 52], "e1_share": 52, "final": False}))
    gate = RUN + ["gate", "--out", str(tmp_path / "g"), "--jobs", str(jobs), "--carry5-items", str(items),
                  "--design", str(design)]
    # A provisional design (the default human set not all measured) stops Stage 0 and the E runs.
    assert subprocess.run(gate + ["--stage0"], env=env, capture_output=True).returncode == 1
    assert subprocess.run(gate + ["--choice", str(choice)], env=env, capture_output=True).returncode == 1
    design.write_text(json.dumps({"levels": [0, 13, 26, 52], "e1_share": 52, "final": True}))
    assert subprocess.run(gate + ["--stage0"], env=env, capture_output=True).returncode == 0
    assert subprocess.run(gate + ["--choice", str(choice)], env=env, capture_output=True).returncode == 0
    assert subprocess.run(gate + ["--choice", str(tmp_path / "none.json")], env=env, capture_output=True).returncode == 1
    jobs.write_text(json.dumps({"jobs": [{"name": "pt_a", "spec": {"name": "tc_20m_L8", "train": {"lr": 0.002}}}]}))
    r = subprocess.run(gate + ["--choice", str(choice)], env=env, capture_output=True, text=True)
    assert r.returncode == 1 and "Stage 0 chose 0.001" in r.stdout
    os.remove(fake)
    jobs.write_text(json.dumps({"jobs": [{"name": "pt_a", "spec": {"name": "tc_20m_L8", "train": {"lr": 0.001}}}]}))
    assert subprocess.run(gate + ["--choice", str(choice)], env=env, capture_output=True).returncode == 1


# ---- amendment 2 (23 Oct 2026): failed screens are rerun; the amended Carry-5 pools ----
def _screen(d, lr, rc, final_val, first=9.8, last=4.5):
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"lr": lr, "rc": rc, "final_val": final_val, "first_loss": first,
                                                "last_loss": last, "diverged": final_val is None,
                                                "tok_per_s_steady": 1.0}))


def test_choose_reads_reruns_of_failed_screens(tmp_path):
    a1, a2 = tmp_path / "attempt1", tmp_path / "rerun"
    # Attempt 1: L2 finished at 5e-4 only, its other two crashed (rc 1, finite losses: FAILED, not
    # diverged); L8 crashed in all three, one of them after a single step.
    _screen(a1 / "screen_20m_L2_lr0", 5e-4, 0, 4.44)
    _screen(a1 / "screen_20m_L2_lr1", 1e-3, 1, None, first=None, last=None)
    _screen(a1 / "screen_20m_L2_lr2", 2e-3, 1, None, first=None, last=None)
    for k, lr in enumerate((5e-4, 1e-3, 2e-3)):
        _screen(a1 / f"screen_20m_L8_lr{k}", lr, 1, None, last=9.8 if k == 2 else 5.2)
    choice = tmp_path / "lr_choice.json"
    r = subprocess.run(RUN + ["choose", "--art", str(a1), "--out", str(choice)], capture_output=True, text=True)
    assert r.returncode != 0 and "FAILED" in r.stderr and not choice.exists()
    # The rerun: L2's two failed screens finish (2e-3 diverges by the registered definition), L8's three finish.
    _screen(a2 / "screen_20m_L2_lr1", 1e-3, 0, 4.30)
    _screen(a2 / "screen_20m_L2_lr2", 2e-3, 0, 9.9, first=9.8, last=9.9)
    for k, (lr, v) in enumerate(((5e-4, 4.40), (1e-3, 4.25), (2e-3, 4.255))):
        _screen(a2 / f"screen_20m_L8_lr{k}", lr, 0, v)
    r = subprocess.run(RUN + ["choose", "--art", str(a1), "--art", str(a2), "--out", str(choice)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads(choice.read_text())
    assert out["lr"] == {"20m_L2": 1e-3, "20m_L8": 1e-3}  # L8: 2e-3 within 0.01 of 1e-3, so the smaller wins
    l2 = {s["lr"]: s for s in out["screens"]["20m_L2"]}
    assert l2[2e-3]["status"] == "DIVERGED" and l2[5e-4]["art"] == str(a1) and l2[1e-3]["art"] == str(a2)
    assert [x["status"] for x in l2[1e-3]["attempts"]] == ["FAILED", "FINISHED"]
    # A valid screen run twice is refused (no choosing among reruns).
    choice.unlink()
    _screen(a2 / "screen_20m_L2_lr0", 5e-4, 0, 4.10)
    r = subprocess.run(RUN + ["choose", "--art", str(a1), "--art", str(a2), "--out", str(choice)],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "never rerun" in r.stderr


def test_screen_status_on_the_attempt1_receipts():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tcr", ROOT / "tools" / "colab_chat" / "tiny_chat_run.py")
    tcr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcr)
    base = ROOT / "experiments" / "tiny_chat_2026-10" / "stage0" / "attempt1"
    st = {p.parent.name: tcr.screen_status(json.loads(p.read_text())) for p in base.glob("*/screen_*/summary.json")}
    assert len(st) == 27 and sorted(set(st.values())) == ["FAILED", "FINISHED"]
    assert sorted(k for k, v in st.items() if v == "FINISHED") == [
        "screen_20m_L16_lr1", "screen_20m_L16_lr2", "screen_20m_L2_lr0", "screen_20m_L32_lr2", "screen_20m_L4_lr0"]
    assert tcr.screen_status({"rc": 0, "final_val": float("nan"), "first_loss": 9.0}) == "DIVERGED"
    assert tcr.screen_status({"rc": 1, "first_loss": 9.0, "last_loss": float("nan")}) == "DIVERGED"


def test_carry5_pools_reproduce_the_sanity_receipts():
    """carry5_pools over the live model's per-item records equals carry5_eval's own family means,
    leaves the instruction family out of the primary pool, and reads generation compliance."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("tcr", ROOT / "tools" / "colab_chat" / "tiny_chat_run.py")
    tcr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcr)
    for run, n_primary in (("fc_all", 888), ("gen7", 127)):
        d = ROOT / "experiments" / "carry5_2026-10-23" / "sanity" / run
        p = tcr.carry5_pools(d / "live" / "items.jsonl")
        ref = json.loads((d / "summary.json").read_text())["models"]["live"]["fc"]["acc"]
        for mode, fams in p["acc"]["family"].items():
            for fam, v in fams.items():
                assert round(v["mean"], 4) == ref[mode]["family"][fam]["mean"], (run, mode, fam)
        assert p["acc"]["primary"]["twin"]["n"] == n_primary
        num = sum(ref["twin"]["family"][f]["mean"] * ref["twin"]["family"][f]["n"] for f in tcr.PRIMARY_FAMILIES)
        assert abs(p["acc"]["primary"]["twin"]["mean"] - num / n_primary) < 1e-3
    assert p["instruction_complies_gen"]["self"] == {"mean": 0.0, "n": 21}  # gen7: complies 0/21


# ---- amendment 3 (23 Oct 2026): the floor pilot's rate and the gate's pilot checks ----
def _pilot_lr(tmp_path, arts, *extra):
    out = tmp_path / "pilot_lr.json"
    if out.exists():
        out.unlink()
    r = subprocess.run(RUN + ["pilot_lr", *[x for a in arts for x in ("--art", str(a))], "--out", str(out), *extra],
                       capture_output=True, text=True)
    return r, (json.loads(out.read_text()) if out.exists() else None)


def test_pilot_lr_registered_rule_then_fallbacks(tmp_path):
    a1, a2 = tmp_path / "attempt1", tmp_path / "rerun"
    _screen(a1 / "screen_20m_L4_lr0", 5e-4, 0, 4.34)
    _screen(a1 / "screen_20m_L4_lr1", 1e-3, 1, None, first=None, last=None)
    _screen(a1 / "screen_20m_L16_lr1", 1e-3, 0, 4.24)
    _screen(a1 / "screen_20m_L16_lr2", 2e-3, 0, 4.48)
    _screen(a1 / "screen_20m_L32_lr2", 2e-3, 0, 4.28)
    for k, lr in enumerate((5e-4, 1e-3, 2e-3)):
        _screen(a1 / f"screen_20m_L8_lr{k}", lr, 1, None, first=None, last=None)
    # Nothing for L8 yet: the pilot waits, and without --session-ended no fallback applies.
    r, out = _pilot_lr(tmp_path, [a1, a2])
    assert r.returncode != 0 and out is None and "waits" in r.stderr
    # The session ended with every L8 screen failed again: F2, the smaller of the neighbours' rates.
    for k, lr in enumerate((5e-4, 1e-3, 2e-3)):
        _screen(a2 / f"screen_20m_L8_lr{k}", lr, 1, None, first=None, last=None)
    r, out = _pilot_lr(tmp_path, [a1, a2], "--session-ended")
    assert r.returncode == 0, r.stderr
    assert out["lr"] == 5e-4 and out["source"] == "F2_neighbours_20m_L4_20m_L16" and out["pilot_only"]
    # One L8 screen finished on the rerun: F1, the rule over L8's finished screens.
    import shutil
    shutil.rmtree(a2 / "screen_20m_L8_lr1")
    _screen(a2 / "screen_20m_L8_lr1", 1e-3, 0, 4.20)
    r, out = _pilot_lr(tmp_path, [a1, a2], "--session-ended")
    assert r.returncode == 0 and out["lr"] == 1e-3 and out["source"] == "F1_finished_screens"
    # All three resolved: section 4's rule exactly, as choose would apply it (tie within 0.01 -> smaller).
    for k, (lr, v) in ((0, (5e-4, 4.205)), (2, (2e-3, 4.30))):
        shutil.rmtree(a2 / f"screen_20m_L8_lr{k}")
        _screen(a2 / f"screen_20m_L8_lr{k}", lr, 0, v)
    r, out = _pilot_lr(tmp_path, [a1, a2])
    assert r.returncode == 0 and out["lr"] == 5e-4 and out["source"] == "registered"
    # All three diverged: the first rate of section 4's downward extension.
    for k, lr in enumerate((5e-4, 1e-3, 2e-3)):
        shutil.rmtree(a2 / f"screen_20m_L8_lr{k}")
        _screen(a2 / f"screen_20m_L8_lr{k}", lr, 0, 9.9, first=9.8, last=9.9)
    r, out = _pilot_lr(tmp_path, [a1, a2])
    assert r.returncode == 0 and out["lr"] == 2.5e-4 and out["source"] == "all_diverged"


def test_pilot_lr_f3_and_refusal(tmp_path):
    a1 = tmp_path / "attempt1"
    _screen(a1 / "screen_20m_L2_lr1", 1e-3, 0, 4.4)
    _screen(a1 / "screen_20m_L2_lr2", 2e-3, 0, 4.3)
    _screen(a1 / "screen_20m_L8_lr0", 5e-4, 1, None, first=None, last=None)
    r, out = _pilot_lr(tmp_path, [a1], "--session-ended")  # L8 is the shallowest-but-one: no L4/L16 screens
    assert r.returncode == 0 and out["lr"] == 1e-3 and out["source"] == "F3_smallest_finished_20m"
    b = tmp_path / "b"
    _screen(b / "screen_20m_L8_lr0", 5e-4, 1, None, first=None, last=None)
    r, out = _pilot_lr(tmp_path, [b], "--session-ended")
    assert r.returncode != 0 and out is None and "Jonathan decides" in r.stderr


def test_gate_pilot_lr_and_verdict(tmp_path):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"name": "pt", "spec": {"name": "tc_20m_L8", "train": {"lr": 0.0005}}}]}))
    items = tmp_path / "items.jsonl"
    items.write_text("{}" + chr(10))
    fake = tmp_path / "c5.py"
    fake.write_text("")
    env = {**os.environ, "TINY_CHAT_CARRY5": str(fake)}
    design = tmp_path / "design.json"
    design.write_text(json.dumps({"levels": [0, 17, 35, 70], "e1_share": 35, "final": True}))
    choice, plr, verdict = tmp_path / "lr_choice.json", tmp_path / "pilot_lr.json", tmp_path / "verdict.json"
    gate = RUN + ["gate", "--out", str(tmp_path / "g"), "--jobs", str(jobs), "--carry5-items", str(items),
                  "--design", str(design), "--choice", str(choice)]
    ok = lambda *x: subprocess.run(gate + list(x), env=env, capture_output=True).returncode == 0  # noqa: E731
    assert not ok("--pilot-lr", str(plr))                       # no choice and no pilot rate
    plr.write_text(json.dumps({"shape": "20m_L8", "lr": 5e-4, "source": "F2"}))
    assert ok("--pilot-lr", str(plr)) and not ok()               # the pilot rate serves the pilot's gate only
    choice.write_text(json.dumps({"lr": {"20m_L8": 1e-3}}))
    assert not ok("--pilot-lr", str(plr))                        # Stage 0's choice wins over the pilot rate
    jobs.write_text(json.dumps({"jobs": [{"name": "pt", "spec": {"name": "tc_20m_L8", "train": {"lr": 0.001}}}]}))
    assert ok() and not ok("--pilot-verdict", str(verdict), "--need-pilot", "PASS")
    verdict.write_text(json.dumps({"verdict": "FAIL"}))
    assert not ok("--pilot-verdict", str(verdict), "--need-pilot", "PASS")
    assert ok("--pilot-verdict", str(verdict), "--need-pilot", "FAIL")
    verdict.write_text(json.dumps({"verdict": "PASS"}))
    assert ok("--pilot-verdict", str(verdict), "--need-pilot", "PASS")


# ---- amendment 3, A3.3 (23 Oct 2026): the recursion arm ----
posix_only = pytest.mark.skipif(os.name == "nt", reason="fake build tools are POSIX shell scripts")


def test_choose_rec_four_rates_drop_and_merge(tmp_path):
    art = tmp_path / "s0_rec"
    rates = (2.5e-4, 5e-4, 1e-3, 2e-3)
    for k, (lr, v) in enumerate(zip(rates, (4.30, 4.295, 4.20, 4.50))):
        _screen(art / f"screen_20m_R4_lr{k}", lr, 0, v)  # 1e-3 best; nothing within 0.01 below it
    for k, (lr, v) in enumerate(zip(rates, (4.31, 4.305, 4.40, 4.60))):
        _screen(art / f"screen_20m_R2_lr{k}", lr, 0, v)  # 2.5e-4 and 5e-4 tied: the smaller wins
    for k, lr in enumerate(rates):
        _screen(art / f"screen_20m_MC7_lr{k}", lr, 0, 9.9, first=9.8, last=9.9)  # all four diverged
    out = tmp_path / "lr_choice_rec.json"
    r = subprocess.run(RUN + ["choose_rec", "--art", str(art), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.loads(out.read_text())
    assert d["lr"] == {"20m_R2": 2.5e-4, "20m_R4": 1e-3} and list(d["dropped"]) == ["20m_MC7"]
    assert d["grid"] == list(rates)
    # Section 4's plain choose still refuses a shape whose screens all diverged, and a three-rate grid.
    r = subprocess.run(RUN + ["choose", "--art", str(art), "--out", str(tmp_path / "plain.json")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    # R8's screens, on the R8 rule, add their cell to the same file; a cell is chosen once.
    r8 = tmp_path / "s0_rec_r8"
    for k, (lr, v) in enumerate(zip(rates, (4.10, 4.05, 4.30, 4.40))):
        _screen(r8 / f"screen_20m_R8_lr{k}", lr, 0, v)
    r = subprocess.run(RUN + ["choose_rec", "--art", str(r8), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["lr"]["20m_R8"] == 5e-4
    r = subprocess.run(RUN + ["choose_rec", "--art", str(r8), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode != 0 and "already holds" in r.stderr
    # A screen at an unregistered rate is refused.
    bad = tmp_path / "bad"
    for k, lr in enumerate((1.25e-4, 5e-4, 1e-3, 2e-3)):
        _screen(bad / f"screen_20m_R2_lr{k}", lr, 0, 4.3)
    r = subprocess.run(RUN + ["choose_rec", "--art", str(bad), "--out", str(tmp_path / "x.json")],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "registered" in r.stderr


def test_gate_recursion_needs(tmp_path):
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"name": "pt", "spec": {"name": "tc_20m_R2", "train": {"lr": 0.00025}}}]}))
    items = tmp_path / "items.jsonl"
    items.write_text("{}" + chr(10))
    fake = tmp_path / "c5.py"
    fake.write_text("")
    env = {**os.environ, "TINY_CHAT_CARRY5": str(fake)}
    design = tmp_path / "design.json"
    design.write_text(json.dumps({"levels": [0, 17, 35, 70], "e1_share": 35, "final": True}))
    choice, verdict = tmp_path / "lr_choice_rec.json", tmp_path / "verdict.json"
    cuda, dec = tmp_path / "cuda_gate.json", tmp_path / "rec_decision.json"
    choice.write_text(json.dumps({"lr": {"20m_R2": 2.5e-4}}))
    verdict.write_text(json.dumps({"verdict": "PASS"}))
    gate = RUN + ["gate", "--out", str(tmp_path / "g"), "--jobs", str(jobs), "--carry5-items", str(items),
                  "--design", str(design), "--choice", str(choice), "--pilot-verdict", str(verdict),
                  "--need-pilot", "PASS"]
    ok = lambda *x: subprocess.run(gate + list(x), env=env, capture_output=True).returncode == 0  # noqa: E731
    assert ok() and not ok("--need-cuda-gate", str(cuda))
    cuda.write_text(json.dumps({"ok": False}))
    assert not ok("--need-cuda-gate", str(cuda))
    cuda.write_text(json.dumps({"ok": True}))
    assert ok("--need-cuda-gate", str(cuda)) and not ok("--need-rec-r8", str(dec))
    dec.write_text(json.dumps({"run_r8": False}))
    assert not ok("--need-rec-r8", str(dec))
    dec.write_text(json.dumps({"run_r8": True}))
    assert ok("--need-cuda-gate", str(cuda), "--need-rec-r8", str(dec))
    # The Stage 0 form (the arm's screens) checks the pilot and the R8 rule too.
    assert ok("--stage0", "--need-rec-r8", str(dec))
    verdict.write_text(json.dumps({"verdict": "FAIL"}))
    assert not ok("--stage0") and not ok()


def test_probe_and_screen_refuse_without_a_passed_cuda_gate(tmp_path):
    g = tmp_path / "cuda_gate"
    g.mkdir()
    (g / "cuda_gate.json").write_text(json.dumps({"ok": False, "failed": ["./test_recursion"]}))
    for mode, extra in (("probe", ["--steps", "300"]), ("screen", ["--lr", "0.001", "--steps", "1500", "--seed", "101"])):
        out = tmp_path / mode
        r = subprocess.run(RUN + [mode, "--out", str(out), "--spec", "unused.json", "--corpus", "unused.ids",
                                  "--cuda-gate", str(g), *extra], capture_output=True, text=True, cwd=ROOT)
        s = json.loads((out / "summary.json").read_text())
        assert r.returncode == 1 and "CUDA gate did not pass" in s["refused"] and s["status"] == "FAILED"
    import importlib.util
    spec = importlib.util.spec_from_file_location("tcr", ROOT / "tools" / "colab_chat" / "tiny_chat_run.py")
    tcr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcr)
    assert tcr.screen_status(s) == "FAILED"  # a refusal says nothing about the rate: rerun, never chosen from


def fake_build(d, fail=()):
    """A build directory of shell scripts standing in for the validation binaries: each prints the
    MICROTORCH_* variables it saw; those in `fail` exit 1. A fake make records its arguments."""
    d.mkdir(parents=True, exist_ok=True)
    _, legs = TCR.validate_plan(str(ROOT / "tools" / "colab_cuda_validate.sh"))
    names = {b for _, b in legs} | {"test_a", "test_b"}
    for n in names:
        p = d / n
        p.write_text("#!/bin/sh\necho \"$0 dev=$MICROTORCH_DEVICE ops=$MICROTORCH_DEVICE_OPS "
                     "cublas=$MICROTORCH_CUBLAS res=$MICROTORCH_STEP_RESIDENCY\"\n"
                     + ("exit 1\n" if n in fail else "exit 0\n"))
        p.chmod(0o755)
    mk = d / "fake_make"
    mk.write_text("#!/bin/sh\necho \"$@\" > \"$(pwd)/make_args.txt\"\nexit 0\n")
    mk.chmod(0o755)
    return mk


def _tcr():
    import importlib.util
    spec = importlib.util.spec_from_file_location("tcr", ROOT / "tools" / "colab_chat" / "tiny_chat_run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TCR = _tcr()


@posix_only
def test_cuda_gate_runs_every_leg_and_the_production_kernels(tmp_path):
    build = tmp_path / "build"
    mk = fake_build(build, fail={"test_b"})
    script = tmp_path / "validate.sh"
    script.write_text("#!/usr/bin/env bash\n# comment ./test_x\nmake microtorch test_a test_b \\\n"
                      "     test_recursion -j\"$(nproc)\"\nMICROTORCH_DEVICE=cuda ./test_a\n./test_b\n"
                      "MICROTORCH_DEVICE=cuda MICROTORCH_DEVICE_OPS=1 ./test_recursion\necho done\n")
    env = {**os.environ, "TINY_CHAT_MAKE": str(mk), "MICROTORCH_CUBLAS": "1", "MICROTORCH_STEP_RESIDENCY": "1",
           "MICROTORCH_DEVICE": "cuda", "MICROTORCH_DEVICE_OPS": "1"}
    r = subprocess.run(RUN + ["cuda_gate", "--out", str(tmp_path / "g"), "--script", str(script), "--build-dir",
                              str(build)], capture_output=True, text=True, env=env, cwd=ROOT)
    res = json.loads((tmp_path / "g" / "cuda_gate.json").read_text())
    assert r.returncode == 1 and not res["ok"] and res["failed"] == ["{} ./test_b"]
    assert (build / "make_args.txt").read_text().split()[1:] == ["microtorch", "test_a", "test_b", "test_recursion"]
    legs = res["legs"]
    assert [x["binary"] for x in legs] == ["test_a", "test_b", "test_recursion", "test_recursion"]  # all ran
    # Script legs see only their own variables (jobs_vm.py's are removed) ...
    assert "cublas= res=" in legs[2]["output"] and "dev=cuda ops=1" in legs[2]["output"]
    # ... and the last runs test_recursion with the kernels mtstudio trains with, minus residency and deferral.
    assert legs[3]["production_kernels"] and "cublas=1 res=" in legs[3]["output"]
    assert legs[1]["log_tail"] and "log_tail" not in legs[0]
    fake_build(build)  # every leg passes
    r = subprocess.run(RUN + ["cuda_gate", "--out", str(tmp_path / "g2"), "--script", str(script), "--build-dir",
                              str(build)], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0 and json.loads((tmp_path / "g2" / "cuda_gate.json").read_text())["ok"]


@posix_only
def test_fresh_vm_runs_the_cuda_gate_before_a_probe(tmp_path):
    """On a replaced VM the gate job is home and not rerun: the first probe runs the gate there (the
    real validation script's legs, here stand-ins) and refuses when it fails."""
    build = tmp_path / "build"
    mk = fake_build(build, fail={"test_cuda_ops"})
    g = tmp_path / "cuda_gate"
    env = {**os.environ, "TINY_CHAT_MAKE": str(mk), "JOBS_MTSTUDIO": str(build / "mtstudio")}
    out = tmp_path / "probe"
    r = subprocess.run(RUN + ["probe", "--out", str(out), "--spec", "unused.json", "--corpus", "unused.ids",
                              "--cuda-gate", str(g)], capture_output=True, text=True, env=env, cwd=ROOT)
    s = json.loads((out / "summary.json").read_text())
    res = json.loads((g / "cuda_gate.json").read_text())
    assert r.returncode == 1 and "test_cuda_ops" in s["refused"] and not res["ok"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("tcr", ROOT / "tools" / "colab_chat" / "tiny_chat_run.py")
    tcr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tcr)
    legs = tcr.validate_plan(str(ROOT / "tools" / "colab_cuda_validate.sh"))[1]  # every leg of the script, ...
    assert len(res["legs"]) == len(legs) + 1 and res["legs"][-1]["production_kernels"]  # ... then test_recursion


@needs_bin
def test_recursion_probe_data_rebuild_and_post_batching(built, tmp_path):
    """A recursed toy model (1 + [1] x 15, effective depth 16) through the arm's commands: a probe on a
    'fresh VM' whose corpus is missing rebuilds it with the jobs file's data job; the post job's SFT
    batches 16 x 2 by effective depth."""
    t, out = built
    env = {**os.environ, "JOBS_MTSTUDIO": BIN}
    spec = json.loads(shape_spec(t, out).read_text())
    spec["arch"]["custom"]["recursion"] = {"prelude": 1, "block": 1, "repeats": 15, "coda": 0, "schedule": "fixed",
                                           "inject": "none"}
    spec["name"] = "tc_rec"
    sp = tmp_path / "shape_rec.json"
    sp.write_text(json.dumps(spec))
    data = tmp_path / "data2"
    jobs = tmp_path / "jobs.json"
    jobs.write_text(json.dumps({"jobs": [{"name": "data", "cmd": DATA + [
        "--out", "{out}", "--vocab", str(t / "v.gguf"), "--fineweb", str(t / "fw"), "--shares", "50", "--mix-bytes",
        "150000", "--max-epochs", "4", "--T", "64", "--seed", "1", "--human-dir", str(t / "human"),
        "--human-fetch", ""]}]}))
    r = subprocess.run(RUN + ["probe", "--out", str(tmp_path / "probe"), "--spec", str(sp), "--corpus",
                              str(data / "mix_s50.ids"), "--steps", "60", "--data-job", str(jobs)],
                       capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    pr = json.loads((tmp_path / "probe" / "summary.json").read_text())
    assert (data / "mix_s50.ids").exists() and pr["tok_per_s_steady"] and pr["arch"]["recursion"]["repeats"] == 15
    pt = dict(spec, out_dir=str(tmp_path / "pt"))
    (tmp_path / "pt.json").write_text(json.dumps(pt))
    p = subprocess.run([BIN, "run", str(tmp_path / "pt.json")], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:]
    fake = tmp_path / "carry5_fake.py"
    fake.write_text(FAKE_CARRY5)
    items = tmp_path / "items.jsonl"
    items.write_text("{}" + chr(10))
    sft = {"data": {"vocab": str(t / "v.gguf"), "vocab_cap": 300, "T": 64, "sft_examples": str(out / "sft_human.jsonl")},
           "train": {"objective": "sft", "steps": 4, "lr": 0.001, "batch": 32, "eval_every": 2, "eval_batch": 4,
                     "checkpoint_every": 2, "gradmap_every": 0, "seed": 1, "optimizer": "adamw"},
           "export": {"formats": ["safetensors", "gguf"]}}
    (tmp_path / "sft.json").write_text(json.dumps(sft))
    post = tmp_path / "post_r1"
    r = subprocess.run(RUN + ["post", "--out", str(post), "--spec", str(sp), "--pt-weights",
                              str(tmp_path / "pt" / "tc_rec.safetensors"), "--pt-dir", str(tmp_path / "pt"),
                              "--sft-template", str(tmp_path / "sft.json"), "--seed", "1", "--cell", "R16",
                              "--share", "50", "--panel", f"fineweb={out / 'panel_fineweb.ids'}",
                              "--carry5-items", str(items)],
                       capture_output=True, text=True, env={**env, "TINY_CHAT_CARRY5": str(fake)}, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    ss = json.loads((post / "sft_spec.json").read_text())
    assert (ss["train"]["batch"], ss["train"]["accum"]) == (16, 2) and ss["arch"]["custom"]["recursion"]["repeats"] == 15
    s = json.loads((post / "summary.json").read_text())
    assert s["primary"] is not None and s["panels"]["fineweb"] is not None
