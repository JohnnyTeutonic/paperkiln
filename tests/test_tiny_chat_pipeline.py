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
    fake.write_text("import json, os, sys\n"
                    "arg = lambda k: sys.argv[sys.argv.index(k) + 1]\n"
                    "assert sys.argv[1] == 'run' and arg('--scorer') == 'mtstudio'\n"
                    "spec = json.load(open(arg('--model').split('=', 1)[1]))\n"
                    "assert os.path.exists(os.path.join(spec['out_dir'], spec['name'] + '.safetensors'))\n"
                    "m = 0.5 if spec['name'].endswith('r1') else 0.6\n"
                    "acc = {'gold': 0.65, 'self': m, 'twin': 0.7, 'noctx': 0.25}\n"
                    "fc = {k: {mode: {'all': {'mean': v, 'n': 100}} for mode, v in acc.items()}"
                    " for k in ('acc', 'acc_norm')}\n"
                    "os.makedirs(arg('--out'), exist_ok=True)\n"
                    "json.dump({'meta': {'gen': arg('--gen'), 'port': arg('--port')}, 'models': {'m': {'fc': fc}}},"
                    " open(os.path.join(arg('--out'), 'summary.json'), 'w'))\n")
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
