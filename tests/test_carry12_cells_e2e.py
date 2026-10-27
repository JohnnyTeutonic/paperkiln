"""Carry-12's cells end to end on CPU at toy size (PREREGISTRATION_DRAFT.md's licence: the jobs files exercised
end to end before the freeze). For every machine file tools/colab_chat/carry12_<cell>_r<seed>.json:

- the data jobs (tiny_chat_data.py at T 1,024 with --carry12 and cell D's long set; E1's at T 512 for W512) on toy
  FineWeb and toy dialogues;
- the pretraining job's own spec (the jobs file's, with only the width, depth, vocabulary, corpus and step count
  made toy: T 1,024, 16 x 1 or 8 x 2, state tokens for S-gist, attention residuals for X), by `mtstudio run`, which
  exports safetensors and GGUF;
- every post job's own command line (the jobs file's, with the VM's paths mapped here): SFT (16 x 1,024 per step by
  --sft-batching spec; D's on the long set; W512's E1 SFT at T 512), the zero-rate panels, Carry-5 with ember
  generation after the spot-check and base-model Carry-5, then Carry-12 (forced choice in gold, self, near, twin and
  noctx; self history to the target at --max-new 48 by ember ec3201b after its spot-check, every reply through the
  parity gate; W512: forced choice only), on subsets of Carry-5 (every 26th item) and Carry-12 (every 110th);
- analyze.py on the post summaries (with C12_TOY_SEEDS=1,2,3 the registered path, otherwise --partial).

Needs POSIX, MTSTUDIO_BIN and TINY_CHAT_TEST_EMBER_DIR: an ember CPU build at ec3201b laid out as
tiny_chat_run.ensure_ember leaves it (EMBER_DIR/build/tinyllama_server and ember_build.json naming that commit).
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))
import bpe  # noqa: E402
import carry5_eval as ce  # noqa: E402
from test_tiny_chat_pipeline import DATA, WORDS, dialogues, fineweb  # noqa: E402

BIN = os.environ.get("MTSTUDIO_BIN")
EMBER_DIR = os.environ.get("TINY_CHAT_TEST_EMBER_DIR")
EMBER = "ec3201b04e6f2fcb6dffd3496b0c72b3bec0c54a"
SEEDS = [int(x) for x in os.environ.get("C12_TOY_SEEDS", "1").split(",")]
COLAB = ROOT / "tools" / "colab_chat"
LONG = ROOT / "experiments" / "tiny_chat_long_2026-10"
C = "/content/chat"
TOY = {"B": (64, 2), "Sgist": (64, 2), "P16": (64, 4), "X": (64, 4), "W512": (64, 2)}  # (d, layers)

pytestmark = pytest.mark.skipif(os.name == "nt" or not BIN or not EMBER_DIR,
                                reason="needs POSIX, MTSTUDIO_BIN and TINY_CHAT_TEST_EMBER_DIR (ember at ec3201b)")


def jobs(cell, seed):
    return {j["name"]: j for j in json.loads((COLAB / f"carry12_{cell}_r{seed}.json").read_text())["jobs"]}


def toy_arch(arch, cell):
    a = json.loads(json.dumps(arch))
    d, L = TOY[cell]
    a["custom"].update({"d": d, "layers": L, "heads": 2, "d_ff": 2 * d})
    return a


TOY_TOKENS = 2048   # tokens per step at toy size: the cells' 16 x 1,024 does not fit a CPU at T 1,024


def toy_spec(spec, cell, t, corpus, steps=4):
    s = json.loads(json.dumps(spec))
    s.pop("_shape", None)
    s["arch"] = toy_arch(s["arch"], cell)
    tr = s["train"]  # the same layout (accumulation kept) with fewer rows: 2 x 1 or 1 x 2 at T 1,024, 4 x 1 at 512
    tr["batch"] = max(1, TOY_TOKENS // (s["data"]["T"] * tr.get("accum", 1)))
    s["data"].update({"vocab": str(t / "v.gguf"), "vocab_cap": 4096, "corpus": str(corpus)})
    s["train"].update({"steps": steps, "eval_every": steps, "eval_windows": 2, "checkpoint_every": 1000,
                       "lr_schedule": {"warmup_steps": 1, "decay": "cosine", "min_lr_frac": 0.1}})
    return s


def toy_sft(path, t):
    s = json.loads((ROOT / path).read_text())
    s["data"].update({"vocab": str(t / "v.gguf"), "vocab_cap": 4096,
                      "sft_examples": s["data"]["sft_examples"].replace(C, str(t))})
    s["train"].update({"steps": 3, "eval_every": 3, "eval_batch": 2, "checkpoint_every": 1000,
                       "lr_schedule": {"warmup_steps": 1, "decay": "cosine", "min_lr_frac": 0.1}})
    p = t / ("toy_" + Path(path).name)
    p.write_text(json.dumps(s))
    return p


def run(argv, env=None, timeout=7200):
    p = subprocess.run(argv, capture_output=True, text=True, cwd=ROOT, env=env, timeout=timeout)
    assert p.returncode == 0, " ".join(map(str, argv))[:600] + "\n" + p.stdout[-3000:] + p.stderr[-3000:]
    return p


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    """The toy corpora, vocabulary, item subsets and both data jobs. With C12_TOY_DIR set, a persistent folder
    (on ext4) whose finished steps a rerun keeps: the whole run resumes step by step (data jobs, pretraining runs,
    post jobs, each resumable itself)."""
    if os.environ.get("C12_TOY_DIR"):
        t = Path(os.environ["C12_TOY_DIR"])
        if (t / "data512" / "data.ledger.json").exists() and (t / "data" / "data.ledger.json").exists():
            return t
        shutil.rmtree(t, ignore_errors=True)
        t.mkdir(parents=True)
    else:
        t = tmp_path_factory.mktemp("c12cells")
    fineweb(t / "fw", 2000)
    # the zero-rate panels hold windows of T + 1 = 1,025 tokens in a 5% held-out part: 21k+ tokens each
    val = (t / "fw" / "pretrain_train.txt").read_bytes()
    (t / "fw" / "val.txt").write_bytes(val)
    d = t / "human" / "toy"
    dialogues(d, 12000)
    rows = [json.loads(x) for x in (d / "dialogues.jsonl").read_text(encoding="utf-8").splitlines()]
    for i in range(300):  # long dialogues: 4 exchanges each (D's --long-sft 3:4 at toy size)
        a = WORDS[i % len(WORDS)]
        ms = []
        for k in range(4):
            ms += [{"role": "user", "content": f"tell me about the {a} {i} {k}"},
                   {"role": "assistant", "content": f"the {a} sat on a mat {k}"}]
        rows.append({"messages": ms, "uri": f"toy/long/{i}"})
    (d / "dialogues.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    c5 = ce.read_jsonl(ce.SET)[::26]
    c12 = ce.read_jsonl(ROOT / "corpus" / "eval" / "carry12_v1.jsonl")[::110]
    (t / "c5.jsonl").write_text("".join(json.dumps(x) + "\n" for x in c5), encoding="utf-8")
    (t / "c12.jsonl").write_text("".join(json.dumps(x) + "\n" for x in c12), encoding="utf-8")
    text = " ".join(WORDS) * 30 + " " + " ".join(
        x["user"] + " " + x["gold"] for it in c5 + c12 for x in it["turns"]) + " " + " ".join(
        it["frame"].format(c) for it in c5 + c12 for c in it["candidates"]) + " where is in and 0 1 2 3"
    bpe.write_gguf(str(t / "v.gguf"), bpe.train(text, 700))
    common = ["--vocab", str(t / "v.gguf"), "--fineweb", str(t / "fw"), "--shares", "35", "--mix-bytes", "300000",
              "--max-epochs", "4", "--seed", "1", "--human-dir", str(t / "human"), "--human-fetch", ""]
    run(DATA + ["--out", str(t / "data"), "--T", "1024", "--carry12", "--long-sft", "3:4", *common])
    run(DATA + ["--out", str(t / "data512"), "--T", "512", *common])
    for f in ("mix_s35.ids", "sft_human.jsonl", "sft_human_long4.jsonl", "panel_fineweb.ids", "panel_dialogue.ids"):
        assert (t / "data" / f).exists(), f
    return t


def pretrain(t, cell, seed):
    """The jobs file's pretraining spec at toy width and depth (W512: E1's shape spec at T 512, standing in for
    E1's home weights)."""
    if cell == "W512":
        spec = json.loads((ROOT / "experiments" / "tiny_chat_2026-10" / "specs" / "shape_60m_L8.json").read_text())
        spec["train"]["seed"] = seed
        name, corpus = f"tc_e1_60m_L8_r{seed}", t / "data512" / "mix_s35.ids"
    else:
        pt = jobs(cell, seed)[f"pt_c12_{cell}_r{seed}"]
        spec, name, corpus = pt["spec"], pt["model_name"], t / "data" / "mix_s35.ids"
    s = toy_spec(spec, cell, t, corpus)
    s.update({"name": name, "out_dir": str(t / f"pt_{name}")})
    p = t / f"pt_{name}.json"
    w = t / f"pt_{name}" / f"{name}.safetensors"
    if not (w.exists() and (t / f"pt_{name}" / "result.json").exists()):  # a resumed toy run keeps finished ones
        p.write_text(json.dumps(s))
        run([BIN, "run", str(p)])
    return w


def post(t, machine, cmd, cell, seed, weights, art):
    """The jobs file's post command with the VM's paths mapped to the toy's."""
    a = list(cmd)
    a[0] = sys.executable
    out = art / f"c12_{machine}_r{seed}" / f"post_c12_{cell}_r{seed}"
    a[a.index("{out}")] = str(out)
    i = a.index("--spec")
    shape = json.loads((ROOT / a[i + 1]).read_text())
    ts = toy_spec(shape, "W512" if cell == "W512" else machine, t, t / "data" / "mix_s35.ids")
    sp = t / f"shape_{cell}_r{seed}.json"
    sp.write_text(json.dumps(ts))
    a[i + 1] = str(sp)
    a[a.index("--pt-weights") + 1] = str(weights)
    a[a.index("--pt-dir") + 1] = str(weights.parent)
    i = a.index("--sft-template")
    a[i + 1] = str(toy_sft(a[i + 1], t))
    a = [x.replace(C, str(t)) if x.startswith(("fineweb=", "dialogue=")) else x for x in a]
    a[a.index("--carry5-items") + 1] = str(t / "c5.jsonl")
    a[a.index("--carry12-items") + 1] = str(t / "c12.jsonl")
    env = {**os.environ, "JOBS_MTSTUDIO": BIN, "TINY_CHAT_EMBER_DIR": EMBER_DIR, "TINY_CHAT_EMBER_APT": "0",
           "TINY_CHAT_SFT_TOKENS_PER_STEP": str(TOY_TOKENS), "TINY_CHAT_PANEL_WINDOWS": "16"}
    run(a, env=env)
    return json.loads((out / "summary.json").read_text())


@pytest.fixture(scope="module")
def cells_home(toy):
    """Every machine file's jobs at toy size, seeds C12_TOY_SEEDS: ART/c12_<machine>_r<seed>/post_c12_<cell>_r<seed>."""
    art = toy / "art"
    got = {}
    for seed in SEEDS:
        for machine in ("B", "Sgist", "P16", "X"):
            j = jobs(machine, seed)
            w = pretrain(toy, machine, seed)
            for cell in [machine] + (["D", "W512"] if machine == "B" else []):
                cmd = j[f"post_c12_{cell}_r{seed}"]["cmd"]
                ww = pretrain(toy, "W512", seed) if cell == "W512" else w
                got[(cell, seed)] = post(toy, machine, cmd, cell, seed, ww, art)
    return art, got


def test_post_summaries(cells_home):
    art, got = cells_home
    for (cell, seed), s in got.items():
        c12 = s["carry12"]
        assert "error" not in c12, (cell, c12.get("error"))
        assert c12["items"] == 14 and c12["meta"]["self_scope"] == "target" and c12["meta"]["near_self"] is False
        assert c12["max_new"] == 48 and c12["items_sha256"]
        assert s["sft_evals"] and set(s["panels"]) == {"fineweb", "dialogue"}
        if cell == "W512":
            assert s["primary"] is None and s["carry5"] == {"skipped": "post --no-carry5 (Carry-12 alone)"}
            assert c12["context"] == 512 and c12["meta"]["gen"] == "none" and s["Y12"] is None
            assert s["Y12_gold"] is not None and "sft_batching" not in s
            continue
        assert c12["context"] == 1024 and s["sft_batching"]["mode"] == "spec"
        # the cells' layout at toy tokens per step: accumulation 2 at 16 layers (P16, X), 1 otherwise
        assert (s["sft_batching"]["batch"], s["sft_batching"]["accum"]) == ((1, 2) if cell in ("P16", "X") else (2, 1))
        # Carry-5 first: ember at ec3201b with EMBER_DETERMINISTIC=1 after its spot-check (or mtstudio, flagged)
        g5 = s["gen_engine"]
        assert g5["ember_commit_pinned"] == EMBER and g5["ember_env"] == {"EMBER_GEMM_TF32": "0",
                                                                         "EMBER_DETERMINISTIC": "1"}
        assert s["primary"] is not None and s["carry5"]["meta"]["gen"] == g5["engine"]
        # then Carry-12: the same ember build, its own 20-item spot-check (here every item of the subset), the gate
        e12 = c12["gen_engine"]
        assert c12["ember_build"]["commit"] == EMBER and e12["requested"] == "ember"
        assert e12["ember_env"] == {"EMBER_GEMM_TF32": "0", "EMBER_DETERMINISTIC": "1"}
        assert e12["spot"]["items"] == 14
        assert s["Y12"] is not None and s["Y12_gold"] is not None
        if e12["engine"] == "ember":
            assert c12["gate"]["replies"] > 0 and c12["gate"]["argmax_identical"] == c12["gate"]["replies"]
        for mode in ("gold", "self", "near", "twin", "noctx"):
            assert c12["pools"]["acc"]["primary"][mode]["n"] > 0, (cell, mode)
        assert c12["over_ctx"]["gold"] == 0, (cell, c12["over_ctx"])
        out = art / f"c12_{'B' if cell == 'D' else cell}_r{seed}" / f"post_c12_{cell}_r{seed}"
        for f in json.loads((COLAB / f"carry12_{'B' if cell == 'D' else cell}_r{seed}.json").read_text())["jobs"]:
            if f["name"] == f"post_c12_{cell}_r{seed}":
                missing = [o for o in f["outputs"] if not (out / o).exists()]
                assert not missing, (cell, missing)  # every output the jobs file brings home exists
    # D trained on the long set from B's weights; S-gist's GGUF declares its state tokens; X's its residuals
    for seed in SEEDS:
        assert got[("D", seed)]["pt_weights_sha256"] == got[("B", seed)]["pt_weights_sha256"]
        assert got[("D", seed)]["sft_data_sha256"] != got[("B", seed)]["sft_data_sha256"]
        g = art / f"c12_Sgist_r{seed}" / f"post_c12_Sgist_r{seed}" / "sft" / f"tc_sft_Sgist_r{seed}.gguf"
        assert ce.state_template(str(g))[1] == 4
        assert got[("X", seed)]["arch"]["attn_res"] == {"form": "full", "scale": "sources"}


def test_rerun_is_idempotent(cells_home, toy):
    """A relaunch of a finished post job returns at once (summary present, Carry-12 included)."""
    art, got = cells_home
    seed = SEEDS[0]
    cmd = jobs("X", seed)[f"post_c12_X_r{seed}"]["cmd"]
    out = art / f"c12_X_r{seed}" / f"post_c12_X_r{seed}"
    before = (out / "summary.json").read_text()
    w = toy / f"pt_tc_c12_X_r{seed}" / f"tc_c12_X_r{seed}.safetensors"
    post(toy, "X", cmd, "X", seed, w, art)
    assert (out / "summary.json").read_text() == before


def test_analyze_on_the_toy_outputs(cells_home, tmp_path):
    art, got = cells_home
    tool = [sys.executable, str(LONG / "analyze.py"), "--art", str(art), "--lstar", "8", "--out",
            str(tmp_path / "a.json")]
    if SEEDS != [1, 2, 3]:
        tool.append("--partial")
    p = subprocess.run(tool, capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-3000:]
    res = json.loads((tmp_path / "a.json").read_text())["carry12"]
    assert set(res["cells"]) == {"W512", "B", "D", "Sgist", "P16", "X"}
    assert all(res["cells"][c]["Y12_gold"]["n"] == len(SEEDS) for c in res["cells"])
    if SEEDS == [1, 2, 3]:
        for k in ("HT_B_minus_W512_Y12_gold", "HD_D_minus_B", "HS1_Sgist_minus_B", "HX_gap_D_X",
                  "HX_gap_correction_slope", "HX_H_gap", "HX_main_X_minus_P16", "HX_isoplot_residual"):
            assert "verdict" in res[k], k
        assert set(res["validity_gate"]) == set(res["cells"])
    else:
        assert res["label"] == "DESCRIPTIVE, PARTIAL"
    shutil.copyfile(tmp_path / "a.json", art / "analysis_toy.json")
