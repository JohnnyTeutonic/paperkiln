"""The tiny-chat post job's generation through ember.cpp (tools/colab_chat/tiny_chat_run.py, 24 Oct 2026;
experiments/tiny_chat_2026-10/PREREGISTRATION.md, implementation note of that date).

The decision logic (spot-check comparison, engine choice, the llama_recursive CPU retry, fallbacks, the
jobs files) runs anywhere. The end-to-end test trains toy llama, fixed-recursion and MoR models, runs the
post job with --gen ember on a real ember.cpp build and the real tools/carry5_eval.py, and needs POSIX,
MTSTUDIO_BIN and TINY_CHAT_TEST_EMBER_DIR: an EMBER_DIR that tiny_chat_run.ensure_ember has built (a CPU
build: TINY_CHAT_EMBER_CMAKE=-DHAS_CUDA=OFF).
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_tiny_chat_pipeline import BIN, RUN, TCR, built, shape_spec  # noqa: E402,F401 (built: the fixture)

EMBER_DIR = os.environ.get("TINY_CHAT_TEST_EMBER_DIR")


def rec(reply, stop="eos", used=0):
    return {"reply": reply, "stop": stop, "history_used": used, "dropped": False}


def item(i, tweak=None):
    r = {"id": i, "self": [rec(f"s{k}", used=k) for k in range(5)], "gold": rec("g", used=2), "twin": rec("t"),
         "noctx": rec("n")}
    if tweak:
        tweak(r)
    return r


def test_compare_gens_counts_items_and_replies():
    a = {i: item(i) for i in ("a", "b", "c")}
    b = {"a": item("a"), "b": item("b", lambda r: r["self"][3].update(reply="other"))}
    res = TCR.compare_gens(a, b, ["a", "b", "c"])
    assert (res["items"], res["items_identical"], res["replies"], res["replies_identical"]) == (3, 1, 24, 15)
    diff = {r["id"]: r for r in res["differing"]}
    assert diff["b"]["diff"][0]["reply"] == "self4" and diff["b"]["diff"][0]["mtstudio"]["reply"] == "other"
    assert diff["c"]["missing"] == ["mtstudio"]
    # The stop reason and the history used count as well as the text.
    b2 = {"a": item("a", lambda r: r["twin"].update(stop="length"))}
    assert TCR.compare_gens(a, b2, ["a"])["items_identical"] == 0


class Fake:
    """Stand-ins for the build, the GGUF and the spot-check: SPOTS maps an EMBER_IR_DEVICE value to the
    identical count the spot-check returns (40 passes)."""

    def __init__(self, monkeypatch, spots, cuda=True, build_error=None):
        self.calls = []
        self.spots = spots

        def ensure_ember(log, commit=None):
            return {"error": build_error} if build_error else {"bin": "/x/tinyllama_server",
                                                               "commit": commit or TCR.EMBER_COMMIT,
                                                               "cuda": cuda, "built_here": True}

        def spot_check(spec, items, work, ember, env, tag):
            self.calls.append((tag, env.get("EMBER_IR_DEVICE"), env.get("EMBER_GEMM_TF32"), ember["gpu_layers"]))
            k = self.spots[env.get("EMBER_IR_DEVICE")]
            return {"identical": k, "items_identical": k, "n": 40, "pass": k == 40}

        monkeypatch.setattr(TCR, "ensure_ember", ensure_ember)
        monkeypatch.setattr(TCR, "ensure_f32_gguf", lambda spec, w, log: ("m.gguf", "ab" * 32))
        monkeypatch.setattr(TCR, "spot_check", spot_check)


LLAMA = {"d": 32, "layers": 2, "heads": 2, "d_ff": 64}
REC = {**LLAMA, "recursion": {"prelude": 1, "block": 1, "repeats": 3, "coda": 0, "schedule": "fixed"}}
MOR = {**LLAMA, "recursion": {"prelude": 1, "block": 1, "repeats": 3, "coda": 0, "schedule": "mor"}}


def choose(tmp_path, custom):
    return TCR.choose_engine("spec.json", "w.safetensors", "items.jsonl", str(tmp_path), custom)


def test_engine_ember_on_a_full_spot_check(tmp_path, monkeypatch):
    f = Fake(monkeypatch, {None: 40})
    d = choose(tmp_path, LLAMA)
    assert d["engine"] == "ember" and d["flag"] is None and d["family"] == "llama" and d["ir_device"] is None
    assert d["ember"] == {"bin": "/x/tinyllama_server", "gpu_layers": "-1"} and d["ember_env"] == {"EMBER_GEMM_TF32": "0"}
    assert f.calls == [("default", None, "0", "-1")]
    # Decided once per run: a rerun reads gen_engine.json and spot-checks nothing.
    f.spots = {None: 0}
    assert choose(tmp_path, LLAMA)["engine"] == "ember" and len(f.calls) == 1


def test_engine_mtstudio_for_the_whole_run_below_n(tmp_path, monkeypatch):
    Fake(monkeypatch, {None: 39})
    d = choose(tmp_path, LLAMA)
    assert d["engine"] == "mtstudio" and "39/40" in d["flag"] and "mtstudio throughout" in d["flag"]


def test_recursion_retries_on_the_cpu_interpreter(tmp_path, monkeypatch):
    f = Fake(monkeypatch, {None: 31, "cpu": 40})
    d = choose(tmp_path, REC)
    assert d["family"] == "llama_recursive" and d["engine"] == "ember" and d["ir_device"] == "cpu"
    assert d["ember_env"] == {"EMBER_GEMM_TF32": "0", "EMBER_IR_DEVICE": "cpu"}
    assert "31/40" in d["flag"] and "EMBER_IR_DEVICE=cpu passed" in d["flag"]
    assert [c[:2] for c in f.calls] == [("default", None), ("ir_cpu", "cpu")]
    # Both fail: mtstudio, with both counts in the flag.
    Fake(monkeypatch, {None: 31, "cpu": 38})
    d = choose(tmp_path / "b", REC)
    assert d["engine"] == "mtstudio" and "31/40" in d["flag"] and "EMBER_IR_DEVICE=cpu 38/40" in d["flag"]
    # Passing on the device: ember's CUDA interpreter generates, unflagged.
    Fake(monkeypatch, {None: 40})
    d = choose(tmp_path / "c", REC)
    assert d["engine"] == "ember" and d["ir_device"] == "gpu" and d["flag"] is None


def test_mor_runs_on_embers_cpu_path_and_falls_back_like_the_rest(tmp_path, monkeypatch):
    f = Fake(monkeypatch, {None: 40})
    d = choose(tmp_path, MOR)
    assert d["family"] == "llama_mor" and d["engine"] == "ember" and d["ir_device"] == "cpu" and len(f.calls) == 1
    Fake(monkeypatch, {None: 12})
    d = choose(tmp_path / "b", MOR)
    assert d["engine"] == "mtstudio" and "12/40" in d["flag"]  # no CPU retry: MoR is on the CPU already


def test_an_unbuildable_ember_falls_back_flagged(tmp_path, monkeypatch):
    f = Fake(monkeypatch, {None: 40}, build_error="ember cmake failed (rc 1)")
    d = choose(tmp_path, LLAMA)
    assert d["engine"] == "mtstudio" and "ember unavailable" in d["flag"] and f.calls == []


def test_jobs_files_generate_with_ember_except_the_pilot():
    colab = ROOT / "tools" / "colab_chat"
    for name in ("e5", "e1", "e1_60m", "e5_60m", "e1_rec", "e1_rec_r8", "e1_mor", "asha"):
        doc = json.loads((colab / f"tiny_chat_{name}.json").read_text(encoding="utf-8"))
        posts = [j for j in doc["jobs"] if "post" in j.get("cmd", [])]
        assert posts, name
        for j in posts:
            assert j["cmd"][-2:] == ["--gen", "ember"], (name, j["name"])
        assert TCR.EMBER_COMMIT[:12] in doc["_comment"], name
    pilot = json.loads((colab / "tiny_chat_pilot.json").read_text(encoding="utf-8"))
    assert all("--gen" not in j.get("cmd", []) for j in pilot["jobs"])  # it ran with mtstudio generation


# ---------------------------------------------------------------- end to end (POSIX, a real ember build)
e2e = pytest.mark.skipif(os.name == "nt" or BIN is None or not EMBER_DIR,
                         reason="needs POSIX, MTSTUDIO_BIN and TINY_CHAT_TEST_EMBER_DIR (an ember build)")


@e2e
@pytest.mark.parametrize("kind", ["llama", "recursive", "mor"])
def test_post_generates_with_ember_after_a_spot_check(built, tmp_path, kind):  # noqa: F811 (the fixture)
    t, out = built
    env = {**os.environ, "JOBS_MTSTUDIO": BIN, "TINY_CHAT_EMBER_DIR": EMBER_DIR, "TINY_CHAT_EMBER_APT": "0"}
    spec = json.loads(shape_spec(t, out).read_text())
    spec["data"]["T"] = 256  # Carry-5's conversations fit (the toy's 64 would drop most history)
    if kind == "recursive":
        spec["arch"]["custom"]["recursion"] = {"prelude": 1, "block": 1, "repeats": 3, "coda": 0, "schedule": "fixed",
                                               "inject": "none"}
    if kind == "mor":
        spec["arch"]["custom"].update(layers=3)
        spec["arch"]["custom"]["recursion"] = {"prelude": 1, "block": 1, "repeats": 3, "coda": 1, "schedule": "mor",
                                               "inject": "none"}
    spec["name"] = f"tc_{kind}"
    sp = tmp_path / f"shape_{kind}.json"
    sp.write_text(json.dumps(spec))
    pt = dict(spec, out_dir=str(tmp_path / "pt"))
    (tmp_path / "pt.json").write_text(json.dumps(pt))
    p = subprocess.run([BIN, "run", str(tmp_path / "pt.json")], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:]
    # Carry-5's first 60 items: the spot-check takes items 0, 26 and 52, the run all 60.
    lines = (ROOT / "corpus" / "eval" / "carry5_v1.jsonl").read_text(encoding="utf-8").splitlines()[:60]
    items = tmp_path / "items.jsonl"
    items.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sft = {"data": {"vocab": str(t / "v.gguf"), "vocab_cap": 300, "T": 256, "sft_examples": str(out / "sft_human.jsonl")},
           "train": {"objective": "sft", "steps": 4, "lr": 0.001, "batch": 32, "eval_every": 2, "eval_batch": 4,
                     "checkpoint_every": 2, "gradmap_every": 0, "seed": 1, "optimizer": "adamw"},
           "export": {"formats": ["safetensors", "gguf"]}}
    (tmp_path / "sft.json").write_text(json.dumps(sft))
    post = tmp_path / "post"
    argv = RUN + ["post", "--out", str(post), "--spec", str(sp), "--pt-weights", str(tmp_path / "pt" / f"tc_{kind}.safetensors"),
                  "--pt-dir", str(tmp_path / "pt"), "--sft-template", str(tmp_path / "sft.json"), "--seed", "1",
                  "--cell", kind, "--share", "50", "--carry5-items", str(items), "--gen", "ember"]
    r = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    s = json.loads((post / "summary.json").read_text())
    g = s["gen_engine"]
    assert g["ember_build"]["commit"] == TCR.EMBER_COMMIT, g
    assert g["spot"]["pass"] and g["spot"]["n"] == 3 and g["spot"]["replies_identical"] == 24, g["spot"]
    assert g["engine"] == "ember" and s["carry5"]["meta"]["gen"] == "ember" and "gen_flag" not in s
    assert g["spot"]["env"]["EMBER_GEMM_TF32"] == "0"
    assert g["family"] == {"llama": "llama", "recursive": "llama_recursive", "mor": "llama_mor"}[kind]
    if kind != "llama":
        assert g["ir_device"] == "cpu"  # a CPU build; MoR is always on ember's CPU path
    # ember returns ids, so mtstudio score checks every self-history reply of the whole run as well.
    # (A reply whose history was dropped for the context is not gated: carry5_eval leaves it out.)
    assert 250 <= g["full_gate"]["replies"] <= 300, g["full_gate"]
    assert g["full_gate"]["argmax_identical"] == g["full_gate"]["replies"], g["full_gate"]
    assert os.path.islink(post / "carry5" / "ember_cwd" / "debugging.log")  # ember's INFO log discarded
    assert s["primary"] is not None and len(open(post / "carry5" / "m" / "gen.jsonl").readlines()) == 60
    # A rerun is idempotent and keeps the decision.
    r = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0 and "present" in r.stdout
