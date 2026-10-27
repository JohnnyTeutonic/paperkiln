"""experiments/tiny_chat_long_2026-10/make_jobs.py: Carry-12's cells' specs and jobs files (PREREGISTRATION_DRAFT.md
sections 6 and 10). The committed files are the script's output byte for byte; the pretraining specs are the specs
Stage 0 probed at T 1,024 (arch, data, train, export); every rate is lr_choice.json's for its plain shape; the post
jobs carry Carry-12 after Carry-5 at ember ec3201b with EMBER_DETERMINISTIC=1; B's machine runs D (sft_human_long4,
B's weights) and W512 (E1's uploaded weights, E1's SFT, Carry-12 forced choice); the caps follow the measured total;
and every file's gate refuses until the frozen PREREGISTRATION.md exists."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LONG = ROOT / "experiments" / "tiny_chat_long_2026-10"
COLAB = ROOT / "tools" / "colab_chat"
sys.path.insert(0, str(COLAB))
import tiny_chat_run as tcr  # noqa: E402


def mk():
    spec = importlib.util.spec_from_file_location("c12_make_jobs", LONG / "make_jobs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def jobs(cell, seed):
    return json.loads((COLAB / f"carry12_{cell}_r{seed}.json").read_text(encoding="utf-8"))


def by_name(doc):
    return {j["name"]: j for j in doc["jobs"]}


def arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_files_are_the_scripts_output(tmp_path, monkeypatch):
    m = mk()
    assert m.lstar() == 8 and m.cells() == ["B", "Sgist", "P16", "X"]
    for cell in m.cells():
        assert json.loads((ROOT / m.cell_spec_path(cell)).read_text()) == m.cell_spec(cell)
    for path, doc in m.sft_templates().items():
        assert json.loads((ROOT / path).read_text()) == doc
    hs, _, _ = m.hours()
    for cell in m.cells():
        for seed in m.SEEDS:
            name, doc, _ = m.machine_doc(cell, seed, hs[cell])
            assert json.loads((COLAB / name).read_text()) == doc, name


def test_pretraining_specs_are_the_probed_specs_and_rates():
    choice = json.loads((ROOT / "experiments" / "tiny_chat_2026-10" / "stage0" / "lr_choice.json").read_text())["lr"]
    for cell in ("B", "Sgist", "P16", "X"):
        probe = json.loads((LONG / "specs" / f"probe60_{cell}_T1024.json").read_text())
        for seed in (1, 2, 3):
            pt = by_name(jobs(cell, seed))[f"pt_c12_{cell}_r{seed}"]
            spec = pt["spec"]
            assert pt["model_name"] == f"tc_c12_{cell}_r{seed}" and "snapshot_steps" not in pt  # no rungs
            assert spec["train"]["seed"] == seed
            for k in ("arch", "data", "export"):
                assert spec[k] == probe[k], (cell, k)
            assert {**spec["train"], "seed": 1} == probe["train"], cell
            sid = spec["name"][len("tc_"):]
            assert sid == ("60m_L8" if cell in ("B", "Sgist") else "60m_L16") and spec["train"]["lr"] == choice[sid]
            tr = spec["train"]
            assert tr["batch"] * tr["accum"] * spec["data"]["T"] == 16384 and spec["data"]["T"] == 1024
            assert spec["data"]["corpus"] == "/content/chat/data/mix_s35.ids" and tr["steps"] == 10000
    assert jobs("Sgist", 1)["jobs"][3]["spec"]["arch"]["state_tokens"] == {"k": 4, "mask": "turn_state"}
    assert jobs("X", 1)["jobs"][3]["spec"]["arch"]["custom"]["attn_res"] == {"form": "full", "scale": "sources"}


def test_post_jobs_carry12_after_carry5_at_the_pin():
    for cell in ("B", "Sgist", "P16", "X"):
        for seed in (1, 2, 3):
            doc = jobs(cell, seed)
            j = by_name(doc)
            assert doc["max_parallel"] == 1 and list(j)[:4] == ["cuda_gate", "data", "gate", f"pt_c12_{cell}_r{seed}"]
            assert "--T" in j["data"]["cmd"] and arg(j["data"]["cmd"], "--T") == "1024"
            assert "--carry12" in j["data"]["cmd"] and arg(j["data"]["cmd"], "--long-sft") == "10:4"
            posts = [cell] + (["D"] if cell == "B" else [])
            for c in posts:
                p = j[f"post_c12_{c}_r{seed}"]
                cmd = p["cmd"]
                assert p["after"] == p["init_job"] == f"pt_c12_{cell}_r{seed}"
                assert arg(cmd, "--ember-commit") == "ec3201b04e6f2fcb6dffd3496b0c72b3bec0c54a"
                assert "--ember-deterministic" in cmd and arg(cmd, "--gen") == "ember"
                assert arg(cmd, "--carry12-items") == "corpus/eval/carry12_v1.jsonl" and arg(cmd, "--carry12-spot") == "20"
                assert arg(cmd, "--carry5-items") == "corpus/eval/carry5_v1.jsonl" and arg(cmd, "--sft-batching") == "spec"
                assert arg(cmd, "--cell") == c and arg(cmd, "--seed") == str(seed) and arg(cmd, "--share") == "35"
                assert arg(cmd, "--pt-weights") == f"/content/chat/pt_c12_{cell}_r{seed}/tc_c12_{cell}_r{seed}.safetensors"
                sft = arg(cmd, "--sft-template")
                assert sft.endswith("sft_T1024_long4.json" if c == "D" else "sft_T1024.json")
                assert {"summary.json", "carry12/summary.json", "carry12/m/items.jsonl", "carry5/m/items.jsonl",
                        f"sft/tc_sft_{c}_r{seed}.gguf"} <= set(p["outputs"])
                shape = json.loads((ROOT / arg(cmd, "--spec")).read_text())
                t = json.loads((ROOT / sft).read_text())
                b, a = tcr.sft_batching("spec", shape, t["data"]["T"])
                assert b * a * t["data"]["T"] == 16384 and (b, a) == ((16, 1) if cell in ("B", "Sgist") else (8, 2))
            if cell == "B":
                w = j[f"post_c12_W512_r{seed}"]["cmd"]
                assert "--no-carry5" in w and arg(w, "--carry12-gen") == "none" and arg(w, "--cell") == "W512"
                assert arg(w, "--pt-weights") == f"/content/chat/upload/tc_e1_60m_L8_r{seed}.safetensors"
                assert arg(w, "--spec") == "experiments/tiny_chat_2026-10/specs/shape_60m_L8.json"
                assert "--sft-batching" not in w  # E1's own SFT batching (32 x 1 at L8)
                assert j["data512"]["light"] and arg(j["data512"]["cmd"], "--T") == "512"
                e1 = json.loads((ROOT / "experiments" / "tiny_chat_2026-10" / "specs" / "sft.json").read_text())
                w512 = json.loads((LONG / "specs" / "sft_w512.json").read_text())
                e1["data"]["sft_examples"] = "/content/chat/data512/sft_human.jsonl"
                assert {k: v for k, v in w512.items() if k != "_comment"} == \
                    {k: v for k, v in e1.items() if k != "_comment"}
                assert f"--upload tc_e1_60m_L8_r{seed}.safetensors=" in doc["_comment"]
            else:
                assert not any(n.startswith(("post_c12_D", "post_c12_W512", "data512")) for n in j)


def test_caps_and_not_launched():
    m = mk()
    hs, w512, _ = m.hours()
    total = json.loads((LONG / "stage0" / "total.json").read_text())
    assert abs(sum(3 * h for h in hs.values()) - 3 * w512 - total["core_vm_hours"]) < 0.5
    for cell, h in hs.items():
        doc = jobs(cell, 1)
        assert f"--max-usd {m.cap(h)} " in doc["_comment"] and "--commit FREEZE_COMMIT" in doc["_comment"]
        assert "NOT LAUNCHED" in doc["_comment"]
        assert m.cap(h) >= h * 0.59  # a cap under the estimate would stop the machine early
    drv = ROOT.parent / "scripts" / "colab_driver.json"  # the workspace's watchdog entries, when present
    if drv.exists():
        cells = [f"carry12_{c}_r{r}.json" for c in hs for r in (1, 2, 3)]
        assert not any(d.get("enabled") and any(f in d["command"] for f in cells)
                       for d in json.loads(drv.read_text())["drivers"])


def test_gate_refuses_until_the_frozen_file(tmp_path):
    g = by_name(jobs("X", 2))["gate"]["cmd"]
    assert arg(g, "--need-file") == "experiments/tiny_chat_long_2026-10/PREREGISTRATION.md"
    assert arg(g, "--need-cuda-gate") == "experiments/tiny_chat_long_2026-10/stage0/cuda_gate_c12_s0.json"
    cmd = [sys.executable] + g[1:]
    cmd[cmd.index("{out}")] = str(tmp_path)
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    got = json.loads((tmp_path / "gate.json").read_text())
    if (LONG / "PREREGISTRATION.md").exists():  # frozen: the gate passes
        assert p.returncode == 0 and got == {"ok": True, "problems": []}
    else:
        assert p.returncode == 1 and got["problems"] == [
            "experiments/tiny_chat_long_2026-10/PREREGISTRATION.md missing: this block launches only once that "
            "file is committed"]
