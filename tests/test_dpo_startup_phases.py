"""Actual tiny CPU DPO off/on equivalence; no CUDA timing or production admission."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import pytest
from test_mtstudio_chat import BIN, write_vocab_gguf
from test_mtstudio_sft import EOT, FACTS, WORDS, qa

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(BIN is None, reason="set MTSTUDIO_BIN")
def test_phase_telemetry_preserves_tiny_dpo_checkpoint_and_rng(tmp_path):
    write_vocab_gguf(tmp_path / "v.gguf", ["<unk>", EOT] + WORDS, eos_id=1)
    examples = [qa(a,b) for a,b in FACTS] * 3 + [qa(0,0)]
    (tmp_path / "sft.jsonl").write_text("".join(json.dumps(v)+"\n" for v in examples))
    pairs = [{"prompt": f"what is {a} plus {b} ?", "chosen": f"the answer is {a+b} .", "rejected": f"the answer is {(a+b+1)%20} ."} for a,b in (FACTS * 6)[:35]]
    (tmp_path / "pairs.jsonl").write_text("".join(json.dumps(v)+"\n" for v in pairs))
    base = {"name": "base", "arch": {"preset":"llama-tiny", "rope_heads":"all", "custom":{"d":16,"layers":1,"heads":2,"d_ff":32}}, "data":{"vocab":str(tmp_path/"v.gguf"),"vocab_cap":64,"T":16,"sft_examples":str(tmp_path/"sft.jsonl")}, "train":{"objective":"sft","steps":1,"eval_every":1,"checkpoint_every":1,"seed":3,"lr":.001,"batch":2,"gradmap_every":0}, "export":{"formats":["safetensors"]},"out_dir":str(tmp_path/"base")}
    env = {k:v for k,v in os.environ.items() if k not in ["MICROTORCH_DPO_PHASES","MICROTORCH_DPO_CHECKPOINT_PUBLISH_V1","DPO_NUMERICAL_SOURCE_COMMIT"]}
    env.update(MICROTORCH_DEVICE="cpu",MICROTORCH_DEVICE_OPS="0",MICROTORCH_STEP_RESIDENCY="0",MICROTORCH_DEFER_DOWNLOADS="0",OMP_NUM_THREADS="1")
    def run(spec, extra):
        path=tmp_path/(Path(spec["out_dir"]).name+".json");path.write_text(json.dumps(spec))
        done=subprocess.run([BIN,"run",str(path)],env=env|extra,capture_output=True,text=True,timeout=120)
        assert done.returncode==0,done.stderr
        return done
    base_done=run(base,{"MICROTORCH_DPO_PHASES":"1"})
    assert "DPO_STARTUP_PHASE " not in base_done.stderr
    # Test-only hook uses the production full-state publisher, so RNG output
    # means its actual saved deterministic-reconstruction record, not a new
    # claim to serialize the engine's internal state bytes.
    hook=tmp_path/"fixture_hook.py"
    hook.write_text("import argparse,sys\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(ROOT/"tools/colab_chat"))+")\nimport dpo_checkpoint_core_v1 as cp\np=argparse.ArgumentParser();p.add_argument('action');\nfor x in ['out','step','seed','batch','accum','source','spec']:p.add_argument('--'+x)\na=p.parse_args()\nif a.action=='publish': cp.publish(a.out,int(a.step),int(a.seed),int(a.batch),int(a.accum),a.source,'b'*64)\n")
    outputs={};logs={}
    for mode in ("unset","off","on"):
        spec=json.loads(json.dumps(base));spec["name"]="phase_po";spec["out_dir"]=str(tmp_path/mode)
        spec["data"].pop("sft_examples");spec["data"]["pref_pairs"]=str(tmp_path/"pairs.jsonl")
        spec["train"].update(objective="provenance_po",steps=2,eval_every=2,init_from=str(tmp_path/"base/base.safetensors"),po={"beta":.1,"weight_mode":"uniform","nll":.2})
        extra={"MICROTORCH_DPO_CHECKPOINT_PUBLISH_V1":str(hook),"DPO_NUMERICAL_SOURCE_COMMIT":"a"*40}
        if mode!="unset": extra["MICROTORCH_DPO_PHASES"]="1" if mode=="on" else "0"
        done=run(spec,extra)
        assert "DPO_STARTUP_PHASE " not in done.stdout
        rows=[json.loads(line.split("DPO_STARTUP_PHASE ",1)[1]) for line in done.stderr.splitlines() if line.startswith("DPO_STARTUP_PHASE ")]
        if mode!="on": assert rows==[]
        else:
            labels=[r["phase"] for r in rows]
            assert labels[:3]==["reference_model_load_begin","reference_model_load_end","reference_cache_begin"]
            assert labels[-3:]==["reference_cache_end","first_training_step_begin","first_training_step_end"]
            progress=[r["pairs_completed"] for r in rows if r["phase"]=="reference_cache_progress"]
            assert progress==[1,32,35]
            assert all(r["diagnostic_only"] is True and r["pairs_total"]==35 for r in rows)
            assert [r["elapsed_seconds"] for r in rows]==sorted(r["elapsed_seconds"] for r in rows)
        root=Path(spec["out_dir"])
        (root/"cli_stdout.txt").write_text(done.stdout)
        (root/"cli_stderr.txt").write_text(done.stderr)
        outputs[mode]={n:(root/n).read_bytes() for n in ["model.safetensors","optim.safetensors","state.txt","phase_po.safetensors"]}
        manifest=json.loads((root/".dpo_checkpoint_v1/CURRENT/manifest.json").read_text())
        outputs[mode]["rng"]=manifest["rng"]
        logs[mode] = [json.loads(x) for x in (root/"events.jsonl").read_text().splitlines()]
        assert [r["step"] for r in logs[mode] if r["event"]=="step"]==[1,2]
        assert not any(r.get("event")=="dpo_phase" for r in logs[mode])
    assert outputs["unset"]==outputs["off"]==outputs["on"]
    assert outputs["on"]["rng"]["draws"]==4
    for mode in ("unset","off","on"):
        logs[mode]=[r for r in logs[mode] if r["event"]=="step"]
    assert logs["unset"]==logs["off"]==logs["on"]

    report={"actual_cuda":False,"steps":2,"seed":3,"batch":2,"model_optimizer_state_export_byte_identical":True,"rng_reconstruction_record":outputs["on"]["rng"],"phase_event_stream_unchanged":True,"numerical_step_events_identical":True,"file_sha256":{n:hashlib.sha256(b).hexdigest() for n,b in outputs["on"].items() if isinstance(b,bytes)}}
    (tmp_path/"phase_equivalence.json").write_text(json.dumps(report,indent=2)+"\n")
