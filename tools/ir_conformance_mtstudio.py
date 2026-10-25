"""paperkiln's side of the keramos conformance suite (spec/IR_SPEC.md section 11; keramos is vendored in spec/):
mtstudio's adapter for the runner, and the training of the suite's pk_ models.

    MTSTUDIO_BIN=build/mtstudio python tools/ir_conformance_mtstudio.py [--only ID] [--json OUT]
    MTSTUDIO_BIN=build/mtstudio python tools/ir_conformance_mtstudio.py --build-models OUT [--only NAME]

The first runs spec/tools/ir_conformance.py with --adapter mtstudio (other runner options pass through).
--build-models trains every pk_ model of the suite with mtstudio (tiny word or BPE vocabularies, d 16,
30-40 steps, the gate tests' recipes) into OUT/<name>/ (<name>.gguf with the IR, <name>.safetensors and
mtstudio.json, the run's spec). They go into keramos's conformance/models/, which then regenerates its
expected values (python tools/ir_conformance.py --regenerate there).
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "tools"))
from ir_conformance import Adapter, _decode_tensors, _encode_tensors, enc  # noqa: E402
import ir_conformance  # noqa: E402

# ---------------------------------------------------------------- the pk_ models' recipes

D, H = 16, 2



def _recipes():
    llama = {"preset": "llama-tiny", "custom": {"d": D, "layers": 2, "heads": H}}
    gpt2 = {"preset": "gpt2-nano", "custom": {"d": D, "layers": 2, "heads": H}}

    def with_custom(base, **kw):
        a = copy.deepcopy(base)
        a["custom"].update(kw)
        return a

    r = {
        "pk_llama": (llama, "core"),
        "pk_moe": (dict(copy.deepcopy(llama), moe={"experts": 4, "top_k": 2, "aux_loss": 0.01, "z_loss": 0.001}), "core"),
        "pk_gpt2": (gpt2, "core"),
        "pk_gpt2_swa": (with_custom(gpt2, attention="swa", window=4, sinks=2), "core"),
        "pk_kimi": (with_custom(gpt2, attention="kimi"), "state"),
        "pk_srd": (with_custom(gpt2, attention="srd"), "state"),
        "pk_attnres": (with_custom(gpt2, attention="attnres"), "core"),
        "pk_flex_highway": (with_custom(gpt2, activation="relu", residual="highway", attention="swa", window=4, sinks=2),
                            "core"),
        "pk_flex_expr": (with_custom(gpt2, norm="expr:(x - rowmean(x)) / sqrt(rowmean((x - rowmean(x))^2) + 1e-5)",
                                     activation="expr:relu(x)^2", position="rope"), "core"),
        "pk_mamba": ({"preset": "mamba-tiny", "custom": {"d": D, "layers": 2, "heads": H, "d_state": 8}}, "state"),
        "pk_recursive": ({"preset": "llama-tiny", "custom": {"d": D, "heads": H, "recursion": {
            "prelude": 1, "block": 1, "repeats": 2, "coda": 1, "inject": "add"}}}, "recursion"),
        "pk_mor": ({"preset": "llama-tiny", "custom": {"d": D, "heads": H, "recursion": {
            "prelude": 1, "block": 1, "repeats": 3, "coda": 1, "schedule": "mor", "inject": "none"}}}, "recursion"),
        "pk_llama_attnres": (with_custom(llama, attn_res={"form": "full"}), "core"),
        # llama_attnres under recursion: each application's own pseudo-queries, "tensor_index": "application".
        "pk_llama_attnres_rec": ({"preset": "llama-tiny", "custom": {"d": D, "heads": H, "attn_res": {"form": "full"},
                                  "recursion": {"prelude": 1, "block": 1, "repeats": 2, "coda": 1,
                                                "inject": "none"}}}, "recursion"),
        "pk_state": (dict(copy.deepcopy(llama), state_tokens={"k": 2, "mask": "turn_state"}), "state"),
        # The recurrence lane (IR_SPEC 6.10): kimi's projections around a recurrence node whose body is a
        # decayed elu+1 linear attention (no hand-written op computes it).
        "pk_recurrence": (with_custom(gpt2, attention="recurrence", recurrence=DECAYED_LINEAR_ATTENTION), "recurrence"),
    }
    return r


# The pk_recurrence body: section 6.4's linear attention with the running sums decayed by 0.9375 per token.
_PHI = "where({x}, {x} + 1, (exp(min(0, {x})) - 1) + 1)"
DECAYED_LINEAR_ATTENTION = {
    "index": {"h": H, "e": D // H}, "inputs": ["q[h,e]", "k[h,e]", "v[h,e]"],
    "let": ["pk[h,e] = " + _PHI.format(x="k[h,e]")],
    "update": ["num[h,e] = 0.9375 * num[h,e] + pk[h,e] * v[h,e]", "den[h,e] = 0.9375 * den[h,e] + pk[h,e]"],
    "readout": "y[h,e] = (" + _PHI.format(x="q[h,e]") + " * num[h,e]) / max(den[h,e], 1e-08)"}


# ---------------------------------------------------------------- mtstudio's adapter (section 11.3)

class MtstudioAdapter(Adapter):
    """paperkiln's C++ engine. It runs its own exports, not documents: model cases and forward-only
    traces of llama-family models go through `mtstudio score` (--kv-cache, --kv-step 1, --evict); op
    cases of expr through expr_eval; recurrence op and gradient cases and the digest vectors through
    ir_eval. Everything else is SKIP."""
    name, profiles = "mtstudio", {"core", "state", "recursion", "chat", "recurrence"}
    score_only = True  # it runs its own exports, not documents: no open/forward fallback

    def __init__(self, binary=None):
        self.bin = binary or os.environ.get("MTSTUDIO_BIN") or str(ROOT / "build" / "mtstudio")
        if not Path(self.bin).is_file():
            raise SystemExit(f"mtstudio adapter: no binary at {self.bin} (set MTSTUDIO_BIN)")
        ee = os.environ.get("EXPR_EVAL_BIN") or str(Path(self.bin).parent / "expr_eval")
        self.expr_eval = ee if Path(ee).is_file() else None
        ie = os.environ.get("IR_EVAL_BIN") or str(Path(self.bin).parent / "ir_eval")
        self.ir_eval = ie if Path(ie).is_file() else None

    def _ir_eval(self, req):
        if self.ir_eval is None:
            raise NotImplementedError("no ir_eval binary beside mtstudio (set IR_EVAL_BIN)")
        r = subprocess.run([self.ir_eval], input=json.dumps([req]), capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-500:])
        got = json.loads(r.stdout)[0]
        if "skip" in got:
            raise NotImplementedError(got["skip"])
        if "error" in got:
            raise RuntimeError(got["error"])
        return _decode_tensors(got)

    def digest(self, text, prose=False):
        return self._ir_eval({"method": "digest", "text": text, "prose": prose})

    def run_node_grad(self, node, inputs, weights, out_grad):
        if node["op"] != "recurrence":
            raise NotImplementedError("ir_eval differentiates recurrence nodes")
        return self._ir_eval({"method": "run_node_grad", "node": node, "inputs": _encode_tensors(inputs),
                              "weights": _encode_tensors(weights), "out_grad": enc(out_grad)})["grads"]

    def score(self, model_dir, prompt, cont, kv_cache=False, evict=False, kv_step=0):
        d = Path(model_dir).resolve()
        mt = d / "mtstudio.json"
        if not mt.is_file():
            raise NotImplementedError("no mtstudio run record (not a paperkiln-trained model)")
        spec = json.loads(mt.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="irconf_") as tmp:
            spec["out_dir"] = str(d)
            spec.setdefault("data", {})["vocab"] = str(d / (d.name + ".gguf"))
            spec["name"] = d.name
            sp = Path(tmp) / "spec.json"
            sp.write_text(json.dumps(spec), encoding="utf-8")
            req, res = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
            req.write_text(json.dumps({"prompt_ids": list(prompt), "continuation_ids": list(cont)}) + "\n",
                           encoding="utf-8")
            cmd = [self.bin, "score", str(sp), "--in", str(req), "--out", str(res)]
            if kv_cache:
                cmd.append("--kv-cache")
                if evict:
                    cmd.append("--evict")
                if kv_step:
                    cmd += ["--kv-step", str(kv_step)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                raise RuntimeError(r.stderr.strip()[-500:])
            g = json.loads(res.read_text().splitlines()[0])
        out = {"logprobs": g["logprobs"], "argmax": g["argmax"]}
        if "depth" in g:
            out["depth"] = g["depth"]
        if "kv_rows" in g:
            out["kv_rows"] = g["kv_rows"]
        return out

    def run_node(self, node, inputs, weights, state):
        if node["op"] == "recurrence":
            got = self._ir_eval({"method": "run_node", "node": node, "inputs": _encode_tensors(inputs),
                                 "weights": _encode_tensors(weights)})
            return {"outputs": got["outputs"], "state": got["state"]}
        if node["op"] != "expr" or self.expr_eval is None:
            raise NotImplementedError("mtstudio runs whole models; expr_eval runs expr nodes, ir_eval recurrences")
        xs = [inputs[n] for n in node["in"]]
        case = {"expr": node["expr"], "inputs": [x.astype(np.float64).tolist() for x in xs]}
        r = subprocess.run([self.expr_eval], input=json.dumps([case]), capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-500:])
        got = json.loads(r.stdout)[0]
        if "error" in got:
            raise RuntimeError(got["error"])
        val = np.array([[float(v) if not isinstance(v, str) else float(v.replace("inf", "inf")) for v in row]
                        for row in got["out"]], dtype=np.float32)
        return {"outputs": {node["out"]: val}, "state": {}}


# ---------------------------------------------------------------- training the pk_ models

def build_models(out, only=None):
    sys.path.insert(0, str(ROOT / "tests"))
    from test_mtstudio_chat import BIN, write_vocab_gguf
    from test_mtstudio_moe import WORDS, corpus
    import bpe
    import test_state_tokens_ir as sti
    if BIN is None:
        raise SystemExit("--build-models needs MTSTUDIO_BIN")
    out = Path(out)
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    write_vocab_gguf(work / "v.gguf", ["<unk>"] + WORDS)
    (work / "c.txt").write_text(corpus(), encoding="utf-8")
    text = sti.corpus()
    bpe.write_gguf(str(work / "vs.gguf"), bpe.train(text, 400))
    (work / "cs.txt").write_text(text, encoding="utf-8")
    for name, (arch, _) in _recipes().items():
        if only and only not in name:
            continue
        state = "state_tokens" in arch
        steps = 40 if name == "pk_mor" else 30
        spec = {"name": name, "arch": arch,
                "data": {"vocab": str(work / ("vs.gguf" if state else "v.gguf")),
                         "corpus": str(work / ("cs.txt" if state else "c.txt")), "T": 64 if state else 16,
                         "max_tokens": 0},
                "train": {"steps": steps, "lr": 0.003, "eval_every": 1000, "checkpoint_every": 1000, "batch": 2,
                          "seed": 3, "gradmap_every": 0},
                "export": {"formats": ["safetensors", "gguf"]}, "out_dir": str(work / name)}
        sp = work / f"{name}.json"
        sp.write_text(json.dumps(spec), encoding="utf-8")
        r = subprocess.run([BIN, "run", str(sp)], capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise SystemExit(f"{name}: mtstudio run failed\n{r.stderr[-2000:]}")
        ev = [json.loads(x) for x in (work / name / "events.jsonl").read_text().splitlines() if x.strip()]
        ex = {e["format"]: e["path"] for e in ev if e["event"] == "export"}
        dst = out / name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ex["gguf"], dst / f"{name}.gguf")
        shutil.copyfile(ex["safetensors"], dst / f"{name}.safetensors")
        rec = {k: v for k, v in spec.items() if k not in ("out_dir",)}
        rec["data"] = {k: v for k, v in spec["data"].items() if k not in ("vocab", "corpus")}
        if arch["preset"] == "llama-tiny":
            rec["arch"] = dict(rec["arch"], rope_heads="all")
        (dst / "mtstudio.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        print(f"built {name}: {os.path.getsize(dst / (name + '.gguf'))} + "
              f"{os.path.getsize(dst / (name + '.safetensors'))} bytes")
    shutil.rmtree(work)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--build-models" in argv:
        i = argv.index("--build-models")
        out = argv[i + 1]
        only = argv[argv.index("--only") + 1] if "--only" in argv else None
        build_models(Path(out), only)
        return 0
    if "--adapter" not in argv:
        argv = ["--adapter", "mtstudio"] + argv
    return ir_conformance.main(argv, adapters={"mtstudio": MtstudioAdapter})


if __name__ == "__main__":
    sys.exit(main())
