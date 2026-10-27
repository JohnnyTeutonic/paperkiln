#!/usr/bin/env python3
"""ONNX Runtime GenAI (CPU) against the keramos conformance suite: an in-process adapter (IR_SPEC 11.3).

    python adapters/ortgenai/ortgenai_adapter.py --adapter ortgenai --json OUT.json [--only tp_]
           [--ort-patch none|unfused_rope,no_gqa,gemma3_config] [--ort-loader export|gguf] [--ort-dir DIR]

Run from the keramos root with the pinned environment (requirements.txt beside this file): CPU execution
provider only, float32, 2 intra-op threads. ONNX Runtime GenAI never reads the IR document. Each GGUF is
exported to an HF checkpoint by adapters/common/gguf_to_hf.py, and the builder (build_onnx.py, which runs
onnxruntime_genai.models.builder with -p fp32 -e cpu) makes model.onnx and genai_config.json from it, as a
user of a converted checkpoint would; --ort-loader gguf gives the builder the GGUF itself (its own weight
loader) with the export's config. Both are cached under --ort-dir by the GGUF's sha256 and the settings.
--ort-patch names builder patches for diagnosis (build_onnx.py lists them); the receipt uses none. Where the
file declares no eos id (the tp_ files), the build sets the configuration's eos_token_id to the file's unknown
id, since the builder stops without one (build_onnx.py --eos-fallback); no suite case acts on it.

Methods (FINDINGS.txt has the reasoning):
  open      export + build (cached); og.Model with the CPU provider. Skip for architecture "paperkiln" (IR_SPEC
            9.1), for a file whose own document carries semantics its architecture cannot (adapters/hf's
            rule), for a replacement document (ir), for evict true, where the builder declines the
            architecture (its NotImplementedError, by name), and where onnxruntime's CPU kernel refuses the
            built graph's shape (a one-token forward at load fails with the kernel's head_size check; its
            message is the skip reason). Any other build or run error is an error (FAIL): the builder
            stopping with an exception on an architecture it lists (Gemma 3's KeyError) is one.
  forward   Generator.append_tokens(ids) (the first call is the prefill, later ones extend the sequence);
            logits from Generator.get_output("logits"), the decoder's [1, n, vocab] output for the rows this
            call ran (the builder does not prune the LM head by default). kv_rows: the generator's sequence
            length per layer (its past/present buffers are max_length or window sized, shared in place, so
            the length is what says which rows are held); kv_positions 0 .. len - 1; recurrent {}; depth [].
  truncate  Generator.rewind_to(n); n = 0, or a rewind the generator refuses, re-prefills (IR_SPEC 6.9).
            The result's "_how" says which.
  snapshot, restore  the ids at the snapshot; restore rewinds (rewind_to) when the session's ids still
            begin with them, else re-prefills them. "_how" says which.
  An error leaves the session as it was: the generator is rebuilt and the session's ids re-prefilled.
  check, run_node, run_node_grad, digest, render_chat, dequantize: skip (no IR validation, no single
            nodes, no digest, no IR template records; GGUF dequantisation is gguf-py's).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "adapters" / "common"))
sys.path.insert(0, str(ROOT / "adapters" / "hf"))
import ir_conformance as rc  # noqa: E402
import ir_validate  # noqa: E402
import gguf_to_hf as g2h  # noqa: E402
from hf_adapter import ir_only_reason  # noqa: E402  (the same rule for what an HF config can carry)

THREADS = 2
# onnxruntime's CPU GroupQueryAttention preconditions (probe.py's kernel section: head_size % 8 always,
# head_size % 16 with do_rotary): the kernel's own messages, matched to tell a refused shape from any other failure.
KERNEL_REFUSALS = ("head_size must be a multiple of 8", "head_size shall be a multiple of 16")


class Refused(NotImplementedError):
    pass


class ORTGenAIAdapter(rc.Adapter):
    name, backend, exact = "onnxruntime-genai", "cpu", False
    profiles = {"core", "thirdparty", "state"}

    def __init__(self, patch="none", loader="export", ort_dir=None):
        import onnxruntime as ort
        import onnxruntime_genai as og
        self.og, self.ort = og, ort
        self.patch, self.loader = patch, loader
        self.dir = Path(ort_dir or os.path.expanduser("~/ort_keramos"))
        self.models, self.sessions, self.n = {}, {}, 0
        self.load_notes = {}

    def hello(self):
        import transformers
        import torch
        return {"engine": self.name,
                "version": f"onnxruntime-genai {self.og.__version__}, onnxruntime {self.ort.__version__}; builder under "
                           f"transformers {transformers.__version__}, torch {torch.__version__}",
                "backend": self.backend, "exact": self.exact, "profiles": sorted(self.profiles),
                "settings": {"precision": "fp32", "execution_provider": "cpu", "intra_op_num_threads": THREADS,
                             "builder_patch": self.patch, "loader": self.loader,
                             "exporter": f"adapters/common/gguf_to_hf.py v{g2h.EXPORTER_VERSION}"},
                "methods": ["hello", "open", "forward", "truncate", "snapshot", "restore", "close"]}

    # ------------------------------------------------------------ building and loading

    def build_dir(self, gguf):
        """(export dir, onnx dir, sha256) for a GGUF under this adapter's settings."""
        sha = hashlib.sha256(Path(gguf).read_bytes()).hexdigest()
        stem = f"{Path(gguf).stem}-{sha[:12]}"
        tag = f"{self.loader}-{self.patch.replace(',', '+')}"
        return self.dir / "export" / stem, self.dir / "onnx" / f"{stem}-{tag}", sha

    def build(self, gguf):
        """Export and build (cached). Returns the onnx dir; raises Refused or RuntimeError."""
        ex, out, _ = self.build_dir(gguf)
        if not (ex / "config.json").is_file():
            g2h.export(gguf, ex)
        status = out / "keramos_build.json"
        if status.is_file():
            st = json.loads(status.read_text(encoding="utf-8"))
        else:
            src = Path(gguf) if self.loader == "gguf" else ex
            meta_raw, *_ = ir_validate.read_gguf(gguf)
            unk = meta_raw.get("tokenizer.ggml.unknown_token_id", (None, 0))[1]
            cmd = [sys.executable, str(HERE / "build_onnx.py"), "--input", str(src), "--config", str(ex),
                   "--output", str(out), "--cache", str(self.dir / "cache"), "--patch", self.patch,
                   "--eos-fallback", str(int(unk))]
            env = dict(os.environ, OMP_NUM_THREADS=str(THREADS), MKL_NUM_THREADS=str(THREADS), HF_HUB_OFFLINE="1",
                       CUDA_VISIBLE_DEVICES="")
            r = subprocess.run(cmd, capture_output=True, text=True, env=env)
            out.mkdir(parents=True, exist_ok=True)
            (out / "keramos_build.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
            last = (r.stderr.strip().splitlines() or [""])[-1]
            st = {"returncode": r.returncode, "command": cmd[1:], "last": last}
            status.write_text(json.dumps(st, indent=1) + "\n", encoding="utf-8")
        cls = json.loads((ex / "config.json").read_text(encoding="utf-8"))["architectures"][0]
        why = st["last"].split(": ", 1)[-1].replace(str(Path(gguf)), "<gguf>").replace(str(ex), "<export>")
        if st["returncode"] == 3:
            raise Refused(f"the builder declines {cls} ({self.loader} input): {why}")
        if st["returncode"] != 0:
            raise RuntimeError(f"the builder fails on {cls} ({self.loader} input): {why}")
        return out

    def _model(self, gguf):
        ex, out, sha = self.build_dir(gguf)
        if sha in self.models:
            return self.models[sha]
        meta_raw, *_ = ir_validate.read_gguf(gguf)
        meta = {k: v[1] for k, v in meta_raw.items()}
        arch = meta.get("general.architecture")
        if arch not in g2h.FAMILIES:
            raise NotImplementedError(f"general.architecture {arch!r}: no HF checkpoint for the builder (IR_SPEC 9.1: "
                                      f"a file no llama.cpp-architecture graph computes)")
        if "paperkiln.ir" in meta:
            why = ir_only_reason(json.loads(meta["paperkiln.ir"]), meta)
            if why:
                raise NotImplementedError(f"IR-only semantics: {why}; the builder would compute plain {arch} "
                                          f"(llama.cpp FINDINGS 7, K1)")
        entry = self.load(self.build(gguf))
        entry["arch"] = arch
        self.models[sha] = entry
        self.load_notes[Path(gguf).stem] = entry["notes"]
        return entry

    def load(self, out):
        """og.Model of a built directory (CPU provider, 2 threads), with the kernel check below."""
        out = Path(out)
        gcfg = json.loads((out / "genai_config.json").read_text(encoding="utf-8"))
        cfg = self.og.Config(str(out))
        cfg.clear_providers()  # the CPU provider: no other is appended
        cfg.overlay(json.dumps({"model": {"decoder": {"session_options": {
            "intra_op_num_threads": THREADS, "inter_op_num_threads": 1}}}}))
        model = self.og.Model(cfg)
        entry = {"model": model, "ctx": int(gcfg["model"]["context_length"]),
                 "vocab": int(gcfg["model"]["vocab_size"]), "layers": int(gcfg["model"]["decoder"]["num_hidden_layers"]),
                 "dir": out, "genai_config": gcfg}
        # The kernel's shape preconditions, by the engine itself: one token through a throwaway generator.
        try:
            g = self._generator(entry)
            g.append_tokens([1])
            del g
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            hit = next((k for k in KERNEL_REFUSALS if k in msg), None)
            if hit:
                why = msg.split("Status Message: ")[-1].strip()
                entry["refused"] = (f"onnxruntime's CPU kernel refuses the built graph: {why} "
                                    f"(head_size {gcfg['model']['decoder']['head_size']})")
            else:
                raise RuntimeError(f"onnxruntime-genai: one-token forward at load: {type(e).__name__}: {msg}") from e
        notes = {"onnx_dir": str(out), "builder_patch": self.patch, "loader": self.loader,
                 "attention_op": _attention_ops(out / gcfg["model"]["decoder"]["filename"]),
                 "sliding_window": gcfg["model"]["decoder"].get("sliding_window"), "refused": entry.get("refused")}
        entry["notes"] = notes
        return entry

    def session_of(self, entry):
        """A session dict over a loaded entry (for probe.py: built directories outside open's cache)."""
        return {"m": entry, "ids": [], "g": None, "snap": {}}

    def _generator(self, m):
        p = self.og.GeneratorParams(m["model"])
        p.set_search_options(max_length=m["ctx"], do_sample=False)
        return self.og.Generator(m["model"], p)

    # ------------------------------------------------------------ sessions

    def open(self, gguf, ir=None, evict=None):
        if ir is not None:
            raise NotImplementedError("a replacement document: the builder runs the HF config, never a document")
        if evict is True:
            raise NotImplementedError("evict true: onnxruntime-genai has no state tokens or turn_state eviction")
        m = self._model(gguf)
        if m.get("refused"):
            raise NotImplementedError(m["refused"])
        self.n += 1
        sid = f"s{self.n}"
        self.sessions[sid] = {"m": m, "ids": [], "g": None, "snap": {}}
        return {"session": sid, "vocab": m["vocab"], "applications": m["layers"], "state_k": 0, "_notes": m["notes"]}

    def close(self, session):
        self.sessions.pop(session, None)

    def _append(self, s, ids):
        """Append ids to the session's generator; the logits [n, vocab] of those rows."""
        if s["g"] is None:
            s["g"] = self._generator(s["m"])
        s["g"].append_tokens(list(ids))
        lg = np.asarray(s["g"].get_output("logits"), dtype=np.float32)
        lg = lg.reshape(-1, lg.shape[-1])
        if lg.shape[0] != len(ids):
            raise RuntimeError(f"logits output has {lg.shape[0]} rows for {len(ids)} ids (pruned LM head?)")
        return lg

    def _reprefill(self, s, ids):
        s["g"] = None
        if ids:
            self._append(s, ids)
        s["ids"] = list(ids)

    def _report(self, s, report, logits=None):
        L = len(s["ids"])
        res = {"len": L}
        if "logits" in report:
            res["logits"] = logits
        if "depth" in report:
            res["depth"] = []
        held = 0 if s["g"] is None else len(s["g"].get_sequence(0))
        if "kv_rows" in report:
            res["kv_rows"] = {"kv": [held] * s["m"]["layers"]}
        if "kv_positions" in report:
            res["kv_positions"] = {"kv": [list(range(held))] * s["m"]["layers"]}
        if "recurrent" in report:
            res["recurrent"] = {}
        return res

    def forward(self, session, ids, report):
        s = self.sessions[session]
        if not ids:
            raise RuntimeError("forward: ids is empty (IR_SPEC 11.3)")
        before = list(s["ids"])
        try:
            logits = self._append(s, ids)
        except Exception as e:
            self._reprefill(s, before)
            raise RuntimeError(f"onnxruntime-genai append_tokens: {type(e).__name__}: {e}") from e
        s["ids"] = before + list(ids)
        return self._report(s, report, logits)

    def _rewind(self, s, n):
        """Bring the session to its first n ids: rewind_to where the generator allows it, else re-prefill."""
        L = len(s["ids"])
        if n == L:
            return "nothing to remove"
        if n == 0 or s["g"] is None:
            self._reprefill(s, s["ids"][:n])
            return "cleared" if n == 0 else f"re-prefill of {n} ids"
        try:
            s["g"].rewind_to(n)
            s["ids"] = s["ids"][:n]
            return f"Generator.rewind_to({n})"
        except Exception as e:  # noqa: BLE001
            self._reprefill(s, s["ids"][:n])
            return f"re-prefill of {n} ids (rewind_to refused: {str(e)[:90]})"

    def truncate(self, session, n, report):
        s = self.sessions[session]
        L = len(s["ids"])
        if not 0 <= n <= L:
            raise RuntimeError(f"truncate to {n}, outside 0 .. len {L} (IR_SPEC 11.3)")
        how = self._rewind(s, n)
        res = self._report(s, [r for r in report if r != "logits"])
        res["_how"] = how
        return res

    def snapshot(self, session, tag):
        s = self.sessions[session]
        s["snap"][tag] = list(s["ids"])
        return {"len": len(s["ids"])}

    def restore(self, session, tag, report):
        s = self.sessions[session]
        if tag not in s["snap"]:
            raise RuntimeError(f"restore: no snapshot {tag!r} in this session")
        want = s["snap"][tag]
        if s["ids"][:len(want)] == want and len(want) > 0:
            how = self._rewind(s, len(want))
        else:
            self._reprefill(s, want)
            how = f"re-prefill of {len(want)} ids"
        res = self._report(s, [r for r in report if r != "logits"])
        res["_how"] = how
        return res

    # ------------------------------------------------------------ what onnxruntime-genai has no path for

    def check(self, gguf, ir):
        raise NotImplementedError("onnxruntime-genai has no IR validation (it never reads the document)")

    def run_node(self, node, inputs, weights, state):
        raise NotImplementedError("onnxruntime-genai runs whole models, not single IR nodes")

    def run_node_grad(self, node, inputs, weights, out_grad):
        raise NotImplementedError("onnxruntime-genai runs whole models, not single IR nodes")

    def digest(self, text, prose=False):
        raise NotImplementedError("onnxruntime-genai has no IR digest")

    def render_chat(self, record, messages, generation_prompt=False, thinking=None):
        raise NotImplementedError("onnxruntime-genai renders the tokenizer's Jinja template, not IR template records")

    def dequantize(self, type_name, n, raw):
        raise NotImplementedError("onnxruntime-genai reads ONNX initialisers; GGUF dequantisation is gguf-py's")


def _attention_ops(onnx_path):
    """{op_type: count} of the attention nodes the builder emitted, with GQA's do_rotary."""
    import onnx
    m = onnx.load(str(onnx_path), load_external_data=False)
    out = {}
    for n in m.graph.node:
        if n.op_type in ("GroupQueryAttention", "Attention", "MultiHeadAttention", "RotaryEmbedding"):
            key = n.op_type
            if n.op_type == "GroupQueryAttention":
                at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
                key += f"(do_rotary={at.get('do_rotary', 0)}, local_window_size={at.get('local_window_size', -1)})"
            out[key] = out.get(key, 0) + 1
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    opts = {"patch": "none", "loader": "export", "ort_dir": None}
    for flag, key in (("--ort-patch", "patch"), ("--ort-loader", "loader"), ("--ort-dir", "ort_dir")):
        if flag in argv:
            i = argv.index(flag)
            opts[key] = argv[i + 1]
            del argv[i:i + 2]
    return rc.main(argv, adapters={"ortgenai": lambda: ORTGenAIAdapter(**opts)})


if __name__ == "__main__":
    sys.exit(main())
