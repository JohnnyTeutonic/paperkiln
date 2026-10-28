#!/usr/bin/env python3
"""A real checkpoint's GGUF header, without its weights: llama.cpp b6000's converter in its remote mode (config and
tokenizer JSON downloaded, every tensor's name, dtype and shape read from the safetensors headers by HTTP range
requests, nothing else), its metadata and tensor infos as it would write them, keramos's gguf_ir document built
from them and embedded (paperkiln.ir, its version and digest), and the header alone written (no tensor data).
Then keramos's ir_check.py checks the file: the document rules, every tensor the document names against the
file's tensor infos and shapes, rule 38 (the embedded document is the graph the builder makes from the file).

    python3 tools/thirdparty_header_check.py HF_ID OUT_DIR [KERAMOS_TOOLS (default spec/tools)]

For checkpoints too large for the laptop (LOCAL_LLM_STACK.md section 5.3: DeepSeek-V2-Lite, the DeepSeek-V3
layout): the builder and the validator on the file llama.cpp would write, with no weight downloaded. Writes
OUT_DIR/<id>.header.gguf (metadata and tensor infos, no data), <id>.ir.json and <id>.summary.json. Needs
huggingface_hub and llama.cpp b6000 at ~/llamacpp_b6000.
"""
import json
import os
from pathlib import Path
import sys

model_id, out_dir = sys.argv[1], Path(sys.argv[2])
ktools = sys.argv[3] if len(sys.argv) > 3 else str(Path(__file__).resolve().parent.parent / "spec" / "tools")
LLAMACPP = os.path.expanduser("~/llamacpp_b6000")
sys.path.insert(0, LLAMACPP)
sys.path.insert(0, os.path.join(LLAMACPP, "gguf-py"))
sys.path.insert(0, ktools)
import gguf  # noqa: E402
import convert_hf_to_gguf as conv  # noqa: E402
import gguf_ir  # noqa: E402
import ir_check  # noqa: E402
import ir_validate  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

out_dir.mkdir(parents=True, exist_ok=True)
local = Path(snapshot_download(repo_id=model_id, allow_patterns=["*.json"], cache_dir=str(out_dir / "hf_cache")))
hparams = conv.ModelBase.load_hparams(local)
cls = conv.ModelBase.from_model_architecture(hparams["architectures"][0])
V = int(hparams["vocab_size"])


def placeholder_vocab(self):  # the graph is under test, not the tokenizer
    self.gguf_writer.add_tokenizer_model("gpt2")
    self.gguf_writer.add_tokenizer_pre("default")
    self.gguf_writer.add_token_list([f"<t{i}>" for i in range(V)])
    self.gguf_writer.add_token_types([gguf.TokenType.NORMAL] * V)


cls.set_vocab = placeholder_vocab
safe = model_id.replace("/", "__")
path = out_dir / f"{safe}.header.gguf"
model = cls(local, gguf.LlamaFileType.MOSTLY_BF16, path, remote_hf_model_id=model_id)
model.prepare_tensors()
model.prepare_metadata(vocab_only=False)
w = model.gguf_writer
kv = w.kv_data[0]
meta = {k: v.value for k, v in kv.items()}
meta["_n_tokens"] = V
shapes = {name: [int(x) for x in info.shape] for name, info in w.tensors[0].items()}
doc = gguf_ir.build({k: v for k, v in meta.items()}, shapes)
text = json.dumps(doc, separators=(",", ":"))
w.add_string("paperkiln.ir", text)
w.add_uint32("paperkiln.ir_version", int(doc["ir_version"]))
w.add_string("paperkiln.ir_sha256", ir_validate.digest(text))
w.write_header_to_file(path=path)
w.write_kv_data_to_file()
w.write_ti_data_to_file()
w.close()
(out_dir / f"{safe}.ir.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
errs = ir_check.check(str(path))
nodes = doc["block"]["nodes"]
summary = {
    "model": model_id, "architecture": meta["general.architecture"], "tensors": len(shapes),
    "header_bytes": os.path.getsize(path), "digest": ir_validate.digest(text),
    "config": {k: hparams.get(k) for k in ("first_k_dense_replace", "scoring_func", "topk_method", "n_group",
                                           "topk_group", "norm_topk_prob", "routed_scaling_factor", "q_lora_rank",
                                           "kv_lora_rank", "rope_scaling", "num_hidden_layers")},
    "metadata": {k: v for k, v in meta.items() if k.startswith("deepseek2.") and not isinstance(v, list)},
    "ir": {"n_layers": doc["model"]["n_layers"], "block_nodes": len(nodes),
           "apply": sorted({json.dumps(n["apply"]) for n in nodes if "apply" in n}),
           "moe_ffn": [{k: v for k, v in n.items() if k in ("n_experts", "top_k", "gate_norm", "scoring",
                                                            "selection_bias", "groups")}
                       for n in nodes if n["op"] == "moe_ffn"],
           "rope_freq_factors": [n.get("freq_factors") for n in nodes if n["op"] == "rope"],
           "mla_scale": [n["scale"] for n in nodes if n["op"] == "mla_attention"],
           "affine": [n["scale"] for n in nodes if n["op"] == "affine"]},
    "ir_check": errs or "OK"}
(out_dir / f"{safe}.summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=1))
