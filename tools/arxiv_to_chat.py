#!/usr/bin/env python3
"""arXiv paper -> extracted architecture -> trained model -> chat, as one command.

    python3 tools/arxiv_to_chat.py 2302.13971 --run-dir runs/arxiv/llama \\
        --mtstudio build/mtstudio --ember-bin ~/ember_bin/338b24a/cpu/tinyllama_server \\
        --ember-www ~/ember_bin/338b24a/www --chat "Hello, how are you?"

The chain (LOCAL_LLM_STACK.md section 7, "arXiv extractor -> model ->
conversation"); every stage writes into the run directory and is skipped
when its output already exists, so a run restarts where it stopped:

  gate      the relevance gate (papers/llm_extract.py run_gate): one Claude
            call reads the paper's title and abstract (arXiv API) and
            returns in_scope, out_of_scope or spurious with a reason and an
            evidence phrase that must occur in the abstract. Anything but a
            verified in_scope REFUSES before extraction (exit 2,
            refusal.json with the reason and evidence; nothing extracted or
            trained) -> relevance.json. Skipped only with
            --skip-relevance-gate, which the manifest records. The key is
            ANTHROPIC_API_KEY (or --env-file, that line only); a cached
            verdict needs no key.
  extract   papers/fetch.py on the paper's LaTeX (fetched, or --tex, or a
            cached --extraction file: works offline), plus a scan of the
            source for mechanisms the extractor's lattice cannot express
            -> extraction.json
  spec      extraction -> mtstudio specs (pretrain + SFT). REFUSES, with
            the reason, when the architecture maps onto no family mtstudio
            trains and ember serves, or when a flavour the family needs is
            unresolved or contested: never an engine default in place of
            something the paper did not say. Size (d, layers, heads, T) is
            the run's choice, reported beside the paper's values; the
            mechanism is the paper's. -> spec_report.json, *_spec.json
            (exit 2 and refusal.json on a refusal)
            An LLM extraction (papers/llm_extract.py, --extraction) may
            carry the paper's new operation as an expr: a verified expr in
            a slot the flex family hosts (the FFN activation over one
            input; the norm over the row, e(x) * g + b with flex's learned
            per-feature g and b, a paper's own per-channel affine tail
            mapped onto them) is parsed by spec/tools/ir_expr.py, probed for
            finite f32 values and written as custom.activation / custom.norm
            = "expr:<e>" (mtstudio checks its gradient against finite
            differences before training; the GGUF's IR carries the expr
            node ember runs). A needs_new_primitive op, an unverified one,
            or an expr in any other slot refuses. RoPE with the llama
            block's flavours is the llama family; RoPE with any other
            flavour flex hosts is flex with position rope. The llama
            family also builds an extracted n_kv_heads (grouped-query
            attention at the paper's ratio of query to key/value heads,
            multi-query as one kv head; custom.kv_heads), a stated
            rope_theta, NEOX pairing and a partial rotary fraction
            (custom.rope_theta, rope_pairing, rope_dims) and a per-head
            q/k norm (custom.qk_norm); flex refuses them by name.
            Since 26 Oct 2026 (LOCAL_LLM_STACK.md 11 item 4.2) flex also
            hosts an expr over more inputs, by the extraction's input roles
            (a gate projection; learned scalars or per-channel vectors, with
            their stated inits, per norm place where the paper gives one;
            custom.activation_inputs / norm_inputs), an expr on the embed
            output (custom.embed), and a stated norm placement per sublayer
            (pre, post, sandwich, branch_post, none; custom.norm_placement).
            A state-space paper with ssm mamba1 builds the mamba preset from
            its d_state, d_conv, expand and dt_rank (custom.ssm "mamba1");
            --assert settles a size or a q/k norm width the paper leaves out.
            T5's bucketed relative bias (positional t5_bias) builds flex
            with position t5_relative from the paper's rel_buckets and
            rel_max_distance (stated and verified, or --assert; keramos
            1.3's attention.position_bias, which ember runs); a T5-bias
            paper that states neither refuses by name. A
            structural op in the block graph that no family wires there
            (conv1d_causal on q, k, v, a moe_ffn, ...) refuses by name.
  pretrain  mtstudio run, objective lm on --corpus -> pretrain/
  sft       mtstudio run, objective sft on --sft-examples (the first
            --sft-limit examples that fit T; longer ones are dropped and
            counted in sft_fit.json and the manifest, never truncated),
            init_from the pretrained weights -> sft/
  export    mtstudio export --formats gguf: GGUF with the IR embedded
            (rope_heads stated explicitly for llama) -> export/
  chat      ember serves the GGUF (an IR-only load for every non-llama
            family) next to `mtstudio chat`; greedy parity over --parity-n
            frozen prompts (/chat contract: reply, token count, stop
            reason), then the --chat turns through ember's /chat with
            history -> parity.jsonl, chat.jsonl
  --engine numpy instead runs the GGUF's IR in tools/ir_run.py (paperkiln's
  reference interpreter, single-turn) as the stand-in engine.

manifest.json records the paper id, extraction hash, the specs, corpus and
SFT hashes, the GGUF sha256, the engine binary hash and the parity result.
Exit codes: 0 done, 2 refused, 1 any other failure.

--emit-jobs FILE writes the pretrain + SFT pair as jobs for
tools/colab_chat/jobs_vm.py (Colab trains; this command then resumes from
the downloaded run with --import-sft DIR, which re-points the run's spec at
local data and continues from export).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "papers"))
sys.path.insert(0, os.path.join(ROOT, "spec", "tools"))  # keramos
sys.path.insert(0, HERE)

EXIT_REFUSED = 2
STAGES = ("gate", "extract", "spec", "pretrain", "sft", "export", "chat")

# Mechanisms papers/fetch.py has no field for. Seen in the title or abstract
# (where a paper states its contribution), or in a first-person sentence of
# the body ("we use ..."), they make the architecture something the
# extracted flavours alone would silently mis-describe, so the spec refuses.
# Elsewhere in the body (related work, "unlike X ...") they are reported.
MECHANISM_CUES = {
    "state-space model": r"state[- ]spaces?\b|\bSSMs?\b|\bMamba\b|\bS4\b",
    "linear or recurrent attention": r"linear[- ]attention|linear transformers?|\bRWKV\b|retentive network|\bRetNet\b",
    "mixture of experts": r"mixture[- ]of[- ]experts|\bMoE\b|sparsely[- ]gated",
    "grouped or multi-query attention": r"grouped[- ]query attention|\bGQA\b|multi[- ]query attention|\bMQA\b",
    "sparse or banded attention": r"sparse attention|banded (?:sparse )?attention|sparse[- ]banded|"
                                  r"dilated (?:sliding )?window|block[- ]sparse",
    "encoder-decoder or encoder-only model": r"encoder[- ]decoder|masked language model",
}
FIRST_PERSON = re.compile(r"\bwe (?:also |instead |then )?(?:use|adopt|employ|apply|leverage|introduce|propose|"
                          r"replace|follow|build|train)|\bour (?:model|architecture|design|network)s?\b",
                          re.IGNORECASE)
CONTRAST = re.compile(r"\bunlike\b|instead of|rather than|future work|\bnot\b|n't\b|\bwithout\b|"
                      r"compared? (?:to|with)|\bbaselines?\b|\bprior\b|\bprevious(?:ly)?\b|\bexisting\b",
                      re.IGNORECASE)

# The families this chain builds: mtstudio trains them, the exporter writes
# GGUF + IR for them, ember (338b24a) runs them from the IR.
LEGAL = {
    "norm": {"layernorm", "rmsnorm"},
    "activation": {"gelu", "relu", "swiglu"},
    "positional": {"rope", "learned", "sinusoidal"},
    "residual": {"residual", "highway", "plain"},
    "attention": {"exact", "swa"},
}
SPEC_KEY = {"norm": "norm", "activation": "activation", "positional": "position",
            "residual": "residual", "attention": "attention"}
# A paper's new operation written as an expr (papers/llm_extract.py) can take
# these flavour slots; the flex family hosts it as custom.<key> = "expr:<e>"
# over this many inputs: the FFN activation over the up-projection, the norm
# over the row (e(x) * g + b with a learned per-feature gain g and bias b,
# tools/parity_model.hpp FlexConfig; a norm the paper writes with its own
# per-channel parameters keeps them as flex's gain and bias when they form a
# trailing affine, see split_norm). Since 26 Oct 2026 an expr may read more
# inputs, by role (the extraction's new_ops inputs): the activation a gate
# projection and learned scalars or per-channel vectors, the norm learned
# scalars or vectors inside the row expression (DyT's alpha); flex hosts them
# as arch.custom.activation_inputs / norm_inputs. Anything else an expr
# replaces (an attention score, a routing rule) is refused: no family here
# builds it.
EXPR_SLOTS = {"activation": 1, "norm": 1, "embed": 1}
# What each input role is in flex's activation_inputs / norm_inputs.
ROLE_KINDS = {"activation": {"input": "up", "gate": "gate", "learned_scalar": "scalar", "learned_vector": "channel"},
              "norm": {"input": "row", "learned_scalar": "scalar", "learned_vector": "channel"},
              "embed": {"input": "row", "learned_scalar": "scalar", "learned_vector": "channel"}}
# The flex family's norm placements (parity::FlexConfig place_attn / place_ffn); the llama family is pre-norm.
FLEX_PLACEMENTS = ("pre", "post", "sandwich", "branch_post", "none")
FLEX_NORM_PLACEMENT = "pre"
# Block-graph ops (papers/llm_extract.py IR_OPS) that no flavour field carries: their presence in an
# extraction's block_ops refuses (build_spec), except Mamba-1's conv1d_causal and selective_scan on Mamba-1.
GRAPH_ONLY_OPS = {"conv1d_causal", "selective_scan", "ssm_scan", "linear_attention", "moe_ffn", "depth_attend",
                  "list", "append", "mul_col", "rms_row", "unsupported"}
# Where a hosted expr is probed (spec/tools/ir_expr.py, f32): every value finite.
EXPR_PROBE = (-20.0, 20.0, 0.25)
NEW_OP_CUE = re.compile(r"^new op (\S+) ")


class Refused(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg):
    print(f"[arxiv_to_chat] {msg}", flush=True)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# extract

def abstract_of(tex):
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        m = re.search(r"\\abstract\s*\{(.*?)\}\s*\n", tex, re.S)
    return m.group(1) if m else ""


def mechanism_scan(tex, title):
    """Cues for mechanisms outside the extractor's lattice, by place found."""
    import fetch

    head = fetch.detex((title or "") + "\n" + abstract_of(tex))
    body = fetch.detex(tex)
    sentences = re.split(r"(?<=[.!?])\s+", body)
    hits = []
    for mech, pat in MECHANISM_CUES.items():
        rx = re.compile(pat, re.IGNORECASE)
        m = rx.search(head)
        if m:
            hits.append({"mechanism": mech, "where": "title/abstract", "blocking": True,
                         "evidence": " ".join(head[max(0, m.start() - 80):m.end() + 60].split())})
            continue
        adopted, mentioned = None, 0
        for s in sentences:
            if rx.search(s):
                mentioned += 1
                if adopted is None and FIRST_PERSON.search(s) and not CONTRAST.search(s):
                    adopted = " ".join(s.split())[:240]
        if adopted:
            hits.append({"mechanism": mech, "where": "first-person sentence", "blocking": True,
                         "evidence": adopted})
        elif mentioned:
            hits.append({"mechanism": mech, "where": f"{mentioned} other sentence(s)", "blocking": False,
                         "evidence": "mentioned, not adopted (related work, contrast or negation)"})
    return hits


def stage_gate(a, run, manifest):
    """The relevance gate before extraction. Returns the gate record (with
    "passes"), or {"skipped": ...} under --skip-relevance-gate. On a block,
    writes refusal.json and the manifest's refused list."""
    if a.skip_relevance_gate:
        log("relevance gate SKIPPED (--skip-relevance-gate)")
        return {"skipped": True, "flag": "--skip-relevance-gate", "passes": True}
    import llm_extract

    path = os.path.join(run, "relevance.json")
    if os.path.exists(path) and read_json(path).get("arxiv_id") == a.arxiv_id:
        g = read_json(path)
    else:
        if a.env_file:
            llm_extract.load_env_key(a.env_file)
        g = llm_extract.run_gate(a.arxiv_id, None, a.gate_tag, a.gate_budget_usd)
        write_json(path, g)
    if g["passes"]:
        log(f"relevance gate: in_scope ({g['architecture_kind']}): {g['reason']}")
        return g
    reasons = llm_extract.gate_refusal_reasons(g)
    write_json(os.path.join(run, "refusal.json"), {
        "arxiv_id": a.arxiv_id, "title": g.get("title"), "stage": "relevance_gate", "verdict": g["verdict"],
        "reason": g["reason"], "architecture_kind": g["architecture_kind"], "evidence": g["evidence"],
        "evidence_found": g["evidence_found"], "reasons": reasons})
    manifest["refused"] = reasons
    print(f"\nREFUSED arXiv:{a.arxiv_id} ({g.get('title') or 'untitled'}) at the relevance gate: "
          "nothing extracted, nothing trained.")
    for x in reasons:
        print("  - " + x)
    return g


def stage_extract(a, run):
    out = os.path.join(run, "extraction.json")
    if os.path.exists(out):
        return read_json(out)
    if a.extraction:
        ext = read_json(a.extraction)
        if ext.get("arxiv_id") != a.arxiv_id:
            raise SystemExit(f"--extraction is for arXiv:{ext.get('arxiv_id')}, not {a.arxiv_id}")
        ext.setdefault("source", {})["cached_from"] = os.path.abspath(a.extraction)
        write_json(out, ext)
        return ext
    import fetch

    src = os.path.join(run, "source.tex.gz")
    if a.tex:
        tex = open(a.tex, encoding="utf-8", errors="replace").read()
    elif os.path.exists(src):
        tex = gzip.open(src, "rt", encoding="utf-8").read()
    else:
        log(f"fetching arXiv:{a.arxiv_id} source")
        tex = fetch.fetch_source(a.arxiv_id)
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.write(tex)
    arch = fetch.extract(a.arxiv_id, tex)
    ext = fetch.to_json(arch)
    ext["mechanism_scan"] = mechanism_scan(tex, arch.title)
    ext["source"] = {"tex_sha256": hashlib.sha256(tex.encode("utf-8")).hexdigest(),
                     "from": a.tex or f"https://arxiv.org/e-print/{a.arxiv_id}",
                     "extractor": "papers/fetch.py", "extractor_sha256": sha256(fetch.__file__)}
    write_json(out, ext)
    return ext


# --------------------------------------------------------------------------
# spec

# Beyond the flavours: the operator may settle a q/k norm's width, Mamba-1's sizes and T5's bucket count and
# maximum distance when the paper leaves them out.
ASSERTABLE = {**{k: None for k in LEGAL}, "qk_norm": ("per_head", "none"), "d_state": int, "d_conv": int,
              "expand": int, "dt_rank": int, "rel_buckets": int, "rel_max_distance": int}


def parse_asserts(items):
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep or k not in ASSERTABLE:
            raise SystemExit(f"--assert wants FIELD=VALUE with FIELD in {sorted(ASSERTABLE)}, got {it!r}")
        rule = ASSERTABLE[k]
        if rule is int:
            if not v.isdigit() or int(v) < 1:
                raise SystemExit(f"--assert {k} wants a positive integer, got {v!r}")
        elif rule and v not in rule:
            raise SystemExit(f"--assert {k} wants one of {', '.join(rule)}, got {v!r}")
        out[k] = v
    return out


def mamba_spec(fields, llm_ver, size, asserts, placement, refusals, report):
    """A Mamba-1 paper on the mamba preset (arch.custom.ssm "mamba1", parity::Mamba1Block: pre-RMSNorm,
    in_proj, the causal depthwise conv, the selective scan, out_proj, the residual). d_state, d_conv and
    expand are the paper's (stated, verified) or the operator's (--assert); dt_rank is the paper's rule
    ceil(d_model / K), its number scaled to this run's width, the operator's, or by default Mamba's own
    ceil(d / 16). The block's norm is RMSNorm, pre-norm: another stated norm or placement refuses."""
    ok = ("verified", "quote_only")
    custom = {"ssm": "mamba1", "d": size["d"], "layers": size["layers"]}
    for k in ("d_state", "d_conv", "expand"):
        r = llm_ver.get(k) or {}
        if k in asserts:
            custom[k] = int(asserts[k])
            report.append(("ASSERTED", f"{k} = {custom[k]} (by the operator, --assert; paper: "
                                       f"{r.get('value') or 'not stated'})"))
        elif r.get("check") in ok and (r.get("value") or "").strip():
            v = float(r["value"])
            if v != int(v) or v < 1:
                refusals.append(f"{k} = {r['value']} is not a positive integer")
                continue
            custom[k] = int(v)
            report.append(("MATCHED", f"{k} = {custom[k]} (“{' '.join((r.get('quote') or '').split())[:120]}”)"))
        else:
            refusals.append(f"{k} is unresolved: the paper does not state Mamba-1's {k} "
                            f"({r.get('check') or 'not_stated'}); assert it with --assert {k}=VALUE if you know it")
    r = llm_ver.get("dt_rank") or {}
    v = (r.get("value") or "").replace(" ", "")
    if "dt_rank" in asserts:
        custom["dt_rank"] = int(asserts["dt_rank"])
        report.append(("ASSERTED", f"dt_rank = {custom['dt_rank']} (by the operator, --assert)"))
    elif r.get("check") in ok and v:
        import llm_extract as lx
        rule = lx.DT_RANK_RULE.match(v)
        if rule:
            custom["dt_rank"] = max(1, -(-size["d"] // int(rule.group(1))))
            report.append(("MATCHED", f"dt_rank = ceil(d/{rule.group(1)}) = {custom['dt_rank']} at this run's d"))
        else:
            dm = fields.get("d_model")
            if not dm:
                refusals.append(f"dt_rank = {v} without the paper's d_model: it cannot be scaled to this run")
            else:
                custom["dt_rank"] = max(1, round(float(v) * size["d"] / float(dm["value"])))
                report.append(("SCALED", f"dt_rank {v} of d_model {dm['value']} -> {custom['dt_rank']} at d = "
                                         f"{size['d']} (the paper's ratio)"))
    else:
        report.append(("DEFAULT", f"dt_rank = ceil(d/16) = {-(-size['d'] // 16)}: not stated by the paper; "
                                  "Mamba-1's own rule (parity::Mamba1Config dt_rank 0)"))
    nf = fields.get("norm")
    if nf and nf.get("verdict") in ("used", "inherited") and nf["value"] != "rmsnorm":
        refusals.append(f"norm = {nf['value']} ({nf.get('evidence', '')[:120]}): the Mamba-1 block here normalises "
                        "with RMSNorm, so training it would silently approximate the paper")
    if any(p != FLEX_NORM_PLACEMENT for p in placement.values()):
        refusals.append(f"norm placement {placement}: the Mamba-1 block here is pre-norm")
    for k in ("activation", "positional", "attention"):
        f = fields.get(k)
        if f:
            report.append(("NOTED", f"{k} = {f['value']} ({f.get('verdict')}): the Mamba-1 block is fixed (SiLU "
                                    "gate, no positions, no attention); not a knob here"))
    if refusals:
        raise Refused(refusals)
    paper = {k: fields[k]["value"] for k in ("d_model", "n_layers") if k in fields}
    report.append(("SIZED", f"d={size['d']} layers={size['layers']} T={size['T']} chosen for this run "
                            f"(paper: {paper or 'no dims extracted'}); the mechanism is the paper's"))
    return "mamba", {"preset": "mamba-tiny", "custom": custom}, report


def t5_sizes(llm_ver, asserts, attention, refusals, report):
    """T5's bucketed relative position bias (positional t5_bias) as the flex family's position t5_relative
    (parity::FlexConfig rel_buckets, rel_max_distance; keramos IR_SPEC 5.15 attention.position_bias, which
    ember runs since 1ab6a25). The bucket count and the maximum distance are the paper's (stated, verified)
    or the operator's (--assert); neither has a default here, so a T5-bias paper that states neither refuses
    by name. Returns {rel_buckets, rel_max_distance}, the resolved ones; problems go to refusals."""
    ok = ("verified", "quote_only")
    out = {}
    for k, what in (("rel_buckets", "number of buckets"), ("rel_max_distance", "maximum distance")):
        r = llm_ver.get(k) or {}
        v = (r.get("value") or "").strip()
        if k in asserts:
            out[k] = int(asserts[k])
            report.append(("ASSERTED", f"{k} = {out[k]} (by the operator, --assert; paper: {v or 'not stated'})"))
        elif r.get("check") in ok and v:
            try:
                n = float(v)
            except ValueError:
                n = None
            if n is None or n != int(n) or n < 1:
                refusals.append(f"{k} = {v} (T5's {what}) is not a positive integer")
                continue
            out[k] = int(n)
            report.append(("MATCHED", f"{k} = {out[k]} (“{' '.join((r.get('quote') or '').split())[:120]}”)"))
        else:
            refusals.append(f"positional = t5_bias, but {k} (T5's {what}) is unresolved: the paper does not state "
                            f"it ({r.get('check') or 'not_stated'}); assert it with --assert {k}=VALUE if you know it")
    if len(out) == 2 and (out["rel_buckets"] < 2 or out["rel_max_distance"] <= out["rel_buckets"] // 2):
        refusals.append(f"rel_buckets = {out['rel_buckets']} with rel_max_distance = {out['rel_max_distance']}: "
                        "T5's bucketing needs at least 2 buckets and a maximum distance above half of them "
                        "(the exact buckets below it)")
    if attention != "exact":
        refusals.append(f"positional = t5_bias with attention = {attention}: the flex family adds T5's bias to "
                        "exact attention only")
    return out


def expr_host_check(expr, n_inputs, values=None):
    """An expr the flex family is to host: parsed by spec/tools/ir_expr.py (the
    grammar both engines implement) with n_inputs inputs, and finite in f32
    over EXPR_PROBE. values: per input, None (the probe row: the slot's own
    rows and a gate) or a number (a learned input, held at its init). Returns
    (canonical text, list of problems)."""
    import numpy as np
    import ir_expr

    try:
        node = ir_expr.parse(expr, n_inputs)
    except ir_expr.ExprError as e:
        return expr, [f"does not parse in the IR grammar (spec/tools/ir_expr.py): {e}"]
    lo, hi, step = EXPR_PROBE
    xs = np.arange(lo, hi + step / 2, step, dtype=np.float32).reshape(1, -1)
    values = values or [None] * n_inputs
    ins = [xs if v is None else np.full_like(xs, np.float32(v)) for v in values]
    with np.errstate(all="ignore"):
        y = ir_expr.evaluate(node, ins)
    bad = xs[~np.isfinite(y)]
    if bad.size:
        return expr, [f"non-finite in f32 at x = {', '.join(f'{v:g}' for v in bad[:6])}"
                      f"{' ...' if bad.size > 6 else ''} (probe [{lo:g}, {hi:g}] step {step:g}"
                      + (", learned inputs at their init" if any(v is not None for v in values) else "") + ")"]
    return expr, []


LEARNED_PARAM = re.compile(r"\blearn(?:ed|able|t)\b|\bparameter\b|\btrain(?:ed|able)\b|tf\.Variable|nn\.Parameter",
                           re.IGNORECASE)
PER_CHANNEL = re.compile(r"per[- ](?:channel|feature|dimension|element)|\bvector\b|\bchannel", re.IGNORECASE)
SCALAR = re.compile(r"\bscalar\b|\bsingle\b", re.IGNORECASE)
GATE = re.compile(r"\bgat(?:e|ing)\b", re.IGNORECASE)


def input_roles(inputs):
    """Each new-op input's role: the extraction's role (papers/llm_extract.py
    INPUT_ROLES, since 26 Oct 2026), or for an older extraction read from its
    meaning: the first input is the tensor the op stands on (x), a learned
    per-channel vector or scalar, a gate projection, else other."""
    import llm_extract as lx

    roles = []
    for k, i in enumerate(inputs):
        r = (i.get("role") or "").strip()
        if r in lx.INPUT_ROLES:
            roles.append(r)
            continue
        m = i.get("meaning") or ""
        if k == 0:
            roles.append("input")
        elif LEARNED_PARAM.search(m) and PER_CHANNEL.search(m):
            roles.append("learned_vector")
        elif LEARNED_PARAM.search(m) and SCALAR.search(m):
            roles.append("learned_scalar")
        elif GATE.search(m):
            roles.append("gate")
        else:
            roles.append("other")
    return roles


def input_init(i, d=None):
    """(init, how) of a learned input: its stated init when the extraction
    verified it (a number in its own quote, or sqrt(d_model) at this run's
    width d), else 1.0 by default."""
    import llm_extract as lx

    init = i.get("init") or {}
    if (i.get("init_check") in ("verified", "quote_only") and d
            and lx.INIT_SQRT_D.match((init.get("value") or "").replace(" ", ""))):
        return math.sqrt(d), f"stated as sqrt(d_model): {math.sqrt(d):g} at this run's d = {d}"
    if i.get("init_check") in ("verified", "quote_only"):
        v = lx.parse_number(init.get("value") or "")
        if v is not None and math.isfinite(v):
            return float(v), f"stated ({' '.join((init.get('quote') or '').split())[:80]!r})"
    return 1.0, "not stated: 1.0 (paperkiln's default)"


def split_norm(expr, inputs, roles):
    """A norm expr over the row and the paper's own learned inputs, split into
    its row expression core and a trailing affine on learned per-channel
    vectors: core*g, (core + a)*g, core*g + a, core + a (either operand
    order). flex's expr norm is core * g + b with g, b learned per feature,
    the same family of functions ((core + a) * g = core * g + a * g; core
    alone gains a scale/shift the paper's form may lack, reported). Learned
    inputs left inside the core (a scalar such as DyT's alpha, or a vector)
    stay the core's inputs, renamed x, a, b, ... in order. Returns (core
    canonical, [(role, input)] of the core in its input order, form) or
    (None, reason, None)."""
    import llm_extract as lx

    names = [i.get("name") for i in inputs]
    vec = {n for n, r in zip(names, roles) if r == "learned_vector"}
    node = lx.parse_expr(expr)

    def is_param(n):
        return n[0] == "var" and n[1] in vec

    def split_add(n):  # core + a, a + core -> (core, a)
        if n[0] == "bin" and n[1] == "+":
            if is_param(n[3]):
                return n[2], n[3][1]
            if is_param(n[2]):
                return n[3], n[2][1]
        return n, None

    def split_mul(n):  # core * g, g * core -> (core, g)
        if n[0] == "bin" and n[1] == "*":
            if is_param(n[3]):
                return n[2], n[3][1]
            if is_param(n[2]):
                return n[3], n[2][1]
        return n, None

    core, shift = split_add(node)
    core, gain = split_mul(core)
    inside = False
    if shift is None and gain is not None:
        inner, pre = split_add(core)
        if pre is not None:
            core, shift, inside = inner, pre, True
    if gain is not None and gain == shift:
        return None, f"its learned {gain} is both the gain and the shift", None
    form = ("(core + a)*g" if inside else "core*g + a") if gain and shift else \
        "core*g" if gain else "core + a" if shift else "core"
    used = lx.expr_vars(core)
    tail = {p for p in (gain, shift) if p}
    if tail & used:
        return None, "a learned vector of its affine tail also enters the row expression", None
    row = [n for n, r in zip(names, roles) if r == "input"]
    if len(row) != 1 or row[0] not in used:
        return None, "it has no single row input inside its expression", None
    keep = [n for n in names if n in used]
    rename = {n: ("x" if k == 0 else chr(ord("a") + k - 1)) for k, n in enumerate(keep)}

    def sub(n):
        kind = n[0]
        if kind == "var":
            return ("var", rename[n[1]])
        if kind == "neg":
            return ("neg", sub(n[1]))
        if kind == "bin":
            return ("bin", n[1], sub(n[2]), sub(n[3]))
        if kind == "call":
            return ("call", n[1], [sub(a) for a in n[2]])
        return n

    return lx.expr_canonical(sub(core)), [(roles[names.index(n)], inputs[names.index(n)]) for n in keep], form


def judge_new_ops(ext, size=None):
    """The new operations of an LLM extraction (papers/llm_extract.py), each
    hosted or refused. Returns (hosted {flavour: expr}, refusals, report,
    names judged, the extracted value each hosted expr replaces {flavour:
    "expr:<canonical>"}, the hosted expr's inputs {flavour: flex
    activation_inputs / norm_inputs}, only for an expr over more than its own
    row). A rule-based extraction has none."""
    llm = ext.get("llm") or {}
    ver = llm.get("verification") or {}
    ops = ver.get("new_ops") or []
    flav = ver.get("flavours") or {}
    hosted, replaces, slot_inputs, refusals, report = {}, {}, {}, [], []
    d = (size or {}).get("d")
    # Where each new op sits in the extraction's block graph: the embed section, the attention
    # sublayer's norm (before the first attention), the FFN's (after it) or the head's.
    nodes = ver.get("block_ops") or []
    first_attn = next((n["i"] for n in nodes if n.get("subgraph") == "block" and n.get("op") == "attention"), None)

    def slots_of(nm):
        out = set()
        for n in nodes:
            if n.get("new_op") != nm:
                continue
            sub = n.get("subgraph")
            out.add("final" if sub == "head" else "embed" if sub == "embed" else
                    "attn" if first_attn is not None and n["i"] < first_attn else "ffn")
        return out

    by_name = {o.get("name"): o for o in ops}
    used = {(r.get("value") or "")[7:] for r in flav.values() if (r.get("value") or "").startswith("new_op:")}
    nv = (flav.get("norm") or {}).get("value") or ""
    norm_op = nv[7:] if nv.startswith("new_op:") else None
    # A norm's variants (DyT on LLaMA: one alpha init in the attention sublayer's norm, another in the
    # FFN's and the head's): the same expression over the same input roles, used by no flavour, only at
    # norm places of the graph. They are the norm op, with their own inits where they sit.
    variants = {}
    base = by_name.get(norm_op)
    for o in ops:
        nm = o.get("name")
        if base is None or nm == norm_op or nm in used or o.get("kind") != "expr":
            continue
        same = ((o.get("expr_check") or {}).get("canonical") == (base.get("expr_check") or {}).get("canonical")
                and input_roles(o.get("inputs") or []) == input_roles(base.get("inputs") or []))
        if same and slots_of(nm) and slots_of(nm) <= {"attn", "ffn", "final"}:
            variants[nm] = sorted(slots_of(nm))
    for o in ops:
        name, kind = o.get("name", ""), o.get("kind")
        if name in variants:
            report.append(("NORM-VARIANT", f"new op {name}: the norm {norm_op}'s expression at the "
                                           f"{', '.join(variants[name])} norm(s) of the block graph, with its own "
                                           "inits (flex norm_inputs init_<place>)"))
            continue
        where = f" ({o['where']})" if o.get("where") else ""
        quote = " ".join((o.get("quote") or "").split())[:200]
        if kind == "needs_new_primitive":
            refusals.append(f"new op {name} needs a new primitive{where}: {o.get('reason') or 'no reason given'}; "
                            f"no family here builds it and no expr can state it, so the chain refuses rather "
                            f"than approximates (paper: “{quote}”)")
            continue
        if kind != "expr":
            refusals.append(f"new op {name} has kind {kind!r} (expected expr or needs_new_primitive)")
            continue
        chk = o.get("expr_check") or {}
        if o.get("check") != "verified" or o.get("errors") or not chk.get("ok"):
            refusals.append(f"new op {name} = {chk.get('expr') or '?'} is unverified{where}: "
                            + "; ".join(o.get("errors") or ["no passing expr check"]))
            continue
        users = sorted(k for k, r in flav.items() if (r.get("value") or "") == f"new_op:{name}")
        if not users and slots_of(name) == {"embed"}:
            users = ["embed"]  # an expr on the embed output (DyT's learned scalar after the embedding)
        slot = [k for k in users if k in EXPR_SLOTS]
        if not users or users != slot or len(slot) != 1:
            refusals.append(f"new op {name} = {chk['canonical']} (expr, replaces "
                            f"{o.get('replaces') or 'unstated'}) is used by "
                            f"{', '.join(users) if users else 'no flavour the chain places'}: the flex family "
                            f"hosts an expr only as {' or '.join(sorted(EXPR_SLOTS))}, so nothing here builds it")
            continue
        k = slot[0]
        inputs = o.get("inputs") or []
        roles = input_roles(inputs)
        meanings = "; ".join(f"{i.get('name')} = {i.get('meaning')} ({r})" for i, r in zip(inputs, roles))
        built, how, core_inputs = chk["canonical"], None, list(zip(roles, inputs))
        if k == "norm" and len(inputs) > 1:
            built, core_inputs, how = split_norm(chk["canonical"], inputs, roles)
            if built is None:
                refusals.append(f"new op {name} = {chk['canonical']} (expr in the norm slot): {core_inputs} "
                                f"(inputs: {meanings})")
                continue
        kinds = [ROLE_KINDS[k].get(r) for r, _ in core_inputs]
        own = "up" if k == "activation" else "row"  # norm, embed
        if None in kinds or kinds.count(own) != 1 or kinds.count("gate") > 1:
            allowed = ("the up-projection (input), one gate projection (gate), learned scalars and learned "
                       "per-channel vectors" if k == "activation" else
                       "the row (input) and learned scalars or per-channel vectors")
            refusals.append(f"new op {name} = {chk['canonical']} reads {len(inputs)} inputs ({meanings}): the flex "
                            f"{k} hosts an expr over {allowed}, so this form is not built")
            continue
        if k != "embed" and flav[k].get("check") not in ("verified", "quote_only"):
            continue  # the flavour's own verdict refuses it, by name, below
        spec_inputs, values, init_notes = [], [], []
        for (r, i), kd in zip(core_inputs, kinds):
            if kd in ("scalar", "channel"):
                v, how_init = input_init(i, d)
                entry = {"kind": kd, "init": v}
                if k == "norm" and variants:
                    # Per place: this op's init where it sits, each variant's where it sits.
                    j = inputs.index(i)
                    for nm, places in [(name, sorted(slots_of(name)))] + list(variants.items()):
                        src = (by_name[nm].get("inputs") or [])[j]
                        pv, _ = input_init(src, d)
                        for pl in places:
                            if pv != v:
                                entry[f"init_{pl}"] = pv
                    how_init += "; " + ", ".join(f"{key[5:]} {val:g}" for key, val in entry.items()
                                                 if key.startswith("init_")) if len(entry) > 2 else ""
                spec_inputs.append(entry)
                values.append(v)
                init_notes.append(f"{i.get('name')} ({kd}, init {v:g}: {how_init})")
            else:
                spec_inputs.append(kd)
                values.append(None)
        expr, problems = expr_host_check(built, len(core_inputs), values)
        if problems:
            refusals.append(f"new op {name} = {expr}: " + "; ".join(problems))
            continue
        hosted[k] = expr
        replaces[k] = "expr:" + chk["canonical"]
        if len(core_inputs) > 1:
            slot_inputs[k] = spec_inputs
        lo, hi, step = EXPR_PROBE
        report.append(("EXPR", f"{k} = expr:{expr} (new op {name}, replaces {o.get('replaces') or k}; verified "
                               f"quote{where}: “{quote[:120]}”); parses in the IR grammar "
                               f"(spec/tools/ir_expr.py), finite in f32 on [{lo:g}, {hi:g}]; mtstudio checks its "
                               "gradient against finite differences before training"))
        if len(core_inputs) > 1:
            gate_note = " and a gate projection d -> d_ff (mlp.c_gate, biased as flex's linears)" if "gate" in kinds else ""
            report.append(("EXPR-INPUTS", f"{k}_inputs = {json.dumps(spec_inputs)}: the expr's inputs in order"
                                          f"{gate_note}" + (f"; learned: {', '.join(init_notes)}" if init_notes
                                                            else "")))
        if k == "norm" and len(inputs) > 1:
            tail = [i.get("name") for i, r in zip(inputs, roles) if r == "learned_vector"
                    and i not in [ci for _, ci in core_inputs]]
            report.append(("NORM-AFFINE", (
                f"norm: the paper's {chk['canonical']} is {how} over the row expression"
                + (f" with its learned per-channel {', '.join(tail)}" if tail else "")
                + "; flex computes expr(...) * gain + bias, both learned per feature (init 1, 0): " + {
                    "(core + a)*g": "(core + a) * g = core * g + a * g (gain g, bias a * g), the same functions, another "
                                    "parametrisation",
                    "core*g + a": "the same form",
                    "core*g": "the same form plus a per-feature bias b the paper's form does not have",
                    "core + a": "the same form plus a per-feature gain g the paper's form does not have",
                    "core": "a per-feature gain and bias the paper's form does not have"}[how])))
        elif k == "norm":
            report.append(("NORM-AFFINE", "norm: flex computes expr(x) * gain + bias, both learned per feature "
                                          "(init 1, 0), as layernorm's affine; the paper's expr states the row "
                                          "expression only"))
    return hosted, refusals, report, {o.get("name", "") for o in ops}, replaces, slot_inputs


def ancestor_fill(ext):
    """Flavours an LLM extraction leaves not stated, taken from its verified
    inheritance: the curated ancestor table (papers/fetch.py ANCESTORS), the
    lookup the rule-based path makes, never generation. Only when the reply
    names an ancestor for the paper's own model (inherits, with its quote
    found in the source and resolved to a curated entry) and itself marks a
    value inherited from it; never over a stated or unverified value.
    Returns {flavour: field entry with verdict "inherited"}."""
    ver = (ext.get("llm") or {}).get("verification") or {}
    out = (ext.get("llm") or {}).get("output") or {}
    inh = ver.get("inherits") or {}
    anc = inh.get("resolved")
    if not anc or not inh.get("quote_found"):
        return {}
    import fetch

    table = fetch.ANCESTORS.get(anc) or {}
    marked = [k for k, r in (out.get("flavours") or {}).items() if r.get("status") == "inherited"]
    marked += ["family"] if (out.get("family") or {}).get("status") == "inherited" else []
    if not table or not marked:
        return {}
    fl = ver.get("flavours") or {}
    filled = {}
    for k in ("norm", "activation", "positional", "residual", "attention"):
        if k in table and (fl.get(k) or {}).get("check", "not_stated") == "not_stated" \
                and k not in ext.get("fields", {}):
            filled[k] = {"value": table[k], "verdict": "inherited",
                         "evidence": f"curated ancestor table: fetch.ANCESTORS[{anc!r}][{k!r}] = {table[k]}; the "
                                     f"paper's model inherits from {inh.get('ancestor')} (verified quote, "
                                     f"{inh.get('where') or 'no location'}: "
                                     f"“{' '.join((inh.get('quote') or '').split())[:160]}”; the "
                                     f"reply marks {', '.join(sorted(marked))} inherited) and does not restate it"}
    return filled


def build_spec(ext, size, asserts=None, skip_scan=False):
    """Extraction -> (arch dict for an mtstudio spec, report). Raises Refused
    with every reason found. size: d, layers, heads, d_ff, T."""
    asserts = asserts or {}
    report, refusals = [], []
    fields = dict(ext.get("fields", {}))
    inherited = ancestor_fill(ext)
    fields.update(inherited)

    # New operations (LLM extraction): an expr in a slot the flex family
    # hosts is built; every other new op refuses with its reason.
    hosted, op_refusals, op_report, op_names, op_replaces, op_inputs = judge_new_ops(ext, size)
    refusals += op_refusals
    report += op_report
    llm_out = (ext.get("llm") or {}).get("output") or {}
    llm_flav = llm_out.get("flavours") or {}
    llm_all = (ext.get("llm") or {}).get("verification") or {}
    llm_ver = llm_all.get("flavours") or {}
    ok_checks = ("verified", "quote_only")

    # Norm placement per sublayer: a stated (verified or quoted) placement the flex family builds
    # (parity::FlexConfig place_attn / place_ffn); the per-sublayer fields win over the shared one.
    # Anything else would be trained as pre-norm, a silent approximation, so it refuses. Unstated: pre.
    placement = {}
    for sub, key in (("attn", "norm_placement_attn"), ("ffn", "norm_placement_ffn"), ("attn", "norm_placement"),
                     ("ffn", "norm_placement")):
        r = llm_ver.get(key) or {}
        v = (r.get("value") or "").strip()
        if sub in placement or not v or r.get("check") not in ok_checks:
            continue
        placement[sub] = v
        where = f"{key} = {v} ({r.get('where') or 'no location'})"
        if v not in FLEX_PLACEMENTS:
            if sub == "ffn" and key == "norm_placement" and placement.get("attn") == v:
                continue  # the shared field, refused once
            refusals.append(f"{where}: no family here builds that placement (flex takes "
                            f"{', '.join(FLEX_PLACEMENTS)} per sublayer), so training it would silently "
                            "approximate the paper")
        else:
            report.append(("MATCHED", f"norm placement, {'attention' if sub == 'attn' else 'FFN'} sublayer: {where}"))
    placement = {sub: placement.get(sub, FLEX_NORM_PLACEMENT) for sub in ("attn", "ffn")}
    non_pre = any(v != FLEX_NORM_PLACEMENT for v in placement.values())
    # Projection biases are not a knob: flex's linears carry biases, llama's do not.
    for k in ("qkv_bias", "mlp_bias"):
        r = llm_ver.get(k) or {}
        if (r.get("value") or "") in ("true", "false") and r.get("check") != "not_stated":
            report.append(("NOTED", f"{k} = {r['value']} ({r.get('status')}, {r.get('check')}): not a knob; flex's "
                                    "projections carry biases, the llama family's do not"))

    scan = ext.get("mechanism_scan")
    if scan is None:
        if skip_scan:
            report.append(("UNCHECKED", "mechanism scan skipped (--skip-mechanism-scan): the extraction "
                                        "carries no scan of the source"))
        else:
            refusals.append("the extraction carries no mechanism scan (a cached file made without the "
                            "source); re-extract from the LaTeX (--tex or a fetch), or pass "
                            "--skip-mechanism-scan to accept that explicitly")
    nkv, nh = fields.get("n_kv_heads"), fields.get("n_heads")
    gqa = bool(nkv and nh and nkv["value"] != nh["value"])
    # What only the llama family builds (mtstudio's arch.custom.kv_heads, rope_pairing, rope_dims, rope_theta):
    # (reason, custom key, value or a function of the run's head count). A flex model refuses them below.
    llama_only = []
    # Mamba-1 (a state-space family with ssm mamba1, both verified): the mamba preset's selective scan
    # (arch.custom.ssm "mamba1" with the paper's d_state, d_conv, expand, dt_rank) builds it.
    fam = llm_all.get("family") or {}
    ssm = llm_ver.get("ssm") or {}
    mamba1 = (fam.get("check") in ok_checks and fam.get("value") == "state_space" and
              ssm.get("check") in ok_checks and ssm.get("value") == "mamba1")
    for h in scan or []:
        m = NEW_OP_CUE.match(h.get("mechanism", ""))
        if m and m.group(1) in op_names:
            continue  # judged above from the extraction's verified new-op record
        if h["blocking"] and h.get("mechanism") == "state-space model" and mamba1:
            report.append(("NOTED", f"{h['mechanism']}: {h['where']}, {h['evidence'][:160]} (built as Mamba-1: "
                                    "the extraction's ssm = mamba1)"))
            continue
        if h["blocking"] and h.get("mechanism") == "grouped or multi-query attention" and gqa:
            # The extracted n_kv_heads builds it (below); without the field it still refuses.
            report.append(("NOTED", f"{h['mechanism']}: {h['where']}, {h['evidence']} (built from n_kv_heads)"))
            continue
        if h["blocking"]:
            refusals.append(f"mechanism outside the extractor's lattice: {h['mechanism']} ({h['where']}: "
                            f"\u201c{h['evidence']}\u201d); no family here builds it, and training the "
                            "extracted flavours alone would silently approximate the paper")
        else:
            report.append(("NOTED", f"{h['mechanism']}: {h['where']}, {h['evidence']}"))

    # Structural ops in the extraction's block graph that the flavours do not carry (Primer's depthwise
    # convolutions on q, k and v as conv1d_causal nodes, 26 Oct 2026): no family below builds them there,
    # so a graph that has them refuses rather than trains without them. Mamba-1's own two are its block.
    seen_ops = set()
    for n in llm_all.get("block_ops") or []:
        op = n.get("op")
        if op not in GRAPH_ONLY_OPS or (mamba1 and op in ("conv1d_causal", "selective_scan")) or \
                (op == "unsupported" and n.get("new_op") in op_names):
            continue
        key = (op, n.get("role", ""))
        if key in seen_ops:
            continue
        seen_ops.add(key)
        refusals.append(f"block op {op} ({n.get('role') or 'no role'}, {n.get('subgraph')} subgraph) in the paper's "
                        "block graph: no family here builds it in that place, and training the extracted flavours "
                        "without it would silently approximate the paper")

    if mamba1:
        return mamba_spec(fields, llm_ver, size, asserts, placement, refusals, report)

    flav = {}
    for k in ("norm", "activation", "positional", "residual", "attention"):
        f = fields.get(k)
        if k in asserts:
            flav[k] = asserts[k]
            report.append(("ASSERTED", f"{k} = {asserts[k]} (by the operator, --assert; paper: "
                                       f"{f['value'] if f else 'unresolved'})"))
            continue
        if f is None:
            llm = bool(ext.get("llm"))
            if k == "residual":
                flav[k] = "residual"
                report.append(("DEFAULT", "residual = residual: " + (
                    "the LLM extraction found no stated residual wiring (a highway or plain stream would be "
                    "a stated departure)" if llm else
                    "the extractor's only positive signature is a highway/gated stream, and the paper "
                    "shows none")))
            elif k == "attention":
                flav[k] = "exact"
                report.append(("DEFAULT", "attention = exact: " + (
                    "the LLM extraction found no stated attention variant (windowed, linear or sparse "
                    "attention would be a stated departure)" if llm else
                    "no sliding-window signature in the paper (the extractor is one-sided by design)")))
            else:
                refusals.append(f"{k} is unresolved: the paper does not state it and names no ancestor "
                                f"that does (mentions: {ext.get('mentions', {}).get(k) or 'none'}); "
                                f"assert it with --assert {k}=VALUE if you know it")
            continue
        v, verdict = str(f["value"]), f.get("verdict")
        default = {"residual": "residual", "attention": "exact"}.get(k)
        if verdict == "unverified" and default and v == default and \
                (llm_flav.get(k) or {}).get("status") == "inherited":
            # The paper does not state it (the reply marks it inherited, and
            # the curated table has no entry to check it against): the
            # default rule for an unstated residual / attention applies, as
            # when the field is absent. A departure (highway, swa) inherited
            # without a check still refuses below.
            flav[k] = v
            report.append(("DEFAULT", f"{k} = {v}: not stated by the paper (the extraction marks it inherited, "
                                      f"unverified: {'; '.join(f.get('reasons') or [])}); "
                                      + ("a highway or plain stream" if k == "residual" else
                                         "windowed, linear or sparse attention") + " would be a stated departure"))
            continue
        if verdict not in ("used", "inherited"):
            ru = f.get("runner_up") or {}
            refusals.append(f"{k} = {v} has verdict {verdict!r}"
                            + (f" (runner-up {ru.get('value')})" if ru else "")
                            + f": not asserted by the extractor; settle it with --assert {k}=VALUE")
            continue
        flav[k] = v
        if k in inherited:
            report.append(("INHERITED", f"{k} = {v}: {f['evidence']}"))
        else:
            report.append(("MATCHED", f"{k} = {v} ({verdict}: \u201c{f['evidence'][:120]}\u201d)"))

    for k, v in flav.items():
        if k in asserts:
            pass
        elif v.startswith("new_primitive:"):
            continue  # its new op's refusal above says why
        elif v.startswith("expr:"):
            if hosted.get(k) is not None and v == op_replaces.get(k):
                flav[k] = "expr:" + hosted[k]  # a norm's row expression, its affine tail flex's own
                continue
            if not any(v[5:] in r for r in op_refusals):
                refusals.append(f"{k} = {v}: an expr with no verified new-op record that the flex family "
                                "hosts (papers/llm_extract.py writes one); not built")
            continue
        if k == "positional" and v == "t5_bias":
            continue  # built on flex from its bucket count and maximum distance (t5_sizes, below)
        if v not in LEGAL[k]:
            refusals.append(f"{k} = {v} has no engine support (mtstudio takes {sorted(LEGAL[k])})")
    t5 = t5_sizes(llm_ver, asserts, flav.get("attention", "exact"), refusals, report) \
        if flav.get("positional") == "t5_bias" else None

    if nkv and nh is None:
        refusals.append(f"n_kv_heads = {nkv['value']} without n_heads: the key/value sharing cannot be sized")
    elif gqa:
        q_heads, kv_heads = int(nh["value"]), int(nkv["value"])
        if kv_heads < 1 or q_heads % kv_heads:
            refusals.append(f"n_kv_heads = {kv_heads} does not divide n_heads = {q_heads}")
        else:
            llama_only.append((f"n_kv_heads = {kv_heads} of n_heads = {q_heads} (grouped/multi-query attention)",
                               "kv_heads", ("kv", q_heads, kv_heads)))

    # Q/K norm (the llama family's per-head RMSNorm on q and k, arch.custom.qk_norm): per_head builds it;
    # full width and an unstated width refuse (llama's is per head), unless the operator settles it.
    r = llm_ver.get("qk_norm") or {}
    qk = asserts.get("qk_norm") or ((r.get("value") or "").strip() if r.get("check") in ok_checks else "")
    if qk:
        where = (f"qk_norm = {qk} (by the operator, --assert)" if "qk_norm" in asserts else
                 f"qk_norm = {qk} ({r.get('where') or 'no location'})")
        if qk == "per_head":
            llama_only.append((where, "qk_norm", True))
        elif qk == "full":
            refusals.append(f"{where}: the queries and keys normalised over their whole width; the llama "
                            "family's q/k norm is per head, so nothing here builds it")
        elif qk == "unspecified":
            refusals.append(f"{where}: the paper normalises the queries and keys but does not say over which "
                            "width; the llama family builds a per-head norm: settle it with --assert "
                            "qk_norm=per_head if that is the paper's")
        elif qk != "none":
            refusals.append(f"{where}: not a q/k norm the chain knows (per_head, full, unspecified, none)")

    family, custom = None, {}
    # RoPE with the llama block's flavours (rmsnorm, swiglu, residual stream,
    # exact attention, pre-norm) is the llama family; RoPE with any other
    # flavour the flex family hosts (an expr activation or norm, layernorm,
    # gelu, relu, highway, swa, another norm placement) is flex with position
    # rope, every head rotated as in the llama block (tools/parity_model.hpp
    # FlexConfig).
    llama_block = all(flav.get(k, want) == want for k, want in (
        ("norm", "rmsnorm"), ("activation", "swiglu"), ("residual", "residual"), ("attention", "exact"))) \
        and not non_pre and "embed" not in hosted
    if flav.get("positional") == "rope":
        # flex rotates every head with interleaved pairs and theta 10000; the llama family takes a stated theta,
        # NEOX pairs and a partial rotary fraction (arch.custom.rope_theta, rope_pairing, rope_dims).
        for k, ok, key in (("rope_theta", lambda v: float(v) == 10000.0, "rope_theta"),
                           ("rope_pairing", lambda v: v == "interleaved", "rope_pairing"),
                           ("rope_fraction", lambda v: float(v) == 1.0, "rope_dims")):
            r = llm_ver.get(k) or {}
            v = (r.get("value") or "").strip()
            if v and r.get("check") in ("verified", "quote_only"):
                try:
                    good = ok(v)
                    val = float(v) if k != "rope_pairing" else v
                except ValueError:
                    good, val = False, None
                if not good:
                    where = f"{k} = {v} ({r.get('where') or 'no location'})"
                    if val is None or (k == "rope_pairing" and v != "neox") or \
                            (k == "rope_fraction" and not 0.0 < val < 1.0) or (k == "rope_theta" and val <= 0):
                        refusals.append(f"{where}: no rope here builds it")
                    else:
                        llama_only.append((where, key, ("frac", val) if k == "rope_fraction" else val))
    if refusals:
        raise Refused(refusals)
    if flav.get("positional") == "rope" and llama_block:
        family = "llama"
        custom["position"] = "rope"
        hd = size["d"] // size["heads"] if size["heads"] else 0
        for why, key, val in llama_only:
            if key == "kv_heads":
                _, q_heads, kv_heads = val
                if kv_heads == 1:  # multi-query: one key/value head at any width
                    custom["kv_heads"] = 1
                    report.append(("MATCHED", f"{why}: kv_heads = 1 (multi-query attention)"))
                    continue
                if (size["heads"] * kv_heads) % q_heads:
                    raise Refused([f"{why}: this run's {size['heads']} heads cannot keep the paper's ratio of "
                                   f"{q_heads // kv_heads} query heads per key/value head; choose --heads as a "
                                   f"multiple of {q_heads // kv_heads}"])
                custom["kv_heads"] = size["heads"] * kv_heads // q_heads
                report.append(("MATCHED", f"{why}: kv_heads = {custom['kv_heads']} of this run's {size['heads']} "
                                          "heads (the paper's ratio)"))
            elif key == "rope_dims":
                dims = int(round(val[1] * hd / 2.0)) * 2
                if dims < 2 or dims > hd:
                    raise Refused([f"{why}: {val[1]} of this run's head width {hd} is no even rotated width"])
                custom["rope_dims"] = dims
                report.append(("MATCHED", f"{why}: rope_dims = {dims} of the head width {hd}"))
            else:
                custom[key] = val
                report.append(("MATCHED", f"{why}: {key} = {val} (the llama family)"))
    else:
        if llama_only:
            raise Refused([f"{why}: only the llama family builds it (rmsnorm, swiglu, rope, exact attention, "
                           "residual stream); this paper's flavours make it flex" for why, _, _ in llama_only])
        family = "flex"
        for k in ("norm", "activation", "positional", "residual", "attention"):
            custom[SPEC_KEY[k]] = flav[k]
        if t5:
            custom["position"] = "t5_relative"
            custom.update(t5)
            report.append(("MATCHED", f"position = t5_relative on the flex family: T5's bucketed relative bias, "
                                      f"{t5['rel_buckets']} buckets up to distance {t5['rel_max_distance']} "
                                      "(parity::FlexConfig; keramos IR_SPEC 5.15 attention.position_bias, which "
                                      "ember runs)"))
        for k, ins in op_inputs.items():
            if k != "embed" and custom.get(SPEC_KEY[k]) == "expr:" + hosted[k]:
                custom[f"{k}_inputs"] = ins
        if "embed" in hosted:
            custom["embed"] = "expr:" + hosted["embed"]
            if "embed" in op_inputs:
                custom["embed_inputs"] = op_inputs["embed"]
            report.append(("MATCHED", f"embed = expr:{hosted['embed']} on the embed output, before the first "
                                      "block (parity::FlexConfig embed_expr)"))
        if non_pre:
            custom["norm_placement"] = (placement["attn"] if placement["attn"] == placement["ffn"] else
                                        {"attn": placement["attn"], "ffn": placement["ffn"]})
            report.append(("MATCHED", f"norm_placement = {json.dumps(custom['norm_placement'])} on the flex family "
                                      "(parity::FlexConfig place_attn / place_ffn); the norm before the head stays "
                                      "unless both sublayers are post-norm"))
        if flav.get("positional") == "rope":
            report.append(("MATCHED", "position = rope on the flex family (the paper's other flavours are not the "
                                      "llama block's): every head rotated, interleaved pairs, theta 10000, as "
                                      "the llama block and the IR rope node"))

    T = size["T"]
    custom.update({"d": size["d"], "layers": size["layers"], "heads": size["heads"]})
    if size.get("d_ff"):
        custom["d_ff"] = size["d_ff"]
    paper = {k: fields[k]["value"] for k in ("d_model", "n_layers", "n_heads", "d_ff", "context_length")
             if k in fields}
    report.append(("SIZED", f"d={size['d']} layers={size['layers']} heads={size['heads']} T={T}"
                            f"{' d_ff=' + str(size['d_ff']) if size.get('d_ff') else ''} chosen for this run "
                            f"(paper: {paper or 'no dims extracted'}); the mechanism is the paper's"))
    if size["d"] % size["heads"]:
        raise Refused([f"d={size['d']} is not divisible by heads={size['heads']}"])

    if family == "flex" and custom["attention"] == "swa":
        wf = fields.get("window_size")
        if wf is None:
            raise Refused(["attention = swa but the paper's window size is unresolved"])
        w = int(wf["value"])
        if w >= T:
            scaled = max(4, T // 4)
            report.append(("SCALED", f"window {w} -> {scaled}: window >= T={T} would make the attention "
                                     "exact and un-test the mechanism"))
            w = scaled
        custom["window"] = w
        sf = fields.get("n_sinks")
        custom["sinks"] = int(sf["value"]) if sf else 0
        report.append(("MATCHED" if sf else "DEFAULT",
                       f"sinks = {custom['sinks']}" + ("" if sf else " (the paper names no sink tokens; "
                                                                     "mtstudio's own default is 1)")))
    arch = {"custom": custom}
    if family == "llama":
        arch["rope_heads"] = "all"
        report.append(("MATCHED", "rope_heads = all (GGUF engines rotate every head; contract point 2)"))
    return family, arch, report


def make_specs(a, run, family, arch):
    name = "arxiv_" + re.sub(r"[^0-9A-Za-z]+", "_", a.arxiv_id)
    common_data = {"vocab": os.path.abspath(a.vocab), "vocab_cap": a.vocab_cap, "T": a.T}
    pre = {
        "name": name + "_base",
        "arch": arch,
        "data": {**common_data, "corpus": os.path.abspath(a.corpus), "max_tokens": a.max_tokens},
        "train": {"steps": a.pretrain_steps, "lr": a.lr, "batch": a.batch,
                  "eval_every": max(1, a.pretrain_steps // 4),
                  "checkpoint_every": min(a.checkpoint_every or a.pretrain_steps, a.pretrain_steps) or 1,
                  "gradmap_every": 0, "seed": a.seed, "optimizer": "adamw"},
        "export": {"formats": ["safetensors"]},
        "out_dir": os.path.join(run, "pretrain"),
    }
    sft = {
        "name": name,
        "arch": arch,
        "data": {**common_data, "sft_examples": os.path.join(run, "sft_examples.jsonl")},
        "train": {"steps": a.sft_steps, "lr": a.sft_lr, "batch": a.batch,
                  "eval_every": max(1, a.sft_steps // 2),
                  "checkpoint_every": min(a.checkpoint_every or a.sft_steps, a.sft_steps) or 1,
                  "gradmap_every": 0, "seed": a.seed, "optimizer": "adamw", "objective": "sft",
                  "init_from": os.path.join(run, "pretrain", name + "_base.safetensors")},
        "export": {"formats": ["safetensors"]},
        "out_dir": os.path.join(run, "sft"),
    }
    for s, warm in ((pre, a.lr_warmup), (sft, a.sft_lr_warmup)):
        if a.lr_decay != "none" or warm:
            s["train"]["lr_schedule"] = {"warmup_steps": warm, "decay": a.lr_decay, "min_lr_frac": 0.1}
    return name, pre, sft


def stage_spec(a, run, ext, manifest):
    rep_path = os.path.join(run, "spec_report.json")
    if os.path.exists(rep_path) and os.path.exists(os.path.join(run, "sft_spec.json")):
        rep = read_json(rep_path)
        return rep["name"], rep["family"]
    size = {"d": a.d, "layers": a.layers, "heads": a.heads, "d_ff": a.d_ff, "T": a.T}
    try:
        family, arch, report = build_spec(ext, size, parse_asserts(a.assert_), a.skip_mechanism_scan)
    except Refused as r:
        reasons = r.args[0]
        write_json(os.path.join(run, "refusal.json"), {"arxiv_id": a.arxiv_id, "title": ext.get("title"),
                                                       "reasons": reasons})
        manifest["refused"] = reasons
        print(f"\nREFUSED arXiv:{a.arxiv_id} ({ext.get('title') or 'untitled'}): no spec, nothing trained.")
        for x in reasons:
            print("  - " + x)
        return None, None
    name, pre, sft = make_specs(a, run, family, arch)
    write_json(os.path.join(run, "pretrain_spec.json"), pre)
    write_json(os.path.join(run, "sft_spec.json"), sft)
    write_json(rep_path, {"name": name, "family": family, "report": report})
    print(f"\nspec for arXiv:{a.arxiv_id} ({ext.get('title') or 'untitled'}): family {family}")
    for b, d in report:
        print(f"  [{b}] {d}")
    return name, family


# --------------------------------------------------------------------------
# train / export

def mtstudio(a, args, log_path):
    env = {**os.environ, "OMP_NUM_THREADS": str(a.threads)}
    with open(log_path, "a", encoding="utf-8") as lf:
        p = subprocess.run([a.mtstudio, *args], stdout=lf, stderr=subprocess.STDOUT, env=env)
    if p.returncode:
        tail = open(log_path, encoding="utf-8", errors="replace").read()[-1500:]
        raise SystemExit(f"mtstudio {args[0]} failed ({p.returncode}); {log_path} ends:\n{tail}")


def sft_fit(src, vocab, T, limit, dst):
    """The SFT slice mtstudio trains on: examples in file order whose prompt
    tokens + completion tokens + the end token fit T + 1 (load_sft's bound:
    one row x = seq[0..T-1], y = seq[1..T]), up to `limit` (0 = all); longer
    ones are dropped and counted, never truncated. Tokenised as load_sft
    does (prompt and completion separately, tools/bpe.py = mtstudio's BPE).
    Writes dst atomically; returns the counts."""
    import bpe

    tok = bpe.read_gguf(vocab)
    seen = kept = dropped = 0
    longest_kept = longest_dropped = 0
    with open(src, encoding="utf-8") as f, open(dst + ".tmp", "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            seen += 1
            ex = json.loads(line)
            n = len(tok.encode(ex["prompt"])) + len(tok.encode(ex["completion"])) + 1
            if n > T + 1:
                dropped += 1
                longest_dropped = max(longest_dropped, n)
                continue
            g.write(line if line.endswith("\n") else line + "\n")
            kept += 1
            longest_kept = max(longest_kept, n)
            if limit and kept >= limit:
                break
    os.replace(dst + ".tmp", dst)
    return {"source": os.path.abspath(src), "T": T, "limit": limit, "seen": seen, "kept": kept,
            "dropped_longer_than_T": dropped, "longest_kept_tokens": longest_kept,
            "longest_dropped_tokens": longest_dropped,
            "rule": "len(prompt) + len(completion) + 1 end token <= T + 1 (mtstudio load_sft), file order"}


def stage_train(a, run, which, name):
    spec = os.path.join(run, f"{which}_spec.json")
    out = os.path.join(run, which)
    stem = name + ("_base" if which == "pretrain" else "")
    weights = os.path.join(out, stem + ".safetensors")
    if os.path.exists(os.path.join(out, "result.json")) and os.path.exists(weights):
        return weights
    if which == "sft":
        dst = os.path.join(run, "sft_examples.jsonl")
        if not os.path.exists(dst):
            fit = sft_fit(a.sft_examples, a.vocab, a.T, a.sft_limit, dst)
            write_json(os.path.join(run, "sft_fit.json"), fit)
            log(f"sft: {fit['kept']} examples kept (first {a.sft_limit or 'all'} that fit T={a.T}); "
                f"{fit['dropped_longer_than_T']} of the {fit['seen']} read were longer and dropped")
    log(f"{which}: mtstudio run ({read_json(spec)['train']['steps']} steps)")
    t0 = time.time()
    mtstudio(a, ["run", spec], os.path.join(run, f"{which}.log"))
    log(f"{which}: done in {time.time() - t0:.0f} s")
    if not os.path.exists(weights):
        raise SystemExit(f"{which}: mtstudio finished without {weights}")
    return weights


def stage_export(a, run, name):
    out = os.path.join(run, "export")
    gguf = os.path.join(out, name + ".gguf")
    if os.path.exists(gguf):
        return gguf
    os.makedirs(out, exist_ok=True)
    log("export: mtstudio export --formats gguf")
    mtstudio(a, ["export", os.path.join(run, "sft", "spec.json"), "--from",
                 os.path.join(run, "sft", name + ".safetensors"), "--formats", "gguf", "--out-dir", out],
             os.path.join(run, "export.log"))
    if not os.path.exists(gguf):
        raise SystemExit(f"export wrote no {gguf}; see {run}/export.log")
    return gguf


def import_sft(a, run, name):
    """A run trained elsewhere (Colab, jobs_vm.py): copy its out_dir in as
    sft/ and point its spec at this machine's vocabulary."""
    dst = os.path.join(run, "sft")
    if os.path.exists(os.path.join(dst, "result.json")):
        return
    shutil.copytree(a.import_sft, dst, dirs_exist_ok=True)
    sp = os.path.join(dst, "spec.json")
    s = read_json(sp if os.path.exists(sp) else os.path.join(run, "sft_spec.json"))
    s["data"]["vocab"] = os.path.abspath(a.vocab)
    s["out_dir"] = dst
    write_json(sp, s)
    if not os.path.exists(os.path.join(dst, name + ".safetensors")):
        raise SystemExit(f"--import-sft {a.import_sft} has no {name}.safetensors")


# --------------------------------------------------------------------------
# serve + chat

def listeners(port):
    out = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if re.search(rf":{port}\s", ln)]


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


class Server:
    """A server this command starts, owns and stops (by its own pid only)."""

    def __init__(self, label, argv, port, expect, log_path, env=None):
        if listeners(port):
            raise SystemExit(f"port {port} is already in use: {listeners(port)}")
        self.label, self.port = label, port
        self.log = open(log_path, "w", encoding="utf-8")
        self.p = subprocess.Popen(argv, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                  start_new_session=True)
        deadline = time.time() + 300
        while True:
            if self.p.poll() is not None:
                raise SystemExit(f"{label} exited ({self.p.returncode}); see {log_path}")
            try:
                self.health = get(f"http://127.0.0.1:{port}/health")
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                if time.time() > deadline:
                    self.stop()
                    raise SystemExit(f"{label} did not answer /health on port {port}")
                time.sleep(1)
        ls = listeners(port)
        if len(ls) != 1 or f"pid={self.p.pid}," not in ls[0]:
            self.stop()
            raise SystemExit(f"port {port}: expected only {label} (pid {self.p.pid}) listening, got {ls}")
        if expect not in self.health:
            self.stop()
            raise SystemExit(f"{label} /health does not name {expect}: {self.health}")
        log(f"{label} up on {port} (pid {self.p.pid}): {self.health[:160]}")

    def chat(self, body):
        return post(f"http://127.0.0.1:{self.port}/chat", body)

    def stop(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.SIGTERM)
            try:
                self.p.wait(15)
            except subprocess.TimeoutExpired:
                self.p.kill()
                self.p.wait()
        self.log.close()


def greedy_body(prompt, max_new, history=None):
    # The /chat contract (point 4); repetition penalty left out = off, since
    # mtstudio's /chat does not take it yet.
    b = {"user_input": prompt, "max_new_tokens": max_new, "temperature": 1.0, "top_k": 1, "top_p": 1.0,
         "seed": 0}
    if history:
        b["history"] = history
    return b


def stage_chat(a, run, name, gguf, manifest):
    import bpe

    prompts = [json.loads(x)["prompt"] for x in open(a.prompts, encoding="utf-8") if x.strip()][:a.parity_n]
    tok = bpe.read_gguf(gguf)
    mts = Server("mtstudio chat", [a.mtstudio, "chat", os.path.join(run, "sft"), "--port", str(a.port_mts),
                                   "--host", "127.0.0.1"], a.port_mts, f'"model":"{name}"',
                 os.path.join(run, "mtstudio_chat.log"), env={**os.environ, "OMP_NUM_THREADS": str(a.threads)})
    try:
        ref = [mts.chat(greedy_body(p, a.max_new)) for p in prompts]
        if a.engine == "numpy":
            ref += [mts.chat(greedy_body(u, a.max_new)) for u in a.chat]
    finally:
        mts.stop()

    if a.engine == "numpy":
        return chat_numpy(a, run, gguf, prompts, ref, manifest)

    eng = Server("ember", [a.ember_bin, gguf, str(a.port_engine), "127.0.0.1", a.ember_www, "0"],
                 a.port_engine, f'"model":"{gguf}"',  # ember's /health names the GGUF it loaded
                 os.path.join(run, "ember.log"), env={**os.environ, "OMP_NUM_THREADS": str(a.threads)})
    rows, same = [], {"reply": 0, "tokens": 0, "stop": 0, "prompt_tokens": 0}
    try:
        for i, (p, m) in enumerate(zip(prompts, ref)):
            e = eng.chat(greedy_body(p, a.max_new))
            py_n = len(tok.encode("<|user|>" + p + "<|assistant|>"))
            r = {"i": i, "prompt": p, "ember": e["reply"], "mtstudio": m["reply"],
                 "ember_tokens": e["tokens"], "mts_tokens": m["tokens"],
                 "ember_stop": e["stop_reason"], "mts_stop": m["stop_reason"],
                 "ember_prompt_tokens": e.get("prompt_tokens"), "bpe_py_prompt_tokens": py_n}
            rows.append(r)
            same["reply"] += e["reply"] == m["reply"]
            same["tokens"] += e["tokens"] == m["tokens"]
            same["stop"] += e["stop_reason"] == m["stop_reason"]
            same["prompt_tokens"] += e.get("prompt_tokens") == py_n
        history, turns = [], []
        for u in a.chat:
            e = eng.chat(greedy_body(u, a.max_new, history) if a.chat_greedy else
                         {"user_input": u, "max_new_tokens": a.max_new, "temperature": 0.7, "top_k": 40,
                          "top_p": 0.9, "seed": 1, "history": history})
            turns.append({"user": u, "reply": e["reply"], "stop_reason": e["stop_reason"],
                          "tokens": e["tokens"], "history_turns_used": e.get("history_turns_used")})
            history.append({"user": u, "assistant": e["reply"]})
    finally:
        eng.stop()
    with open(os.path.join(run, "parity.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(run, "chat.jsonl"), "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    n = len(rows)
    parity = {"engine": "ember", "n": n, "identical_replies": same["reply"], "same_token_count": same["tokens"],
              "same_stop_reason": same["stop"], "prompt_tokens_match_bpe_py": same["prompt_tokens"],
              "pass": same["reply"] == n and same["tokens"] == n and same["stop"] == n,
              "settings": greedy_body("", a.max_new) | {"user_input": None},
              "prompts": os.path.relpath(a.prompts, ROOT) if a.prompts.startswith(ROOT) else a.prompts,
              "prompts_sha256": sha256(a.prompts), "ember_health": eng.health, "mtstudio_health": mts.health}
    print(f"\nparity (greedy, ember vs mtstudio /chat): {same['reply']}/{n} identical replies, "
          f"{same['tokens']}/{n} token counts, {same['stop']}/{n} stop reasons, "
          f"{same['prompt_tokens']}/{n} prompt token counts == bpe.py")
    for r in rows:
        if r["ember"] != r["mtstudio"]:
            print(f"  DIFF [{r['i']}] ember {r['ember'][:70]!r} | mtstudio {r['mtstudio'][:70]!r}")
    print_chat(turns)
    return parity, turns


def chat_numpy(a, run, gguf, prompts, ref, manifest):
    """tools/ir_run.py as the engine: it reruns the IR from position 0 each
    step, single-turn only, so the chat turns are separate prompts."""
    receipt = os.path.join(run, "mtstudio_greedy.jsonl")
    allp = prompts + list(a.chat)
    with open(receipt, "w", encoding="utf-8") as f:
        for i, (p, m) in enumerate(zip(allp, ref)):
            f.write(json.dumps({"i": i, "prompt": p, "mtstudio": m["reply"]}) + "\n")
    out = os.path.join(run, "parity.jsonl")
    p = subprocess.run([sys.executable, os.path.join(HERE, "ir_run.py"), gguf, receipt, HERE, out,
                        "--max-new", str(a.max_new)], capture_output=True, text=True)
    print(p.stdout)
    if p.returncode:
        raise SystemExit(f"ir_run.py failed: {p.stderr[-1500:]}")
    rows = [json.loads(x) for x in open(out, encoding="utf-8")]
    n = len(prompts)
    same = sum(r["same"] for r in rows[:n])
    turns = [{"user": u, "reply": r["ir_numpy"], "tokens": r["tokens"]} for u, r in zip(a.chat, rows[n:])]
    with open(os.path.join(run, "chat.jsonl"), "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    print_chat(turns)
    return {"engine": "numpy (tools/ir_run.py)", "n": n, "identical_replies": same, "pass": same == n,
            "settings": greedy_body("", a.max_new) | {"user_input": None},
            "prompts_sha256": sha256(a.prompts)}, turns


def print_chat(turns):
    if turns:
        print("\nchat:")
    for t in turns:
        print(f"  user:      {t['user']}\n  assistant: {t['reply']}")


# --------------------------------------------------------------------------
# Colab jobs

def emit_jobs(a, run, name, path):
    """The pretrain + SFT pair as jobs_vm.py jobs, data paths on the VM."""
    pre, sft = read_json(os.path.join(run, "pretrain_spec.json")), read_json(os.path.join(run, "sft_spec.json"))

    def vm(p, base):  # absolute VM paths as given; relative ones under base
        return p if p.startswith("/") else base + p
    vm_vocab = vm(a.vm_vocab, "/content/pk/")
    jobs = []
    for s in (pre, sft):
        s = json.loads(json.dumps(s))
        s.pop("out_dir", None)
        s.pop("name", None)
        s["data"]["vocab"] = vm_vocab
        s["export"] = {"formats": ["safetensors", "gguf"]}
        # A reclaimed VM resumes from the last checkpoint the runner relayed.
        s["train"]["checkpoint_every"] = min(500, s["train"]["steps"])
        if "corpus" in s["data"]:
            s["data"]["corpus"] = vm(a.vm_corpus, "/content/pk/")
            if s["data"]["corpus"].endswith(".ids"):
                s["data"]["max_tokens"] = 0  # an .ids corpus is read whole (pretrain steps set the tokens seen)
        else:
            s["data"]["sft_examples"] = vm(a.vm_sft_upload, "/content/chat/upload/")
            s["train"].pop("init_from", None)
        jobs.append(s)
    short = re.sub(r"[^0-9A-Za-z]+", "_", a.arxiv_id)
    doc = read_json(path) if os.path.exists(path) else {"_comment": "", "max_parallel": 2, "jobs": []}
    doc["jobs"] = [j for j in doc["jobs"] if j["name"] not in (f"base_{short}", f"sft_{short}")]
    extra = [x for x in a.vm_after.split(",") if x] if a.vm_after else []
    sft_extra = [x for x in a.vm_sft_after.split(",") if x] if a.vm_sft_after else []
    new = [{"name": f"base_{short}", "model_name": name + "_base", "spec": jobs[0], "relay_every_steps": 2000},
           {"name": f"sft_{short}", "model_name": name, "after": [f"base_{short}", *sft_extra],
            "init_job": f"base_{short}", "spec": jobs[1]}]
    if extra:
        new[0]["after"] = extra
    # Deferred downloads: validated for llama, and for gpt2 and flex since defect D1's fix (25 Oct 2026,
    # tests/test_gpt2_device.cpp; deferred runs bit-identical to write-through in tools/device_eval_gate.py).
    # The other families keep them off until they are gated.
    if read_json(os.path.join(run, "spec_report.json"))["family"] not in ("llama", "gpt2", "flex"):
        for j in new:
            j["env"] = {"MICROTORCH_DEFER_DOWNLOADS": 0}
    doc["jobs"] += new
    write_json(path, doc)
    log(f"jobs for arXiv:{a.arxiv_id} written to {path}")


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("arxiv_id")
    ap.add_argument("--run-dir", required=True)
    gt = ap.add_argument_group("relevance gate (before extraction)")
    gt.add_argument("--skip-relevance-gate", action="store_true",
                    help="run without the gate (recorded in the manifest)")
    gt.add_argument("--gate-tag", default="relevance_gate", help="ledger/budget tag for the gate call")
    gt.add_argument("--gate-budget-usd", type=float, default=None, help="refuse the call past this spend")
    gt.add_argument("--env-file", help="read ANTHROPIC_API_KEY (that line only) from this dotenv file")
    src = ap.add_argument_group("extraction source (default: fetch from arXiv)")
    src.add_argument("--extraction", help="cached extraction.json (offline)")
    src.add_argument("--tex", help="local LaTeX source")
    sp = ap.add_argument_group("spec")
    sp.add_argument("--assert", dest="assert_", action="append", metavar="FIELD=VALUE",
                    help="operator assertion of a flavour the paper leaves unresolved (recorded)")
    sp.add_argument("--skip-mechanism-scan", action="store_true")
    sp.add_argument("--d", type=int, default=128)
    sp.add_argument("--layers", type=int, default=2)
    sp.add_argument("--heads", type=int, default=4)
    sp.add_argument("--d-ff", type=int, default=0, help="0 = family default (3d llama, 4d flex)")
    sp.add_argument("--T", type=int, default=128)
    tr = ap.add_argument_group("training")
    tr.add_argument("--mtstudio", default=os.path.join(ROOT, "build", "mtstudio"))
    tr.add_argument("--corpus", default=os.path.join(ROOT, "corpus/data/tinychat_hf_sample_v1/pretrain_train.txt"))
    tr.add_argument("--vocab", default=os.path.join(
        ROOT, "corpus/data/vocab_fineweb_tinychat_bpe8k_v1/fineweb_tinychat_bpe8k.gguf"))
    tr.add_argument("--vocab-cap", type=int, default=8192)
    tr.add_argument("--max-tokens", type=int, default=400000)
    tr.add_argument("--sft-examples", default=os.path.expanduser("~/pkchat/fwsft/sft_mix.jsonl"))
    tr.add_argument("--sft-limit", type=int, default=2000, help="first N SFT examples (0 = all)")
    tr.add_argument("--pretrain-steps", type=int, default=300)
    tr.add_argument("--sft-steps", type=int, default=150)
    tr.add_argument("--batch", type=int, default=4)
    tr.add_argument("--lr", type=float, default=3e-3)
    tr.add_argument("--sft-lr", type=float, default=1e-3)
    tr.add_argument("--lr-warmup", type=int, default=0, help="pretrain warmup steps")
    tr.add_argument("--sft-lr-warmup", type=int, default=0)
    tr.add_argument("--lr-decay", choices=("none", "cosine"), default="none", help="both stages")
    tr.add_argument("--seed", type=int, default=1)
    tr.add_argument("--threads", type=int, default=2)
    tr.add_argument("--checkpoint-every", type=int, default=0,
                    help="steps between checkpoints (0 = at the end only); a stopped stage resumes "
                         "from its last checkpoint when rerun")
    tr.add_argument("--import-sft", help="a trained SFT out_dir from elsewhere (Colab): skip pretrain + sft")
    ch = ap.add_argument_group("serve and chat")
    ch.add_argument("--engine", choices=("ember", "numpy"), default="ember")
    ch.add_argument("--ember-bin", default=os.path.expanduser("~/ember_bin/338b24a/cpu/tinyllama_server"))
    ch.add_argument("--ember-www", default=os.path.expanduser("~/ember_bin/338b24a/www"))
    ch.add_argument("--port-mts", type=int, default=8121)
    ch.add_argument("--port-engine", type=int, default=8122)
    ch.add_argument("--prompts", default=os.path.join(ROOT, "corpus/eval/tinychat_hf_prompts_v1.jsonl"))
    ch.add_argument("--parity-n", type=int, default=10)
    ch.add_argument("--max-new", type=int, default=40)
    ch.add_argument("--chat", action="append", default=[], help="a user turn (repeatable; one conversation)")
    ch.add_argument("--chat-greedy", action="store_true", help="greedy chat turns (default: t 0.7, top-k 40)")
    ap.add_argument("--stop-after", choices=STAGES)
    ap.add_argument("--emit-jobs", help="append this paper's pretrain + SFT jobs to a jobs_vm.py JSON")
    ap.add_argument("--vm-corpus", default="corpus/data/fineweb_edu_sample_v1/pretrain_train.txt")
    ap.add_argument("--vm-vocab", default="corpus/data/vocab_fineweb_tinychat_bpe8k_v1/fineweb_tinychat_bpe8k.gguf")
    ap.add_argument("--vm-sft-upload", default="sft_dialogue_v1.jsonl",
                    help="an uploaded file's name, or an absolute VM path (a data job's output)")
    ap.add_argument("--vm-after", default="", help="jobs the pretrain job waits for (comma list), e.g. the mix")
    ap.add_argument("--vm-sft-after", default="", help="more jobs the SFT job waits for, e.g. the SFT set's builder")
    a = ap.parse_args(argv)

    run = os.path.abspath(a.run_dir)
    os.makedirs(run, exist_ok=True)
    mpath = os.path.join(run, "manifest.json")
    manifest = read_json(mpath) if os.path.exists(mpath) else {"arxiv_id": a.arxiv_id, "stages": {}}
    manifest.pop("refused", None)

    def done(stage, **kv):
        manifest["stages"][stage] = {"at": time.strftime("%Y-%m-%dT%H:%M:%S"), **kv}
        write_json(mpath, manifest)

    g = stage_gate(a, run, manifest)
    if g.get("skipped"):
        manifest["relevance_gate"] = "SKIPPED by --skip-relevance-gate"
        done("gate", skipped=True, flag="--skip-relevance-gate")
    else:
        manifest["relevance_gate"] = g["verdict"] if g["passes"] else f"REFUSED ({g['verdict']})"
        done("gate", verdict=g["verdict"], passes=g["passes"], reason=g["reason"],
             architecture_kind=g["architecture_kind"], evidence=g["evidence"],
             evidence_found=g["evidence_found"], gate=g["gate"], **({} if g["passes"] else {"refused": True}))
        if not g["passes"]:
            return EXIT_REFUSED
    if a.stop_after == "gate":
        return 0

    ext = stage_extract(a, run)
    manifest["title"] = ext.get("title")
    done("extract", extraction_sha256=sha256(os.path.join(run, "extraction.json")),
         source=ext.get("source"), mechanism_scan=ext.get("mechanism_scan"))
    if a.stop_after == "extract":
        return 0

    name, family = stage_spec(a, run, ext, manifest)
    if name is None:
        done("spec", refused=True)
        return EXIT_REFUSED
    rep = read_json(os.path.join(run, "spec_report.json"))
    done("spec", family=family, name=name, report=rep["report"],
         pretrain_spec=read_json(os.path.join(run, "pretrain_spec.json")),
         sft_spec=read_json(os.path.join(run, "sft_spec.json")))
    if a.emit_jobs:
        emit_jobs(a, run, name, a.emit_jobs)
    if a.stop_after == "spec":
        return 0

    if a.import_sft:
        import_sft(a, run, name)
        done("sft", imported_from=os.path.abspath(a.import_sft),
             weights_sha256=sha256(os.path.join(run, "sft", name + ".safetensors")))
    else:
        w = stage_train(a, run, "pretrain", name)
        done("pretrain", corpus=a.corpus, corpus_sha256=sha256(a.corpus), max_tokens=a.max_tokens,
             vocab_sha256=sha256(a.vocab), weights_sha256=sha256(w),
             result=read_json(os.path.join(run, "pretrain", "result.json")))
        if a.stop_after == "pretrain":
            return 0
        w = stage_train(a, run, "sft", name)
        fit_path = os.path.join(run, "sft_fit.json")
        done("sft", sft_source=a.sft_examples, sft_limit=a.sft_limit,
             sft_fit=read_json(fit_path) if os.path.exists(fit_path) else None,
             sft_slice_sha256=sha256(os.path.join(run, "sft_examples.jsonl")), weights_sha256=sha256(w),
             result=read_json(os.path.join(run, "sft", "result.json")))
    if a.stop_after == "sft":
        return 0

    gguf = stage_export(a, run, name)
    irp = os.path.join(run, "export", name + ".ir.json")
    ir = read_json(irp) if os.path.exists(irp) else {}
    rope = [n.get("rope_heads") for n in ir.get("block", {}).get("nodes", []) if n.get("op") == "rope"]
    done("export", gguf=gguf, gguf_sha256=sha256(gguf), ir_sha256=sha256(irp) if ir else None,
         family_hint=ir.get("family_hint"), rope_heads=rope[0] if rope else None)
    if a.stop_after == "export":
        return 0

    parity, turns = stage_chat(a, run, name, gguf, manifest)
    if a.engine == "ember":
        parity["ember_bin"] = a.ember_bin
        parity["ember_bin_sha256"] = sha256(a.ember_bin)
    done("chat", parity=parity, chat=turns)
    manifest["parity_pass"] = parity["pass"]
    write_json(mpath, manifest)
    log(f"run directory: {run}")
    return 0 if parity["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
