#!/usr/bin/env python3
"""Export a trained paperkiln chat model as a static, zero-server web chat.

    python3 tools/export_web_chat.py RUN_DIR OUT_DIR [--vocab VOCAB.gguf] [--force]

RUN_DIR is a `mtstudio run` output directory (spec.json, <name>.safetensors,
events.jsonl, usually <name>.gguf and card.json). OUT_DIR becomes a folder
any static host can serve (GitHub Pages, Firebase Hosting, `python3 -m
http.server`); the model runs in the visitor's browser (web/chat/paperkiln.js):

    OUT_DIR/index.html            the chat page
    OUT_DIR/paperkiln.js          inference, plain JavaScript
    OUT_DIR/firebase.json         Firebase Hosting config (public: ".")
    OUT_DIR/.nojekyll             GitHub Pages: no Jekyll processing
    OUT_DIR/model/manifest.json   architecture, file names, sizes, sha256
    OUT_DIR/model/vocab.json      the token list, id order
    OUT_DIR/model/<name>.safetensors   the exported weights, unchanged
    OUT_DIR/model/card.json       the model card, when the run has one

Architecture comes from the run's own events.jsonl "model" event (what the
engine built), else from the spec resolved as mtstudio resolves it. The
vocabulary is read as LoadedLM reads it: the spec's vocab GGUF when it can
be found, else the run's exported GGUF, capped at data.vocab_cap; its size
must equal the embedding rows in the weights. For the llama family the
manifest carries rope_heads ("all" or "first"), resolved as mtstudio
resolves it for a trained run: the model event's value, "first" when the
model event predates the field, else the spec's arch.rope_heads, else
"first". Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEB = HERE.parent / "web" / "chat"
MANIFEST_SCHEMA = "paperkiln.web_chat/1"

# Mirrors PRESETS in tools/mtstudio.cpp (name -> d, layers, heads, T).
PRESETS = {
    "gpt2-nano": (128, 2, 4, 128), "llama-tiny": (128, 2, 4, 128),
    "gpt2-small": (256, 4, 8, 256), "kimi-tiny": (128, 2, 4, 128),
    "srd-tiny": (128, 2, 4, 128), "attnres-tiny": (128, 2, 4, 128),
    "swa-tiny": (128, 2, 4, 128),
}
# What paperkiln.js runs. Other attention kinds fail here, not in the browser.
SUPPORTED_ATTENTION = {"exact"}


# ---- GGUF: the token list only ---------------------------------------------
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
            10: "<Q", 11: "<q", 12: "<d"}


def read_gguf_tokens(path: Path) -> list[str]:
    """tokenizer.ggml.tokens from a GGUF (v2/v3) file, in id order."""
    b = path.read_bytes()
    p = 0

    def take(fmt):
        nonlocal p
        v = struct.unpack_from(fmt, b, p)[0]
        p += struct.calcsize(fmt)
        return v

    def string():
        nonlocal p
        n = take("<Q")
        s = b[p:p + n]
        p += n
        return s

    def value(vt):
        if vt in _SCALARS:
            return take(_SCALARS[vt])
        if vt == 8:
            return string()
        if vt == 9:
            et, n = take("<I"), take("<Q")
            return [value(et) for _ in range(n)]
        raise ValueError(f"{path}: unknown GGUF value type {vt}")

    if b[:4] != b"GGUF":
        raise ValueError(f"{path} is not a GGUF file")
    p = 4
    version = take("<I")
    if version < 2:
        raise ValueError(f"{path}: GGUF v{version} is not supported (v2 or later)")
    take("<Q")  # tensor count
    n_meta = take("<Q")
    for _ in range(n_meta):
        key = string().decode("utf-8", "replace")
        v = value(take("<I"))
        if key == "tokenizer.ggml.tokens":
            return [t.decode("utf-8", "replace") for t in v]
    raise ValueError(f"{path} has no tokenizer.ggml.tokens")


# ---- safetensors header ------------------------------------------------------
def safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n == 0 or n > (1 << 26):
            raise ValueError(f"{path}: implausible safetensors header length")
        return json.loads(f.read(n))


# ---- architecture --------------------------------------------------------------
def arch_from_events(run: Path) -> dict | None:
    ev = run / "events.jsonl"
    if not ev.exists():
        return None
    model = None
    for line in ev.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(e, dict) and e.get("event") == "model":
            model = e
    return model


def rope_heads_of(ev: dict | None, spec: dict) -> str:
    """RoPE head coverage of a trained run (mtstudio recorded_rope_heads):
    a model event without the field is a run from before it existed."""
    if ev is not None:
        return ev.get("rope_heads") or "first"
    v = spec.get("arch", {}).get("rope_heads")
    if v not in (None, "all", "first"):
        raise ValueError(f"arch.rope_heads must be all or first, got {v!r}")
    return v or "first"


def arch_from_spec(spec: dict) -> dict:
    """parse_spec's family resolution (tools/mtstudio.cpp), dims and flavours."""
    arch = spec.get("arch", {})
    a = {"family": "gpt2", "attention": "exact", "d": 128, "layers": 2, "heads": 4, "T": 128,
         "norm": "", "activation": "", "position": "", "residual": "", "d_ff": 0}
    p = arch.get("preset")
    if p:
        if p not in PRESETS:
            raise ValueError(f"unknown preset {p}")
        a["d"], a["layers"], a["heads"], a["T"] = PRESETS[p]
        for pre in ("kimi", "srd", "swa", "attnres"):
            if p.startswith(pre):
                a["attention"] = pre
        if p.startswith("llama"):
            a["family"] = "llama"
    for k, v in arch.get("custom", {}).items():
        if k in a:
            a[k] = v
    if a["position"] == "rope":
        a["family"] = "llama"
    elif a["norm"] or a["activation"] or a["position"] or a["residual"]:
        a["family"] = "flex"
    if a["family"] == "gpt2" and a["layers"] != 2 and a["attention"] in ("exact", "swa"):
        a["family"] = "flex"
    a["T"] = spec.get("data", {}).get("T", a["T"])
    if a["family"] == "llama":
        a.update(norm="rmsnorm", activation="swiglu", position="rope", residual="residual",
                 d_ff=a["d_ff"] or 3 * a["d"])
    else:
        a.update(norm=a["norm"] or "layernorm", activation=a["activation"] or "gelu",
                 position=a["position"] or "learned", residual=a["residual"] or "residual",
                 d_ff=a["d_ff"] or 4 * a["d"])
    return a


def find_vocab(run: Path, spec: dict, name: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    rel = spec.get("data", {}).get("vocab", "")
    candidates = []
    if rel:
        rp = Path(rel)
        candidates += [rp] if rp.is_absolute() else [Path.cwd() / rp, run.parent.parent / rp, run / rp]
    candidates.append(run / f"{name}.gguf")
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError("no vocabulary found (tried " + ", ".join(map(str, candidates))
                            + "); pass --vocab VOCAB.gguf")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export(run: Path, out: Path, vocab_path: str | None = None, force: bool = False) -> dict:
    spec_path = run / "spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"{run} has no spec.json (not a mtstudio run directory?)")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    name = spec.get("name", "model")
    weights = run / f"{name}.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found (export safetensors when training)")

    ev = arch_from_events(run)
    a = arch_from_spec(spec)
    if ev:
        for k in ("family", "attention", "d", "layers", "heads", "T", "d_ff", "norm",
                  "activation", "position", "residual"):
            if ev.get(k) is not None:
                a[k] = ev[k]
    if a["attention"] not in SUPPORTED_ATTENTION or a["family"] not in ("llama", "flex", "gpt2"):
        raise ValueError(f"{a['family']}/{a['attention']} attention is not supported by the "
                         "browser engine yet (exact attention in the llama, flex and gpt2 "
                         "families is)")

    vpath = find_vocab(run, spec, name, vocab_path)
    tokens = read_gguf_tokens(vpath)
    cap = spec.get("data", {}).get("vocab_cap", 0)
    if cap and cap < len(tokens):
        tokens = tokens[:cap]
    header = safetensors_header(weights)
    emb = header.get("embed_tokens.weight") or header.get("wte.weight")
    if not emb:
        raise ValueError(f"{weights} has neither embed_tokens.weight nor wte.weight")
    dtypes = {v["dtype"] for k, v in header.items() if k != "__metadata__"}
    if not dtypes <= {"F32", "F16", "BF16"}:
        raise ValueError(f"{weights} has dtypes {sorted(dtypes)}; the browser reads F32/F16/BF16")
    if emb["shape"][0] != len(tokens):
        raise ValueError(f"vocabulary {vpath} gives {len(tokens)} words but the weights have "
                         f"{emb['shape'][0]} embedding rows; pass the run's vocab with --vocab")
    if emb["shape"][1] != a["d"]:
        raise ValueError(f"weights have d={emb['shape'][1]}, architecture says d={a['d']}")

    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"{out} is not empty (use --force to overwrite)")
    (out / "model").mkdir(parents=True, exist_ok=True)
    wdst = out / "model" / weights.name
    shutil.copyfile(weights, wdst)
    vdst = out / "model" / "vocab.json"
    vdst.write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
    eos = tokens.index("<|endoftext|>") if "<|endoftext|>" in tokens else -1
    card = run / "card.json"
    if card.exists():
        shutil.copyfile(card, out / "model" / "card.json")
    elif (out / "model" / "card.json").exists():
        (out / "model" / "card.json").unlink()

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "name": name,
        "family": a["family"], "attention": a["attention"],
        "d": a["d"], "layers": a["layers"], "heads": a["heads"], "d_ff": a["d_ff"], "T": a["T"],
        "norm": a["norm"], "activation": a["activation"], "position": a["position"],
        "residual": a["residual"],
        "rope_theta": 10000.0 if a["family"] == "llama" else None,
        "rope_heads": rope_heads_of(ev, spec) if a["family"] == "llama" else None,
        "norm_eps": 1e-5,
        "tied_embeddings": "embed_tokens.weight" in header and "lm_head.weight" not in header,
        "vocab_size": len(tokens), "eos_id": eos,
        "weights": {"file": weights.name, "bytes": wdst.stat().st_size, "sha256": sha256(wdst),
                    "dtypes": sorted(dtypes)},
        "vocab": {"file": "vocab.json", "bytes": vdst.stat().st_size, "sha256": sha256(vdst),
                  "source": vpath.name},
        "card": "card.json" if card.exists() else None,
        "prompt_template": "user: <text> assistant:",
    }
    (out / "model" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                 encoding="utf-8")
    for f in ("index.html", "paperkiln.js", "firebase.json"):
        shutil.copyfile(WEB / f, out / f)
    (out / ".nojekyll").write_text("", encoding="utf-8")  # GitHub Pages: serve files as they are
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--vocab", help="vocab GGUF (default: the spec's, else RUN_DIR/<name>.gguf)")
    ap.add_argument("--force", action="store_true", help="write into a non-empty OUT_DIR")
    args = ap.parse_args(argv)
    try:
        m = export(args.run_dir, args.out_dir, args.vocab, args.force)
    except (OSError, ValueError) as e:
        print(f"export_web_chat: {e}", file=sys.stderr)
        return 1
    total = sum(p.stat().st_size for p in args.out_dir.rglob("*") if p.is_file())
    print(f"export_web_chat: {m['name']} ({m['family']}, d={m['d']}, layers={m['layers']}, "
          f"vocab {m['vocab_size']}) -> {args.out_dir} ({total / 1e6:.2f} MB)")
    print(f"  try it: cd {args.out_dir} && python3 -m http.server 8000  "
          "(then open http://127.0.0.1:8000/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
