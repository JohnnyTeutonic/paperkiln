"""Counts-only lexical overlap diagnostic. GPT/Codex, 7 November 2026.

The remote packet contains keyed fingerprints, never target text or case IDs.
This is a lexical diagnostic, not a semantic contamination certificate.
"""
import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import time

WORDS = re.compile(r"[a-z0-9]+")
SEP = b"<|endoftext|>"
WIDTH = 16


def words(text):
    return WORDS.findall(text.lower())


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tag(word, key):
    return hashlib.blake2b(word.encode(), key=key, digest_size=WIDTH).digest()


def confirm(ws, key):
    return hmac.new(key, " ".join(ws).encode(), hashlib.sha256).hexdigest()


def make_packet(texts, key):
    targets = {}
    for text in texts:
        ws = words(text)
        if not ws:
            raise ValueError("Empty target")
        n = min(8, len(ws))
        for i in range(len(ws) - n + 1):
            seq = ws[i:i + n]
            pattern = b"".join(tag(w, key) for w in seq)
            targets[pattern.hex()] = {"pattern": pattern.hex(), "n": n,
                                      "confirmation": confirm(seq, key)}
    return {"version": 1, "normalisation": "lowercase ASCII alphanumeric words",
            "rule": "contiguous 8 words; whole target when shorter than 8",
            "key": key.hex(), "targets": list(targets.values()),
            "target_texts": len(texts)}


def count_document(text, packet, cache):
    key = bytes.fromhex(packet["key"])
    ws = words(text)
    if len(cache) > 200000:
        cache.clear()
    encoded = b"".join(cache.setdefault(w, tag(w, key)) if w not in cache
                       else cache[w] for w in ws)
    counts = {}
    for t in packet["targets"]:
        pattern = bytes.fromhex(t["pattern"])
        start = 0
        while True:
            pos = encoded.find(pattern, start)
            if pos < 0:
                break
            start = pos + 1
            if pos % WIDTH == 0:
                i = pos // WIDTH
                if confirm(ws[i:i + t["n"]], key) == t["confirmation"]:
                    n = str(t["n"])
                    counts[n] = counts.get(n, 0) + 1
    return counts


def atomic(path, obj):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def scan(path, packet, progress, state_path, file_key, block_size=8 << 20):
    """Checkpoint only at complete document boundaries; never across documents."""
    state = progress.setdefault(file_key, {"offset": 0, "documents": 0,
                                          "documents_hit": 0, "windows_by_length": {}})
    cache = {}
    with open(path, "rb") as f:
        f.seek(state["offset"])
        buf = b""
        while True:
            chunk = f.read(block_size)
            parts = (buf + chunk).split(SEP)
            buf = parts.pop()
            if not chunk:
                parts += [buf] if buf.strip() else []
                buf = b""
            for doc in parts:
                hits = count_document(doc.decode("utf-8", errors="strict"), packet, cache)
                state["documents"] += 1
                state["documents_hit"] += bool(hits)
                for n, value in hits.items():
                    state["windows_by_length"][n] = state["windows_by_length"].get(n, 0) + value
            state["offset"] = f.tell() - len(buf)
            atomic(state_path, progress)
            if not chunk:
                return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--packet", required=True)
    p.add_argument("--manifest", required=True,
                   help="JSON list: name,text,text_sha256,ids,ids_sha256")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if Path(a.out).exists():
        raise SystemExit("Refusing to replace a final receipt")
    packet = json.loads(Path(a.packet).read_text())
    if packet.get("version") != 1:
        raise SystemExit("Unknown packet version")
    manifest = json.loads(Path(a.manifest).read_text())
    identity = {"packet_sha256": digest(a.packet), "manifest_sha256": digest(a.manifest),
                "worker_sha256": digest(__file__)}
    checkpoint = a.out + ".checkpoint.json"
    progress = json.loads(Path(checkpoint).read_text()) if Path(checkpoint).exists() else {"identity": identity}
    if progress["identity"] != identity:
        raise SystemExit("Checkpoint input/source drift")
    started = time.monotonic()
    for item in manifest:
        for field, expected in [("text", "text_sha256"), ("ids", "ids_sha256")]:
            if digest(item[field]) != item[expected]:
                raise SystemExit("Corpus hash mismatch")
        scan(item["text"], packet, progress, checkpoint, item["name"])
    atomic(a.out, {"status": "COMPLETE_REQUIRES_CODEX_REVIEW", "identity": identity,
                   "corpora": {i["name"]: progress[i["name"]] for i in manifest},
                   "manifest": manifest, "elapsed_this_process_seconds": time.monotonic() - started,
                   "limitations": "Lexical only; short common phrases are not proof of contamination. No target text or IDs emitted."})
    Path(checkpoint).unlink()


if __name__ == "__main__":
    main()
