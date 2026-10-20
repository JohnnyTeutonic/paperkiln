"""Shared fetcher discipline (docs/decisions/0001-posttrain-charter.md, section 9):
pinned source, deterministic output path, idempotent, sha256 verified, loud
failure on mismatch. Every corpus fetcher imports this; nothing else writes
under corpus/data/.
"""
from __future__ import annotations

import hashlib
import os
import sys

CORPUS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(CORPUS_DIR, "data")      # gitignored: bytes, not records
HASH_DIR = os.path.join(CORPUS_DIR, "hashes")    # committed: the record


class HashMismatch(RuntimeError):
    pass


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def out_path(corpus_id, filename):
    d = os.path.join(DATA_DIR, corpus_id)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def expected_hashes(corpus_id):
    """{filename: sha256} from corpus/hashes/<id>.sha256 (sha256sum format),
    or {} when the corpus has not been pinned yet."""
    p = os.path.join(HASH_DIR, f"{corpus_id}.sha256")
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                h, name = line.split(None, 1)
                out[name.lstrip("*")] = h
    return out


def pin_hashes(corpus_id, files, note):
    """Write corpus/hashes/<id>.sha256 the first time a corpus is fetched.
    Refuses to overwrite: a pinned hash changes only by a new corpus id."""
    p = os.path.join(HASH_DIR, f"{corpus_id}.sha256")
    if os.path.exists(p):
        raise RuntimeError(f"{p} exists; a pinned corpus is never re-pinned")
    os.makedirs(HASH_DIR, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {note}\n")
        for name, h in sorted(files.items()):
            f.write(f"{h}  {name}\n")


def finish(corpus_id, paths, note, pin=False):
    """Verify every produced file against its pinned hash. With pin=True and
    no pin on record, record the hashes instead (first fetch of a new id)."""
    got = {os.path.basename(p): sha256_file(p) for p in paths}
    want = expected_hashes(corpus_id)
    if not want:
        if not pin:
            raise HashMismatch(f"{corpus_id}: no pinned hashes; run with --pin once, "
                               "then commit corpus/hashes/")
        pin_hashes(corpus_id, got, note)
        print(f"{corpus_id}: PINNED {got}")
        return got
    for name, h in got.items():
        if want.get(name) != h:
            raise HashMismatch(f"{corpus_id}/{name}: sha256 {h} != pinned {want.get(name)}")
    print(f"{corpus_id}: OK (sha256 verified: {', '.join(sorted(got))})")
    return got


def already_done(corpus_id, filenames):
    """Idempotence: every file exists and matches its pin."""
    want = expected_hashes(corpus_id)
    if not want:
        return False
    for name in filenames:
        p = os.path.join(DATA_DIR, corpus_id, name)
        if not os.path.exists(p) or sha256_file(p) != want.get(name):
            return False
    return True


def die(msg):
    print(f"FETCH FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


# ---- dialogue corpora (chat_sft.py, dialogue_pt.py) ------------------------
# The chat template's markers; a source dialogue containing one would corrupt
# the template it is rendered into, so it is dropped (reason "markers").
MARKERS = ("<|user|>", "<|assistant|>", "<|endoftext|>")
# System messages that may be stripped: empty or generic text that sets no
# instruction the replies could depend on. Any other system message drops the
# dialogue (reason "system_instruction"): its replies may obey an instruction
# the rendered text would no longer contain.
BOILERPLATE_SYSTEM = frozenset({
    "",
    "you are a helpful assistant.",
    "you are a helpful assistant",
    "you are a helpful ai assistant.",
    "you are a helpful ai assistant",
    "you are an ai assistant.",
    "you are an ai assistant",
})


def parquet_rows(repo, rev, files, columns):
    """(uri, row dict) for every row of the given parquet files of a pinned dataset revision."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    for fn in files:
        f = pq.ParquetFile(hf_hub_download(repo, fn, repo_type="dataset", revision=rev))
        i = 0
        for batch in f.iter_batches(batch_size=2048, columns=columns):
            for row in batch.to_pylist():
                yield f"{repo}@{rev[:8]}/{fn}#{i}", row
                i += 1


def normalise_dialogue(msgs):
    """(turns, reason). turns: alternating user/assistant messages from a user turn, ending
    on an assistant turn (a trailing unanswered user turn is trimmed), or None with the
    reason: "system_instruction", "markers", "empty_turn", "non_alternating", "too_short".
    A boilerplate system message (BOILERPLATE_SYSTEM) is stripped; reason "system_stripped"
    then accompanies kept turns, "kept" otherwise."""
    stripped = False
    out = []
    for m in msgs:
        role, text = m["role"], str(m["content"] or "").strip()
        if role == "system":
            if out or text.lower() not in BOILERPLATE_SYSTEM:
                return None, "system_instruction"
            stripped = True
            continue
        if role not in ("user", "assistant"):
            return None, "non_alternating"
        if any(k in text for k in MARKERS):
            return None, "markers"
        if not text:
            return None, "empty_turn"
        if role != ("user" if len(out) % 2 == 0 else "assistant"):
            return None, "non_alternating"
        out.append({"role": role, "content": text})
    if out and out[-1]["role"] == "user":
        out.pop()
    if len(out) < 2:
        return None, "too_short"
    return out, "system_stripped" if stripped else "kept"


def render_dialogue(turns):
    """The chat template (LOCAL_LLM_STACK.md section 3, point 3) for a whole dialogue, one
    document: <|user|>U<|assistant|>A<|endoftext|> per exchange, then a newline."""
    return "".join(f"<|user|>{turns[k]['content']}<|assistant|>{turns[k + 1]['content']}<|endoftext|>"
                   for k in range(0, len(turns) - 1, 2)) + "\n"


def dialogue_key(turns):
    """Duplicate key: sha256 of the dialogue's lowercased words, so whitespace, case and
    punctuation differences do not hide a duplicate."""
    import re
    text = " ".join(w for w in re.split(r"[^0-9a-z]+", " ".join(t["content"] for t in turns).lower()) if w)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
