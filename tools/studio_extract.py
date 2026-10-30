#!/usr/bin/env python3
"""Studio's explicitly budgeted LLM extraction worker; no credential discovery."""
from __future__ import annotations
import argparse
import json
import math
import os
import re
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers"))
import llm_extract as lx


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(value, out, indent=2, ensure_ascii=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def extract(arxiv_id, output, budget, tag, *, source_loader=None, extractor=None, client_factory=None):
    if not isinstance(arxiv_id, str) or len(arxiv_id) > 32 or not re.fullmatch(
            r"(?:\d{4}\.\d{4,5}(?:v\d+)?|[a-z][a-z-]{1,12}(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", arxiv_id):
        raise ValueError("A bare arXiv identifier is required")
    if not math.isfinite(budget) or not 0 < budget <= 5:
        raise ValueError("Extraction spend cap must be finite, positive and at most $5")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY is not configured in the Studio server environment")
    tex = (source_loader or lx.get_tex)(arxiv_id)
    result = (extractor or lx.run_extraction)(
        arxiv_id, tex, (client_factory or lx.make_client)(), tag, budget,
        lx.MODEL, lx.EFFORT, lx.MAX_TOKENS, True)
    # Rebuild the handoff from the actual raw response and current source. Do
    # not trust an asserted field or cached verification record supplied by it.
    sent = lx.prepare_source(tex)
    raw = result["llm"]["output"]
    verified = lx.verify(raw, sent)
    source = dict(result.get("source") or {})
    source.update(tex_sha256=lx.sha256_text(tex), sent_sha256=lx.sha256_text(sent))
    result = lx.to_extraction(arxiv_id, raw, verified, source)
    atomic_json(output, result)
    return result


def watch_parent(parent_pid):
    """Stop this worker if its owning Studio process exits, including crashes."""
    import threading
    import time

    def watch():
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x00100000, False, parent_pid)  # SYNCHRONIZE only
            if not handle:
                os._exit(1)
            try:
                # WAIT_OBJECT_0 means the exact owning process has exited.
                kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
                os._exit(1)  # owner exited or the ownership wait itself failed
            finally:
                kernel.CloseHandle(handle)
        else:
            while os.getppid() == parent_pid:
                time.sleep(0.2)
            os._exit(1)
    threading.Thread(target=watch, daemon=True, name="studio-parent-watch").start()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arxiv_id")
    parser.add_argument("--out", required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args(argv)
    if args.parent_pid is not None:
        if args.parent_pid <= 0:
            parser.error("--parent-pid must be positive")
        watch_parent(args.parent_pid)
    try:
        extract(args.arxiv_id, args.out, args.budget_usd, args.tag)
    except (Exception, SystemExit) as exc:
        # Provider errors must never echo the explicitly inherited credential.
        message = str(exc)
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key:
            message = message.replace(key, "[redacted]")
        atomic_json(args.out + ".error.json", {"error": message or type(exc).__name__})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
