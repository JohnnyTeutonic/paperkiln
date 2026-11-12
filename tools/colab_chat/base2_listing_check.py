#!/usr/bin/env python3
"""A by-hand check, on the live Colab CLI, of the listing form runner_multi.py takes as an answer. Run it once during
the build, with the build's session up, before the arms start:

    cd /home/jonat/claude_base2_fix3 && python3 tools/colab_chat/base2_listing_check.py pk-b2-data

It takes ONE `colab sessions` listing and judges everything from that one listing (runner_multi.ColabTransport's
answered(), listed() and absent(), each fed the same output). It makes no stop of any kind. It prints one JSON object:
    cli_version        what `colab version` prints after "Version:" (a string; "" when it does not answer)
    sessions_rc        the listing's exit code
    own_line           a line for NAME is in the listing
    own_line_in_form   every such line matches runner_multi.COLAB_SESSION_LINE ("[name] endpoint | Hardware: ")
    own_endpoint       listed() gives NAME an endpoint from that line
    other_lines_in_form the number of lines in that form
    empty_line         the CLI's exact empty-listing line is there
    answered, listed, absent   the transport's three readings of that one listing
    agree              the build's own session line is there, in form, with an endpoint, and answered is true,
                       listed is true and absent is false
Exit 0 only when agree; else 1 (also with no session up: the form was then not seen).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import runner_multi as rm  # noqa: E402  the forms and the transport the runner itself uses


def check(name, colab=None):
    if colab:
        rm.COL = colab
    rm.NAME = name
    vrc, vout = rm.colab("version", timeout=60)
    m = re.search(r"Version:\s*(\S+)", vout or "")
    rc, out = rm.colab("sessions", timeout=120)
    one = (rc, out)
    real = rm.colab
    rm.colab = lambda *a, timeout=240: one   # listed() and absent() read this same listing
    try:
        t = rm.ColabTransport()
        endpoint = t.listed()
        absent = t.absent()
        answered = t.answered(one)
    finally:
        rm.colab = real
    own = [ln for ln in out.splitlines() if ln.startswith(f"[{name}]")]
    res = {"cli_version": m.group(1) if (vrc == 0 and m) else "", "sessions_rc": rc, "own_line": bool(own),
           "own_line_in_form": bool(own) and all(re.match(rm.COLAB_SESSION_LINE, ln) for ln in own),
           "own_endpoint": endpoint is not None,
           "other_lines_in_form": len(re.findall(rm.COLAB_SESSION_LINE, out, re.M)),
           "empty_line": bool(re.search(rm.COLAB_EMPTY_LISTING, out, re.M)),
           "answered": answered, "listed": endpoint is not None, "absent": absent}
    res["agree"] = (res["own_line"] and res["own_line_in_form"] and res["own_endpoint"] and answered
                    and res["listed"] and not absent)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name", help="the session name to look for (the build's: pk-b2-data)")
    ap.add_argument("--colab", default=None, help="the colab CLI (default: runner_multi's)")
    a = ap.parse_args(argv)
    res = check(a.name, a.colab)
    print(json.dumps(res))
    return 0 if res["agree"] else 1


if __name__ == "__main__":
    sys.exit(main())
