#!/usr/bin/env python3
"""A by-hand check, on the live Colab CLI, of the listing forms runner_multi.py takes as an answer. Run it once during
the build, with the build's session up, before the arms start:

    python3 tools/colab_chat/base2_listing_check.py pk-b2-data

It runs `colab sessions` and `colab stop -s` on a made-up name that has no session, and prints one JSON object of
numbers and booleans only (no line of the listing, no name, no endpoint):
    sessions_rc            the listing's exit code
    own_line               a line for NAME is in the listing
    own_line_in_form       every such line matches runner_multi.COLAB_SESSION_LINE ("[name] endpoint | Hardware: ")
    other_lines_in_form    the number of lines in that form
    empty_line             the CLI's exact empty-listing line is there
    answered               runner_multi.ColabTransport.answered() on that listing
    listed                 listed() gives NAME an endpoint
    absent                 absent() for NAME
    absent_made_up         absent() for the made-up name
    stop_made_up_rc        `colab stop -s MADE-UP`'s exit code
    stop_made_up_not_found it said the session was not found
    agree                  all of it is what the runner needs: answered, listed and absent agree with the line
                           for NAME, the made-up name is absent, and its stop changed nothing
Exit 0 when agree, else 1.

Why the stop is safe: the CLI's `stop` with -s looks the name up in its local session store only; a name that is not
there prints "[colab] Session '<name>' not found." and returns, exit 0, without a call to the service
(commands/session.py, read on 12 Nov 2026). The made-up name is NAME plus "-nosuch-" and eight random hex digits.
Without -s the CLI would stop a unique session, so this script always passes -s.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import runner_multi as rm  # noqa: E402  the forms and the transport the runner itself uses


def check(name, colab=None):
    if colab:
        rm.COL = colab
    rm.NAME = name
    t = rm.ColabTransport()
    rc, out = rm.colab("sessions", timeout=120)
    own = [ln for ln in out.splitlines() if ln.startswith(f"[{name}]")]
    res = {"sessions_rc": rc, "own_line": bool(own),
           "own_line_in_form": bool(own) and all(re.match(rm.COLAB_SESSION_LINE, ln) for ln in own),
           "other_lines_in_form": len(re.findall(rm.COLAB_SESSION_LINE, out, re.M)),
           "empty_line": bool(re.search(rm.COLAB_EMPTY_LISTING, out, re.M)),
           "answered": t.answered((rc, out))}
    res["listed"] = t.listed() is not None
    res["absent"] = t.absent()
    made_up = f"{name}-nosuch-{uuid.uuid4().hex[:8]}"
    rm.NAME = made_up
    res["absent_made_up"] = t.absent()
    src, sout = rm.colab("stop", "-s", made_up, timeout=120)
    rm.NAME = name
    res["stop_made_up_rc"] = src
    res["stop_made_up_not_found"] = "not found" in sout
    res["agree"] = (res["answered"] and res["listed"] == (res["own_line"] and res["own_line_in_form"])
                    and res["absent"] == (not res["own_line"]) and res["absent_made_up"]
                    and src == 0 and res["stop_made_up_not_found"])
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
