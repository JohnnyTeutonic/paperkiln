"""Print the compile flags coalfire's own build gave one source file, from its compile_commands.json, as one line of
shell words: everything except the compiler, -o and its argument, -c and the source itself. The dump tool is compiled
with these, so that it sees the headers exactly as the library did (USE_CUDA and friends change class layouts).

  python3 dump_flags.py BUILD/compile_commands.json src/transformer.cpp

Exit 1, printing nothing, if the file has no entry or its entry has no command.
"""
import json
import shlex
import sys
from pathlib import Path


def flags(db_path, source):
    for e in json.loads(Path(db_path).read_text()):
        if not Path(e.get("file", "")).as_posix().endswith("/" + source.strip("/")):
            continue
        words = e["arguments"] if "arguments" in e else shlex.split(e.get("command", ""))
        if not words:
            return None
        out, skip = [], False
        for w in words[1:]:
            if skip:
                skip = False
                continue
            if w == "-o":
                skip = True
                continue
            if w == "-c" or w == e["file"] or Path(w).as_posix().endswith("/" + source.strip("/")):
                continue
            out.append(w)
        return out
    return None


def main():
    f = flags(sys.argv[1], sys.argv[2])
    if not f:
        return 1
    print(" ".join(shlex.quote(w) for w in f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
