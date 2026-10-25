"""spec/ is keramos at the pinned tag (spec/KERAMOS_PIN.json; tools/keramos_pin.py, tools/sync_keramos.sh):
no vendored file is edited, missing or added, and, where a keramos checkout is at hand (KERAMOS_DIR or
the sibling ../keramos) with the pinned commit, the pin's manifest is that commit's and its tag names it.
Standard library only."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import keramos_pin as kp  # noqa: E402

PIN = kp.load_pin()
SRC = Path(os.environ.get("KERAMOS_DIR") or ROOT.parent / "keramos")


def _has_commit(src, commit):
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(src), "cat-file", "-e", commit + "^{commit}"],
                              capture_output=True).returncode == 0
    except OSError:
        return False


def test_pin_names_a_tag_and_a_full_commit():
    assert PIN["tag"].startswith("v") and len(PIN["tag"].split(".")) == 3
    assert len(PIN["commit"]) == 40 and int(PIN["commit"], 16) >= 0
    assert "IR_SPEC.md" in PIN["files"] and "ir.schema.json" in PIN["files"]


def test_vendored_tree_is_the_pinned_commit():
    msgs = kp.differences(PIN)
    assert msgs == [], "spec/ differs from keramos " + PIN["tag"] + " (edit keramos, tag, sync):\n" + "\n".join(msgs)


def test_an_edit_is_detected(tmp_path):
    pin = {"tag": "v0.0.0", "files": {"a.md": kp.blob_sha1(ROOT / "spec" / "IR_SPEC.md")}}
    (tmp_path / "a.md").write_bytes((ROOT / "spec" / "IR_SPEC.md").read_bytes().replace(b"\r\n", b"\n"))
    assert kp.differences(pin, tmp_path) == []
    (tmp_path / "a.md").write_bytes(b"edited\n")
    (tmp_path / "b.md").write_bytes(b"new\n")
    assert kp.differences(pin, tmp_path) == ["not in keramos v0.0.0: b.md", "edited: a.md"]


@pytest.mark.skipif(not _has_commit(SRC, PIN["commit"]), reason="no keramos checkout with the pinned commit")
def test_manifest_and_tag_match_keramos():
    assert kp.manifest(SRC, PIN["commit"]) == PIN["files"]
    assert kp.tag_commit(SRC, PIN["tag"]) == PIN["commit"]
