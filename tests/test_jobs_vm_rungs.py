"""tools/colab_chat/jobs_vm.py's rung and successive-halving features (experiments/tiny_chat_2026-10/
PREREGISTRATION.md amendment 1), with a stand-in trainer: snapshots at the snapshot steps, a light
rung job running beside a training job that fills the main pool, a rung that can never come (the
run ended before it) ending as lost, promotion decisions holding, running or failing candidates,
and after_terminal against a normal "after". POSIX only: the manager reads /proc, as on the VM.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOBS_VM = ROOT / "tools" / "colab_chat" / "jobs_vm.py"
pytestmark = pytest.mark.skipif(os.name == "nt", reason="jobs_vm.py runs on the Colab VM (Linux /proc)")

TRAINER = """#!{py}
# Stand-in for `mtstudio run SPEC`: a checkpoint (model.safetensors, then state.txt) every step.
import json, os, sys, time
spec = json.load(open(sys.argv[2]))
out = spec["out_dir"]
os.makedirs(out, exist_ok=True)
for step in range(1, spec["train"]["steps"] + 1):
    time.sleep(0.6)
    open(os.path.join(out, "model.safetensors"), "w").write("weights at %d" % step)
    open(os.path.join(out, "state.txt"), "w").write("%d\\n{{}}\\n" % step)
json.dump({{"done": True}}, open(os.path.join(out, "result.json"), "w"))
"""

READER = """import os, sys
# Copies the snapshot's state.txt beside the training job's current step into this job's output.
snap, live, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
step = open(os.path.join(live, "state.txt")).readline().strip()
open(os.path.join(out, "seen.txt"), "w").write(open(snap).read().split()[0] + " " + step)
"""

WRITER = """import json, os, sys
os.makedirs(sys.argv[1], exist_ok=True)
json.dump({"decisions": {"r1": {"a": True, "b": False}}}, open(os.path.join(sys.argv[1], "decisions.json"), "w"))
"""


def test_rungs_promotions_and_terminal_after(tmp_path):
    chat, repo = tmp_path / "chat", tmp_path / "repo"
    chat.mkdir()
    repo.mkdir()
    trainer = tmp_path / "trainer"
    trainer.write_text(TRAINER.format(py=sys.executable))
    trainer.chmod(0o755)
    (repo / "reader.py").write_text(READER)
    (repo / "writer.py").write_text(WRITER)
    (repo / "touch.py").write_text("import os, sys\nos.makedirs(sys.argv[1], exist_ok=True)\n"
                                   "open(os.path.join(sys.argv[1], 'ok'), 'w').write('ok')\n")
    py = sys.executable
    pt = f"{chat}/pt"
    cfg = {"max_parallel": 1, "max_parallel_light": 1, "jobs": [
        {"name": "pt", "snapshot_steps": [2, 4], "spec": {"train": {"steps": 6}}},
        {"name": "rung_2", "light": True, "rung_of": {"job": "pt", "step": 2},
         "cmd": [py, "reader.py", f"{pt}/rung_2/state.txt", pt, "{out}"]},
        {"name": "rung_9", "light": True, "rung_of": {"job": "pt", "step": 9}, "cmd": [py, "touch.py", "{out}"]},
        {"name": "promo", "light": True, "cmd": [py, "writer.py", "{out}"]},
        {"name": "cand_a", "after": "promo", "cmd": [py, "touch.py", "{out}"],
         "promoted_by": {"files": [f"{chat}/promo/decisions.json"], "round": "r1", "key": "a"}},
        {"name": "cand_b", "after": "promo", "cmd": [py, "touch.py", "{out}"],
         "promoted_by": {"files": [f"{tmp_path}/absent.json", f"{chat}/promo/decisions.json"], "round": "r1",
                         "key": "b"}},
        {"name": "term", "light": True, "after_terminal": True, "after": ["cand_b", "rung_9"],
         "cmd": [py, "touch.py", "{out}"]},
        {"name": "casc", "after": ["cand_b"], "cmd": [py, "touch.py", "{out}"]}]}
    jf = tmp_path / "jobs.json"
    jf.write_text(json.dumps(cfg))
    env = {**os.environ, "JOBS_CHAT_DIR": str(chat), "JOBS_REPO": str(repo), "JOBS_MTSTUDIO": str(trainer),
           "JOBS_TICK": "0.1"}
    p = subprocess.run([py, str(JOBS_VM), str(jf)], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    status = {n: (chat / f"STATUS_{n}").read_text().strip() for n in
              ("pt", "rung_2", "rung_9", "promo", "cand_a", "cand_b", "term", "casc")}
    assert status == {"pt": "FINISHED", "rung_2": "FINISHED", "rung_9": "FAILED", "promo": "FINISHED",
                      "cand_a": "FINISHED", "cand_b": "FAILED", "term": "FINISHED", "casc": "FAILED"}, status
    # The snapshots hold their own steps, and the light rung job ran while the training job (the whole
    # main pool) was still going.
    assert (chat / "pt" / "rung_2" / "model.safetensors").read_text() == "weights at 2"
    assert (chat / "pt" / "rung_4" / "state.txt").read_text().split()[0] == "4"
    snap_step, live_step = map(int, (chat / "rung_2" / "seen.txt").read_text().split())
    assert snap_step == 2 and live_step < 6
    assert "rung lost" in (chat / "rung_9_train.log").read_text()
    assert "not promoted" in (chat / "cand_b_train.log").read_text()
    assert (chat / "cand_a" / "ok").exists() and not (chat / "casc" / "ok").exists()
