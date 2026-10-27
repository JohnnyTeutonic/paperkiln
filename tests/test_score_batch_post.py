"""tiny_chat_run.py post --score-batch-tokens (opt-in, LOCAL_LLM_WORK E5.2): 0, the default, leaves Carry-5's
argv as every registered arm runs it; N adds carry5_eval's --score-batch-tokens N where mtstudio scores."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "colab_chat"))
import tiny_chat_run as tcr  # noqa: E402


def test_default_argv_unchanged_and_opt_in(monkeypatch):
    base = tcr.carry5_argv("s.json", "items.jsonl", "out", "mtstudio", 8100)
    assert "--score-batch-tokens" not in base
    monkeypatch.setattr(tcr, "SCORE_BATCH_TOKENS", 16384)
    on = tcr.carry5_argv("s.json", "items.jsonl", "out", "mtstudio", 8100)
    assert on[:len(base)] == base and on[len(base):] == ["--score-batch-tokens", "16384"]
    assert "--score-batch-tokens" not in tcr.carry5_argv("s.json", "i", "o", "mtstudio", 1, scorer="none")


def test_post_takes_the_flag():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "colab_chat" / "tiny_chat_run.py"), "post", "--help"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "--score-batch-tokens" in r.stdout
