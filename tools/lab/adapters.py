"""The only door to frozen code. Claude Opus 5.5, 7 Nov 2026.

mixed_gen.py, mine_score.py, family_exposure.py and analytic_rubric_v2.py live in the workspace root's AI_ML/chat_screens_claude, are import-only, and are not copied here. Each load checks the file's hash against a pin and fails closed,
so a changed scorer cannot slip under a registered read. LAB_SCREENS_DIR names the folder. Nothing else in the lab imports from it, and the core runs without it.
"""
from __future__ import annotations

import ast
import os
import sys
import types
from pathlib import Path

from schema import sha_file

SCREENS_ENV = "LAB_SCREENS_DIR"
DEFAULT = "/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/AI_ML/chat_screens_claude"
PINS = {
    "analysis/correction_base_receipts/mine_score.py": "989c1bcb5ee52b0badbb6304f839307a443a983f5f9a1f6274959e6a58d4e722",
    "family_exposure.py": "7e22c598694da6b9f8d4a1819f76dd5b46cc7c3f43f24a2a7d963be241584340",
    "mixed_gen.py": "71a98de13376af597c8cae27c2696808286ba0b814d305758ed086bab561872e",
    "correction_split.py": "da785f1c9bf880fdd274de588fd31041273e1d0711a4c89a6b4de3f8fd940ff2",
    "analytic_rubric_v2.py": "3519e8b7e0dc0d4081db18b8c07176ebc9598eef43bef1eb21cec10a6d6cfeb1",
}


class FrozenChanged(Exception):
    pass


def screens_dir():
    d = Path(os.environ.get(SCREENS_ENV, DEFAULT))
    return d if d.is_dir() else None


def verify_pins(root=None):
    root = Path(root) if root else screens_dir()
    if root is None:
        raise FrozenChanged("no screens folder")
    out = {}
    for rel, pin in PINS.items():
        got = sha_file(root / rel)
        if got != pin:
            raise FrozenChanged(f"{rel} has changed")
        out[rel] = got[:12]
    return out


def load_frozen(rel, root=None):
    """Runs the file's definitions, and leaves out a bare top-level main() call, which mine_score.py has: importing must not start its analysis. The file itself is unchanged and its hash pinned.
    The folder goes at the end of sys.path, so its modules (it has its own scenarios.py) never shadow the lab's."""
    root = Path(root) if root else screens_dir()
    if root is None:
        raise FrozenChanged("no screens folder")
    path = root / rel
    if rel not in PINS or sha_file(path) != PINS[rel]:
        raise FrozenChanged(f"{rel} is not at its pin")
    if str(root) not in sys.path:
        sys.path.append(str(root))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and getattr(n.value.func, "id", "") == "main")]
    mod = types.ModuleType("frozen_" + path.stem)
    mod.__file__ = str(path)
    sys.modules[mod.__name__] = mod
    exec(compile(tree, str(path), "exec"), mod.__dict__)
    return mod


def mine_score():
    return load_frozen("analysis/correction_base_receipts/mine_score.py")


def leg_counts(packet_path, gguf_path, seed, steps):
    """family_exposure's realised recount for one packet under the trainer's draw rule. An absolute packet path passes through its Path join, so the module is used unchanged. Needs numpy, through the frozen module's own imports."""
    fe = load_frozen("family_exposure.py")
    enc, _, _ = fe.N.tokeniser(str(gguf_path))
    return fe.leg_counts(fe.Packet(str(packet_path), enc), seed, steps)


def to_exposure(counts):
    """family_exposure's fields into the ledger's exposure record."""
    return {f: {"draws": v["draws"], "sampled_supervised": v["sampled_supervised"], "sampled_decision": v["sampled_decision"], "sampled_decision_lr": v["sampled_decision_lr"], "distinct_scenarios": v["scenarios_drawn"]} for f, v in counts.items()}
