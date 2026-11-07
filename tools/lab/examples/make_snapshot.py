"""Writes history_observational_v1.json from the workspace root's stored analysis files. Numbers only. Claude Opus 5.5, 7 Nov 2026."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import adapters as AD  # noqa: E402
import history as H  # noqa: E402

fe = AD.load_frozen("family_exposure.py")
snap = H.snapshot(AD.screens_dir(), fe)
HERE.joinpath("history_observational_v1.json").write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
print(len(snap["families"]), "families")
