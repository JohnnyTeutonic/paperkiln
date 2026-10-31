"""Compatibility entry point for the frozen, isolated S-full V2 finalizer.

The integrated runner has additional DPO support and is NOT the runner whose
bytes the S-full source freeze attests. Execute the reviewed isolated finalizer
instead; its original absolute anchors remain authoritative. This entry point
is not evidence that the integrated checkout equals that historical freeze.
"""

from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path

FROZEN_TARGET = Path(
    "/home/jonat/pk_sfull_operation_v2_20261101/tools/colab_chat/finalize_sfull_operational_v2.py"
)
FROZEN_TARGET_SHA256 = (
    "7bc4ecbc2345efa8cfa01b1b8095bcb0d34aa3599c825f095b0d99a93331a449"
)
if hashlib.sha256(FROZEN_TARGET.read_bytes()).hexdigest() != FROZEN_TARGET_SHA256:
    raise ValueError("isolated S-full finalizer source differs")
_spec = importlib.util.spec_from_file_location(
    "isolated_sfull_operational_v2", FROZEN_TARGET
)
_delegate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_delegate)
for _name in dir(_delegate):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_delegate, _name)


if __name__ == "__main__":
    raise SystemExit(_delegate.main())
