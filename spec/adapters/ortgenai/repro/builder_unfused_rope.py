# main's builder.py with a separate RotaryEmbedding node when head_size % 16 != 0 (and unpacked Q/K/V).
import os
import runpy
import sys

MB = os.path.expanduser("~/issues_work/ortgenai_main/src/python/py/models/builder.py")
sys.path.insert(0, os.path.dirname(MB))
from builders.base import Model  # noqa: E402

orig = Model.is_fused_rope_supported
Model.is_fused_rope_supported = lambda self: orig(self) and self.head_size % 16 == 0
sys.argv = [MB] + sys.argv[1:] + ["--extra_options", "fuse_qkv=false"]
runpy.run_path(MB, run_name="__main__")
