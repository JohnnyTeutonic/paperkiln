# main's builder.py with Gemma3Model's local RoPE caches built without the global layers' linear factor.
import os
import runpy
import sys

MB = os.path.expanduser("~/issues_work/ortgenai_main/src/python/py/models/builder.py")
sys.path.insert(0, os.path.dirname(MB))
from builders.gemma import Gemma3Model  # noqa: E402


def multi_cache(self):
    self.cos_cache_global_name, self.sin_cache_global_name = "cos_cache_global", "sin_cache_global"
    super(Gemma3Model, self).make_rotary_embedding_caches(
        cos_cache_name=self.cos_cache_global_name, sin_cache_name=self.sin_cache_global_name)
    self.rope_attrs["create_caches"] = True
    self.rope_attrs["theta"] = self.rope_local_theta
    self.rope_attrs["rescale_factors"] = 1          # the fix
    self.cos_cache_local_name, self.sin_cache_local_name = "cos_cache_local", "sin_cache_local"
    super(Gemma3Model, self).make_rotary_embedding_caches(
        cos_cache_name=self.cos_cache_local_name, sin_cache_name=self.sin_cache_local_name)


Gemma3Model.make_rotary_embedding_multi_cache = multi_cache
sys.argv = [MB] + sys.argv[1:]
runpy.run_path(MB, run_name="__main__")
