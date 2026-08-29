# -*- coding: utf-8 -*-
"""paperkiln-fetch: arXiv id in, evidence-carrying architecture out.

Public surface re-exported from the vendored extractor (source of
truth: papers/fetch.py in the paperkiln repo; sync-checked by
tools/sync_fetch_pkg.py there).
"""
from ._fetch import Arch, Finding, extract, fetch_source, to_json  # noqa: F401
from .hf_emit import build_hf_config  # noqa: F401

__version__ = "0.1.0"
__all__ = ["Arch", "Finding", "extract", "fetch_source", "to_json",
           "build_hf_config", "__version__"]
