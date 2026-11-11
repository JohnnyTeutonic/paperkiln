"""Tests only (tests/hub_standin.py). A test puts this folder on PYTHONPATH and sets HUB_STANDIN_ROOT; every python
process started under that environment then has huggingface_hub.HfApi replaced by the on-disk stand-in, so a shell
script's own calls to hf_store.py and base2_corpus.py go through the real hf_store.HfBackend and reach no network."""
import os
import sys

if os.environ.get("HUB_STANDIN_ROOT"):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import huggingface_hub

    import hub_standin

    huggingface_hub.HfApi = hub_standin.DiskApi
