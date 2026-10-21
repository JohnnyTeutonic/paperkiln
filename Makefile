# Convenience targets. The C++ build is CMake (see README); this file only
# fetches data. `make corpus` reproduces, byte for byte and hash-verified,
# every corpus and vocabulary the repository's numbers were produced on.
PYTHON ?= python3

.PHONY: corpus corpus-test

corpus:
	$(PYTHON) corpus/fetchers/tinystories.py all
	$(PYTHON) corpus/fetchers/tinychat.py all
	$(PYTHON) corpus/fetchers/vocab_chat7b.py
	$(PYTHON) corpus/fetchers/vocab_tinystories_valid.py
	$(PYTHON) corpus/fetchers/tinyarith.py
	$(PYTHON) corpus/fetchers/eval_heldout.py

corpus-test:
	$(PYTHON) -m pytest -q tests/test_corpus_quality.py tests/test_tinyarith.py tests/test_eval_frozen.py tests/test_decontam.py \
	    tests/test_dialogue_pt.py tests/test_mix_pretrain.py
