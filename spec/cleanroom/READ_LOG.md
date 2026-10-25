# Clean-room read log

Every file opened while writing the clean-room implementation (cleanroom/rust/). Directory
listings (names only) of keramos/, conformance/ and examples/ were also taken. Nothing under
tools/, tests/, microtorch/ or ember.cpp/ was opened.

- keramos/README.md
- keramos/conformance/README.md
- keramos/.gitignore
- keramos/conformance/manifest.json
- keramos/conformance/expected_fail.json
- keramos/conformance/receipts/index.json (pass/fail/skip counts only; the per-engine receipts were not opened)
- keramos/IR_SPEC.md (all of it)
- keramos/ir.schema.json
- keramos/conformance/ops/*.json (every op fixture: nodes, shapes, classes)
- keramos/conformance/models/*/case.json (structure; pk_llama in full)
- keramos/conformance/models/*/*.gguf (metadata, tensor tables and the embedded documents, read with my
  own GGUF reader; the .safetensors and mtstudio.json files were not opened)
- keramos/conformance/state/*.json (step and expectation structure)
- keramos/conformance/refuse/*.json and refuse/*.gguf (documents, rule numbers, GGUF metadata)
- keramos/conformance/digest/vectors.json
- keramos/conformance/quant/blocks.json
- keramos/conformance/grad/recurrence.json (request shapes only; gradients not implemented)
- keramos/conformance/recurrence/*.ir.json
- The runner `tools/ir_conformance.py` was run as a black box (`--help`, and against adapters); its
  source was not opened. A logging stub adapter recorded the request formats it sends.
- keramos/examples/*.ir.json (validated and digested by the clean-room engine; read as data)
- keramos/IR_SPEC.md diff of commit e3b77ec (v1.1.4, section 4.6), which landed during this work
