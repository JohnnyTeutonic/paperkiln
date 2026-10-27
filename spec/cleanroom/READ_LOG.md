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

## Revision 1.3 and 1.4 pass (27 October 2026)

Every file opened while bringing `cleanroom/rust/` from revision 1.2 to revision 1.4 (keramos v1.4.1,
working tree at v1.4.1-2-g9b6d5c6). Nothing under `tools/`, `tests/` or `adapters/`, no per-engine receipt,
no `.safetensors` or `mtstudio.json`, nothing of paperkiln or ember.cpp and not `LOCAL_LLM_STACK.md` was
opened or searched.

- keramos/IR_SPEC.md (all of it, revision 1.4, v1.4.1)
- keramos/ir.schema.json (`op_attention` and `op_linear_attention` printed; the whole schema is compiled
  into the engine, as before)
- keramos/conformance/README.md
- keramos/conformance/expected_fail.json (the `keramos-cleanroom` entries)
- keramos/conformance/receipts/index.json (counts only: the `cleanroom.json` and `ir_numpy.json` runs)
- keramos/conformance/ops/attention.json (the `position_bias*` and `relative_kv*` cases: nodes and the
  shapes of inputs, weights and expected values)
- keramos/conformance/refuse/position_bias_*.json, relative_kv_*.json, architecture_*.json (documents and
  rule numbers), and the four refuse/architecture_*.gguf (metadata, tensor tables and embedded documents,
  read with a GGUF reader written for this pass, scratch only)
- keramos/conformance/models/pk_flex_t5/pk_flex_t5.gguf and pk_flex_shaw/pk_flex_shaw.gguf (metadata,
  tensor tables, embedded documents, same reader), and `general.architecture` of every models/*/*.gguf
- keramos/cleanroom/READ_LOG.md, FINDINGS.md, rust/Cargo.toml and rust/src (schema.rs, validate.rs,
  model.rs, ops.rs, main.rs, tensor.rs, json.rs; gguf.rs and session.rs searched)
- Git: `git log --oneline -3 -- IR_SPEC.md` and `git describe --tags` (no history of `tools/`)
- The runner `tools/ir_conformance.py` was run as a black box (`--help`, and the whole suite with
  `--adapter cmd:... --json`) from a copy of the tree on WSL ext4; its source was not opened.

## Phase 2 (v1.4.2 text, 28 October 2026)

Work in the scratch clone of keramos (`.../scratchpad/e7/keramos`), whose `IR_SPEC.md` holds the v1.4.2
text uncommitted. Opened or run, in addition to the earlier pass's files:

- `git diff HEAD -- IR_SPEC.md` in the clone (every changed line of the text, Appendix D whole: D.1 to D.7,
  3.1's architecture and GGUF paragraphs, rules 38 to 40, 3.5, 5.15's Q16 bullet, 6.9, 7.5, 8.1.1, 8.1.3,
  8.1.4, 8.1.6, 11.1 to 11.3, 12.5, 12.6), and `git status --short` there
- keramos/conformance/chat/render.json as data: the records, each case's name, record and keys; the inputs
  of the 39 refused cases were sent to the engine by a driver script that printed only the engine's own
  refusal messages. No case's expected text was printed or read.
- keramos/conformance/chat/parse_reply.json as data: its comment, each case's name and keys
- keramos/cleanroom/rust/src: session.rs (`run_block`, `truncate`), json.rs and gguf.rs (types and accessors)
- The runner, as a black box: `--only refuse/architecture`, `--only mor_window`, `--only accept/`,
  `--only chat/` and the whole suite, from the WSL copy `~/e7_keramos` of the clone
- `kcr check` on the four rule 38 refusal GGUFs (the engine's own messages)
- `scripts/cpu_budget.py list`

No embedded document of a `tp_` model was opened in this phase: Appendix D's graphs were written from
the appendix alone, and the acceptance cases checked them. Nothing under `tools/`, `tests/` or
`adapters/`, no per-engine receipt, no `.safetensors` or `mtstudio.json`, nothing of paperkiln or ember and
not `LOCAL_LLM_STACK.md` was opened or searched (`git status` lists modified `tools/` files in the clone;
their diffs were not opened).
