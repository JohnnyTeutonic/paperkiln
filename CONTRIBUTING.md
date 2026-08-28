# Contributing to microtorch

microtorch is an educational-yet-research-capable LLM framework: every major
component of the modern transformer training stack, in readable C++. The bar
for contributions follows from that: readable first, receipted second, fast
third.

## Building

```bash
git clone https://github.com/JohnnyTeutonic/microtorch
cd microtorch && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
ctest --output-on-failure
```

The trainer core (`transformer_core`) resolves from a sibling
`../transformer_cpp` checkout if present, otherwise from the vendored copy in
`third_party/transformer_cpp` (refresh with `tools/sync_vendor.sh`). A clone
builds with no siblings.

`-DMICROTORCH_CUDA=ON` routes `device::matmul` through the CUDA kernel tree
(needs the CUDA toolkit; validated on T4).

## The receipt discipline

Every trainable component and every tunable knob in this repo carries a
receipt: a gradient check against finite differences, a wired-smoke run where
the loss falls, or a deterministic probe that proves the code path binds.
Contributions follow the same rule:

- **New ops**: add a finite-difference gradcheck to `tests/test_gradcheck.cpp`.
  Use `tests/check.hpp` `CHECK()` (plain `assert()` vanishes under Release).
- **New modules/mechanisms**: module-level FD gradcheck plus a short training
  run showing loss falls (steps and numbers in the PR description).
- **New knobs**: an A/B or a deterministic probe demonstrating the knob binds
  (see `swa_check` in transformer_cpp for the pattern). A knob that parses but
  does not bind is worse than no knob.
- Research claims (docs/SPARSE_ATTENTION.md territory) additionally follow
  falsifiers-first: state what result would kill the idea before running it.

## Tests

- `ctest` runs the full suite; individual binaries live in `build/`
  (`test_gradcheck`, `test_llama`, `test_nn`, ...).
- Tests that write then reopen files must write to `$TMPDIR`/`/tmp`, never the
  working directory.

## Style

- `clang-format` 18 with the repo `.clang-format`; CI enforces it
  (`clang-format --dry-run -Werror`). Format before committing:
  `clang-format -i <files>`.
- Comments state constraints the code cannot show; they are not narration.
- Parameter names follow Hugging Face conventions where a mapping exists
  (`layers.N.self_attn.q_proj.weight`), so exporters need no rename tables.

## Commit hygiene

- One logical change per commit; subject line says what the tree can now do.
- Whitespace-only commits are listed in `.git-blame-ignore-revs`.
