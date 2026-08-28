#pragma once
// Phase 1c: the "torch.load story" docs/DESIGN.md section 3 chose -- safetensors,
// not pickle. Format: 8-byte LE header length, JSON header
// {name: {dtype, shape, data_offsets}}, raw little-endian tensor bytes.
#include <map>
#include <string>

#include "microtorch/primitives.hpp"

namespace microtorch {

// Reads every F32 tensor into a Matrix: [n] -> [1, n], [r, c] -> [r, c].
// Tensors with >2 dims are SKIPPED (returned in `skipped`), because in HF
// GPT-2 checkpoints those are exactly the non-parameters -- e.g.
// "h.N.attn.bias" is the [1,1,ctx,ctx] causal-mask buffer, which shares a
// suffix with the real bias "h.N.attn.c_attn.bias"; skipping by RANK
// rather than by name-pattern cannot mistake one for the other.
// A leading "transformer." prefix is stripped so GPT2Model and
// GPT2LMHeadModel exports map onto the same module names.
std::map<std::string, Matrix> load_safetensors(
    const std::string& path, std::map<std::string, std::string>* skipped = nullptr);

// Writes a state_dict as F32 safetensors, the inverse of load_safetensors:
// round-trips through HF tooling (safetensors.torch.load_file reads it).
// Every Matrix is written [rows, cols]; the map's sorted order fixes the
// tensor layout so identical dicts produce byte-identical files.
void save_safetensors(const std::string& path, const std::map<std::string, Matrix>& tensors);

}  // namespace microtorch
