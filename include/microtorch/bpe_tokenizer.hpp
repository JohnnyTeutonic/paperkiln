#pragma once
// Byte-level BPE encoder/decoder: the C++ side of tools/bpe.py (the
// reference, which also trains). Byte-identical to it: tests/test_bpe.py
// encodes a corpus with both and compares ids.
//
// The implementation is coalfire's standalone byte_bpe_tokenizer.hpp
// (ROADMAP C5), vendored unchanged in third_party/transformer_cpp/include/:
// one tokenizer for both trainers. The pre-tokenization rules and the
// vocabulary layout are documented there. This header keeps paperkiln's
// names, microtorch::bpe::{specials, Tokenizer, GgufTokenizer,
// read_gguf_tokenizer, load_gguf}. The vendored file is included by its
// path, so every target and tool that compiles with -I include (bpe_encode,
// llama3_encode, the experiment gates) keeps building without transformer_core's
// include directory; a MICROTORCH_TCPP_SIBLING build still uses the vendored
// copy of this one header.
#include "../../third_party/transformer_cpp/include/byte_bpe_tokenizer.hpp"

namespace microtorch {
namespace bpe = ::byte_bpe;
}  // namespace microtorch
