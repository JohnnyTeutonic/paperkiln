# microtorch — autograd tape + nn API over transformer_core's primitives.
#
# The seam docs/DESIGN.md section 1 proposed: consume transformer_cpp via
# add_subdirectory and link the existing transformer_core STATIC target.
# EXCLUDE_FROM_ALL so only transformer_core is built (as a dependency),
# not that repo's executables. CUDA stays off here: phase 1a is CPU-first
# per docs/history/PHASE0_KERNEL_AUDIT.md section 6.
cmake_minimum_required(VERSION 3.18)
project(microtorch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA support (docs/DESIGN.md section 8.2). OFF by default: the local machine
# never builds or runs CUDA (validated on Colab). ON: transformer_cpp's
# CUDA_ENABLED kernel tree is compiled and device::matmul can dispatch to
# CudaMatrix::matmul after device::set(Device::CUDA).
option(MICROTORCH_CUDA "Route device::matmul through transformer_core CUDA kernels" OFF)
if(MICROTORCH_CUDA)
    set(CUDA_ENABLED ON CACHE BOOL "transformer_cpp CUDA kernels" FORCE)
endif()

# transformer_core source resolution: the development layout keeps
# transformer_cpp as a SIBLING (Jonathan's machine, richest history);
# clones from GitHub use the VENDORED copy in third_party/ (synced via
# tools/sync_vendor.sh). CI discovered the sibling-only assumption made
# the pushed repo unbuildable by anyone else (workflow run #13).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../transformer_cpp/CMakeLists.txt)
    set(MICROTORCH_TCPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../transformer_cpp)
elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/transformer_cpp/CMakeLists.txt)
    set(MICROTORCH_TCPP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/transformer_cpp)
else()
    message(FATAL_ERROR "transformer_cpp sources not found (sibling dir or "
                        "third_party/transformer_cpp)")
endif()
message(STATUS "transformer_core from: ${MICROTORCH_TCPP_DIR}")
add_subdirectory(${MICROTORCH_TCPP_DIR}
                 ${CMAKE_BINARY_DIR}/transformer_cpp EXCLUDE_FROM_ALL)

add_library(microtorch STATIC
    src/autograd.cpp
    src/ops.cpp
    src/nn.cpp
    src/device.cpp
    src/device_cache.cpp
    src/safetensors.cpp
    src/gguf.cpp
    src/quant.cpp
    src/llama.cpp
    src/kimi_linear.cpp
    src/srd.cpp
    src/cerebellum.cpp
    src/mamba.cpp
)
target_include_directories(microtorch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(microtorch PUBLIC transformer_core
                                 PRIVATE nlohmann_json::nlohmann_json)
if(MICROTORCH_CUDA)
    # USE_CUDA/CUDA_AVAILABLE mirror transformer_cpp's own compile defs
    # (its add_compile_definitions is directory-scoped and does not reach
    # microtorch's translation units).
    target_compile_definitions(microtorch PUBLIC MICROTORCH_CUDA USE_CUDA CUDA_AVAILABLE)
    # Phase B1 residency (docs/CUDA_PHASE_B.md): microtorch-owned CUDA TU.
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    target_sources(microtorch PRIVATE src/cuda_resident.cu)
    # Phase B2.1a op set (docs/CUDA_PHASE_B2.md section 4): same TU rules.
    target_sources(microtorch PRIVATE src/cuda_ops.cu)
endif()

enable_testing()
add_executable(test_gradcheck tests/test_gradcheck.cpp)
target_link_libraries(test_gradcheck PRIVATE microtorch)
add_test(NAME gradcheck COMMAND test_gradcheck)

add_executable(test_nn tests/test_nn.cpp)
target_link_libraries(test_nn PRIVATE microtorch)

add_executable(test_resident_parity tests/test_resident_parity.cpp)
target_link_libraries(test_resident_parity PRIVATE microtorch)

add_executable(test_step_residency tests/test_step_residency.cpp)
target_link_libraries(test_step_residency PRIVATE microtorch)

add_executable(test_cuda_ops tests/test_cuda_ops.cpp)
target_link_libraries(test_cuda_ops PRIVATE microtorch)
add_test(NAME step_residency COMMAND test_step_residency)

add_executable(test_highway tests/test_highway.cpp)
target_link_libraries(test_highway PRIVATE microtorch)

add_executable(test_deep_swa tests/test_deep_swa.cpp)
target_link_libraries(test_deep_swa PRIVATE microtorch)
add_test(NAME nn COMMAND test_nn)

add_executable(test_kimi_linear tests/test_kimi_linear.cpp)
target_link_libraries(test_kimi_linear PRIVATE microtorch)
add_test(NAME kimi_linear COMMAND test_kimi_linear)

add_executable(test_cerebellum tests/test_cerebellum.cpp)
target_link_libraries(test_cerebellum PRIVATE microtorch)
add_test(NAME cerebellum COMMAND test_cerebellum)

add_executable(test_mamba tests/test_mamba.cpp)
target_link_libraries(test_mamba PRIVATE microtorch)
add_test(NAME mamba COMMAND test_mamba)

add_executable(test_train_utils tests/test_train_utils.cpp)
target_link_libraries(test_train_utils PRIVATE microtorch)
add_test(NAME train_utils COMMAND test_train_utils)

add_executable(test_gguf_export tests/test_gguf_export.cpp)
target_link_libraries(test_gguf_export PRIVATE microtorch)
add_test(NAME gguf_export COMMAND test_gguf_export)

add_executable(test_lora_quant tests/test_lora_quant.cpp)
target_link_libraries(test_lora_quant PRIVATE microtorch)
add_test(NAME lora_quant COMMAND test_lora_quant)

add_executable(test_llama tests/test_llama.cpp)
target_link_libraries(test_llama PRIVATE microtorch)
add_test(NAME llama COMMAND test_llama)

add_executable(test_batching tests/test_batching.cpp)
target_link_libraries(test_batching PRIVATE microtorch)
add_test(NAME batching COMMAND test_batching)

add_executable(test_checkpointing tests/test_checkpointing.cpp)
target_link_libraries(test_checkpointing PRIVATE microtorch)
add_test(NAME checkpointing COMMAND test_checkpointing)

add_executable(test_muon tests/test_muon.cpp)
target_link_libraries(test_muon PRIVATE microtorch)
add_test(NAME muon COMMAND test_muon)

add_executable(test_inplace tests/test_inplace.cpp)
target_link_libraries(test_inplace PRIVATE microtorch)
add_test(NAME inplace COMMAND test_inplace)

add_executable(test_flex tests/test_flex.cpp)
target_link_libraries(test_flex PRIVATE microtorch)
add_test(NAME flex COMMAND test_flex)

add_executable(test_swa tests/test_swa.cpp)
target_link_libraries(test_swa PRIVATE microtorch)
add_test(NAME swa COMMAND test_swa)

add_executable(test_attn_res tests/test_attn_res.cpp)
target_link_libraries(test_attn_res PRIVATE microtorch)
add_test(NAME attn_res COMMAND test_attn_res)

add_executable(test_srd tests/test_srd.cpp)
target_link_libraries(test_srd PRIVATE microtorch)
add_test(NAME srd COMMAND test_srd)

# Phase-1c success test; needs a checkpoint + reference on disk, so it is a
# tool rather than a registered test (tools/make_gpt2_reference.py builds
# the reference).
add_executable(gpt2_parity tools/gpt2_parity.cpp)
target_link_libraries(gpt2_parity PRIVATE microtorch nlohmann_json::nlohmann_json)

# Phase-2b Llama-family test: load Qwen 2B safetensors and verify RMSNorm + RoPE
# work end-to-end. Needs a checkpoint on disk (downloaded separately).
add_executable(qwen_parity tools/qwen_parity.cpp)
target_link_libraries(qwen_parity PRIVATE microtorch nlohmann_json::nlohmann_json)

# Phase 3a: Benchmark Kimi Linear vs standard attention
add_executable(benchmark_attention tools/benchmark_attention.cpp)
target_link_libraries(benchmark_attention PRIVATE microtorch)

# Phase 3c: Benchmark Mamba vs Kimi vs standard attention
add_executable(benchmark_mamba tools/benchmark_mamba.cpp)
target_link_libraries(benchmark_mamba PRIVATE microtorch)

# Phase 3 Integration: Unified test of all three mechanisms
add_executable(phase3_integration tools/phase3_integration_test.cpp)
target_link_libraries(phase3_integration PRIVATE microtorch)

# Semantic E2E: trained GGUF -> state_dict -> re-export (gguf + safetensors);
# tinyllama.cpp generation parity between original and re-export is the gate.
add_executable(gguf_roundtrip tools/gguf_roundtrip.cpp)
target_link_libraries(gguf_roundtrip PRIVATE microtorch)

# SRD graduation gate: TinyStories training parity, 4 lanes
# (exact / kimi / srd / srd-falsifier) on identical batches.
add_executable(srd_parity tools/srd_parity.cpp)
target_link_libraries(srd_parity PRIVATE microtorch)

# SRD needle-in-haystack: synthetic associative recall, the retrieval-
# structured eval the T=256 regime negative pointed to.
add_executable(srd_needle tools/srd_needle.cpp)
target_link_libraries(srd_needle PRIVATE microtorch)

# M1 (docs/STUDIO_PLAN.md): the run-spec driver. One JSON describes the
# lifecycle; the driver executes it and emits the JSONL event stream.
add_executable(mtstudio tools/mtstudio.cpp)
target_link_libraries(mtstudio PRIVATE microtorch nlohmann_json::nlohmann_json)

# Python bindings (optional; requires Python dev headers). pybind11 is
# fetched at configure time so no vendored copy lives in-tree.
option(MICROTORCH_BUILD_PYTHON "Build the pybind11 Python module" OFF)
if(MICROTORCH_BUILD_PYTHON)
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.6)
    FetchContent_MakeAvailable(pybind11)
    pybind11_add_module(_microtorch python/bindings.cpp)
    target_link_libraries(_microtorch PRIVATE microtorch)
endif()
