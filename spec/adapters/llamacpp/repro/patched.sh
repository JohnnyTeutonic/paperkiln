# Measure two candidate fixes on llama.cpp master, then restore the tree and rebuild:
#   gptneox graph with LLM_FFN_GELU_ERF; build_attn_mha without the f32 -> f16 cast of K/V.
set -e
W=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad/issues/llama
O=~/issues_work/lc
L=~/issues_work/llama_master
PY=~/hf_keramos/venv/bin/python
cd $L
sed -i 's/LLM_FFN_GELU, LLM_FFN_SEQ, il)/LLM_FFN_GELU_ERF, LLM_FFN_SEQ, il)/' src/models/gptneox.cpp
sed -i 's/if (k->type == GGML_TYPE_F32) {/if (false \&\& k->type == GGML_TYPE_F32) {/; s/if (v->type == GGML_TYPE_F32) {/if (false \&\& v->type == GGML_TYPE_F32) {/' src/llama-graph.cpp
git diff --stat
cmake --build build -j2 --target llama > ~/issues_work/patched_build.log 2>&1
cd $O
./dump_logits_master pythia-160m-f32.gguf ids.txt pythia_master_gelu_erf.f32 2>/dev/null
OMP_NUM_THREADS=2 $PY $W/pythia_compare.py ~/issues_work/pythia-160m cmp pythia_master_gelu_erf.f32 2>&1 | grep -v -i "warn\|Loading"
echo "=== fa_f32 with the cast removed"
./fa_f32_master smollm2-135m-f32.gguf 2>/dev/null
cd $L
git checkout -- src/models/gptneox.cpp src/llama-graph.cpp
git diff --stat
cmake --build build -j2 --target llama > ~/issues_work/restore_build.log 2>&1
echo restored
cd $O && ./fa_f32_master smollm2-135m-f32.gguf 2>/dev/null | tail -2
