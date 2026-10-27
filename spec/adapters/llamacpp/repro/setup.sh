# Build the repro programs against llama.cpp master and b6000; make the models.
set -e
W=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad/issues/llama
O=~/issues_work/lc
mkdir -p $O
L=~/issues_work/llama_master
for p in fa_f32 swa_rm dump_logits; do
  [ -f $W/$p.cpp ] || continue
  g++ -O2 -std=c++17 $W/$p.cpp -I $L/include -I $L/ggml/include -L $L/build/bin -lllama -lggml -lggml-base \
      -Wl,-rpath,$L/build/bin -o $O/${p}_master && echo built ${p}_master
done
PY=~/hf_keramos/venv/bin/python; CPY=$PY
export PYTHONPATH=$L/gguf-py:$HOME/issues_work/sp
$PY $W/make_gemma3.py $O/gemma3_w512.gguf 512 2>&1 | tail -1
SNAP=$(ls -d ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/*/ | head -1)
[ -f $O/smollm2-135m-f32.gguf ] || $CPY $L/convert_hf_to_gguf.py $SNAP --outtype f32 --outfile $O/smollm2-135m-f32.gguf 2>&1 | tail -2
ls -la $O
