# Pythia-160m: tokenizer files, f32 GGUF; build dump_logits; dump stock master logits.
set -e
W=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad/issues/llama
O=~/issues_work/lc
L=~/issues_work/llama_master
PY=~/hf_keramos/venv/bin/python
P=~/issues_work/pythia-160m
mkdir -p $P
SNAP=$(ls -d ~/.cache/huggingface/hub/models--EleutherAI--pythia-160m/snapshots/*/ | head -1)
ln -sf $SNAP/config.json $P/config.json
ln -sf $SNAP/model.safetensors $P/model.safetensors
cd $P
[ -f tokenizer.json ] || $PY -c "
from huggingface_hub import hf_hub_download
for f in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
    print(hf_hub_download('EleutherAI/pythia-160m', f, local_dir='.'))
"
ls -la $P
export PYTHONPATH=$L/gguf-py:$HOME/issues_work/sp
[ -f $O/pythia-160m-f32.gguf ] || ~/ort_keramos/venv_tf4/bin/python $L/convert_hf_to_gguf.py $P --outtype f32 --outfile $O/pythia-160m-f32.gguf 2>&1 | tail -1
g++ -O2 -std=c++17 $W/dump_logits.cpp -I $L/include -I $L/ggml/include -L $L/build/bin -lllama -lggml -lggml-base \
    -Wl,-rpath,$L/build/bin -o $O/dump_logits_master
cd $O
unset PYTHONPATH
$PY $W/pythia_compare.py $P ids
./dump_logits_master pythia-160m-f32.gguf ids.txt pythia_master_stock.f32 2>/dev/null
OMP_NUM_THREADS=2 $PY $W/pythia_compare.py $P cmp pythia_master_stock.f32 2>&1 | grep -v -i warn
