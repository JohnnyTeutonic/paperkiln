W=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad/issues/llama
L=~/issues_work/llama_master; O=~/issues_work/lc
cd $L && git status --short | head; cd $O
g++ -O2 -std=c++17 $W/swa_rm.cpp -I $L/include -I $L/ggml/include -L $L/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$L/build/bin -o swa_rm_master
for a in "2048 600 0" "2048 1900 0" "2048 600 1"; do ./swa_rm_master gemma3_w512.gguf $a 2>/dev/null; done
