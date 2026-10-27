O=~/issues_work/lc
cd $O
echo "=== fa_f32 master, SmolLM2-135M f32"
./fa_f32_master smollm2-135m-f32.gguf 2>/dev/null
echo "=== swa_rm master, gemma3 w512"
for args in "2048 600 0" "2048 1900 0" "2048 600 1"; do ./swa_rm_master gemma3_w512.gguf $args 2>/dev/null; done
