set -e
# ReLU^2 Wins (arXiv 2402.03804) paper -> chat on the CPU, the GoLU run's scale. The SFT set is
# TinyChat unfiltered: arxiv_to_chat keeps the first 2000 that fit T and counts the rest.
MT=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch
IN=~/arxiv_chain_runs/inputs
mkdir -p $IN
cp --update=none $MT/corpus/data/tinychat_hf_sample_v1/pretrain_train.txt $IN/
cp --update=none $MT/corpus/data/vocab_fineweb_tinychat_bpe8k_v1/fineweb_tinychat_bpe8k.gguf $IN/
cp --update=none $MT/corpus/eval/tinychat_hf_prompts_v1.jsonl $IN/
cp $MT/experiments/arxiv_chain_2026-10-23/2402.03804.extraction.json $IN/
cd $MT
python3 tools/arxiv_to_chat.py 2402.03804 --run-dir ~/arxiv_chain_runs/relu2wins \
  --extraction $IN/2402.03804.extraction.json \
  --mtstudio ~/expr_build/mtstudio \
  --corpus $IN/pretrain_train.txt --vocab $IN/fineweb_tinychat_bpe8k.gguf \
  --sft-examples ~/pkchat/fwsft/sft_tinychat.jsonl --sft-limit 2000 \
  --d 128 --layers 2 --heads 4 --T 64 --batch 4 \
  --pretrain-steps 600 --sft-steps 200 --lr-warmup 50 --sft-lr-warmup 20 --lr-decay cosine \
  --threads 2 --checkpoint-every 150 \
  --ember-bin ~/ember_ir_build/tinyllama_server --ember-www $MT/../ember.cpp/www \
  --prompts $IN/tinychat_hf_prompts_v1.jsonl --parity-n 40 --max-new 40 --chat-greedy \
  --chat "Hello! What is your name?" --chat "Can you tell me a short story about a cat?" "$@"
