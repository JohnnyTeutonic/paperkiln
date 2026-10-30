set -e
cd /mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete
python3 scripts/cpu_budget.py run --session paperkiln --job hf-flex-llama-clean-build --cores 1 --resume incremental-cmake-build-hf-flex-llama -- cmake --build /home/jonat/hf_flex_llama_clean_20261030/build --target mtstudio --parallel 1 > /home/jonat/hf_flex_llama_build_20261030.log 2>&1
