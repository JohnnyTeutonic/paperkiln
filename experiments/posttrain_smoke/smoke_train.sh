set -e
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad
cd ~/mtrel && timeout 900 make -j4 mtstudio 2>&1 | grep -E "error|Built target mtstudio" | head -20
mkdir -p ~/pt_smoke && rm -rf ~/pt_smoke/r20
cd ~/pt_smoke && timeout 3000 ~/mtrel/mtstudio run "$S/smoke_r20.json" > ~/pt_smoke/train.log 2>&1
tail -3 ~/pt_smoke/train.log; grep '"eval"' ~/pt_smoke/r20/events.jsonl | tail -3; ls ~/pt_smoke/r20
