set -e
S=/mnt/c/Users/jonat/AppData/Local/Temp/claude/c--Users-jonat-OneDrive-Documents-research-portfolio-complete/ff539744-c077-4107-914b-96a0afc19e79/scratchpad
R=/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch
cd ~/pt_smoke
P=$R/corpus/eval/tinyarith_r20_dev_v1.jsonl
for V in exact_match synthetic_judge; do
  python3 $R/tools/posttrain/rejection_sample.py --mtstudio ~/mtrel/mtstudio --verify ~/mtrel/posttrain_verify \
    --spec $S/smoke_r20.json --prompts $P --verifier $V --n 8 --temp 1.0 --seed 17 --keep top1 --out-dir ~/pt_smoke/rs_$V
done
~/mtrel/mtstudio sample-set $S/smoke_r20.json --prompts $P --out ~/pt_smoke/replay.jsonl --n 8 --temp 1.0 --seed 17 --max-new 16
cmp ~/pt_smoke/replay.jsonl ~/pt_smoke/rs_exact_match/samples.jsonl && echo "REPLAY: byte-identical"
python3 $R/tools/posttrain/audit_sample.py draw ~/pt_smoke/rs_exact_match/scored.jsonl --n 12 --seed 1 --out-dir ~/pt_smoke/audit
head -c 600 ~/pt_smoke/rs_exact_match/scored.jsonl; echo
python3 - <<'PY'
import json
for v in ("exact_match","synthetic_judge"):
    rs=[json.loads(l) for l in open(f"/home/jonat/pt_smoke/rs_{v}/scored.jsonl")]
    print(v, "mean reward", round(sum(r["reward"] for r in rs if r["reward"]>=0)/len(rs),3))
em=[json.loads(l) for l in open("/home/jonat/pt_smoke/rs_exact_match/scored.jsonl")]
sj=[json.loads(l) for l in open("/home/jonat/pt_smoke/rs_synthetic_judge/scored.jsonl")]
assert [a["completion"] for a in em]==[b["completion"] for b in sj]
fooled=sum(1 for a,b in zip(em,sj) if b["reward"]==1 and a["reward"]==0); print("judge accepts, truth rejects:", fooled, "of", len(em))
import collections; print("completion shapes:", collections.Counter(" ".join(r["completion"].split()[:3]) for r in em).most_common(4))
PY
