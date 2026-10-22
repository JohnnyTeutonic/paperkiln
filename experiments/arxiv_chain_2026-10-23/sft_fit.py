"""Keep the SFT examples whose prompt + completion fit a T-token window
(tools/bpe.py tokens, plus one for the end token, with a margin), in file
order, up to N. Usage: sft_fit.py microtorch_dir vocab.gguf in.jsonl out.jsonl T N"""
import json
import sys

mt, vocab, src, dst, T, N = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
sys.path.insert(0, mt + "/tools")
import bpe  # noqa: E402

tok = bpe.read_gguf(vocab)
kept = seen = 0
lengths = []
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as g:
    for line in f:
        if not line.strip():
            continue
        seen += 1
        ex = json.loads(line)
        n = len(tok.encode(ex["prompt"] + ex["completion"])) + 1
        if n <= T - 4:
            g.write(line if line.endswith("\n") else line + "\n")
            kept += 1
            lengths.append(n)
            if kept >= N:
                break
print(json.dumps({"seen": seen, "kept": kept, "window": T, "max_tokens_kept": max(lengths),
                  "mean_tokens_kept": round(sum(lengths) / len(lengths), 1)}))
