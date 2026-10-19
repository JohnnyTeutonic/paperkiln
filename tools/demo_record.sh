#!/usr/bin/env bash
# The SCREEN-RECORDING orchestrator (companion to RECORDING_GUIDE.md).
# Everything rides ONE localhost server — the diff-to-paper page AND the
# live dashboard — so there are no file:// URLs to fight from Windows,
# and the browser tabs are auto-opened for you via Windows interop.
# Training runs in the BACKGROUND while the dashboard fills in live; the
# terminal streams the milestones meanwhile.
#
#   tools/demo_record.sh CORPUS.txt VOCAB.gguf [TINYLLAMA_BIN]
#
# The browser drives EVERYTHING: one page (demo.html) with a
# [paper -> architecture] / [live dashboard] toggle, and the dashboard
# carries a ▶ TRAIN button that launches the armed spec via POST /train.
# The terminal only narrates, waits for the done event, and runs the
# payoff (Atlas row + chat server). Storyboard (studio tab opens FIRST
# so the flow reads fetch -> evidence -> train -> chat, nothing
# pre-decided on camera; the up-front fetch.py call is network
# insurance only):
#   1  page   [studio] fetch 1706.03762 live — architecture lands in
#             the editable builder, the block diagram redraws
#   2  page   [evidence] tab — every value cites the paper's LaTeX
#   3  page   ▶ TRAIN — loss curve, node-graph glow, then artifact
#             download links
#   4  page   terminal starts tinyllama_server on the exported gguf;
#             the in-page chat panel talks to it (endpoint :8081)
# In VSCode: one Simple Browser tab (the script prints the URL). See
# RECORDING_GUIDE.md.
set -e
cd "$(dirname "$0")/.."   # repo root, wherever the script is launched from

# Arguments optional: defaults point at the standard sibling-repo layout
# (override by passing paths, or via DEMO_CORPUS / DEMO_VOCAB / DEMO_TL).
_SIB=$(cd .. 2>/dev/null && pwd)/coalfire.cpp
CORPUS=${1:-${DEMO_CORPUS:-$_SIB/data/tinystories-txt/train-0-small.txt}}
VOCAB=${2:-${DEMO_VOCAB:-$_SIB/releases/chat7b.gguf}}
TL=${3:-${DEMO_TL:-$HOME/tlbuild/tinyllama}}
[ -f "$CORPUS" ] || { echo "corpus not found: $CORPUS"; \
  echo "usage: demo_record.sh [CORPUS.txt] [VOCAB.gguf] [TINYLLAMA_BIN]"; exit 1; }
[ -f "$VOCAB" ] || { echo "vocab gguf not found: $VOCAB"; \
  echo "usage: demo_record.sh [CORPUS.txt] [VOCAB.gguf] [TINYLLAMA_BIN]"; exit 1; }
MT=$(command -v ./mtstudio || command -v build/mtstudio || echo "$HOME/mtrel/mtstudio")
OUT=/tmp/mtdemo_rec
STEPS=${DEMO_STEPS:-300}
cue()  { printf "\n\033[1;33m>>> %s\033[0m\n" "$*"; }
step() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
pause() { read -rp $'\033[1;33m[enter to continue]\033[0m '; }
# Inside VSCode's integrated terminal nothing is spawned externally —
# the whole demo lives in one Simple Browser tab. Detection is belt and
# braces (TERM_PROGRAM can be lost through nested shells); DEMO_NO_OPEN=1
# forces it. Elsewhere, the ONE url auto-opens in the Windows browser.
IN_VSCODE=""
{ [ "${TERM_PROGRAM:-}" = "vscode" ] || [ -n "${VSCODE_IPC_HOOK_CLI:-}" ] ||
  [ -n "${VSCODE_GIT_ASKPASS_MAIN:-}" ] || [ -n "${DEMO_NO_OPEN:-}" ]; } && IN_VSCODE=1
openurl() {
  if [ -n "$IN_VSCODE" ]; then
    return 0  # the one-tab instruction is printed once, up front
  fi
  (cd /mnt/c && /mnt/c/Windows/System32/cmd.exe /c start "$1") \
    > /dev/null 2>&1 || cue "open manually: $1"
}
cleanup() { kill "${SERVE_PID:-}" "${RUN_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT

rm -rf "$OUT"; mkdir -p "$OUT"

step "warming the arXiv cache (insurance only — the on-camera fetch is live)"
python3 papers/fetch.py 1706.03762 --json /tmp/mtdemo_arch.json \
    --emit-html "$OUT/paper.html" 2>/dev/null || true

# The one-tab page: paper view and live dashboard behind an in-page
# toggle, so a VSCode Simple Browser tab never needs a companion.
cat > "$OUT/demo.html" <<'HTML'
<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>paperkiln demo</title><style>
  * { box-sizing:border-box; margin:0; }
  body { background:#0d1117; height:100vh; display:flex; flex-direction:column;
         font:13px ui-monospace,Consolas,monospace; }
  nav { padding:.45rem .8rem; border-bottom:1px solid #30363d; }
  button { background:#161b22; color:#c9d1d9; border:1px solid #30363d;
           border-radius:6px; padding:.3rem .9rem; font:inherit; cursor:pointer;
           margin-right:.5rem; }
  button.on { background:#1f6feb; border-color:#1f6feb; color:#fff; }
  iframe { flex:1; border:0; width:100%; }
</style></head><body>
<nav>
  <button id="bd" class="on">studio</button>
  <button id="bp">evidence: paper &rarr; architecture</button>
</nav>
<iframe id="fr" src="/"></iframe>
<script>
  const fr = document.getElementById('fr');
  const bp = document.getElementById('bp'), bd = document.getElementById('bd');
  bp.onclick = () => { fr.src = '/paper.html'; bp.classList.add('on'); bd.classList.remove('on'); };
  bd.onclick = () => { fr.src = '/';           bd.classList.add('on'); bp.classList.remove('on'); };
</script></body></html>
HTML

# The spec is written up front and ARMED into the server: the page's
# Train button launches it — every action after this line happens in
# the browser, not the terminal.
cat > /tmp/mtdemo_rec_spec.json <<EOF
{
  "name": "demo",
  "arch": {"preset": "llama-tiny"},
  "data": {"corpus": "$CORPUS", "vocab": "$VOCAB", "vocab_cap": 4096, "T": 128},
  "train": {"steps": $STEPS, "batch": 4, "eval_every": 25, "checkpoint_every": 100,
             "gradmap_every": 5},
  "export": {"formats": ["safetensors", "gguf"]},
  "out_dir": "$OUT"
}
EOF
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OMP_WAIT_POLICY=PASSIVE
"$MT" serve "$OUT" 8080 /tmp/mtdemo_rec_spec.json > /dev/null 2>&1 &
SERVE_PID=$!
sleep 1
echo ""
if [ -n "$IN_VSCODE" ]; then
  cue "ONE TAB, INSIDE VSCODE — the only setup step:"
  cue "  Ctrl+Shift+P  ->  'Simple Browser: Show'  ->  paste:"
else
  cue "your browser is opening this page (the only page there is):"
fi
echo ""
echo "        http://localhost:8080/demo.html"
echo ""
openurl "http://localhost:8080/demo.html"
cue "Everything happens IN THE PAGE from here:"
echo "     1. [studio] opens first — type 1706.03762 in the 'from paper'"
echo "        box and hit fetch: the architecture lands in the editable"
echo "        builder and the DIAGRAM redraws, live off arXiv"
echo "     2. [evidence] tab — every extracted value cites the paper's LaTeX"
echo "     3. back on [studio], press \xe2\x96\xb6 TRAIN — loss curve descends,"
echo "        node-graph glows; exported .safetensors/.gguf become"
echo "        download links when it finishes"
echo "     4. this terminal then starts the chat server — talk to the model"
echo "        in the page's chat panel"
echo ""
cue "This terminal will notice when training finishes and run the payoff."
printf "     waiting for the Train button"
until grep -q '"event":"done"' "$OUT/events.jsonl" 2>/dev/null; do
  printf "."
  sleep 2
done
echo " done!"
step "the run is now an Atlas data point"
python3 tools/atlas_extract.py "$OUT" | head -20

step "chat with it — a separately written engine reads the GGUF"
TLS=${DEMO_TLS:-$HOME/tlbuild/tinyllama_server}
if [ -x "$TLS" ]; then
  "$TLS" "$OUT/demo.gguf" 8081 localhost > /dev/null 2>&1 &
  CHAT_PID=$!
  trap 'cleanup; kill "${CHAT_PID:-}" 2>/dev/null || true' EXIT
  sleep 2
  cue "chat server is UP — use the page's chat panel (endpoint :8081)"
  step "paper to chatting model, one stack, every claim receipted"
  cue "leave this running while you record; Ctrl-C here ends the demo"
  wait "$CHAT_PID"
elif [ -x "$TL" ]; then
  "$TL" "$OUT/demo.gguf" "$OUT/demo.gguf" 4 prompt "once upon a time" \
      --max-tokens 40 -ngl 0 --top-k 1 --raw-prompt 2>/dev/null | tail -2
  step "done — paper to chatting model, one stack, every claim receipted"
fi
