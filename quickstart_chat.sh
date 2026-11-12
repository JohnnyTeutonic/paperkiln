#!/usr/bin/env bash
# quickstart_chat.sh: from a fresh clone to chatting with your own small language model.
#
#   bash quickstart_chat.sh          # asks a few questions, then does everything
#   bash quickstart_chat.sh --help   # the non-interactive flags
#
# What it does, in order: checks your tools (and tells you exactly how to get any
# that are missing), builds paperkiln, makes the TinyChat practice corpus, trains a
# model on your CPU, writes a plain-language card about the model, then starts a
# chat page in your browser. Optionally it gives you a public link so someone else
# can chat to your model too.
#
# Linux, macOS and Windows through WSL2. Native Windows is not supported yet.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
BUILD="${PAPERKILN_BUILD_DIR:-$ROOT/build}"
PORT=8080
MODE="" PRESET="" SHARE="" ASSUME_YES=0 NO_BROWSER=0
# The ready-made model as a browser bundle (tools/export_web_chat.py): it runs in the browser, so no build is needed.
# The release asset's SHA-256 is pinned here and checked before anything is unpacked: a changed or damaged
# download stops the script. It matches the digest GitHub reports for the asset (release chat-v1, 1,728,786 bytes).
# Another bundle needs both variables: PAPERKILN_DEFAULT_MODEL_URL and PAPERKILN_DEFAULT_MODEL_SHA256.
DEFAULT_MODEL_URL="https://github.com/JohnnyTeutonic/paperkiln/releases/download/chat-v1/tinychat-v2-web.tar.gz"
DEFAULT_MODEL_SHA256="16ae917498bc3a30e816b140e2fbced1d6b8a62426607a077e9fb1f1e9dd0fb2"
if [ -n "${PAPERKILN_DEFAULT_MODEL_URL:-}" ]; then
  DEFAULT_MODEL_URL="$PAPERKILN_DEFAULT_MODEL_URL"
  DEFAULT_MODEL_SHA256="${PAPERKILN_DEFAULT_MODEL_SHA256:-}"
fi
TOOLS_DIR="$HOME/.paperkiln/bin"

usage() {
  cat <<EOF
Usage: bash quickstart_chat.sh [options]
  --mode default|train   chat with the ready-made model, or train your own
  --preset quick|better  quick: about 10 minutes on a laptop CPU; better: about an hour
  --share | --no-share   give the chat a public link (Cloudflare quick tunnel) or keep it on this machine
  --port N               local port for the chat page (default 8080)
  --no-browser           print the chat address instead of opening a browser
  --yes                  accept the defaults instead of asking
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --share) SHARE=yes; shift ;;
    --no-share) SHARE=no; shift ;;
    --port) PORT="$2"; shift 2 ;;
    --no-browser) NO_BROWSER=1; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1"; usage; exit 2 ;;
  esac
done

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
say()  { printf '  %s\n' "$*"; }
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31mStopped:\033[0m %s\n' "$*" >&2; exit 1; }

# ask "question" default -> echoes the answer (default when --yes or empty input)
ask() {
  local q="$1" def="$2" ans
  if [ "$ASSUME_YES" = 1 ]; then echo "$def"; return; fi
  read -r -p "  $q [$def] " ans </dev/tty || ans=""
  echo "${ans:-$def}"
}
yes_no() { case "$(ask "$1 (y/n)" "$2")" in y|Y|yes|Yes) return 0 ;; *) return 1 ;; esac; }

# ---------------------------------------------------------------- 1. where are we
step "Checking your system"
OS="$(uname -s)"; ARCH="$(uname -m)"; PLATFORM=""
case "$OS" in
  Linux)
    if grep -qi microsoft /proc/version 2>/dev/null; then PLATFORM=wsl; else PLATFORM=linux; fi ;;
  Darwin) PLATFORM=macos ;;
  MINGW*|MSYS*|CYGWIN*)
    bold "This looks like Windows (Git Bash or similar)."
    say "paperkiln runs on Windows through WSL2, a Linux environment built into Windows."
    say "1. Open PowerShell as Administrator and run:   wsl --install"
    say "2. Restart, open the new 'Ubuntu' app, and run this script again from there:"
    say "     git clone https://github.com/JohnnyTeutonic/paperkiln && cd paperkiln && bash quickstart_chat.sh"
    exit 0 ;;
  *) die "unrecognised system '$OS'. Linux, macOS and WSL2 are supported." ;;
esac
say "System: $PLATFORM ($ARCH)"
yes_no "Is that right?" y || die "tell us what you are running: https://github.com/JohnnyTeutonic/paperkiln/issues"

if [ -z "$MODE" ]; then
  bold ""
  bold "What would you like to do?"
  say "1) Chat with the ready-made model (downloads it; ready in about a minute; no compiler needed; it runs in your browser)"
  say "2) Train your own model on this computer (needs a C++ compiler; about 10 minutes for the quick model)"
  case "$(ask "Choose 1 or 2" 2)" in 1) MODE=default ;; *) MODE=train ;; esac
fi

# ---------------------------------------------------------------- 2. tools
have() { command -v "$1" >/dev/null 2>&1; }

install_hint() {
  # $1 = what is missing (compiler|cmake|python)
  local pm=""
  if [ "$PLATFORM" = macos ]; then pm=brew
  elif have apt-get; then pm=apt
  elif have dnf; then pm=dnf
  elif have pacman; then pm=pacman
  fi
  case "$1:$pm" in
    compiler:apt)    echo "sudo apt-get update && sudo apt-get install -y build-essential" ;;
    compiler:dnf)    echo "sudo dnf install -y gcc-c++ make" ;;
    compiler:pacman) echo "sudo pacman -S --needed base-devel" ;;
    compiler:brew)   echo "xcode-select --install" ;;
    cmake:apt)       echo "sudo apt-get install -y cmake" ;;
    cmake:dnf)       echo "sudo dnf install -y cmake" ;;
    cmake:pacman)    echo "sudo pacman -S --needed cmake" ;;
    cmake:brew)      echo "brew install cmake" ;;
    python:apt)      echo "sudo apt-get install -y python3" ;;
    python:dnf)      echo "sudo dnf install -y python3" ;;
    python:pacman)   echo "sudo pacman -S --needed python" ;;
    python:brew)     echo "brew install python" ;;
    *)               echo "" ;;
  esac
}

# Offer the system install (needs sudo), or a no-sudo route where one exists.
need() {
  local what="$1" cmd
  cmd="$(install_hint "$what")"
  bold "Missing: $what"
  case "$what" in
    cmake)
      say "No administrator rights? CMake installs for just you with:  python3 -m pip install --user cmake"
      if have python3 && yes_no "Install CMake for just you now, without sudo?" y; then
        python3 -m pip install --user cmake && export PATH="$HOME/.local/bin:$PATH" && return 0
      fi ;;
    compiler)
      say "No administrator rights? With conda installed:  conda install -c conda-forge cxx-compiler" ;;
  esac
  if [ -n "$cmd" ]; then
    say "The usual install command (asks for your password):  $cmd"
    if yes_no "Run it now?" y; then eval "$cmd"; return 0; fi
  fi
  die "install $what, then run this script again."
}

have python3 || need python
if [ "$MODE" = train ]; then
  have cmake || need cmake
  if ! have c++ && ! have g++ && ! have clang++; then need compiler; fi
  say "Tools: python3, cmake $(cmake --version | head -1 | awk '{print $3}'), C++ compiler found."
else
  have curl || die "curl is needed to download the model (it ships with almost every system)."
  say "Tools: python3 and curl found."
fi

# ---------------------------------------------------------------- 3. build
JOBS="$( (have nproc && nproc) || sysctl -n hw.ncpu 2>/dev/null || echo 2)"
MTSTUDIO="$BUILD/mtstudio"

build_mtstudio() {
  step "Building paperkiln (a few minutes the first time)"
  [ -e third_party/transformer_cpp ] || [ -e ../coalfire.cpp ] || [ -e ../transformer_cpp ] \
    || die "third_party/transformer_cpp is missing; re-clone with: git clone --recursive https://github.com/JohnnyTeutonic/paperkiln"
  cmake -S . -B "$BUILD" -DCMAKE_BUILD_TYPE=Release >"$BUILD.configure.log" 2>&1 \
    || { tail -20 "$BUILD.configure.log"; die "CMake configuration failed (full log: $BUILD.configure.log)."; }
  cmake --build "$BUILD" --target mtstudio -j "$JOBS" >"$BUILD.build.log" 2>&1 \
    || { tail -30 "$BUILD.build.log"; die "the build failed (full log: $BUILD.build.log)."; }
  say "Built $MTSTUDIO"
}

sha256_of() {
  if have sha256sum; then sha256sum "$1" | awk '{print $1}'
  elif have shasum; then shasum -a 256 "$1" | awk '{print $1}'
  else return 1; fi
}

# fetch_default_model URL SHA256 DEST: download into DEST.tar.gz.part (an interrupted download resumes on
# the next run), check its SHA-256 against the pinned digest before unpacking anything, refuse members
# that could land outside DEST (absolute paths, "..", links, devices), unpack into a temporary folder
# and only then move it into place, so a failure never leaves a half-unpacked model behind.
fetch_default_model() {
  local url="$1" want="$2" dest="$3" part got top tmp bad attempt=1
  [[ "$want" =~ ^[0-9a-f]{64}$ ]] \
    || die "no SHA-256 to check $url against: set PAPERKILN_DEFAULT_MODEL_SHA256 to the bundle's 64-character digest."
  sha256_of /dev/null >/dev/null 2>&1 || die "sha256sum (or shasum) is needed to check the download."
  mkdir -p "$(dirname "$dest")"
  part="$dest.tar.gz.part"
  while :; do
    curl -fL --retry 3 --progress-bar -C - -o "$part" "$url" \
      || die "the download from $url stopped. Run the script again to resume it, or choose 2 to train your own model."
    got="$(sha256_of "$part")"
    [ "$got" = "$want" ] && break
    rm -f "$part"
    [ "$attempt" -ge 2 ] && die "the downloaded model's SHA-256 is $got, not the pinned $want. Nothing was unpacked; the download was deleted."
    say "The download did not match its pinned SHA-256; fetching it once more from the start."
    attempt=2
  done
  say "SHA-256 checked: $got"
  tar -tzf "$part" >"$part.list" 2>/dev/null || { rm -f "$part" "$part.list"; die "the download is not a readable .tar.gz."; }
  top="$(head -1 "$part.list" | cut -d/ -f1)"
  case "$top" in ""|.|..) rm -f "$part.list"; die "the bundle does not unpack into one folder; nothing was unpacked." ;; esac
  bad="$(awk -v top="$top" '{ n = $0; if (n != top && index(n, top "/") != 1) { print; next }
                             k = split(n, c, "/"); for (i = 1; i <= k; i++) if (c[i] == "..") { print; next } }' "$part.list")"
  bad="$bad$(tar -tvzf "$part" | cut -c1 | grep -v '^[-d]$' || true)"
  rm -f "$part.list"
  [ -z "$bad" ] || die "the bundle holds entries outside its folder, links or special files; nothing was unpacked: $bad"
  tmp="$dest.tmp.$$"
  rm -rf "$tmp" && mkdir -p "$tmp"
  tar -xzf "$part" -C "$tmp" --strip-components=1 --no-same-owner \
    || { rm -rf "$tmp"; die "unpacking the bundle failed; nothing was installed."; }
  { [ -f "$tmp/index.html" ] && [ -f "$tmp/model/manifest.json" ]; } \
    || { rm -rf "$tmp"; die "the bundle has no index.html or model/manifest.json; nothing was installed."; }
  printf '%s\n' "$want" >"$tmp/.bundle_sha256"
  rm -rf "$dest" && mv "$tmp" "$dest" && rm -f "$part"
  say "Unpacked into $dest"
}

# ---------------------------------------------------------------- 4. the model
if [ "$MODE" = train ]; then
  if [ -z "$PRESET" ]; then
    bold ""
    bold "How long can you wait?"
    say "1) Quick: a model of about 450,000 parameters, about 10 minutes on a laptop CPU"
    say "2) Better: a bigger model, about an hour on a CPU"
    case "$(ask "Choose 1 or 2" 1)" in 2) PRESET=better ;; *) PRESET=quick ;; esac
  fi
  SPEC="specs/tinychat-$PRESET.json"
  OUT="runs/tinychat-$PRESET"
  [ -x "$MTSTUDIO" ] || build_mtstudio

  step "Making the TinyChat practice corpus"
  # Several passes over a small inventory beat one pass over a large one at this size
  # (transformer_cpp CHAT_EXPERIMENTS.md): about five epochs for each preset.
  if [ "$PRESET" = better ]; then N=4000; else N=1500; fi
  python3 tools/get_tinychat_data.py --version 2 --dialogues "$N"

  step "Training ($PRESET). Progress is shown below; you can leave it running."
  say "Trained models are saved in $OUT"
  "$MTSTUDIO" run "$SPEC" | python3 tools/train_progress.py "$SPEC"
else
  OUT="runs/tinychat-default-web"
  # A bundle unpacked by this script carries the digest it was checked against; anything else is fetched again.
  if [ ! -f "$OUT/index.html" ] || [ "$(cat "$OUT/.bundle_sha256" 2>/dev/null)" != "$DEFAULT_MODEL_SHA256" ]; then
    step "Downloading the ready-made chat model"
    fetch_default_model "$DEFAULT_MODEL_URL" "$DEFAULT_MODEL_SHA256" "$OUT"
  fi
fi

# ---------------------------------------------------------------- 5. the card
if [ "$MODE" = train ] && [ ! -f "$OUT/card.md" ] && [ -f tools/model_card.py ]; then
  step "Writing a plain-language card about your model"
  python3 tools/model_card.py "$OUT" --probe --mtstudio "$MTSTUDIO" || say "(the card could not be written; chatting still works)"
fi
if [ -f "$OUT/card.md" ]; then
  bold ""
  bold "About the model you are about to chat with"
  sed 's/^/  /' "$OUT/card.md"
fi

# ---------------------------------------------------------------- 6. chat, and maybe share
if [ -z "$SHARE" ]; then
  bold ""
  bold "Share it?"
  say "A public link lets anyone you send it to chat with your model while this script runs."
  say "It is anonymous and free (a Cloudflare quick tunnel); nothing about you is published."
  if yes_no "Make a public link?" n; then SHARE=yes; else SHARE=no; fi
fi

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

step "Starting the chat server"
if [ "$MODE" = train ]; then
  "$MTSTUDIO" chat "$OUT" --port "$PORT" --host 127.0.0.1 >"$OUT/chat.log" 2>&1 &
  HEALTH="health"
else
  # The ready-made model runs in the browser; Python's built-in web server just hands out the files.
  python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$OUT" >"$OUT/chat.log" 2>&1 &
  HEALTH="model/manifest.json"
fi
PIDS+=($!)
for _ in $(seq 1 60); do
  curl -fs "http://127.0.0.1:$PORT/$HEALTH" >/dev/null 2>&1 && break
  kill -0 "${PIDS[0]}" 2>/dev/null || { tail -20 "$OUT/chat.log"; die "the chat server stopped (log: $OUT/chat.log)."; }
  sleep 1
done
LOCAL_URL="http://127.0.0.1:$PORT/"

PUBLIC_URL=""
if [ "$SHARE" = yes ]; then
  CF="$(command -v cloudflared || true)"
  if [ -z "$CF" ] && [ -x "$TOOLS_DIR/cloudflared" ]; then CF="$TOOLS_DIR/cloudflared"; fi
  if [ -z "$CF" ]; then
    say "The public link needs Cloudflare's small 'cloudflared' program (about 35 MB, from github.com/cloudflare)."
    if yes_no "Download it into $TOOLS_DIR?" y; then
      mkdir -p "$TOOLS_DIR"
      case "$PLATFORM:$ARCH" in
        linux:x86_64|wsl:x86_64)   asset=cloudflared-linux-amd64 ;;
        linux:aarch64|wsl:aarch64) asset=cloudflared-linux-arm64 ;;
        macos:x86_64)              asset=cloudflared-darwin-amd64.tgz ;;
        macos:arm64)               asset=cloudflared-darwin-arm64.tgz ;;
        *) die "no cloudflared download for $PLATFORM/$ARCH; install it from https://github.com/cloudflare/cloudflared" ;;
      esac
      url="https://github.com/cloudflare/cloudflared/releases/latest/download/$asset"
      if [[ "$asset" == *.tgz ]]; then curl -fsSL "$url" | tar -xz -C "$TOOLS_DIR"
      else curl -fsSL "$url" -o "$TOOLS_DIR/cloudflared"; fi
      chmod +x "$TOOLS_DIR/cloudflared"; CF="$TOOLS_DIR/cloudflared"
    fi
  fi
  if [ -n "$CF" ]; then
    "$CF" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >"$OUT/tunnel.log" 2>&1 &
    PIDS+=($!)
    for _ in $(seq 1 30); do
      PUBLIC_URL="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$OUT/tunnel.log" | head -1 || true)"
      [ -n "$PUBLIC_URL" ] && break
      sleep 1
    done
    [ -n "$PUBLIC_URL" ] || say "(the public link did not come up; see $OUT/tunnel.log. Local chat still works.)"
  fi
fi

open_browser() {
  case "$PLATFORM" in
    macos) open "$1" ;;
    wsl)   (have wslview && wslview "$1") || cmd.exe /c start "" "$1" 2>/dev/null ;;
    linux) have xdg-open && xdg-open "$1" >/dev/null 2>&1 ;;
  esac || true
}

bold ""
bold "Your model is ready."
say "Chat on this computer:  $LOCAL_URL"
[ -n "$PUBLIC_URL" ] && say "Public link to share:    $PUBLIC_URL   (anyone with it can chat; it ends when you stop this script)"
say "Stop everything with Ctrl+C."
[ "$NO_BROWSER" = 1 ] || open_browser "$LOCAL_URL"
wait "${PIDS[0]}"
