#!/usr/bin/env bash
# Vendor keramos (the IR's specification, schema, reference tools and conformance suite) at a tag into
# spec/, and record the tag, its commit and the commit's manifest in spec/KERAMOS_PIN.json.
#
#   tools/sync_keramos.sh v1.1.0 [--overwrite-local]
#
# The source is a keramos checkout: KERAMOS_DIR, or the sibling ../keramos. Only the tag's committed
# tree is copied (git archive), never the checkout's working files. spec/ is keramos's: changes land
# in keramos first, are tagged there, then synced here. So this script refuses to run while spec/
# differs from the current pin (tools/keramos_pin.py check lists the files), unless
# --overwrite-local is given: port the local edits to keramos first.
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${KERAMOS_DIR:-$HERE/../keramos}"
DST="$HERE/spec"
TAG="$1"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
[ -n "$TAG" ] || { echo "usage: tools/sync_keramos.sh TAG [--overwrite-local]"; exit 2; }
[ -d "$SRC/.git" ] || { echo "no keramos checkout at $SRC (set KERAMOS_DIR)"; exit 1; }
git -c safe.directory='*' -C "$SRC" rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null \
    || { echo "keramos at $SRC has no tag $TAG"; exit 1; }
if [ "$2" != "--overwrite-local" ] && [ -f "$DST/KERAMOS_PIN.json" ]; then
    if ! "$PY" "$HERE/tools/keramos_pin.py" check > /dev/null; then
        echo "spec/ differs from its pinned keramos tag; the pinned tree is canonical:"
        "$PY" "$HERE/tools/keramos_pin.py" check || true
        echo "port the local edits to keramos (and tag), or pass --overwrite-local"
        exit 1
    fi
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git -c safe.directory='*' -C "$SRC" archive --format=tar "refs/tags/$TAG^{commit}" | tar -x -C "$TMP"
# Emptied file by file (OneDrive can hold a directory handle open), then the empty directories.
mkdir -p "$DST"
find "$DST" -mindepth 1 -type f -delete
find "$DST" -mindepth 1 -depth -type d -empty -delete
(cd "$TMP" && tar -cf - .) | (cd "$DST" && tar -xf -)
"$PY" "$HERE/tools/keramos_pin.py" write "$SRC" "$TAG"
"$PY" "$HERE/tools/keramos_pin.py" check
