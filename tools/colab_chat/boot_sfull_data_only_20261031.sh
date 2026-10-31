#!/usr/bin/env bash
set -euo pipefail
COMMIT=$1
JOBS=$2
mkdir -p /content/chat/upload
exec >> /content/chat/boot.log 2>&1
trap 'echo "data-only boot failed at line $LINENO"; echo BOOT_FAILED > /content/chat/STATUS' ERR
if [ ! -d /content/pk/.git ]; then
  git clone -q https://github.com/JohnnyTeutonic/paperkiln.git /content/pk
fi
git -C /content/pk checkout -q "$COMMIT"
test "$(git -C /content/pk rev-parse HEAD)" = "$COMMIT"
cd /content/pk
python3 tools/colab_chat/provenance.py /content/chat/provenance.json
pip -q install huggingface_hub pandas pyarrow tokenizers==0.23.3
# Run the original pinned fetcher. Do not rewrite its expected hashes.
python3 corpus/fetchers/fineweb_edu.py shard0_v1
# The hash file uses relative names, so validate from its corpus directory.
(cd corpus/data/fineweb_edu_shard0_v1 && sha256sum -c ../../hashes/fineweb_edu_shard0_v1.sha256)
if pgrep -f "[j]obs_vm.py" >/dev/null; then
  echo 'existing jobs_vm process; refuse duplicate'
  exit 1
fi
echo RUNNING > /content/chat/STATUS
nohup bash -c 'python3 -u tools/colab_chat/jobs_vm.py "$1" > /content/chat/jobs.log 2>&1; rc=$?; python3 tools/colab_chat/sfull_data_only_finish_20261031.py --jobs-exit "$rc" || { printf "JOBS_FAILED\n" > /content/chat/STATUS; exit 1; }' gate "$JOBS" >/dev/null 2>&1 &
