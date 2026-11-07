#!/usr/bin/env bash
# Jarvis VM boot prelude (runner_multi.py --backend jarvis, jarvis_transport.py): run over ssh as ubuntu on
# every new VM before prepare() and boot_jobs.sh, so the VM looks like the Colab VM boot_jobs.sh expects.
# The Jarvis filesystem is at /home/jl_fs; /content becomes a link to the campaign's folder on it, so
# /content/chat and /content/pk are Colab's paths and survive the VM. Idempotent.
#   bash jarvis_prelude.sh COMMIT CAMPAIGN
# Unlike runpod_prelude.sh nothing here is best-effort: any failed step ends the script non-zero, the
# runner destroys the VM, and no job runs on a half-prepared machine. Jobs run as ubuntu, not root.
# Tests set PK_FS_ROOT, PK_CONTENT and PK_CUDA_DIR; a VM uses the defaults.
set -euo pipefail
COMMIT=$1
CAMPAIGN=$2
FS_ROOT=${PK_FS_ROOT:-/home/jl_fs}
CONTENT=${PK_CONTENT:-/content}
CUDA_DIR=${PK_CUDA_DIR:-/usr/local/cuda}
CUDA_RUN=cuda_12.4.1_550.54.15_linux.run  # the same CUDA as RunPod's image, so the sm_89 build matches
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/$CUDA_RUN
CUDA_MD5=afc99bab1d8c6579395d851d948ca3c1  # NVIDIA's md5sum.txt for 12.4.1, read 7 Nov 2026
CMAKE_VERSION=3.31.6
trap 'echo "jarvis prelude failed at line $LINENO"' ERR
echo "=== jarvis prelude $(date -u) commit $COMMIT campaign $CAMPAIGN ==="

# ssh answers about 30 s after create, while cloud-init may still hold apt. rc 2 is "done, with warnings".
if command -v cloud-init > /dev/null; then
  rc=0
  timeout 300 cloud-init status --wait > /dev/null || rc=$?
  case $rc in
    0|2) ;;
    *) echo "cloud-init did not finish (rc $rc)"; exit 4 ;;
  esac
fi

# The GPU must be the study GPU, whatever the API said.
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)
if [ "$gpu" != "NVIDIA L4" ]; then echo "GPU is '$gpu', not NVIDIA L4: refused"; exit 3; fi
cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\)\..*/\1/p')
echo "host driver CUDA major: ${cuda:-unknown}"
if [ -z "$cuda" ] || [ "$cuda" -lt 12 ]; then echo "host CUDA driver older than 12: refused"; exit 3; fi

# The filesystem must really be the virtiofs mount: a plain folder here would lose every checkpoint with the VM.
fstype=$(findmnt -n -o FSTYPE --target "$FS_ROOT")
if [ "$fstype" != virtiofs ]; then echo "$FS_ROOT is on $fstype, not virtiofs: refused"; exit 5; fi
if [ -e "$CONTENT" ] && [ ! -L "$CONTENT" ]; then echo "$CONTENT exists and is not a link: refused"; exit 5; fi
sudo -n mkdir -p "$FS_ROOT/$CAMPAIGN"
sudo -n chown "$(id -u):$(id -g)" "$FS_ROOT/$CAMPAIGN"
sudo -n ln -sfn "$FS_ROOT/$CAMPAIGN" "$CONTENT"
mkdir -p "$CONTENT/chat/upload"
touch "$CONTENT/chat/.write_test"
rm "$CONTENT/chat/.write_test"

# A previous VM's process ids mean nothing here; jobs_vm.py's lock and the jobs' pid files would make it
# refuse to start, or adopt a stranger.
rm -f "$CONTENT"/chat/jobs_vm.pid "$CONTENT"/chat/*.pid "$CONTENT/chat/STATUS" "$CONTENT/chat/provenance.json"
# Resume uploads begun for the old VM are void: prepare() decides afresh what goes up.
find "$CONTENT/chat/upload" -mindepth 2 -maxdepth 2 -type f -delete
# A build (and its clone) from another commit on this filesystem is never reused.
if [ -d "$CONTENT/pk/.git" ]; then
  head=$(git -C "$CONTENT/pk" rev-parse HEAD 2> /dev/null || echo none)
  case "$head" in
    "$COMMIT"*) echo "paperkiln build on the filesystem is at $head" ;;
    *) echo "paperkiln on the filesystem is at $head, not $COMMIT: removed"; rm -rf "${CONTENT:?}/pk" ;;
  esac
fi

# Jobs are started over ssh and must outlive it.
sudo -n loginctl enable-linger "$(id -un)"

# What boot_jobs.sh uses that the VM lacks. Ubuntu 22.04's pip predates --root-user-action, so it is not passed.
APT=(sudo -n env DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 -o Acquire::Retries=3)
for attempt in 1 2 3; do
  if "${APT[@]}" update -qq; then break; fi
  if [ "$attempt" = 3 ]; then echo "apt-get update failed three times"; exit 6; fi
  sleep 20
done
"${APT[@]}" install -y -q rsync procps git build-essential libboost-regex-dev python3-pip > /dev/null
sudo -n -H python3 -m pip -q install huggingface_hub pandas pyarrow tokenizers numpy "cmake==$CMAKE_VERSION"
hash -r
have=$(cmake --version | sed -n 's/^cmake version //p')
if [ "$have" != "$CMAKE_VERSION" ]; then echo "cmake is $have, not $CMAKE_VERSION: refused"; exit 7; fi

# nvcc: RunPod's image brings it, this VM may not, or may bring another release. ENV_PREAMBLE puts
# /usr/local/cuda/bin first on PATH for every exec, so this is the nvcc the build finds.
nvcc_release() { "$CUDA_DIR/bin/nvcc" --version 2> /dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p'; }
if [ "$(nvcc_release || true)" != 12.4 ]; then
  echo "installing the CUDA 12.4.1 toolkit"
  tmp=$(mktemp -d)
  curl -fsSL --retry 3 -o "$tmp/$CUDA_RUN" "$CUDA_URL"
  (cd "$tmp" && echo "$CUDA_MD5  $CUDA_RUN" | md5sum -c --quiet -)
  sudo -n sh "$tmp/$CUDA_RUN" --silent --toolkit
  rm -rf "$tmp"
  [ -e "$CUDA_DIR" ] || sudo -n ln -s /usr/local/cuda-12.4 "$CUDA_DIR"
fi
if [ "$(nvcc_release || true)" != 12.4 ]; then echo "nvcc at $CUDA_DIR is not 12.4: refused"; exit 8; fi
echo "$CUDA_DIR/lib64" | sudo -n tee /etc/ld.so.conf.d/pk-cuda.conf > /dev/null
sudo -n ldconfig

nproc
free -g
df -h "$CONTENT/"
echo "prelude done $(date -u)"
