#!/usr/bin/env python3
"""Local driver for the first-wave screening session (microtorch/AGENTS.md path C; the root workspace's
AI_ML/chat_screens_claude/PROTOCOL.md v1.2 and SESSION.md). Run in a foreground WSL process from the paperkiln
checkout's root:

    python3 experiments/chat_screens_2026-11-04/drive.py FULL_COMMIT_SHA RUN_ID [SEED SCREENS [CAP_MINUTES]]

CAP_MINUTES (30-120, A4) sets an absolute session cap; deadlines then follow capped().

SEED (default 1) is the spec's train.seed; SCREENS (default S0,S1,S2,S3, wave 1) is a comma-separated list of the
screens this session trains, in order. Both are pinned in the input manifest (amendment A1, wave 2).

FULL_COMMIT_SHA is this experiment's commit on GitHub (vm_run.sh is fetched at it); the trainer is built on the VM
from the numerical source pinned in vm_run.sh. Session name claude-screens-RUN_ID. Outputs:
/mnt/c/ml_artifacts/chat_screens_claude/runs/RUN_ID/ (driver.log, staging/, one folder per screen holding model.gguf,
manifest.json and train.log, and run_RUN_ID/out/ with the VM's logs and status.json); the top-level files of out/
and every screen's manifest are copied to this experiment's receipts/RUN_ID/, with archive.sha256 hashing every
retrieved file, after the VM is gone.

Before any session is created, every input is hashed against PINS (the protocol's pins and PACKETS.md); a mismatch
creates nothing. After creation the inputs are uploaded, then hashed on the VM against the manifest; a part that
differs is uploaded again (at most twice). While the VM trains, each screen it marks READY is collected at once: its
parts are downloaded, hashed, joined and the whole GGUF hashed against the VM's manifest. A screen counts as home only
then. No screen is ever retrained by the driver.

Deadlines from the driver's start (DEADLINES, minutes): polling ends at 100, workload-stop attempts at 104,
retrieval (the final logs and any screen still uncollected) at 112, shutdown attempts at 117, the final inventory at
120: the session's 2 L4-hour cap (100 for sessions of three screens or fewer). Every external call has a timeout capped by the deadline that governs it; a call
past its deadline is not started; a timed-out child is killed.

Safeguards carried over from the Mamba scan driver (experiments/mamba_parallel_scan_2026-11-03/drive.py): the launch
publishes vm_run.sh atomically and a launch without acknowledgement is reconciled by a read-only probe, never blindly
repeated; read-only and idempotent calls are retried; a run that did not finish is stopped through the VM script's
published PID before retrieval; each retrieval extracts into its own staging folder; shutdown is certified only by an
authoritative session listing that shows neither our session name, nor the endpoint we owned, nor any endpoint that
was not there before our creation call.
Exit status:
  0  every gate passed, status.json matches run, commit, source and manifest, and every screen is home verified
  1  the experiment failed (a gate failed; screens already home and receipts are preserved)
  2  transport, receipt or validation could not be completed, a screen is not home, or a workload stop was not
     confirmed
  3  cleanup unresolved: a VM may still be running (the log names it and how to stop it)
Test hooks (DRIVE_TEST_*) are honoured only with DRIVE_TEST_HARNESS=1, and refused otherwise.
"""
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

TEST = os.environ.get("DRIVE_TEST_HARNESS") == "1"


def hook(name, default):
    return os.environ.get(name) if TEST and os.environ.get(name) else default


COL = hook("DRIVE_TEST_COLAB", os.path.expanduser("~/.local/bin/colab"))
# Section 37.2 (Jonathan, 10 Nov 2026): the session GPU is a launch setting, L4 unless DRIVE_GPU names another allowed type
GPU = os.environ.get("DRIVE_GPU", "L4")
if GPU not in ("L4", "A100"):
    raise SystemExit(f"DRIVE_GPU {GPU!r} is not an allowed session GPU (L4, A100)")
CLI_PY = hook("DRIVE_TEST_CLI_PY", os.path.expanduser("~/.local/share/uv/tools/google-colab-cli/bin/python"))
REAP = hook("DRIVE_TEST_REAP", f"{sys.executable} tools/colab_reap_orphans.py").split()
ADOPT = hook("DRIVE_TEST_ADOPT", f"{CLI_PY} tools/colab_adopt.py").split()
HERE = Path(__file__).resolve().parent
ARTIFACTS = Path(hook("DRIVE_TEST_ARTIFACTS", "/mnt/c/ml_artifacts/chat_screens_claude/runs"))
RECEIPTS = Path(hook("DRIVE_TEST_RECEIPTS", str(HERE / "receipts")))
MINUTE = float(hook("DRIVE_TEST_MINUTE_SECONDS", "60"))
LOG_FAIL_AFTER = int(hook("DRIVE_TEST_LOG_FAIL_AFTER", "0"))
RAISE_ON = hook("DRIVE_TEST_RAISE_ON", "")
PART_BYTES = int(hook("DRIVE_TEST_PART_BYTES", str(45 * 1024 * 1024)))  # runner_multi.py: 45 MiB both ways
SOURCE = "1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1"
SCREENS = ("S0", "S1", "S2", "S3")  # wave 1, the default
ALL_SCREENS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S24", "S25", "S26", "S27", "S21", "S22", "S23", "S28", "S29", "S30", "S31", "S32", "S33", "S34", "S35", "S36", "S37", "S38", "S39", "S40", "S41", "S42", "S43", "S44", "S45", "S46", "S47", "S48", "S49", "S50", "S51", "S52", "S53", "S54", "S55", "S56", "S57", "S58", "S61", "S62", "S63", "S64", "S65", "S66", "S67", "S68")
COMMON_INPUTS = ("vocab.gguf", "arm_a_lr3e5.json", "recipes.json")
DEFAULT_BASE = "sfull_sft1500"  # a screen starts from this unless its recipe names a "base" (A19)


def capped(cap):
    """Deadlines for an absolute session cap of `cap` minutes (A4): polling ends 10 minutes before the cap, stop
    attempts 8, retrieval 5, shutdown 3."""
    return (cap - 10, cap - 8, cap - 5, cap - 3, cap)


ST_MIN_LEFT = float(hook("DRIVE_TEST_ST_MIN_LEFT", "6"))  # minutes needed to start the safetensors fetch
# A18: training minutes per arm, whole minutes, by shape (L2 23, L4 26, L8 32, L16 38, L32 45, 60M 57). Used by launch_need only; the VM never reads them.
SCREEN_MINUTES = {"S35": 32, "S36": 45, "S37": 23, "S38": 38, "S39": 26, "S40": 23, "S41": 32, "S42": 45, "S43": 38, "S44": 26, "S45": 57, "S46": 57, "S47": 23, "S48": 45, "S49": 45, "S50": 45, "S54": 86, "S55": 43, "S56": 43, "S57": 43, "S58": 43, "S61": 57, "S62": 57, "S63": 57, "S64": 29, "S65": 29, "S66": 29, "S67": 29, "S68": 29}
# A18: (step, bar). val_loss at the first evaluation at or past the step, above the bar or not finite, ends the session once this arm is home. Later arms are never stopped.
EARLY_STOP = {"S35": (1024, 1.88)}
# A20: the same read at the pre-set cut-off, logged. It stops nothing: each arm is alone in its session, and a kill on a bar set from 31 rows costs more than it saves (PROTOCOL.md section 28).
EARLY_READ = {"S54": (1024, 2.09), "S55": (512, 2.65)}


def recipe(screen):
    """A screen's recipes.json entry (A9), empty if it has none."""
    try:
        rec = json.loads(Path(PINS["recipes.json"][0]).read_text())
    except (KeyError, OSError, ValueError):
        rec = {}
    return rec.get(screen, {})


def screen_steps(screen):
    """A screen's training steps: its recipes.json override (A9), else arm A's 256."""
    return int(recipe(screen).get("train.steps", 256))


def screen_base(screen):
    """The base a screen starts from (A19): each base is the pin <base>.safetensors."""
    return recipe(screen).get("base", DEFAULT_BASE)


def bases_of(screens):
    return tuple(sorted({screen_base(s) for s in screens}))


def part_stem(base):
    """Part names: base.partNN for the default base, so its copy on the HF repo is not sent again, <base>.partNN for others."""
    return "base" if base == DEFAULT_BASE else base


def st_excludes(st_home):
    """tar patterns: the safetensors parts of arms verified home stay out of the log archive. An arm not home keeps them, since the VM then holds the only copy."""
    return "".join(f' --exclude="screens/{s}/model.safetensors.part*"' for s in sorted(st_home))


def launch_need(screens):
    """Minutes needed between launch and the polling deadline: the build (about 2.5) and about 8.7 minutes of
    training per 256 steps per screen (wave 3, seed 1), with margin. The driver does not launch with less.
    `screens` is a count (arm A recipes) or a sequence of screen names (steps from recipes.json). A screen in
    SCREEN_MINUTES takes its own minutes, since the 9 minutes per 256 steps are S-full's shape."""
    timed = 0
    if not isinstance(screens, int):
        timed = sum(SCREEN_MINUTES.get(s, 0) for s in screens)
        screens = sum(screen_steps(s) / 256 for s in screens if s not in SCREEN_MINUTES)
    return float(hook("DRIVE_TEST_LAUNCH_NEED", str(3 + timed + 9 * screens)))


def required_gates(screens):
    return {"provenance", "checkout", "inputs", "configure", "build", "specs"} | {
        f"{g}_{s}" for s in screens for g in ("train", "package")}
DEADLINES = (100, 104, 112, 117, 120)  # four screens (wave 1)
DEADLINES_SHORT = (80, 84, 92, 97, 100)  # three or fewer screens (wave 2): keeps three sessions within the week
VM_IN = "/content/screens_in"
# The inputs' private Hugging Face dataset repo (Jonathan, 4 Nov 2026), seeded with hf_seed.py: the base parts and every
# other pinned file under their VM names. The token (scoped to that repo only) lives in HF_TOKEN_FILE, mode 600, and is
# never printed; it is copied to our own VM only for the download and deleted there afterwards. Without the token file
# the driver uploads every input instead.
HF_REPO = hook("DRIVE_TEST_HF_REPO", "JohnnyTeutonic/chat-screens-inputs")
HF_TOKEN_FILE = Path(hook("DRIVE_TEST_HF_TOKEN_FILE", os.path.expanduser("~/.config/claude_screens/hf_token")))
# The repo commit hf_seed.py verified, written there after verification; downloads pin it (an immutable revision).
HF_REVISION_FILE = Path(hook("DRIVE_TEST_HF_REVISION_FILE", os.path.expanduser("~/.config/claude_screens/hf_revision")))

# Every input, by its name on the VM: (local path, SHA-256). PROTOCOL.md section 2 and PACKETS.md.
ART = "/mnt/c/ml_artifacts"
PINS = {
    "sfull_sft1500.safetensors": (
        f"{ART}/tiny_chat/c12_Sfull_r1_cont_v3_20261101/post_c12_Sfull_r1/sft/.checkpoint_v2/"
        "generation-bc6c1bfa7feb44bc9f13e4df158440a3/model.safetensors",
        "886e7db5dc2dc457f01d5f21847e5d63dc50e02785562d167d253e46c0f834f1"),
    "vocab.gguf": (f"{ART}/capacity/vocab/vocab_llama3_bpe.gguf",
                   "0b880aa073a7e27c0770d5ecc5215b55d6d415080d20bd3dc7b169ecabadf198"),
    "S0.jsonl": (f"{ART}/sfull_uplift_20261103/token-packet-v3/train.jsonl",
                 "a319c322b0d7a5f134ecd22223b8c7f6b11277794a8ed485ee0aaf085ad59602"),
    "S1.jsonl": (f"{ART}/chat_screens_claude/packets/v2/S1/train.jsonl",
                 "cd6e9d35b9ff9acc11f1cbd824f55b73923a2c520eddedeb93e9787de17d47c3"),
    "S2.jsonl": (f"{ART}/chat_screens_claude/packets/v2/S2/train.jsonl",
                 "efb36b697e07888aa63fe525733f9f5759844043e6f8a1351f763d79afe4e46d"),
    "S3.jsonl": (f"{ART}/chat_screens_claude/packets/v2/S3/train.jsonl",
                 "e623e241a776bd1acdc3dd9c5cb9f204c93bc2944dca41f83948421118f33272"),
    "S4.jsonl": (f"{ART}/chat_screens_claude/packets/w2/S4/train.jsonl",
                 "6d24b3bd3b83f2964737fdf74f952fbdbfe8c5b0f4fdb0c8fe6e294d5009011f"),
    "S5.jsonl": (f"{ART}/chat_screens_claude/packets/w3/S5/train.jsonl",
                 "cc1cd7f75665c00424037d68df630053bde18ee057b5b1cc07a1df58ededadf5"),
    # A4 template-diversity arms: S6 = D_low, S7 = D_medium, S8 = D_high
    "S6.jsonl": (f"{ART}/chat_screens_claude/packets/w4/D_low/train.jsonl",
                 "beb51b848eb9e87592e294204b49e92a5dcceed817d3070988851312e2e5117d"),
    "S7.jsonl": (f"{ART}/chat_screens_claude/packets/w4/D_medium/train.jsonl",
                 "18c2ed659bf3a375467fcef59626863fa9e66d4fcecb6c14b0a98cfd9c4c3ac1"),
    "S8.jsonl": (f"{ART}/chat_screens_claude/packets/w4/D_high/train.jsonl",
                 "fd5907163fc3da2683af4f7b17a7d9a50adc146c35cc04f777e68fee75aabfd0"),
    # A7 contrastive arms: S9 = CX (same fact, different operations), S10 = CU (matched different-fact control);
    # A8 format-only arm S11
    "S9.jsonl": (f"{ART}/chat_screens_claude/packets/a7/S9/train.jsonl",
                 "5368bf3f99b158426954e4c8b437fb464dcb0c97ed83a4decdfea1cb73f81c02"),
    "S10.jsonl": (f"{ART}/chat_screens_claude/packets/a7/S10/train.jsonl",
                  "dcebc46ac0d509f3df35b50cdc5a9c6426cbc8bae9f7d8269c205486e0b62c16"),
    "S11.jsonl": (f"{ART}/chat_screens_claude/packets/a7/S11/train.jsonl",
                  "2d5dafa8dc0b3bcc925cb8edc5606a0aeacaae3c335e452d92308a2ffd454ef2"),
    # A9 dose arms: S12 and S13 train the same packet under different recipes; S14 is the control (S2 data)
    "S12.jsonl": (f"{ART}/chat_screens_claude/packets/a9/S12/train.jsonl", "9719eeb1c20f5c45bd6ed802b10e456003d3435a7fdc3f242731c4683807496f"),
    "S13.jsonl": (f"{ART}/chat_screens_claude/packets/a9/S12/train.jsonl", "9719eeb1c20f5c45bd6ed802b10e456003d3435a7fdc3f242731c4683807496f"),
    "S14.jsonl": (f"{ART}/chat_screens_claude/packets/a9/S14/train.jsonl", "ad6e4ba0aaae1d92fbaa9bfbfc4f289df17243f6ba01c3bb7f6d335f15b01479"),
    # A10: S15 multi-scenario, S16 the existing grid, at matched exposure (PROTOCOL.md section 19)
    "S15.jsonl": (f"{ART}/chat_screens_claude/packets/a10/S15/train.jsonl", "ac96a2d418ca3c2bac19679d5e2abe52e6fc212f1f42176f112e24f497345569"),
    "S16.jsonl": (f"{ART}/chat_screens_claude/packets/a10/S16/train.jsonl", "f7d05b869f1e3c3a1971179d62417cf1b91a13926e6e9283ee95fedc1a74939d"),
    # A11 (section 20.1): S24 grid, S25 half human half grid, S26 half grid half base, S27 human
    "S24.jsonl": (f"{ART}/chat_screens_claude/packets/a11b/S24/train.jsonl", "9c9c26fc5c905ec2176877f21123c7416354f90217adff555acc6cbfe7878931"),
    "S25.jsonl": (f"{ART}/chat_screens_claude/packets/a11b/S25/train.jsonl", "32ed04621ee1e1eb1725f941d733ec5b06ac5f9cf5387c8334205ae78791f78b"),
    "S26.jsonl": (f"{ART}/chat_screens_claude/packets/a11b/S26/train.jsonl", "193b0ba2b6bc370cff35d7fbdf693a4dc46eef1328026ea6e6b47c5ee5a4b7c4"),
    "S27.jsonl": (f"{ART}/chat_screens_claude/packets/a11b/S27/train.jsonl", "b20f46b2f6c016f2c609010968935a70ab527fb80327d5a1eec9f5acb5cf86ea"),
    # A12 (section 21): S21 / S22 / S23 with 68 / 272 / 1,088 distinct instructions
    "S21.jsonl": (f"{ART}/chat_screens_claude/packets/a12/S21/train.jsonl", "2a66d0e01e8eb7183f4820c3e8e11e5b090acc6e79843f1c3bed021e2d016c37"),
    "S22.jsonl": (f"{ART}/chat_screens_claude/packets/a12/S22/train.jsonl", "e68b2ed7f67c2b0950962e481c51659671c749296571a7bf0a966feb2acbe552"),
    "S23.jsonl": (f"{ART}/chat_screens_claude/packets/a12/S23/train.jsonl", "ae9f593d08877ab96e9b0f67f657ba1d5697552e2b5d6011ba12d434fd568ec5"),
    # A14 (section 22): S28 H-graft, S29 H-LLM (hybrids at 30% of sampled supervised tokens)
    "S28.jsonl": (f"{ART}/chat_screens_claude/packets/a14/S28/train.jsonl", "25b70169db28e140b4a6f951eb01258237aa4bdab20dd4660f3e2519f57ff5f8"),
    "S29.jsonl": (f"{ART}/chat_screens_claude/packets/a14/S29/train.jsonl", "b62155cd3bcf9de2fa3b39d198f7a97ec837b6366566dfefb0567af9f4da550d"),
    # A15 (section 23): S30, S31 (hybrid with A12 instruction half, at 30% of sampled supervised tokens)
    "S30.jsonl": (f"{ART}/chat_screens_claude/packets/a15/S30/train.jsonl", "4b761d345af7ec37126b339c4c30bef703e5ef71ab9df4658aea551599cdf60b"),
    "S31.jsonl": (f"{ART}/chat_screens_claude/packets/a15/S31/train.jsonl", "00ec274aff7df7762b5085fe67615c20e7630a669f4b5aa9156908d4cbff7eb0"),
    # A16 (section 24): S32, S33, S34 (paired contrast data; S34 is S32's packet at 2,048 steps)
    "S32.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S33.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S33/train.jsonl", "12818666ddeae11a96d928a9e0636cacb93cf72dc2019785f45f9454f53c5685"),
    "S34.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    # A19 (section 27): S51 M1 and S52 C0 from S34, S53 M2 from S30; S52 is S34's own packet, S34's and S30's exported weights are bases
    "s34_a16.safetensors": (f"{ART}/chat_screens_claude/runs/a16-s1-S34-20261106/S34/model.safetensors", "7a0562f01da0f517314934646c5814c3d67aac2864664b84e1f89c6e248ba457"),
    "s30_a15.safetensors": (f"{ART}/chat_screens_claude/runs/a15-s1-S30-20261106/S30/model.safetensors", "332d1c2bcede7d40b1f25b2ece3be9b0b13a50941c77beb5282bfffeab7d4933"),
    "S51.jsonl": (f"{ART}/chat_screens_claude/packets/a19/M1/train.jsonl", "5d7a022081363972766dc5bfc13754b7f6a32e2c21208eae88c03fa822dfe9e1"),
    "S52.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S53.jsonl": (f"{ART}/chat_screens_claude/packets/a19/M2/train.jsonl", "58113f54d69f8d3697d40f1ff8945ef4e3234be8b9117c4782a07dfa2f02f270"),
    # A18 (section 26): depth for width. S35 to S44 and the rate arms train S32's packet; each base is the tc_ export. S45 to S48 are pinned only if their branch opens: A18 later pins
    "S35.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S36.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S37.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S38.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S39.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S40.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S41.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S42.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S43.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S44.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S49.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S50.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "tc_e1_L2_r1.safetensors": (f"{ART}/tiny_chat/e1/pt_e1_L2_r1/tc_e1_L2_r1.safetensors", "763c4f38c2c1c890481f9895cb47d164b5d2f7e758563a34a9c072bd1fd70700"),
    "tc_e1_L2_r2.safetensors": (f"{ART}/tiny_chat/e1/pt_e1_L2_r2/tc_e1_L2_r2.safetensors", "6d56a0433ce378a3641d0904b516a96c18d755c3e9dfe41377815f27a1d19419"),
    "tc_e1_L4_r1.safetensors": (f"{ART}/tiny_chat/e1/pt_e1_L4_r1/tc_e1_L4_r1.safetensors", "f4f5fc323e8d5856ddcc90e21b95e70d9fb202861854faa080774a6319648c1a"),
    "tc_e1_L4_r2.safetensors": (f"{ART}/tiny_chat/e1/pt_e1_L4_r2/tc_e1_L4_r2.safetensors", "42ee6b5d01308a3fa9e483eae4e747337595831787399ad587074dfe9f474f7c"),
    "tc_e5_s35_r1.safetensors": (f"{ART}/tiny_chat/e5/pt_e5_s35_r1/tc_e5_s35_r1.safetensors", "0b6485e241224a0a07407caa55e7e03346b7e31264ba2daa122e531d4d6da956"),
    "tc_e5_s35_r3.safetensors": (f"{ART}/tiny_chat/e5/pt_e5_s35_r3/tc_e5_s35_r3.safetensors", "b8d47042a0bcab9ae77089ef1671c4197f7fbd891efd230004dcd0c85df4e00b"),
    "tc_e1_L16_r1.safetensors": (f"{ART}/tiny_chat/e1/pt_e1_L16_r1/tc_e1_L16_r1.safetensors", "b8d5e4099653aa52b1bc9b46f4793f4072e1f03e4d7dd973ed3f0c4217216559"),
    "tc_e1_L16_r2.safetensors": (f"{ART}/tiny_chat/e1/pt_e1_L16_r2/tc_e1_L16_r2.safetensors", "7e59c4989bec3754a076750da4e52d2325768a84ecf5d7960a08f13bc3f46810"),
    "tc_e1_L32_r1.safetensors": (f"{ART}/tiny_chat/e1/pt_e1_L32_r1/tc_e1_L32_r1.safetensors", "86fc7216c1da122ecfd66e8c43b2e8a0effaf0dfb30a91f61c3df4b8f95507ff"),
    "tc_e1_L32_r2.safetensors": (f"{ART}/tiny_chat/e1/pt_e1_L32_r2/tc_e1_L32_r2.safetensors", "02b11e03185413131883bfd24249d6bb80a64d970ff9dc5ff8bf9758f274f34e"),
    # A20 (section 28): S34's and S30's recipes from October's pk_sft_v1, at 12x768, k = 0, 8x2. S54 trains S32's packet, S55 S30's
    "S54.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S55.jsonl": (f"{ART}/chat_screens_claude/packets/a15/S30/train.jsonl", "4b761d345af7ec37126b339c4c30bef703e5ef71ab9df4658aea551599cdf60b"),
    "pk_sft_v1.safetensors": (f"{ART}/general_chat/sft/pk_sft_v1.safetensors", "eae61755dc6625eb87c54ce3152774596ed4299c65c0462ed3831b0f9d7ee80e"),
    # Lab step 1 (section 33): the balanced fixed mixture at seeds 1 and 2, each packet placed for its seed, and the mixed-row specialist
    "S56.jsonl": (f"{ART}/chat_screens_claude/packets/lab_n1/N1/train.jsonl", "42990b4b3043213425103e23c617fcd6bf528cd19625a6b2debd38f9971d8935"),
    "S57.jsonl": (f"{ART}/chat_screens_claude/packets/lab_n1/N2/train.jsonl", "2a4c2bdb4befa9006d3e90e6ea48ce31b25992478b0ddfb592a6fe50040d3869"),
    "S58.jsonl": (f"{ART}/chat_screens_claude/packets/lab_n1/SP/train.jsonl", "64673f9fdfa6ce8dfc527a29385e028849ecdc172155e057aceb2e1acb78ab1d"),
    # A21 (section 36): S34's and S30's recipes and their 15:15 mixture from A's SFT1500, at S34's shape and block. S61 trains S32's packet, S62 and S64 S30's, S63 the mixture
    "S61.jsonl": (f"{ART}/chat_screens_claude/packets/a16/S32/train.jsonl", "ab0ba685f64f84b1446eb4198bcfdc07c69de0d3a8a6da35057df396ac20024a"),
    "S62.jsonl": (f"{ART}/chat_screens_claude/packets/a15/S30/train.jsonl", "4b761d345af7ec37126b339c4c30bef703e5ef71ab9df4658aea551599cdf60b"),
    "S63.jsonl": (f"{ART}/chat_screens_claude/packets/a21/S63/train.jsonl", "edab09d7d8428bcae4bf97ddd3f86aad2d54ea45a1ceb59cb66762ba0f8c1670"),
    "S64.jsonl": (f"{ART}/chat_screens_claude/packets/a15/S30/train.jsonl", "4b761d345af7ec37126b339c4c30bef703e5ef71ab9df4658aea551599cdf60b"),
    "tc_sft_b1_r1.safetensors": (f"{ART}/tiny_chat/b1_r1/sft_b1_r1/tc_sft_b1_r1.safetensors", "86de03a5b1455ebe13792ac0be944b3eb14b851567ffb89249ee4fae2e1c8dbd"),
    # A22 (section 37): from A's SFT1500 at S64's recipe. S65 trains S14's packet (S2's rows), S66 S30's packet de-canned, S67 S2's rows de-canned and the K2 pool
    "S65.jsonl": (f"{ART}/chat_screens_claude/packets/a9/S14/train.jsonl", "ad6e4ba0aaae1d92fbaa9bfbfc4f289df17243f6ba01c3bb7f6d335f15b01479"),
    "S66.jsonl": (f"{ART}/chat_screens_claude/packets/a22/S66/train.jsonl", "659f3c9989bd19239ac6551f0587f54c2df4703d76ec17bb1afa3d2d70cb3793"),
    "S67.jsonl": (f"{ART}/chat_screens_claude/packets/a22/S67/train.jsonl", "a921f0388a0318067eac854f7c2858fb7a98e011fc9fb78c0056bbc7a2b01270"),
    # A23 (section 38): S68 trains S67's packet with S30's identity rows restored, at S64's recipe from A's SFT1500
    "S68.jsonl": (f"{ART}/chat_screens_claude/packets/a23/S68/train.jsonl", "395af1e8b7ace59351def36aa77d9d7a0afc9bacedd41d057fded68d1ec8d29e"),
    # A18 later pins
    "recipes.json": (str(HERE / "recipes.json"), "85b51cfd00165f49c25cc5f44cfe71bcc302e8f60a31ef6a04eddd9c5d59a3df"),
    "arm_a_lr3e5.json": (str(HERE / "arm_a_lr3e5.json"),
                         "66c718e0b501c751aae0a1cd22eddfdc666a7cfb46a4c074fa41b67aa220125e"),
}
if TEST and os.environ.get("DRIVE_TEST_PINS"):
    PINS = {k: tuple(v) for k, v in json.loads(os.environ["DRIVE_TEST_PINS"]).items()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def launch_code(url, commit, run, manifest_sha, seed=1, screens=SCREENS, root="/content"):
    """The VM-side launch: fetch vm_run.sh and start it detached. Each call writes its own temporary file and renames
    it over the final path, so a delayed duplicate never rewrites a script a bash is reading; vm_run.sh's atomic
    run-directory claim then lets only one launch do any work. (`root` is for tests only.)"""
    return (
        "import os, subprocess, tempfile, urllib.request\n"
        f"src = urllib.request.urlopen({url!r}).read().replace(b'\\r\\n', b'\\n')\n"
        f"fd, tmp = tempfile.mkstemp(prefix='vm_run_{run}.', suffix='.tmp', dir={root!r})\n"
        "with os.fdopen(fd, 'wb') as f:\n"
        "    f.write(src)\n"
        f"os.replace(tmp, '{root}/vm_run_{run}.sh')\n"
        f"subprocess.Popen('VM_GPU={GPU} nohup bash {root}/vm_run_{run}.sh {commit} {run} {manifest_sha} {seed} "
        f"{','.join(screens)} "
        f">> {root}/vm_run_{run}.log 2>&1 &', shell=True)\n"
        "print('LAUNCHED')\n")


class Stop(Exception):
    pass


class Driver:
    def __init__(self, commit, run, seed=1, screens=SCREENS, cap=None):
        self.commit, self.run, self.seed, self.screens, self.cap = commit, run, seed, tuple(screens), cap
        self.required = required_gates(self.screens)
        self.bases = bases_of(self.screens)
        self.st_home = set()
        self.needed = tuple(f"{b}.safetensors" for b in self.bases) + COMMON_INPUTS + tuple(f"{s}.jsonl" for s in self.screens)
        self.name = f"claude-screens-{run}"
        self.out = ARTIFACTS / run
        self.stage = self.out / "staging"
        self.receipts = RECEIPTS / run
        t = time.monotonic()
        self.work_end, self.stop_end, self.retrieve_end, self.shutdown_end, self.inventory_end = (
            t + m * MINUTE for m in (capped(cap) if cap else
                                     DEADLINES if len(self.screens) > 3 else DEADLINES_SHORT))
        self.intent = False
        self.endpoint = None
        self.before = None
        self.cancel = False
        self.log_file = None
        self.log_lines = 0
        self.raised = False
        self.log_failed = False
        self.done = False
        self.stop_confirmed = None
        self.launch_attempted = False
        self.manifest = None
        self.manifest_sha = None
        self.home = {}  # screen -> verified GGUF SHA-256
        self.early_read = {}  # A18: screen -> R at its read step, None if not finite
        self.stop_after = None

    # ---- logging never raises: the file is the record, the terminal and stderr best effort
    def log(self, *parts):
        line = f"{time.strftime('%H:%M:%S')} " + " ".join(str(p) for p in parts)
        try:
            self.log_lines += 1
            if LOG_FAIL_AFTER and self.log_lines > LOG_FAIL_AFTER:
                raise OSError("test: log write failed")
            self.log_file.write(line + "\n")
            self.log_file.flush()
        except (OSError, ValueError, AttributeError):
            self.log_failed = True
            try:
                sys.stderr.write("[log file unavailable] " + line + "\n")
            except (OSError, ValueError):
                pass
        try:
            print(line, flush=True)
        except (OSError, ValueError):
            pass

    def check(self):
        if self.cancel:
            raise Stop("cancelled by signal")

    def call(self, argv, want, deadline, label):
        """Run argv, timeout min(want, time to deadline). (rc, output), or (None, '') if not started, timed out
        (the child is killed) or failed to start. Never raises."""
        left = deadline - time.monotonic()
        if left <= 1:
            self.log(f"{label}: not started, deadline passed")
            return None, ""
        try:
            if RAISE_ON and RAISE_ON in argv and not self.raised:
                self.raised = True
                raise OSError("test: command failed to start")
            r = subprocess.run(argv, capture_output=True, text=True, timeout=min(want, left))
        except subprocess.TimeoutExpired:
            self.log(f"{label}: timed out, killed")
            return None, ""
        except OSError as e:
            self.log(f"{label}: could not run: {e}")
            return None, ""
        tail = " | ".join(r.stderr.strip().splitlines()[-2:])[:240] if r.returncode else ""
        self.log(f"{label}: exit {r.returncode}" + (f" ({tail})" if tail else ""))
        return r.returncode, r.stdout + r.stderr

    def sleep(self, seconds, deadline):
        end = min(time.monotonic() + seconds, deadline)
        while time.monotonic() < end and not self.cancel:
            time.sleep(min(1.0, max(0.0, end - time.monotonic())))

    def sessions(self, deadline):
        """Rows [(name, endpoint)] from an authoritative listing, or None if the listing failed or has a row
        that does not parse (fail closed)."""
        rc, out = self.call([COL, "sessions"], 120, deadline, "sessions")
        if rc != 0:
            return None
        rows = []
        for line in out.splitlines():
            if not line.startswith("["):
                continue
            m = re.match(r"^\[([^\]]+)\]\s+(\S+)", line)
            if not m:
                self.log("unparsed session row:", line)
                return None
            if m.group(1) == "colab":
                continue
            rows.append((m.group(1), m.group(2)))
        return rows

    # ---- phases
    def stage_inputs(self):
        """Local, before any session: every input hashes as pinned; the base weights are split into parts; the
        manifest pins every file and part. Raises Stop on any mismatch (nothing has been created)."""
        self.stage.mkdir(parents=True, exist_ok=False)
        unpinned = sorted(n for n in self.needed if n not in PINS)
        if unpinned:
            raise Stop(f"inputs not pinned: {unpinned}: nothing created")
        files = {}
        for name, (path, want) in PINS.items():
            if name not in self.needed:
                continue
            got = sha256(path)
            if got != want:
                raise Stop(f"input {name} hashes {got}, pinned {want}: nothing created")
            files[name] = want
        bases = {}
        for b in self.bases:
            parts = []
            with open(PINS[f"{b}.safetensors"][0], "rb") as f:
                for i in range(10 ** 6):
                    chunk = f.read(PART_BYTES)
                    if not chunk:
                        break
                    p = self.stage / f"{part_stem(b)}.part{i:02d}"
                    p.write_bytes(chunk)
                    parts.append({"name": p.name, "sha256": hashlib.sha256(chunk).hexdigest()})
            bases[b] = {"sha256": files[f"{b}.safetensors"], "parts": parts}
        self.manifest = {"files": files, "bases": bases, "screen_bases": {s: screen_base(s) for s in self.screens},
                         "arm_a_spec_sha256": files["arm_a_lr3e5.json"],
                         "source": SOURCE, "run_id": self.run, "seed": self.seed, "screens": list(self.screens)}
        man = self.stage / "manifest.json"
        man.write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        self.manifest_sha = sha256(man)
        self.log("inputs pinned and staged:", len(files), "files,", sum(len(v["parts"]) for v in bases.values()), "base parts, manifest",
                 self.manifest_sha)

    def uploads(self):
        """[(local path, VM name)] in upload order: the manifest, then the small inputs, then the base parts."""
        out = [(self.stage / "manifest.json", "manifest.json")]
        out += [(Path(PINS[n][0]), n) for n in self.needed if not n.endswith(".safetensors")]
        out += [(self.stage / p["name"], p["name"]) for v in self.manifest["bases"].values() for p in v["parts"]]
        return out

    def github_preflight(self):
        """Before anything is created: this checkout is at COMMIT and clean in this experiment, GitHub serves the
        same vm_run.sh at COMMIT, and the numerical source is on a remote branch (the VM clones it). Raises Stop."""
        if TEST:
            if os.environ.get("DRIVE_TEST_GITHUB") == "fail":
                raise Stop("github preflight failed (test)")
            self.log("github preflight: simulated under the test harness")
            return
        root = str(HERE.parents[1])
        rc, head = self.call(["git", "-C", root, "rev-parse", "HEAD"], 60, self.work_end, "git HEAD")
        if rc != 0 or head.strip() != self.commit:
            raise Stop(f"checkout HEAD {head.strip()} is not {self.commit}")
        rc, dirty = self.call(["git", "-C", root, "status", "--porcelain", "--", str(HERE),
                               f":(exclude){HERE / 'receipts'}"], 60, self.work_end, "git status")
        if rc != 0 or dirty.strip():
            raise Stop("this experiment has uncommitted changes")
        url = (f"https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/{self.commit}/"
               "experiments/chat_screens_2026-11-04/vm_run.sh")
        try:
            served = urllib.request.urlopen(url, timeout=60).read().replace(b"\r\n", b"\n")
        except OSError as e:
            raise Stop(f"vm_run.sh not served at {self.commit}: {e}")
        local = (HERE / "vm_run.sh").read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(served).digest() != hashlib.sha256(local).digest():
            raise Stop("GitHub serves a different vm_run.sh at this commit")
        rc, _ = self.call(["git", "-C", root, "fetch", "-q", "origin"], 300, self.work_end, "git fetch")
        if rc != 0:
            raise Stop("git fetch failed: remote branches unconfirmed")
        rc, branches = self.call(["git", "-C", root, "branch", "-r", "--contains", SOURCE], 120,
                                 self.work_end, "source on a remote branch")
        if rc != 0 or not branches.strip():
            raise Stop(f"numerical source {SOURCE} is on no remote branch")
        self.log("github preflight: HEAD, clean tree, served vm_run.sh and remote source all confirmed:",
                 branches.split()[0])

    def preflight(self):
        rows = self.sessions(self.work_end)
        if rows is None:
            raise Stop("preflight: session listing failed; not creating anything")
        self.log("preflight sessions:", json.dumps(rows))
        if any(n == self.name for n, _ in rows):
            raise Stop(f"preflight: a session named {self.name} exists; not ours to touch")
        for argv, label in ((REAP, "orphans"), (ADOPT + ["--list"], "adopt --list")):
            rc, out = self.call(argv, 300, self.work_end, label)
            self.log(f"{label}:\n" + out.strip())
            if rc != 0:
                raise Stop(f"preflight: {label} failed")
        self.before = {ep for _, ep in rows}

    def create(self):
        self.intent = True
        self.call([COL, "new", "-s", self.name, "--gpu", GPU], 600, self.work_end, "new")
        for _ in range(3):
            rows = self.sessions(self.work_end)
            if rows is not None:
                mine = [ep for n, ep in rows if n == self.name]
                if len(mine) == 1:
                    self.endpoint = mine[0]
                    self.log("owned session", self.name, "endpoint", self.endpoint)
                    return
                new = sorted({ep for _, ep in rows} - self.before)
                if new:
                    self.log("endpoints new since creation, not under our name:", new)
            self.sleep(MINUTE / 3, self.work_end)
            self.check()
        raise Stop("creation not confirmed under our name")

    def exec_py(self, code, want, deadline, label):
        p = Path(f"/tmp/claude_screens_{self.run}_{label}.py")
        p.write_text(code)
        return self.call([COL, "exec", "-s", self.name, "-f", str(p), "--timeout", str(max(30, want - 60))],
                         want, deadline, label)

    def vm_hashes(self, names):
        """{name: sha256 or None} for files in VM_IN, from one read-only exec (retried), or None if it failed."""
        code = ("import hashlib, json, os\n"
                f"names = {sorted(names)!r}\n"
                "def sha(p):\n"
                "    h = hashlib.sha256()\n"
                "    with open(p, 'rb') as f:\n"
                "        for b in iter(lambda: f.read(1 << 20), b''):\n"
                "            h.update(b)\n"
                "    return h.hexdigest()\n"
                f"print('HASHES ' + json.dumps({{n: (sha(os.path.join({VM_IN!r}, n)) if os.path.isfile("
                f"os.path.join({VM_IN!r}, n)) else None) for n in names}}))\n")
        for attempt in (1, 2, 3, 4):  # read-only, so retried freely (wave 3 seed 1: one failed, the next passed)
            self.check()
            rc, out = self.exec_py(code, 300, self.work_end, f"vm_hashes{attempt}")
            for line in out.splitlines():
                if rc == 0 and line.startswith("HASHES "):
                    try:
                        return json.loads(line[len("HASHES "):])
                    except ValueError:
                        pass
            self.sleep(MINUTE / 6, self.work_end)
        return None

    def hf_fetch(self, names):
        """Have the VM download `names` from HF_REPO into VM_IN with the scoped token, then delete the token file on
        the VM. Returns True if the download step ran, completely or partly (hashes are checked separately, and what
        did not arrive intact is uploaded). Never raises."""
        if not HF_TOKEN_FILE.is_file() or not HF_REVISION_FILE.is_file():
            self.log("no HF token or verified revision: inputs will be uploaded")
            return False
        revision = HF_REVISION_FILE.read_text().strip()
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            self.log("HF revision file malformed: inputs will be uploaded")
            return False
        self.log("HF fetch from", HF_REPO, "at revision", revision)
        code = ("import os, time\n"
                "from huggingface_hub import hf_hub_download\n"
                f"tok_path = os.path.join({VM_IN!r}, '.hf_token')\n"
                "ok = True\n"
                "t = time.monotonic()\n"
                "try:\n"
                "    os.chmod(tok_path, 0o600)\n"
                "    token = open(tok_path).read().strip()\n"
                f"    for n in {sorted(names)!r}:\n"
                "        try:\n"
                f"            hf_hub_download({HF_REPO!r}, n, repo_type='dataset', revision={revision!r}, token=token, local_dir={VM_IN!r})\n"
                "        except Exception as e:\n"
                "            ok = False\n"
                "            print('FAILED', n, type(e).__name__)\n"
                "except Exception as e:\n"
                "    ok = False\n"
                "    print('FAILED setup', type(e).__name__)\n"
                "finally:\n"
                "    if os.path.exists(tok_path):\n"
                "        os.remove(tok_path)\n"
                "print('HF_DONE' if ok else 'HF_PARTIAL', round(time.monotonic() - t))\n")
        # Two attempts (a fetch takes about 25 s; a stalled exec waits for its timeout). Each uploads the token
        # afresh, since the VM-side code deletes it whatever happens.
        for attempt, timeout in ((1, 300), (2, 180)):
            self.check()
            rc = None
            for up in (1, 2):
                rc, _ = self.call([COL, "upload", "-s", self.name, str(HF_TOKEN_FILE), f"{VM_IN}/.hf_token"],
                                  300, self.work_end, f"upload token a{attempt}u{up}")
                if rc == 0:
                    break
            if rc != 0:
                self.remove_token()  # the upload may have landed although the call failed
                return False
            rc, out = self.exec_py(code, timeout, self.work_end, "hf_fetch")
            self.log(f"hf_fetch attempt {attempt}:", " ".join(out.split()[-6:]))
            # HF_PARTIAL still counts as having run: the hash check that follows decides what must be uploaded
            if rc == 0 and ("HF_DONE" in out.split() or "HF_PARTIAL" in out.split()):
                return True
            self.remove_token()
        return False

    def remove_token(self):
        """Best effort: delete the token file on our VM and confirm it is gone (two attempts). Never raises."""
        gone = ("import os\n"
                f"p = os.path.join({VM_IN!r}, '.hf_token')\n"
                "os.path.exists(p) and os.remove(p)\n"
                "print('TOKEN_GONE' if not os.path.exists(p) else 'TOKEN_PRESENT')\n")
        for attempt in (1, 2):
            rc, out = self.exec_py(gone, 120, self.work_end, f"token_remove{attempt}")
            if rc == 0 and "TOKEN_GONE" in out.split():
                return
        self.log("the token file may remain on our VM until it stops")

    def upload(self):
        """Inputs onto the VM: the per-run manifest is uploaded; the data files come from the HF repo when the token
        file exists, else they are uploaded. Then everything is hashed on the VM; what differs is uploaded directly,
        at most twice more. True when every file on the VM matches the manifest."""
        # The first exec on a new session can stall in the local CLI (wave 3, seeds 2-3: two sessions created in
        # the same second; the VM itself was answering). Retry until VM_IN exists, else stop before any upload.
        for attempt in range(1, 5):
            self.check()
            rc, out = self.exec_py(f"import os\nos.makedirs({VM_IN!r}, exist_ok=True)\n"
                                   f"print('READY' if os.path.isdir({VM_IN!r}) else 'NO_DIR')\n", 120,
                                   self.work_end, "mkdir_in")
            if rc == 0 and "READY" in out.split():
                break
            self.log(f"VM not ready for inputs (attempt {attempt}); waiting")
            self.sleep(MINUTE / 2, self.work_end)
        else:
            raise Stop("the VM never answered: no input was uploaded")
        want = {n: sha256(p) for p, n in self.uploads()}
        pending = self.uploads()
        data = [n for _, n in self.uploads() if n != "manifest.json"]
        if self.hf_fetch(data):
            got = self.vm_hashes(want) or {}
            pending = [(p, n) for p, n in self.uploads() if got.get(n) != want[n]]
            self.log(f"after HF fetch, {len(pending)} file(s) still to upload:", [n for _, n in pending])
            if not pending:
                return True
        for round_ in (1, 2, 3):
            base = [n for _, n in pending if re.fullmatch(r".+\.part\d{2}", n)]
            if self.cap and base:
                raise Stop(f"under a {self.cap}-minute cap the base weights are not uploaded directly "
                           f"({len(base)} part(s) missing or wrong on the VM): not launching")
            for local, name in pending:
                self.check()
                for attempt in (1, 2):
                    rc, _ = self.call([COL, "upload", "-s", self.name, str(local), f"{VM_IN}/{name}"], 1800,
                                      self.work_end, f"upload {name} r{round_}a{attempt}")
                    if rc == 0:
                        break
            got = self.vm_hashes(want)
            if got is None:
                self.log("could not hash the uploads on the VM")
                return False
            bad = sorted(n for n in want if got.get(n) != want[n])
            self.log(f"upload round {round_}: {len(want) - len(bad)} of {len(want)} verified on the VM", bad)
            if not bad:
                return True
            pending = [(p, n) for p, n in self.uploads() if n in bad]
        return False

    def launch_state(self):
        code = ("import os, time\n"
                f"pid, run = '/content/vm_{self.run}.pid', '/content/run_{self.run}'\n"
                "for _ in range(90):\n"
                "    if os.path.exists(pid) or os.path.isdir(run):\n"
                "        print('STARTED'); raise SystemExit\n"
                "    time.sleep(1)\n"
                "print('ABSENT')\n")
        for attempt in (1, 2, 3):
            self.check()
            rc, out = self.exec_py(code, 200, self.work_end, f"launch_probe{attempt}")
            if rc == 0 and ("STARTED" in out.split() or "ABSENT" in out.split()):
                return "started" if "STARTED" in out.split() else "absent"
        return None

    def launch(self):
        url = (f"https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/{self.commit}/"
               "experiments/chat_screens_2026-11-04/vm_run.sh")
        code = launch_code(url, self.commit, self.run, self.manifest_sha, self.seed, self.screens)
        for label in ("launch", "launch_retry"):
            self.check()
            self.launch_attempted = True
            rc, out = self.exec_py(code, 200, self.work_end, label)
            if rc == 0 and "LAUNCHED" in out:
                return True
            state = self.launch_state()
            self.log(f"{label}: no acknowledgement; the VM script is {state or 'unknown'}")
            if state == "started":
                return True
            if state is None:
                return False
        return False

    def vm_state(self, deadline, label):
        """(done, [ready screens]) from one read-only exec, or (None, None) if it failed."""
        code = ("import json, os\n"
                f"o = '/content/run_{self.run}/out'\n"
                "def holds(p):\n"
                "    try:\n"
                f"        return open(p).read().strip() == {self.run!r}\n"
                "    except OSError:\n"
                "        return False\n"
                f"ready = [s for s in {list(self.screens)!r} if holds(f'{{o}}/screens/{{s}}/READY')]\n"
                "print('STATE ' + json.dumps({'done': holds(o + '/DONE'), 'ready': ready}))\n")
        rc, out = self.exec_py(code, 200, deadline, label)
        for line in out.splitlines():
            if rc == 0 and line.startswith("STATE "):
                try:
                    s = json.loads(line[len("STATE "):])
                    return bool(s["done"]), [x for x in s["ready"] if x in self.screens]
                except (ValueError, KeyError, TypeError):
                    pass
        return None, None

    def read_eval(self, screen, step, deadline):
        """The first evaluation at or past `step` in the VM's train log: (step, value), value None if not finite; None if there is none yet; False if the read failed. Read-only."""
        code = ("import json, re\n"
                f"step, got = {step}, None\n"
                "try:\n"
                f"    for line in open('/content/run_{self.run}/out/train_{screen}.log', errors='replace'):\n"
                "        if '\"eval\"' not in line:\n"
                "            continue\n"
                "        s, v = re.search(r'\"step\":\\s*(\\d+)', line), re.search(r'\"val_loss\":\\s*([^,}\\s]+)', line)\n"
                "        if s and v and int(s.group(1)) >= step:\n"
                "            got = [int(s.group(1)), v.group(1)]\n"
                "            break\n"
                "except OSError:\n"
                "    pass\n"
                "print('EVAL ' + json.dumps(got))\n")
        rc, out = self.exec_py(code, 200, deadline, "early_read")
        for line in out.splitlines():
            if rc == 0 and line.startswith("EVAL "):
                try:
                    got = json.loads(line[len("EVAL "):])
                except ValueError:
                    return False
                if got is None:
                    return None
                try:
                    value = float(got[1])
                except ValueError:
                    value = math.inf
                return int(got[0]), value if math.isfinite(value) else None
        return False

    def early_check(self):
        """A18's readout: each arm in EARLY_STOP is read once. A failed read is logged and is not a stop; it is tried again at the next poll. A20's arms in EARLY_READ are read the same way and stop nothing."""
        for s, (step, bar) in {**EARLY_STOP, **EARLY_READ}.items():
            if s not in self.screens or s in self.early_read:
                continue
            got = self.read_eval(s, step, self.work_end)
            if got is False:
                self.log(f"{s}: early read failed: not a stop")
            elif got is not None:
                at, value = got
                self.early_read[s] = value
                over = value is None or value > bar
                if s in EARLY_READ:
                    self.log(f"{s}: R({at}) = {value}, bar {bar}, read only:", "over the bar" if over else "under the bar")
                    continue
                self.log(f"{s}: R({at}) = {value}, bar {bar}:", f"stop once {s} is home" if over else "continues")
                if over:
                    self.stop_after = s

    def collect(self, screen, deadline):
        """Download one READY screen's manifest, parts and log; hash every part; join; hash the GGUF. Only a fully
        verified screen is promoted to OUT/screen. True on success. Never raises."""
        try:
            remote = f"/content/run_{self.run}/out/screens/{screen}"
            work = self.out / f"collect_{screen}"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            rc, _ = self.call([COL, "download", "-s", self.name, f"{remote}/manifest.json",
                               str(work / "manifest.json")], 300, deadline, f"{screen} manifest")
            if rc != 0:
                return False
            man = json.loads((work / "manifest.json").read_text())
            if man.get("screen") != screen or man.get("run_id") != self.run:
                self.log(f"{screen}: manifest identity mismatch", man.get("screen"), man.get("run_id"))
                return False
            for part in man["parts"]:
                name = part["name"]
                if not re.fullmatch(r"model\.gguf\.part\d{2}", name):
                    self.log(f"{screen}: refusing part name", name)
                    return False
                for attempt in (1, 2):
                    rc, _ = self.call([COL, "download", "-s", self.name, f"{remote}/{name}", str(work / name)],
                                      900, deadline, f"{screen} {name} a{attempt}")
                    if rc == 0 and (work / name).is_file() and sha256(work / name) == part["sha256"]:
                        break
                else:
                    self.log(f"{screen}: part {name} not verified")
                    return False
            with open(work / "model.gguf", "wb") as g:
                for part in man["parts"]:
                    with open(work / part["name"], "rb") as f:
                        shutil.copyfileobj(f, g, 1 << 20)
            if sha256(work / "model.gguf") != man["gguf_sha256"]:
                self.log(f"{screen}: joined GGUF hash differs from the VM's manifest")
                return False
            for part in man["parts"]:
                (work / part["name"]).unlink()
            for extra in ("train.log", "result.json"):
                self.call([COL, "download", "-s", self.name, f"{remote}/{extra}", str(work / extra)], 300,
                          deadline, f"{screen} {extra}")
                want = man.get(extra.replace(".", "_") + "_sha256")
                if not (work / extra).is_file() or sha256(work / extra) != want:
                    self.log(f"{screen}: {extra} missing or not as the VM hashed it")
                    return False
            if screen in SCREEN_MINUTES:   # A18: what the scorer reads beside the weights, here in case the VM is lost before its logs archive. Best effort: the archive holds them too
                for extra in (f"spec_{screen}.json", f"train_{screen}_check.json"):
                    self.call([COL, "download", "-s", self.name, f"/content/run_{self.run}/out/{extra}", str(work / extra)], 120, deadline, f"{screen} {extra}")
            (work).rename(self.out / screen)
            self.home[screen] = man["gguf_sha256"]
            self.log(f"{screen}: HOME, GGUF {man['gguf_sha256']} ({man['gguf_bytes']} bytes)")
            # A10: the safetensors comes only after the screen is home, and only with time to spare
            if man.get("st_parts"):
                # A18: no arm follows the last one, so nothing needs the time up to the stop deadline
                st_end = self.retrieve_end if screen in SCREEN_MINUTES and screen == self.screens[-1] else deadline
                if (st_end - time.monotonic()) / MINUTE >= ST_MIN_LEFT:
                    self.collect_safetensors(screen, remote, self.out / screen, man, st_end)
                else:
                    self.log(f"{screen}: under {ST_MIN_LEFT} minutes left; the safetensors stays on the VM")
            return True
        except (OSError, ValueError, KeyError, TypeError) as e:
            self.log(f"{screen}: collection failed:", repr(e))
            return False

    def collect_safetensors(self, screen, remote, work, man, deadline):
        """The run's exported safetensors (A10 parity check), in parts like the GGUF, each hash-verified, then joined
        and checked. Optional: on any failure the screen still comes home with its verified GGUF, without
        model.safetensors, and the log says so. Never raises."""
        ok = True
        for part in man["st_parts"]:
            name = part["name"]
            if not re.fullmatch(r"model\.safetensors\.part\d{2}", name):
                self.log(f"{screen}: refusing safetensors part name", name)
                ok = False
                break
            got = False
            for attempt in (1, 2):
                rc, _ = self.call([COL, "download", "-s", self.name, f"{remote}/{name}", str(work / name)],
                                  900, deadline, f"{screen} {name} a{attempt}")
                if rc == 0 and (work / name).is_file() and sha256(work / name) == part["sha256"]:
                    got = True
                    break
            if not got:
                self.log(f"{screen}: safetensors part {name} not verified")
                ok = False
                break
        if ok:
            with open(work / "model.safetensors", "wb") as g:
                for part in man["st_parts"]:
                    with open(work / part["name"], "rb") as f:
                        shutil.copyfileobj(f, g, 1 << 20)
            if sha256(work / "model.safetensors") != man.get("st_sha256"):
                self.log(f"{screen}: joined safetensors hash differs from the VM's manifest")
                ok = False
        if not ok:
            (work / "model.safetensors").unlink(missing_ok=True)
            self.log(f"{screen}: the GGUF comes home without its safetensors")
        else:
            self.st_home.add(screen)
            self.log(f"{screen}: safetensors {man['st_sha256']} ({man['st_bytes']} bytes)")
        for p in work.glob("model.safetensors.part*"):
            p.unlink()
        return ok

    def poll(self):
        while time.monotonic() < self.work_end and not self.cancel:
            self.sleep(2 * MINUTE, self.work_end)
            if self.cancel:
                break
            self.call(ADOPT + ["--refresh"], 180, self.work_end, "refresh")
            done, ready = self.vm_state(self.work_end, "poll")
            if done is None:
                continue
            self.early_check()
            for s in ready:
                if s not in self.home:
                    self.collect(s, self.work_end)
            if done:
                return True
            # S35 finishes and comes home first, so a stop is never blind. Its safetensors, if it did not arrive, stays in the logs archive: st_excludes keeps an arm not home.
            if self.stop_after in self.home and self.screens[-1] != self.stop_after:
                self.log(f"early stop: {self.stop_after} is home, the arms after it are cancelled")
                self.cancel = True
        return False

    def stop_workload(self):
        """As the Mamba driver: SIGTERM the VM script's published PID (checked to be ours); its handler stops the
        running step's process group, writes stop.json and exits through the trap that writes status.json and DONE.
        True (interrupted or completed), False (not confirmed) or None (no launch was attempted)."""
        if not self.launch_attempted:
            self.log("workload stop: no launch was attempted")
            return None
        code = (
            "import json, os, signal, time\n"
            f"run, script = {self.run!r}, '/content/vm_run_{self.run}.sh'\n"
            f"pid_file, done, stop = '/content/vm_{self.run}.pid', '/content/run_{self.run}/out/DONE', "
            f"'/content/run_{self.run}/out/stop.json'\n"
            "def reply(result, **kw):\n"
            "    print('STOP_REPLY ' + json.dumps(dict(result=result, **kw))); raise SystemExit\n"
            "def done_ok():\n"
            "    try:\n"
            "        return open(done).read().strip() == run\n"
            "    except OSError:\n"
            "        return False\n"
            "for _ in range(60):\n"
            "    if os.path.exists(pid_file):\n"
            "        break\n"
            "    time.sleep(1)\n"
            "else:\n"
            "    reply('not_observed')\n"
            "pid = int(open(pid_file).read())\n"
            "try:\n"
            "    ours = script.encode() in open(f'/proc/{pid}/cmdline', 'rb').read()\n"
            "except OSError:\n"
            "    ours = False\n"
            "if ours and not done_ok():\n"
            "    try:\n"
            "        os.kill(pid, signal.SIGTERM)\n"
            "    except ProcessLookupError:\n"
            "        pass\n"
            "for _ in range(120):\n"
            "    if done_ok():\n"
            "        break\n"
            "    time.sleep(1)\n"
            "s = json.load(open(stop)) if os.path.exists(stop) else None\n"
            "if not done_ok():\n"
            "    reply('unconfirmed', stop=s, signalled=ours)\n"
            "if s is None:\n"
            "    reply('completed', signalled=ours)\n"
            "gone = s.get('step_group_gone') is True or s.get('step_started') is False\n"
            "reply('interrupted' if gone else 'unconfirmed', stop=s, signalled=ours)\n")
        result, out = None, ""
        for attempt in (1, 2):
            rc, out = self.exec_py(code, 260, self.stop_end, f"stop_workload{attempt}")
            reply = None
            for line in out.splitlines():
                if line.startswith("STOP_REPLY "):
                    try:
                        reply = json.loads(line[len("STOP_REPLY "):])
                    except ValueError:
                        reply = None
            result = reply.get("result") if rc == 0 and isinstance(reply, dict) else None
            if result is not None:
                break
        confirmed = result in ("interrupted", "completed")
        self.log("workload stop:", f"confirmed ({result})" if confirmed else
                 f"NOT CONFIRMED ({result}): outputs may still be changing", out.strip()[-300:])
        return confirmed

    def retrieve(self):
        """The VM's logs and status (without the GGUF parts), then any READY screen not yet home."""
        code = ("import subprocess\n"
                f"subprocess.run('cp /content/vm_run_{self.run}.log /content/run_{self.run}/out/ 2>/dev/null', "
                "shell=True)\n"
                f"r = subprocess.run('tar czf /content/screens_{self.run}.tgz --warning=no-file-changed "
                f"--exclude=\"model.gguf.part*\"{st_excludes(self.st_home)} -C /content run_{self.run}/out', shell=True)\n"
                "print('PACKED' if r.returncode in (0, 1) else 'PACK_FAILED %d' % r.returncode)\n")
        archive = self.out / "logs.tgz"
        prefix = f"run_{self.run}/out"
        got_logs = False
        for attempt in (1, 2):
            archive.unlink(missing_ok=True)
            rc, out = self.exec_py(code, 300, self.retrieve_end, f"pack{attempt}")
            if rc != 0 or "PACKED" not in out:
                continue
            rc, _ = self.call([COL, "download", "-s", self.name, f"/content/screens_{self.run}.tgz", str(archive)],
                              600, self.retrieve_end, f"download logs {attempt}")
            if rc != 0 or not archive.is_file():
                continue
            stage = self.out / f"extract_{attempt}"
            try:
                shutil.rmtree(stage, ignore_errors=True)
                with tarfile.open(archive) as t:
                    members = t.getmembers()
                    bad = [m.name for m in members if not (m.isfile() or m.isdir())
                           or not (m.name == prefix or m.name.startswith(prefix + "/")) or ".." in Path(m.name).parts]
                    if bad:
                        self.log("refusing archive members", bad[:5])
                        continue
                    t.extractall(stage, members=members)
                (stage / f"run_{self.run}").rename(self.out / f"run_{self.run}")
                shutil.rmtree(stage, ignore_errors=True)
                got_logs = True
                break
            except (tarfile.TarError, OSError) as e:
                self.log(f"archive attempt {attempt} unreadable:", e)
        if not got_logs:
            self.log("LOGS NOT RETRIEVED: the VM holds the only copy, and cleanup will stop it")
        _, ready = self.vm_state(self.retrieve_end, "final_state")
        for s in ready or []:
            if s not in self.home:
                self.collect(s, self.retrieve_end)
        return got_logs

    def copy_receipts(self):
        """Receipts from whatever came home: the VM's top-level out/ files when the logs came back, and in every
        case the input manifest, every screen's manifest, result and log hashes, and home.json."""
        src = self.out / f"run_{self.run}" / "out"
        try:
            self.receipts.mkdir(parents=True, exist_ok=False)
            if src.is_dir():
                for f in src.iterdir():
                    if f.is_file():
                        shutil.copy2(f, self.receipts / f.name)
            if (self.out / "input_manifest.json").is_file():
                shutil.copy2(self.out / "input_manifest.json", self.receipts / "input_manifest.json")
            for s in self.screens:
                if (self.out / s / "manifest.json").is_file():
                    shutil.copy2(self.out / s / "manifest.json", self.receipts / f"{s}_manifest.json")
            lines = []
            for f in sorted(p for p in self.out.rglob("*") if p.is_file() and "staging" not in p.parts
                            and p.name != "driver.log"):
                lines.append(f"{sha256(f)}  {f.relative_to(self.out).as_posix()}\n")
            tmp = self.receipts / "archive.sha256.tmp"
            tmp.write_text("".join(lines))
            tmp.rename(self.receipts / "archive.sha256")
            (self.receipts / "home.json").write_text(json.dumps(self.home, indent=2, sort_keys=True))
            return True
        except Exception as e:
            self.log("receipt copy failed:", repr(e))
            return False

    def validate(self):
        """True: validated pass. False: a well-formed status of a failed experiment. None: missing or malformed."""
        out = self.out / f"run_{self.run}" / "out"
        try:
            s = json.loads((out / "status.json").read_text())
        except (OSError, ValueError) as e:
            self.log("status.json missing or unreadable:", e)
            return None
        gates = s.get("gates") if isinstance(s, dict) else None
        if not isinstance(gates, dict) or not all(type(v) is int for v in gates.values()) \
                or type(s.get("exit_status")) is not int or not isinstance(s.get("pass"), bool):
            self.log("status.json malformed:", json.dumps(s)[:500])
            return None
        if s.get("run_id") != self.run or s.get("commit") != self.commit or s.get("source") != SOURCE \
                or s.get("head") != SOURCE or s.get("manifest_sha256") != self.manifest_sha:
            self.log("status.json identity mismatch:", s.get("run_id"), s.get("commit"), s.get("source"),
                     s.get("head"), s.get("manifest_sha256"))
            return None
        problems = []
        if set(gates) != self.required:
            problems.append(f"gate set differs: missing {sorted(self.required - set(gates))}, "
                            f"extra {sorted(set(gates) - self.required)}")
        if s.get("seed") != self.seed or s.get("screens") != list(self.screens):
            problems.append(f"seed or screens differ: {s.get('seed')} {s.get('screens')}")
        failed = sorted(k for k, v in gates.items() if v != 0)
        if failed:
            problems.append(f"failed gates {failed}")
        if s["exit_status"] != 0:
            problems.append(f"VM exit status {s['exit_status']}")
        if s["pass"] is not True:
            problems.append("pass is not true")
        try:
            stop = json.loads((out / "stop.json").read_text()) if (out / "stop.json").exists() else {}
        except (OSError, ValueError):
            stop = {"step_group_gone": None}
        if (out / "QUIESCENCE_UNCONFIRMED").exists() or stop.get("step_group_gone", True) is not True:
            self.log("quiescence not confirmed by the VM script")
            self.stop_confirmed = False
        self.log("status.json:", json.dumps(s))
        self.log("validation:", "PASS" if not problems else "; ".join(problems))
        return not problems

    def cleanup(self):
        if not self.intent:
            return True
        for attempt in range(1, 6):
            try:
                if time.monotonic() >= self.shutdown_end:
                    break
                self.call([COL, "stop", "-s", self.name], 120, self.shutdown_end, f"stop {attempt}")
                time.sleep(max(0.0, min(MINUTE / 6, self.shutdown_end - time.monotonic())))  # not cancellable
                rows = self.sessions(self.shutdown_end)
                if rows is None:
                    continue
                names = {n for n, _ in rows}
                endpoints = {ep for _, ep in rows}
                # Only an unnamed [?] row can be a VM our creation call allocated; named rows (parallel sibling
                # sessions included) belong to their names.
                unnamed = {ep for n, ep in rows if n == "?"}
                new = unnamed - (self.before or set()) - ({self.endpoint} if self.endpoint else set())
                ours_listed = self.name in names or (self.endpoint is not None and self.endpoint in endpoints)
                if not ours_listed and not new:
                    self.log(f"CLEANUP OK: {self.name} ({self.endpoint}) and no new endpoint listed")
                    return True
                if self.endpoint is not None and self.endpoint in endpoints and self.name not in names:
                    self.call(ADOPT + ["--adopt", f"{self.endpoint}={self.name}"], 180, self.shutdown_end, "adopt")
                if new:
                    self.log("endpoints new since our creation call, ownership not established:", sorted(new))
            except Exception as e:
                self.log(f"cleanup attempt {attempt} error:", repr(e))
        self.log(f"CLEANUP UNRESOLVED: {self.name} (endpoint {self.endpoint}) may still be running, or a VM our "
                 f"creation call allocated is unnamed. Inspect `{COL} sessions`; stop ours with "
                 f"`{COL} stop -s {self.name}` or adopt the endpoint first (tools/colab_adopt.py --adopt EP=NAME).")
        return False

    def main(self):
        try:
            self.out.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"{self.out} exists: refusing to reuse a run folder", file=sys.stderr)
            return 2
        try:
            self.log_file = open(self.out / "driver.log", "a")
        except OSError:
            self.log_file = None
        self.log(f"run {self.run} commit {self.commit} source {SOURCE} session {self.name} gpu {GPU}"
                 + (" [TEST HARNESS]" if TEST else ""))
        logs = receipts = False
        valid = None
        try:
            self.stage_inputs()
            self.github_preflight()
            self.preflight()
            self.check()
            self.create()
            self.check()
            if not self.upload():
                raise Stop("inputs not verified on the VM: not launching")
            left, need = (self.work_end - time.monotonic()) / MINUTE, launch_need(self.screens)
            if left < need:
                raise Stop(f"too late to launch: {left:.1f} minutes to the polling deadline, {need:g} needed for "
                           f"{len(self.screens)} screen(s)")
            launched = self.launch()
            done = self.poll() if launched else False
            self.done = done
            self.log("launched", launched, "done", done, "cancelled", self.cancel, "home", sorted(self.home))
            if not done:
                self.stop_confirmed = self.stop_workload()
            logs = self.retrieve()
            valid = self.validate() if logs else None
        except Stop as e:
            self.log("stopped:", e)
            if self.launch_attempted and not self.done:
                self.stop_confirmed = self.stop_workload()
                logs = self.retrieve()
                valid = self.validate() if logs else None
        except Exception as e:
            self.log("error:", repr(e))
        finally:
            clean = False
            try:
                clean = self.cleanup()
            except Exception as e:
                self.log("cleanup error:", repr(e))
            for label, argv in (("final sessions", [COL, "sessions"]), ("final orphans", REAP)):
                rc, out = self.call(argv, 120, self.inventory_end, label)
                self.log(f"{label}:\n" + out.strip())
            if (self.stage / "manifest.json").is_file():
                shutil.copy2(self.stage / "manifest.json", self.out / "input_manifest.json")
            if self.out.joinpath("staging").exists():
                shutil.rmtree(self.out / "staging", ignore_errors=True)  # copies of pinned inputs only
            receipts = self.copy_receipts()
            all_home = set(self.home) == set(self.screens)
            code = 3 if not clean else 2 if not (logs and receipts and valid is not None) else 0 if valid else 1
            if receipts and not logs:
                self.log("receipts hold only what came home: the VM logs were not retrieved")
            if code == 0 and not (self.done and all_home):
                self.log(f"validated status but done {self.done}, home {sorted(self.home)}: exit 2")
                code = 2
            if self.log_failed:
                code = max(code, 2)
            if self.stop_confirmed is False:
                self.log("the workload stop was not confirmed: exit at least 2")
                code = max(code, 2)
            self.log(f"exit {code} (logs {logs}, receipts {receipts}, valid {valid}, cleanup {clean}, "
                     f"home {sorted(self.home)})")
            try:
                if self.log_file:
                    self.log_file.close()
                if receipts:
                    shutil.copy2(self.out / "driver.log", self.receipts / "driver.log")
            except OSError:
                code = max(code, 2)
            return code


def main():
    if not TEST and any(k.startswith("DRIVE_TEST_") for k in os.environ):
        print("DRIVE_TEST_* is set but DRIVE_TEST_HARNESS is not: refusing a real run with test hooks",
              file=sys.stderr)
        return 2
    args = sys.argv[1:]
    cap = None
    if len(args) == 5:
        if not re.fullmatch(r"[0-9]{2,3}", args[4]) or not 30 <= int(args[4]) <= 120:
            print(__doc__, file=sys.stderr)
            return 2
        cap = int(args.pop())
    if len(args) not in (2, 4) or not re.fullmatch(r"[0-9a-f]{40}", args[0]) \
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", args[1]) \
            or (len(args) == 4 and not (re.fullmatch(r"[1-9]", args[2])
                                        and re.fullmatch(r"S[0-9]{1,2}(,S[0-9]{1,2})*", args[3])
                                        and set(args[3].split(",")) <= set(ALL_SCREENS)
                                        and len(set(args[3].split(","))) == len(args[3].split(",")))):
        print(__doc__, file=sys.stderr)
        return 2
    seed, screens = (int(args[2]), args[3].split(",")) if len(args) == 4 else (1, SCREENS)
    d = Driver(args[0], args[1], seed, screens, cap)

    def on_signal(signum, _frame):
        d.cancel = True

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, on_signal)
    return d.main()


if __name__ == "__main__":
    sys.exit(main())
