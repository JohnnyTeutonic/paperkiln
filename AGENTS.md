# AGENTS.md — paperkiln (microtorch)

Instructions for any coding agent (Codex, Claude, others) working in this repository.
Most of this file is about Google Colab, because that is where agents waste money.

---

## Colab: the rules that stop wasted compute

The author pays for Colab compute units. A session left running with nothing
watching it burns units for hours. Every rule below was learned by losing units
or runs. Follow them exactly; do not improvise a new launch pattern.

### 0. Never

- **Never use the local GPU** (RTX 4060 laptop) for training or CUDA work. It has
  forced machine restarts three times. Colab by default; local only if the author
  says so.
- **Never leave a session running** at the end of your turn unless a driver you
  started is actively watching it and you have told the author so.
- **Never press "Terminate other sessions"** in the Colab web UI. It kills
  working VMs too.
- **Never update the CLI** (`colab update`, `uv tool install -U ...`). The tools
  here are written against version 0.6.0; ignore the upgrade banner.
- **Never delete a `HALTED` file** in an arm's output directory. It means the arm
  was shown it cannot complete as configured. The runner refuses to launch while
  it exists, on purpose.
- **Never edit a pre-registered sweep** (`experiments/transfer_s*/sweep_*.json`,
  anything beside a `PREREGISTRATION.md`) or the frozen `analyze.py` next to it.
- **Never `pkill -f <pattern>`** where the pattern appears in your own command
  line; it kills your own shell. Use pid files or bracket the pattern
  (`[s]upervisor.py`).

### 1. Where things are

| What | Path (inside WSL) |
|---|---|
| Colab CLI | `~/.local/bin/colab` (runs in WSL Ubuntu only, OAuth already set up) |
| CLI's own Python (needed by the adopt tool) | `~/.local/share/uv/tools/google-colab-cli/bin/python` |
| Session keys the CLI holds | `~/.config/colab-cli/sessions.json` |
| Repo from WSL | `/mnt/c/Users/jonat/OneDrive/Documents/research_portfolio_complete/microtorch` |
| Large run outputs and checkpoints | `/mnt/c/ml_artifacts/<study>/...` (outside OneDrive on purpose) |
| mtsweep arm driver | `tools/colab_transfer_runner.py` |
| Multi-job coordinator | `../tools/colab_supervisor/supervisor.py` (repo root, not microtorch) |
| Orphan recovery | `tools/colab_adopt.py`, `tools/colab_reap_orphans.py` |
| CUDA test gate (run on a VM) | `tools/colab_cuda_validate.sh` |

### 2. Preflight: do all of this before creating any session

Run from WSL. From Windows, write a `.sh` file and run it with
`wsl bash -c "bash '/mnt/c/.../script.sh'"`. Do **not** pass variables, `;`
chains or `/mnt/c` paths inline through Git Bash or PowerShell into
`wsl bash -c '...'`: the paths get mangled and variables arrive empty.

```bash
COL=~/.local/bin/colab
$COL sessions                     # 1. auth works, and what is ALREADY running
python3 tools/colab_reap_orphans.py   # 2. list "[?]" orphans (listing only; no --kill)
~/.local/share/uv/tools/google-colab-cli/bin/python tools/colab_adopt.py --list
```

Then check:

1. **Is someone else's job running?** A named session you did not create belongs to
   a live driver. Leave it alone.
2. **Are there `[?]` orphans?** These are live VMs whose key the CLI lost after a
   single 404/401. They hold a GPU slot and burn units. If one is plainly yours or
   plainly dead, stop it with `colab_reap_orphans.py --kill` (it will not touch
   named sessions). If it may be another driver's work in progress, adopt it
   instead: `colab_adopt.py --adopt <endpoint>=<name>`.
3. **Slots.** At most **three** concurrent GPU sessions in practice (measured
   13 Sep 2026; a fourth returns `TooManyAssignmentsError` or `Precondition
   Failed`). `supervisor.py` still says `MAX_SESSIONS = 4` from an older
   measurement; plan for three.
4. **`HALTED`?** `ls /mnt/c/ml_artifacts/<study>/<arm>/HALTED`. If present, stop
   and tell the author.
5. **Budget.** Ask the author before any job expected to use more than a few GPU
   hours. Say the GPU type, the number of sessions and the expected hours.

### 3. Pick exactly one launch path

**A. An mtsweep arm (a `sweep_*.json`): use `tools/colab_transfer_runner.py`.**

```bash
python3 tools/colab_transfer_runner.py \
    --sweep experiments/<study>/sweep_X.json \
    --session <short-name> \
    --local-out /mnt/c/ml_artifacts/<study>/<arm> \
    --expect <number of runs> --gpu L4 --jobs 4 --max-hours 8
```

It is idempotent: re-running the same command resumes from what is already in
`--local-out`. It builds the CUDA binary once and caches it per GPU and source
commit; it relays finished runs every tick and partial checkpoints every
`--partial-every` seconds; it refreshes the session key before the hourly expiry;
it adopts its own orphaned VM instead of creating a new one; and it stops the
session when the arm is complete or the deadline passes. Use `--shard K/N` with a
different `--session` per VM to split an arm. `--gpu L4` is the study GPU for any
claim-carrying arm; never mix T4 and L4 inside one arm. T4 is for probes only.

**B. A job with its own boot script and resume artifact: add it to `supervisor.py`.**
Admission rule, both halves required: the job checkpoints far more often than
VMs are reclaimed, **and** its `pull()` brings that checkpoint home **and** its
`uploads()` pushes it back on relaunch. A checkpoint nobody downloads restarts
from step 0 on every reclaim. Then `python3 supervisor.py status`,
`supervisor.py run`, `supervisor.py stop` (pid file), `supervisor.py reap`.

**C. A one-off under two hours** (a CUDA validation, a short probe): one script,
run as one unit — `new`, upload, exec, download, `stop` — never as separate tool
calls with gaps between them (sessions get reclaimed between upload and exec).
Always pass an explicit `--timeout` to `exec`/`run` (the default is 30 seconds
and silently kills real work), and always end with `colab stop -s <name>` even on
failure (`trap` it).

### 4. Keeping the driver alive (the WSL trap)

WSL tears the distro down about eight seconds after the last `wsl.exe` session
exits, and kills every background process with it. `nohup ... &`, `setsid`, and
`Start-Process -WindowStyle Hidden wsl ...` all die. The driver then stops
relaying and the VMs keep burning, unwatched. This is the worst case.

What works:

- Run the driver as a **foreground** process in a shell that stays open for the
  whole job: a terminal tab the author leaves open, or your own long-running
  command if your harness keeps it alive (not a fire-and-forget background call).
- **Verify a few minutes later**, not just at launch:
  `ps -o pid=,etime=,args= -C python3` should show the driver with a growing
  elapsed time, and its log should have advanced.
- If you cannot keep a process alive for the job's duration, **do not launch**.
  Give the author the exact command to run in a WSL tab instead.

### 5. Monitoring: how to tell alive from dead

- **Session alive** = the control-plane listing (`colab sessions`) shows
  `[<name>]`. Do **not** decide a session is dead because a `colab exec` probe was
  slow: exec takes 5 to 15 minutes against a busy VM, and killing it "to recover"
  destroys in-flight runs.
- **Work alive** = a separate check that the process and its outputs exist on the
  VM; require two consecutive failures before re-provisioning.
- **Progress** = results counted on local disk (`--local-out`), not the VM's log.
- **Hung stream** = the local driver log has not been modified for more than
  15 minutes while a session exists. Stop the driver by pid and restart the same
  command (it resumes).
- **Lifetime.** Headless CLI sessions die at around two hours whatever you do;
  keys expire hourly. Any work unit must fit inside that or checkpoint and relay
  inside it (paths A and B do this; path C must be sized under two hours).

### 6. Data movement limits

- Uploads over about 100 MB fail; the tools chunk anything over 32 to 45 MB.
  Downloads are fine; log downloads truncate at about 52 KB.
- The home uplink is slow (0.2 to 0.5 MB/s). Every relaunch re-uploads the vocab
  (58 MB), the cached binary and any resume checkpoints. Prefer fewer sessions over
  more when resume payloads are large.
- `git archive` on this Windows box writes CRLF. A `.sh` unpacked on a VM dies with
  `pipefail\r`. Fix on the VM: `find . -name "*.sh" -exec sed -i "s/\r$//" {} +`.
- Google Drive cannot be an unattended checkpoint store (OAuth grant per VM).
- WSL `/tmp` is wiped without warning. Outputs go under `/mnt/c/ml_artifacts/` or
  `~/`, and receipts are copied into `experiments/<study>/receipts/` as soon as an
  arm completes, before any analysis.

### 7. Finish every turn clean

```bash
~/.local/bin/colab sessions          # must show only sessions a live driver owns
python3 tools/colab_reap_orphans.py   # must list no "[?]" orphans
```

Report to the author: which sessions exist, which driver owns each, where its
log is, how many runs are banked locally, and the exact command to resume.
If a session exists that no running driver owns, stop it.

---

## Other working rules for this repository

- Do not touch the author's uncommitted work-in-progress files; check
  `git status` and leave modified files you did not create alone.
- Tests that write and reopen files must write to a temp directory, not the
  OneDrive tree (WSL can wedge on OneDrive file handles). Build and run test
  suites on ext4 (`~/mtrel`, `~/mtdbg`) against the source tree.
- Tests use `tests/check.hpp` `CHECK()`, never bare `assert()` (it vanishes in
  Release builds).
- Every experiment keeps its receipts (`events.jsonl`, `result.json`, analysis
  output) in the repo. A claim without a receipt is not a result.
- Pre-registered studies: rules are frozen at the licence commit. Changes are
  dated amendments in the pre-registration file, never silent edits.
