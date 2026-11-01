"""Materialize a private source-bound normal-runner boot-function-only amendment."""

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

FROZEN = Path("/home/jonat/pk_sfull_operation_v2_20261101")
CONTROL = "1af22cfd66f95cd40dcd98cf4ffae57cb3de4c61"
NUMERIC = "1ccb693f55d7328d4a8b0b7238fe8a2cafb6edf1"
ARCHIVE = "0ba62855b7fa19f9fe9248ba7edae3b04a124f9cde2aa2f764f4a7680eb7fcd9"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(destination):
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    helper = Path(__file__).with_name("sfull_preboot_recovery_v1.py").read_bytes()
    policy = {
        "origin": "https://github.com/JohnnyTeutonic/paperkiln.git",
        "control_commit": CONTROL,
        "numerical_commit": NUMERIC,
        "control_archive_sha256": ARCHIVE,
        "control_archive_bytes": 388659200,
    }
    policy_data = (json.dumps(policy, indent=2) + "\n").encode()
    (destination / "recovery_helper.py").write_bytes(helper)
    (destination / "policy.json").write_bytes(policy_data)
    boot = (FROZEN / "tools/colab_chat/boot_sfull_v2_20261101.sh").read_bytes()
    capsule = r"""import hashlib,json,os,subprocess,sys,urllib.request
from pathlib import Path
sys.path.insert(0,'/content')
import sfull_preboot_recovery_v1 as recovery
EXPECTED_HELPER=HELPER_HASH
EXPECTED_POLICY=POLICY_HASH
EXPECTED_BOOT=BOOT_HASH

def main():
 control,jobs=sys.argv[1:]
 if control!=CONTROL_VALUE or jobs!='tools/colab_chat/carry12_sfull_v2_r1.json':raise ValueError('original boot arguments differ')
 helper=Path('/content/sfull_preboot_recovery_v1.py');policy_path=Path('/content/sfull_preboot_policy_v1.json')
 if hashlib.sha256(helper.read_bytes()).hexdigest()!=EXPECTED_HELPER or hashlib.sha256(policy_path.read_bytes()).hexdigest()!=EXPECTED_POLICY:raise ValueError('recovery payload differs')
 policy=json.loads(policy_path.read_text())
 recovery.recover('/content',policy['origin'],control,policy['numerical_commit'],policy['control_archive_sha256'],policy['control_archive_bytes'])
 url='https://raw.githubusercontent.com/JohnnyTeutonic/paperkiln/'+control+'/tools/colab_chat/boot_sfull_v2_20261101.sh'
 raw=urllib.request.urlopen(url,timeout=120).read()
 if hashlib.sha256(raw).hexdigest()!=EXPECTED_BOOT:raise ValueError('original frozen boot bytes differ')
 boot=Path('/content/boot_jobs.sh')
 if boot.exists() and hashlib.sha256(boot.read_bytes()).hexdigest()!=EXPECTED_BOOT:raise ValueError('existing boot file differs; preserve for review')
 temp=boot.with_suffix('.verified.tmp');temp.write_bytes(raw);os.replace(temp,boot)
 return subprocess.call(['bash',str(boot),control,jobs])

try:
 raise SystemExit(main())
except recovery.ActiveOwner:
 print('source recovery refused active owner; no source/status overwrite',flush=True)
 raise SystemExit(75)
except Exception:
 Path('/content/chat/STATUS').write_text('BOOT_FAILED'+chr(10))
 raise
"""
    for key, value in {
        "HELPER_HASH": sha(helper),
        "POLICY_HASH": sha(policy_data),
        "BOOT_HASH": sha(boot),
        "CONTROL_VALUE": CONTROL,
    }.items():
        capsule = capsule.replace(key, repr(value))
    capsule_data = capsule.encode()
    compile(capsule, "capsule", "exec")
    (destination / "capsule.py").write_bytes(capsule_data)
    # Reproduce original repo-relative lookup paths without copying frozen source or data.
    for item in FROZEN.iterdir():
        if item.name not in {".git", "tools"}:
            (destination / item.name).symlink_to(
                item, target_is_directory=item.is_dir()
            )
    tools = destination / "tools"
    tools.mkdir()
    chat = tools / "colab_chat"
    chat.mkdir()
    for item in (FROZEN / "tools").iterdir():
        if item.name != "colab_chat":
            (tools / item.name).symlink_to(item, target_is_directory=item.is_dir())
    for item in (FROZEN / "tools/colab_chat").iterdir():
        if item.name != "runner_multi.py":
            (chat / item.name).symlink_to(item, target_is_directory=item.is_dir())
    payloads = {
        "/content/sfull_preboot_recovery_v1.py": helper,
        "/content/sfull_preboot_policy_v1.json": policy_data,
        "/content/sfull_boot_recovery_capsule_v1.py": capsule_data,
    }
    local = {
        str(destination / "recovery_helper.py"): sha(helper),
        str(destination / "policy.json"): sha(policy_data),
        str(destination / "capsule.py"): sha(capsule_data),
    }
    original = (FROZEN / "tools/colab_chat/runner_multi.py").read_text()
    tree = ast.parse(original)
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "boot_launch_code"
    )
    lines = original.splitlines(keepends=True)
    old = "".join(lines[node.lineno - 1 : node.end_lineno])
    function = (
        Path(__file__).with_name("sfull_recovery_boot_function_v1.py.in").read_text()
    )
    function = (
        function.replace("__CONTROL__", repr(CONTROL))
        .replace("__LOCAL_PINS__", repr(local))
        .replace(
            "__PAYLOADS__",
            repr({k: base64.b64encode(v).decode() for k, v in payloads.items()}),
        )
    )
    amended = (
        "".join(lines[: node.lineno - 1]) + function + "".join(lines[node.end_lineno :])
    )
    compile(amended, "runner", "exec")
    path = chat / "runner_multi.py"
    path.write_text(amended)
    newtree = ast.parse(amended)
    newnode = next(
        n
        for n in newtree.body
        if isinstance(n, ast.FunctionDef) and n.name == "boot_launch_code"
    )
    newlines = amended.splitlines(keepends=True)
    reverse = (
        "".join(newlines[: newnode.lineno - 1])
        + old
        + "".join(newlines[newnode.end_lineno :])
    )
    assert reverse == original
    proof = {
        "kind": "NORMAL_RUNNER_BOOT_FUNCTION_ONLY_RECOVERY_V1",
        "original_runner": str(FROZEN / "tools/colab_chat/runner_multi.py"),
        "original_runner_sha256": sha(original.encode()),
        "modified_runner": str(path),
        "modified_runner_sha256": sha(amended.encode()),
        "exact_reverse_restores_original": True,
        "only_function_changed": "boot_launch_code",
        "control_commit_unchanged": CONTROL,
        "numerical_commit_unchanged": NUMERIC,
        "original_boot_sha256": sha(boot),
        "payload_pins": local,
        "no_provider_or_config_action": True,
    }
    (destination / "proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination), indent=2))
