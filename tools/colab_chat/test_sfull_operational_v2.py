import json
from pathlib import Path
from unittest.mock import patch
import pytest
import sfull_operational_v2 as op

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parents[1] / "docs/receipts/sfull_launch_v2_20261101/historical_fixture"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exact_overlay(seed):
    old = json.loads(
        (ORIGINAL / f"tools/colab_chat/carry12_sfull_r{seed}.json").read_text()
    )
    new = json.loads((HERE / f"carry12_sfull_v2_r{seed}.json").read_text())
    assert new == op.expected_jobs(old, seed)
    for a, b in zip(old["jobs"], new["jobs"]):
        if a["name"] not in ("gate", "sfull_inputs"):
            b = json.loads(json.dumps(b))
            b.pop("full_checkpoint_v2", None)
            if a["name"].startswith("pt_"):
                b["spec"]["train"]["checkpoint_every"] = a["spec"]["train"][
                    "checkpoint_every"
                ]
                b["relay_every_steps"] = a["relay_every_steps"]
            if a["name"].startswith("post_"):
                b["cmd"][b["cmd"].index("--sft-template") + 1] = a["cmd"][
                    a["cmd"].index("--sft-template") + 1
                ]
            assert a == b


@pytest.fixture
def validate_tree(tmp_path):
    h, c = tmp_path / "historical", tmp_path / "control"
    for seed in (1, 2, 3):
        rel = f"tools/colab_chat/carry12_sfull_r{seed}.json"
        p = h / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((ORIGINAL / rel).read_bytes())
    ref = (
        h
        / "experiments/tiny_chat_long_2026-10/receipts/sfull_prospective_20261031/data_reference.json"
    )
    ref.parent.mkdir(parents=True)
    ref.write_bytes((ORIGINAL / ref.relative_to(h)).read_bytes())
    v = c / "tools/colab_chat/sfull_verify_inputs_v2.py"
    v.parent.mkdir(parents=True)
    v.write_bytes((HERE / "sfull_verify_inputs_v2.py").read_bytes())
    rel = "experiments/tiny_chat_long_2026-10/specs/sft_T1024.json"
    target = h / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ORIGINAL / rel).read_bytes())
    (c / "tools/colab_chat/sfull_sft_template_v2.json").write_bytes(
        (HERE / "sfull_sft_template_v2.json").read_bytes()
    )
    jobs = tmp_path / "jobs.json"
    jobs.write_bytes((HERE / "carry12_sfull_v2_r1.json").read_bytes())
    with (
        patch.object(
            op.subprocess, "check_output", return_value=(op.NUMERICAL_COMMIT + "\n")
        ),
        patch.object(op.subprocess, "run"),
    ):
        yield h, c, jobs


def test_validate_accepts_exact(validate_tree):
    assert op.validate(*validate_tree)["seed"] == 1


@pytest.mark.parametrize(
    "mutation", ["steps", "seed", "mask", "lr", "post", "v1", "reference", "verifier"]
)
def test_refuses_drift(validate_tree, mutation):
    h, c, j = validate_tree
    if mutation == "verifier":
        (c / "tools/colab_chat/sfull_verify_inputs_v2.py").write_text("wrong")
    elif mutation == "reference":
        (
            h
            / "experiments/tiny_chat_long_2026-10/receipts/sfull_prospective_20261031/data_reference.json"
        ).write_text("{}")
    else:
        x = json.loads(j.read_text())
        pt = next(z for z in x["jobs"] if z["name"].startswith("pt_"))
        if mutation in ("steps", "seed", "lr"):
            pt["spec"]["train"][mutation] = 999
        elif mutation == "mask":
            pt["spec"]["arch"]["state_tokens"]["mask"] = "gist"
        elif mutation == "post":
            x["jobs"][-1]["cmd"][-1] = "123"
        else:
            next(z for z in x["jobs"] if z["name"] == "sfull_inputs")["cmd"][1] = (
                "sfull_verify_inputs.py"
            )
        j.write_text(json.dumps(x))
    with pytest.raises(ValueError):
        op.validate(h, c, j)


@pytest.mark.parametrize(
    "status,rc,expected",
    [
        ("FINISHED", 0, 0),
        ("FAILED", 0, 1),
        ("WAITING", 0, 1),
        ("FINISHED", 1, 1),
        ("MISSING", 0, 1),
    ],
)
def test_finish_checks_every_job(tmp_path, status, rc, expected):
    jobs = {"jobs": [{"name": "a"}, {"name": "b"}]}
    (tmp_path / "STATUS_a").write_text("FINISHED\n")
    if status != "MISSING":
        (tmp_path / "STATUS_b").write_text(status + "\n")
    assert op.finish(tmp_path, jobs, rc) == expected
    assert (tmp_path / "STATUS").read_text().strip() == (
        "FINISHED" if expected == 0 else "JOBS_FAILED"
    )


def test_pinned_boot_selection():
    import runner_multi as runner

    code = runner.boot_launch_code(
        "a" * 40,
        "tools/colab_chat/carry12_sfull_v2_r1.json",
        "tools/colab_chat/boot_sfull_v2_20261101.sh",
    )
    compile(code, "boot", "exec")
    assert "/" + ("a" * 40) + "/tools/colab_chat/boot_sfull_v2_20261101.sh" in code


def test_real_manage_error_sets_failure_marker(tmp_path):
    import subprocess

    p = subprocess.run(
        [
            "python3",
            str(HERE / "sfull_operational_v2.py"),
            "manage",
            "--historical",
            str(tmp_path / "absent"),
            "--chat",
            str(tmp_path),
            "--jobs",
            str(tmp_path / "absent.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert p.returncode != 0
    assert (tmp_path / "STATUS").read_text() == "BOOT_FAILED\n"
