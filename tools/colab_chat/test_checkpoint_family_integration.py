from pathlib import Path
import pytest
import runner_multi as runner

SF = "tools/colab_chat/boot_sfull_v2_20261101.sh"
DPO = "tools/colab_chat/boot_dpo_checkpoint_v1.sh"


@pytest.mark.parametrize("boot", [SF, DPO, "tools/colab_chat/boot_jobs.sh"])
def test_mixed_campaigns_refused(boot):
    with pytest.raises(ValueError, match="mixed"):
        runner.validate_checkpoint_families(
            [{"full_checkpoint_v2": {}}, {"dpo_full_checkpoint_v1": {}}], boot
        )


@pytest.mark.parametrize(
    "key,boot", [("full_checkpoint_v2", SF), ("dpo_full_checkpoint_v1", DPO)]
)
def test_separate_named_boot_admits_only_own_family(key, boot):
    runner.validate_checkpoint_families([{key: {}}], boot)
    with pytest.raises(ValueError, match="named boot"):
        runner.validate_checkpoint_families(
            [{key: {}}], "tools/colab_chat/boot_jobs.sh"
        )


def test_default_no_optin_unchanged():
    runner.validate_checkpoint_families(
        [{"name": "ordinary", "spec": {}}], "tools/colab_chat/boot_jobs.sh"
    )
    assert "boot_jobs.sh" in runner.boot_launch_code("a" * 40, "jobs.json")


def test_both_boot_choices_compile():
    for boot in [SF, DPO]:
        compile(runner.boot_launch_code("a" * 40, "jobs.json", boot), "boot", "exec")


def test_legacy_checkpoint_functions_unchanged_from_public_base():
    import ast
    import subprocess

    path = Path(runner.__file__)
    repo = path.parents[2]
    before = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "show",
            "6af5ac810add1c212817097edacdbf50bfb50cf4:tools/colab_chat/runner_multi.py",
        ],
        text=True,
    )

    def functions(source):
        tree = ast.parse(source)
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
        }

    old, new = functions(before), functions(path.read_text())
    for name in [
        "upload_file",
        "download_file",
        "relay",
        "fetch_final",
        "adopt_home",
        "volume_has",
        "upload_checkpoint",
    ]:
        assert old[name] == new[name], name
