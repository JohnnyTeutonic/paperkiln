from pathlib import Path
import subprocess
import pytest
import runner_multi as runner

HERE = Path(__file__).resolve().parent


@pytest.mark.parametrize(
    "commit,jobs", [("short", "jobs.json"), ("a" * 40, "bad;touch.json"), ("", "")]
)
def test_bad_boot_arguments_publish_failure_before_network(tmp_path, commit, jobs):
    source = (HERE / "boot_dpo_checkpoint_v1.sh").read_text()
    # Execute the exact validation prefix with only its absolute VM root mapped to tmp.
    prefix = source.split("if [ ! -d /content/pk/.git ]; then", 1)[0]
    script = tmp_path / "prefix.sh"
    script.write_text(prefix.replace("/content", str(tmp_path)))
    result = subprocess.run(
        ["bash", str(script), commit, jobs], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert (tmp_path / "chat/STATUS").read_text() == "BOOT_FAILED\n"


def test_named_boot_selected_without_changing_default():
    code = runner.boot_launch_code(
        "a" * 40,
        "/home/jonat/dpo/jobs_v1.json",
        "tools/colab_chat/boot_dpo_checkpoint_v1.sh",
    )
    compile(code, "boot", "exec")
    assert "boot_dpo_checkpoint_v1.sh" in code
    assert "boot_jobs.sh" in runner.boot_launch_code("a" * 40, "jobs.json")
