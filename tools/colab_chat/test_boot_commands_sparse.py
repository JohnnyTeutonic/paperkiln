import json
import os
from pathlib import Path
import subprocess
import time

import pytest


def git(*args, cwd):
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def boot(tmp_path):
    source = tmp_path / 'source'
    runtime = source / 'tools/colab_chat'
    runtime.mkdir(parents=True)
    (runtime / 'run.json').write_text(json.dumps({'jobs': [{'name': 'one', 'cmd': ['unused']}]}))
    (runtime / 'provenance.py').write_text('')
    (runtime / 'jobs_vm.py').write_text(
        "import os,pathlib\np=pathlib.Path(os.environ['JOBS_CHAT_DIR'])/'started'\np.write_text('yes')\n")
    (source / 'unneeded').mkdir()
    (source / 'unneeded/payload.bin').write_bytes(os.urandom(2 << 20))
    git('init', '-q', cwd=source)
    git('config', 'user.name', 'Test', cwd=source)
    git('config', 'user.email', 'test@example.invalid', cwd=source)
    git('add', '.', cwd=source)
    git('commit', '-qm', 'frozen fixture', cwd=source)
    bare = tmp_path / 'origin.git'
    git('clone', '--bare', str(source), str(bare), cwd=tmp_path)
    git('config', 'uploadpack.allowFilter', 'true', cwd=bare)
    fakebin = tmp_path / 'bin'
    fakebin.mkdir()
    pgrep = fakebin / 'pgrep'
    pgrep.write_text('#!/bin/sh\nexit 1\n')
    pgrep.chmod(0o755)
    hook = tmp_path / 'hook.sh'
    hook.write_text(Path(__file__).with_name('boot_commands_sparse.sh').read_text())
    root = tmp_path / 'content'
    env = dict(os.environ, COMMAND_BOOT_ROOT=str(root), COMMAND_BOOT_REPO=bare.as_uri(),
               PATH=str(fakebin) + os.pathsep + os.environ['PATH'])
    return source, git('rev-parse', 'HEAD', cwd=source), hook, root, env


def test_exact_sparse_snapshot_executes_without_unneeded_blob(boot):
    _, commit, hook, root, env = boot
    subprocess.run(['bash', str(hook), commit, 'tools/colab_chat/run.json'], env=env, check=True)
    for _ in range(100):
        if (root / 'chat/started').exists():
            break
        time.sleep(0.01)
    assert (root / 'chat/started').read_text() == 'yes'
    assert git('rev-parse', 'HEAD', cwd=root / 'pk') == commit
    assert not (root / 'pk/unneeded/payload.bin').exists()
    assert sum(p.stat().st_size for p in (root / 'pk/.git/objects/pack').glob('*.pack')) < 100_000


def test_training_job_refused_before_manager_starts(boot):
    source, _, hook, root, env = boot
    (source / 'tools/colab_chat/run.json').write_text(json.dumps({'jobs': [{'name': 'train', 'spec': 'x'}]}))
    git('add', '.', cwd=source)
    git('commit', '-qm', 'training refusal fixture', cwd=source)
    commit = git('rev-parse', 'HEAD', cwd=source)
    git('push', env['COMMAND_BOOT_REPO'], 'HEAD:refs/heads/test', cwd=source)
    result = subprocess.run(['bash', str(hook), commit, 'tools/colab_chat/run.json'], env=env)
    assert result.returncode != 0
    assert (root / 'chat/STATUS').read_text().strip() == 'BOOT_FAILED'
    assert not (root / 'chat/started').exists()


def test_short_or_unscoped_inputs_refused_without_checkout(boot):
    _, commit, hook, root, env = boot
    for revision, jobs in ((commit[:8], 'tools/colab_chat/run.json'), (commit, 'elsewhere/run.json')):
        result = subprocess.run(['bash', str(hook), revision, jobs], env=env)
        assert result.returncode != 0
    assert not root.exists()
