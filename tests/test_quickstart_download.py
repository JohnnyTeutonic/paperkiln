"""The ready-made model path of quickstart_chat.sh: the pinned SHA-256 is checked before anything is
unpacked, unsafe archives are refused, an interrupted download resumes, and nothing half-unpacked is
left behind. Offline: the bundles are built here and fetched through file:// URLs.

    python3 -m pytest tests/test_quickstart_download.py -q
"""
import hashlib
import io
import json
import os
import pathlib
import shutil
import socket
import subprocess
import tarfile
import time
import urllib.request

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "quickstart_chat.sh"
PINNED_URL = "https://github.com/JohnnyTeutonic/paperkiln/releases/download/chat-v1/tinychat-v2-web.tar.gz"
PINNED_SHA256 = "16ae917498bc3a30e816b140e2fbced1d6b8a62426607a077e9fb1f1e9dd0fb2"

pytestmark = pytest.mark.skipif(not (shutil.which("bash") and shutil.which("curl") and shutil.which("tar")),
                                reason="needs bash, curl and tar")


def _add(tf, name, data=b"", kind=tarfile.REGTYPE, linkname=""):
    info = tarfile.TarInfo(name)
    info.type, info.linkname, info.mtime = kind, linkname, 0
    info.mode = 0o755 if kind == tarfile.DIRTYPE else 0o644
    if kind == tarfile.REGTYPE:
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    else:
        tf.addfile(info)


def bundle(path, extra=()):
    """A minimal browser bundle (index.html, model/manifest.json) plus any extra (name, data, kind, link)."""
    with tarfile.open(path, "w:gz") as tf:
        _add(tf, "tinychat-web/", kind=tarfile.DIRTYPE)
        _add(tf, "tinychat-web/index.html", b"<!doctype html><title>chat</title>")
        _add(tf, "tinychat-web/model/", kind=tarfile.DIRTYPE)
        _add(tf, "tinychat-web/model/manifest.json", json.dumps({"name": "test"}).encode())
        for e in extra:
            _add(tf, *e)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def clone(tmp_path):
    """The script alone in a fresh folder: the default mode needs nothing else from the repository."""
    d = tmp_path / "clone"
    d.mkdir()
    shutil.copy(SCRIPT, d / "quickstart_chat.sh")
    return d


def run(clone, url, sha=None, port=None, timeout=60):
    env = dict(os.environ, PAPERKILN_DEFAULT_MODEL_URL=url)
    env.pop("PAPERKILN_DEFAULT_MODEL_SHA256", None)
    if sha is not None:
        env["PAPERKILN_DEFAULT_MODEL_SHA256"] = sha
    args = ["bash", "quickstart_chat.sh", "--yes", "--mode", "default", "--no-share", "--no-browser",
            "--port", str(port or free_port())]
    return subprocess.run(args, cwd=clone, env=env, capture_output=True, text=True, timeout=timeout,
                          stdin=subprocess.DEVNULL)


def serve_until_up(clone, url, sha):
    """Start the script, wait for the bundle's manifest over HTTP, stop it; returns the page body."""
    port = free_port()
    env = dict(os.environ, PAPERKILN_DEFAULT_MODEL_URL=url, PAPERKILN_DEFAULT_MODEL_SHA256=sha)
    p = subprocess.Popen(["bash", "quickstart_chat.sh", "--yes", "--mode", "default", "--no-share",
                          "--no-browser", "--port", str(port)], cwd=clone, env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/model/manifest.json", timeout=1) as r:
                    assert json.load(r) == {"name": "test"}
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=1) as r:
                    return r.read().decode()
            except OSError:
                if p.poll() is not None:
                    pytest.fail(f"the script stopped: {p.stdout.read()}")
                time.sleep(0.2)
        pytest.fail("the chat page never came up")
    finally:
        try:
            os.killpg(p.pid, 15)
        except ProcessLookupError:
            pass
        p.wait(timeout=10)


def test_pinned_default_is_the_release_digest():
    text = SCRIPT.read_text()
    assert f'DEFAULT_MODEL_URL="{PINNED_URL}"' in text
    assert f'DEFAULT_MODEL_SHA256="{PINNED_SHA256}"' in text


def test_checked_bundle_installs_and_serves(clone, tmp_path):
    sha = bundle(tmp_path / "b.tar.gz")
    page = serve_until_up(clone, (tmp_path / "b.tar.gz").as_uri(), sha)
    assert "<title>chat</title>" in page
    out = clone / "runs" / "tinychat-default-web"
    assert (out / ".bundle_sha256").read_text().strip() == sha
    assert not (clone / "runs" / "tinychat-default-web.tar.gz.part").exists()
    assert not list((clone / "runs").glob("*.tmp.*"))


def test_wrong_digest_unpacks_nothing(clone, tmp_path):
    bundle(tmp_path / "b.tar.gz")
    r = run(clone, (tmp_path / "b.tar.gz").as_uri(), "0" * 64)
    assert r.returncode != 0 and "not the pinned" in r.stderr
    runs = clone / "runs"
    assert not (runs / "tinychat-default-web").exists()
    assert not (runs / "tinychat-default-web.tar.gz.part").exists()


def test_overridden_url_needs_a_digest(clone, tmp_path):
    bundle(tmp_path / "b.tar.gz")
    r = run(clone, (tmp_path / "b.tar.gz").as_uri(), sha=None)
    assert r.returncode != 0 and "PAPERKILN_DEFAULT_MODEL_SHA256" in r.stderr
    assert not (clone / "runs" / "tinychat-default-web").exists()


@pytest.mark.parametrize("extra,why", [
    (("tinychat-web/../escaped.txt", b"x"), "outside"),
    (("other/file.txt", b"x"), "outside"),
    (("tinychat-web/link", b"", tarfile.SYMTYPE, "/etc/passwd"), "links"),
    (("tinychat-web/hard", b"", tarfile.LNKTYPE, "tinychat-web/index.html"), "links"),
])
def test_unsafe_members_refused_even_with_the_right_digest(clone, tmp_path, extra, why):
    sha = bundle(tmp_path / "b.tar.gz", [extra])
    r = run(clone, (tmp_path / "b.tar.gz").as_uri(), sha)
    assert r.returncode != 0 and why in r.stderr, r.stderr
    assert not (clone / "runs" / "tinychat-default-web").exists()
    assert not (tmp_path / "escaped.txt").exists() and not (clone / "runs" / "escaped.txt").exists()


def test_interrupted_download_resumes(clone, tmp_path):
    sha = bundle(tmp_path / "b.tar.gz", [("tinychat-web/model/weights.bin", os.urandom(200_000))])
    data = (tmp_path / "b.tar.gz").read_bytes()
    (clone / "runs").mkdir()
    (clone / "runs" / "tinychat-default-web.tar.gz.part").write_bytes(data[: len(data) // 2])
    serve_until_up(clone, (tmp_path / "b.tar.gz").as_uri(), sha)
    assert (clone / "runs" / "tinychat-default-web" / "model" / "weights.bin").exists()


def test_corrupt_partial_is_restarted_once(clone, tmp_path):
    sha = bundle(tmp_path / "b.tar.gz")
    (clone / "runs").mkdir()
    (clone / "runs" / "tinychat-default-web.tar.gz.part").write_bytes(b"garbage from an earlier attempt")
    serve_until_up(clone, (tmp_path / "b.tar.gz").as_uri(), sha)


def test_unchecked_earlier_install_is_replaced(clone, tmp_path):
    # A folder unpacked by the earlier script (no recorded digest) is fetched again and checked.
    old = clone / "runs" / "tinychat-default-web"
    (old / "model").mkdir(parents=True)
    (old / "index.html").write_text("stale")
    sha = bundle(tmp_path / "b.tar.gz")
    page = serve_until_up(clone, (tmp_path / "b.tar.gz").as_uri(), sha)
    assert "stale" not in page
