#!/usr/bin/env python3
"""Publish a trained paperkiln model to the Hugging Face Hub as a community model.

    export HF_TOKEN=hf_...            # a write token from huggingface.co/settings/tokens
    python3 tools/publish_hf.py runs/tinychat-better --repo your-name/tinychat-better --dry-run
    python3 tools/publish_hf.py runs/tinychat-better --repo your-name/tinychat-better

A run directory gives the exported weights (safetensors and GGUF), spec.json, result.json,
card.json, and a README.md built from card.md with the Hub's metadata header. For a release
with a hand-written card and chosen files (quantisations, hashes from a log):

    python3 tools/publish_hf.py --repo you/pk_base_v1 --card card.md \\
        --file pk_base_v1.gguf --file pk_base_v1_q4_k_m.gguf --file spec.json \\
        --sha256 expected.sha256 --stage stage/pk_base_v1 --dry-run
    python3 tools/publish_hf.py ... --stage stage/pk_base_v1 --verify dl/pk_base_v1

- `--sha256 FILE` holds "HASH  NAME" lines (sha256sum's format); every data file must be listed
  and match before anything is staged or uploaded.
- `--stage DIR` writes the repo as a folder (data files hard-linked where the filesystem allows,
  else copied; README.md; SHA256SUMS of the data files) and uploads it with the Hub's resumable
  large-folder upload: rerunning the same command after an interruption continues it.
- `--verify DIR` downloads every data file back into DIR (which must be empty or absent) and
  checks its sha256; `--verify-only` does just that.

The token is read from HF_TOKEN (or from `--env-file`, from which only HF_TOKEN and
HUGGING_FACE_HUB_TOKEN are loaded) and is never written to disk or to the console.

Needs `pip install huggingface_hub`. Optional: this is not how the quickstart gets its
ready-made model (that is a GitHub Release, so downloading needs no account).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def load_env_file(path: str, keys=TOKEN_KEYS) -> None:
    """The named KEY=VALUE lines into os.environ, without echoing them; other lines are ignored."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        if k in keys:
            os.environ.setdefault(k, v.strip().strip("\"'"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sha256_list(path) -> dict:
    """sha256sum lines ("HASH  NAME" or "HASH *NAME") -> {basename: hash}."""
    out = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"{path}:{n}: not a sha256sum line")
        name = Path(parts[1].lstrip("*").strip()).name
        if name in out and out[name] != parts[0].lower():
            raise ValueError(f"{path}:{n}: {name} listed twice with different hashes")
        out[name] = parts[0].lower()
    return out


def gguf_string_kv(path, key):
    """A string value from a GGUF's metadata, or None (no key, not a GGUF, other type)."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    try:
        with open(path, "rb") as f:
            magic, _ver, _nt, nkv = struct.unpack("<4sIQQ", f.read(24))
            if magic != b"GGUF":
                return None

            def rstr():
                (n,) = struct.unpack("<Q", f.read(8))
                return f.read(n)

            def skip(t):
                if t == 8:
                    rstr()
                elif t == 9:
                    et, n = struct.unpack("<IQ", f.read(12))
                    if et in sizes:
                        f.seek(sizes[et] * n, 1)
                    else:
                        for _ in range(n):
                            skip(et)
                else:
                    f.seek(sizes[t], 1)

            for _ in range(nkv):
                k = rstr().decode("utf-8", "replace")
                (t,) = struct.unpack("<I", f.read(4))
                if k == key and t == 8:
                    return rstr().decode("utf-8", "replace")
                skip(t)
    except (OSError, struct.error, KeyError):
        return None
    return None


def hub_readme(run: Path, name: str, repo: str, license_id: str = "mit", gguf=None) -> str:
    card = (run / "card.md").read_text(encoding="utf-8") if (run / "card.md").exists() else \
        f"# {name}\n\nA small language model trained with paperkiln.\n"
    tok = gguf_string_kv(gguf, "tokenizer.ggml.model") if gguf else None
    word = tok in (None, "word")
    front = "\n".join([
        "---",
        f"license: {license_id}",
        "library_name: paperkiln",
        "pipeline_tag: text-generation",
        "tags:",
        "- paperkiln",
        "- tiny",
        "- word-level" if word else "- gguf",
        "---",
        "",
    ])
    if word:
        tok_note = ("The GGUF carries a word-level vocabulary (`tokenizer.ggml.model = word`), so it loads in "
                    "paperkiln and ember.cpp, not in tools that expect a subword tokenizer.\n")
    else:
        tok_note = (f"The GGUF carries its own subword vocabulary (`tokenizer.ggml.model = {tok}`) "
                    "and loads in paperkiln and ember.cpp.\n")
    howto = (
        "\n## Chat with it\n\n"
        "```bash\n"
        "git clone https://github.com/JohnnyTeutonic/paperkiln && cd paperkiln\n"
        f"huggingface-cli download {repo} --local-dir runs/{name}\n"
        "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target mtstudio -j\n"
        f"./build/mtstudio chat runs/{name}    # then open http://127.0.0.1:8080/\n"
        "```\n\n" + tok_note
    )
    return front + card.rstrip() + "\n" + howto


def card_readme(card_path: str, license_id: str) -> str:
    """A hand-written card: used as written when it has a metadata header, else given a minimal one."""
    text = Path(card_path).read_text(encoding="utf-8")
    if text.startswith("---\n"):
        return text
    return "\n".join(["---", f"license: {license_id}", "pipeline_tag: text-generation",
                      "tags:", "- paperkiln", "---", ""]) + text


def stage_folder(stage: Path, files: dict, readme: str, sums: str) -> None:
    """files {published name: source path}. Hard links where possible, copies otherwise. A stage left by an
    earlier run is reused only if every data file in it is the same file or has the same content."""
    stage.mkdir(parents=True, exist_ok=True)
    allowed = set(files) | {"README.md", "SHA256SUMS", ".cache"}
    extra = [p.name for p in stage.iterdir() if p.name not in allowed]
    if extra:
        raise SystemExit(f"stage {stage} holds other files ({', '.join(sorted(extra))}); use an empty folder")
    for name, src in files.items():
        dst = stage / name
        if dst.exists():
            if os.path.samefile(dst, src) or sha256_file(dst) == sha256_file(src):
                continue
            raise SystemExit(f"stage {stage}: {name} differs from {src}; use an empty folder")
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    (stage / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    (stage / "SHA256SUMS").write_text(sums, encoding="utf-8", newline="\n")


def verify_download(api, repo: str, expected: dict, out_dir: Path) -> int:
    """Download each file in expected {name: sha256} into out_dir and compare; returns the mismatch count."""
    from huggingface_hub import hf_hub_download
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"verify: {out_dir} is not empty; give an empty or new folder")
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    bad = 0
    for name, want in expected.items():
        p = hf_hub_download(repo, name, repo_type="model", local_dir=str(out_dir), token=api.token)
        got = sha256_file(p)
        ok = got == want
        bad += not ok
        print(f"  {'OK ' if ok else 'BAD'} {name}  {got}")
    print(f"verify: {len(expected) - bad}/{len(expected)} files match")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", nargs="?", help="a paperkiln run directory (spec.json, NAME.gguf, ...)")
    ap.add_argument("--repo", required=True, help="the Hub model repo, e.g. your-name/tinychat-better")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--env-file", help="a file with HF_TOKEN=...; only the token lines are loaded, never printed")
    ap.add_argument("--dry-run", action="store_true", help="show (and stage) what would be uploaded; no network")
    ap.add_argument("--file", action="append", default=[], help="a file to publish under its own name (repeatable)")
    ap.add_argument("--card", help="the model card (Markdown); used as written if it starts with a --- header")
    ap.add_argument("--license", default="mit", help="Hub licence id for a generated header (default mit)")
    ap.add_argument("--sha256", help="expected hashes (sha256sum format); every data file must match")
    ap.add_argument("--stage", help="write the repo as a folder here and upload it resumably")
    ap.add_argument("--verify", help="after the upload, download every data file into this empty folder and "
                                     "check its sha256")
    ap.add_argument("--verify-only", action="store_true", help="only the --verify download and check")
    args = ap.parse_args(argv)

    files = {}  # published name -> local path
    readme = None
    if args.run_dir:
        run = Path(args.run_dir)
        spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
        name = spec["name"]
        for f in (f"{name}.safetensors", f"{name}.gguf", "spec.json", "result.json", "card.json"):
            if (run / f).exists():
                files[f] = run / f
        missing = [f for f in (f"{name}.safetensors", f"{name}.gguf", "spec.json") if f not in files]
        if missing:
            print(f"missing in {run}: {', '.join(missing)} (train and export the run first)")
            return 1
        readme = hub_readme(run, name, args.repo, args.license, files.get(f"{name}.gguf"))
    for f in args.file:
        p = Path(f)
        if not p.is_file():
            print(f"no such file: {f}")
            return 1
        if p.name in ("README.md", "SHA256SUMS"):
            print(f"{p.name} is written by this tool; pass the card with --card")
            return 1
        files[p.name] = p
    if args.card:
        readme = card_readme(args.card, args.license)
    if not files:
        print("nothing to publish: give a run directory or --file")
        return 1
    if readme is None:
        print("no model card: give a run directory with card.md, or --card")
        return 1

    hashes = {n: sha256_file(p) for n, p in files.items()}
    if args.sha256:
        want = read_sha256_list(args.sha256)
        bad = [n for n in files if want.get(n) != hashes[n]]
        if bad:
            for n in bad:
                print(f"sha256 mismatch: {n} is {hashes[n]}, expected {want.get(n, '(not listed)')}")
            return 1
    sums = "".join(f"{hashes[n]}  {n}\n" for n in sorted(files))

    print(f"repo: {args.repo} ({'private' if args.private else 'public'})")
    for n in sorted(files):
        print(f"  {n}  {files[n].stat().st_size / 1e6:.2f} MB  sha256 {hashes[n]}"
              + ("  (matches --sha256)" if args.sha256 else ""))
    print(f"  README.md  (model card, {len(readme)} characters)")
    print("  SHA256SUMS  (the data files' hashes)")
    if args.stage:
        stage_folder(Path(args.stage), files, readme, sums)
        print(f"staged in {args.stage}")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return 0

    if args.env_file:
        load_env_file(args.env_file)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("no token: set HF_TOKEN to a write token (huggingface.co/settings/tokens) or pass --env-file")
        return 1
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("needs huggingface_hub: python3 -m pip install --user huggingface_hub")
        return 1
    api = HfApi(token=token)

    if not args.verify_only:
        api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        if args.stage:
            api.upload_large_folder(repo_id=args.repo, folder_path=args.stage, repo_type="model",
                                    private=args.private, num_workers=1)
        else:
            for n in sorted(files):
                api.upload_file(path_or_fileobj=str(files[n]), path_in_repo=n, repo_id=args.repo,
                                commit_message=f"paperkiln: {n}")
            api.upload_file(path_or_fileobj=sums.encode("utf-8"), path_in_repo="SHA256SUMS",
                            repo_id=args.repo, commit_message="paperkiln: SHA256SUMS")
            api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md",
                            repo_id=args.repo, commit_message="paperkiln: model card")
        print(f"published: https://huggingface.co/{args.repo}")
    if args.verify:
        return 1 if verify_download(api, args.repo, hashes, Path(args.verify)) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
