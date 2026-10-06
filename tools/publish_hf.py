#!/usr/bin/env python3
"""Publish a trained paperkiln chat model to the Hugging Face Hub as a community model.

    export HF_TOKEN=hf_...            # a write token from huggingface.co/settings/tokens
    python3 tools/publish_hf.py runs/tinychat-better --repo your-name/tinychat-better --dry-run
    python3 tools/publish_hf.py runs/tinychat-better --repo your-name/tinychat-better

Uploads what someone needs to chat with the model or load it elsewhere: the exported
weights (safetensors and GGUF), spec.json, result.json, card.json, and a README.md
built from the model card with the Hub's metadata header. The token is read from
HF_TOKEN (or from `--env-file`, which is loaded without printing anything) and is
never written to disk or to the console.

Needs `pip install huggingface_hub`. Optional: this is not how the quickstart gets its
ready-made model (that is a GitHub Release, so downloading needs no account).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_env_file(path: str) -> None:
    """KEY=VALUE lines into os.environ, without echoing them."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip().removeprefix("export ").strip(), v.strip().strip("\"'"))


def hub_readme(run: Path, name: str, repo: str) -> str:
    card = (run / "card.md").read_text(encoding="utf-8") if (run / "card.md").exists() else \
        f"# {name}\n\nA small language model trained with paperkiln.\n"
    front = "\n".join([
        "---",
        "license: mit",
        "library_name: paperkiln",
        "pipeline_tag: text-generation",
        "tags:",
        "- paperkiln",
        "- tiny",
        "- word-level",
        "---",
        "",
    ])
    howto = (
        "\n## Chat with it\n\n"
        "```bash\n"
        "git clone https://github.com/JohnnyTeutonic/paperkiln && cd paperkiln\n"
        f"huggingface-cli download {repo} --local-dir runs/{name}\n"
        "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --target mtstudio -j\n"
        f"./build/mtstudio chat runs/{name}    # then open http://127.0.0.1:8080/\n"
        "```\n\n"
        "The GGUF carries a word-level vocabulary (`tokenizer.ggml.model = word`), so it loads in "
        "paperkiln and ember.cpp, not in tools that expect a subword tokenizer.\n"
    )
    return front + card.rstrip() + "\n" + howto


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--repo", required=True, help="the Hub model repo, e.g. your-name/tinychat-better")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--env-file", help="a file with HF_TOKEN=..., loaded without printing")
    ap.add_argument("--dry-run", action="store_true", help="show what would be uploaded; no network")
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    spec = json.loads((run / "spec.json").read_text(encoding="utf-8"))
    name = spec["name"]
    files = [f for f in (f"{name}.safetensors", f"{name}.gguf", "spec.json", "result.json", "card.json")
             if (run / f).exists()]
    missing = [f for f in (f"{name}.safetensors", f"{name}.gguf", "spec.json") if f not in files]
    if missing:
        print(f"missing in {run}: {', '.join(missing)} (train and export the run first)")
        return 1
    readme = hub_readme(run, name, args.repo)

    print(f"repo: {args.repo} ({'private' if args.private else 'public'})")
    for f in files:
        print(f"  {f}  {(run / f).stat().st_size / 1e6:.2f} MB")
    print(f"  README.md  (model card + Hub metadata, {len(readme)} characters)")
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
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    for f in files:
        api.upload_file(path_or_fileobj=str(run / f), path_in_repo=f, repo_id=args.repo,
                        commit_message=f"paperkiln: {f}")
    api.upload_file(path_or_fileobj=readme.encode("utf-8"), path_in_repo="README.md",
                    repo_id=args.repo, commit_message="paperkiln: model card")
    print(f"published: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
