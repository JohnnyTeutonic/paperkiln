"""Explicit V2 operational amendment to the preserved, absolute original finalizer.

Scientific calculation/parity/floor code is delegated unchanged. This module adds
exact full runtime-spec validation and binds the new checkpoint-I/O source.
"""

from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = Path("/home/jonat/pk_sfull_packet_20261031")
EXPERIMENT = ORIGINAL / "experiments/tiny_chat_long_2026-10"
PROOF = REPO / "docs/receipts/sfull_launch_v2_20261101"
sys.path.insert(0, str(EXPERIMENT))
spec = importlib.util.spec_from_file_location(
    "preserved_sfull_v2_delegate", EXPERIMENT / "finalize_sfull.py"
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
original_sources = old.sources
original_finalizer = old.Finalizer
original_launch = old.LAUNCH
old.LAUNCH = PROOF / "launch_pin_v2.json"


def expected_specs(seed):
    jobs = json.loads((HERE / f"carry12_sfull_v2_r{seed}.json").read_text())["jobs"]
    pt = next(j for j in jobs if j["name"].startswith("pt_"))
    pt_spec = json.loads(json.dumps(pt["spec"]))
    pt_spec.update(name=pt["model_name"], out_dir=f"/content/chat/{pt['name']}")
    shape = json.loads((EXPERIMENT / "specs/cell_Sfull.json").read_text())
    sft = json.loads((HERE / "sfull_sft_template_v2.json").read_text())
    sft_spec = {
        "name": f"tc_sft_Sfull_r{seed}",
        "arch": shape["arch"],
        "data": sft["data"],
        "train": {
            **sft["train"],
            "batch": 16,
            "accum": 1,
            "seed": seed,
            "init_from": f"/content/chat/pt_c12_Sfull_r{seed}/tc_c12_Sfull_r{seed}.safetensors",
        },
        "export": sft["export"],
        "out_dir": f"/content/chat/post_c12_Sfull_r{seed}/sft",
    }
    return pt_spec, sft_spec


def validate_runtime_specs(art):
    missing = []
    for seed in (1, 2, 3):
        base = Path(art) / f"c12_Sfull_r{seed}"
        paths = [
            base / f"pt_c12_Sfull_r{seed}/pt_c12_Sfull_r{seed}_spec.json",
            base / f"post_c12_Sfull_r{seed}/sft_spec.json",
        ]
        for path, expected in zip(paths, expected_specs(seed)):
            if not path.is_file():
                missing.append(str(path))
                continue
            if json.loads(path.read_text()) != expected:
                raise ValueError(
                    "V2 runtime spec differs beyond exact checkpoint amendment: "
                    + str(path)
                )
    return missing


def sources():
    files = original_sources() + [original_launch, Path(__file__)]
    files += [
        HERE / n
        for n in [
            "sfull_operational_v2.py",
            "sfull_checkpoint_v2.py",
            "sfull_checkpoint_transport_v2.py",
            "runner_multi.py",
            "boot_sfull_v2_20261101.sh",
            "sfull_verify_inputs_v2.py",
            "sfull_sft_template_v2.json",
            "carry12_sfull_v2_r1.json",
            "carry12_sfull_v2_r2.json",
            "carry12_sfull_v2_r3.json",
        ]
    ]
    numeric = Path("/home/jonat/pk_sfull_numeric_io_v2_20261101")
    files += [
        numeric / "tools/mtstudio.cpp",
        numeric / "include/microtorch/sfull_checkpoint_bridge_v2.hpp",
    ]
    return list(dict.fromkeys(files))


class Finalizer(original_finalizer):
    def inputs(self):
        missing = validate_runtime_specs(self.art)
        result = super().inputs()
        result["pending"] = list(dict.fromkeys([*result["pending"], *missing]))
        result["ready"] = not result["pending"]
        return result


def main():
    # Require this new explicit freeze; never silently consume the old source freeze.
    if "--source-freeze" not in sys.argv:
        sys.argv += ["--source-freeze", str(PROOF / "source_freeze_v2.json")]
    old.sources = sources
    old.Finalizer = Finalizer
    return old.main()


if __name__ == "__main__":
    raise SystemExit(main())
