"""Independent binary64 B1/B3 diagnostic; not a historical arithmetic oracle."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            h.update(data)
    return h.hexdigest()


def weights(path):
    with Path(path).open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
    result = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            raise ValueError("reference requires stored FP32 tensors")
        view = np.memmap(
            path,
            dtype="<f4",
            mode="r",
            offset=8 + size + info["data_offsets"][0],
            shape=tuple(info["shape"]),
        )
        result[name] = np.asarray(view, dtype=np.float64)
    return result


def forward(ids, positions, w):
    # This specific registered B architecture only: d768/L8/H12/FF2240, no biases,
    # all-head interleaved RoPE, causal attention, RMS+SwiGLU, tied head.
    d, heads, width = 768, 12, 64
    x = w["embed_tokens.weight"][ids].copy()
    length = len(ids)
    epsilon = float(np.float32(1e-5))
    angles = np.arange(length, dtype=np.float64)[:, None] * 10000.0 ** (
        -np.arange(0, width, 2, dtype=np.float64)[None, :] / width
    )
    cosine, sine = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]

    def rms(value, gain):
        return (
            value
            / np.sqrt(np.mean(value * value, axis=1, keepdims=True) + epsilon)
            * gain
        )

    def rotate(value):
        value = value.reshape(length, heads, width)
        even, odd = value[:, :, 0::2].copy(), value[:, :, 1::2].copy()
        value[:, :, 0::2] = even * cosine - odd * sine
        value[:, :, 1::2] = even * sine + odd * cosine
        return value.transpose(1, 0, 2)

    future = np.triu(np.ones((length, length), dtype=bool), 1)
    for layer in range(8):
        prefix = f"layers.{layer}."
        norm = rms(x, w[prefix + "input_layernorm.weight"])
        q = rotate(norm @ w[prefix + "self_attn.q_proj.weight"])
        k = rotate(norm @ w[prefix + "self_attn.k_proj.weight"])
        v = (
            (norm @ w[prefix + "self_attn.v_proj.weight"])
            .reshape(length, heads, width)
            .transpose(1, 0, 2)
        )
        score = (q @ k.transpose(0, 2, 1)) * 0.125
        score[:, future] = -np.inf
        probability = np.exp(score - np.max(score, axis=2, keepdims=True))
        probability /= np.sum(probability, axis=2, keepdims=True)
        attended = (probability @ v).transpose(1, 0, 2).reshape(length, d)
        x = x + attended @ w[prefix + "self_attn.o_proj.weight"]
        norm = rms(x, w[prefix + "post_attention_layernorm.weight"])
        gate = norm @ w[prefix + "mlp.gate_proj.weight"]
        up = norm @ w[prefix + "mlp.up_proj.weight"]
        x = x + (gate / (1.0 + np.exp(-gate)) * up) @ w[prefix + "mlp.down_proj.weight"]
    hidden = rms(x, w["norm.weight"])
    return hidden[positions] @ w["embed_tokens.weight"].T


def run(art, output):
    art, output = Path(art), Path(output)
    source_hash = sha(Path(__file__))
    requests = json.loads((art / "requests.json").read_text())
    inventory = json.loads((art / "inventory.json").read_text())
    if output.exists():
        raise FileExistsError("completed diagnostic is immutable")
    checkpoint = output.with_suffix(".checkpoint.json")
    report = {
        "scope": "Independent binary64 mathematical diagnostic from stored FP32 weights, not historical/golden arithmetic",
        "source_sha256": source_hash,
        "request_sha256": sha(art / "requests.json"),
        "predeclared_checks": {
            "max_absolute_full_logits_error": 0.001,
            "ordinary_native_top2_gap": 0.001,
            "ordinary_argmax_must_match": True,
        },
        "transcendentals": "NumPy binary64 exp/sin/cos/power/sqrt and GEMM/reductions; no native float32 RoPE table or float32 accumulation emulation. Epsilon is exact promoted float32(1e-5).",
        "rows": [],
    }
    started = time.monotonic()
    for run_name in ("c12_B_r1", "c12_B_r3"):
        info = inventory["runs"][run_name]
        path = Path(info["weights"])
        if sha(path) != info["sources_sha256"][str(path)]:
            raise ValueError("checkpoint bytes changed")
        stat = path.stat()
        w = weights(path)
        selected = [x for x in requests if x["run"] == run_name]
        native = np.fromfile(
            art / (run_name + "_full.logits.f32"), dtype="<f4"
        ).reshape(-1, 16384)
        offset = 0
        for request in selected:
            if time.monotonic() - started > 1200:
                raise ValueError("20-minute reference budget exceeded")
            row = request["request"]
            p, c = row["prompt_ids"], row["continuation_ids"]
            ids = p + c[:-1]
            result = forward(ids, np.arange(len(p) - 1, len(p) - 1 + len(c)), w)
            if not np.isfinite(result).all():
                raise ValueError("nonfinite binary64 result")
            original = native[offset : offset + len(c)].astype(np.float64)
            differences = np.max(np.abs(result - original), axis=1)
            raw = art / (
                run_name
                + "_"
                + row["key"].split(":")[-1].replace("|", "_")
                + ".fp64.f64.gz"
            )
            with raw.open("xb") as stream:
                stream.write(gzip.compress(result.astype("<f8").tobytes(), mtime=0))
            positions = []
            for j, chosen in enumerate(c):
                values, old = result[j].copy(), original[j].copy()
                values[0], old[0] = -np.inf, -np.inf
                top = np.argsort(-values, kind="stable")[:2]
                oldtop = np.argsort(-old, kind="stable")[:2]
                gap = float(old[oldtop[0]] - old[oldtop[1]])
                positions.append(
                    {
                        "position": j,
                        "saved_chosen": chosen,
                        "native_argmax": int(oldtop[0]),
                        "fp64_argmax": int(top[0]),
                        "fp64_runner_up": int(top[1]),
                        "fp64_top2_gap": float(values[top[0]] - values[top[1]]),
                        "fp64_argmax_minus_saved": float(
                            values[top[0]] - values[chosen]
                        ),
                        "native_top2_gap": gap,
                        "ordinary": gap > 0.001,
                        "max_abs_error": float(differences[j]),
                    }
                )
            report["rows"].append(
                {
                    "key": row["key"],
                    "run": run_name,
                    "weight_sha256": info["sources_sha256"][str(path)],
                    "raw_file": raw.name,
                    "raw_sha256": sha(raw),
                    "positions": positions,
                    "compatibility_pass": bool(
                        np.max(differences) <= 0.001
                        and all(
                            x["native_argmax"] == x["fp64_argmax"]
                            for x in positions
                            if x["ordinary"]
                        )
                    ),
                }
            )
            if (
                sha(Path(__file__)) != source_hash
                or path.stat().st_mtime_ns != stat.st_mtime_ns
            ):
                raise ValueError("source/checkpoint changed")
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            temporary.replace(checkpoint)
            offset += len(c)
            print(
                row["key"],
                "max_error",
                float(np.max(differences)),
                "compatibility",
                report["rows"][-1]["compatibility_pass"],
                flush=True,
            )
        del w
    report["compatibility_pass"] = all(
        row["compatibility_pass"] for row in report["rows"]
    )
    report["seconds"] = time.monotonic() - started
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    checkpoint.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--art", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.art, args.output)
