"""Actual CPU depth integration: source model -> F32 GGUF/IR -> NumPy and Ember.

Requires MTSTUDIO_BIN and EMBER_IR_LOGITS_BIN. Set PARITY_DEPTH_PROOF_DIR to
retain the exact tiny runs and row evidence; each case directory is fresh.
No provider calls, GPU execution or model-quality claim.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec" / "tools"))
from test_mtstudio_chat import write_vocab_gguf  # noqa: E402
import ir_check  # noqa: E402
import ir_numpy  # noqa: E402

BIN = os.environ.get("MTSTUDIO_BIN")
EMBER = os.environ.get("EMBER_IR_LOGITS_BIN")
BASE = os.environ.get("MTSTUDIO_BASE_BIN")
BASE_SHA256 = "3cba15295005af59affc1418c11c013b1ecf0f4accf8c4441df4cc6b003dd52a"
pytestmark = pytest.mark.skipif(not BIN or not EMBER, reason="actual mtstudio and CPU Ember binaries required")
TOKENS = ["<unk>", "the", "cat", "sat", "on", "mat", "."]
SEQUENCES = [[1], [1, 2, 3], [6, 5, 4, 3, 2, 1]]
SOURCE_FILES = ["tools/parity_model.hpp", "tools/mtstudio.cpp", "src/ops.cpp"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def invoke(command, directory, name, **kwargs):
    result = subprocess.run(
        [str(v) for v in command], capture_output=True, text=True, errors="replace", timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, **kwargs
    )
    (directory / (name + ".stdout")).write_text(result.stdout, encoding="utf-8")
    (directory / (name + ".stderr")).write_text(result.stderr, encoding="utf-8")
    return result


def directory_for(tmp_path, name):
    proof = os.environ.get("PARITY_DEPTH_PROOF_DIR")
    directory = Path(proof) / name if proof else tmp_path / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def make_spec(directory, attention, layers):
    write_vocab_gguf(directory / "vocab.gguf", TOKENS)
    (directory / "corpus.txt").write_text("the cat sat on the mat . " * 100, encoding="utf-8")
    spec = {
        "name": "depth_gate", "arch": {"preset": "gpt2-nano",
        "custom": {"d": 16, "heads": 2, "layers": layers, "attention": attention}},
        "data": {"vocab": str(directory / "vocab.gguf"), "corpus": str(directory / "corpus.txt"),
                 "T": 12, "max_tokens": 0},
        "train": {"steps": 3, "batch": 1, "seed": 9, "lr": 0.001,
                  "eval_every": 3, "checkpoint_every": 3, "gradmap_every": 0},
        "export": {"formats": ["safetensors", "gguf"]}, "out_dir": str(directory / "trained"),
    }
    path = directory / "spec.json"
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return path, spec


def read_weights(path):
    meta, tensors, offset, raw, _ = ir_check.read_gguf(path)
    weights = {}
    for name, dims, kind, location in tensors:
        assert kind == 0, name
        weights[name] = np.frombuffer(raw, dtype="<f4", count=int(np.prod(dims)),
                                     offset=offset + location).reshape(tuple(reversed(dims)))
    return meta, weights


def compare_rows(actual, reference):
    actual, reference = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    assert actual.shape == reference.shape
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    error = float(np.max(np.abs(actual - reference))) / max(1.0, float(np.max(np.abs(reference))))
    assert error <= 1e-5, error
    assert np.array_equal(actual.argmax(axis=1), reference.argmax(axis=1))
    return error


def native_scores(spec, directory, requests, label, binary=None):
    req, out, rows = (directory / (label + suffix) for suffix in (".in.jsonl", ".out.jsonl", ".rows.f32"))
    req.write_text("".join(json.dumps(row) + "\n" for row in requests), encoding="utf-8")
    result = invoke([binary or BIN, "score", spec, "--in", req, "--out", out, "--logits", rows], directory, label)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == len(requests)
    assert all(not row["over_ctx"] for row in records)
    return records, np.fromfile(rows, dtype="<f4").reshape(-1, len(TOKENS))


@pytest.mark.parametrize("attention", ["kimi", "srd", "recurrence"])
@pytest.mark.parametrize("layers", [1, 3])
def test_trained_depth_metadata_full_rows_and_greedy(tmp_path, attention, layers):
    directory = directory_for(tmp_path, f"{attention}_{layers}")
    identity = {"mtstudio": sha(BIN), "ember": sha(EMBER),
                "sources": {name: sha(ROOT / name) for name in SOURCE_FILES}}
    spec_path, spec = make_spec(directory, attention, layers)
    trained = invoke([BIN, "run", spec_path], directory, "train")
    assert trained.returncode == 0, trained.stdout + trained.stderr
    out = Path(spec["out_dir"])
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    model = next(row for row in events if row["event"] == "model")
    done = next(row for row in events if row["event"] == "done")
    assert done["final_step"] == 3
    assert model["layers"] == layers and model["attention"] == attention and model["family"] == "gpt2"
    gguf = Path(next(row["path"] for row in events if row.get("event") == "export" and row.get("format") == "gguf"))
    checked = invoke([sys.executable, ROOT / "spec/tools/ir_check.py", gguf], directory, "ir_check")
    assert checked.returncode == 0, checked.stdout + checked.stderr
    meta, weights = read_weights(gguf)
    doc = json.loads(meta["paperkiln.ir"][1])
    assert doc["model"]["n_layers"] == doc["block"]["repeat"] == layers
    assert doc["family_hint"] == attention
    assert meta["general.architecture"][1] == "paperkiln"
    assert {int(name.split(".")[1]) for name in weights if name.startswith("blk.")} == set(range(layers))
    assert meta["paperkiln.block_count"][1] == layers
    assert model["params"] == sum(value.size for value in weights.values())
    # Independent constructor count: untied table/head + positions + final norm;
    # 12*d*d+13*d per GPT2-style block, plus SRD's d->d/4->d predictor.
    d, vocab, ctx = 16, len(TOKENS), 12
    shell = 2 * vocab * d + ctx * d + 2 * d
    block = 12 * d * d + 13 * d
    if attention == "srd":
        hidden = max(1, d // 4)
        block += 2 * d * hidden + hidden + d
    assert model["params"] == shell + layers * block
    card_result = invoke([sys.executable, ROOT / "tools/model_card.py", out, "--spec", spec_path],
                         directory, "card")
    assert card_result.returncode == 0, card_result.stdout + card_result.stderr
    card = json.loads((out / "card.json").read_text())
    assert card["architecture"]["layers"] == layers
    assert card["size_and_training"]["params"] == model["params"]

    references, states = [], []
    for ids in SEQUENCES:
        state = {}
        reference = ir_numpy.forward(doc, weights, ids, states=state)
        assert set(state) == ({"rec"} if attention == "recurrence" else {"lin"})
        assert all(set(apps) == set(range(layers)) for apps in state.values())
        references.append(reference)
        states.append({slot: {str(app): {name: list(value.shape) for name, value in tensors.items()}
                              for app, tensors in apps.items()} for slot, apps in state.items()})
    request = [{"prompt_ids": ids[:1], "continuation_ids": ids[1:] + [1]} for ids in SEQUENCES]
    _, source_rows = native_scores(spec_path, directory, request, "source_rows")
    np_rows = np.concatenate(references, axis=0)
    native_error = compare_rows(source_rows, np_rows)

    ir_path = gguf.with_suffix(".ir.json")
    environment = {**os.environ, "EMBER_IR": str(ir_path), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    ember = subprocess.run([EMBER, str(gguf)], input=json.dumps(SEQUENCES),
                           capture_output=True, text=True, timeout=120, env=environment)
    (directory / "ember_rows.json").write_text(ember.stdout, encoding="utf-8")
    (directory / "ember.stderr").write_text(ember.stderr, encoding="utf-8")
    assert ember.returncode == 0, ember.stderr
    actual = json.loads(ember.stdout)
    assert len(actual) == len(SEQUENCES)
    ember_errors = {mode: max(compare_rows(row[mode], ref) for row, ref in zip(actual, references))
                    for mode in ("prefill", "decode")}

    continuations, expected_rows = [], []
    for prompt in SEQUENCES:
        context, continuation, trace = list(prompt), [], []
        for _ in range(3):
            row = ir_numpy.forward(doc, weights, context)[-1]
            selectable = row.copy()
            selectable[0] = -np.inf  # source score contract excludes unknown token
            token = int(np.argmax(selectable))
            continuation.append(token)
            trace.append(row)
            context.append(token)
        continuations.append({"prompt_ids": prompt, "continuation_ids": continuation})
        expected_rows.extend(trace)
    scored, greedy_rows = native_scores(spec_path, directory, continuations, "greedy")
    assert all(row["argmax"] == req["continuation_ids"] for row, req in zip(scored, continuations))
    greedy_error = compare_rows(greedy_rows, expected_rows)
    for row, logits in zip(scored, np.asarray(expected_rows).reshape(len(SEQUENCES), 3, -1)):
        targets = row["argmax"]
        lp = []
        for values, target in zip(logits.astype(np.float64), targets):
            maximum = values.max()
            lp.append(float(values[target] - maximum - np.log(np.exp(values - maximum).sum())))
        assert np.max(np.abs(np.asarray(row["logprobs"]) - lp)) <= 1e-5 * max(1.0, max(abs(v) for v in lp))

    ember_greedy = {}
    for mode in ("prefill", "decode"):
        replies = []
        traces = []
        for prompt in SEQUENCES:
            context, reply = list(prompt), []
            for _ in range(3):
                run = subprocess.run([EMBER, str(gguf)], input=json.dumps([context]), capture_output=True,
                                     text=True, timeout=120, env=environment)
                assert run.returncode == 0, run.stderr
                trace = json.loads(run.stdout)
                assert len(trace) == 1
                row = np.asarray(trace[0][mode], np.float64)[-1]
                assert np.isfinite(row).all()
                row[0] = -np.inf
                token = int(row.argmax())
                reply.append(token)
                context.append(token)
                traces.append({"context": context[:-1], "response": trace[0], "stderr": run.stderr})
            replies.append(reply)
        assert replies == [row["continuation_ids"] for row in continuations]
        ember_greedy[mode] = len(replies)
        (directory / f"ember_greedy_{mode}.json").write_text(json.dumps(traces), encoding="utf-8")

    assert sha(BIN) == identity["mtstudio"] and sha(EMBER) == identity["ember"]
    assert identity["sources"] == {name: sha(ROOT / name) for name in SOURCE_FILES}
    receipt = {"status": "PASS_CPU_ONLY", "attention": attention, "layers": layers, "steps": 3,
               "params": model["params"], "spec_sha256": sha(spec_path), "gguf_sha256": sha(gguf),
               "ir_sha256": sha(ir_path), "identity": identity, "numpy_source": native_error,
               "numpy_ember": ember_errors, "source_greedy_rows": greedy_error,
               "ember_free_running_greedy": ember_greedy,
               "greedy_sequences": len(SEQUENCES), "new_tokens_each": 3, "state_shapes": states,
               "rows_each_mode": sum(map(len, SEQUENCES)), "gpu_calls": 0, "provider_calls": 0}
    (directory / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


@pytest.mark.parametrize("attention", ["kimi", "srd", "recurrence"])
@pytest.mark.parametrize("depth", [0, -1, 1.5, True])
def test_invalid_depth_refuses_before_output_creation(tmp_path, attention, depth):
    name = f"bad_{attention}_{str(depth).replace('.', '_')}"
    directory = directory_for(tmp_path, name)
    path, spec = make_spec(directory, attention, depth)
    result = invoke([BIN, "run", path], directory, "refused")
    assert result.returncode != 0
    assert not Path(spec["out_dir"]).exists()


def safetensor_bits(path):
    raw = path.read_bytes()
    count = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + count])
    payload = raw[8 + count:]
    return {name: (entry["dtype"], entry["shape"],
                   payload[entry["data_offsets"][0]:entry["data_offsets"][1]])
            for name, entry in header.items() if name != "__metadata__"}


@pytest.mark.skipif(not BASE, reason="preserved historical baseline binary required")
@pytest.mark.parametrize("attention", ["kimi", "srd", "recurrence"])
def test_default_two_depth_is_bit_identical_to_historical_binary(tmp_path, attention):
    assert sha(BASE) == BASE_SHA256  # exact author-Flex receipt binary
    directory = directory_for(tmp_path, f"historical_{attention}_2")
    initial = {"current": sha(BIN), "baseline": sha(BASE),
               "sources": {name: sha(ROOT / name) for name in SOURCE_FILES}}
    records = {}
    for label, binary in (("before", BASE), ("after", BIN)):
        case = directory / label
        case.mkdir()
        spec_path, spec = make_spec(case, attention, 2)
        # Omit the override to exercise the historical/default preset depth.
        del spec["arch"]["custom"]["layers"]
        spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        result = invoke([binary, "run", spec_path], case, "train")
        assert result.returncode == 0, result.stdout + result.stderr
        events = [json.loads(line) for line in (Path(spec["out_dir"]) / "events.jsonl").read_text().splitlines()]
        model = next(row for row in events if row["event"] == "model")
        done = next(row for row in events if row["event"] == "done")
        assert model["layers"] == 2 and done["final_step"] == 3
        exported = {row["format"]: Path(row["path"]) for row in events if row.get("event") == "export"}
        requests = [{"prompt_ids": ids[:1], "continuation_ids": ids[1:] + [1]} for ids in SEQUENCES]
        score, rows = native_scores(spec_path, case, requests, "source_rows", binary)
        meta, weights = read_weights(exported["gguf"])
        doc = json.loads(meta["paperkiln.ir"][1])
        greedy = []
        for prompt in SEQUENCES:
            context, continuation = list(prompt), []
            for _ in range(3):
                values = ir_numpy.forward(doc, weights, context)[-1].copy()
                values[0] = -np.inf
                token = int(values.argmax())
                continuation.append(token)
                context.append(token)
            greedy.append({"prompt_ids": prompt, "continuation_ids": continuation})
        greedy_score, greedy_rows = native_scores(spec_path, case, greedy, "greedy", binary)
        assert [row["argmax"] for row in greedy_score] == [row["continuation_ids"] for row in greedy]
        records[label] = {"weights": safetensor_bits(exported["safetensors"]),
                          "score": score, "rows": rows, "greedy_score": greedy_score,
                          "greedy_rows": greedy_rows, "greedy": greedy,
                          "spec_sha256": sha(spec_path), "gguf_sha256": sha(exported["gguf"]),
                          "safetensors_sha256": sha(exported["safetensors"])}
    before, after = records["before"], records["after"]
    assert before["weights"] == after["weights"]  # every named dtype/shape/raw value byte
    assert before["greedy"] == after["greedy"]
    for key in ("rows", "greedy_rows"):
        assert np.array_equal(before[key].view(np.uint32), after[key].view(np.uint32))
    for key in ("score", "greedy_score"):
        assert [row["argmax"] for row in before[key]] == [row["argmax"] for row in after[key]]
        for left, right in zip(before[key], after[key]):
            assert np.array_equal(np.asarray(left["logprobs"], np.float64).view(np.uint64),
                                  np.asarray(right["logprobs"], np.float64).view(np.uint64))
    assert sha(BASE) == initial["baseline"] and sha(BIN) == initial["current"]
    assert initial["sources"] == {name: sha(ROOT / name) for name in SOURCE_FILES}
    receipt = {"status": "PASS_CPU_BITWISE", "attention": attention, "layers": 2,
               "steps": 3, "identity": initial, "tensor_count": len(before["weights"]),
               "tensor_raw_sha256": {name: hashlib.sha256(values[2]).hexdigest()
                                     for name, values in before["weights"].items()},
               "all_named_tensor_dtype_shape_and_value_bytes_equal": True,
               "full_score_row_bits_equal": True, "greedy_score_row_bits_equal": True,
               "greedy_ids_and_float64_logprob_bits_equal": True,
               "scored_full_rows": len(before["rows"]), "greedy_sequences": len(SEQUENCES),
               "new_tokens_each": 3, "gpu_calls": 0, "provider_calls": 0}
    for label, record in records.items():
        receipt[label] = {key: record[key] for key in ("spec_sha256", "gguf_sha256", "safetensors_sha256")}
    (directory / "validation.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
