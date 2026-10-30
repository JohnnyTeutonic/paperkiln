import numpy as np
import pytest

from bf16_attention_reference import (
    ARRAY_NAMES,
    HEADER,
    MAGIC,
    bf16,
    read_output,
    reference,
    validate_shape,
    write_input,
)


def test_rne_ties_and_specials():
    x = np.array(
        [1.00390625, 1.01171875, -1.00390625, -1.01171875, 0.0, -0.0, np.inf, -np.inf],
        dtype=np.float32,
    )
    expected = np.array(
        [1.0, 1.015625, -1.0, -1.015625, 0.0, -0.0, np.inf, -np.inf], dtype=np.float32
    )
    assert np.array_equal(bf16(x).view(np.uint32), expected.view(np.uint32))
    for bits in (0x7F800001, 0x7FFFFFFF, 0xFF800001):
        assert np.isnan(bf16(np.array([bits], dtype=np.uint32).view(np.float32)))[0]


def test_all_bf16_values_are_idempotent():
    bits = np.arange(65536, dtype=np.uint32) << 16
    x = bits.view(np.float32)
    finite = np.isfinite(x)
    assert np.array_equal(bf16(x[finite]).view(np.uint32), bits[finite])


@pytest.mark.parametrize("causal", [False, True])
def test_surrogate_matches_smooth_full_precision_gradient(causal):
    rng = np.random.default_rng(149)
    q, k, v, dy = [
        rng.normal(0, 0.2, (1, 2, 3, 8)).astype(np.float32) for _ in range(4)
    ]
    got = reference(q, k, v, dy, causal, low_precision=False)

    def loss(xs):
        q_, k_, v_ = [x.astype(np.float64) for x in xs]
        s = q_ @ k_.swapaxes(-1, -2) / np.sqrt(q.shape[-1])
        if causal:
            s = np.where(np.triu(np.ones((3, 3), dtype=bool), 1), -np.inf, s)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        return float(np.sum((p @ v_) * dy.astype(np.float64)))

    original = [x.astype(np.float64) for x in (q, k, v)]
    for i, name in enumerate(("dq", "dk", "dv")):
        fd = np.empty_like(original[i])
        eps = 1e-5
        for index in np.ndindex(original[i].shape):
            plus = [x.copy() for x in original]
            minus = [x.copy() for x in original]
            plus[i][index] += eps
            minus[i][index] -= eps
            fd[index] = (loss(plus) - loss(minus)) / (2 * eps)
        np.testing.assert_allclose(got[name], fd, rtol=2e-5, atol=2e-7)
    if causal:
        assert np.count_nonzero(np.triu(got["prob"], 1)) == 0
        assert np.count_nonzero(np.triu(got["ds"], 1)) == 0
    np.testing.assert_allclose(got["prob"].sum(-1), 1, atol=1e-7)


def test_batches_and_heads_are_isolated():
    rng = np.random.default_rng(19)
    xs = [rng.normal(size=(2, 3, 7, 8)).astype(np.float32) for _ in range(4)]
    full = reference(*xs)
    for b in range(2):
        for h in range(3):
            one = reference(*(x[b : b + 1, h : h + 1] for x in xs))
            for name in ARRAY_NAMES:
                np.testing.assert_array_equal(
                    full[name][b : b + 1, h : h + 1], one[name]
                )


def test_binary_contract_and_invalid_data(tmp_path):
    xs = [np.zeros((1, 2, 3, 8), dtype=np.float32) for _ in range(4)]
    path = tmp_path / "in.bin"
    write_input(path, *xs, True, True)
    assert HEADER.unpack_from(path.read_bytes()) == (MAGIC, 1, 1, 2, 3, 8, 1, 1)
    with pytest.raises(FileExistsError):
        write_input(path, *xs, True, True)
    out = tmp_path / "out.bin"
    ref = reference(*xs)
    out.write_bytes(
        path.read_bytes()[: HEADER.size]
        + b"".join(ref[n].astype("<f4").tobytes() for n in ARRAY_NAMES)
    )
    for name, array in read_output(out).items():
        np.testing.assert_array_equal(array, ref[name])
    out.write_bytes(out.read_bytes() + b"extra")
    with pytest.raises(ValueError, match="length"):
        read_output(out)
    out.write_bytes(b"bad")
    with pytest.raises(ValueError, match="short"):
        read_output(out)
    xs[0][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        write_input(tmp_path / "bad.bin", *xs, True, True)


@pytest.mark.parametrize(
    "shape",
    [(0, 1, 1, 8), (1, 1, 2, 7), (1, 1, 2049, 8), (16, 32, 1, 8), (1, 1, 1, 264)],
)
def test_unsupported_layout_refused(shape):
    with pytest.raises(ValueError):
        validate_shape(shape)


def test_stage_audit_rejects_corrupted_gradient():
    from bf16_attention_check import audit_output

    rng = np.random.default_rng(49)
    xs = [rng.normal(size=(1, 1, 3, 8)).astype(np.float32) for _ in range(4)]
    got = reference(*xs)
    audit_output(xs, got, True, True)
    got["dk"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="per-stage"):
        audit_output(xs, got, True, True)


def test_completed_matrix_replays_without_execution_and_detects_header_drift(
    tmp_path, monkeypatch
):
    import json
    import subprocess
    import sys
    import bf16_attention_check as gate

    monkeypatch.setattr(gate, "SHAPES", [(1, 1, 3, 8)])
    binary = tmp_path / "fake-probe"
    binary.write_bytes(b"unit-test-only; not a CUDA executable")
    output = tmp_path / "unit-results"
    calls = []
    corrupt_header = [False]
    mutate_binary = [False]

    def fake_run(args, stdout, stderr, timeout):
        calls.append(args)
        data = __import__("pathlib").Path(args[1]).read_bytes()
        _, _, b, h, s, d, causal, low = HEADER.unpack_from(data)
        xs = np.frombuffer(data, "<f4", offset=HEADER.size).reshape(4, b, h, s, d)
        got = reference(*xs, causal=bool(causal), low_precision=bool(low))
        __import__("pathlib").Path(args[2]).write_bytes(
            data[: HEADER.size]
            + b"".join(got[n].astype("<f4").tobytes() for n in ARRAY_NAMES)
        )
        if corrupt_header[0]:
            target = __import__("pathlib").Path(args[2])
            changed = bytearray(target.read_bytes())
            changed[28:32] = (1 - low).to_bytes(4, "little")
            target.write_bytes(changed)
        if mutate_binary[0]:
            binary.write_bytes(binary.read_bytes() + b"changed-during-execution")
        stdout.write(
            json.dumps(
                {
                    "device": "NVIDIA L4",
                    "sm": 89,
                    "device_uuid": "0" * 32,
                    "iterations": 20,
                    "forward_backward_ms": 1.0,
                }
            ).encode()
        )
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    argv = ["gate", "--binary", str(binary), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    gate.main()
    assert len(calls) == 4
    monkeypatch.setattr(sys, "argv", argv + ["--replay"])
    gate.main()
    assert len(calls) == 4
    # Editing both case JSON and final report must not override the captured
    # stdout or expected case identity/mask/precision metadata.
    receipt = output / "case00_mask0_bf160.json"
    report = output / "report.json"
    original_receipt, original_report = receipt.read_bytes(), report.read_bytes()
    for field, value in (("case", "WRONG_CASE"), ("shape", [1, 1, 4, 8]),
                         ("causal", True), ("bf16", True), ("timing", {
                             "device": "NVIDIA L4", "sm": 89, "device_uuid": "0" * 32,
                             "iterations": 20, "forward_backward_ms": -999})):
        changed = json.loads(original_receipt)
        changed[field] = value
        receipt.write_text(json.dumps(changed))
        altered_report = json.loads(original_report)
        altered_report["cases"][0] = changed
        report.write_text(json.dumps(altered_report))
        with pytest.raises(AssertionError, match="case receipt differs"):
            gate.main()
        assert not (output / "HALTED").exists()  # replay must remain read-only
        assert len(calls) == 4
        receipt.write_bytes(original_receipt)
        report.write_bytes(original_report)
    victim = output / "case00_mask1_bf161.out.bin"
    raw = bytearray(victim.read_bytes())
    raw[24:28] = (0).to_bytes(4, "little")
    victim.write_bytes(raw)
    with pytest.raises(AssertionError):
        gate.main()
    assert len(calls) == 4
    # Fresh executions must bind the binary header and reject a binary
    # replaced while the child was running, not merely pin it at startup.
    for option, directory in ((corrupt_header, "fresh-wrong-header"),
                              (mutate_binary, "fresh-binary-drift")):
        option[0] = True
        fresh = tmp_path / directory
        monkeypatch.setattr(sys, "argv", ["gate", "--binary", str(binary), "--output", str(fresh)])
        with pytest.raises(AssertionError):
            gate.main()
        assert (fresh / "HALTED").is_file()
        assert not (fresh / "report.json").exists()
        option[0] = False


def test_optimized_python_refuses_before_any_results_created(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    output = tmp_path / "must-not-exist"
    gate = Path(__file__).with_name("bf16_attention_check.py")
    result = subprocess.run([sys.executable, "-O", str(gate), "--binary", "unused", "--output", str(output)],
                            text=True, capture_output=True)
    assert result.returncode != 0
    assert "optimized Python disables verification" in result.stderr
    assert not output.exists()


def test_halted_result_cannot_be_retried(tmp_path, monkeypatch):
    import sys
    import bf16_attention_check as gate

    binary = tmp_path / "fake-probe"
    binary.write_bytes(b"not executed")
    output = tmp_path / "halt"
    output.mkdir()
    (output / "HALTED").write_text("preserved failure")
    monkeypatch.setattr(
        sys, "argv", ["gate", "--binary", str(binary), "--output", str(output)]
    )
    with pytest.raises(RuntimeError, match="HALTED"):
        gate.main()
