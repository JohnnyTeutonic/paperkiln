"""Independent binary64 replay of the registered selective-scan arrays and gradients."""

import argparse
import json
from pathlib import Path
import numpy as np

BOUND = 3e-5


def need(condition, message):
    if not condition:
        raise ValueError(message)


def fixtures(sl, batch, d, n, softplus, skip):
    sizes = [
        (sl * batch, d),
        (sl * batch, d),
        (sl * batch, n),
        (sl * batch, n),
        (d, n),
        (1, d),
    ]
    values = []
    for v, (rows, cols) in enumerate(sizes):
        a = np.array(
            [((i * 7 + v * 5 + 3) % 23 - 11) / 19 for i in range(rows * cols)],
            dtype=np.float64,
        )
        if v == 1:
            a = np.array(
                [
                    [-20, -0.7, 0, 19.75, 20, 20.25][i % 6]
                    if softplus
                    else (i % 5 - 2) / 10
                    for i in range(a.size)
                ]
            )
        if v == 4:
            a = np.array([-0.2 - (i % 7) / 25 for i in range(a.size)])
        values.append(a.astype(np.float32).astype(np.float64).reshape(rows, cols))
    return values


def upstream(T, D):
    return np.array(
        [((i * 3 + 2) % 17 - 8) / 13 for i in range(T * D)], dtype=np.float32
    ).astype(np.float64).reshape(T, D) / (T * D)


def oracle(inputs, sl, softplus, skip):
    x, dt, B, C, A, Dskip = inputs
    T, D = x.shape
    N = A.shape[1]
    batch = T // sl
    states = np.zeros((batch, sl + 1, D, N), dtype=np.float64)
    delta = np.where(dt <= 20, np.logaddexp(0, dt), dt) if softplus else dt.copy()
    y = np.zeros_like(x)
    derivatives = [np.zeros_like(a) for a in inputs]
    dx, ddt, dB, dC, dA, dD = derivatives
    for q in range(batch):
        for t in range(sl):
            row = q * sl + t
            dl = delta[row]
            prev = states[q, t]
            states[q, t + 1] = (
                prev * np.exp(dl[:, None] * A)
                + (x[row] * dl)[:, None] * B[row][None, :]
            )
            y[row] = (states[q, t + 1] * C[row][None, :]).sum(axis=1)
            if skip:
                y[row] += x[row] * Dskip[0]
    dy = upstream(T, D)
    for q in range(batch):
        carry = np.zeros((D, N))
        for t in range(sl - 1, -1, -1):
            row = q * sl + t
            dl = delta[row]
            a = np.exp(dl[:, None] * A)
            ds = dy[row, :, None] * C[row, None, :] + carry
            dpre = ds * states[q, t] * a
            dC[row] = (dy[row, :, None] * states[q, t + 1]).sum(axis=0)
            dA += dpre * dl[:, None]
            dB[row] = (ds * (x[row] * dl)[:, None]).sum(axis=0)
            d_xdl = (ds * B[row, None, :]).sum(axis=1)
            d_delta = (dpre * A).sum(axis=1) + d_xdl * x[row]
            dx[row] = d_xdl * dl
            if skip:
                dx[row] += dy[row] * Dskip[0]
                dD[0] += dy[row] * x[row]
            ddt[row] = d_delta
            if softplus:
                ddt[row] *= np.where(dt[row] <= 20, 1 / (1 + np.exp(-dt[row])), 1)
            carry = ds * a
    return y, derivatives


def inventory():
    expected = set()
    for sp in (False, True):
        for skip in (False, True):
            full = 63 if skip else 31
            for mask in range(full + 1):
                expected.add((2, 2, 3, 2, sp, skip, mask, "deferred"))
            for sl, batch, d, n in (
                (1, 1, 1, 1),
                (3, 2, 3, 2),
                (9, 2, 5, 3),
                (4, 2, 2, 5),
            ):
                expected.add((sl, batch, d, n, sp, skip, full, "deferred"))
            for mode in ("write_through", "host_forward", "host_backward"):
                expected.add((3, 2, 3, 2, sp, skip, full, mode))
    return expected


def replay(path, require_cuda=False):
    parsed = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.startswith("{")
    ]
    need(
        parsed
        and parsed[-1]
        == {
            "pass": True,
            "cases": 220,
            "actual_cuda": require_cuda,
            "devcheck": True,
            "finite_difference": True,
        },
        "strict final proof differs",
    )
    rows = [x for x in parsed if x.get("raw_case") is True]
    need(len(rows) == 220, "case count differs")
    seen = set()
    worst = 0
    quantities = 0
    inputkeys = ("x", "dt", "B", "C", "A", "Dskip")
    gradkeys = ("dx", "ddt", "dB", "dC", "dA", "dD")
    for r in rows:
        sl, batch, d, n, sp, skip, mask, mode = (
            r[k] for k in ("seq", "batch", "d", "n", "softplus", "skip", "mask", "mode")
        )
        need(
            all(type(v) is int for v in (sl, batch, d, n, mask))
            and type(sp) is bool
            and type(skip) is bool,
            "metadata types differ",
        )
        key = (sl, batch, d, n, sp, skip, mask, mode)
        need(key in inventory() and key not in seen, "unregistered or duplicate case")
        seen.add(key)
        need(r.get("actual_cuda") is require_cuda, "case device proof differs")
        expected_fwd = 1 if require_cuda and mode != "host_forward" else 0
        expected_bwd = 1 if require_cuda and mask and mode != "host_backward" else 0
        need(
            type(r.get("device_fwd")) is int and r["device_fwd"] == expected_fwd,
            "forward kernel proof differs",
        )
        need(
            type(r.get("device_bwd")) is int and r["device_bwd"] == expected_bwd,
            "backward kernel proof differs",
        )
        need(
            type(r.get("worst")) in (int, float)
            and np.isfinite(r["worst"])
            and 0 <= r["worst"] < BOUND,
            "self metric differs",
        )
        inputs = fixtures(sl, batch, d, n, sp, skip)
        for k, ref in zip(inputkeys, inputs):
            if k == "Dskip" and not skip:
                need(k not in r, "unexpected skip input")
                continue
            value = np.asarray(r[k], dtype=np.float64)
            need(
                value.shape == ref.shape and np.isfinite(value).all(),
                "input shape or finite contract differs",
            )
            need(np.max(np.abs(value - ref)) < 1e-7, "registered input fixture differs")
        y, grads = oracle(inputs, sl, sp, skip)
        for k, ref, requested in [("y", y, True)] + [
            (k, a, bool(mask & (1 << i)))
            for i, (k, a) in enumerate(zip(gradkeys, grads))
        ]:
            if not requested:
                need(k not in r, "unexpected unrequested gradient")
                continue
            a = np.asarray(r[k], dtype=np.float64)
            need(
                a.shape == ref.shape and np.isfinite(a).all(),
                "result shape/nonfinite differs",
            )
            error = float(np.max(np.abs(a - ref) / (1 + np.abs(ref))))
            need(error < BOUND, (key, k, error))
            worst = max(worst, error)
            quantities += 1
    need(seen == inventory(), "registered matrix incomplete")
    return {
        "pass": True,
        "actual_cuda": require_cuda,
        "cases": 220,
        "quantities": quantities,
        "independent_numpy_replay": True,
        "worst_normalized_error": worst,
        "bound": BOUND,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("log")
    p.add_argument("--require-cuda", action="store_true")
    a = p.parse_args()
    print(json.dumps(replay(a.log, a.require_cuda), indent=2))
