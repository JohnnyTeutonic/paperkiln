import importlib.util
from pathlib import Path
import numpy as np
import pytest

s = importlib.util.spec_from_file_location(
    "replay", Path(__file__).with_name("replay_gemm_transpose.py")
)
q = importlib.util.module_from_spec(s)
s.loader.exec_module(q)


def fixture():
    rows = ["DEVICE NVIDIA L4"]
    for f in range(4):
        for m, n, k in q.SHAPES:
            for mode in range(3):
                a = np.array(
                    [(int((i * 17 + 3) % 41) - 20) / 31 for i in range(m * k)],
                    dtype=np.float32,
                )
                b = np.array(
                    [(int((i * 13 + 7) % 43) - 21) / 29 for i in range(k * n)],
                    dtype=np.float32,
                )
                init = np.array(
                    [(int(i % 11) - 5) / 17 for i in range(m * n)], dtype=np.float32
                )
                aa = a.reshape((k, m) if f & 2 else (m, k))
                bb = b.reshape((n, k) if f & 1 else (k, n))
                ref = (aa.T if f & 2 else aa).astype(np.float64) @ (
                    bb.T if f & 1 else bb
                ).astype(np.float64)
                if mode:
                    ref += init.reshape(m, n)
                c = ref.astype(np.float32).reshape(-1)
                err = float(np.max(abs(c.reshape(m, n) - ref) / (1 + abs(ref))))
                rows.append(f"CASE {f} {m} {n} {k} {mode} {err:.17g}")
                for name, v in (("A", a), ("B", b), ("I", init), ("C", c), ("OLD", c)):
                    rows.append(name + "".join(" " + float(x).hex() for x in v))
    rows.append("PASS 96")
    return "\n".join(rows) + "\n"


def test_complete():
    assert q.replay(fixture())["cases"] == 96


@pytest.mark.parametrize("kind", range(9))
def test_reject(kind):
    t = fixture()
    if kind == 0:
        t = t.replace("NVIDIA L4", "CPU")
    if kind == 1:
        t = t.replace("CASE 0 1 1 0 0", "CASE 0 2 1 0 0", 1)
    if kind == 2:
        t = t.replace("PASS 96", "PASS 95")
    if kind == 3:
        t = t + "junk\n"
    if kind == 4:
        t = t.replace("I -0x1.", "I -0x2.", 1)
    if kind == 5:
        t = t.replace("C 0x0.0p+0", "C nan", 1)
    if kind == 6:
        t = t.replace("OLD 0x0.0p+0", "OLD 0x1.0p+0", 1)
    if kind == 7:
        t = t.replace("CASE 0 1 1 0 0 0", "CASE 0 1 1 0 0 nan", 1)
    if kind == 8:
        t = t.replace("A\n", "A 0x0.0p+0\n", 1)
    with pytest.raises(ValueError):
        q.replay(t)


def test_timing():
    rows = ["DEVICE NVIDIA L4"]
    for f in range(4):
        for pair in range(7):
            for leg in range(2):
                rows.append(
                    f"TIMING {f} {pair} {int((not leg) if pair % 2 else bool(leg))} 0.1"
                )
    rows.append("TIMING_PASS 56")
    text = "\n".join(rows)
    q.timings(text)
    for bad in (
        text.replace("0.1", "nan", 1),
        text.replace("TIMING_PASS 56", "TIMING_PASS 55"),
        text + "\njunk",
        text.replace("TIMING 0 0 0", "TIMING 0 0 1", 1),
    ):
        with pytest.raises(ValueError):
            q.timings(bad)
