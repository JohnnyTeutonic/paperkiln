"""The IR's expr op: the numpy reference (tools/ir_expr.py) on its own, and
against an engine's expr evaluator (paperkiln's tools/expr_eval.cpp, say).

The reference alone: the grammar (accepted and rejected strings, with the
column a rejection names), hand-computed values, nearest-float literals. With
the CLI (EXPR_EVAL_BIN; skipped otherwise): random
inputs through every primitive and the expressions papers use, the C++
result within ROUNDING (|a - b| <= 1e-5 * max(1, max|ref|), the IR's CPU
tolerance) of the reference, non-finite entries in the same places, and the
two parsers rejecting the same strings at the same columns.
"""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ir_expr  # noqa: E402
from ir_expr import ExprError, evaluate, parse  # noqa: E402

F = np.float32
TOL = 1e-5

GOOD = [
    ("x", 1), ("a", 2), ("  x * sigmoid(1.702*x) ", 1), ("max(x, 0)^2", 1), ("where(x, x, 0.01*x)", 1),
    ("x/(1+abs(x))", 1), ("(x-rowmean(x))/sqrt(rowmean((x-rowmean(x))^2)+1e-5)", 1), ("x*tanh(a)", 2),
    ("where(b, x, a) - rowmax(x) + rowsum(a)", 3), ("relu(a)*x", 2), ("-x^-2 + 2^3^2 - -1", 1),
    (".5 + 2. + 1.5E+3 + 1e-5 + pi", 1), ("exp(log(sqrt(erf(min(x, 1)))))", 1), ("relu(x)\t", 1), ("3", 1),
]
# (text, inputs, the column the message names) -- tests/test_expr.cpp's list
BAD = [
    ("", 1, 1), ("x +", 1, 4), ("x + * 2", 1, 5), ("2x", 1, 2), ("x ** 2", 1, 4), ("y", 1, 1), ("a", 1, 1), ("b", 2, 1),
    ("c", 3, 1), ("foo(x)", 1, 1), ("exp x", 1, 5), ("max(x)", 1, 1), ("where(x, 1)", 1, 1),
    ("exp(x, x)", 1, 1), ("(x", 1, 3), ("x)", 1, 2), ("1e", 1, 1), (".", 1, 1), ("1e999", 1, 1),
    ("x % 2", 1, 3), ("x = 1", 1, 3), ("pi(x)", 1, 1), ("sigmoid", 1, 8), ("max(x,,0)", 1, 7),
    ("(" * 300 + "x" + ")" * 300, 1, None),
]

# Every primitive, and the activations and gates papers write.
PARITY = [
    ("x", 1), ("x + 0.5", 1), ("0.5 - x", 1), ("x * 1.702", 1), ("x / 3", 1), ("3 / x", 1), ("-x", 1),
    ("x ^ 2", 1), ("2 ^ x", 1), ("abs(x) ^ 0.75", 1), ("exp(x)", 1), ("log(x)", 1), ("sqrt(x)", 1),
    ("tanh(x)", 1), ("erf(x)", 1), ("abs(x)", 1), ("sigmoid(x)", 1), ("relu(x)", 1), ("max(x, 0.5)", 1),
    ("min(x, -0.5)", 1), ("where(x, x, 0.01*x)", 1), ("rowsum(x)", 1), ("rowmean(x)", 1), ("rowmax(x)", 1),
    ("max(x, 0)^2", 1), ("x*sigmoid(1.702*x)", 1), ("x/(1+abs(x))", 1),
    ("0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))", 1), ("0.5*x*(1 + erf(x/sqrt(2)))", 1),
    ("(x-rowmean(x))/sqrt(rowmean((x-rowmean(x))^2)+1e-5)", 1), ("x - rowmax(x) - log(rowsum(exp(x - rowmax(x))))", 1),
    ("x*tanh(a)", 2), ("relu(a)*x", 2), ("where(a, x, -x) + max(x, a) * min(a, 1)", 2),
    ("(x - a) / (1 + exp(rowsum(x*a)/7))", 2), ("x * sigmoid(a) + b", 3), ("2 * pi", 1), ("rowsum(3)", 1), ("where(-1, 1, 2)", 1), ("0.1 + 0.2 - 0.3", 1),
]


def _bin():
    cand = os.environ.get("EXPR_EVAL_BIN")
    return cand if cand and Path(cand).is_file() else None


BIN = _bin()
needs_bin = pytest.mark.skipif(BIN is None, reason="set EXPR_EVAL_BIN to a built expr_eval")


def cpp(cases):
    """Run expr_eval on a list of cases; returns its list of results."""
    r = subprocess.run([BIN], input=json.dumps(cases), capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def as_array(rows):
    conv = {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}
    return np.array([[conv[v] if isinstance(v, str) else v for v in row] for row in rows], dtype=F)


def column(msg):
    return int(msg.split(" at column ")[1].split()[0])


# ---- the reference on its own ----

@pytest.mark.parametrize("text,n", GOOD)
def test_grammar_accepts(text, n):
    parse(text, n)


@pytest.mark.parametrize("text,n,col", BAD)
def test_grammar_rejects_naming_the_column(text, n, col):
    with pytest.raises(ExprError) as e:
        parse(text, n)
    assert str(e.value).startswith("expr: ")
    if col is not None:
        assert column(str(e.value)) == col, str(e.value)


def test_values_by_hand():
    x = np.array([[-2.0, -0.5, 0.0, 3.0], [1.0, 2.0, -1.0, 0.25]], F)
    b = np.array([[1.0, -1.0, 2.0, 0.5], [-3.0, 0.0, 4.0, 1.0]], F)
    assert evaluate("max(x, 0)^2", x).tolist() == [[0, 0, 0, 9], [1, 4, 0, 0.0625]]
    assert evaluate("where(x, x, 0.01*x)", x).tolist() == [[F(0.01) * F(-2), F(0.01) * F(-0.5), 0, 3],
                                                           [1, 2, F(0.01) * F(-1), 0.25]]
    assert evaluate("rowsum(x)", x).tolist() == [[0.5] * 4, [2.25] * 4]
    assert evaluate("rowmean(x)", x).tolist() == [[0.125] * 4, [0.5625] * 4]
    assert evaluate("rowmax(x)", x).tolist() == [[3] * 4, [2] * 4]
    assert evaluate("x * a + 1", [x, b]).tolist() == [[-1, 1.5, 1, 2.5], [-2, 1, -3, 1.25]]
    assert evaluate("min(x, a)", [x, b]).tolist() == [[-2, -1, 0, 0.5], [-3, 0, -1, 0.25]]
    assert evaluate("a", [b, x]).tolist() == x.tolist()  # a is the second input
    assert evaluate("relu(x) - abs(x)", x).tolist() == [[-2, -0.5, 0, 0], [0, 0, -1, 0]]
    # precedence and associativity
    one = np.zeros((1, 1), F)
    for text, want in [("1 + 2 * 3", 7), ("(1 + 2) * 3", 9), ("8 / 2 / 2", 2), ("1 - 2 - 3", -4),
                       ("2 ^ 3 ^ 2", 512), ("-2 ^ 2", -4), ("2 ^ -1", 0.5), ("- - 3", 3)]:
        assert evaluate(text, one)[0, 0] == want, text
    # every step rounds to f32: 0.1 + 0.2 in f32 is not the double sum
    assert evaluate("0.1 + 0.2", one)[0, 0] == F(0.1) + F(0.2)
    assert evaluate("2 * pi", one)[0, 0] == F(2) * F(math.pi)
    assert evaluate("x", x).dtype == np.float32 and evaluate("3", x).shape == (2, 4)


def test_literals_are_the_nearest_float():
    for lit in ["0.1", "1.702", "1e-5", "0.044715", "3.4028234e38", "1.4e-45", "7e-46", "16777217",
                "0.30000001192092896", "2.", ".5", "1.5E+3"]:
        node = parse(lit)
        want = F(float(lit))  # differs from strtof only on a double-rounding tie, none of these
        assert node.value == want, lit
    # 16777217 = 2^24 + 1 is a tie between 2^24 and 2^24 + 2: ties to even.
    assert parse("16777217").value == F(16777216)


def test_rowsum_adds_in_column_order():
    # 1e8 + 1 - 1e8 + ... : pairwise or reordered summation gives another answer.
    x = np.array([[1e8, 1.0, -1e8, 1.0, 1.0]], F)
    s = F(0)
    for v in x[0]:
        s = F(s + v)
    assert evaluate("rowsum(x)", x)[0, 0] == s == F(2)


def test_inputs_checked():
    with pytest.raises(ExprError):
        evaluate("x + a", [np.zeros((2, 3), F), np.zeros((2, 4), F)])
    with pytest.raises(ExprError):
        parse("x", 0)


# ---- against an engine's evaluator (EXPR_EVAL_BIN) ----

@needs_bin
def test_cpp_matches_numpy_on_random_inputs():
    rng = np.random.default_rng(20261022)
    cases, refs = [], []
    for text, n in PARITY:
        for T, w in [(3, 7), (5, 64)]:
            xs = [rng.normal(0.0, 1.5, (T, w)).astype(F) for _ in range(n)]
            xs[0][0, :2] = [0.0, -0.0]  # the kinks and signed zeros
            cases.append({"expr": text, "inputs": [v.tolist() for v in xs]})
            refs.append(evaluate(text, xs))
    got = cpp(cases)
    worst = 0.0
    for (case, ref, out) in zip(cases, refs, got):
        assert "out" in out, (case["expr"], out)
        a = as_array(out["out"])
        assert a.shape == ref.shape, case["expr"]
        fin = np.isfinite(ref)
        assert np.array_equal(fin, np.isfinite(a)), case["expr"]
        assert np.array_equal(np.isnan(ref), np.isnan(a)), case["expr"]
        assert np.array_equal(a[~fin & ~np.isnan(ref)], ref[~fin & ~np.isnan(ref)]), case["expr"]  # same infinities
        if fin.any():
            scale = max(1.0, float(np.abs(ref[fin]).max()))
            err = float(np.abs(a[fin].astype(np.float64) - ref[fin]).max()) / scale
            assert err <= TOL, (case["expr"], err)
            worst = max(worst, err)
    print(f"{len(cases)} cases, worst scaled difference {worst:.2e}")


@needs_bin
def test_cpp_and_numpy_reject_the_same_strings_at_the_same_columns():
    cases = [{"expr": t, "n_inputs": n} for t, n in GOOD] + [{"expr": t, "n_inputs": n} for t, n, _ in BAD]
    got = cpp(cases)
    for case, out in zip(cases, got):
        try:
            parse(case["expr"], case["n_inputs"])
            py = None
        except ExprError as e:
            py = str(e)
        if py is None:
            assert out == {"ok": True}, (case, out)
        else:
            assert "error" in out, (case, out)
            if " at column " in py:
                assert column(out["error"]) == column(py), (case["expr"], out["error"], py)
