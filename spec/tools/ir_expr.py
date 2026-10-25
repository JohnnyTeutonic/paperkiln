"""numpy f32 reference for the IR's `expr` op: runtime-defined elementwise ops.

The grammar and semantics are microtorch/ir.hpp's (op `expr`) and
microtorch/expr.hpp's; ops::expr (src/expr.cpp) is the C++ implementation,
and tests/test_ir_expr.py compares the two. Every primitive is computed in
f32 and rounded to f32 after it; row sums add in ascending column order.

    parse(text, n_inputs=1) -> Node          raises ExprError (a ValueError)
    evaluate(text_or_node, inputs) -> [T, w] float32 array

    >>> import numpy as np
    >>> evaluate("max(x, 0)^2", np.array([[-1.0, 2.0]], np.float32))
    array([[0., 4.]], dtype=float32)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import math
import re

import numpy as np

try:  # erf: scipy's ufunc if present, else math.erf per element (both double, rounded to f32)
    from scipy.special import erf as _erf_ufunc
except ImportError:  # pragma: no cover - depends on the environment
    _erf_ufunc = None

F32 = np.float32
MAX_INPUTS = 23  # x, then a .. v (x the first input, a the second; as papers/llm_extract.py)
MAX_DEPTH = 200
FUNCS = {"exp": 1, "log": 1, "sqrt": 1, "tanh": 1, "sigmoid": 1, "erf": 1, "abs": 1, "relu": 1,
         "max": 2, "min": 2, "where": 3, "rowsum": 1, "rowmean": 1, "rowmax": 1}
PI = F32(math.pi)  # the nearest float to pi


class ExprError(ValueError):
    pass


@dataclass
class Node:
    kind: str           # "num", "in", "neg", "+", "-", "*", "/", "^", or a function name
    column: int         # 1-based, where the node starts in the text
    value: np.float32 = F32(0)  # "num"
    input: int = 0              # "in": 0 for x, 1 for a, 2 for b, ...
    args: list = field(default_factory=list)


_NUM = re.compile(r"(\d+\.?\d*|\.\d+)([eE][+-]?\d*)?")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _nearest_f32(lit: str) -> np.float32:
    """The nearest float to the decimal literal, ties to even (what strtof returns)."""
    with np.errstate(over="ignore"):
        f = F32(float(lit))
    if not np.isfinite(f):
        return f
    exp = re.search(r"[eE]([+-]?\d+)$", lit)
    if exp and abs(int(exp.group(1))) > 400:  # beyond double range: float() already decided
        return f
    q = Fraction(lit)
    with np.errstate(over="ignore"):  # the neighbour of the largest float is inf, dropped below
        cands = {f, np.nextafter(f, F32(np.inf)), np.nextafter(f, F32(-np.inf))}
    cands = [c for c in cands if np.isfinite(c)]
    return min(cands, key=lambda c: (abs(Fraction(float(c)) - q), int(c.view(np.uint32)) & 1))


class _Parser:
    def __init__(self, text: str, n_inputs: int):
        self.s, self.n = text, n_inputs
        self.toks = self._lex()
        self.pos = 0

    def fail(self, what: str, column: int):
        raise ExprError(f'expr: {what} at column {column} in "{self.s}"')

    def _lex(self):
        s, i, toks = self.s, 0, []
        while i < len(s):
            ch, col = s[i], i + 1
            if ch in " \t":
                i += 1
            elif ch.isascii() and (ch.isdigit() or ch == "."):
                m = _NUM.match(s, i)
                if not m:
                    self.fail("malformed number", col)
                if m.group(2) is not None and not re.search(r"\d$", m.group(2)):
                    self.fail("malformed number (exponent without digits)", col)
                lit = m.group(0)
                v = _nearest_f32(lit)
                if not np.isfinite(v):
                    self.fail(f"number {lit} is out of float range", col)
                toks.append(("num", lit, v, col))
                i = m.end()
            elif ch.isascii() and (ch.isalpha() or ch == "_"):
                m = _NAME.match(s, i)
                toks.append(("name", m.group(0), None, col))
                i = m.end()
            elif ch in "+-*/^":
                toks.append(("op", ch, None, col))
                i += 1
            elif ch in "(),":
                toks.append((ch, ch, None, col))
                i += 1
            else:
                self.fail(f"unexpected character '{ch}'", col)
        toks.append(("end", "end", None, len(s) + 1))
        return toks

    def peek(self):
        return self.toks[self.pos]

    def next(self):
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def at_op(self, op):
        t = self.peek()
        return t[0] == "op" and t[1] == op

    def enter(self, depth):
        if depth > MAX_DEPTH:
            self.fail(f"nesting deeper than {MAX_DEPTH}", self.peek()[3])

    def parse(self) -> Node:
        if self.toks[0][0] == "end":
            self.fail("empty expression", 1)
        root = self.expr(0)
        t = self.peek()
        if t[0] != "end":
            self.fail(f"expected an operator or the end, got '{t[1]}'", t[3])
        return root

    def expr(self, depth):
        self.enter(depth)
        left = self.term(depth + 1)
        while self.at_op("+") or self.at_op("-"):
            op = self.next()
            left = Node(op[1], op[3], args=[left, self.term(depth + 1)])
        return left

    def term(self, depth):
        self.enter(depth)
        left = self.unary(depth + 1)
        while self.at_op("*") or self.at_op("/"):
            op = self.next()
            left = Node(op[1], op[3], args=[left, self.unary(depth + 1)])
        return left

    def unary(self, depth):
        self.enter(depth)
        if self.at_op("-"):
            op = self.next()
            return Node("neg", op[3], args=[self.unary(depth + 1)])
        return self.power(depth + 1)

    def power(self, depth):
        self.enter(depth)
        base = self.primary(depth + 1)
        if self.at_op("^"):
            op = self.next()
            return Node("^", op[3], args=[base, self.unary(depth + 1)])
        return base

    def primary(self, depth):
        self.enter(depth)
        kind, text, value, col = self.next()
        if kind == "num":
            return Node("num", col, value=value)
        if kind == "(":
            inner = self.expr(depth + 1)
            t = self.peek()
            if t[0] != ")":
                self.fail(f"expected ')', got '{t[1]}'", t[3])
            self.next()
            return inner
        if kind != "name":
            self.fail("expected an operand, got the end" if kind == "end" else f"expected an operand, got '{text}'",
                      col)
        if text in FUNCS:
            if self.peek()[0] != "(":
                self.fail(f"'{text}' is a function: expected '('", self.peek()[3])
            self.next()
            args = []
            while True:
                args.append(self.expr(depth + 1))
                t = self.peek()
                if t[0] == ",":
                    self.next()
                    continue
                if t[0] != ")":
                    self.fail(f"expected ',' or ')', got '{t[1]}'", t[3])
                self.next()
                break
            k = FUNCS[text]
            if len(args) != k:
                self.fail(f"'{text}' takes {k} argument{'' if k == 1 else 's'}, got {len(args)}", col)
            return Node(text, col, args=args)
        if self.peek()[0] == "(":
            self.fail(f"unknown function '{text}'", col)
        if text == "pi":
            return Node("num", col, value=PI)
        idx = self.n  # x is the first input, a the second, b the third, ...
        if text == "x":
            idx = 0
        elif len(text) == 1 and "a" <= text < chr(ord("a") + self.n - 1):
            idx = ord(text) - ord("a") + 1
        if idx >= self.n:
            names = ", ".join(["x"] + [chr(ord("a") + k - 1) for k in range(1, self.n)])
            self.fail(f"unknown name '{text}' (inputs: {names}; constant: pi)", col)
        return Node("in", col, input=idx)


def parse(text: str, n_inputs: int = 1) -> Node:
    if not 1 <= n_inputs <= MAX_INPUTS:
        raise ExprError(f"expr: {n_inputs} inputs (1 to {MAX_INPUTS})")
    return _Parser(text, n_inputs).parse()


# ---- evaluation ----

def _erf(v):
    if _erf_ufunc is not None:
        return F32(_erf_ufunc(np.float64(v))) if np.ndim(v) == 0 else _erf_ufunc(v.astype(np.float64)).astype(F32)
    if np.ndim(v) == 0:
        return F32(math.erf(float(v)))
    return np.vectorize(lambda t: math.erf(float(t)), otypes=[np.float64])(v).astype(F32)


def _sigmoid(v):
    one = F32(1)
    return one / (one + np.exp(-v))  # ops::sigmoid's host formula, each step in f32


UNARY = {
    "neg": np.negative, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "tanh": np.tanh, "abs": np.abs,
    "sigmoid": _sigmoid, "erf": _erf,
    "relu": lambda v: np.where(v > 0, v, F32(0)),  # max(0, v) as ops::relu (0 for v <= 0)
}
BINARY = {
    "+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide,
    "^": np.power,  # np.power, never ** (which turns x**2 into a square)
    "max": lambda a, b: np.where(a >= b, a, b), "min": lambda a, b: np.where(a <= b, a, b),
}


def _f32(v):
    """A scalar stays an np.float32; arrays are checked to be float32."""
    if np.ndim(v) == 0:
        return F32(v)
    assert v.dtype == F32, v.dtype
    return v


def _eval(n: Node, xs, T, w):
    k = n.kind
    if k == "num":
        return n.value
    if k == "in":
        return xs[n.input]
    if k in UNARY:
        return _f32(UNARY[k](_eval(n.args[0], xs, T, w)))
    if k in BINARY:
        return _f32(BINARY[k](_eval(n.args[0], xs, T, w), _eval(n.args[1], xs, T, w)))
    if k == "where":
        c, a, b = (_eval(m, xs, T, w) for m in n.args)
        return _f32(np.where(c > 0, a, b))
    full = np.broadcast_to(_eval(n.args[0], xs, T, w), (T, w))
    if k == "rowmax":  # the first largest entry, scanning ascending columns
        m = full[:, 0].copy()
        for j in range(1, w):
            m = np.where(full[:, j] > m, full[:, j], m)
        return m[:, None]
    s = np.zeros(T, F32)
    for j in range(w):  # ascending column order, one f32 add at a time
        s = s + full[:, j]
    s = s[:, None]
    return s if k == "rowsum" else _f32(s / F32(w))


def evaluate(expr, inputs) -> np.ndarray:
    """expr (a string or a parse() result) over inputs (one [T, w] array or a list of them) -> [T, w] float32."""
    if isinstance(inputs, np.ndarray):
        inputs = [inputs]
    xs = [np.asarray(v, dtype=F32) for v in inputs]
    if not xs or any(v.ndim != 2 or v.shape != xs[0].shape for v in xs):
        raise ExprError("IR_SPEC 5.19 (rule 37): expr: every input must have one shape [T, w]")
    node = parse(expr, len(xs)) if isinstance(expr, str) else expr
    T, w = xs[0].shape
    with np.errstate(all="ignore"):
        r = _eval(node, xs, T, w)
    return np.array(np.broadcast_to(r, (T, w)), dtype=F32)


if __name__ == "__main__":  # pragma: no cover
    import doctest
    doctest.testmod()
