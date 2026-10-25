"""The IR's `recurrence` op (spec/IR_SPEC.md section 6.10, profile recurrence): a per-token state update
declared in the document, its bodies written in an indexed extension of the `expr` grammar.

    compile_node(node) -> Program          parse and check a recurrence node (RecurrenceError names the rule)
    errors(node)        -> [message]       the same checks as messages (tools/ir_validate.py; stdlib only)
    run(prog, ins, weights, state=None, dtype=np.float32) -> (y [T, w], state {name: array})
    fd_grads(prog, ins, weights, out_grad, h=1e-5) -> {name: gradient}   central differences in float64
    rewrite(doc) -> doc                    every linear_attention, ssm_scan, conv1d_causal and selective_scan
                                           node replaced by its recurrence (section 6.10.5)

A node:
    {"op": "recurrence", "in": ["q", "k", "v"], "state": "lin",
     "index": {"h": 2, "e": 4},
     "inputs": ["q[h,e]", "k[h,e]", "v[h,e]"],          one declaration per value of `in`
     "weights": {"A[r,j]": "blk.{i}.ssm_a"},            optional: declaration -> tensor name
     "init": ["s[r] = h0[r]"],                          optional: a state tensor's initial value
     "let": ["pk[h,e] = ..."],                          optional, in order
     "update": ["num[h,e] = num[h,e] + pk[h,e] * v[h,e]", ...],   every state tensor, once
     "readout": "y[h,e] = ...",                         the output row
     "out": "a"}
Parsing and checking use the standard library only; run() and fd_grads() need numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

SECTION = "6.10"
FUNCS = {"exp": 1, "log": 1, "sqrt": 1, "tanh": 1, "sigmoid": 1, "erf": 1, "abs": 1, "relu": 1,
         "max": 2, "min": 2, "where": 3}
LIBM = {"exp", "log", "tanh", "erf", "sigmoid", "^"}  # results that differ by an ulp between libms
RESERVED = set(FUNCS) | {"pi", "sum", "rowsum", "rowmean", "rowmax", "per_layer"}  # per_layer: 3.8
MAX_DEPTH = 200  # as expr (section 5.19)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_NUM = re.compile(r"(\d+\.?\d*|\.\d+)([eE][+-]?\d*)?")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class RecurrenceError(ValueError):
    def __init__(self, rule, msg):
        super().__init__(f"IR_SPEC {SECTION} (rule {rule}): {msg}")
        self.rule = rule


@dataclass
class Node:
    kind: str          # num pi read index neg + - * / ^ <function> sum
    col: int
    text: str = ""     # num: the literal; read / index / sum: the name
    ixs: list = field(default_factory=list)   # read: [(index name or None, integer offset or constant)]
    args: list = field(default_factory=list)


# ---------------------------------------------------------------- parsing

class _Parser:
    def __init__(self, text, where):
        self.s, self.where = text, where
        self.toks = self._lex()
        self.pos = 0

    def fail(self, what, col):
        raise RecurrenceError(31, f'{self.where}: {what} at column {col} in "{self.s}"')

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
                try:
                    if not math.isfinite(float(lit)) or abs(float(lit)) > 3.4028235677973366e38:
                        self.fail(f"number {lit} is out of float range", col)
                except OverflowError:
                    self.fail(f"number {lit} is out of float range", col)
                toks.append(("num", lit, col))
                i = m.end()
            elif ch.isascii() and (ch.isalpha() or ch == "_"):
                m = _NAME.match(s, i)
                toks.append(("name", m.group(0), col))
                i = m.end()
            elif ch in "+-*/^=":
                toks.append(("op", ch, col))
                i += 1
            elif ch in "(),[]":
                toks.append((ch, ch, col))
                i += 1
            else:
                self.fail(f"unexpected character '{ch}'", col)
        toks.append(("end", "end", len(s) + 1))
        return toks

    def peek(self):
        return self.toks[self.pos]

    def next(self):
        t = self.toks[self.pos]
        self.pos += 1
        return t

    def at(self, kind, text=None):
        t = self.peek()
        return t[0] == kind and (text is None or t[1] == text)

    def expect(self, kind, what):
        t = self.next()
        if t[0] != kind:
            self.fail(f"expected {what}, got '{t[1]}'", t[2])
        return t

    def enter(self, depth):
        if depth > MAX_DEPTH:
            self.fail(f"nesting deeper than {MAX_DEPTH}", self.peek()[2])

    def decl(self):
        """name '[' index (',' index)* ']'"""
        t = self.expect("name", "a name")
        self.expect("[", "'['")
        idx = [self.expect("name", "an index name")[1]]
        while self.at(","):
            self.next()
            idx.append(self.expect("name", "an index name")[1])
        self.expect("]", "']'")
        return t[1], idx, t[2]

    def end(self):
        t = self.peek()
        if t[0] != "end":
            self.fail(f"expected an operator or the end, got '{t[1]}'", t[2])

    def expr(self, depth):
        self.enter(depth)
        left = self.term(depth + 1)
        while self.at("op", "+") or self.at("op", "-"):
            op = self.next()
            left = Node(op[1], op[2], args=[left, self.term(depth + 1)])
        return left

    def term(self, depth):
        self.enter(depth)
        left = self.unary(depth + 1)
        while self.at("op", "*") or self.at("op", "/"):
            op = self.next()
            left = Node(op[1], op[2], args=[left, self.unary(depth + 1)])
        return left

    def unary(self, depth):
        self.enter(depth)
        if self.at("op", "-"):
            op = self.next()
            return Node("neg", op[2], args=[self.unary(depth + 1)])
        return self.power(depth + 1)

    def power(self, depth):
        self.enter(depth)
        base = self.primary(depth + 1)
        if self.at("op", "^"):
            op = self.next()
            return Node("^", op[2], args=[base, self.unary(depth + 1)])
        return base

    def ix(self):
        t = self.next()
        if t[0] == "num":
            if not t[1].isdigit():
                self.fail(f"an index constant is an integer, not {t[1]}", t[2])
            return (None, int(t[1]))
        if t[0] != "name":
            self.fail(f"expected an index, got '{t[1]}'", t[2])
        off = 0
        if self.at("op", "+") or self.at("op", "-"):
            sign = 1 if self.next()[1] == "+" else -1
            c = self.next()
            if c[0] != "num" or not c[1].isdigit():
                self.fail("an index offset is an integer", c[2])
            off = sign * int(c[1])
        return (t[1], off)

    def primary(self, depth):
        self.enter(depth)
        kind, text, col = self.next()
        if kind == "num":
            return Node("num", col, text=text)
        if kind == "(":
            inner = self.expr(depth + 1)
            self.expect(")", "')'")
            return inner
        if kind != "name":
            self.fail("expected an operand, got the end" if kind == "end" else f"expected an operand, got '{text}'",
                      col)
        if text in ("rowsum", "rowmean", "rowmax"):
            self.fail(f"'{text}' is not defined in a recurrence body (write sum(index, ...))", col)
        if text == "sum":
            self.expect("(", "'('")
            j = self.expect("name", "the summed index")
            self.expect(",", "','")
            body = self.expr(depth + 1)
            self.expect(")", "')'")
            return Node("sum", col, text=j[1], args=[body])
        if text in FUNCS:
            if not self.at("("):
                self.fail(f"'{text}' is a function: expected '('", self.peek()[2])
            self.next()
            args = [self.expr(depth + 1)]
            while self.at(","):
                self.next()
                args.append(self.expr(depth + 1))
            self.expect(")", "',' or ')'")
            if len(args) != FUNCS[text]:
                k = FUNCS[text]
                self.fail(f"'{text}' takes {k} argument{'' if k == 1 else 's'}, got {len(args)}", col)
            return Node(text, col, args=args)
        if text == "pi":
            return Node("pi", col)
        if self.at("("):
            self.fail(f"unknown function '{text}'", col)
        if self.at("["):
            self.next()
            ixs = [self.ix()]
            while self.at(","):
                self.next()
                ixs.append(self.ix())
            self.expect("]", "']'")
            return Node("read", col, text=text, ixs=ixs)
        return Node("index", col, text=text)


def parse_decl(text, where):
    p = _Parser(text, where)
    name, idx, _ = p.decl()
    p.end()
    return name, idx


def parse_assign(text, where):
    p = _Parser(text, where)
    name, idx, _ = p.decl()
    p.expect("op", "'='")
    e = p.expr(0)
    p.end()
    return name, idx, e


# ---------------------------------------------------------------- checking

@dataclass
class Def:
    name: str
    index: list
    expr: Node
    where: str


@dataclass
class Program:
    index: dict                     # index name -> size
    inputs: list                    # [(name, [index names])], one per value of `in`
    weights: list                   # [(name, [index names], tensor name)]
    init: list                      # [Def]
    let: list                       # [Def]
    update: list                    # [Def], one per state tensor, in order
    readout: Def
    shapes: dict                    # every tensor name -> [sizes]
    exact: bool                     # no libm function and no '^': EXACT, else ROUNDING

    def numel(self, name):
        n = 1
        for s in self.shapes[name]:
            n *= s
        return n

    @property
    def state(self):
        return [d.name for d in self.update]

    @property
    def out_width(self):
        return self.numel(self.readout.name)


def _uses(e, kinds):
    return e.kind in kinds or any(_uses(a, kinds) for a in e.args)


def _check_expr(e, scope, readable, shapes, index, where):
    if e.kind == "read":
        if e.text not in readable:
            raise RecurrenceError(32, f"{where}: reads {e.text}, which this body cannot read "
                                      f"(it reads {', '.join(sorted(readable)) or 'nothing'})")
        rank = len(shapes[e.text])
        if len(e.ixs) != rank:
            raise RecurrenceError(32, f"{where}: {e.text} has rank {rank}, read with {len(e.ixs)} indices")
        for name, _ in e.ixs:
            if name is not None and name not in scope:
                raise RecurrenceError(32, f"{where}: index {name} is not in scope (in scope: {', '.join(scope)})")
    elif e.kind == "index":
        if e.text not in scope:
            what = "a tensor (write it with its indices)" if e.text in shapes else "not an index in scope"
            raise RecurrenceError(32, f"{where}: {e.text} is {what}")
    elif e.kind == "sum":
        if e.text not in index:
            raise RecurrenceError(32, f"{where}: sum over {e.text}, which is not a declared index")
        if e.text in scope:
            raise RecurrenceError(32, f"{where}: sum over {e.text}, which is already in scope")
        _check_expr(e.args[0], scope + [e.text], readable, shapes, index, where)
        return
    for a in e.args:
        _check_expr(a, scope, readable, shapes, index, where)


def compile_node(nd):
    """The checked Program of a recurrence node (rules 31-33 of section 10.1); raises RecurrenceError."""
    ins = nd.get("in") if isinstance(nd.get("in"), list) else []
    index = nd.get("index")
    if not isinstance(index, dict) or not index:
        raise RecurrenceError(31, "index must map index names to sizes")
    names = {}

    def declare(name, kind):
        if not isinstance(name, str) or not _IDENT.match(name) or name in RESERVED:
            raise RecurrenceError(31, f"{name!r} is not a name (an identifier, not a function, pi or sum)")
        if name in names:
            raise RecurrenceError(31, f"{name} is declared twice ({names[name]} and {kind})")
        names[name] = kind

    for k, v in index.items():
        declare(k, "index")
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise RecurrenceError(31, f"index {k}: size {v!r} is not a positive integer")
    shapes = {}

    def sizes(idx, where):
        if len(set(idx)) != len(idx):
            raise RecurrenceError(32, f"{where}: an index repeats in {idx}")
        for i in idx:
            if i not in index:
                raise RecurrenceError(32, f"{where}: {i} is not a declared index")
        return [index[i] for i in idx]

    decl_in = nd.get("inputs")
    if not isinstance(decl_in, list) or len(decl_in) != len(ins) or not ins:
        raise RecurrenceError(31, f"inputs must declare each of the {len(ins)} values of in (at least one)")
    inputs = []
    for j, t in enumerate(decl_in):
        if not isinstance(t, str):
            raise RecurrenceError(31, f"inputs[{j}] is not a declaration string")
        name, idx = parse_decl(t, f"inputs[{j}]")
        declare(name, "input")
        shapes[name] = sizes(idx, f"inputs[{j}]")
        inputs.append((name, idx))
    weights = []
    w = nd.get("weights", {})
    if not isinstance(w, dict):
        raise RecurrenceError(31, "weights must map declarations to tensor names")
    for t, tensor in w.items():
        name, idx = parse_decl(t, f"weights {t!r}")
        declare(name, "weight")
        if not isinstance(tensor, str) or not tensor:
            raise RecurrenceError(31, f"weight {name}: the tensor name must be a string")
        shapes[name] = sizes(idx, f"weights {t!r}")
        weights.append((name, idx, tensor))

    def defs(key, required=False):
        v = nd.get(key, [] if not required else None)
        if key == "readout":
            return v
        if not isinstance(v, list) or (required and not v):
            raise RecurrenceError(31, f"{key} must be a list of assignments" + (" (at least one)" if required else ""))
        out = []
        for j, t in enumerate(v):
            if not isinstance(t, str):
                raise RecurrenceError(31, f"{key}[{j}] is not an assignment string")
            name, idx, e = parse_assign(t, f"{key}[{j}]")
            out.append(Def(name, idx, e, f"{key}[{j}]"))
        return out

    update = defs("update", required=True)
    for d in update:
        declare(d.name, "state")
        shapes[d.name] = sizes(d.index, d.where)
    lets = defs("let")
    init = defs("init")
    ro = nd.get("readout")
    if not isinstance(ro, str):
        raise RecurrenceError(31, "readout must be one assignment string")
    rname, ridx, rexpr = parse_assign(ro, "readout")
    readout = Def(rname, ridx, rexpr, "readout")
    inputs_n = {n for n, _ in inputs}
    weights_n = {n for n, _, _ in weights}
    state_n = {d.name for d in update}
    seen_init = set()
    for d in init:
        if d.name not in state_n:
            raise RecurrenceError(33, f"{d.where}: init assigns {d.name}, which is not a state tensor")
        if d.name in seen_init:
            raise RecurrenceError(33, f"{d.where}: {d.name} is initialised twice")
        seen_init.add(d.name)
        if sizes(d.index, d.where) != shapes[d.name]:
            raise RecurrenceError(33, f"{d.where}: {d.name}'s sizes differ from its update's")
        _check_expr(d.expr, list(d.index), weights_n, shapes, index, d.where)
    let_n = set()
    for d in lets:
        _check_expr(d.expr, list(d.index), inputs_n | weights_n | state_n | let_n, shapes, index, d.where)
        declare(d.name, "let")
        shapes[d.name] = sizes(d.index, d.where)
        let_n.add(d.name)
    for d in update:
        _check_expr(d.expr, list(d.index), inputs_n | weights_n | state_n | let_n, shapes, index, d.where)
    _check_expr(rexpr, list(ridx), inputs_n | weights_n | state_n | let_n, shapes, index, "readout")
    declare(rname, "readout")
    shapes[rname] = sizes(ridx, "readout")
    exact = not any(_uses(d.expr, LIBM) for d in init + lets + update + [readout])
    return Program(index=dict(index), inputs=inputs, weights=weights, init=init, let=lets, update=update,
                   readout=readout, shapes=shapes, exact=exact)


def errors(nd, slot=None):
    """Rules 31-33 as messages: the node, and with its slot declaration, the slot's fit (rule 33)."""
    try:
        prog = compile_node(nd)
    except RecurrenceError as e:
        return [str(e)]
    out = []
    if slot is not None:
        ts = {t.get("name"): t for t in slot.get("tensors", []) if isinstance(t, dict)}
        want = {d.name: prog.shapes[d.name] for d in prog.update}
        if set(ts) != set(want):
            out.append(str(RecurrenceError(33, f"recurrent state {slot.get('name')}: tensors {sorted(ts)} are not the "
                                               f"update's state tensors {sorted(want)}")))
        inits = {d.name for d in prog.init}
        for n, t in ts.items():
            if n in want and t.get("shape") != want[n]:
                out.append(str(RecurrenceError(33, f"recurrent state {slot.get('name')}: {n} has shape "
                                                   f"{t.get('shape')}, the update gives {want[n]}")))
            if n in want and (t.get("init") == "body") != (n in inits):
                out.append(str(RecurrenceError(33, f"recurrent state {slot.get('name')}: {n}'s init is "
                                                   f"{t.get('init')!r}, but the node "
                                                   f"{'initialises' if n in inits else 'does not initialise'} it")))
    return out


# ---------------------------------------------------------------- evaluation (numpy)

def _np():
    import numpy as np
    return np


def _literal(text, dt):
    """The nearest float to the literal (strtof), as dt."""
    import ir_expr
    return dt(ir_expr._nearest_f32(text))


def _fn(name, dt):
    np = _np()
    one = dt(1)
    if name == "erf":
        try:
            from scipy.special import erf as _e
            return lambda v: _e(np.asarray(v, np.float64)).astype(dt)
        except ImportError:
            return lambda v: np.vectorize(lambda t: math.erf(float(t)), otypes=[np.float64])(v).astype(dt)
    return {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "tanh": np.tanh, "abs": np.abs,
            "sigmoid": lambda v: one / (one + np.exp(-v)),
            "relu": lambda v: np.where(v > 0, v, dt(0))}[name]


class _Eval:
    def __init__(self, prog, dt):
        self.p, self.dt = prog, dt

    def shape(self, scope):
        return tuple(self.p.index[i] for i in scope)

    def ev(self, e, scope, env):
        np, dt, k = _np(), self.dt, e.kind
        if k == "num":
            return _literal(e.text, dt)
        if k == "pi":
            return dt(np.float32(math.pi))
        if k == "index":
            a = scope.index(e.text)
            sh = [1] * len(scope)
            sh[a] = self.p.index[e.text]
            return np.arange(self.p.index[e.text], dtype=dt).reshape(sh)
        if k == "read":
            X = env[e.text]
            coords = []
            for name, off in e.ixs:
                if name is None:
                    coords.append(np.array(off).reshape([1] * len(scope)))
                else:
                    a = scope.index(name)
                    sh = [1] * len(scope)
                    sh[a] = self.p.index[name]
                    coords.append((np.arange(self.p.index[name]) + off).reshape(sh))
            coords = np.broadcast_arrays(*coords) if coords else []
            mask = np.ones(coords[0].shape, bool)
            idx = []
            for c, n in zip(coords, X.shape):
                mask &= (c >= 0) & (c < n)
                idx.append(np.clip(c, 0, n - 1))
            return np.where(mask, X[tuple(idx)], dt(0)).astype(dt)  # outside the tensor: +0
        if k == "sum":
            j = e.text
            body = np.broadcast_to(self.ev(e.args[0], scope + [j], env), self.shape(scope + [j]))
            acc = np.zeros(self.shape(scope), dt)
            for jj in range(self.p.index[j]):  # ascending, from +0, one rounded add at a time
                acc = (acc + body[..., jj]).astype(dt)
            return acc
        a = [self.ev(x, scope, env) for x in e.args]
        if k == "neg":
            return (-a[0]).astype(dt) if hasattr(a[0], "astype") else dt(-a[0])
        if k == "+":
            r = a[0] + a[1]
        elif k == "-":
            r = a[0] - a[1]
        elif k == "*":
            r = a[0] * a[1]
        elif k == "/":
            r = a[0] / a[1]
        elif k == "^":
            r = np.power(a[0], a[1])
        elif k == "max":
            r = np.where(a[0] >= a[1], a[0], a[1])
        elif k == "min":
            r = np.where(a[0] <= a[1], a[0], a[1])
        elif k == "where":
            r = np.where(a[0] > 0, a[1], a[2])
        else:
            r = _fn(k, dt)(a[0])
        return np.asarray(r).astype(dt) if np.ndim(r) else dt(r)

    def define(self, d, env):
        np = _np()
        v = self.ev(d.expr, list(d.index), env)
        return np.array(np.broadcast_to(v, self.shape(d.index)), dtype=self.dt)


def initial_state(prog, weights, dt=None):
    """{state tensor: array}: the init bodies' values, +0 elsewhere."""
    np = _np()
    dt = dt or np.float32
    ev = _Eval(prog, dt)
    env = {n: np.asarray(weights[n], dt).reshape(prog.shapes[n]) for n, _, _ in prog.weights}
    st = {d.name: np.zeros(prog.shapes[d.name], dt) for d in prog.update}
    with np.errstate(all="ignore"):
        for d in prog.init:
            st[d.name] = ev.define(d, env)
    return st


def run(prog, ins, weights, state=None, dtype=None):
    """The recurrence over T rows from `state` (default: the initial state): y [T, out width] and the state
    after the last row. ins: one [T, w] array per input; weights: {weight name: array} (any shape with the
    declared element count)."""
    np = _np()
    dt = dtype or np.float32
    ev = _Eval(prog, dt)
    wenv = {}
    for n, _, _ in prog.weights:
        a = np.asarray(weights[n], dt)
        if a.size != prog.numel(n):
            raise RecurrenceError(34, f"weight {n} has {a.size} elements, its declaration {prog.numel(n)}")
        wenv[n] = a.reshape(prog.shapes[n])
    xs = [np.asarray(x, dt) for x in ins]
    T = xs[0].shape[0]
    for (n, _), x in zip(prog.inputs, xs):
        if x.ndim != 2 or x.shape[0] != T or x.shape[1] != prog.numel(n):
            raise RecurrenceError(34, f"input {n} is {list(x.shape)}, its declaration [T, {prog.numel(n)}]")
    st = initial_state(prog, weights, dt) if state is None else {k: np.asarray(v, dt).reshape(prog.shapes[k])
                                                                  for k, v in state.items()}
    y = np.zeros((T, prog.out_width), dt)
    with np.errstate(all="ignore"):
        for t in range(T):
            env = dict(wenv)
            env.update(st)
            for (n, _), x in zip(prog.inputs, xs):
                env[n] = x[t].reshape(prog.shapes[n])
            for d in prog.let:
                env[d.name] = ev.define(d, env)
            new = {d.name: ev.define(d, env) for d in prog.update}  # every update reads the state before
            env.update(new)
            st = new
            y[t] = ev.define(prog.readout, env).reshape(-1)
    return y, st


def fd_grads(prog, ins, weights, out_grad, h=1e-5):
    """d sum(y * out_grad) / d(each input element, each weight element), by central differences of the
    recurrence evaluated in float64 (its literals still the nearest floats)."""
    np = _np()
    g = np.asarray(out_grad, np.float64)
    base_in = [np.asarray(x, np.float64) for x in ins]
    base_w = {k: np.asarray(v, np.float64) for k, v in weights.items()}

    def f(xs, ws):
        return float(np.sum(run(prog, xs, ws, dtype=np.float64)[0] * g))

    out = {}
    for j, (name, _) in enumerate(prog.inputs):
        d = np.zeros_like(base_in[j])
        for pos in np.ndindex(*d.shape):
            step = h * max(1.0, abs(base_in[j][pos]))
            hi = [x.copy() for x in base_in]
            lo = [x.copy() for x in base_in]
            hi[j][pos] += step
            lo[j][pos] -= step
            d[pos] = (f(hi, base_w) - f(lo, base_w)) / (2 * step)
        out[name] = d
    for name, _, _ in prog.weights:
        d = np.zeros_like(base_w[name])
        for pos in np.ndindex(*d.shape):
            step = h * max(1.0, abs(base_w[name][pos]))
            hi, lo = dict(base_w), dict(base_w)
            hi[name] = base_w[name].copy()
            lo[name] = base_w[name].copy()
            hi[name][pos] += step
            lo[name][pos] -= step
            d[pos] = (f(base_in, hi) - f(base_in, lo)) / (2 * step)
        out[name] = d
    return out


# ---------------------------------------------------------------- the four hand-written ops as recurrences

def _num(v):
    """A float attribute as a body literal (the shortest decimal of its double: strtof reads the same float)."""
    s = repr(float(v))
    return s[:-2] if s.endswith(".0") else s


def _plain(nd, keys):
    for k in keys:
        if isinstance(nd.get(k), dict):
            raise ValueError(f"{nd['op']}: per-layer {k} has no single recurrence")


def _phi(x):
    """elu + 1 (section 6.4), the same operations: x + 1 for x > 0, else (exp(x) - 1) + 1; min(0, x) keeps exp
    from overflowing in the branch where() does not take (and is x itself wherever that branch is taken)."""
    return f"where({x}, {x} + 1, (exp(min(0, {x})) - 1) + 1)"


def linear_attention(nd):
    _plain(nd, ("heads", "head_dim", "den_floor"))
    H, E = nd["heads"], nd["head_dim"]
    return {"op": "recurrence", "in": list(nd["in"]), "state": nd["state"], "index": {"h": H, "e": E},
            "inputs": ["q[h,e]", "k[h,e]", "v[h,e]"],
            "let": [f"pk[h,e] = {_phi('k[h,e]')}"],
            "update": ["num[h,e] = num[h,e] + pk[h,e] * v[h,e]", "den[h,e] = den[h,e] + pk[h,e]"],
            "readout": f"y[h,e] = ({_phi('q[h,e]')} * num[h,e]) / max(den[h,e], {_num(nd['den_floor'])})",
            "out": nd["out"]}


def ssm_scan(nd):
    _plain(nd, ("n",))
    n = nd["n"]
    r = {"op": "recurrence", "in": list(nd["in"]), "state": nd["state"], "index": {"r": n, "j": n, "o": 1},
         "inputs": ["u[r]"],
         "weights": {"A[r,j]": nd["a"], "B[r]": nd["b"], "C[r]": nd["c"], "D[o]": nd["d"]},
         "update": ["s[r] = sum(j, A[r,j] * s[j]) + B[r] * u[r]"],
         "readout": "y[r] = C[r] * s[r] + D[0] * u[r]", "out": nd["out"]}
    if "tensor_index" in nd:
        r["tensor_index"] = nd["tensor_index"]
    return r


def conv1d_causal(nd):
    _plain(nd, ("channels", "kernel"))
    C, K = nd["channels"], nd["kernel"]
    w = {"W[c,k]": nd["weight"]}
    if nd.get("bias"):
        w["Bias[c]"] = nd["bias"]
    r = {"op": "recurrence", "in": list(nd["in"]), "state": nd["state"], "index": {"c": C, "k": K, "r": K - 1},
         "inputs": ["u[c]"], "weights": w,
         # xx: the K - 1 earlier inputs (the state, oldest first), then this token's
         "let": [f"xx[k,c] = where(k - {_num(K - 1.5)}, u[c], x[k,c])"],
         "update": ["x[r,c] = xx[r + 1,c]"],
         "readout": "y[c] = sum(k, W[c,k] * xx[k,c])" + (" + Bias[c]" if nd.get("bias") else ""),
         "out": nd["out"]}
    if "tensor_index" in nd:
        r["tensor_index"] = nd["tensor_index"]
    return r


def selective_scan(nd):
    _plain(nd, ("channels", "d_state", "dt_softplus"))
    D, N = nd["channels"], nd["d_state"]
    w = {"A[c,n]": nd["a"]}
    if nd.get("d"):
        w["Dk[c]"] = nd["d"]
    lets = []
    if nd.get("dt_softplus", True):  # softplus with llama.cpp's threshold 20; log(1 + e) for log1p (ROUNDING)
        lets.append("delta[c] = where(dt[c] - 20, dt[c], log(1 + exp(min(20, dt[c]))))")
        dl = "delta[c]"
    else:
        dl = "dt[c]"
    lets.append(f"xd[c] = x[c] * {dl}")
    r = {"op": "recurrence", "in": list(nd["in"]), "state": nd["state"], "index": {"c": D, "n": N},
         "inputs": ["x[c]", "dt[c]", "Bm[n]", "Cm[n]"], "weights": w, "let": lets,
         "update": [f"s[c,n] = s[c,n] * exp({dl} * A[c,n]) + Bm[n] * xd[c]"],
         "readout": "y[c] = sum(n, s[c,n] * Cm[n])" + (" + Dk[c] * x[c]" if nd.get("d") else ""),
         "out": nd["out"]}
    if "tensor_index" in nd:
        r["tensor_index"] = nd["tensor_index"]
    return r


REWRITES = {"linear_attention": linear_attention, "ssm_scan": ssm_scan, "conv1d_causal": conv1d_causal,
            "selective_scan": selective_scan}


def slot_for(node, name):
    """The recurrent slot a recurrence node reads: its state tensors, init 0 or "body"."""
    prog = compile_node(node)
    inits = {d.name for d in prog.init}
    return {"name": name, "kind": "recurrent", "op": "recurrence", "dtype": "f32",
            "tensors": [{"name": d.name, "shape": prog.shapes[d.name], "init": "body" if d.name in inits else 0}
                        for d in prog.update]}


def rewrite(doc):
    """The document with every hand-written recurrent op replaced by its recurrence (section 6.10.5) and each
    of their slots by the recurrence's slot (same name, same tensors)."""
    import copy
    out = copy.deepcopy(doc)
    blk = out["block"]
    slots = {s["name"]: j for j, s in enumerate(blk.get("state", []))}
    for j, nd in enumerate(blk["nodes"]):
        f = REWRITES.get(nd.get("op"))
        if f is None:
            continue
        new = f(nd)
        blk["nodes"][j] = new
        blk["state"][slots[nd["state"]]] = slot_for(new, nd["state"])
    if out.get("family_hint") and not out["family_hint"].endswith("_recurrence"):
        out["family_hint"] = out["family_hint"] + "_recurrence"
    return out
