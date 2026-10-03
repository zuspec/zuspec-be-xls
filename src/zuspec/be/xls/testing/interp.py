"""Interpret an :mod:`xir` package: the X0 stand-in for XLS's IR interpreter.

Op semantics are coded from ``ir_semantics.md`` (pinned release), section
named on each. Values follow :mod:`xir`: bits are unsigned ``int``s, tuples are
``tuple``s, arrays are ``list``s.

**Procs** run as Kahn processes over finite input streams
(:func:`run_proc`). One activation evaluates every node in list order, which
is a topological order and, for our emitter, program order for effects. An
activation whose blocking ``receive`` finds its stream empty is abandoned at
that point: effects already evaluated in it (sends before the receive) stand,
later ones do not happen, and state is not updated. That is what the SV
reference does when its input runs out mid-iteration.
"""
from __future__ import annotations

import dataclasses as dc
from typing import Dict, List, Optional, Sequence, Union

from ..xir import (ArrayType, BitsType, Function, Node, Package, Proc,
                   TOKEN, TupleType, Type, Value, zero_value)


class XlsAssertionError(Exception):
    def __init__(self, message: str, label: Optional[str] = None):
        super().__init__(message)
        self.label = label


class _Blocked(Exception):
    pass


def _mask(w: int) -> int:
    return (1 << w) - 1


def to_signed(v: int, w: int) -> int:
    v &= _mask(w)
    return v - (1 << w) if w and v >> (w - 1) else v


def _width(n: Node) -> int:
    assert isinstance(n.type, BitsType), n
    return n.type.width


def _div(op: str, a: int, b: int, w: int) -> int:
    """``ir_semantics.md`` "Arithmetic binary operations", notes ** and ***."""
    m = _mask(w)
    if op == "udiv":
        return m if b == 0 else a // b
    if op == "umod":
        return 0 if b == 0 else a % b
    sa, sb = to_signed(a, w), to_signed(b, w)
    if op == "sdiv":
        if sb == 0:
            # max positive if the dividend is non-negative, else max negative
            return (m >> 1) if sa >= 0 else (1 << (w - 1)) if w else 0
        q = abs(sa) // abs(sb)          # rounds toward zero
        q = q if (sa >= 0) == (sb >= 0) else -q
        return q & m
    if op == "smod":
        if sb == 0:
            return 0
        r = abs(sa) % abs(sb)           # sign follows the left operand
        return (r if sa >= 0 else -r) & m
    raise AssertionError(op)


class Interpreter:
    def __init__(self, pkg: Package):
        self.pkg = pkg

    # -- functions --------------------------------------------------------
    def call(self, fn: Union[str, Function], args: Sequence[Value]) -> Value:
        f = self.pkg.function(fn) if isinstance(fn, str) else fn
        if len(args) != len(f.params):
            raise ValueError(f"{f.name}: expects {len(f.params)} arguments")
        env: Dict[int, Value] = {}
        for p, a in zip(f.params, args):
            env[id(p)] = _norm(a, p.type)
        for n in f.nodes:
            env[id(n)] = self._eval(n, env, None)
        return env[id(f.ret)]

    # -- nodes --------------------------------------------------------------
    def _eval(self, n: Node, env: Dict[int, Value], act: Optional["_Activation"]) -> Value:
        op = n.op
        ops = [env[id(o)] for o in n.operands]
        a = n.attrs
        if op == "literal":
            return a["value"]
        if op == "identity":
            return ops[0]
        if op in ("not", "neg"):
            w = _width(n)
            return (~ops[0] if op == "not" else -ops[0]) & _mask(w)
        if op in ("and", "or", "xor", "nand", "nor"):
            # "Variadic bitwise operations"
            w = _width(n)
            r = ops[0]
            for x in ops[1:]:
                r = r & x if op in ("and", "nand") else r | x if op in ("or", "nor") else r ^ x
            if op in ("nand", "nor"):
                r = ~r
            return r & _mask(w)
        if op in ("and_reduce", "or_reduce", "xor_reduce"):
            w = _width(n.operands[0])
            x = ops[0]
            if op == "and_reduce":
                return int(x == _mask(w))
            if op == "or_reduce":
                return int(x != 0)
            return bin(x).count("1") & 1
        if op in ("add", "sub"):
            w = _width(n)
            return (ops[0] + ops[1] if op == "add" else ops[0] - ops[1]) & _mask(w)
        if op in ("umul", "smul"):
            # result = product truncated to the result width
            w = _width(n)
            if op == "umul":
                return (ops[0] * ops[1]) & _mask(w)
            wa, wb = _width(n.operands[0]), _width(n.operands[1])
            return (to_signed(ops[0], wa) * to_signed(ops[1], wb)) & _mask(w)
        if op in ("udiv", "sdiv", "umod", "smod"):
            return _div(op, ops[0], ops[1], _width(n))
        if op in ("eq", "ne"):
            r = _eq(ops[0], ops[1])
            return int(r if op == "eq" else not r)
        if op in ("ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"):
            x, y = ops
            if op[0] == "s":
                w = _width(n.operands[0])
                x, y = to_signed(x, w), to_signed(y, w)
            return int({"lt": x < y, "le": x <= y, "gt": x > y, "ge": x >= y}[op[1:]])
        if op in ("shll", "shrl", "shra"):
            # "Shift operations": over-shift gives 0, or all sign bits for shra
            w = _width(n)
            x, s = ops
            if op == "shll":
                return 0 if s >= w else (x << s) & _mask(w)
            if op == "shrl":
                return 0 if s >= w else x >> s
            sx = to_signed(x, w)
            return (sx >> min(s, w)) & _mask(w)
        if op == "zero_ext":
            return ops[0]
        if op == "sign_ext":
            w0 = _width(n.operands[0])
            return to_signed(ops[0], w0) & _mask(_width(n))
        if op == "bit_slice":
            return (ops[0] >> a["start"]) & _mask(a["width"])
        if op == "dynamic_bit_slice":
            # out-of-bounds bits read as 0
            w0 = _width(n.operands[0])
            s = ops[1]
            return 0 if s >= w0 else (ops[0] >> s) & _mask(a["width"])
        if op == "bit_slice_update":
            # bits at or above the operand width are ignored
            w = _width(n)
            x, s, v = ops
            if s >= w:
                return x
            vw = _width(n.operands[2])
            m = (_mask(vw) << s) & _mask(w)
            return (x & ~m | (v << s) & m) & _mask(w)
        if op == "concat":
            # operand 0 is the most significant
            r = 0
            for o, v in zip(n.operands, ops):
                r = (r << _width(o)) | v
            return r
        if op == "sel":
            cases = [env[id(c)] for c in a["cases"]]
            s = ops[0]
            if s < len(cases):
                return cases[s]
            return env[id(a["default"])]
        if op == "priority_sel":
            cases = [env[id(c)] for c in a["cases"]]
            s = ops[0]
            for i, c in enumerate(cases):
                if (s >> i) & 1:
                    return c
            return env[id(a["default"])]
        if op == "tuple":
            return tuple(ops)
        if op == "tuple_index":
            return ops[0][a["index"]]
        if op == "array":
            return list(ops)
        if op == "array_index":
            # out-of-bounds indices clamp to the last element
            v = ops[0]
            for idx in a.get("indices", []):
                i = env[id(idx)]
                v = v[min(i, len(v) - 1)]
            return v
        if op == "array_update":
            # any out-of-bounds index leaves the array unchanged
            idxs = [env[id(i)] for i in a.get("indices", [])]
            return _array_update(ops[0], ops[1], idxs)
        if op == "invoke":
            return self.call(a["to_apply"], ops)
        if op == "after_all":
            return TOKEN
        if op == "assert":
            if ops[1] == 0:
                raise XlsAssertionError(a["message"], a.get("label"))
            return TOKEN
        if act is None:
            raise ValueError(f"{op} is only valid in a proc")
        return act.effect(n, ops, env)


def _eq(x: Value, y: Value) -> bool:
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(_eq(a, b) for a, b in zip(x, y))
    return x == y


def _array_update(arr: list, v: Value, idxs: List[int]) -> list:
    if not idxs:
        return v
    i = idxs[0]
    if i >= len(arr):
        return arr
    out = list(arr)
    out[i] = _array_update(arr[i], v, idxs[1:])
    return out


def _norm(v: Value, t: Type) -> Value:
    if isinstance(t, BitsType):
        return int(v) & _mask(t.width)
    if isinstance(t, TupleType):
        return tuple(_norm(e, et) for e, et in zip(v, t.elems))
    if isinstance(t, ArrayType):
        return [_norm(e, t.elem) for e in v]
    return v


# ---------------------------------------------------------------------------
# Procs
# ---------------------------------------------------------------------------

@dc.dataclass
class ProcResult:
    outputs: Dict[str, List[Value]]
    state: Dict[str, Value]
    activations: int
    #: Why the run stopped: "blocked" (a receive ran out of input),
    #: "limit" (max_activations reached) or "assert".
    stop: str
    blocked_on: Optional[str] = None
    assertion: Optional[XlsAssertionError] = None


class _Activation:
    def __init__(self, proc: Proc, inputs: Dict[str, List[Value]],
                 cursor: Dict[str, int], state: Dict[str, Value],
                 outputs: Dict[str, List[Value]]):
        self.proc = proc
        self.inputs, self.cursor = inputs, cursor
        self.state = state
        self.outputs = outputs
        self.next: Dict[str, Value] = {}

    def effect(self, n: Node, ops: List[Value], env: Dict[int, Value]) -> Value:
        a = n.attrs
        pred = env[id(a["predicate"])] if "predicate" in a else 1
        if n.op == "state_read":
            return self.state[a["state_element"]]
        if n.op == "next_value":
            if pred:
                st = a["state_element"]
                if st in self.next:
                    raise ValueError(f"two next_value nodes fired for {st}")
                self.next[st] = env[id(a["value"])]
            return ()
        ch = self.proc.channel(a["channel"])
        if n.op == "receive":
            # "receive": a false predicate yields zero-filled data
            if not pred:
                return (TOKEN, zero_value(ch.type))
            q = self.inputs.get(ch.name, [])
            i = self.cursor.get(ch.name, 0)
            if i >= len(q):
                raise _Blocked(ch.name)
            self.cursor[ch.name] = i + 1
            return (TOKEN, _norm(q[i], ch.type))
        if n.op == "send":
            if pred:
                self.outputs.setdefault(ch.name, []).append(ops[1])
            return TOKEN
        raise ValueError(f"op {n.op!r} is not supported by the interpreter")


def run_proc(pkg: Package, proc: Union[str, Proc],
             inputs: Dict[str, Sequence[Value]],
             max_activations: int = 10000) -> ProcResult:
    p = pkg.proc(proc) if isinstance(proc, str) else proc
    interp = Interpreter(pkg)
    ins = {k: list(v) for k, v in inputs.items()}
    cursor: Dict[str, int] = {}
    state = {s.name: s.init for s in p.state}
    outputs: Dict[str, List[Value]] = {c.name: [] for c in p.channels if c.direction == "send"}
    count = 0
    while count < max_activations:
        act = _Activation(p, ins, cursor, state, outputs)
        env: Dict[int, Value] = {}
        try:
            for n in p.nodes:
                env[id(n)] = interp._eval(n, env, act)
        except _Blocked as b:
            return ProcResult(outputs, state, count, "blocked", blocked_on=str(b))
        except XlsAssertionError as e:
            return ProcResult(outputs, state, count, "assert", assertion=e)
        state = {**state, **act.next}
        count += 1
    return ProcResult(outputs, state, count, "limit")
