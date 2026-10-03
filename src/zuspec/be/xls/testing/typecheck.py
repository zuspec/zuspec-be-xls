"""Type-check an :mod:`xir` package against XLS's per-op rules.

The rules are coded from ``ir_semantics.md`` (XLS docs, pinned release), one
function per op group, with the doc section named on each. They are kept
apart from :class:`xir.Builder`, which computes the same types on the emitter
side, so the two act as checks on each other.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Union

from ..xir import (ArrayType, BitsType, Function, Node, Package, Proc,
                   TOKEN_T, TupleType, Type)


class XlsCheckError(Exception):
    pass


def _err(n: Node, msg: str) -> XlsCheckError:
    return XlsCheckError(f"{n.ref} ({n.op}): {msg}")


def _need_bits(n: Node, x: Node) -> int:
    if not isinstance(x.type, BitsType):
        raise _err(n, f"operand {x.ref} must be bits, is {x.type}")
    return x.type.width


def _arity(n: Node, k: int) -> None:
    if len(n.operands) != k:
        raise _err(n, f"expects {k} operand(s), has {len(n.operands)}")


def _expect(n: Node, t: Type) -> None:
    if n.type != t:
        raise _err(n, f"result type should be {t}, is declared {n.type}")


_NARY = {"and", "or", "xor", "nand", "nor"}
_ARITH_SAME = {"add", "sub", "sdiv", "smod", "udiv", "umod"}
_MUL = {"umul", "smul"}
_EQ = {"eq", "ne"}
_CMP = {"ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"}
_SHIFT = {"shll", "shrl", "shra"}
_REDUCE = {"and_reduce", "or_reduce", "xor_reduce"}


def check_node(n: Node, pkg: Package, body: Union[Function, Proc]) -> None:
    op = n.op
    o = n.operands
    if op in ("not", "identity", "neg"):
        # "Unary bitwise operations", "Arithmetic unary operations"
        _arity(n, 1)
        if op != "identity":
            _need_bits(n, o[0])
        _expect(n, o[0].type)
    elif op in _NARY:
        # "Variadic bitwise operations": one or more identically-typed bits
        if not o:
            raise _err(n, "needs at least one operand")
        w = _need_bits(n, o[0])
        for x in o:
            if _need_bits(n, x) != w:
                raise _err(n, "operands must have the same width")
        _expect(n, BitsType(w))
    elif op in _REDUCE:
        # "Bitwise Reduction Operations"
        _arity(n, 1)
        _need_bits(n, o[0])
        _expect(n, BitsType(1))
    elif op in _ARITH_SAME:
        # "Arithmetic binary operations": same width except the multiplies
        _arity(n, 2)
        w = _need_bits(n, o[0])
        if _need_bits(n, o[1]) != w:
            raise _err(n, "operands must have the same width")
        _expect(n, BitsType(w))
    elif op in _MUL:
        _arity(n, 2)
        _need_bits(n, o[0]), _need_bits(n, o[1])
        if not isinstance(n.type, BitsType):
            raise _err(n, "result must be bits")
    elif op in _EQ:
        # "Equality comparison operations": identically-typed operands
        _arity(n, 2)
        if o[0].type != o[1].type:
            raise _err(n, f"operand types differ: {o[0].type} vs {o[1].type}")
        _expect(n, BitsType(1))
    elif op in _CMP:
        # "Numeric comparison operations": identically-typed bits operands
        _arity(n, 2)
        if _need_bits(n, o[0]) != _need_bits(n, o[1]):
            raise _err(n, "operands must have the same width")
        _expect(n, BitsType(1))
    elif op in _SHIFT:
        # "Shift operations": result = operand width; amount any width
        _arity(n, 2)
        w = _need_bits(n, o[0])
        _need_bits(n, o[1])
        _expect(n, BitsType(w))
    elif op in ("zero_ext", "sign_ext"):
        # "Extension operations": new_bit_count >= N
        _arity(n, 1)
        w = _need_bits(n, o[0])
        nb = n.attrs.get("new_bit_count")
        if not isinstance(nb, int) or nb < w:
            raise _err(n, f"new_bit_count={nb} must be >= {w}")
        _expect(n, BitsType(nb))
    elif op == "bit_slice":
        # "bit_slice": width(operand) >= start + width
        _arity(n, 1)
        w = _need_bits(n, o[0])
        s, wd = n.attrs.get("start"), n.attrs.get("width")
        if not isinstance(s, int) or not isinstance(wd, int) or s < 0 or wd < 0 or s + wd > w:
            raise _err(n, f"start={s} width={wd} does not fit bits[{w}]")
        _expect(n, BitsType(wd))
    elif op == "dynamic_bit_slice":
        # The width <= operand-width rule is not in ir_semantics.md; XLS's
        # verifier enforces it ("Expected operand 0 of dynamic_bit_slice to
        # have at least N bits").
        _arity(n, 2)
        _need_bits(n, o[1])
        if n.attrs["width"] > _need_bits(n, o[0]):
            raise _err(n, "width exceeds the operand width")
        _expect(n, BitsType(n.attrs["width"]))
    elif op == "bit_slice_update":
        # "bit_slice_update": result is the operand's type
        _arity(n, 3)
        w = _need_bits(n, o[0])
        _need_bits(n, o[1]), _need_bits(n, o[2])
        _expect(n, BitsType(w))
    elif op == "concat":
        # "concat": result is the sum of the operand widths
        _expect(n, BitsType(sum(_need_bits(n, x) for x in o)))
    elif op == "literal":
        _arity(n, 0)
        if "value" not in n.attrs:
            raise _err(n, "missing value")
    elif op == "sel":
        # "sel": default iff len(cases) < 2**width(selector)
        _arity(n, 1)
        sw = _need_bits(n, o[0])
        cases = n.attrs.get("cases") or []
        if not cases:
            raise _err(n, "needs at least one case")
        default = n.attrs.get("default")
        if len(cases) > (1 << sw):
            raise _err(n, "more cases than the selector can address")
        if (default is None) != (len(cases) == (1 << sw)):
            raise _err(n, "default must be given iff the cases do not cover the selector")
        for c in cases + ([default] if default is not None else []):
            if c.type != n.type:
                raise _err(n, f"case {c.ref} is {c.type}, result is {n.type}")
    elif op == "priority_sel":
        # "priority_sel": one selector bit per case, default required
        _arity(n, 1)
        cases = n.attrs.get("cases") or []
        if _need_bits(n, o[0]) != len(cases):
            raise _err(n, "selector width must equal the number of cases")
        default = n.attrs.get("default")
        if default is None:
            raise _err(n, "default is required")
        for c in cases + [default]:
            if c.type != n.type:
                raise _err(n, f"case {c.ref} is {c.type}, result is {n.type}")
    elif op == "tuple":
        _expect(n, TupleType(tuple(x.type for x in o)))
    elif op == "tuple_index":
        _arity(n, 1)
        t = o[0].type
        i = n.attrs.get("index")
        if not isinstance(t, TupleType) or not isinstance(i, int) or not 0 <= i < len(t.elems):
            raise _err(n, f"index={i} invalid for {t}")
        _expect(n, t.elems[i])
    elif op == "array":
        # "array": n-th operand is the n-th element
        if not o:
            raise _err(n, "empty arrays are not supported")
        for x in o:
            if x.type != o[0].type:
                raise _err(n, "elements must have the same type")
        _expect(n, ArrayType(o[0].type, len(o)))
    elif op in ("array_index", "array_update"):
        k = 1 if op == "array_index" else 2
        _arity(n, k)
        t = o[0].type
        for idx in n.attrs.get("indices", []):
            _need_bits(n, idx)
            if not isinstance(t, ArrayType):
                raise _err(n, "more indices than array dimensions")
            t = t.elem
        if op == "array_index":
            _expect(n, t)
        else:
            if o[1].type != t:
                raise _err(n, f"update value is {o[1].type}, element is {t}")
            _expect(n, o[0].type)
    elif op == "invoke":
        try:
            callee = pkg.function(n.attrs["to_apply"])
        except KeyError:
            raise _err(n, f"no function {n.attrs.get('to_apply')!r}")
        if [x.type for x in o] != [p.type for p in callee.params]:
            raise _err(n, f"argument types do not match {callee.name}")
        _expect(n, callee.return_type)
    elif op == "after_all":
        for x in o:
            if x.type != TOKEN_T:
                raise _err(n, f"operand {x.ref} is not a token")
        _expect(n, TOKEN_T)
    elif op in ("receive", "send"):
        if not isinstance(body, Proc):
            raise _err(n, "channel ops are only allowed in procs")
        try:
            ch = body.channel(n.attrs["channel"])
        except KeyError:
            raise _err(n, f"no channel {n.attrs.get('channel')!r}")
        if o[0].type != TOKEN_T:
            raise _err(n, "first operand must be a token")
        pred = n.attrs.get("predicate")
        if pred is not None and pred.type != BitsType(1):
            raise _err(n, "predicate must be bits[1]")
        if op == "receive":
            if ch.direction != "receive":
                raise _err(n, f"channel {ch.name} is not an input")
            _arity(n, 1)
            blocking = n.attrs.get("blocking", True)
            want = (TupleType((TOKEN_T, ch.type)) if blocking
                    else TupleType((TOKEN_T, ch.type, BitsType(1))))
            _expect(n, want)
        else:
            if ch.direction != "send":
                raise _err(n, f"channel {ch.name} is not an output")
            _arity(n, 2)
            if o[1].type != ch.type:
                raise _err(n, f"data is {o[1].type}, channel carries {ch.type}")
            _expect(n, TOKEN_T)
    elif op in ("state_read", "next_value"):
        if not isinstance(body, Proc):
            raise _err(n, "state ops are only allowed in procs")
        try:
            st = body.state_element(n.attrs["state_element"])
        except KeyError:
            raise _err(n, f"no state element {n.attrs.get('state_element')!r}")
        if op == "state_read":
            _expect(n, st.type)
        else:
            if n.attrs["value"].type != st.type:
                raise _err(n, f"value is {n.attrs['value'].type}, state is {st.type}")
            pred = n.attrs.get("predicate")
            if pred is not None and pred.type != BitsType(1):
                raise _err(n, "predicate must be bits[1]")
            _expect(n, TupleType(()))
    elif op == "assert":
        _arity(n, 2)
        if o[0].type != TOKEN_T or o[1].type != BitsType(1):
            raise _err(n, "operands must be (token, bits[1])")
        if not isinstance(n.attrs.get("message"), str):
            raise _err(n, "message is required")
        _expect(n, TOKEN_T)
    else:
        raise _err(n, f"op {op!r} is not in the checked subset")


def _check_body(pkg: Package, body: Union[Function, Proc], names: Dict[str, Node]) -> None:
    seen = set(id(p) for p in getattr(body, "params", []))
    for n in body.nodes:
        if n.ref in names:
            raise _err(n, "name is not unique")
        names[n.ref] = n
        for x in n.all_operands():
            if id(x) not in seen:
                raise _err(n, f"operand {x.ref} is not defined before use")
        check_node(n, pkg, body)
        seen.add(id(n))


def check_package(pkg: Package) -> None:
    """Raise :class:`XlsCheckError` on the first violation."""
    ids: Dict[int, Node] = {}
    for body in pkg.members():
        names: Dict[str, Node] = {}
        if isinstance(body, Function):
            for p in body.params:
                names[p.ref] = p
        _check_body(pkg, body, names)
        if isinstance(body, Function):
            if body.ret is None or body.ret not in body.nodes and body.ret not in body.params:
                raise XlsCheckError(f"fn {body.name}: no return value")
        else:
            reads = [n.attrs["state_element"] for n in body.nodes if n.op == "state_read"]
            for st in body.state:
                if reads.count(st.name) != 1:
                    # "state_read": every state element has a state_read
                    raise XlsCheckError(
                        f"proc {body.name}: state element {st.name} has {reads.count(st.name)} state_read nodes")
        for n in list(getattr(body, "params", [])) + body.nodes:
            if n.id in ids and ids[n.id] is not n:
                raise _err(n, f"id {n.id} is not unique in the package")
            ids[n.id] = n
    if pkg.top is not None:
        names = [m.name for m in pkg.members()]
        if pkg.top not in names:
            raise XlsCheckError(f"top {pkg.top!r} is not defined")
