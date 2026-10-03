"""Static-subset typing: the width and signedness of every IR expression,
computed bottom-up (``xls-phase0.md`` R1, decision D-7).

**The contract the front end meets.** SV widths depend on context; the front
end resolves them and writes the result into the IR as explicit casts. After
that, every expression's type follows from its operands alone:

* a variable has its declared type;
* ``ExprCast(DataTypeInt(w, s), x)`` is ``(w, s)``. It truncates, or extends
  by **x's** signedness (sign-extends a signed ``x``), as SV does;
* a bare integer constant is unsigned, as wide as its value needs (at least
  1 bit). A ``bool`` constant is 1 bit. Anything wider or signed comes wrapped
  in a cast;
* binary arithmetic and bitwise ops, ``?:`` arms and comparisons need equal
  operand widths. The result is signed only if both operands are (SV 11.8.1);
* a shift has its left operand's type; the amount is any width, read
  unsigned;
* comparisons, logical ops, reductions and bit selects are 1-bit unsigned;
* concatenation and replication are unsigned, as wide as their parts.

:func:`infer` is the rule for one node given its operands' types. The lowering
calls it on every expression it lowers, so a violation stops the lowering with
the expression's source location.
"""
from __future__ import annotations

import dataclasses as dc
from typing import Callable, Optional, Union

import zuspec.ir.core as ir


class StaticSubsetError(Exception):
    """An IR construct is outside the static subset, or ill-typed in it."""

    def __init__(self, msg: str, loc: Optional[ir.Loc] = None):
        self.msg = msg
        self.loc = loc
        where = ""
        if loc is not None:
            where = f"{loc.file or '<unknown>'}:{loc.line}:{loc.pos}: "
        super().__init__(where + msg)


@dc.dataclass(frozen=True)
class IntT:
    width: int
    signed: bool = False

    def __str__(self) -> str:
        return f"{'s' if self.signed else 'u'}{self.width}"


@dc.dataclass(frozen=True)
class ArrT:
    elem: "T"
    size: int

    def __str__(self) -> str:
        return f"{self.elem}[{self.size}]"


T = Union[IntT, ArrT]

BIT = IntT(1, False)


def from_ir(dt: ir.DataType, loc: Optional[ir.Loc] = None) -> T:
    """The static-subset type of an IR data type."""
    if isinstance(dt, ir.DataTypeEnum):
        width = dt.width or max(1, max((v for v in dt.items.values()), default=0).bit_length())
        return IntT(width, False)
    if isinstance(dt, ir.DataTypeInt):
        if dt.bits <= 0:
            raise StaticSubsetError(f"integer type without a width: {dt}", loc)
        return IntT(dt.bits, bool(dt.signed))
    if isinstance(dt, ir.DataTypeArray):
        if dt.size is None or dt.size <= 0:
            raise StaticSubsetError("array without a fixed size", loc)
        return ArrT(from_ir(dt.element_type, loc), dt.size)
    raise StaticSubsetError(f"type {type(dt).__name__} is not in the static subset", loc)


def const_type(v: object, loc: Optional[ir.Loc] = None) -> IntT:
    if isinstance(v, bool):
        return BIT
    if isinstance(v, int):
        if v < 0:
            raise StaticSubsetError(
                f"negative constant {v} needs an explicit signed cast", loc)
        return IntT(max(1, v.bit_length()), False)
    raise StaticSubsetError(f"constant {v!r} is not an integer", loc)


def _int(t: T, what: str, loc) -> IntT:
    if not isinstance(t, IntT):
        raise StaticSubsetError(f"{what} must be an integer, is an array ({t})", loc)
    return t


def _same(a: IntT, b: IntT, what: str, loc) -> IntT:
    if a.width != b.width:
        raise StaticSubsetError(
            f"{what}: operand widths differ ({a} vs {b}); the front end must "
            f"insert a cast", loc)
    return IntT(a.width, a.signed and b.signed)


_ARITH = {ir.BinOp.Add, ir.BinOp.Sub, ir.BinOp.Mult, ir.BinOp.Div, ir.BinOp.Mod,
          ir.BinOp.BitAnd, ir.BinOp.BitOr, ir.BinOp.BitXor}
_SHIFT = {ir.BinOp.LShift, ir.BinOp.RShift, ir.BinOp.ARShift}
_LOGIC = {ir.BinOp.And, ir.BinOp.Or}
_REDUCE = {ir.UnaryOp.AndReduce, ir.UnaryOp.OrReduce, ir.UnaryOp.XorReduce}


def infer(e: ir.Expr, sub: Callable[[ir.Expr], T]) -> T:
    """The type of *e*, given ``sub`` that types its sub-expressions.

    Variable references, calls and channel ops are typed by the caller, which
    knows the scope; this covers the operators.
    """
    loc = e.loc
    if isinstance(e, ir.ExprConstant):
        return const_type(e.value, loc)
    if isinstance(e, ir.ExprCast):
        if not isinstance(e.target_type, (ir.DataTypeInt, ir.DataTypeEnum)):
            raise StaticSubsetError("only casts to an integer type are supported", loc)
        _int(sub(e.value), "cast operand", loc)
        return from_ir(e.target_type, loc)
    if isinstance(e, ir.ExprSigned):
        t = _int(sub(e.value), "$signed operand", loc)
        return IntT(t.width, True)
    if isinstance(e, ir.ExprBin):
        if e.op in _ARITH:
            return _same(_int(sub(e.lhs), "operand", loc), _int(sub(e.rhs), "operand", loc),
                         e.op.name, loc)
        if e.op in _SHIFT:
            _int(sub(e.rhs), "shift amount", loc)
            return _int(sub(e.lhs), "shifted value", loc)
        if e.op in _LOGIC:
            _int(sub(e.lhs), "operand", loc), _int(sub(e.rhs), "operand", loc)
            return BIT
        raise StaticSubsetError(f"operator {e.op.name} is not in the static subset", loc)
    if isinstance(e, ir.ExprCompare):
        if len(e.ops) != 1 or len(e.comparators) != 1:
            raise StaticSubsetError("chained comparisons are not supported", loc)
        if e.ops[0] not in (ir.CmpOp.Eq, ir.CmpOp.NotEq, ir.CmpOp.Lt, ir.CmpOp.LtE,
                            ir.CmpOp.Gt, ir.CmpOp.GtE):
            raise StaticSubsetError(f"comparison {e.ops[0].name} is not supported", loc)
        _same(_int(sub(e.left), "operand", loc), _int(sub(e.comparators[0]), "operand", loc),
              e.ops[0].name, loc)
        return BIT
    if isinstance(e, ir.ExprBool):
        for v in e.values:
            _int(sub(v), "operand", loc)
        return BIT
    if isinstance(e, ir.ExprUnary):
        t = _int(sub(e.operand), "operand", loc)
        if e.op in (ir.UnaryOp.Not,) or e.op in _REDUCE:
            return BIT
        return t
    if isinstance(e, ir.ExprIfExp):
        _int(sub(e.test), "condition", loc)
        a, b = sub(e.body), sub(e.orelse)
        if isinstance(a, ArrT) or isinstance(b, ArrT):
            if a != b:
                raise StaticSubsetError(f"?: arms differ ({a} vs {b})", loc)
            return a
        return _same(a, b, "?:", loc)
    if isinstance(e, ir.ExprConcat):
        if not e.values:
            raise StaticSubsetError("empty concatenation", loc)
        return IntT(sum(_int(sub(v), "concatenation operand", loc).width for v in e.values))
    if isinstance(e, ir.ExprReplicate):
        if e.count <= 0:
            raise StaticSubsetError("replication count must be positive", loc)
        return IntT(e.count * _int(sub(e.value), "replicated value", loc).width)
    if isinstance(e, ir.ExprPartSelect):
        t = _int(sub(e.value), "part-select operand", loc)
        _int(sub(e.base), "part-select base", loc)
        if e.width <= 0 or e.width > t.width:
            raise StaticSubsetError(
                f"part select of {e.width} bits from a {t.width}-bit value", loc)
        return IntT(e.width)
    if isinstance(e, ir.ExprSubscript):
        if isinstance(e.slice, ir.ExprSlice):
            raise StaticSubsetError("use ExprPartSelect for part selects", loc)
        base = sub(e.value)
        _int(sub(e.slice), "index", loc)
        if isinstance(base, ArrT):
            return base.elem
        return BIT
    if isinstance(e, ir.ExprList):
        if not e.elts:
            raise StaticSubsetError("empty array literal", loc)
        ts = [sub(x) for x in e.elts]
        for x in ts[1:]:
            if x != ts[0]:
                raise StaticSubsetError(f"array literal elements differ ({ts[0]} vs {x})", loc)
        return ArrT(ts[0], len(ts))
    raise StaticSubsetError(f"expression {type(e).__name__} is not in the static subset", loc)
