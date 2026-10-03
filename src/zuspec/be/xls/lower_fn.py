"""Lower static-subset zuspec IR to XLS IR dataflow: expressions, statements
and pure functions (``xls-phase0.md`` §3.2, §3.3, BX-3).

The SV meaning is normative (``chisel-xls-assessment.md`` §4.4). Where XLS's
op differs from SV, the lowering inserts the difference explicitly:

* **array read out of bounds** reads 0 in SV, but XLS ``array_index`` clamps,
  so the read is guarded with ``sel`` (E17);
* **a signed bit or part-select base** can be negative. SV reads the bits that
  are in range and 0 for the rest; XLS reads the start as unsigned. The
  lowering pads the value with zeros below bit 0 and shifts the start by the
  pad (E13);
* **division** is rejected unless the divisor is a non-zero constant or the
  division sits under a condition ``d != 0`` on the same value (D-1).

Statements become SSA: every assignment rebinds a name to a new node, ``if``
lowers both arms and joins with ``sel`` (S2), constant-bound loops unroll
(S4, D-4), and ``return`` is predicated (S6). The lowering keeps the *path
predicate* (the conjunction of the enclosing conditions) for effects; procs
(``lower_proc.py``) add the channel effects on top.
"""
from __future__ import annotations

import dataclasses as dc
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import zuspec.ir.core as ir

from . import xir
from .width import ArrT, BIT, IntT, StaticSubsetError, T, const_type, from_ir, infer

#: The most iterations an unrolled loop may take.
MAX_UNROLL = 65536

Key = Tuple[str, object]


def xtype(t: T) -> xir.Type:
    if isinstance(t, IntT):
        return xir.BitsType(t.width)
    return xir.ArrayType(xtype(t.elem), t.size)


def zero_of(t: T) -> xir.Value:
    return xir.zero_value(xtype(t))


def var_key(e: ir.Expr) -> Optional[Key]:
    """The variable an expression names, or ``None``."""
    if isinstance(e, ir.ExprRefField) and isinstance(e.base, ir.TypeExprRefSelf):
        return ("field", e.index)
    if isinstance(e, ir.ExprRefLocal):
        return ("local", e.name)
    if isinstance(e, ir.ExprRefParam):
        return ("param", e.name)
    return None


def _mask(w: int) -> int:
    return (1 << w) - 1


def _signed(v: int, w: int) -> int:
    return v - (1 << w) if w and (v >> (w - 1)) & 1 else v


@dc.dataclass
class FnInfo:
    """A lowered function: what a call site needs."""
    ir_fn: ir.Function
    xls_fn: xir.Function
    params: List[T]
    returns: T


class _Flow:
    """Per-path lowering state: SSA bindings, path predicate, effect token."""

    def __init__(self, env: Dict[Key, xir.Node], conds: List[xir.Node],
                 tok: Optional[xir.Node]):
        self.env = env
        self.conds = conds          # path conditions (1-bit nodes), innermost last
        self.tok = tok

    def fork(self, cond: xir.Node) -> "_Flow":
        return _Flow(dict(self.env), self.conds + [cond], self.tok)


def src_path(f: Optional[str]) -> str:
    """A source path as recorded in the IR: absolute, so a tool run from any
    directory can find it. Pseudo-files (``<...>``) are kept as they are."""
    if not f:
        return "<unknown>"
    return f if f.startswith("<") else os.path.abspath(f)


def pos_of(pkg: xir.Package, loc: Optional[ir.Loc]) -> List[xir.Pos]:
    """An XLS ``pos=`` for *loc*. Loc is 1-based (line, column); XLS positions
    are 0-based."""
    if loc is None or not loc.line:
        return []
    fno = pkg.file_number(src_path(loc.file))
    return [(fno, max(0, loc.line - 1), max(0, (loc.pos or 1) - 1))]


class BodyLowerer:
    """Lowers statements and expressions into one XLS function or proc body."""

    def __init__(self, pkg: xir.Package, body: Union[xir.Function, xir.Proc],
                 types: Dict[Key, T], names: Dict[Key, str],
                 funcs: Dict[str, FnInfo]):
        self.pkg = pkg
        self.b = xir.Builder(pkg, body)
        self.types = dict(types)
        self.names = dict(names)
        self.funcs = funcs
        self.flow = _Flow({}, [], None)
        self._used_names: set = set()
        self._lit_cache: Dict[Tuple[str, object], xir.Node] = {}

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def error(self, msg: str, node: Optional[ir.Base] = None) -> StaticSubsetError:
        return StaticSubsetError(msg, node.loc if node is not None else None)

    def set_loc(self, loc: Optional[ir.Loc]) -> None:
        p = pos_of(self.pkg, loc)
        if p:
            self.b.pos = p

    def unique_name(self, base: str) -> str:
        base = "".join(c if c.isalnum() or c == "_" else "_" for c in base) or "v"
        if base[0].isdigit():
            base = "_" + base
        name, i = base, 0
        while name in self._used_names:
            i += 1
            name = f"{base}__{i}"
        self._used_names.add(name)
        return name

    def name_node(self, n: xir.Node, base: str) -> None:
        """Give a fresh node a readable name (the variable it was assigned to)."""
        if n.name is None and n.op not in ("literal", "param", "state_read"):
            n.name = self.unique_name(base)

    def lit(self, value: int, width: int) -> xir.Node:
        return self.b.bits_lit(value & _mask(width), width)

    @staticmethod
    def lit_value(n: xir.Node) -> Optional[xir.Value]:
        return n.attrs["value"] if n.op == "literal" else None

    def truth(self, n: xir.Node) -> xir.Node:
        """SV truthiness of an integer: ``x != 0`` (1-bit values are themselves)."""
        if n.type == xir.BitsType(1):
            return n
        return self.fold(self.b.compare("ne", n, self.lit(0, n.type.width)))

    def and_(self, *xs: xir.Node) -> Optional[xir.Node]:
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        if len(xs) == 1:
            return xs[0]
        return self.fold(self.b.nary("and", *xs))

    def not_(self, x: xir.Node) -> xir.Node:
        return self.fold(self.b.unop("not", x))

    def pred(self) -> Optional[xir.Node]:
        return self.and_(*self.flow.conds)

    def resize(self, n: xir.Node, t: IntT, width: int) -> xir.Node:
        return self.fold(self.b.resize(n, width, t.signed))

    # ------------------------------------------------------------------
    # Constant folding (for unrolled-loop control and tidy output)
    # ------------------------------------------------------------------
    _FOLD = {"add", "sub", "umul", "smul", "and", "or", "xor", "not", "neg",
             "eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge",
             "shll", "shrl", "shra", "zero_ext", "sign_ext", "bit_slice", "concat",
             "sel", "and_reduce", "or_reduce", "xor_reduce"}

    def fold(self, n: xir.Node) -> xir.Node:
        """Replace *n* (just built) by a literal if its operands are literals."""
        if n.op not in self._FOLD or not isinstance(n.type, xir.BitsType):
            return n
        ops = list(n.all_operands())
        if not ops or any(o.op != "literal" for o in ops):
            return n
        v = self._eval_lit(n)
        if v is None:
            return n
        body = self.b.body
        if body.nodes and body.nodes[-1] is n:
            body.nodes.pop()
        return self.b.literal(v, n.type)

    def _eval_lit(self, n: xir.Node) -> Optional[int]:
        w = n.type.width
        vals = [o.attrs["value"] for o in n.operands]
        ws = [o.type.width for o in n.operands]
        op = n.op
        m = _mask(w)
        if op == "add":
            return (vals[0] + vals[1]) & m
        if op == "sub":
            return (vals[0] - vals[1]) & m
        if op == "umul":
            return (vals[0] * vals[1]) & m
        if op == "smul":
            return (_signed(vals[0], ws[0]) * _signed(vals[1], ws[1])) & m
        if op in ("and", "or", "xor"):
            r = vals[0]
            for v in vals[1:]:
                r = r & v if op == "and" else r | v if op == "or" else r ^ v
            return r
        if op == "not":
            return ~vals[0] & m
        if op == "neg":
            return -vals[0] & m
        if op in ("eq", "ne"):
            return int((vals[0] == vals[1]) == (op == "eq"))
        if op in ("ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"):
            a, c = vals
            if op[0] == "s":
                a, c = _signed(a, ws[0]), _signed(c, ws[1])
            return int({"lt": a < c, "le": a <= c, "gt": a > c, "ge": a >= c}[op[1:]])
        if op == "shll":
            return 0 if vals[1] >= w else (vals[0] << vals[1]) & m
        if op == "shrl":
            return 0 if vals[1] >= w else vals[0] >> vals[1]
        if op == "shra":
            return (_signed(vals[0], w) >> min(vals[1], w)) & m
        if op == "zero_ext":
            return vals[0]
        if op == "sign_ext":
            return _signed(vals[0], ws[0]) & m
        if op == "bit_slice":
            return (vals[0] >> n.attrs["start"]) & m
        if op == "concat":
            r = 0
            for v, vw in zip(vals, ws):
                r = (r << vw) | v
            return r
        if op == "sel":
            cases = [c.attrs["value"] for c in n.attrs["cases"]]
            s = vals[0]
            return cases[s] if s < len(cases) else n.attrs["default"].attrs["value"]
        if op == "and_reduce":
            return int(vals[0] == _mask(ws[0]))
        if op == "or_reduce":
            return int(vals[0] != 0)
        if op == "xor_reduce":
            return bin(vals[0]).count("1") & 1
        return None

    # ------------------------------------------------------------------
    # Expressions
    # ------------------------------------------------------------------
    def expr(self, e: ir.Expr) -> Tuple[xir.Node, T]:
        saved = self.b.pos
        self.set_loc(e.loc)
        try:
            return self._expr(e)
        finally:
            self.b.pos = saved

    def int_expr(self, e: ir.Expr, what: str = "operand") -> Tuple[xir.Node, IntT]:
        n, t = self.expr(e)
        if not isinstance(t, IntT):
            raise self.error(f"{what} must be an integer, is an array ({t})", e)
        return n, t

    def _expr(self, e: ir.Expr) -> Tuple[xir.Node, T]:
        key = var_key(e)
        if key is not None:
            if key not in self.types:
                raise self.error(f"reference to an unknown variable {key[1]!r}", e)
            if key not in self.flow.env:
                raise self.error(f"variable {self.names.get(key, key[1])!r} is read before it is set", e)
            return self.flow.env[key], self.types[key]

        if isinstance(e, ir.ExprCall):
            return self.call(e)
        if isinstance(e, ir.ExprAwait):
            raise self.error("a channel operation may only appear as a statement "
                             "or the right-hand side of an assignment", e)

        # Operators: lower the operands, then type the node with width.infer.
        if isinstance(e, ir.ExprIfExp):
            return self._ifexp(e)

        lowered: Dict[int, Tuple[xir.Node, T]] = {}

        def sub(c: ir.Expr) -> T:
            if id(c) not in lowered:
                lowered[id(c)] = self.expr(c)
            return lowered[id(c)][1]

        t = infer(e, sub)
        L = lambda c: lowered[id(c)][0]          # noqa: E731
        Ty = lambda c: lowered[id(c)][1]         # noqa: E731
        b = self.b

        if isinstance(e, ir.ExprConstant):
            return self.lit(int(e.value), t.width), t
        if isinstance(e, ir.ExprCast):
            return self.resize(L(e.value), Ty(e.value), t.width), t
        if isinstance(e, ir.ExprSigned):
            return L(e.value), t
        if isinstance(e, ir.ExprBin):
            return self._bin(e, L(e.lhs), Ty(e.lhs), L(e.rhs), Ty(e.rhs), t), t
        if isinstance(e, ir.ExprCompare):
            a, c = L(e.left), L(e.comparators[0])
            signed = Ty(e.left).signed and Ty(e.comparators[0]).signed
            op = {ir.CmpOp.Eq: "eq", ir.CmpOp.NotEq: "ne", ir.CmpOp.Lt: "lt",
                  ir.CmpOp.LtE: "le", ir.CmpOp.Gt: "gt", ir.CmpOp.GtE: "ge"}[e.ops[0]]
            if op not in ("eq", "ne"):
                op = ("s" if signed else "u") + op
            return self.fold(b.compare(op, a, c)), t
        if isinstance(e, ir.ExprBool):
            xs = [self.truth(L(v)) for v in e.values]
            if len(xs) == 1:
                return xs[0], t
            return self.fold(b.nary("and" if e.op == ir.BoolOp.And else "or", *xs)), t
        if isinstance(e, ir.ExprUnary):
            x = L(e.operand)
            if e.op == ir.UnaryOp.Invert:
                return self.fold(b.unop("not", x)), t
            if e.op == ir.UnaryOp.Not:
                return self.not_(self.truth(x)), t
            if e.op == ir.UnaryOp.USub:
                return self.fold(b.unop("neg", x)), t
            if e.op == ir.UnaryOp.UAdd:
                return x, t
            op = {ir.UnaryOp.AndReduce: "and_reduce", ir.UnaryOp.OrReduce: "or_reduce",
                  ir.UnaryOp.XorReduce: "xor_reduce"}[e.op]
            return self.fold(b.reduce(op, x)), t
        if isinstance(e, ir.ExprConcat):
            return self.fold(b.concat(*[L(v) for v in e.values])), t
        if isinstance(e, ir.ExprReplicate):
            return self.fold(b.concat(*([L(e.value)] * e.count))), t
        if isinstance(e, ir.ExprPartSelect):
            return self.part_read(L(e.value), L(e.base), Ty(e.base), e.width), t
        if isinstance(e, ir.ExprSubscript):
            base_t = Ty(e.value)
            if isinstance(base_t, ArrT):
                return self.array_read(L(e.value), base_t, L(e.slice), Ty(e.slice)), t
            return self.part_read(L(e.value), L(e.slice), Ty(e.slice), 1), t
        if isinstance(e, ir.ExprList):
            elems = [L(x) for x in e.elts]
            if all(x.op == "literal" for x in elems):
                return b.literal([x.attrs["value"] for x in elems], xtype(t)), t
            return b.array(*elems), t
        raise self.error(f"expression {type(e).__name__} is not supported", e)

    def _bin(self, e: ir.ExprBin, a: xir.Node, at: IntT, c: xir.Node, ct: IntT,
             t: IntT) -> xir.Node:
        b = self.b
        op = e.op
        if op in (ir.BinOp.Add, ir.BinOp.Sub):
            return self.fold(b.binop("add" if op == ir.BinOp.Add else "sub", a, c))
        if op == ir.BinOp.Mult:
            return self.fold(b.mul("smul" if t.signed else "umul", a, c, t.width))
        if op in (ir.BinOp.Div, ir.BinOp.Mod):
            self._check_divisor(e, c)
            name = ("s" if t.signed else "u") + ("div" if op == ir.BinOp.Div else "mod")
            return b.binop(name, a, c)
        if op in (ir.BinOp.BitAnd, ir.BinOp.BitOr, ir.BinOp.BitXor):
            name = {ir.BinOp.BitAnd: "and", ir.BinOp.BitOr: "or", ir.BinOp.BitXor: "xor"}[op]
            return self.fold(b.nary(name, a, c))
        if op == ir.BinOp.LShift:
            return self.fold(b.shift("shll", a, c))
        if op == ir.BinOp.RShift:
            return self.fold(b.shift("shrl", a, c))
        if op == ir.BinOp.ARShift:
            return self.fold(b.shift("shra" if at.signed else "shrl", a, c))
        if op in (ir.BinOp.And, ir.BinOp.Or):
            return self.fold(b.nary("and" if op == ir.BinOp.And else "or",
                                    self.truth(a), self.truth(c)))
        raise self.error(f"operator {op.name} is not supported", e)

    def _check_divisor(self, e: ir.ExprBin, d: xir.Node) -> None:
        """D-1: a divisor must be a non-zero constant, or guarded by ``d != 0``
        on the same value (the same SSA node) in an enclosing condition."""
        v = self.lit_value(d)
        if v is not None:
            if v == 0:
                raise self.error("division by the constant zero", e)
            return
        for c in self.flow.conds:
            if self._implies_nonzero(c, d):
                return
        raise self.error(
            "division by a value that may be zero: guard it with "
            "`if (d != 0)` or `d != 0 ? n / d : ...` (xls-phase0.md D-1)", e)

    @staticmethod
    def _unextended(n: xir.Node) -> xir.Node:
        # Extension keeps a value non-zero, so look through it on both sides:
        # `if (b != 0)` compares a widened b (the literal 0 is 32 bits).
        while n.op in ("zero_ext", "sign_ext"):
            n = n.operands[0]
        return n

    def _implies_nonzero(self, c: xir.Node, d: xir.Node) -> bool:
        d = self._unextended(d)
        if c.op == "ne":
            a, z = c.operands
            for x, y in ((a, z), (z, a)):
                if self._unextended(x) is d and y.op == "literal" and y.attrs["value"] == 0:
                    return True
        if c.op == "and":
            return any(self._implies_nonzero(x, d) for x in c.operands)
        return False

    def _ifexp(self, e: ir.ExprIfExp) -> Tuple[xir.Node, T]:
        c, ct = self.int_expr(e.test, "condition")
        c = self.truth(c)
        outer = self.flow
        self.flow = outer.fork(c)
        try:
            body = self.expr(e.body)
        finally:
            self.flow = outer
        self.flow = outer.fork(self.not_(c))
        try:
            orelse = self.expr(e.orelse)
        finally:
            self.flow = outer
        types = {id(e.test): ct, id(e.body): body[1], id(e.orelse): orelse[1]}
        t = infer(e, lambda x: types[id(x)])
        return self.mux(c, body[0], orelse[0]), t

    def mux(self, c: xir.Node, if_true: xir.Node, if_false: xir.Node) -> xir.Node:
        if if_true is if_false:
            return if_true
        v = self.lit_value(c)
        if v is not None:
            return if_true if v else if_false
        return self.b.mux(c, if_true, if_false)

    # -- selects ---------------------------------------------------------
    def _padded(self, x: xir.Node, start: xir.Node, start_t: IntT,
                width: int) -> Tuple[xir.Node, xir.Node]:
        """For a possibly negative signed start: pad *x* with *width* zeros
        below bit 0 and move the start up by *width*, so bits below 0 read
        as 0 and a start below ``-width`` lands out of range."""
        k = max(start_t.width, (x.type.width + width).bit_length()) + 1
        s = self.fold(self.b.sign_ext(start, k)) if k > start_t.width else start
        s = self.fold(self.b.binop("add", s, self.lit(width, k)))
        # A negative sum is a large unsigned start, i.e. out of range.
        return self.fold(self.b.concat(x, self.lit(0, width))), s

    def part_read(self, x: xir.Node, start: xir.Node, start_t: IntT, width: int) -> xir.Node:
        """SV ``x[start +: width]``: bits outside ``x`` read as 0."""
        w = x.type.width
        sv = self.lit_value(start)
        if sv is not None:
            sv = _signed(sv, start_t.width) if start_t.signed else sv
            if 0 <= sv and sv + width <= w:
                return self.fold(self.b.bit_slice(x, sv, width))
            if sv >= w or sv + width <= 0:
                return self.lit(0, width)
        if start_t.signed:
            x, start = self._padded(x, start, start_t, width)
        return self.b.dynamic_bit_slice(x, start, width)

    def part_write(self, x: xir.Node, start: xir.Node, start_t: IntT,
                   value: xir.Node) -> xir.Node:
        """SV ``x[start +: W] = value``: bits outside ``x`` are not written."""
        w, width = x.type.width, value.type.width
        sv = self.lit_value(start)
        if sv is not None:
            sv = _signed(sv, start_t.width) if start_t.signed else sv
            if sv >= w or sv + width <= 0:
                return x
            if 0 <= sv and sv + width <= w:
                parts = []
                if sv + width < w:
                    parts.append(self.fold(self.b.bit_slice(x, sv + width, w - sv - width)))
                parts.append(value)
                if sv > 0:
                    parts.append(self.fold(self.b.bit_slice(x, 0, sv)))
                return self.fold(self.b.concat(*parts))
        if start_t.signed:
            px, ps = self._padded(x, start, start_t, width)
            upd = self.b.bit_slice_update(px, ps, value)
            return self.fold(self.b.bit_slice(upd, width, w))
        return self.b.bit_slice_update(x, start, value)

    def unsigned_index(self, idx: xir.Node, idx_t: IntT, size: int) -> xir.Node:
        """An index XLS can read as unsigned without aliasing.

        XLS reads every index as unsigned. A signed index is sign-extended
        one bit past what *size* needs, so a negative value becomes a large
        unsigned one, which is out of bounds, as in SV. Without that, a
        narrow negative index (``s3`` -4 is ``0b100``) would alias index 4.
        """
        if not idx_t.signed:
            return idx
        k = max(idx_t.width, size.bit_length()) + 1
        return self.fold(self.b.sign_ext(idx, k))

    def in_bounds(self, idx: xir.Node, size: int) -> xir.Node:
        """``idx < size`` for an index from :meth:`unsigned_index`."""
        w = idx.type.width
        if (1 << w) <= size:
            return self.lit(1, 1)           # every index value is in range
        k = max(w, size.bit_length())
        i = self.fold(self.b.zero_ext(idx, k)) if k > w else idx
        return self.fold(self.b.compare("ult", i, self.lit(size, k)))

    def array_read(self, a: xir.Node, at: ArrT, idx: xir.Node, idx_t: IntT) -> xir.Node:
        """SV ``a[i]``: out of bounds reads 0 (XLS clamps, so guard it)."""
        iv = self.lit_value(idx)
        if iv is not None:
            iv = _signed(iv, idx_t.width) if idx_t.signed else iv
            if 0 <= iv < at.size:
                return self.b.array_index(a, [idx])
            return self.b.literal(zero_of(at.elem), xtype(at.elem))
        idx = self.unsigned_index(idx, idx_t, at.size)
        elem = self.b.array_index(a, [idx])
        ok = self.in_bounds(idx, at.size)
        if self.lit_value(ok) == 1:
            return elem
        zero = self.b.literal(zero_of(at.elem), xtype(at.elem))
        return self.mux(ok, elem, zero)

    # -- calls -----------------------------------------------------------
    def call(self, e: ir.ExprCall) -> Tuple[xir.Node, T]:
        name = None
        if isinstance(e.func, ir.ExprAttribute):
            name = e.func.attr
        elif isinstance(e.func, (ir.ExprRefUnresolved,)):
            name = e.func.name
        if name is None or name not in self.funcs:
            raise self.error(f"call to an unknown function {name!r}", e)
        info = self.funcs[name]
        if len(e.args) != len(info.params):
            raise self.error(f"{name}: expects {len(info.params)} arguments, got {len(e.args)}", e)
        args = []
        for a, pt in zip(e.args, info.params):
            n, t = self.expr(a)
            args.append(self.coerce(n, t, pt, a, f"argument of {name}"))
        return self.b.invoke(info.xls_fn, args), info.returns

    def coerce(self, n: xir.Node, t: T, want: T, e: ir.Expr, what: str) -> xir.Node:
        """Check that *n* (type *t*) can be stored as *want*.

        Widths must match (the front end casts); signedness may differ, since
        bits are bits. A bare constant takes the target width if it fits.
        """
        if isinstance(want, ArrT) or isinstance(t, ArrT):
            if t != want:
                raise self.error(f"{what}: type {t} does not match {want}", e)
            return n
        if t.width == want.width:
            return n
        if isinstance(e, ir.ExprConstant) and isinstance(e.value, int) and \
                0 <= e.value < (1 << want.width):
            return self.lit(int(e.value), want.width)
        raise self.error(f"{what}: a {t.width}-bit value where {want.width} bits are "
                         f"needed; the front end must insert a cast", e)

    # ------------------------------------------------------------------
    # Assignment targets
    # ------------------------------------------------------------------
    def target_type(self, tgt: ir.Expr) -> T:
        key = var_key(tgt)
        if key is not None:
            if key not in self.types:
                raise self.error(f"assignment to an unknown variable {key[1]!r}", tgt)
            return self.types[key]
        if isinstance(tgt, ir.ExprSubscript):
            bt = self.target_type(tgt.value)
            return bt.elem if isinstance(bt, ArrT) else BIT
        if isinstance(tgt, ir.ExprPartSelect):
            return IntT(tgt.width)
        if isinstance(tgt, ir.ExprConcat):
            return IntT(sum(self.target_type(v).width for v in tgt.values))
        raise self.error(f"cannot assign to {type(tgt).__name__}", tgt)

    def assign(self, tgt: ir.Expr, n: xir.Node, t: T, src: ir.Expr) -> None:
        want = self.target_type(tgt)
        n = self.coerce(n, t, want, src, "assignment")
        self._store(tgt, n)

    def _store(self, tgt: ir.Expr, n: xir.Node) -> None:
        key = var_key(tgt)
        if key is not None:
            self.name_node(n, self.names.get(key, str(key[1])))
            self.flow.env[key] = n
            return
        if isinstance(tgt, ir.ExprSubscript):
            base, bt = self.expr(tgt.value)
            idx, it = self.int_expr(tgt.slice, "index")
            if isinstance(bt, ArrT):
                # array_update leaves the array unchanged out of bounds (E18).
                idx = self.unsigned_index(idx, it, bt.size)
                self._store(tgt.value, self.b.array_update(base, n, [idx]))
            else:
                self._store(tgt.value, self.part_write(base, idx, it, n))
            return
        if isinstance(tgt, ir.ExprPartSelect):
            base, _ = self.int_expr(tgt.value)
            start, st = self.int_expr(tgt.base, "part-select base")
            self._store(tgt.value, self.part_write(base, start, st, n))
            return
        if isinstance(tgt, ir.ExprConcat):
            hi = n.type.width
            for part in tgt.values:
                w = self.target_type(part).width
                self._store(part, self.fold(self.b.bit_slice(n, hi - w, w)))
                hi -= w
            return
        raise self.error(f"cannot assign to {type(tgt).__name__}", tgt)

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------
    def stmts(self, body: Sequence[ir.Stmt]) -> None:
        for s in body:
            saved = self.b.pos
            self.set_loc(s.loc)
            try:
                self.stmt(s)
            finally:
                self.b.pos = saved

    def stmt(self, s: ir.Stmt) -> None:
        if isinstance(s, ir.StmtAnnAssign):
            self.declare(s)
        elif isinstance(s, ir.StmtAssign):
            if len(s.targets) != 1:
                raise self.error("multiple assignment targets are not supported", s)
            if isinstance(s.value, ir.ExprAwait):
                self.await_assign(s.targets[0], s.value, s)
            else:
                n, t = self.expr(s.value)
                self.assign(s.targets[0], n, t, s.value)
        elif isinstance(s, ir.StmtExpr):
            if isinstance(s.expr, ir.ExprAwait):
                self.await_stmt(s.expr, s)
            elif isinstance(s.expr, ir.ExprCall):
                self.expr(s.expr)      # a pure call: no effect
            else:
                raise self.error("expression statement has no effect", s)
        elif isinstance(s, ir.StmtIf):
            self.if_(s)
        elif isinstance(s, ir.StmtMatch):
            self.stmts([self.match_to_if(s)])
        elif isinstance(s, ir.StmtWhile):
            self.while_(s)
        elif isinstance(s, ir.StmtRepeat):
            self.repeat(s)
        elif isinstance(s, ir.StmtReturn):
            self.return_(s)
        elif isinstance(s, ir.StmtAssert):
            self.assert_(s)
        elif isinstance(s, ir.StmtPass):
            pass
        elif isinstance(s, (ir.StmtBreak, ir.StmtContinue)):
            raise self.error("break/continue are not supported in the static subset (X0)", s)
        else:
            raise self.error(f"statement {type(s).__name__} is not in the static subset", s)

    def declare(self, s: ir.StmtAnnAssign) -> None:
        key = var_key(s.target)
        if key is None or key[0] != "local":
            raise self.error("a declaration must name a local (ExprRefLocal)", s)
        if s.ir_type is None:
            raise self.error(f"local {key[1]!r} has no type (ir_type)", s)
        t = from_ir(s.ir_type, s.loc)
        self.types[key] = t
        self.names[key] = str(key[1])
        if s.value is not None:
            n, vt = self.expr(s.value)
            n = self.coerce(n, vt, t, s.value, f"initializer of {key[1]}")
        else:
            # SV automatic variables start at 0 (2-state) on every entry (P3).
            n = self.b.literal(zero_of(t), xtype(t))
        self.flow.env[key] = n
        self.name_node(n, str(key[1]))

    def if_(self, s: ir.StmtIf) -> None:
        c, _ = self.int_expr(s.test, "condition")
        c = self.truth(c)
        cv = self.lit_value(c)
        if cv is not None:                 # constant condition: one arm
            self.stmts(s.body if cv else s.orelse)
            return
        outer = self.flow
        then = outer.fork(c)
        self.flow = then
        self.stmts(s.body)
        els = outer.fork(self.not_(c))
        els.tok = then.tok               # effects stay in program order (D-6)
        self.flow = els
        self.stmts(s.orelse)
        self.flow = outer
        outer.tok = els.tok
        for key in list(outer.env.keys()):
            a, b = then.env.get(key), els.env.get(key)
            if a is not b:
                outer.env[key] = self.mux(c, a, b)
                self.name_node(outer.env[key], self.names.get(key, str(key[1])))

    def match_to_if(self, s: ir.StmtMatch) -> ir.Stmt:
        """``case`` as a first-match-wins ``if`` chain (S3)."""
        chain: List[ir.Stmt] = []
        for case in reversed(s.cases):
            test = self._pattern_test(s.subject, case.pattern, s)
            if case.guard is not None:
                test = ir.ExprBool(op=ir.BoolOp.And, values=[test, case.guard]) \
                    if test is not None else case.guard
            if test is None:                       # default
                chain = list(case.body)
            else:
                chain = [ir.StmtIf(test=test, body=list(case.body), orelse=chain, loc=case.loc)]
        return chain[0] if len(chain) == 1 else ir.StmtIf(
            test=ir.ExprConstant(value=True), body=chain, orelse=[])

    def _pattern_test(self, subject: ir.Expr, p: ir.Pattern, s) -> Optional[ir.Expr]:
        if isinstance(p, ir.PatternValue):
            return ir.ExprCompare(left=subject, ops=[ir.CmpOp.Eq], comparators=[p.value],
                                  loc=p.loc or s.loc)
        if isinstance(p, ir.PatternOr):
            tests = [self._pattern_test(subject, x, s) for x in p.patterns]
            if any(t is None for t in tests):
                return None
            return ir.ExprBool(op=ir.BoolOp.Or, values=tests)
        if isinstance(p, ir.PatternAs) and p.pattern is None and p.name is None:
            return None
        raise self.error(f"case pattern {type(p).__name__} is not supported", s)

    def _const_cond(self, e: ir.Expr, s: ir.Stmt) -> bool:
        c, _ = self.int_expr(e, "loop condition")
        v = self.lit_value(self.truth(c))
        if v is None:
            raise self.error("loop condition does not have a constant value at "
                             "elaboration; only constant-bound loops are supported (D-4)", s)
        return bool(v)

    def while_(self, s: ir.StmtWhile) -> None:
        n = 0
        while self._const_cond(s.test, s):
            n += 1
            if n > MAX_UNROLL:
                raise self.error(f"loop runs more than {MAX_UNROLL} iterations", s)
            self.stmts(s.body)

    def repeat(self, s: ir.StmtRepeat) -> None:
        cnt, _ = self.int_expr(s.count, "repeat count")
        count = self.lit_value(cnt)
        if count is None:
            raise self.error("repeat count is not a constant", s)
        key = var_key(s.iterator) if s.iterator is not None else None
        if key is not None and key not in self.types:
            self.types[key] = IntT(32, True)
            self.names[key] = str(key[1])
        for i in range(count):
            if key is not None:
                self.flow.env[key] = self.lit(i, self.types[key].width)
            self.stmts(s.body)

    # -- function-only / proc-only hooks ------------------------------------
    def return_(self, s: ir.StmtReturn) -> None:
        raise self.error("return is only allowed in a function", s)

    def assert_(self, s: ir.StmtAssert) -> None:
        c, _ = self.int_expr(s.test, "assertion")
        c = self.truth(c)
        p = self.pred()
        ok = c if p is None else self.fold(self.b.nary("or", self.not_(p), c))
        msg = "assertion failed"
        if isinstance(s.msg, ir.ExprConstant) and isinstance(s.msg.value, str):
            msg = s.msg.value
        if s.loc is not None:
            msg = f"{msg} @ {src_path(s.loc.file)}:{s.loc.line}"
        tok = self.flow.tok if self.flow.tok is not None else self.token_source()
        self.flow.tok = self.b.assert_(tok, ok, msg)

    def token_source(self) -> xir.Node:
        return self.b.literal(xir.TOKEN, xir.TOKEN_T)

    def await_assign(self, tgt: ir.Expr, e: ir.ExprAwait, s: ir.Stmt) -> None:
        raise self.error("channel operations are only allowed in a proc's run()", s)

    def await_stmt(self, e: ir.ExprAwait, s: ir.Stmt) -> None:
        raise self.error("channel operations are only allowed in a proc's run()", s)


class FunctionLowerer(BodyLowerer):
    """A pure function (S5): parameters in, one return value out."""

    def __init__(self, pkg: xir.Package, fn: ir.Function, funcs: Dict[str, FnInfo]):
        self.ir_fn = fn
        self.xfn = xir.Function(name=fn.name)
        types: Dict[Key, T] = {}
        names: Dict[Key, str] = {}
        args = fn.args.args if fn.args is not None else []
        for a in args:
            if a.datatype is None:
                raise StaticSubsetError(f"{fn.name}: argument {a.arg!r} has no type", a.loc or fn.loc)
            types[("param", a.arg)] = from_ir(a.datatype, a.loc or fn.loc)
            names[("param", a.arg)] = a.arg
        if fn.returns is None:
            raise StaticSubsetError(f"{fn.name}: a function must return a value", fn.loc)
        self.ret_t = from_ir(fn.returns, fn.loc)
        super().__init__(pkg, self.xfn, types, names, funcs)
        self.params = [types[("param", a.arg)] for a in args]
        for a in args:
            p = self.b.param(a.arg, xtype(types[("param", a.arg)]))
            self._used_names.add(a.arg)
            self.flow.env[("param", a.arg)] = p
        self.done: Optional[xir.Node] = None
        self.ret: Optional[xir.Node] = None

    def lower(self) -> FnInfo:
        self.set_loc(self.ir_fn.loc)
        self.ret = self.b.literal(zero_of(self.ret_t), xtype(self.ret_t))
        self.done = self.lit(0, 1)
        self.stmts(self.ir_fn.body)
        ret = self.ret
        if not self.xfn.nodes or self.xfn.nodes[-1] is not ret:
            # 'ret' goes on the last node, as XLS prints it.
            ret = self.b.unop("identity", ret)
        self.xfn.ret = ret
        xir.remove_dead_nodes(self.xfn)
        return FnInfo(self.ir_fn, self.xfn, self.params, self.ret_t)

    def pred(self) -> Optional[xir.Node]:
        # Statements after a taken return have no effect.
        live = None if self.lit_value(self.done) == 0 else self.not_(self.done)
        return self.and_(*self.flow.conds, live)

    def return_(self, s: ir.StmtReturn) -> None:
        if s.value is None:
            raise self.error("a function must return a value", s)
        n, t = self.expr(s.value)
        n = self.coerce(n, t, self.ret_t, s.value, "return value")
        active = self.pred()
        if active is None:
            self.ret, self.done = n, self.lit(1, 1)
            return
        self.ret = self.mux(active, n, self.ret)
        self.done = self.fold(self.b.nary("or", self.done, active))


def lower_functions(pkg: xir.Package, fns: Sequence[ir.Function],
                    funcs: Optional[Dict[str, FnInfo]] = None) -> Dict[str, FnInfo]:
    """Lower pure functions in dependency order (callees first).

    Recursion is rejected: XLS functions cannot recurse.
    """
    funcs = {} if funcs is None else funcs
    by_name = {f.name: f for f in fns}
    state: Dict[str, str] = {}

    def callees(f: ir.Function) -> List[str]:
        out: List[str] = []

        def walk(o):
            if isinstance(o, ir.ExprCall):
                if isinstance(o.func, ir.ExprAttribute):
                    out.append(o.func.attr)
                elif isinstance(o.func, ir.ExprRefUnresolved):
                    out.append(o.func.name)
            if isinstance(o, ir.Base):
                for fld in dc.fields(o):
                    walk(getattr(o, fld.name))
            elif isinstance(o, list):
                for x in o:
                    walk(x)
        walk(f.body)
        return [c for c in out if c in by_name]

    def visit(name: str) -> None:
        if state.get(name) == "done":
            return
        if state.get(name) == "active":
            raise StaticSubsetError(f"function {name!r} is recursive", by_name[name].loc)
        state[name] = "active"
        for c in callees(by_name[name]):
            visit(c)
        info = FunctionLowerer(pkg, by_name[name], funcs).lower()
        pkg.functions.append(info.xls_fn)
        funcs[name] = info
        state[name] = "done"

    for f in fns:
        visit(f.name)
    return funcs
