"""XLS IR object model: types, values, nodes, functions, procs and packages.

This is the emitter's view of XLS IR. It covers the subset that
``zuspec-be-xls`` emits. The printer (``printer.py``) turns it into ``.ir``
text in the syntax of the pinned XLS release (see ``XLS_SYNTAX_VERSION``).

**Values.** A bits value is a Python ``int`` holding the *unsigned* bit pattern.
A tuple value is a Python ``tuple``, an array value a Python ``list``, and the
token value is :data:`TOKEN`. A value has no type of its own; it is always
read together with the :class:`Type` that describes it.

**Nodes.** Every node has an op name, a result type, positional ``operands``
and keyword ``attrs``. An attr value is an ``int``, ``bool``, ``str``, a
:class:`Node`, a list of nodes, or (for ``literal``) a value. Keeping a node
generic lets the printer, our parser and our interpreter share one shape,
while the per-op rules live in the builder (result types) and, separately, in
``testing/typecheck.py`` (checks).
"""
from __future__ import annotations

import dataclasses as dc
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

#: The xlsynth release whose IR text syntax the printer targets. The syntax
#: references are ``ir_converter_main`` output from this release (see
#: ``tests/unit/data/xls_ref``).
XLS_SYNTAX_VERSION = "xlsynth v0.59.0"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Type:
    def flat_bit_count(self) -> int:
        raise NotImplementedError



#: Words XLS IR's parser takes as keywords (pinned release), so they cannot
#: name a node, parameter, state element or channel. Found by trying each as a
#: parameter name with opt_main.
IR_KEYWORDS = frozenset({
    "bits", "block", "chan", "clock", "file_number", "fn", "instantiation", "next",
    "package", "proc", "reg", "ret", "stage", "token", "top"})


def ir_ident(name: str) -> str:
    """*name* as an XLS IR identifier: a keyword gets a trailing `_`
    (an SV argument named `block` is the IR parameter `block_`)."""
    return name + "_" if name in IR_KEYWORDS else name

@dc.dataclass(frozen=True)
class BitsType(Type):
    width: int

    def __str__(self) -> str:
        return f"bits[{self.width}]"

    def flat_bit_count(self) -> int:
        return self.width


@dc.dataclass(frozen=True)
class TupleType(Type):
    elems: Tuple[Type, ...]

    def __str__(self) -> str:
        return "(" + ", ".join(str(e) for e in self.elems) + ")"

    def flat_bit_count(self) -> int:
        return sum(e.flat_bit_count() for e in self.elems)


@dc.dataclass(frozen=True)
class ArrayType(Type):
    elem: Type
    size: int

    def __str__(self) -> str:
        return f"{self.elem}[{self.size}]"

    def flat_bit_count(self) -> int:
        return self.elem.flat_bit_count() * self.size


@dc.dataclass(frozen=True)
class TokenType(Type):
    def __str__(self) -> str:
        return "token"

    def flat_bit_count(self) -> int:
        return 0


def bits(width: int) -> BitsType:
    return BitsType(width)


def tuple_t(*elems: Type) -> TupleType:
    return TupleType(tuple(elems))


def array_t(elem: Type, size: int) -> ArrayType:
    return ArrayType(elem, size)


TOKEN_T = TokenType()
UNIT_T = TupleType(())


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

class _Token:
    def __repr__(self) -> str:
        return "TOKEN"


TOKEN = _Token()

Value = Union[int, tuple, list, _Token]


def zero_value(t: Type) -> Value:
    if isinstance(t, BitsType):
        return 0
    if isinstance(t, TupleType):
        return tuple(zero_value(e) for e in t.elems)
    if isinstance(t, ArrayType):
        return [zero_value(t.elem) for _ in range(t.size)]
    if isinstance(t, TokenType):
        return TOKEN
    raise TypeError(f"no zero value for {t}")


def format_value(v: Value, t: Type) -> str:
    """Print a value in XLS literal syntax (bits values in decimal)."""
    if isinstance(t, BitsType):
        return str(int(v))
    if isinstance(t, TupleType):
        return "(" + ", ".join(format_value(e, et) for e, et in zip(v, t.elems)) + ")"
    if isinstance(t, ArrayType):
        return "[" + ", ".join(format_value(e, t.elem) for e in v) + "]"
    if isinstance(t, TokenType):
        return "token"
    raise TypeError(f"cannot format a value of type {t}")


# ---------------------------------------------------------------------------
# Data layout (ir_semantics.md "Data types"; data_layout.md)
# ---------------------------------------------------------------------------

def flatten(v: Value, t: Type) -> int:
    """Flatten a value to one bit vector, in the layout of XLS's generated
    Verilog ports (X1-4, checked against ``codegen_main`` v0.59.0).

    Tuple element 0 goes in the *most* significant bits, like an SV packed
    struct's first member. Array element 0 goes in the *least* significant
    bits, like an SV packed array ``[N-1:0]``: codegen writes the literal
    ``[a, b, c]`` as ``{c, b, a}``.

    The DSLX ``as`` cast between arrays and bits uses the other array order
    (``u8[2] as u16`` lowers to ``concat(a[0], a[1])``). The lowering never
    relies on this function for SV packed arrays; it builds the ``concat`` in
    SV order explicitly.
    """
    if isinstance(t, BitsType):
        return int(v)
    if isinstance(t, TupleType):
        out = 0
        for e, et in zip(v, t.elems):
            out = (out << et.flat_bit_count()) | flatten(e, et)
        return out
    if isinstance(t, ArrayType):
        out = 0
        w = t.elem.flat_bit_count()
        for e in reversed(v):
            out = (out << w) | flatten(e, t.elem)
        return out
    if isinstance(t, TokenType):
        return 0
    raise TypeError(f"cannot flatten {t}")


def unflatten(x: int, t: Type) -> Value:
    """Inverse of :func:`flatten`."""
    if isinstance(t, BitsType):
        return x & ((1 << t.width) - 1)
    if isinstance(t, TupleType):
        out = []
        shift = t.flat_bit_count()
        for et in t.elems:
            shift -= et.flat_bit_count()
            out.append(unflatten(x >> shift, et))
        return tuple(out)
    if isinstance(t, ArrayType):
        w = t.elem.flat_bit_count()
        return [unflatten(x >> (w * i), t.elem) for i in range(t.size)]
    if isinstance(t, TokenType):
        return TOKEN
    raise TypeError(f"cannot unflatten {t}")


# ---------------------------------------------------------------------------
# Nodes and containers
# ---------------------------------------------------------------------------

#: A source position: (file number, line, column), all zero-based.
Pos = Tuple[int, int, int]


@dc.dataclass(eq=False)
class Node:
    op: str
    type: Type
    operands: List["Node"] = dc.field(default_factory=list)
    attrs: Dict[str, Any] = dc.field(default_factory=dict)
    id: int = 0
    name: Optional[str] = None
    pos: List[Pos] = dc.field(default_factory=list)

    @property
    def ref(self) -> str:
        """The name other nodes use to refer to this one."""
        return ir_ident(self.name) if self.name is not None else f"{self.op}.{self.id}"

    def all_operands(self) -> Iterator["Node"]:
        """Positional operands, then node-valued attrs, in print order."""
        yield from self.operands
        for v in self.attrs.values():
            if isinstance(v, Node):
                yield v
            elif isinstance(v, list):
                for e in v:
                    if isinstance(e, Node):
                        yield e

    def __repr__(self) -> str:
        return f"<Node {self.ref}: {self.type}>"


@dc.dataclass(eq=False)
class Function:
    name: str
    params: List[Node] = dc.field(default_factory=list)
    nodes: List[Node] = dc.field(default_factory=list)   # excludes params
    ret: Optional[Node] = None
    pos: List[Pos] = dc.field(default_factory=list)

    @property
    def return_type(self) -> Type:
        assert self.ret is not None
        return self.ret.type


@dc.dataclass(eq=False)
class ChannelInterface:
    """A channel on a proc's interface (``proc p<ch: T in>``)."""
    name: str
    type: Type
    direction: str                       # "receive" | "send"
    kind: str = "streaming"
    strictness: str = "proven_mutually_exclusive"
    flow_control: str = "ready_valid"
    flop_kind: str = "none"


@dc.dataclass(eq=False)
class StateElement:
    name: str
    type: Type
    init: Value
    pos: List[Pos] = dc.field(default_factory=list)


@dc.dataclass(eq=False)
class Proc:
    name: str
    channels: List[ChannelInterface] = dc.field(default_factory=list)
    state: List[StateElement] = dc.field(default_factory=list)
    nodes: List[Node] = dc.field(default_factory=list)
    #: State elements XLS's optimizer split off for non-synthesizable logic
    #: (asserts); appears only in ``opt_main`` output.
    non_synth: List[str] = dc.field(default_factory=list)

    def channel(self, name: str) -> ChannelInterface:
        for c in self.channels:
            if c.name == name:
                return c
        raise KeyError(name)

    def state_element(self, name: str) -> StateElement:
        for s in self.state:
            if s.name == name:
                return s
        raise KeyError(name)


@dc.dataclass(eq=False)
class Package:
    name: str
    files: Dict[int, str] = dc.field(default_factory=dict)
    functions: List[Function] = dc.field(default_factory=list)
    procs: List[Proc] = dc.field(default_factory=list)
    top: Optional[str] = None
    _next_id: int = 1

    def new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def note_id(self, i: int) -> None:
        """Keep ``new_id`` above an id that was assigned elsewhere (parser)."""
        if i >= self._next_id:
            self._next_id = i + 1

    def file_number(self, path: str) -> int:
        for n, p in self.files.items():
            if p == path:
                return n
        n = len(self.files)
        self.files[n] = path
        return n

    def function(self, name: str) -> Function:
        for f in self.functions:
            if f.name == name:
                return f
        raise KeyError(name)

    def proc(self, name: str) -> Proc:
        for p in self.procs:
            if p.name == name:
                return p
        raise KeyError(name)

    def members(self) -> List[Union[Function, Proc]]:
        return list(self.functions) + list(self.procs)


#: Ops kept even when nothing uses their value.
EFFECT_OPS = {"send", "receive", "next_value", "assert", "state_read", "cover", "trace"}


def remove_dead_nodes(body: Union[Function, Proc]) -> None:
    """Drop nodes that neither have an effect nor feed one (or the return)."""
    live = set()
    work = [n for n in body.nodes if n.op in EFFECT_OPS]
    if isinstance(body, Function) and body.ret is not None:
        work.append(body.ret)
    while work:
        n = work.pop()
        if id(n) in live:
            continue
        live.add(id(n))
        work.extend(n.all_operands())
    body.nodes = [n for n in body.nodes if id(n) in live]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class XlsTypeError(Exception):
    """An operation was built with operands it does not accept."""


def _bw(n: Node, what: str) -> int:
    if not isinstance(n.type, BitsType):
        raise XlsTypeError(f"{what}: expected bits, got {n.type} ({n.ref})")
    return n.type.width


class Builder:
    """Appends nodes to a function or proc body, computing result types.

    The result-type rules here are the emitter's. ``testing/typecheck.py``
    re-derives them independently from ``ir_semantics.md``.
    """

    _NARY = {"and", "or", "xor", "nand", "nor"}
    _BIN_SAME = {"add", "sub", "udiv", "sdiv", "umod", "smod"}
    _CMP = {"eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"}
    _SHIFT = {"shll", "shrl", "shra"}
    _REDUCE = {"and_reduce", "or_reduce", "xor_reduce"}

    def __init__(self, pkg: Package, body: Union[Function, Proc]):
        self.pkg = pkg
        self.body = body
        self.pos: List[Pos] = []        # applied to each new node

    # -- generic --------------------------------------------------------
    def node(self, op: str, t: Type, operands: Sequence[Node] = (),
             name: Optional[str] = None, **attrs) -> Node:
        n = Node(op=op, type=t, operands=list(operands), attrs=dict(attrs),
                 id=self.pkg.new_id(), name=name, pos=list(self.pos))
        self.body.nodes.append(n)
        return n

    def param(self, name: str, t: Type) -> Node:
        assert isinstance(self.body, Function)
        n = Node(op="param", type=t, id=self.pkg.new_id(), name=name)
        self.body.params.append(n)
        return n

    # -- values ---------------------------------------------------------
    def literal(self, value: Value, t: Type, name: Optional[str] = None) -> Node:
        if isinstance(t, BitsType):
            value = int(value) & ((1 << t.width) - 1)
        return self.node("literal", t, name=name, value=value)

    def bits_lit(self, value: int, width: int) -> Node:
        return self.literal(value, BitsType(width))

    # -- bitwise / arithmetic ------------------------------------------
    def nary(self, op: str, *xs: Node) -> Node:
        assert op in self._NARY
        w = _bw(xs[0], op)
        for x in xs[1:]:
            if _bw(x, op) != w:
                raise XlsTypeError(f"{op}: operand widths differ")
        return self.node(op, BitsType(w), xs)

    def unop(self, op: str, x: Node) -> Node:
        assert op in ("not", "neg", "identity")
        return self.node(op, x.type if op == "identity" else BitsType(_bw(x, op)), [x])

    def binop(self, op: str, a: Node, b: Node) -> Node:
        assert op in self._BIN_SAME
        if _bw(a, op) != _bw(b, op):
            raise XlsTypeError(f"{op}: operand widths differ ({a.type}, {b.type})")
        return self.node(op, a.type, [a, b])

    def mul(self, op: str, a: Node, b: Node, width: int) -> Node:
        assert op in ("umul", "smul")
        _bw(a, op), _bw(b, op)
        return self.node(op, BitsType(width), [a, b])

    def compare(self, op: str, a: Node, b: Node) -> Node:
        assert op in self._CMP
        if op in ("eq", "ne"):
            if a.type != b.type:
                raise XlsTypeError(f"{op}: operand types differ")
        elif _bw(a, op) != _bw(b, op):
            raise XlsTypeError(f"{op}: operand widths differ")
        return self.node(op, BitsType(1), [a, b])

    def shift(self, op: str, x: Node, amt: Node) -> Node:
        assert op in self._SHIFT
        _bw(amt, op)
        return self.node(op, BitsType(_bw(x, op)), [x, amt])

    def reduce(self, op: str, x: Node) -> Node:
        assert op in self._REDUCE
        _bw(x, op)
        return self.node(op, BitsType(1), [x])

    # -- width ----------------------------------------------------------
    def zero_ext(self, x: Node, width: int) -> Node:
        if _bw(x, "zero_ext") == width:
            return x
        if width < x.type.width:
            raise XlsTypeError("zero_ext: narrowing")
        return self.node("zero_ext", BitsType(width), [x], new_bit_count=width)

    def sign_ext(self, x: Node, width: int) -> Node:
        if _bw(x, "sign_ext") == width:
            return x
        if width < x.type.width:
            raise XlsTypeError("sign_ext: narrowing")
        return self.node("sign_ext", BitsType(width), [x], new_bit_count=width)

    def bit_slice(self, x: Node, start: int, width: int) -> Node:
        w = _bw(x, "bit_slice")
        if start == 0 and width == w:
            return x
        if start < 0 or width < 0 or start + width > w:
            raise XlsTypeError(f"bit_slice: [{start} +: {width}] out of bits[{w}]")
        return self.node("bit_slice", BitsType(width), [x], start=start, width=width)

    def dynamic_bit_slice(self, x: Node, start: Node, width: int) -> Node:
        _bw(start, "dynamic_bit_slice")
        if width > _bw(x, "dynamic_bit_slice"):
            # Not in ir_semantics.md; XLS's verifier rejects it (found by the
            # X0 differential test against opt_main).
            raise XlsTypeError("dynamic_bit_slice: width exceeds the operand width")
        return self.node("dynamic_bit_slice", BitsType(width), [x, start], width=width)

    def bit_slice_update(self, x: Node, start: Node, value: Node) -> Node:
        _bw(start, "bit_slice_update"), _bw(value, "bit_slice_update")
        return self.node("bit_slice_update", BitsType(_bw(x, "bit_slice_update")),
                         [x, start, value])

    def concat(self, *xs: Node) -> Node:
        if len(xs) == 1:
            return xs[0]
        return self.node("concat", BitsType(sum(_bw(x, "concat") for x in xs)), xs)

    def resize(self, x: Node, width: int, signed: bool) -> Node:
        """Truncate or extend *x* to *width* (sign- or zero-filled)."""
        w = _bw(x, "resize")
        if width < w:
            return self.bit_slice(x, 0, width)
        if width > w:
            return self.sign_ext(x, width) if signed else self.zero_ext(x, width)
        return x

    # -- selection ------------------------------------------------------
    def sel(self, selector: Node, cases: Sequence[Node],
            default: Optional[Node] = None) -> Node:
        sw = _bw(selector, "sel")
        t = cases[0].type
        for c in list(cases) + ([default] if default is not None else []):
            if c.type != t:
                raise XlsTypeError(f"sel: case types differ ({c.type} vs {t})")
        full = len(cases) == (1 << sw)
        if full and default is not None:
            raise XlsTypeError("sel: a default is not allowed when cases cover the selector")
        if not full and default is None:
            raise XlsTypeError("sel: a default is required")
        attrs: Dict[str, Any] = {"cases": list(cases)}
        if default is not None:
            attrs["default"] = default
        return self.node("sel", t, [selector], **attrs)

    def mux(self, c: Node, if_true: Node, if_false: Node) -> Node:
        """``c ? if_true : if_false`` for a 1-bit *c*."""
        if if_true is if_false:
            return if_true
        return self.sel(c, [if_false, if_true])

    def priority_sel(self, selector: Node, cases: Sequence[Node], default: Node) -> Node:
        if _bw(selector, "priority_sel") != len(cases):
            raise XlsTypeError("priority_sel: selector width must equal the case count")
        return self.node("priority_sel", default.type, [selector],
                         cases=list(cases), default=default)

    # -- aggregates -----------------------------------------------------
    def tuple(self, *xs: Node) -> Node:
        return self.node("tuple", TupleType(tuple(x.type for x in xs)), xs)

    def tuple_index(self, t: Node, index: int, name: Optional[str] = None) -> Node:
        if not isinstance(t.type, TupleType):
            raise XlsTypeError(f"tuple_index: {t.ref} is {t.type}")
        return self.node("tuple_index", t.type.elems[index], [t], name=name, index=index)

    def array(self, *xs: Node) -> Node:
        return self.node("array", ArrayType(xs[0].type, len(xs)), xs)

    def array_index(self, a: Node, indices: Sequence[Node]) -> Node:
        t = a.type
        for _ in indices:
            if not isinstance(t, ArrayType):
                raise XlsTypeError("array_index: too many indices")
            t = t.elem
        return self.node("array_index", t, [a], indices=list(indices))

    def array_update(self, a: Node, v: Node, indices: Sequence[Node]) -> Node:
        return self.node("array_update", a.type, [a, v], indices=list(indices))

    # -- calls ----------------------------------------------------------
    def invoke(self, fn: Function, args: Sequence[Node]) -> Node:
        return self.node("invoke", fn.return_type, args, to_apply=fn.name)

    # -- effects --------------------------------------------------------
    def after_all(self, *toks: Node) -> Node:
        return self.node("after_all", TOKEN_T, toks)

    def receive(self, tok: Node, channel: ChannelInterface,
                predicate: Optional[Node] = None, blocking: bool = True) -> Node:
        """A blocking receive is (token, data); a non-blocking one is
        (token, data, valid), data 0 when no value was waiting."""
        attrs: Dict[str, Any] = {}
        if predicate is not None:
            attrs["predicate"] = predicate
        if not blocking:
            attrs["blocking"] = False
        attrs["channel"] = channel.name
        t = TupleType((TOKEN_T, channel.type)) if blocking else \
            TupleType((TOKEN_T, channel.type, BitsType(1)))
        return self.node("receive", t, [tok], **attrs)

    def send(self, tok: Node, data: Node, channel: ChannelInterface,
             predicate: Optional[Node] = None) -> Node:
        attrs: Dict[str, Any] = {}
        if predicate is not None:
            attrs["predicate"] = predicate
        attrs["channel"] = channel.name
        return self.node("send", TOKEN_T, [tok, data], **attrs)

    def state_read(self, st: StateElement, name: Optional[str] = None) -> Node:
        return self.node("state_read", st.type, name=name, state_element=st.name)

    def next_value(self, st: StateElement, value: Node,
                   predicate: Optional[Node] = None) -> Node:
        attrs: Dict[str, Any] = {"state_element": st.name, "value": value}
        if predicate is not None:
            attrs["predicate"] = predicate
        return self.node("next_value", UNIT_T, **attrs)

    def assert_(self, tok: Node, cond: Node, message: str,
                label: Optional[str] = None) -> Node:
        attrs: Dict[str, Any] = {"message": message}
        if label is not None:
            attrs["label"] = label
        return self.node("assert", TOKEN_T, [tok, cond], **attrs)
