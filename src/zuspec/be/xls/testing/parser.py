"""Parse XLS IR text into the :mod:`xir` object model.

This parser is the X0 stand-in for XLS's own ``ir_parser``. It is written
from the IR text that XLS prints (``ir_converter_main`` / ``opt_main`` output
of the pinned release), **not** from our printer, so a round trip through
both is a real check of the printer.

It reads the subset the emitter produces plus what XLS prints for it:
functions, procs with proc-scoped channels (``proc p<ch: T in>``), explicit
``state_read`` / ``next_value``, ``file_number`` lines and ``pos=``.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from ..xir import (ArrayType, BitsType, ChannelInterface, Function, Node,
                   Package, Proc, StateElement, TOKEN, TOKEN_T, TokenType,
                   TupleType, Type, Value)


class IrParseError(Exception):
    pass


_TOKEN_RE = re.compile(r"""
    (?P<ws>\s+|//[^\n]*)
  | (?P<str>"(?:[^"\\]|\\.)*")
  | (?P<num>-?(?:0x[0-9a-fA-F_]+|0b[01_]+|[0-9][0-9_]*))
  | (?P<arrow>->)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_.]*)
  | (?P<punct>[()\[\]{}<>:=,])
""", re.VERBOSE)

#: Attrs whose values name other nodes. ``value`` does too, except on
#: ``literal`` where it is the literal value.
_NODE_ATTRS = {"predicate", "value", "cases", "default", "indices"}


class _Tok:
    __slots__ = ("kind", "text", "line")

    def __init__(self, kind: str, text: str, line: int):
        self.kind, self.text, self.line = kind, text, line

    def __repr__(self) -> str:
        return f"{self.kind}:{self.text!r}@{self.line}"


def _lex(text: str) -> List[_Tok]:
    out: List[_Tok] = []
    i, line = 0, 1
    while i < len(text):
        m = _TOKEN_RE.match(text, i)
        if m is None:
            raise IrParseError(f"line {line}: unexpected character {text[i]!r}")
        kind = m.lastgroup
        s = m.group(kind)
        if kind != "ws":
            out.append(_Tok(kind, s, line))
        line += s.count("\n")
        i = m.end()
    out.append(_Tok("eof", "", line))
    return out


def _int(s: str) -> int:
    s = s.replace("_", "")
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    if s.startswith("0x"):
        v = int(s[2:], 16)
    elif s.startswith("0b"):
        v = int(s[2:], 2)
    else:
        v = int(s, 10)
    return -v if neg else v


def coerce_value(raw: Any, t: Type) -> Value:
    """Turn a parsed raw value (ints, lists, tuples) into a value of type *t*."""
    if isinstance(t, BitsType):
        if not isinstance(raw, int):
            raise IrParseError(f"expected a bits value for {t}, got {raw!r}")
        return raw & ((1 << t.width) - 1)
    if isinstance(t, TupleType):
        if not isinstance(raw, tuple) or len(raw) != len(t.elems):
            raise IrParseError(f"expected a {len(t.elems)}-tuple for {t}, got {raw!r}")
        return tuple(coerce_value(r, et) for r, et in zip(raw, t.elems))
    if isinstance(t, ArrayType):
        if not isinstance(raw, list) or len(raw) != t.size:
            raise IrParseError(f"expected {t.size} array elements for {t}, got {raw!r}")
        return [coerce_value(r, t.elem) for r in raw]
    if isinstance(t, TokenType):
        return TOKEN
    raise IrParseError(f"cannot coerce to {t}")


class _Parser:
    def __init__(self, text: str):
        self.toks = _lex(text)
        self.i = 0

    # -- token helpers ---------------------------------------------------
    def peek(self, k: int = 0) -> _Tok:
        return self.toks[self.i + k]

    def next(self) -> _Tok:
        t = self.toks[self.i]
        self.i += 1
        return t

    def at(self, text: str) -> bool:
        t = self.peek()
        return t.kind in ("punct", "ident", "arrow") and t.text == text

    def expect(self, text: str) -> _Tok:
        t = self.next()
        if t.text != text:
            raise IrParseError(f"line {t.line}: expected {text!r}, got {t.text!r}")
        return t

    def ident(self) -> str:
        t = self.next()
        if t.kind != "ident":
            raise IrParseError(f"line {t.line}: expected a name, got {t.text!r}")
        return t.text

    def integer(self) -> int:
        t = self.next()
        if t.kind != "num":
            raise IrParseError(f"line {t.line}: expected a number, got {t.text!r}")
        return _int(t.text)

    # -- types -----------------------------------------------------------
    def type(self) -> Type:
        if self.at("bits"):
            self.next()
            self.expect("[")
            t: Type = BitsType(self.integer())
            self.expect("]")
        elif self.at("token"):
            self.next()
            t = TOKEN_T
        elif self.at("("):
            self.next()
            elems = []
            while not self.at(")"):
                elems.append(self.type())
                if not self.at(")"):
                    self.expect(",")
            self.expect(")")
            t = TupleType(tuple(elems))
        else:
            raise IrParseError(f"line {self.peek().line}: expected a type, got {self.peek().text!r}")
        while self.at("["):
            self.next()
            t = ArrayType(t, self.integer())
            self.expect("]")
        return t

    # -- raw values ------------------------------------------------------
    def raw_value(self) -> Any:
        t = self.peek()
        if t.kind == "num":
            return _int(self.next().text)
        if t.kind == "ident" and t.text == "token":
            self.next()
            return TOKEN
        if t.kind == "ident" and t.text == "bits":     # bits[N]:value
            self.type()
            self.expect(":")
            return self.raw_value()
        if self.at("["):
            self.next()
            elems = []
            while not self.at("]"):
                elems.append(self.raw_value())
                if not self.at("]"):
                    self.expect(",")
            self.expect("]")
            return elems
        if self.at("("):
            self.next()
            elems = []
            while not self.at(")"):
                elems.append(self.raw_value())
                if not self.at(")"):
                    self.expect(",")
            self.expect(")")
            return tuple(elems)
        raise IrParseError(f"line {t.line}: expected a value, got {t.text!r}")

    def pos_list(self) -> List[Tuple[int, int, int]]:
        self.expect("[")
        out = []
        while not self.at("]"):
            self.expect("(")
            f = self.integer(); self.expect(",")
            l = self.integer(); self.expect(",")
            c = self.integer()
            self.expect(")")
            out.append((f, l, c))
            if not self.at("]"):
                self.expect(",")
        self.expect("]")
        return out

    # -- package ---------------------------------------------------------
    def package(self) -> Package:
        self.expect("package")
        pkg = Package(name=self.ident())
        # Nodes reference functions by name; resolve after all are read.
        while self.peek().kind != "eof":
            if self.at("file_number"):
                self.next()
                num = self.integer()
                s = self.next()
                if s.kind != "str":
                    raise IrParseError(f"line {s.line}: expected a file name")
                pkg.files[num] = bytes(s.text[1:-1], "utf-8").decode("unicode_escape")
                continue
            top = False
            if self.at("top"):
                self.next()
                top = True
            if self.at("fn"):
                self.next()
                f = self.function(pkg)
                pkg.functions.append(f)
                if top:
                    pkg.top = f.name
            elif self.at("proc"):
                self.next()
                p = self.proc(pkg)
                pkg.procs.append(p)
                if top:
                    pkg.top = p.name
            else:
                t = self.peek()
                raise IrParseError(f"line {t.line}: unexpected {t.text!r} at package scope")
        return pkg

    def _param_attrs(self) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        while self.peek().kind == "ident" and self.peek(1).text == "=":
            key = self.ident()
            self.expect("=")
            attrs[key] = self.pos_list() if key == "pos" else self.integer()
        return attrs

    def function(self, pkg: Package) -> Function:
        f = Function(name=self.ident())
        scope: Dict[str, Node] = {}
        self.expect("(")
        while not self.at(")"):
            name = self.ident()
            self.expect(":")
            t = self.type()
            attrs = self._param_attrs()
            n = Node(op="param", type=t, name=name, id=attrs.get("id", 0),
                     pos=attrs.get("pos", []))
            pkg.note_id(n.id)
            f.params.append(n)
            scope[name] = n
            if not self.at(")"):
                self.expect(",")
        self.expect(")")
        self.expect("->")
        ret_t = self.type()
        self.expect("{")
        while not self.at("}"):
            n, is_ret = self.node_line(pkg, scope)
            f.nodes.append(n)
            if is_ret:
                f.ret = n
        self.expect("}")
        if f.ret is None:
            raise IrParseError(f"fn {f.name}: no 'ret' node")
        if f.ret.type != ret_t:
            raise IrParseError(f"fn {f.name}: returns {f.ret.type}, declared {ret_t}")
        return f

    def proc(self, pkg: Package) -> Proc:
        p = Proc(name=self.ident())
        self.expect("<")
        chan_types: Dict[str, Tuple[Type, str]] = {}
        order: List[str] = []
        while not self.at(">"):
            name = self.ident()
            self.expect(":")
            t = self.type()
            d = self.ident()
            if d not in ("in", "out"):
                raise IrParseError(f"channel {name}: direction must be in/out, got {d}")
            chan_types[name] = (t, d)
            order.append(name)
            if not self.at(">"):
                self.expect(",")
        self.expect(">")
        self.expect("(")
        states: List[Tuple[str, Type, list]] = []
        inits: List[Any] = []
        while not self.at(")"):
            if self.at("init") and self.peek(1).text == "=":
                self.next(); self.expect("=")
                self.expect("{")
                while not self.at("}"):
                    inits.append(self.raw_value())
                    if not self.at("}"):
                        self.expect(",")
                self.expect("}")
            elif self.at("non_synth") and self.peek(1).text == "=":
                self.next(); self.expect("=")
                self.expect("{")
                while not self.at("}"):
                    p.non_synth.append(self.ident())
                    if not self.at("}"):
                        self.expect(",")
                self.expect("}")
            else:
                name = self.ident()
                self.expect(":")
                t = self.type()
                attrs = self._param_attrs()
                states.append((name, t, attrs.get("pos", [])))
            if not self.at(")"):
                self.expect(",")
        self.expect(")")
        if len(inits) != len(states):
            raise IrParseError(f"proc {p.name}: {len(states)} state elements, {len(inits)} init values")
        for (name, t, pos), raw in zip(states, inits):
            p.state.append(StateElement(name=name, type=t, init=coerce_value(raw, t), pos=pos))
        self.expect("{")
        scope: Dict[str, Node] = {}
        ifaces: Dict[str, ChannelInterface] = {}
        while not self.at("}"):
            if self.at("chan_interface"):
                self.next()
                name = self.ident()
                self.expect("(")
                kv: Dict[str, str] = {}
                while not self.at(")"):
                    k = self.ident(); self.expect("=")
                    kv[k] = self.ident()
                    if not self.at(")"):
                        self.expect(",")
                self.expect(")")
                if name not in chan_types:
                    raise IrParseError(f"chan_interface {name}: not on the proc interface")
                t, d = chan_types[name]
                direction = kv.get("direction")
                if (d == "in") != (direction == "receive"):
                    raise IrParseError(f"chan_interface {name}: direction {direction} vs {d}")
                ifaces[name] = ChannelInterface(
                    name=name, type=t, direction=direction,
                    kind=kv.get("kind", "streaming"),
                    strictness=kv.get("strictness", "proven_mutually_exclusive"),
                    flow_control=kv.get("flow_control", "ready_valid"),
                    flop_kind=kv.get("flop_kind", "none"))
                continue
            n, is_ret = self.node_line(pkg, scope)
            if is_ret:
                raise IrParseError(f"proc {p.name}: 'ret' is not allowed in a proc")
            p.nodes.append(n)
        self.expect("}")
        for name in order:
            if name not in ifaces:
                raise IrParseError(f"proc {p.name}: channel {name} has no chan_interface")
            p.channels.append(ifaces[name])
        return p

    def node_line(self, pkg: Package, scope: Dict[str, Node]) -> Tuple[Node, bool]:
        is_ret = False
        if self.at("ret"):
            self.next()
            is_ret = True
        line = self.peek().line
        name = self.ident()
        self.expect(":")
        t = self.type()
        self.expect("=")
        op = self.ident()
        self.expect("(")
        operands: List[Node] = []
        attrs: Dict[str, Any] = {}
        nid: Optional[int] = None
        pos: List[Tuple[int, int, int]] = []

        def ref(nm: str) -> Node:
            if nm not in scope:
                raise IrParseError(f"line {line}: {nm!r} is used before it is defined")
            return scope[nm]

        while not self.at(")"):
            if self.peek().kind == "ident" and self.peek(1).text == "=":
                key = self.ident()
                self.expect("=")
                if key == "id":
                    nid = self.integer()
                elif key == "pos":
                    pos = self.pos_list()
                elif op == "literal" and key == "value":
                    attrs[key] = coerce_value(self.raw_value(), t)
                elif key in _NODE_ATTRS:
                    if self.at("["):
                        self.next()
                        lst = []
                        while not self.at("]"):
                            lst.append(ref(self.ident()))
                            if not self.at("]"):
                                self.expect(",")
                        self.expect("]")
                        attrs[key] = lst
                    else:
                        attrs[key] = ref(self.ident())
                else:
                    tk = self.next()
                    if tk.kind == "num":
                        attrs[key] = _int(tk.text)
                    elif tk.kind == "str":
                        attrs[key] = bytes(tk.text[1:-1], "utf-8").decode("unicode_escape")
                    elif tk.text in ("true", "false"):
                        attrs[key] = tk.text == "true"
                    elif tk.kind == "ident":
                        attrs[key] = tk.text
                    else:
                        raise IrParseError(f"line {tk.line}: bad value for {key}")
            else:
                if attrs:
                    raise IrParseError(f"line {line}: positional operand after a keyword")
                operands.append(ref(self.ident()))
            if not self.at(")"):
                self.expect(",")
        self.expect(")")
        if name in scope:
            raise IrParseError(f"line {line}: {name!r} is defined twice")
        if nid is None:
            m = re.fullmatch(r"[a-z_]+\.(\d+)", name)
            if m is None:
                raise IrParseError(f"line {line}: node {name!r} has no id")
            nid = int(m.group(1))
        pkg.note_id(nid)
        # A name of the form op.N is the default name; keep it implicit.
        explicit = None if name == f"{op}.{nid}" else name
        n = Node(op=op, type=t, operands=operands, attrs=attrs, id=nid,
                 name=explicit, pos=pos)
        scope[name] = n
        return n, is_ret


def parse_package(text: str) -> Package:
    return _Parser(text).package()
