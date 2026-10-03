"""Print the :mod:`xir` object model as XLS IR text.

The syntax follows ``ir_converter_main`` output from the pinned release
(:data:`xir.XLS_SYNTAX_VERSION`):

* ``fn name(a: bits[8] id=1) -> bits[8] { ... }``
* ``proc name<ch: T in, ch2: T out>(st: T, init={v}) { chan_interface ...; ... }``
  -- proc-scoped channels, explicit ``state_read`` / ``next_value``.
* ``name: T = op(operands, key=value, ..., id=N, pos=[(file,line,col)])``
"""
from __future__ import annotations

from typing import Any, List, Union

from .xir import (Function, Node, Package, Proc, format_value, ir_ident)

#: String attrs that print quoted. Every other string attr is a bare name
#: (a channel, state element or function).
_QUOTED = {"message", "label"}


def _fmt_pos(pos) -> str:
    return "[" + ", ".join(f"({f},{l},{c})" for f, l, c in pos) + "]"


def _fmt_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _fmt_attr(node: Node, key: str, v: Any) -> str:
    if node.op == "literal" and key == "value":
        return format_value(v, node.type)
    if isinstance(v, Node):
        return v.ref
    if isinstance(v, list):
        return "[" + ", ".join(e.ref if isinstance(e, Node) else str(e) for e in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return _fmt_str(v) if key in _QUOTED else ir_ident(v)
    raise TypeError(f"cannot print attr {key}={v!r} on {node.ref}")


def format_node(n: Node, is_ret: bool = False) -> str:
    args: List[str] = [o.ref for o in n.operands]
    for k, v in n.attrs.items():
        args.append(f"{k}={_fmt_attr(n, k, v)}")
    args.append(f"id={n.id}")
    if n.pos:
        args.append(f"pos={_fmt_pos(n.pos)}")
    prefix = "ret " if is_ret else ""
    return f"{prefix}{n.ref}: {n.type} = {n.op}({', '.join(args)})"


def format_function(f: Function, top: bool = False) -> str:
    params = ", ".join(f"{p.ref}: {p.type} id={p.id}" for p in f.params)
    assert f.ret is not None, f"function {f.name} has no return value"
    lines = [f"{'top ' if top else ''}fn {ir_ident(f.name)}({params}) -> {f.ret.type} {{"]
    for n in f.nodes:
        lines.append("  " + format_node(n, is_ret=(n is f.ret)))
    lines.append("}")
    return "\n".join(lines)


def format_proc(p: Proc, top: bool = False) -> str:
    chans = ", ".join(
        f"{ir_ident(c.name)}: {c.type} {'in' if c.direction == 'receive' else 'out'}"
        for c in p.channels)
    state = []
    for s in p.state:
        txt = f"{ir_ident(s.name)}: {s.type}"
        if s.pos:
            txt += f" pos={_fmt_pos(s.pos)}"
        state.append(txt)
    if p.state:
        inits = ", ".join(format_value(s.init, s.type) for s in p.state)
        state.append(f"init={{{inits}}}")
    if p.non_synth:
        state.append(f"non_synth={{{', '.join(p.non_synth)}}}")
    lines = [f"{'top ' if top else ''}proc {ir_ident(p.name)}<{chans}>({', '.join(state)}) {{"]
    for c in p.channels:
        lines.append(
            f"  chan_interface {ir_ident(c.name)}(direction={c.direction}, kind={c.kind}, "
            f"strictness={c.strictness}, flow_control={c.flow_control}, "
            f"flop_kind={c.flop_kind})")
    for n in p.nodes:
        lines.append("  " + format_node(n))
    lines.append("}")
    return "\n".join(lines)


def format_package(pkg: Package) -> str:
    out = [f"package {ir_ident(pkg.name)}", ""]
    if pkg.files:
        for num in sorted(pkg.files):
            out.append(f"file_number {num} {_fmt_str(pkg.files[num])}")
        out.append("")
    members: List[str] = []
    for f in pkg.functions:
        members.append(format_function(f, top=(f.name == pkg.top)))
    for p in pkg.procs:
        members.append(format_proc(p, top=(p.name == pkg.top)))
    out.append("\n\n".join(members))
    return "\n".join(out) + "\n"
