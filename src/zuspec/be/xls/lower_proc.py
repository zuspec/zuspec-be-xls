"""Lower a component's ``run()`` to an XLS proc (``xls-phase0.md`` §3.4, BX-4).

The input shape (P1)::

    run():  <prologue>  while True: <B>

* **One iteration of ``B`` is one activation.**
* **State** (P2) is every non-port field, plus every local the prologue
  declares (it outlives the prologue, so it persists). Each becomes a state
  element with one ``state_read`` at the top of the activation and **one
  unpredicated ``next_value``** at the end, its value joined through ``sel``
  (D-5).
* **The initial state** (P4) is the field's ``initial_value`` (or 0), then
  updated by the prologue. The prologue must fold to constants.
* **Locals declared inside ``B``** (``StmtAnnAssign``) are temporaries (P3):
  SSA values that start at 0 every activation and are never state (D-8).
* **Channels** (P9) are the port fields typed ``DataTypeGetIF`` (receive) and
  ``DataTypePutIF`` (send). The channel ops have one canonical form (R6, D-9):

  - get: ``StmtAssign([x], ExprAwait(ExprCall(ExprAttribute(<port>, "get"), [])))``
  - put: ``StmtExpr(ExprAwait(ExprCall(ExprAttribute(<port>, "put"), [v])))``
  - try_get (GAP-3): ``StmtAssign([v], ExprAwait(ExprCall(ExprAttribute(<port>,
    "try_get"), [x])))`` -- a non-blocking receive; ``v`` is its valid bit and
    ``x`` keeps its value when nothing was waiting.

  Each op takes the path predicate (P7) and the effect token, threaded in
  program order (P8, D-6).
* **Rejected in X0** (P10): a channel op inside a loop, two ops on one
  channel in an activation, any other awaited call.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import zuspec.ir.core as ir

from . import xir
from .lower_fn import (BodyLowerer, FnInfo, Key, lower_functions, pos_of, var_key, xtype,
                       zero_of)
from .width import IntT, StaticSubsetError, T, from_ir


class _PrologueLowerer(BodyLowerer):
    """Evaluates the prologue to constants: it may not touch channels."""


class ProcLowerer(BodyLowerer):
    def __init__(self, pkg: xir.Package, proc: xir.Proc, types: Dict[Key, T],
                 names: Dict[Key, str], funcs: Dict[str, FnInfo],
                 channels: Dict[int, xir.ChannelInterface]):
        super().__init__(pkg, proc, types, names, funcs)
        self.channels = channels             # field index -> channel
        self.used: Dict[str, ir.Stmt] = {}   # channel name -> first op
        self.loop_depth = 0
        for c in channels.values():
            self._used_names.add(c.name)

    # -- loops: count depth for P10 -------------------------------------
    def while_(self, s):
        self.loop_depth += 1
        try:
            super().while_(s)
        finally:
            self.loop_depth -= 1

    def repeat(self, s):
        self.loop_depth += 1
        try:
            super().repeat(s)
        finally:
            self.loop_depth -= 1

    # -- channel ops ----------------------------------------------------
    def _port_call(self, e: ir.ExprAwait, s: ir.Stmt) -> Tuple[xir.ChannelInterface, str, ir.ExprCall]:
        call = e.value
        if not (isinstance(call, ir.ExprCall) and isinstance(call.func, ir.ExprAttribute)):
            raise self.error("only port get/put may be awaited in an XLS proc", s)
        key = var_key(call.func.value)
        if key is None or key[0] != "field" or key[1] not in self.channels:
            what = call.func.attr
            raise self.error(f"awaited call {what!r} is not a get/put on a channel port "
                             f"(XLS procs have no time-consuming calls)", s)
        ch = self.channels[key[1]]
        method = call.func.attr
        if self.loop_depth:
            raise self.error(f"channel operation on {ch.name!r} inside a loop: in X0 each "
                             f"channel op runs at most once per activation. Move the loop "
                             f"into state (one iteration per activation)", s)
        if ch.name in self.used:
            first = self.used[ch.name]
            where = f" (first at line {first.loc.line})" if first.loc else ""
            raise self.error(f"second operation on channel {ch.name!r} in one activation"
                             f"{where}; X0 allows one", s)
        self.used[ch.name] = s
        return ch, method, call

    def _tok(self) -> xir.Node:
        if self.flow.tok is None:
            self.flow.tok = self.b.after_all()
        return self.flow.tok

    def await_assign(self, tgt: ir.Expr, e: ir.ExprAwait, s: ir.Stmt) -> None:
        ch, method, call = self._port_call(e, s)
        if method == "try_get" and ch.direction == "receive" and len(call.args) == 1:
            # recv_non_blocking: x = valid ? data : x
            x = call.args[0]
            recv = self.b.receive(self._tok(), ch, predicate=self.pred(), blocking=False)
            self.flow.tok = self.b.tuple_index(recv, 0)
            data = self.b.tuple_index(recv, 1)
            valid = self.b.tuple_index(recv, 2)
            et = from_ir(self._elem_type(ch))
            old, ot = self.expr(x)
            old = self.coerce(old, ot, et, x, f"the value try_get on {ch.name} keeps")
            self.assign(x, self.b.sel(valid, [old, data]), et, x)
            self.assign(tgt, valid, from_ir(ir.DataTypeInt(bits=1, signed=False)), tgt)
            return
        if method != "get" or ch.direction != "receive" or call.args:
            raise self.error(f"expected `x = await {ch.name}.get()` on an input port", s)
        recv = self.b.receive(self._tok(), ch, predicate=self.pred())
        self.flow.tok = self.b.tuple_index(recv, 0)
        data = self.b.tuple_index(recv, 1)
        self.assign(tgt, data, from_ir(self._elem_type(ch)), tgt)

    def await_stmt(self, e: ir.ExprAwait, s: ir.Stmt) -> None:
        ch, method, call = self._port_call(e, s)
        if method == "get":
            raise self.error(f"a get on {ch.name!r} must assign its result: "
                             f"`x = await {ch.name}.get()`", s)
        if method != "put" or ch.direction != "send" or len(call.args) != 1:
            raise self.error(f"expected `await {ch.name}.put(v)` on an output port", s)
        n, t = self.expr(call.args[0])
        n = self.coerce(n, t, from_ir(self._elem_type(ch)), call.args[0], f"data sent on {ch.name}")
        self.flow.tok = self.b.send(self._tok(), n, ch, predicate=self.pred())

    def _elem_type(self, ch: xir.ChannelInterface) -> ir.DataType:
        return self._ch_ir_types[ch.name]


def _split_run(run: ir.Function) -> Tuple[List[ir.Stmt], ir.StmtWhile]:
    body = list(run.body)
    if not body or not isinstance(body[-1], ir.StmtWhile) or not (
            isinstance(body[-1].test, ir.ExprConstant) and body[-1].test.value is True):
        raise StaticSubsetError(
            "run() must end in `forever` (one iteration = one activation, P1)", run.loc)
    for s in body[:-1]:
        if isinstance(s, ir.StmtWhile):
            raise StaticSubsetError("a loop before the forever is not supported", s.loc)
    return body[:-1], body[-1]


def _find_run(comp: ir.DataTypeComponent) -> ir.Function:
    procs = list(getattr(comp, "proc_processes", []) or [])
    for f in procs + list(comp.functions):
        if isinstance(f, ir.Function) and f.name == "run":
            return f
    if len(procs) == 1:
        return procs[0]
    raise StaticSubsetError(f"component {comp.name!r} has no run()", comp.loc)


def lower_component(comp: ir.DataTypeComponent, pkg: Optional[xir.Package] = None,
                    proc_name: Optional[str] = None) -> xir.Package:
    """Lower *comp* (pure functions + ``run()``) to an XLS package with one top proc."""
    pkg = pkg if pkg is not None else xir.Package(name=comp.name or "top")
    fns = [f for f in comp.functions if isinstance(f, ir.Function)
           and f.name != "run" and not f.is_async]
    funcs = lower_functions(pkg, fns)
    run = _find_run(comp)
    prologue, forever = _split_run(run)

    # Fields: channels, constants, state.
    types: Dict[Key, T] = {}
    names: Dict[Key, str] = {}
    channels: Dict[int, xir.ChannelInterface] = {}
    ch_types: Dict[str, ir.DataType] = {}
    state_keys: List[Key] = []
    inits: Dict[Key, xir.Value] = {}
    for i, f in enumerate(comp.fields):
        if f.kind == ir.FieldKind.Port or isinstance(f.datatype, (ir.DataTypeGetIF, ir.DataTypePutIF)):
            dt = f.datatype
            if not isinstance(dt, (ir.DataTypeGetIF, ir.DataTypePutIF)) or dt.element_type is None:
                raise StaticSubsetError(
                    f"port {f.name!r} must be typed DataTypeGetIF/DataTypePutIF with an "
                    f"element type (R7)", f.loc)
            direction = "receive" if isinstance(dt, ir.DataTypeGetIF) else "send"
            channels[i] = xir.ChannelInterface(name=f.name, type=xtype(from_ir(dt.element_type, f.loc)),
                                               direction=direction)
            ch_types[f.name] = dt.element_type
            continue
        key: Key = ("field", i)
        types[key] = from_ir(f.datatype, f.loc)
        names[key] = f.name
        state_keys.append(key)

    # Initial state: field initializers, then the prologue (P4).
    scratch = xir.Package(name="__prologue")
    pl = _PrologueLowerer(scratch, xir.Function(name="__prologue"), types, names, funcs)
    for key in state_keys:
        f = comp.fields[key[1]]
        if f.initial_value is not None:
            n, t = pl.expr(f.initial_value)
            n = pl.coerce(n, t, types[key], f.initial_value, f"initializer of {f.name}")
        else:
            n = pl.b.literal(zero_of(types[key]), xtype(types[key]))
        pl.flow.env[key] = n
    pl.stmts(prologue)
    for key in pl.types:
        if key[0] == "local" and key not in types:   # a prologue local: state
            types[key] = pl.types[key]
            names[key] = pl.names[key]
            state_keys.append(key)
    for key in state_keys:
        n = pl.flow.env[key]
        v = pl.lit_value(n)
        if v is None:
            raise StaticSubsetError(
                f"initial value of {names[key]!r} is not a constant (the prologue must "
                f"fold to constants)", forever.loc)
        inits[key] = v

    # The proc.
    proc = xir.Proc(name=proc_name or comp.name or "top")
    for i in sorted(channels):
        proc.channels.append(channels[i])
    pl2 = ProcLowerer(pkg, proc, types, names, funcs, channels)
    pl2._ch_ir_types = ch_types
    elements: Dict[Key, xir.StateElement] = {}
    for key in state_keys:
        st_name = pl2.unique_name(names[key])
        # A state element sits on its declaration: the field, or (a prologue
        # local) the forever it outlives.
        decl = comp.fields[key[1]].loc if key[0] == "field" else None
        st = xir.StateElement(name=st_name, type=xtype(types[key]), init=inits[key],
                              pos=pos_of(pkg, decl or forever.loc))
        proc.state.append(st)
        elements[key] = st
    for key in state_keys:
        pl2.b.pos = list(elements[key].pos)
        pl2.flow.env[key] = pl2.b.state_read(elements[key], name=elements[key].name)
    pl2.set_loc(forever.loc)
    pl2.stmts(forever.body)
    for key in state_keys:
        # The update sits where its value was computed (the last assignment,
        # or the join of several); an unchanged element, on its declaration.
        st, v = elements[key], pl2.flow.env[key]
        pl2.b.pos = list(v.pos if v.pos and v.op != "state_read" else st.pos)
        pl2.b.next_value(st, v)
    pl2.b.pos = []
    xir.remove_dead_nodes(proc)
    pkg.procs.append(proc)
    pkg.top = proc.name
    return pkg
