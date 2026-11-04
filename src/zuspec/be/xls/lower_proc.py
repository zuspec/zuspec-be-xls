"""Lower a component's ``run()`` to an XLS proc (``xls-phase0.md`` §3.4, BX-4).

The input shape (P1)::

    run():  <prologue>  while True: <B>

* **One iteration of ``B`` is one activation.**
* **State** (P2) is every non-port field, plus every local the prologue
  declares (it outlives the prologue, so it persists). Each becomes a state
  element with one ``state_read`` at the top of the activation and **one
  ``next_value``** at the end, its value joined through ``sel`` (D-5). An
  element read and written only under guards has its ``state_read``
  predicated on the OR of those guards and its ``next_value`` on the OR of its writes'.
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
  program order (P8, D-6), except that effects on exclusive paths are not
  ordered against each other. Channel ops in a constant-bound loop unroll.
* **Several ops on one channel** per activation: allowed when their
  guards are exclusive (``proven_mutually_exclusive``; XLS checks it), or when
  nothing orders one after the other (``runtime_mutually_exclusive``; XLS
  asserts in simulation that at most one fires).
* **Rejected** (P10): two ops on one channel where one is unguarded, or that
  can both fire with another effect between them; two ops on a level; any
  other awaited call.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import zuspec.ir.core as ir

from . import xir
from .lower_fn import (BodyLowerer, FnInfo, Key, lower_functions, pos_of, src_path, var_key,
                       xtype, zero_of)
from .width import IntT, StaticSubsetError, T, from_ir


class _PrologueLowerer(BodyLowerer):
    """Evaluates the prologue to constants: it may not touch channels."""


class _Effect:
    """One effect of the activation: its statement, channel (None for an
    assertion), path conditions, output token, and the earlier effects it
    is ordered after (transitively)."""

    def __init__(self, s: ir.Stmt, ch: Optional[str], conds: List[xir.Node]):
        self.s, self.ch, self.conds = s, ch, conds
        self.tok: Optional[xir.Node] = None
        self.after: set = set()


class ProcLowerer(BodyLowerer):
    def __init__(self, pkg: xir.Package, proc: xir.Proc, types: Dict[Key, T],
                 names: Dict[Key, str], funcs: Dict[str, FnInfo],
                 channels: Dict[int, xir.ChannelInterface]):
        super().__init__(pkg, proc, types, names, funcs)
        self.channels = channels             # field index -> channel
        self.effects: List[_Effect] = []     # in program order
        # A cycle step: one activation is one atomic cycle, so effects
        # on different channels are not ordered at all.
        self.step = False
        # State reads and writes, for the state_read predicate:
        # per state key, the nodes whose value is still the start-of-activation
        # value, and the path predicates of each read and write.
        self.state_dep: Dict[Key, set] = {}
        self.reads: Dict[Key, List[Optional[xir.Node]]] = {}
        self.writes: Dict[Key, List[Optional[xir.Node]]] = {}
        for c in channels.values():
            self._used_names.add(c.name)

    # -- state reads and writes -----------------------------------------
    def _expr(self, e: ir.Expr):
        key = var_key(e)
        if key in self.state_dep and key in self.flow.env and \
                id(self.flow.env[key]) in self.state_dep[key]:
            self.reads[key].append(self.pred())
        return super()._expr(e)

    def _store(self, tgt: ir.Expr, n: xir.Node) -> None:
        key = var_key(tgt)
        if key in self.state_dep:
            self.writes[key].append(self.pred())
        super()._store(tgt, n)

    def mux(self, c: xir.Node, if_true: xir.Node, if_false: xir.Node) -> xir.Node:
        m = super().mux(c, if_true, if_false)
        for dep in self.state_dep.values():
            if id(if_true) in dep or id(if_false) in dep:
                dep.add(id(m))
        return m

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
        if ch.kind == "single_value":
            first = next((x.s for x in self.effects if x.ch == ch.name), None)
            if first is not None:
                where = f" (first at line {first.loc.line})" if first.loc else ""
                raise self.error(f"second operation on level {ch.name!r} in one activation"
                                 f"{where}; a level takes one", s)
        return ch, call.func.attr, call

    # -- the token: effects in program order, except where exclusive -----
    def _begin(self, s: ir.Stmt, ch: Optional[xir.ChannelInterface]) -> Tuple[_Effect, xir.Node]:
        """A new effect and the token it takes. It is ordered after each
        earlier effect whose guard can hold with its own (D-6); effects on
        exclusive paths are not ordered, so XLS can merge operations on one
        channel into one per activation. An earlier operation on the same
        channel whose guard can also hold is left unordered: both firing is
        XLS's runtime assertion."""
        x = _Effect(s, ch.name if ch is not None else None, list(self.flow.conds))
        direct, same = [], []
        for i, y in enumerate(self.effects):
            if self.step and (x.ch is None or y.ch != x.ch):
                continue
            if self.disjoint(x.conds, y.conds):
                continue
            (same if x.ch is not None and y.ch == x.ch else direct).append(i)
        for i in direct:
            x.after |= {i} | self.effects[i].after
        for i in same:
            y = self.effects[i]
            line = f" at line {y.s.loc.line}" if y.s.loc else ""
            if not x.conds or not y.conds:
                raise self.error(
                    f"second operation on channel {x.ch!r} in one activation (the first"
                    f"{line}): one of them is unguarded, so both run. A channel carries "
                    f"one item per activation", s)
            if i in x.after:
                raise self.error(
                    f"operations on channel {x.ch!r} here and{line} can both run in one "
                    f"activation, and another effect lies between them. Make the guards "
                    f"exclusive (if/else, case, or v == k for different constants k), or "
                    f"make the operations adjacent", s)
            ch.strictness = "runtime_mutually_exclusive"
        # only the latest of each chain: the rest are implied
        toks = [self.effects[i].tok for i in direct
                if not any(i in self.effects[j].after for j in direct)]
        tok = toks[0] if len(toks) == 1 else self.b.after_all(*toks)
        return x, tok

    def _end(self, x: _Effect, tok: xir.Node) -> None:
        x.tok = tok
        self.effects.append(x)

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
        x, tok = self._begin(s, None)
        self._end(x, self.b.assert_(tok, ok, msg))

    # -- exclusive guards ---------------------------------------
    def _same(self, a: xir.Node, b: xir.Node, depth: int = 6) -> bool:
        """Structurally the same value (the same node, or the same op on the
        same operands): an unrolled loop rebuilds ``x == i`` each iteration."""
        if a is b:
            return True
        if depth == 0 or a.op != b.op or a.type != b.type or a.op in xir.EFFECT_OPS \
                or len(a.operands) != len(b.operands):
            return False
        plain = lambda n: {k: v for k, v in n.attrs.items()  # noqa: E731
                           if not isinstance(v, (xir.Node, list))}
        return plain(a) == plain(b) and \
            all(self._same(x, y, depth - 1) for x, y in zip(a.operands, b.operands))

    def _eq_const(self, c: xir.Node):
        """(x, k) for a condition ``x == k`` with k a constant."""
        if c.op != "eq" or len(c.operands) != 2:
            return None
        x, y = c.operands
        for u, v in ((x, y), (y, x)):
            k = self.lit_value(v)
            if k is not None and self.lit_value(u) is None:
                return u, k
        return None

    def _excl(self, a: xir.Node, b: xir.Node) -> bool:
        if a.op == "not" and self._same(a.operands[0], b):
            return True
        if b.op == "not" and self._same(b.operands[0], a):
            return True
        ea, eb = self._eq_const(a), self._eq_const(b)
        return ea is not None and eb is not None and ea[1] != eb[1] and \
            self._same(ea[0], eb[0])

    def disjoint(self, xs: List[xir.Node], ys: List[xir.Node]) -> bool:
        """Two paths' conditions cannot both hold: opposite arms of one
        ``if``/``case``, or ``v == j`` and ``v == k`` with j != k."""
        return any(self._excl(a, b) for a in xs for b in ys)

    def await_assign(self, tgt: ir.Expr, e: ir.ExprAwait, s: ir.Stmt) -> None:
        ch, method, call = self._port_call(e, s)
        if method == "try_get" and ch.direction == "receive" and len(call.args) == 1:
            # recv_non_blocking: x = valid ? data : x
            x = call.args[0]
            ef, t0 = self._begin(s, ch)
            recv = self.b.receive(t0, ch, predicate=self.pred(), blocking=False)
            self._end(ef, self.b.tuple_index(recv, 0))
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
        ef, t0 = self._begin(s, ch)
        recv = self.b.receive(t0, ch, predicate=self.pred())
        self._end(ef, self.b.tuple_index(recv, 0))
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
        ef, t0 = self._begin(s, ch)
        self._end(ef, self.b.send(t0, n, ch, predicate=self.pred()))

    def _elem_type(self, ch: xir.ChannelInterface) -> ir.DataType:
        return self._ch_ir_types[ch.name]


def _guard(pl: ProcLowerer, preds: List[Optional[xir.Node]]) -> Optional[xir.Node]:
    """The OR of path predicates; None if any is unguarded (or there are none)."""
    if not preds or any(p is None for p in preds):
        return None
    uniq = list({id(p): p for p in preds}.values())
    return uniq[0] if len(uniq) == 1 else pl.fold(pl.b.nary("or", *uniq))


def _depends_on(n: xir.Node, target: xir.Node) -> bool:
    seen, work = set(), [n]
    while work:
        x = work.pop()
        if x is target:
            return True
        if id(x) not in seen:
            seen.add(id(x))
            work.extend(x.all_operands())
    return False


def _topo_sort(proc: xir.Proc) -> None:
    """Every node after its operands (a state_read predicate is built after
    the read), otherwise in creation order."""
    pos = {id(n): i for i, n in enumerate(proc.nodes)}
    done, out = set(), []

    def visit(n: xir.Node) -> None:
        stack = [(n, False)]
        while stack:
            x, expanded = stack.pop()
            if id(x) in done:
                continue
            if expanded:
                done.add(id(x))
                out.append(x)
                continue
            stack.append((x, True))
            for o in sorted(x.all_operands(), key=lambda o: -pos.get(id(o), -1)):
                if id(o) not in done:
                    stack.append((o, False))

    for n in proc.nodes:
        visit(n)
    proc.nodes = out


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
            ch = xir.ChannelInterface(name=f.name, type=xtype(from_ir(dt.element_type, f.loc)),
                                      direction=direction)
            if f.pragmas.get("fw_flow") == "level":
                # A level: a single_value channel (data only; XLS registers it).
                ch.kind, ch.strictness, ch.flow_control = "single_value", None, "none"
            channels[i] = ch
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
    pl2.step = bool(run.metadata.get("fw_step"))
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
    reads: Dict[Key, xir.Node] = {}
    for key in state_keys:
        pl2.b.pos = list(elements[key].pos)
        reads[key] = pl2.flow.env[key] = pl2.b.state_read(elements[key], name=elements[key].name)
        pl2.state_dep[key] = {id(reads[key])}
        pl2.reads[key], pl2.writes[key] = [], []
    pl2.set_loc(forever.loc)
    pl2.stmts(forever.body)
    for key in state_keys:
        # The update sits where its value was computed (the last assignment,
        # or the join of several); an unchanged element, on its declaration.
        st, v = elements[key], pl2.flow.env[key]
        pl2.b.pos = list(v.pos if v.pos and v.op != "state_read" else st.pos)
        # XLS asserts that a next_value never fires while its read is off,
        # so the read's guard covers the writes too.
        rd = _guard(pl2, pl2.reads[key] + pl2.writes[key])
        if rd is not None and not _depends_on(rd, reads[key]):
            # Used only on a guard: the read takes it, and the update fires
            # only where the element is written, so an activation that does
            # not use it never sees its value.
            reads[key].attrs["predicate"] = rd
            if pl2.writes[key]:
                pl2.b.next_value(st, v, predicate=_guard(pl2, pl2.writes[key]))
        else:
            pl2.b.next_value(st, v)
    pl2.b.pos = []
    xir.remove_dead_nodes(proc)
    _topo_sort(proc)
    pkg.procs.append(proc)
    pkg.top = proc.name
    return pkg


def codegen_flags(comp: ir.DataTypeComponent, pkg: xir.Package) -> List[str]:
    """Codegen flags that *comp*'s proc in *pkg* needs to keep its meaning.

    A cycle step is one cycle: an activation takes the requests offered in
    that cycle. In a pipeline, XLS may schedule the receives in different
    stages, so one activation would take requests offered cycles apart, and
    a request could miss the effect of one whose response has already gone.
    So a step's streaming receives are kept in one stage."""
    run = _find_run(comp)
    if not run.metadata.get("fw_step"):
        return []
    proc = next((p for p in pkg.procs if p.name == pkg.top), None) or \
        (pkg.procs[0] if pkg.procs else None)
    if proc is None:
        return []
    recvs = [xir.ir_ident(c.name) for c in proc.channels
             if c.direction == "receive" and c.kind == "streaming"]
    if len(recvs) < 2:
        return []
    return ["--io_constraints=" + ",".join(f"{recvs[0]}:recv:{c}:recv:0:0"
                                           for c in recvs[1:])]
