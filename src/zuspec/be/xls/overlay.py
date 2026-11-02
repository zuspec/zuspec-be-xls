"""XLS results as a Zuspec overlay (``zuspec-ir-core`` ``docs/results.md``).

:func:`xls_overlay` reads what ``codegen_main`` wrote for one block (the
scheduled or optimized IR, the schedule, the module signature and, when
present, the block IR) and records it as facts on the anchors of the Zuspec
model it was lowered from. Every consumer then reads results through the
overlay, never through XLS's files.

**Mapping.** The lowering stamps every XLS node with ``pos`` from the IR
node's ``Loc``, and XLS keeps those positions through optimization: the first
entry is where the operation is written, the rest are the call sites it was
inlined through. A node's anchor is the expression, statement or declaration
starting at its first position (``exact``), else the statement on that line
(``partial``); positions
naming further non-call expressions (operations XLS merged) make it
``merged``. A node with no position that resolves is ``unmapped`` and is
recorded on the block's anchor. XLS optimizes freely, so this is lossy by
design: lost positions only limit what reports can attribute.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import zuspec.ir.core as ir
from zuspec.ir.core.anchor import AnchorTable, KIND_EXPR, KIND_STMT, neutral_root, qualify
from zuspec.ir.core.results import (DERIVED, EXACT, MERGED, PARTIAL, UNMAPPED, Overlay,
                                    OverlayHeader)

from . import xir
from .read_schedule import Schedule, parse_schedules
from .signature import signature_from_codegen
from .testing.parser import parse_package

ENGINE = "xls"

#: ``reg <name>(<type>...)`` in block IR.
_REG_RE = re.compile(r"^\s*reg\s+([A-Za-z_][A-Za-z0-9_]*)\(\s*bits\[(\d+)\]", re.M)
#: A pipeline register: ``p<stage>_<node name>``, holding the node's value at
#: the end of ``<stage>``.
_PIPE_REG_RE = re.compile(r"^p(\d+)_(.+)$")


class _Resolver:
    """XLS ``pos`` lists -> (anchors, quality)."""

    def __init__(self, anchors: AnchorTable, files: Dict[int, str], fallback: str):
        self.t = anchors
        self.files = {k: os.path.abspath(v) if not v.startswith("<") else v
                      for k, v in files.items()}
        self.fallback = fallback

    def _at(self, p: xir.Pos) -> Tuple[Optional[str], bool]:
        """The anchor a position names, and whether it is exact."""
        f = self.files.get(p[0])
        here = self.t.by_loc((f, p[1] + 1, p[2] + 1))
        exprs = [a for a in here if self.t[a].kind == KIND_EXPR]
        if exprs:
            return exprs[0], True          # the outermost expression starting there
        stmts = [a for a in here if self.t[a].kind == KIND_STMT]
        if stmts:
            return stmts[-1], True         # the innermost statement starting there
        if here:
            return here[-1], True          # a declaration: the function, a member
        stmts = self.t.by_line(f, p[1] + 1)
        return (stmts[0], False) if stmts else (None, False)

    def _is_call(self, anchor: str) -> bool:
        n = self.t.node_of(anchor)
        return isinstance(n, ir.ExprCall)

    def resolve(self, pos: List[xir.Pos]) -> Tuple[List[str], str]:
        if not pos:
            return [self.fallback], UNMAPPED
        first, exact = self._at(pos[0])
        if first is None:
            return [self.fallback], UNMAPPED
        anchors = [first]
        for p in pos[1:]:
            a, ex = self._at(p)
            if a is not None and ex and not self._is_call(a) and \
                    self.t.node_of(a) is not None and a not in anchors:
                anchors.append(a)
        if len(anchors) > 1:
            return anchors, MERGED
        return anchors, EXACT if exact else PARTIAL


def _body(pkg: xir.Package, name: Optional[str]):
    for b in list(pkg.functions) + list(pkg.procs):
        if name is None and b.name == pkg.top or b.name == name:
            return b
    bodies = list(pkg.functions) + list(pkg.procs)
    if name is None and len(bodies) == 1:
        return bodies[0]
    raise ValueError(f"no function or proc {name or pkg.top!r} in the IR")


def xls_overlay(anchors: AnchorTable, *, ir_text: str, schedule_text: Optional[str] = None,
                signature_text: Optional[str] = None, block_ir_text: Optional[str] = None,
                top: Optional[str] = None, block_anchor: Optional[str] = None,
                constraints: Optional[Dict[str, Any]] = None, command: Optional[str] = None,
                engine_version: Optional[str] = None,
                artifacts: Optional[Dict[str, str]] = None) -> Overlay:
    """The overlay for one XLS block.

    *ir_text* is the scheduled IR (``--output_schedule_ir_path``) or the
    optimized IR; *top* names the function or proc in it (default: the package
    top). *block_anchor* is the anchor of the component or function the block
    was built from (default: the anchor named after *top*, if any).
    """
    pkg = parse_package(ir_text)
    body = _body(pkg, top)
    top = body.name
    if block_anchor is None:
        r = neutral_root(top)
        block_anchor = r if r in anchors else next(
            (a for a, e in anchors.entries.items() if e.kind == "type"), r)
    res = _Resolver(anchors, pkg.files, block_anchor)
    sched: Optional[Schedule] = None
    if schedule_text:
        sched = parse_schedules(schedule_text).get(top)
    sig = signature_from_codegen(signature_text) if signature_text else None

    o = Overlay(OverlayHeader(engine=ENGINE, engine_version=engine_version,
                              constraints=dict(constraints or {}), command=command,
                              scope_hashes=dict(anchors.scope_hashes),
                              artifacts=dict(artifacts or {})))

    # -- per node: anchor, stage, delays ----------------------------------------
    by_name: Dict[str, xir.Node] = {}
    node_anchor: Dict[str, Tuple[List[str], str]] = {}
    counts = {q: 0 for q in (EXACT, MERGED, PARTIAL, UNMAPPED)}
    order = {name: i for i, name in enumerate(sched.nodes)} if sched else {}
    for n in body.nodes:
        by_name[n.ref] = n
        if n.op == "param":
            continue
        anchors_q = res.resolve(n.pos)
        node_anchor[n.ref] = anchors_q
        counts[anchors_q[1]] += 1
        tn = sched.nodes.get(n.ref) if sched else None
        for a in anchors_q[0]:
            q = anchors_q[1]
            o.add("xls.node_name", a, n.ref, q)
            if tn is not None:
                o.add("sched.stage", a, tn.stage, q)
                # Node and stage together, for readers that list the schedule
                # by XLS node (fw.hdl StageTLM); codegen adds nodes the
                # schedule does not have, so the two columns do not pair up.
                o.add("xls.node_stage", a, {"node": n.ref, "stage": tn.stage,
                                             "order": order[n.ref]}, q)
                if tn.node_delay_ps is not None:
                    o.add("timing.node_delay_ps", a, tn.node_delay_ps, q)
                if tn.path_delay_ps is not None:
                    o.add("timing.path_delay_ps", a, tn.path_delay_ps, q)

    # Scheduled nodes with no anchor of their own (a function's parameters,
    # which are not body nodes; a proc's params): their stage is still part of
    # the schedule a reader lists by node.
    for name, tn in (sched.nodes.items() if sched else ()):
        if name not in node_anchor:
            o.add("xls.node_stage", block_anchor,
                  {"node": name, "stage": tn.stage, "order": order[name]}, UNMAPPED)

    # -- block-level facts ------------------------------------------------------------
    crit_total = crit_attr = 0
    if sched is not None:
        o.add("sched.stage_count", block_anchor, max(sched.length, 1))
        period = sched.target_clock_period_ps or sched.min_clock_period_ps
        if period:
            o.add("timing.clock_period_ps", block_anchor, period)
        for stage, names in sorted(sched.stages.items()):
            path, total, attr = _critical_path(stage, names, sched, by_name, node_anchor)
            crit_total += total
            crit_attr += attr
            if path is not None:
                o.add("timing.critical_path", block_anchor, path)
    if sig is not None:
        _channel_stages(sig, body, sched)
        o.add("iface.block", block_anchor, ir.signature_to_json(sig))
        o.add("sched.ii", block_anchor, {"value": sig.initiation_interval or 1,
                                         "reliable": not isinstance(body, xir.Proc)})
        for ch in sig.channels:
            ca = f"{block_anchor}.{ch.name}"
            if ca in anchors:
                if ch.stage is not None:
                    o.add("iface.channel_stage", ca, ch.stage)
                if ch.flop_kind is not None:
                    o.add("iface.channel_flop", ca, ch.flop_kind)
    if block_ir_text:
        module = sig.name if sig is not None else top
        # XLS names a pipeline register after the node, made an identifier
        # (`bit_slice.2117` -> `p0_bit_slice_2117`).
        by_ident = {re.sub(r"\W", "_", ref): ref for ref in node_anchor}
        for name, width in _REG_RE.findall(block_ir_text):
            m = _PIPE_REG_RE.match(name)
            ref = by_ident.get(m.group(2)) if m is not None else None
            if ref is None:
                continue
            a_list, q = node_anchor[ref]
            if q == UNMAPPED:
                continue
            stage = int(m.group(1))
            o.add("rtl.register", qualify(a_list[0], "reg", stage),
                  {"module": module, "name": name, "width": int(width), "stage": stage}, DERIVED)

    total = sum(counts.values())
    o.header.coverage = {
        "ops_total": total,
        "ops_by_quality": counts,
        "pct_ops_attributed": round(100.0 * (counts[EXACT] + counts[MERGED]) / total, 1)
        if total else None,
        "pct_crit_delay_attributed": round(100.0 * crit_attr / crit_total, 1)
        if crit_total else None,
    }
    return o


def _critical_path(stage: int, names: List[str], sched: Schedule,
                   by_name: Dict[str, xir.Node], node_anchor):
    """The stage's longest path, end to start: (fact value, delay, attributed delay)."""
    timed = [sched.nodes[n] for n in names if sched.nodes[n].path_delay_ps is not None]
    if not timed:
        return None, 0, 0
    in_stage = {n for n in names}
    cur = max(timed, key=lambda t: t.path_delay_ps)
    chain = []
    while cur is not None:
        chain.append(cur.node)
        node = by_name.get(cur.node)
        want = (cur.path_delay_ps or 0) - (cur.node_delay_ps or 0)
        nxt = None
        if node is not None and want > 0:
            for op in node.all_operands():
                t = sched.nodes.get(op.ref)
                if op.ref in in_stage and t is not None and t.path_delay_ps == want:
                    nxt = t
                    break
        cur = nxt
    chain.reverse()
    total = attr = 0
    path: List[str] = []
    for name in chain:
        d = sched.nodes[name].node_delay_ps or 0
        total += d
        a = node_anchor.get(name)
        if a is not None and a[1] in (EXACT, MERGED):
            attr += d
        if a is not None and a[1] != UNMAPPED and (not path or path[-1] != a[0][0]):
            path.append(a[0][0])
    end = max(timed, key=lambda t: t.path_delay_ps)
    return {"stage": stage, "anchors": path, "delay_ps": end.path_delay_ps}, total, attr


def _channel_stages(sig: ir.BlockSignature, body, sched: Optional[Schedule]) -> None:
    """Fill each channel's stage from where its send or receive is scheduled,
    when the signature does not give it."""
    if sched is None:
        return
    for ch in sig.channels:
        if ch.stage is not None:
            continue
        stages = [sched.nodes[n.ref].stage for n in body.nodes
                  if n.op in ("send", "receive") and n.attrs.get("channel") == ch.name
                  and n.ref in sched.nodes]
        if stages:
            ch.stage = min(stages)
