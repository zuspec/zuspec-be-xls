"""XLS results as a Zuspec overlay (zuspec.be.xls.overlay)."""
import dataclasses as dc
import itertools

import pytest

import zuspec.ir.core as ir
from zuspec.ir.core.anchor import assign_anchors
from zuspec.be.xls import xir
from zuspec.be.xls.overlay import xls_overlay
from zuspec.be.xls.printer import format_package
from zuspec.be.xls.read_schedule import parse_schedules
from zuspec.be.xls.signature import signature_from_codegen

from harness import lower_fns
from irb import assign, bop, c, cmp, decl, fn, if_, l, p, ret, u

SRC = "/src/design.sv"


def stamp(root):
    """Give every IR node its own source position (line n, column 1)."""
    lines = itertools.count(1)
    seen = set()

    def walk(v):
        if isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
            return
        if not isinstance(v, ir.Base) or id(v) in seen or isinstance(v, ir.DataType):
            return
        seen.add(id(v))
        v.loc = ir.Loc(file=SRC, line=next(lines), pos=1)
        for f in dc.fields(v):
            if f.name not in ("loc", "anchor", "provenance", "annotations"):
                walk(getattr(v, f.name))
    walk(root)
    return root


def mac():
    return stamp(fn("mac", [("a", u(8)), ("b", u(8))], u(8), [
        decl("t", u(8), bop(p("a"), "Add", p("b"))),
        if_(cmp(l("t"), "Gt", c(100, u(8))), [assign(l("t"), bop(l("t"), "Sub", c(100, u(8))))]),
        ret(bop(l("t"), "BitXor", p("a"))),
    ]))


def lowered(f):
    t = assign_anchors(f)
    pkg = lower_fns([f], f.name)
    return t, pkg, format_package(pkg)


def test_every_lowered_node_maps_back_exactly():
    """Before XLS optimizes, our side is lossless: every node's position names
    the IR node it was lowered from."""
    t, pkg, text = lowered(mac())
    o = xls_overlay(t, ir_text=text, top="mac")
    cov = o.header.coverage
    assert cov["ops_by_quality"]["exact"] == cov["ops_total"] > 0
    assert set(o.column("xls.node_name")["quality"]) == {"exact"}


def test_schedule_and_critical_path():
    t, pkg, text = lowered(mac())
    body = pkg.functions[0]
    nodes = [n for n in body.nodes if n.op != "param"]
    # A two-stage schedule: first half in stage 0, rest in stage 1; each node 10ps.
    half = len(nodes) // 2
    lines = ["schedules {", '  key: "mac"', "  value {", '    function: "mac"']
    for stage, group in ((0, nodes[:half]), (1, nodes[half:])):
        lines += ["    stages {", f"      stage: {stage}"]
        for i, n in enumerate(group, 1):
            lines += ["      timed_nodes {", f'        node: "{n.ref}"',
                      "        node_delay_ps: 10", f"        path_delay_ps: {10 * i}", "      }"]
        lines += ["    }"]
    lines += ["    length: 2", "    min_clock_period_ps: 500", "  }", "}"]
    o = xls_overlay(t, ir_text=text, schedule_text="\n".join(lines), top="mac")
    col = o.column("sched.stage")
    assert sorted(set(col["value"])) == [0, 1]
    assert o.column("sched.stage_count")["value"] == [2]
    assert o.column("timing.clock_period_ps")["value"] == [500]
    first = nodes[0]
    a = next(a for a, v in zip(o.column("xls.node_name")["anchor"],
                               o.column("xls.node_name")["value"]) if v == first.ref)
    assert a.startswith("mac#v:t")    # the first statement declares t
    paths = o.column("timing.critical_path")["value"]
    assert [p_["stage"] for p_ in paths] == [0, 1]
    assert all(p_["anchors"] for p_ in paths)


def test_lost_positions_are_partial_or_unmapped():
    t, pkg, text = lowered(mac())
    body = pkg.functions[0]
    nodes = [n for n in body.nodes if n.op != "param"]
    stmt_line = t["mac#v:t"].loc[1]
    nodes[0].pos = [(0, stmt_line - 1, 40)]        # right line, no expression starts there
    nodes[1].pos = [(0, 9999, 0)]                  # nowhere
    nodes[2].pos = []                              # no position at all
    o = xls_overlay(t, ir_text=format_package(pkg), top="mac")
    q = o.header.coverage["ops_by_quality"]
    assert q["partial"] == 1 and q["unmapped"] == 2
    unmapped = [a for a, qq in zip(o.column("xls.node_name")["anchor"],
                                   o.column("xls.node_name")["quality"]) if qq == "unmapped"]
    assert set(unmapped) == {"mac"}               # recorded on the block, so totals add up
    assert o.header.coverage["pct_ops_attributed"] < 100.0


def test_inlined_call_sites_do_not_make_a_node_merged():
    t, pkg, text = lowered(mac())
    n = [n for n in pkg.functions[0].nodes if n.op != "param"][0]
    n.pos = n.pos + [(0, 9998, 0)]                  # a call site we have no node for
    o = xls_overlay(t, ir_text=format_package(pkg), top="mac")
    assert o.header.coverage["ops_by_quality"]["merged"] == 0


SIG = """
module_name: "rle"
clock_name: "clk"
reset { name: "rst" asynchronous: false active_low: false }
channel_interfaces {
  channel_name: "input_r"
  direction: CHANNEL_DIRECTION_RECEIVE
  type { type_enum: BITS bit_count: 65 }
  kind: CHANNEL_KIND_STREAMING
  streaming { flow_control: CHANNEL_FLOW_CONTROL_READY_VALID data_port_name: "input_r"
              ready_port_name: "input_r_rdy" valid_port_name: "input_r_vld" }
  flop_kind: FLOP_KIND_SKID
}
"""


def test_signature_reads_flop_kind():
    sig = signature_from_codegen(SIG)
    assert sig.channel("input_r").flop_kind == "skid"
    assert sig.channel("input_r").stage is None


def test_iface_facts_on_the_block():
    t, pkg, text = lowered(mac())
    o = xls_overlay(t, ir_text=text, top="mac", signature_text=SIG)
    blk = o.column("iface.block")
    assert blk["anchor"] == ["mac"] and blk["value"][0]["name"] == "rle"
    assert o.column("sched.ii")["value"] == [{"value": 1, "reliable": True}]


def test_schedule_reader_matches_old_shape():
    text = """schedules { key: "f" value { function: "f"
      stages { stage: 0 timed_nodes { node: "a" node_delay_ps: 0 path_delay_ps: 0 } }
      stages { stage: 1 timed_nodes { node: "add.3" node_delay_ps: 5 path_delay_ps: 5 } }
      length: 2 min_clock_period_ps: 7 } }"""
    s = parse_schedules(text)["f"]
    assert s.length == 2 and s.min_clock_period_ps == 7
    assert s.stages == {0: ["a"], 1: ["add.3"]}
    assert s.nodes["add.3"].node_delay_ps == 5
