"""BX-4: hand-built component IR -> XLS proc, run as a Kahn process."""
import binascii
import os

import pytest

import zuspec.ir.core as ir
from zuspec.be.xls.lower_proc import lower_component
from zuspec.be.xls.width import StaticSubsetError
from harness import roundtrip, run_comp
from irb import (assign, bop, c, call, cast, cmp, component, decl, field, fld, fn, for_,
                 forever, get, idx, if_, in_port, l, out_port, p, put, ret, u, un)


def test_p1_p6_accumulator_matches_xls_reference():
    # The DSLX `acc` proc of tests/unit/data/xls_ref, written as SV-style IR.
    comp = component("acc", [in_port("in_ch", u(8)), out_port("out_ch", u(16)),
                             field("sum", u(16), c(5, u(16)))], [forever(
        decl("x", u(8)),
        assign(l("x"), get(0)),
        decl("s", u(16), bop(fld(2), "Add", cast(u(16), l("x")))),
        if_(cmp(l("x"), "Eq", c(0, u(8))),
            [put(1, l("s")), assign(fld(2), c(0, u(16)))],
            [assign(fld(2), l("s"))]))])
    pkg, r = run_comp(comp, {"in_ch": [3, 0, 7, 0, 1]})
    assert r.outputs["out_ch"] == [8, 7]
    assert r.activations == 5 and r.state["sum"] == 1
    proc = pkg.proc("acc")
    assert [s.name for s in proc.state] == ["sum"]          # x, s are temporaries
    assert sum(n.op == "next_value" for n in proc.nodes) == 1
    assert all("predicate" not in n.attrs for n in proc.nodes if n.op == "next_value")


def test_p2_p3_p4_prologue_state_and_temporaries():
    comp = component("ps", [in_port("i", u(8)), out_port("o", u(8))], [
        decl("count", u(8), c(10, u(8))),                        # prologue local: state
        assign(l("count"), bop(l("count"), "Add", c(1, u(8)))),  # init = 11
        forever(
            decl("t", u(8)),                                     # temporary: 0 each time
            if_(cmp(l("count"), "Eq", c(12, u(8))), [assign(l("t"), c(100, u(8)))]),
            decl("x", u(8)),
            assign(l("x"), get(0)),
            put(1, bop(bop(l("x"), "Add", l("count")), "Add", l("t"))),
            assign(l("count"), bop(l("count"), "Add", c(1, u(8)))))])
    pkg, r = run_comp(comp, {"i": [1, 2, 3]})
    # t is re-initialized every activation: 116 here would mean it leaked as state
    assert r.outputs["o"] == [12, 114, 16]
    assert [(s.name, s.init) for s in pkg.proc("ps").state] == [("count", 11)]


def test_p7_predicated_channel_ops():
    # receive from b only when a's value is odd; send only when the sum is big
    comp = component("pred", [in_port("a", u(8)), in_port("b", u(8)), out_port("o", u(8))], [
        forever(
            decl("x", u(8), None), assign(l("x"), get(0)),
            decl("y", u(8), None),
            if_(idx(l("x"), c(0)), [assign(l("y"), get(1))]),
            if_(cmp(bop(l("x"), "Add", l("y")), "Gt", c(10, u(8))),
                [put(2, bop(l("x"), "Add", l("y")))]))])
    pkg, r = run_comp(comp, {"a": [1, 2, 9, 4, 11], "b": [3, 5, 7]})
    # x=1:y=3 (4, no send); x=2: y=0 (2); x=9: y=5 (14 send); x=4 (4); x=11: y=7 (18 send)
    assert r.outputs["o"] == [14, 18]
    recv = [n for n in pkg.proc("pred").nodes if n.op == "receive"]
    assert "predicate" not in recv[0].attrs and "predicate" in recv[1].attrs


def test_p8_effects_in_program_order():
    comp = component("ord", [in_port("i", u(8)), out_port("o1", u(8)), out_port("o2", u(8))], [
        forever(put(1, c(1, u(8))), decl("x", u(8)), assign(l("x"), get(0)), put(2, l("x")))])
    pkg, r = run_comp(comp, {"i": [7, 8]})
    # The third activation sends on o1, then blocks on i: the first send stands.
    assert r.outputs == {"o1": [1, 1, 1], "o2": [7, 8]}
    ops = [n for n in pkg.proc("ord").nodes if n.op in ("send", "receive")]
    assert [n.attrs["channel"] for n in ops] == ["o1", "i", "o2"]
    # each op takes the previous op's token
    assert ops[1].operands[0] is ops[0]
    assert ops[2].operands[0].operands[0] is ops[1]


def test_s7_assert_in_proc_is_predicated():
    comp = component("ap", [in_port("i", u(8)), out_port("o", u(8))], [forever(
        decl("x", u(8)), assign(l("x"), get(0)),
        if_(cmp(l("x"), "Gt", c(100, u(8))),
            [ir.StmtAssert(test=cmp(l("x"), "NotEq", c(200, u(8))),
                           msg=ir.ExprConstant(value="no 200"))]),
        put(1, l("x")))])
    _, r = run_comp(comp, {"i": [1, 150, 200, 3]})
    assert r.stop == "assert" and r.outputs["o"] == [1, 150]


def _reject(body, match, fields=None):
    comp = component("bad", fields or [in_port("i", u(8)), out_port("o", u(8))], body)
    with pytest.raises(StaticSubsetError, match=match):
        lower_component(comp)


def test_p10_rejections():
    loop_get = [forever(decl("x", u(8)), *for_("k", 2, [assign(l("x"), get(0))]))]
    _reject(loop_get, "inside a loop")
    twice = [forever(decl("x", u(8)), assign(l("x"), get(0)), assign(l("x"), get(0)))]
    _reject(twice, "second operation on channel 'i'")
    bare_get = [forever(ir.StmtExpr(expr=get(0)))]
    _reject(bare_get, "must assign its result")
    tick = [forever(ir.StmtExpr(expr=ir.ExprAwait(value=ir.ExprCall(
        func=ir.ExprAttribute(value=ir.TypeExprRefSelf(), attr="tick"), args=[]))))]
    _reject(tick, "not a get/put on a channel port")
    _reject([decl("x", u(8))], "must end in `forever`")
    untyped = [ir.Field(name="i", datatype=u(8), kind=ir.FieldKind.Port)]
    _reject([forever()], "DataTypeGetIF", fields=untyped)
    nonconst = [decl("x", u(8)), assign(l("x"), get(0)), forever()]
    _reject(nonconst, "channel operations are only allowed")


# ---------------------------------------------------------------------------
# C1: CRC-32 (reflected, poly 0xEDB88320) as a function and a streaming proc
# ---------------------------------------------------------------------------

def crc32_step_fn():
    # crc ^= byte; repeat 8: crc = (crc >> 1) ^ (32'hEDB88320 & {32{crc[0]}})
    return fn("crc32_step", [("crc", u(32)), ("b", u(8))], u(32), [
        assign(p("crc"), bop(p("crc"), "BitXor", cast(u(32), p("b")))),
        *for_("k", 8, [assign(p("crc"), bop(
            bop(p("crc"), "RShift", c(1)), "BitXor",
            bop(c(0xEDB88320, u(32)), "BitAnd",
                ir.ExprReplicate(count=32, value=idx(p("crc"), c(0))))))]),
        ret(p("crc"))])


def test_c1_crc32_proc():
    # input: {last, byte}; on last, send ~crc and restart
    comp = component("crc32", [in_port("i", u(9)), out_port("o", u(32)),
                               field("crc", u(32), c(0xFFFFFFFF, u(32)))], [forever(
        decl("w", u(9)), assign(l("w"), get(0)),
        assign(fld(2), call("crc32_step", fld(2), ir.ExprPartSelect(
            value=l("w"), base=c(0), width=8))),
        if_(idx(l("w"), c(8)), [put(1, un("Invert", fld(2))),
                                assign(fld(2), c(0xFFFFFFFF, u(32)))]))],
        functions=[crc32_step_fn()])
    msgs = [b"123456789", b"hello", b"\x00"]
    stream = []
    for m in msgs:
        stream += [b | (0x100 if k == len(m) - 1 else 0) for k, b in enumerate(m)]
    _, r = run_comp(comp, {"i": stream})
    assert r.outputs["o"] == [binascii.crc32(m) for m in msgs]
    assert r.outputs["o"][0] == 0xCBF43926


def test_every_state_node_has_an_absolute_source_position():
    # XLS errors about state name state_read / next_value nodes and state
    # elements: each needs a pos= to be mapped back to the source.
    def at(x, line, col=3):
        x.loc = ir.Loc(file="rtl/acc.sv", line=line, pos=col)
        return x
    comp = component("acc", [in_port("in_ch", u(8)), out_port("out_ch", u(16)),
                             at(field("sum", u(16), c(5, u(16))), 4),
                             at(field("idle", u(1)), 5)], [at(forever(
        at(decl("x", u(8)), 8),
        at(assign(l("x"), get(0)), 8),
        at(assign(fld(2), bop(fld(2), "Add", cast(u(16), l("x")))), 9),
        at(put(1, fld(2)), 10)), 7)])
    pkg = lower_component(comp)
    roundtrip(pkg)                  # opt_main accepts pos= on state elements
    path = os.path.abspath("rtl/acc.sv")
    assert list(pkg.files.values()) == [path]
    proc = pkg.proc("acc")
    assert {s.name: s.pos for s in proc.state} == {"sum": [(0, 3, 2)], "idle": [(0, 4, 2)]}
    nodes = {(n.op, n.attrs.get("state_element")): n.pos for n in proc.nodes
             if n.op in ("state_read", "next_value")}
    assert nodes[("state_read", "sum")] == [(0, 3, 2)]
    assert nodes[("next_value", "sum")] == [(0, 8, 2)]      # the update on line 9
    assert nodes[("next_value", "idle")] == [(0, 4, 2)]     # unchanged: its declaration
