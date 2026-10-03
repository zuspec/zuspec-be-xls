"""BX-3: hand-built static-subset IR -> XLS IR, checked against SV semantics.

Each test states the SV meaning as a small Python model (2-state, SV widths)
and compares the interpreter (and XLS, when available) against it on
exhaustive or edge-heavy vectors.
"""
import itertools
import random

import pytest

import zuspec.ir.core as ir
from zuspec.be.xls.width import StaticSubsetError
from zuspec.be.xls.testing.interp import Interpreter, XlsAssertionError
from harness import lower_fns, roundtrip, run_fn
from irb import (arr, assign, bop, c, call, cast, cat, cmp, decl, fn, for_, idx, if_,
                 ifx, l, p, ps, ret, s, u, un)


def sx(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) else v


def rng_vals(w, n=24, seed=0):
    r = random.Random(seed + w)
    edge = [0, 1, (1 << w) - 1, 1 << (w - 1), (1 << (w - 1)) - 1]
    return edge + [r.getrandbits(w) for _ in range(n)]


def test_e1_width_casts_mul_high_byte():
    # return 8'((16'(a) * 16'(b)) >> 8)
    f = fn("mulhi", [("a", u(8)), ("b", u(8))], u(8), [
        ret(cast(u(8), bop(bop(cast(u(16), p("a")), "Mult", cast(u(16), p("b"))),
                           "RShift", c(8))))])
    vecs = [(a, b) for a in rng_vals(8, 6) for b in rng_vals(8, 6, 1)]
    assert run_fn([f], "mulhi", vecs) == [((a * b) >> 8) & 0xFF for a, b in vecs]


def test_e14_extension_follows_source_signedness():
    f = fn("ext", [("a", s(4)), ("b", u(4))], u(16), [
        ret(cat(cast(s(8), p("a")), cast(s(8), p("b"))))])
    vecs = [(a, b) for a in range(16) for b in range(16)]
    assert run_fn([f], "ext", vecs) == [((sx(a, 4) & 0xFF) << 8) | b for a, b in vecs]


def test_e6_compare_signedness():
    # a (signed) < b (unsigned): SV compares unsigned; with $signed(b): signed
    f = fn("lt", [("a", s(4)), ("b", u(4))], u(2), [
        ret(cat(cmp(p("a"), "Lt", p("b")),
                cmp(p("a"), "Lt", ir.ExprSigned(value=p("b")))))])
    vecs = list(itertools.product(range(16), range(16)))
    want = [(int(a < b) << 1) | int(sx(a, 4) < sx(b, 4)) for a, b in vecs]
    assert run_fn([f], "lt", vecs) == want


def test_e4_e5_shifts():
    f = fn("sh", [("a", s(8)), ("n", u(4))], u(24), [
        ret(cat(bop(p("a"), "LShift", p("n")), bop(p("a"), "RShift", p("n")),
                bop(p("a"), "ARShift", p("n"))))])
    g = fn("shu", [("a", u(8)), ("n", u(4))], u(8), [ret(bop(p("a"), "ARShift", p("n")))])
    vecs = [(a, n) for a in rng_vals(8, 8) for n in range(16)]
    want = []
    for a, n in vecs:
        sl = 0 if n >= 8 else (a << n) & 0xFF
        sr = 0 if n >= 8 else a >> n
        sa = (sx(a, 8) >> min(n, 8)) & 0xFF
        want.append((sl << 16) | (sr << 8) | sa)
    assert run_fn([f], "sh", vecs) == want
    # >>> on an unsigned operand is a logical shift
    assert run_fn([g], "shu", vecs) == [0 if n >= 8 else a >> n for a, n in vecs]


def test_e7_e8_e9_e10_logic_reduce_select_concat():
    f = fn("mix", [("a", u(4)), ("b", u(4))], u(12), [
        ret(cat(ir.ExprBool(op=ir.BoolOp.And, values=[p("a"), p("b")]),
                ir.ExprBool(op=ir.BoolOp.Or, values=[p("a"), p("b")]),
                un("Not", p("a")), un("AndReduce", p("a")), un("OrReduce", p("a")),
                un("XorReduce", p("a")), ir.ExprReplicate(count=2, value=c(1, u(1))),
                ifx(p("a"), p("b"), un("Invert", p("b")))))])
    vecs = list(itertools.product(range(16), range(16)))
    want = []
    for a, b in vecs:
        bits = [int(bool(a) and bool(b)), int(bool(a) or bool(b)), int(not a),
                int(a == 15), int(a != 0), bin(a).count("1") & 1, 1, 1]
        v = 0
        for x in bits:
            v = (v << 1) | x
        want.append((v << 4) | (b if a else (~b & 0xF)))
    assert run_fn([f], "mix", vecs) == want


def _sv_part_read(x, w, start, width):
    return sum(((x >> (start + k)) & 1) << k for k in range(width) if 0 <= start + k < w)


def test_e12_e13_part_select_read_including_negative_base():
    f = fn("pr", [("x", u(8)), ("i", s(5)), ("j", u(4))], u(12), [
        ret(cat(ps(p("x"), p("i"), 4), ps(p("x"), p("j"), 4), ps(p("x"), c(2), 4)))])
    vecs = [(x, i, j) for x in (0xA5, 0xFF, 0x3C) for i in range(32) for j in range(16)]
    want = [(_sv_part_read(x, 8, sx(i, 5), 4) << 8) | (_sv_part_read(x, 8, j, 4) << 4) |
            ((x >> 2) & 0xF) for x, i, j in vecs]
    assert run_fn([f], "pr", vecs) == want


def test_e11_bit_select_and_write():
    def model(x, i, v):
        i = sx(i, 5)
        r = x
        for k in range(4):
            if 0 <= i + k < 8:
                r = (r & ~(1 << (i + k))) | (((v >> k) & 1) << (i + k))
        return r
    f = fn("pw", [("x", u(8)), ("i", s(5)), ("v", u(4))], u(9), [
        assign(ps(p("x"), p("i"), 4), p("v")),
        ret(cat(idx(p("x"), p("i")), p("x")))])
    vecs = [(x, i, v) for x in (0x00, 0xFF, 0x96) for i in range(32) for v in (0, 0xF, 0x9)]
    want = []
    for x, i, v in vecs:
        r = model(x, i, v)
        bit = (r >> sx(i, 5)) & 1 if 0 <= sx(i, 5) < 8 else 0
        want.append((bit << 8) | r)
    assert run_fn([f], "pw", vecs) == want


def test_e17_e18_array_oob():
    f = fn("ar", [("a", arr(u(8), 5)), ("i", u(3)), ("j", s(3)), ("v", u(8))], u(16), [
        decl("t", u(8), idx(p("a"), p("j"))),
        assign(idx(p("a"), p("i")), p("v")),
        ret(cat(idx(p("a"), p("i")), l("t")))])
    a = [11, 22, 33, 44, 55]
    vecs = [(a, i, j, 99) for i in range(8) for j in range(8)]
    want = []
    for _, i, j, v in vecs:
        sj = sx(j, 3)
        t = a[sj] if 0 <= sj < 5 else 0
        r = v if i < 5 else 0          # write is a no-op out of bounds; read gives 0
        want.append((r << 8) | t)
    assert run_fn([f], "ar", vecs) == want


def test_s3_case_first_match_wins():
    m = ir.StmtMatch(subject=p("x"), cases=[
        ir.StmtMatchCase(pattern=ir.PatternOr(patterns=[ir.PatternValue(value=c(1, u(3))),
                                                        ir.PatternValue(value=c(2, u(3)))]),
                         body=[assign(l("r"), c(10, u(8)))]),
        ir.StmtMatchCase(pattern=ir.PatternValue(value=c(2, u(3))),
                         body=[assign(l("r"), c(20, u(8)))]),
        ir.StmtMatchCase(pattern=ir.PatternAs(), body=[assign(l("r"), c(30, u(8)))]),
    ])
    f = fn("cs", [("x", u(3))], u(8), [decl("r", u(8)), m, ret(l("r"))])
    assert run_fn([f], "cs", [(x,) for x in range(8)]) == [30, 10, 10, 30, 30, 30, 30, 30]


def test_s4_unrolled_for_popcount():
    f = fn("pop", [("x", u(8))], u(4), [
        decl("n", u(4), c(0, u(4))),
        *for_("i", 8, [assign(l("n"), bop(l("n"), "Add",
                                           cast(u(4), idx(p("x"), l("i")))))]),
        ret(l("n"))])
    pkg = lower_fns([f])
    assert not any(n.op in ("ult", "slt") for n in pkg.function("pop").nodes)  # unrolled
    assert run_fn([f], "pop", [(x,) for x in range(256)]) == [bin(x).count("1") for x in range(256)]


def test_s6_early_return_and_s5_calls():
    g = fn("g", [("x", u(8))], u(8), [ret(bop(p("x"), "BitXor", c(0x5A, u(8))))])
    f = fn("f", [("x", u(8))], u(8), [
        if_(cmp(p("x"), "Eq", c(0, u(8))), [ret(c(1, u(8)))]),
        if_(cmp(p("x"), "Eq", c(1, u(8))), [ret(c(7, u(8)))], [assign(p("x"), call("g", p("x")))]),
        ret(bop(call("g", p("x")), "Add", c(1, u(8))))])
    want = [1, 7] + [(((x ^ 0x5A) ^ 0x5A) + 1) & 0xFF for x in range(2, 256)]
    assert run_fn([g, f], "f", [(x,) for x in range(256)]) == want


def test_d1_division_must_be_guarded():
    ok = fn("dv", [("n", u(8)), ("d", u(8))], u(8), [
        if_(cmp(p("d"), "NotEq", c(0, u(8))), [ret(bop(p("n"), "Div", p("d")))]),
        ret(ifx(cmp(p("d"), "NotEq", c(0, u(8))), bop(p("n"), "Mod", p("d")), c(0, u(8))))])
    vecs = [(n, d) for n in rng_vals(8, 4) for d in range(0, 256, 17)]
    assert run_fn([ok], "dv", vecs) == [n // d if d else 0 for n, d in vecs]
    bad = fn("bad", [("n", u(8)), ("d", u(8))], u(8), [ret(bop(p("n"), "Div", p("d")))])
    with pytest.raises(StaticSubsetError, match="may be zero"):
        lower_fns([bad])
    sdiv = fn("sd", [("n", s(8))], s(8), [ret(bop(p("n"), "Div", c(3, s(8))))])
    vals = list(range(256))
    assert run_fn([sdiv], "sd", [(n,) for n in vals]) == \
        [int(sx(n, 8) / 3) & 0xFF for n in vals]


def test_s7_assert_in_function():
    f = fn("af", [("x", u(8))], u(8), [
        ir.StmtAssert(test=cmp(p("x"), "NotEq", c(0, u(8))),
                      msg=ir.ExprConstant(value="x must be non-zero")),
        ret(p("x"))])
    pkg = roundtrip(lower_fns([f]))
    it = Interpreter(pkg)
    assert it.call("af", [5]) == 5
    with pytest.raises(XlsAssertionError, match="non-zero"):
        it.call("af", [0])


def test_width_mismatch_is_a_located_error():
    e = bop(p("a"), "Add", p("b"))
    e.loc = ir.Loc(file="t.sv", line=7, pos=12)
    f = fn("bad", [("a", u(8)), ("b", u(4))], u(8), [ret(e)])
    with pytest.raises(StaticSubsetError) as ex:
        lower_fns([f])
    assert ex.value.loc.line == 7 and "insert a cast" in str(ex.value)
    assert str(ex.value).startswith("t.sv:7:12:")


def test_e18_signed_index_write_does_not_alias():
    # s3 -4 is 0b100: XLS would read it as index 4; SV says out of bounds
    f = fn("aw", [("a", arr(u(8), 5)), ("j", s(3))], arr(u(8), 5), [
        assign(idx(p("a"), p("j")), c(0xEE, u(8))), ret(p("a"))])
    a = [1, 2, 3, 4, 5]
    want = []
    for j in range(8):
        r = list(a)
        if 0 <= sx(j, 3) < 5:
            r[sx(j, 3)] = 0xEE
        want.append(r)
    assert run_fn([f], "aw", [(a, j) for j in range(8)]) == want


@pytest.mark.xls
def test_s7_assert_fires_in_xls_too():
    from zuspec.be.xls.testing.xlstools import XlsToolError, xls_eval_fn
    f = fn("af", [("x", u(8))], u(8), [
        if_(cmp(p("x"), "Gt", c(3, u(8))),
            [ir.StmtAssert(test=cmp(p("x"), "NotEq", c(9, u(8))),
                           msg=ir.ExprConstant(value="nine"))]),
        ret(p("x"))])
    pkg = lower_fns([f])
    assert xls_eval_fn(pkg, "af", [2]) == 2
    assert xls_eval_fn(pkg, "af", [10]) == 10
    with pytest.raises(XlsToolError, match="nine"):
        xls_eval_fn(pkg, "af", [9])
    with pytest.raises(XlsAssertionError):
        Interpreter(roundtrip(pkg)).call("af", [9])
