"""XT-5: op semantics of the interpreter.

``test_doc_*`` are the examples and rules stated in ``ir_semantics.md``,
written down independently of our emitter. ``test_xls_differential`` (marked
``xls``) runs every op in ``OPS`` at several widths on random and edge values
through both our interpreter and XLS (``opt_main`` constant folding).
"""
import random

import pytest

from zuspec.be.xls import xir
from zuspec.be.xls.printer import format_package
from zuspec.be.xls.testing.interp import Interpreter
from zuspec.be.xls.testing.parser import parse_package
from zuspec.be.xls.testing.typecheck import check_package
from xls_helpers import one_fn

B = xir.bits


def ev(pkg, *args):
    check_package(pkg)
    # Round-trip through text so every test also exercises printer + parser.
    pkg2 = parse_package(format_package(pkg))
    return Interpreter(pkg2).call(pkg2.top, list(args))


# ---------------------------------------------------------------------------
# Doc examples and stated rules
# ---------------------------------------------------------------------------

def test_doc_array_index_clamps():
    # "array_index": {10, 2} on bits[3][4][5] is A[4][2]
    t = xir.array_t(xir.array_t(B(3), 4), 5)
    arr = [[(i * 4 + j) % 8 for j in range(4)] for i in range(5)]
    pkg = one_fn([t, B(4), B(4)], lambda b, p: b.array_index(p[0], [p[1], p[2]]))
    assert ev(pkg, arr, 10, 2) == arr[4][2]
    assert ev(pkg, arr, 1, 2) == arr[1][2]
    pkg1 = one_fn([t, B(4)], lambda b, p: b.array_index(p[0], [p[1]]))
    assert ev(pkg1, arr, 1) == arr[1]


def test_doc_array_update_oob_is_noop():
    t = xir.array_t(B(8), 4)
    pkg = one_fn([t, B(8), B(3)], lambda b, p: b.array_update(p[0], p[1], [p[2]]))
    assert ev(pkg, [1, 2, 3, 4], 9, 2) == [1, 2, 9, 4]
    assert ev(pkg, [1, 2, 3, 4], 9, 4) == [1, 2, 3, 4]
    assert ev(pkg, [1, 2, 3, 4], 9, 7) == [1, 2, 3, 4]


def test_doc_shifts_overshift():
    for op, x, s, want in [("shll", 0x81, 1, 0x02), ("shll", 0x81, 8, 0),
                           ("shrl", 0x81, 7, 1), ("shrl", 0x81, 200, 0),
                           ("shra", 0x81, 1, 0xC0), ("shra", 0x81, 8, 0xFF),
                           ("shra", 0x41, 9, 0)]:
        pkg = one_fn([B(8), B(8)], lambda b, p, op=op: b.shift(op, p[0], p[1]))
        assert ev(pkg, x, s) == want, (op, x, s)


def test_doc_division_by_zero():
    # udiv by 0 -> all ones; umod by 0 -> 0; sdiv by 0 -> max pos/neg by
    # dividend sign; smod by 0 -> 0; smod sign follows the dividend.
    cases = [("udiv", 5, 0, 0xFF), ("umod", 5, 0, 0), ("sdiv", 5, 0, 0x7F),
             ("sdiv", 0, 0, 0x7F), ("sdiv", 0xFB, 0, 0x80), ("smod", 0xFB, 0, 0),
             ("sdiv", 0xF9, 2, 0xFD),            # -7 / 2 = -3 (toward zero)
             ("smod", 0xF9, 2, 0xFF),            # -7 % 2 = -1
             ("smod", 7, 0xFE, 1),               #  7 % -2 = 1
             ("sdiv", 0x80, 0xFF, 0x80)]         # -128 / -1 wraps
    for op, a, d, want in cases:
        pkg = one_fn([B(8), B(8)], lambda b, p, op=op: b.binop(op, p[0], p[1]))
        assert ev(pkg, a, d) == want, (op, a, d)


def test_doc_dynamic_bit_slice_oob_zero():
    pkg = one_fn([B(16), B(8)], lambda b, p: b.dynamic_bit_slice(p[0], p[1], 8))
    assert ev(pkg, 0xABCD, 0) == 0xCD
    assert ev(pkg, 0xABCD, 4) == 0xBC
    assert ev(pkg, 0xABCD, 12) == 0x0A
    assert ev(pkg, 0xABCD, 16) == 0


def test_doc_bit_slice_update():
    # the table in "bit_slice_update"
    pkg = one_fn([B(16), B(8), B(8)], lambda b, p: b.bit_slice_update(p[0], p[1], p[2]))
    for start, want in [(0, 0xABFF), (4, 0xAFFD), (12, 0xFBCD), (16, 0xABCD)]:
        assert ev(pkg, 0xABCD, start, 0xFF) == want


def test_doc_concat_msb_first():
    pkg = one_fn([B(4), B(8)], lambda b, p: b.concat(p[0], p[1]))
    assert ev(pkg, 0xA, 0x5C) == 0xA5C


def test_doc_sel_and_priority_sel():
    pkg = one_fn([B(2), B(8), B(8), B(8)],
                 lambda b, p: b.sel(p[0], [p[1], p[2]], default=p[3]))
    assert [ev(pkg, s, 10, 20, 30) for s in range(4)] == [10, 20, 30, 30]
    pkg = one_fn([B(3), B(8), B(8), B(8), B(8)],
                 lambda b, p: b.priority_sel(p[0], [p[1], p[2], p[3]], p[4]))
    assert [ev(pkg, s, 1, 2, 3, 9) for s in range(8)] == [9, 1, 2, 1, 3, 1, 2, 1]


def test_doc_extend_and_reduce():
    pkg = one_fn([B(4)], lambda b, p: b.tuple(
        b.sign_ext(p[0], 8), b.zero_ext(p[0], 8), b.reduce("and_reduce", p[0]),
        b.reduce("or_reduce", p[0]), b.reduce("xor_reduce", p[0])))
    assert ev(pkg, 0x9) == (0xF9, 0x09, 0, 1, 0)
    assert ev(pkg, 0xF) == (0xFF, 0x0F, 1, 1, 0)
    assert ev(pkg, 0x0) == (0, 0, 0, 0, 0)


def test_doc_mul_result_width():
    pkg = one_fn([B(4), B(4)], lambda b, p: b.tuple(
        b.mul("umul", p[0], p[1], 8), b.mul("smul", p[0], p[1], 8),
        b.mul("umul", p[0], p[1], 3)))
    # 0xF*0xF = 225; smul: -1 * -1 = 1; truncated to 3 bits: 225 & 7 = 1
    assert ev(pkg, 0xF, 0xF) == (225, 1, 1)
    assert ev(pkg, 0xF, 2) == (30, 0xFE, 6)


def test_typecheck_rejects_bad_ops():
    from zuspec.be.xls.testing.typecheck import XlsCheckError
    pkg = one_fn([B(8), B(4)], lambda b, p: b.tuple(p[0], p[1]))
    f = pkg.function("f")
    bad = xir.Node(op="add", type=B(8), operands=[f.params[0], f.params[1]], id=99)
    f.nodes.insert(0, bad)
    with pytest.raises(XlsCheckError):
        check_package(pkg)


# ---------------------------------------------------------------------------
# Differential: our interpreter vs XLS, op by op
# ---------------------------------------------------------------------------

def _ops(w):
    """(label, param types, body) for each op at width *w*."""
    bw = [B(w), B(w)]
    out = []
    for op in ("and", "or", "xor", "nand", "nor"):
        out.append((op, bw, lambda b, p, op=op: b.nary(op, p[0], p[1])))
    for op in ("add", "sub", "udiv", "sdiv", "umod", "smod"):
        out.append((op, bw, lambda b, p, op=op: b.binop(op, p[0], p[1])))
    for op in ("eq", "ne", "ult", "ule", "ugt", "uge", "slt", "sle", "sgt", "sge"):
        out.append((op, bw, lambda b, p, op=op: b.compare(op, p[0], p[1])))
    for op in ("shll", "shrl", "shra"):
        out.append((op, [B(w), B(4)], lambda b, p, op=op: b.shift(op, p[0], p[1])))
    for op in ("not", "neg"):
        out.append((op, [B(w)], lambda b, p, op=op: b.unop(op, p[0])))
    for op in ("and_reduce", "or_reduce", "xor_reduce"):
        out.append((op, [B(w)], lambda b, p, op=op: b.reduce(op, p[0])))
    for op in ("umul", "smul"):
        for rw in (w, 2 * w, max(1, w - 1)):
            out.append((f"{op}{rw}", [B(w), B(w + 1)],
                        lambda b, p, op=op, rw=rw: b.mul(op, p[0], p[1], rw)))
    out.append(("sign_ext", [B(w)], lambda b, p: b.sign_ext(p[0], w + 5)))
    out.append(("zero_ext", [B(w)], lambda b, p: b.zero_ext(p[0], w + 5)))
    out.append(("dbs", [B(w), B(4)], lambda b, p: b.dynamic_bit_slice(p[0], p[1], min(3, w))))
    out.append(("bsu", [B(w), B(4), B(3)],
                lambda b, p: b.bit_slice_update(p[0], p[1], p[2])))
    out.append(("concat", bw, lambda b, p: b.concat(p[0], p[1])))
    out.append(("sel", [B(2), B(w), B(w), B(w)],
                lambda b, p: b.sel(p[0], [p[1], p[2], p[3]], default=p[1])))
    out.append(("array", [xir.array_t(B(w), 3), B(2), B(w)],
                lambda b, p: b.tuple(b.array_index(p[0], [p[1]]),
                                     b.array_update(p[0], p[2], [p[1]]))))
    return out


def _random_value(rng, t):
    if isinstance(t, xir.BitsType):
        w = t.width
        edges = [0, 1, (1 << w) - 1, 1 << (w - 1), (1 << (w - 1)) - 1]
        return rng.choice(edges) if rng.random() < 0.3 else rng.getrandbits(w)
    if isinstance(t, xir.ArrayType):
        return [_random_value(rng, t.elem) for _ in range(t.size)]
    raise TypeError(t)


@pytest.mark.xls
@pytest.mark.parametrize("w", [1, 3, 8, 13, 64, 65])
def test_xls_differential(w):
    from zuspec.be.xls.testing.xlstools import xls_eval_fn_batch
    rng = random.Random(1000 + w)
    for label, ptypes, body in _ops(w):
        pkg = one_fn(ptypes, body)
        check_package(pkg)
        vecs = [[_random_value(rng, t) for t in ptypes] for _ in range(40)]
        want = xls_eval_fn_batch(pkg, "f", vecs)
        interp = Interpreter(pkg)
        got = [interp.call("f", v) for v in vecs]
        for v, g, x in zip(vecs, got, want):
            assert g == x, f"{label} w={w} args={v}: interp={g} xls={x}"
