"""XT-3 / XT-4: the interpreter on IR that XLS itself generated from DSLX
(``data/xls_ref``), with expected values worked out from the DSLX source."""
import os

from zuspec.be.xls.testing.interp import Interpreter, run_proc
from zuspec.be.xls.testing.parser import parse_package

REF = os.path.join(os.path.dirname(__file__), "data", "xls_ref")


def load(name):
    with open(os.path.join(REF, name)) as f:
        return parse_package(f.read())


def test_acc_proc():
    # sum starts at 5; each x adds; x == 0 sends the sum and resets it.
    pkg = load("acc_proc.ir")
    r = run_proc(pkg, pkg.top, {"_in_ch": [3, 0, 7, 0, 1]})
    assert r.outputs["_out_ch"] == [8, 7]
    assert r.activations == 5
    assert r.stop == "blocked" and r.blocked_on == "_in_ch"
    assert r.state["__state"] == 1


def test_acc_fn_clamps_index():
    pkg = load("acc_fn.ir")
    it = Interpreter(pkg)
    # a + b[i] + (a >> 1); u2 index is always in range here
    assert it.call(pkg.top, [10, [1, 2, 3, 4], 2]) == (10 + 3 + 5)
    assert it.call(pkg.top, [255, [1, 2, 3, 4], 3]) == (255 + 4 + 127) & 0xFF


def test_twostate_proc_and_assert():
    pkg = load("p2_twostate.ir")
    # state (c, arr): send c, then c+1 and arr[x.b] = x.a
    r = run_proc(pkg, pkg.top, {"_in_ch": [(10, 1), (20, 2), (30, 9)]})
    assert r.outputs["_out_ch"] == [1, 2, 3]
    assert r.state["__state"] == (4, [0, 10, 20, 0])
    r = run_proc(pkg, pkg.top, {"_in_ch": [(10, 1), (255, 2)]})
    assert r.stop == "assert" and r.assertion.label == "never_ff"
    assert r.outputs["_out_ch"] == [1]


def test_twostate_opt_matches_unopt():
    a, b = load("p2_twostate.ir"), load("p2_twostate.opt.ir")
    ins = {"_in_ch": [(i * 7 % 255, i % 16) for i in range(20)]}
    ra, rb = run_proc(a, a.top, ins), run_proc(b, b.top, ins)
    assert ra.outputs == rb.outputs


def test_g_fn():
    pkg = load("p2_g.ir")
    # (a >> 2, b >> 1 (arithmetic, s8), a ++ (b as u8), a[2:6], and_reduce(a))
    assert Interpreter(pkg).call(pkg.top, [0xF4, 0x81]) == (0x3D, 0xC0, 0xF481, 0xD, 0)
