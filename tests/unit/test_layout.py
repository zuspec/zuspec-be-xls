"""XT-6 / X1-4: xir.flatten is the layout of XLS's generated Verilog ports."""
import re

import pytest

from zuspec.be.xls import xir
from zuspec.be.xls.testing.xlstools import run_codegen

T = xir.tuple_t(xir.bits(4), xir.array_t(xir.bits(8), 3),
                xir.tuple_t(xir.bits(2), xir.array_t(xir.bits(3), 2)))
V = (0xa, [0x11, 0x22, 0x33], (0x1, [0x5, 0x2]))


def test_flatten_order_and_round_trip():
    # tuple element 0 in the MSBs; array element 0 in the LSBs
    assert xir.flatten([1, 2, 3], xir.array_t(xir.bits(4), 3)) == 0x321
    assert xir.flatten((1, 2, 3), xir.tuple_t(*[xir.bits(4)] * 3)) == 0x123
    assert xir.flatten(V, T) == (0xa << 32) | (0x332211 << 8) | (0x1 << 6) | (0x2 << 3) | 0x5
    assert xir.unflatten(xir.flatten(V, T), T) == V


def _verilog_concat_value(text: str) -> int:
    """Value of a Verilog concatenation of sized hex literals, e.g. {4'ha, {8'h1, ...}}."""
    out = 0
    for w, h in re.findall(r"(\d+)'h([0-9a-fA-F_]+)", text):
        out = (out << int(w)) | int(h.replace("_", ""), 16)
    return out


@pytest.mark.xls_codegen
def test_x1_4_flatten_matches_codegen_port_layout():
    ty = str(T)
    text = f"""package lay

top proc lay<out: {ty} out>(st: bits[1], init={{0}}) {{
  chan_interface out(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  st: bits[1] = state_read(state_element=st, id=1)
  tok: token = after_all(id=2)
  v: {ty} = literal(value={xir.format_value(V, T)}, id=3)
  send.4: token = send(tok, v, channel=out, id=4)
  next_value.5: () = next_value(state_element=st, value=st, id=5)
}}
"""
    res = run_codegen(text)
    m = re.search(r"wire \[(\d+):0\] v = (\{.*\});", res.verilog)
    assert m, res.verilog
    assert int(m.group(1)) + 1 == T.flat_bit_count()
    assert _verilog_concat_value(m.group(2)) == xir.flatten(V, T)
