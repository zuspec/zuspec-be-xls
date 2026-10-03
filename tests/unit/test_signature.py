"""BX-8 / R9: the predicted block signature for a lowered component."""
import pytest

import zuspec.ir.core as ir
from zuspec.be.xls.lower_proc import lower_component
from zuspec.be.xls.printer import format_package
from zuspec.be.xls.signature import (CodegenOptions, predict_signature, signature_from_codegen,
                                     signature_mismatches)
from zuspec.be.xls.testing.xlstools import run_codegen
from irb import (assign, bop, c, cast, component, decl, field, fld, forever, get, in_port, l,
                 out_port, put, u)


def test_predicted_signature_and_flags():
    comp = component("acc", [in_port("in_ch", u(8)), out_port("out_ch", u(16)),
                             field("sum", u(16), c(5, u(16)))], [forever()])
    sig = predict_signature(comp)
    assert isinstance(sig, ir.BlockSignature) and sig.engine == "xls"
    assert [(ch.name, ch.direction, ch.payload.bits) for ch in sig.channels] == \
        [("in_ch", "in", 8), ("out_ch", "out", 16)]
    assert sig.channel("out_ch").ports == {"data": "out_ch", "valid": "out_ch_vld",
                                           "ready": "out_ch_rdy"}
    assert (sig.clock, sig.reset, sig.latency) == ("clk", "rst", None)
    flags = CodegenOptions(module_name="acc_blk").flags()
    assert "--streaming_channel_valid_suffix=_vld" in flags and "--module_name=acc_blk" in flags


SIG_TEXT = """\
# proto-file: xls/codegen/module_signature.proto
module_name: "acc"
clock_name: "clk"
reset { name: "rst" asynchronous: false active_low: false }
pipeline { latency: 1 initiation_interval: 1 }
channel_interfaces {
  channel_name: "in_ch"
  direction: CHANNEL_DIRECTION_RECEIVE
  type { type_enum: BITS bit_count: 8 }
  streaming { flow_control: CHANNEL_FLOW_CONTROL_READY_VALID
              data_port_name: "in_ch" ready_port_name: "in_ch_rdy" valid_port_name: "in_ch_vld" }
}
channel_interfaces {
  channel_name: "out_ch"
  direction: CHANNEL_DIRECTION_SEND
  type { type_enum: TUPLE tuple_elements { type_enum: BITS bit_count: 4 }
                          tuple_elements { type_enum: BITS bit_count: 12 } }
  streaming { flow_control: CHANNEL_FLOW_CONTROL_READY_VALID
              data_port_name: "out_ch" ready_port_name: "out_ch_rdy" valid_port_name: "out_ch_vld" }
}
"""


def _acc():
    return component("acc", [in_port("in_ch", u(8)), out_port("out_ch", u(16)),
                             field("sum", u(16), c(5, u(16)))], [forever()])


def test_signature_from_codegen_text():
    sig = signature_from_codegen(SIG_TEXT)
    assert (sig.name, sig.clock, sig.reset, sig.reset_active_low) == ("acc", "clk", "rst", False)
    assert (sig.latency, sig.initiation_interval) == (1, 1)
    assert [(ch.name, ch.direction, ch.payload.bits) for ch in sig.channels] == \
        [("in_ch", "in", 8), ("out_ch", "out", 16)]
    assert signature_mismatches(predict_signature(_acc()), sig) == []


def test_signature_mismatches_are_reported():
    sig = signature_from_codegen(SIG_TEXT.replace('"in_ch_rdy"', '"in_ch_ready"')
                                 .replace("bit_count: 8", "bit_count: 9"))
    diffs = signature_mismatches(predict_signature(_acc()), sig)
    assert any("in_ch ports" in d for d in diffs)
    assert any("in_ch payload: predicted 8 bits, codegen 9" in d for d in diffs)
    assert signature_mismatches(predict_signature(_acc(), CodegenOptions(reset=None)),
                                signature_from_codegen(SIG_TEXT)) == \
        ["reset: predicted None, codegen 'rst'"]


@pytest.mark.xls_codegen
def test_x1_3_codegen_signature_matches_prediction():
    # X1-3: XLS codegen, run with CodegenOptions.flags(), reports the predicted ports.
    comp = component("acc", [in_port("in_ch", u(8)), out_port("out_ch", u(16)),
                             field("sum", u(16), c(5, u(16)))], [forever(
        decl("x", u(8)),
        assign(l("x"), get(0)),
        decl("s", u(16), bop(fld(2), "Add", cast(u(16), l("x")))),
        put(1, l("s")),
        assign(fld(2), l("s")))])
    for opts in (CodegenOptions(), CodegenOptions(module_name="acc_blk", reset="rst_n",
                                                  reset_active_low=True,
                                                  data_suffix="_data", valid_suffix="_valid",
                                                  ready_suffix="_ready")):
        res = run_codegen(format_package(lower_component(comp)), opts, stages=2)
        assert signature_mismatches(predict_signature(comp, opts), res.signature) == []
        assert res.signature.latency is not None
        name = opts.module_name or "acc"
        assert f"module {name}(" in res.verilog


def test_actual_signature_takes_sv_types_from_the_prediction():
    from zuspec.be.xls.signature import SignatureMismatch, actual_signature
    comp = _acc()
    comp.fields[0].pragmas["sv_type"] = "byte_t"
    pred = predict_signature(comp)
    sig = actual_signature(pred, signature_from_codegen(SIG_TEXT), sources=["acc.sv"])
    assert sig.channel("in_ch").payload_sv_type == "byte_t"
    assert sig.sources == ["acc.sv"] and sig.latency == 1
    with pytest.raises(SignatureMismatch, match="in_ch payload"):
        actual_signature(pred, signature_from_codegen(SIG_TEXT.replace("bit_count: 8", "bit_count: 7")))
