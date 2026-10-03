"""The block signature we expect XLS codegen to produce (``xls-phase0.md`` BX-8).

This is a prediction: one ready/valid port triple per channel, named by the
codegen naming flags, plus clock and reset. X1-3 runs ``codegen_main`` with
exactly :meth:`CodegenOptions.flags`, reads its ``module_signature`` with
:func:`signature_from_codegen` and compares the two with
:func:`signature_mismatches`.

Latency and initiation interval come from XLS's scheduler, so they are
``None`` here.
"""
from __future__ import annotations

import dataclasses as dc
import re
from typing import Any, Dict, List, Optional, Tuple

import zuspec.ir.core as ir


@dc.dataclass
class CodegenOptions:
    """The ``codegen_main`` naming options the prediction assumes."""
    module_name: Optional[str] = None
    clock: str = "clk"
    reset: Optional[str] = "rst"
    reset_active_low: bool = False
    data_suffix: str = ""
    valid_suffix: str = "_vld"
    ready_suffix: str = "_rdy"

    def flags(self) -> List[str]:
        out = [f"--streaming_channel_data_suffix={self.data_suffix}",
               f"--streaming_channel_valid_suffix={self.valid_suffix}",
               f"--streaming_channel_ready_suffix={self.ready_suffix}"]
        if self.module_name:
            out.append(f"--module_name={self.module_name}")
        if self.reset:
            out.append(f"--reset={self.reset}")
            out.append(f"--reset_active_low={'true' if self.reset_active_low else 'false'}")
        return out


def predict_signature(comp: ir.DataTypeComponent,
                      opts: Optional[CodegenOptions] = None) -> ir.BlockSignature:
    """The block signature for *comp* lowered by :func:`lower_component`."""
    opts = opts or CodegenOptions()
    sig = ir.BlockSignature(name=opts.module_name or comp.name, clock=opts.clock,
                            reset=opts.reset, reset_active_low=opts.reset_active_low,
                            engine="xls")
    for f in comp.fields:
        if not isinstance(f.datatype, (ir.DataTypeGetIF, ir.DataTypePutIF)):
            continue
        direction = "in" if isinstance(f.datatype, ir.DataTypeGetIF) else "out"
        sig.channels.append(ir.ChannelSignature(
            name=f.name, direction=direction, payload=f.datatype.element_type,
            flow_control="ready_valid",
            ports={"data": f.name + opts.data_suffix,
                   "valid": f.name + opts.valid_suffix,
                   "ready": f.name + opts.ready_suffix},
            payload_sv_type=f.pragmas.get("sv_type")))
    return sig


# ---------------------------------------------------------------------------
# The signature XLS reports (codegen_main --output_signature_path)
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r'\s*(?:(#[^\n]*)|([A-Za-z_][\w.]*)|("(?:[^"\\]|\\.)*")|(-?\d+)|([{}:]))')


def parse_textproto(text: str) -> Dict[str, List[Any]]:
    """A minimal protobuf text-format reader: every field maps to the list of its
    values; a message value is itself such a dict."""
    toks: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            if text[pos:].strip():
                raise ValueError(f"textproto: unexpected {text[pos:pos + 20]!r}")
            break
        pos = m.end()
        if m.group(1):
            continue
        if m.group(2):
            toks.append(("id", m.group(2)))
        elif m.group(3):
            toks.append(("str", m.group(3)[1:-1]))
        elif m.group(4):
            toks.append(("int", m.group(4)))
        else:
            toks.append(("p", m.group(5)))
    i = 0

    def message(end: Optional[str]) -> Dict[str, List[Any]]:
        nonlocal i
        out: Dict[str, List[Any]] = {}
        while i < len(toks) and toks[i] != ("p", end):
            kind, name = toks[i]
            if kind != "id":
                raise ValueError(f"textproto: expected a field name, got {name!r}")
            i += 1
            if toks[i] == ("p", ":"):
                i += 1
            if toks[i] == ("p", "{"):
                i += 1
                val: Any = message("}")
                i += 1
            else:
                kind, v = toks[i]
                i += 1
                val = int(v) if kind == "int" else \
                    {"true": True, "false": False}.get(v, v) if kind == "id" else v
            out.setdefault(name, []).append(val)
        return out

    return message(None)


def _one(msg: Dict[str, List[Any]], key: str, default: Any = None) -> Any:
    v = msg.get(key)
    return v[0] if v else default


def _type_bits(t: Dict[str, List[Any]]) -> Optional[int]:
    kind = _one(t, "type_enum")
    if kind == "BITS":
        return _one(t, "bit_count", 0)
    if kind == "TUPLE":
        return sum(_type_bits(e) or 0 for e in t.get("tuple_elements", []))
    if kind == "ARRAY":
        return _one(t, "array_size", 0) * (_type_bits(_one(t, "array_element")) or 0)
    return None


_FLOW = {"CHANNEL_FLOW_CONTROL_READY_VALID": "ready_valid",
         "CHANNEL_FLOW_CONTROL_VALID": "valid", "CHANNEL_FLOW_CONTROL_NONE": "none"}


def signature_from_codegen(text: str) -> ir.BlockSignature:
    """Read the ``ModuleSignatureProto`` text that ``codegen_main`` writes.

    A channel's payload is an unsigned ``DataTypeInt`` of the flattened width;
    XLS does not know the SV type behind it."""
    sig = parse_textproto(text)
    reset = _one(sig, "reset", {})
    pipe = _one(sig, "pipeline", {})
    out = ir.BlockSignature(name=_one(sig, "module_name"), clock=_one(sig, "clock_name"),
                            reset=_one(reset, "name") if reset else None,
                            reset_active_low=bool(_one(reset, "active_low", False)),
                            latency=_one(pipe, "latency") if pipe else None,
                            initiation_interval=_one(pipe, "initiation_interval")
                            if pipe else None, engine="xls")
    for ch in sig.get("channel_interfaces", []):
        stream = _one(ch, "streaming", {})
        ports = {role: _one(stream, f"{role}_port_name")
                 for role in ("data", "valid", "ready")}
        out.channels.append(ir.ChannelSignature(
            name=_one(ch, "channel_name"),
            direction="in" if _one(ch, "direction") == "CHANNEL_DIRECTION_RECEIVE" else "out",
            payload=ir.DataTypeInt(bits=_type_bits(_one(ch, "type", {})), signed=False),
            flow_control=_FLOW.get(_one(stream, "flow_control"), "none"),
            ports={k: v for k, v in ports.items() if v is not None}))
    return out


def _payload_bits(t: ir.DataType) -> Optional[int]:
    bits = getattr(t, "bits", None)
    return bits if isinstance(bits, int) and bits >= 0 else None


def signature_mismatches(predicted: ir.BlockSignature,
                         actual: ir.BlockSignature) -> List[str]:
    """Where *actual* differs from *predicted*; empty when they agree.

    Latency and initiation interval are not compared: the prediction leaves them
    open (``None``), and the scheduler decides them."""
    diffs: List[str] = []
    for attr in ("name", "clock", "reset", "reset_active_low"):
        p, a = getattr(predicted, attr), getattr(actual, attr)
        if p != a:
            diffs.append(f"{attr}: predicted {p!r}, codegen {a!r}")
    pch = {c.name: c for c in predicted.channels}
    ach = {c.name: c for c in actual.channels}
    for name in sorted(pch.keys() - ach.keys()):
        diffs.append(f"channel {name}: predicted, missing from codegen")
    for name in sorted(ach.keys() - pch.keys()):
        diffs.append(f"channel {name}: in codegen, not predicted")
    for name in sorted(pch.keys() & ach.keys()):
        p, a = pch[name], ach[name]
        for attr in ("direction", "flow_control", "ports"):
            if getattr(p, attr) != getattr(a, attr):
                diffs.append(f"channel {name} {attr}: predicted {getattr(p, attr)!r}, "
                             f"codegen {getattr(a, attr)!r}")
        pb, ab = _payload_bits(p.payload), _payload_bits(a.payload)
        if pb != ab:
            diffs.append(f"channel {name} payload: predicted {pb} bits, codegen {ab}")
    return diffs


class SignatureMismatch(Exception):
    pass


def actual_signature(predicted: ir.BlockSignature, codegen: ir.BlockSignature,
                     sources: Optional[List[str]] = None) -> ir.BlockSignature:
    """The signature the integration step uses (xls-phase2.md B3).

    It is XLS's own signature, after checking that it matches *predicted*. XLS
    does not know the source-level payload type names, so those are taken from
    the prediction. The module's *sources* are recorded too."""
    diffs = signature_mismatches(predicted, codegen)
    if diffs:
        raise SignatureMismatch(f"{predicted.name}: XLS codegen's signature differs from "
                                f"the prediction:\n  " + "\n  ".join(diffs))
    out = dc.replace(codegen, channels=[dc.replace(c) for c in codegen.channels],
                     sources=list(sources or []), engine="xls")
    for c in out.channels:
        c.payload_sv_type = predicted.channel(c.name).payload_sv_type
    return out
