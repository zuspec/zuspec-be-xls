"""Several components as one multi-proc XLS package (fw-hdl RQ-F7).

Each member class is lowered to a proc of its own (one per class, shared by
its instances); a top proc named after the group declares the channels
between members as in-package ``chan``s and instantiates each member with
``proc_instantiation``. Codegen with ``--multi_proc`` then gives one module,
with the group's external channels as its ports.

    text = lower_group("grp", members, internal, external)

* *members*: ``[(instance, comp, proc_name)]``;
* *internal*: ``[{"name", "depth", "a": (inst, port), "b": (inst, port)}]``,
  a channel from member port ``a`` (send) to member port ``b`` (receive);
* *external*: ``[{"name", "inst", "port", "flop_kind"}]``, a member port that
  becomes a port of the group (its direction is the member's).

XLS takes one schedule for the package (``#[initiation_interval]`` on a
non-top proc has no effect), so the group shares one set of codegen options.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import zuspec.ir.core as ir

from . import xir
from .lower_proc import lower_component
from .printer import format_package

#: The codegen flags a multi-proc package needs: fw-hdl supplies the FIFO.
MULTI_PROC_FLAGS = ["--multi_proc", "--fifo_module=fw_xls_fifo_wrapper"]


class GroupError(Exception):
    pass


def _ci(name: str, direction: str, kind: str = "streaming", flop: str = "none") -> str:
    return (f"  chan_interface {name}(direction={direction}, kind={kind}, "
            f"strictness=proven_mutually_exclusive, flow_control=ready_valid, "
            f"flop_kind={flop})")


def lower_group(name: str, members: Sequence[Tuple[str, ir.DataTypeComponent, str]],
                internal: Sequence[dict], external: Sequence[dict]) -> str:
    pkg = xir.Package(name=name)
    procs: Dict[str, xir.Proc] = {}
    inst_proc: Dict[str, xir.Proc] = {}
    for inst, comp, proc_name in members:
        if proc_name not in procs:
            lower_component(comp, pkg, proc_name=proc_name)
            procs[proc_name] = pkg.procs[-1]
        inst_proc[inst] = procs[proc_name]
    for p in procs.values():
        for c in p.channels:
            if c.kind != "streaming":
                raise GroupError(f"group {name}: {p.name}.{c.name} is a {c.kind} channel; a "
                                 f"multi-proc group carries streams only")
    pkg.top = None
    # Each member port is bound to exactly one top-level channel.
    bound: Dict[Tuple[str, str], str] = {}
    for e in external:
        bound[(e["inst"], e["port"])] = e["name"]
    for c in internal:
        bound[tuple(c["a"])] = c["name"]
        bound[tuple(c["b"])] = c["name"]

    def ch_of(inst: str, port: str) -> xir.ChannelInterface:
        return inst_proc[inst].channel(port)

    hdr, lines = [], []
    for e in external:
        c = ch_of(e["inst"], e["port"])
        d = "in" if c.direction == "receive" else "out"
        hdr.append(f"{e['name']}: {c.type} {d}")
        lines.append(_ci(e["name"], c.direction, flop=e.get("flop_kind") or "none"))
    for c in internal:
        a, b = ch_of(*c["a"]), ch_of(*c["b"])
        if a.direction != "send" or b.direction != "receive":
            raise GroupError(f"group {name}: channel {c['name']} must run from a send to a "
                             f"receive")
        if str(a.type) != str(b.type):
            raise GroupError(f"group {name}: channel {c['name']}: {a.type} vs {b.type}")
        depth = int(c["depth"])
        lines += [f"  chan {c['name']}({a.type}, id={pkg.new_id()}, kind=streaming, "
                  f"ops=send_receive, flow_control=ready_valid, "
                  f"strictness=proven_mutually_exclusive, fifo_depth={depth}, "
                  f"bypass={'true' if depth == 0 else 'false'}, "
                  f"register_push_outputs=false, register_pop_outputs=false)",
                  _ci(c["name"], "send"), _ci(c["name"], "receive")]
    for inst, comp, proc_name in members:
        p = inst_proc[inst]
        args = []
        for c in p.channels:
            ch = bound.get((inst, c.name))
            if ch is None:
                raise GroupError(f"group {name}: {inst}.{c.name} is not bound")
            args.append(ch)
        lines.append(f"  proc_instantiation {inst}({', '.join(args)}, proc={proc_name})")
    i0, i1, i2 = pkg.new_id(), pkg.new_id(), pkg.new_id()
    lines += [f"  st: () = state_read(state_element=st, id={i0})",
              f"  u: () = tuple(id={i1})",
              f"  nv: () = next_value(state_element=st, value=u, id={i2})"]
    top = [f"top proc {name}<{', '.join(hdr)}>(st: (), init={{()}}) {{"] + lines + ["}"]
    return format_package(pkg).rstrip("\n") + "\n\n" + "\n".join(top) + "\n"


def group_component(name: str, members: Dict[str, ir.DataTypeComponent],
                    external: Sequence[dict]) -> ir.DataTypeComponent:
    """A component with the group's external ports, for signature prediction."""
    fields: List[ir.Field] = []
    for e in external:
        comp = members[e["inst"]]
        f = next(f for f in comp.fields if f.name == e["port"])
        fields.append(ir.Field(name=e["name"], datatype=f.datatype, kind=ir.FieldKind.Port,
                               pragmas=dict(f.pragmas)))
    return ir.DataTypeComponent(name=name, super=None, fields=fields)
