"""Read the ``PackageScheduleProto`` text that ``codegen_main`` writes
(``--output_schedule_path``)."""
from __future__ import annotations

import dataclasses as dc
from typing import Dict, List, Optional

from .signature import parse_textproto


@dc.dataclass
class TimedNode:
    node: str
    stage: int
    node_delay_ps: Optional[int] = None
    path_delay_ps: Optional[int] = None


@dc.dataclass
class Schedule:
    """One function's or proc's schedule."""
    function: str
    length: int
    min_clock_period_ps: Optional[int] = None
    target_clock_period_ps: Optional[int] = None
    #: Node name -> its stage and delays.
    nodes: Dict[str, TimedNode] = dc.field(default_factory=dict)
    #: Every stage the schedule lists, including ones with no nodes.
    stage_ids: List[int] = dc.field(default_factory=list)

    @property
    def stages(self) -> Dict[int, List[str]]:
        """Stage -> node names, in schedule order (empty stages included)."""
        out: Dict[int, List[str]] = {k: [] for k in self.stage_ids}
        for n in self.nodes.values():
            out.setdefault(n.stage, []).append(n.node)
        return out


def _one(d, key, default=None):
    v = d.get(key)
    return v[0] if v else default


def parse_schedules(text: str) -> Dict[str, Schedule]:
    """Function or proc name -> :class:`Schedule`."""
    out: Dict[str, Schedule] = {}
    for entry in parse_textproto(text).get("schedules", []):
        v = _one(entry, "value", {})
        s = Schedule(function=_one(v, "function", _one(entry, "key")),
                     length=_one(v, "length", 0),
                     min_clock_period_ps=_one(v, "min_clock_period_ps"),
                     target_clock_period_ps=_one(v, "target_clock_period_ps"))
        for st in v.get("stages", []):
            k = _one(st, "stage", 0)
            if k not in s.stage_ids:
                s.stage_ids.append(k)
            for tn in st.get("timed_nodes", []):
                name = _one(tn, "node")
                s.nodes[name] = TimedNode(name, k, _one(tn, "node_delay_ps"),
                                          _one(tn, "path_delay_ps"))
            # Older schedules list bare node names.
            for name in st.get("nodes", []):
                s.nodes.setdefault(name, TimedNode(name, k))
        out[s.function] = s
    return out
