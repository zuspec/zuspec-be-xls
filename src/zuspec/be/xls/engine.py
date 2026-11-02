"""XLS as a Zuspec synthesis engine (``zuspec.ir.core.engine``).

``XlsEngine.build`` lowers the IR to XLS IR, runs ``opt_main`` and
``codegen_main`` with the same command ``synth.xls.Codegen`` runs (from
dv-flow-libsynth, which owns the XLS tool knowledge; imported only here), and
returns the RTL with an overlay of XLS's results on the IR's anchors.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import types
from typing import Optional, Sequence

import zuspec.ir.core as ir
from zuspec.ir.core.anchor import AnchorTable
from zuspec.ir.core.engine import BuildResult, Constraints, Reason, Verdict

from . import xir
from .lower_fn import lower_functions
from .lower_proc import lower_component
from .overlay import xls_overlay
from .printer import format_package
from .testing.xlstools import find_tool
from .width import StaticSubsetError


class XlsEngineError(Exception):
    pass


def _lower(roots: Sequence[ir.Base], top: str) -> xir.Package:
    named = {getattr(r, "name", None): r for r in roots}
    node = named.get(top)
    if isinstance(node, ir.DataTypeComponent):
        return lower_component(node)
    fns = [r for r in roots if isinstance(r, ir.Function)]
    if node is None or not fns:
        raise XlsEngineError(f"no component or function named {top!r}")
    pkg = xir.Package(name=top)
    lower_functions(pkg, fns)
    pkg.top = top
    return pkg


class XlsEngine:
    name = "xls"

    def version(self) -> Optional[str]:
        """The xlsynth release, from the manifest of the package that ships
        the tools (edapack ``xlsynth-bin``), when there is one."""
        cg = find_tool("codegen_main")
        if cg is None:
            return None
        man = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(cg))),
                           "manifest.json")
        try:
            with open(man) as f:
                return json.load(f)["release"]["version"]
        except (OSError, KeyError, ValueError):
            return None

    def accepts(self, roots: Sequence[ir.Base], top: str) -> Verdict:
        try:
            _lower(roots, top)
        except StaticSubsetError as e:
            return Verdict(False, [Reason(str(e))])
        except XlsEngineError as e:
            return Verdict(False, [Reason(str(e))])
        return Verdict(True)

    def build(self, roots: Sequence[ir.Base], top: str, anchors: AnchorTable,
              constraints: Constraints, workdir: str) -> BuildResult:
        from dv_flow.libsynth.xls.codegen import (CODEGEN_DEFAULTS, codegen_argv,
                                                  fix_source_comments)
        opt, cg = find_tool("opt_main"), find_tool("codegen_main")
        if opt is None or cg is None:
            raise XlsEngineError("opt_main / codegen_main not found on PATH or in $XLS_BIN")
        params = dict(CODEGEN_DEFAULTS)
        unknown = set(constraints.extra) - set(params)
        if unknown:
            raise XlsEngineError(f"unknown XLS options {sorted(unknown)}; "
                                 f"known: {sorted(params)}")
        params.update(constraints.extra)
        if constraints.pipeline_stages is not None:
            params["pipeline_stages"] = constraints.pipeline_stages
        if constraints.clock_period_ps is not None:
            params["clock_period_ps"] = constraints.clock_period_ps
        if constraints.reset is not None:
            params["reset"] = constraints.reset
            params["reset_active_low"] = constraints.reset_active_low
        p = types.SimpleNamespace(**params)

        pkg = _lower(roots, top)
        module = constraints.module_name or p.module_name or re.sub(r"\W", "_", pkg.top or top)
        os.makedirs(workdir, exist_ok=True)
        path = {k: os.path.join(workdir, f"{module}{ext}") for k, ext in (
            ("ir", ".ir"), ("opt_ir", ".opt.ir"), ("verilog", ".sv"),
            ("signature", ".sig.textproto"), ("schedule", ".schedule.textproto"),
            ("schedule_ir", ".sched.ir"), ("block_ir", ".block.ir"),
            ("line_map", ".line_map.textproto"))}
        log = os.path.join(workdir, f"{module}.xls.log")
        with open(path["ir"], "w") as f:
            f.write(format_package(pkg))
        cmds = [[opt, path["ir"], f"--output_path={path['opt_ir']}"],
                codegen_argv(cg, path["opt_ir"], module, p, p.clock_period_ps,
                             path["schedule"], path["verilog"]) +
                [f"--output_signature_path={path['signature']}",
                 f"--output_schedule_ir_path={path['schedule_ir']}",
                 f"--output_block_ir_path={path['block_ir']}",
                 f"--output_verilog_line_map_path={path['line_map']}"]]
        with open(log, "w") as lf:
            for cmd in cmds:
                lf.write(" ".join(cmd) + "\n")
                lf.flush()
                r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=workdir)
                if r.returncode != 0:
                    raise XlsEngineError(f"{os.path.basename(cmd[0])} failed (see {log})")
        with open(path["opt_ir"]) as f:
            opt_text = f.read()
        if p.source_annotations:
            fix_source_comments(path["verilog"], opt_text, workdir)

        def read(k):
            with open(path[k]) as f:
                return f.read()
        o = xls_overlay(anchors, ir_text=read("schedule_ir"), schedule_text=read("schedule"),
                        signature_text=read("signature"), block_ir_text=read("block_ir"),
                        top=pkg.top, block_anchor=next(
                            (r.anchor for r in roots if getattr(r, "name", None) == top
                             and r.anchor), None),
                        constraints=constraints.to_json(), command=" ".join(cmds[-1]),
                        engine_version=self.version(), artifacts=dict(path))
        return BuildResult(rtl=[path["verilog"]], top_module=module, overlay=o,
                           artifacts=dict(path), log=log)
