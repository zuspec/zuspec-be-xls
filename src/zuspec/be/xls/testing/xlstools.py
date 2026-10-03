"""Optional access to real XLS tools, for checks that X0 can do early.

X0 does not need XLS. When ``opt_main`` is available (``$XLS_BIN/opt_main``
or on ``PATH``), tests marked ``xls`` use it for two X1 checks ahead of time:

* **syntax** (X1-1): XLS parses and verifies what we print;
* **function semantics** (X1-2, functions only): the release ships no
  ``eval_ir_main``, but ``opt_main`` constant-folds a call whose arguments are
  all literals. :func:`xls_eval_fn` wraps the function under test in a
  ``top fn`` that invokes it on literal arguments and reads back the folded
  ``ret literal``.

When ``codegen_main`` is available as well, :func:`run_codegen` turns a package
into Verilog plus XLS's module signature. That gives two more X1 checks:

* **signature** (X1-3): the reported signature matches
  :func:`~zuspec.be.xls.signature.predict_signature`;
* **proc semantics** (X1-2, procs): the generated RTL, simulated, produces the
  same output streams as our interpreter.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import dataclasses as dc
import tempfile
from typing import List, Optional, Sequence, Tuple

import zuspec.ir.core as ir

from .. import xir
from ..printer import format_package
from ..signature import CodegenOptions, signature_from_codegen
from .parser import parse_package


class XlsToolError(Exception):
    pass


def find_tool(name: str) -> Optional[str]:
    d = os.environ.get("XLS_BIN")
    if d:
        p = os.path.join(d, name)
        if os.access(p, os.X_OK):
            return p
    return shutil.which(name)


def opt_main() -> Optional[str]:
    return find_tool("opt_main")


def codegen_main() -> Optional[str]:
    return find_tool("codegen_main")


def run_opt(text: str, extra: Sequence[str] = ()) -> str:
    tool = opt_main()
    if tool is None:
        raise XlsToolError("opt_main not found (set XLS_BIN or add it to PATH)")
    with tempfile.NamedTemporaryFile("w", suffix=".ir", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        r = subprocess.run([tool, path, *extra], capture_output=True, text=True)
    finally:
        os.unlink(path)
    if r.returncode != 0:
        raise XlsToolError(f"opt_main failed:\n{r.stderr}\n--- input ---\n{text}")
    return r.stdout


def xls_eval_fn(pkg: xir.Package, fn: str, args: Sequence[xir.Value]) -> xir.Value:
    """Evaluate ``fn(*args)`` with XLS by constant folding."""
    return xls_eval_fn_batch(pkg, fn, [args])[0]


def xls_eval_fn_batch(pkg: xir.Package, fn: str,
                      arg_sets: Sequence[Sequence[xir.Value]]) -> List[xir.Value]:
    """Evaluate ``fn`` on each argument tuple in one ``opt_main`` run.

    The wrapper returns a tuple with one ``invoke`` per argument tuple; XLS
    folds the whole tuple to one literal.
    """
    return xls_eval_calls(pkg, [(fn, a) for a in arg_sets])


def xls_eval_calls(pkg: xir.Package, calls: Sequence[Tuple[str, Sequence[xir.Value]]],
                   chunk: int = 4000) -> List[xir.Value]:
    """Evaluate many ``(function, args)`` calls, ``chunk`` per ``opt_main`` run."""
    out: List[xir.Value] = []
    for i in range(0, len(calls), chunk):
        out += _xls_eval_chunk(pkg, calls[i:i + chunk])
    return out


def _xls_eval_chunk(pkg: xir.Package, calls) -> List[xir.Value]:
    wrapper = xir.Function(name="__x0_eval")
    b = xir.Builder(pkg, wrapper)
    nodes: List[xir.Node] = []
    for fn, args in calls:
        f = pkg.function(fn)
        lits = [b.literal(a, p.type) for p, a in zip(f.params, args)]
        nodes.append(b.invoke(f, lits))
    wrapper.ret = b.tuple(*nodes)
    saved_top = pkg.top
    pkg.functions.append(wrapper)
    pkg.top = wrapper.name
    try:
        text = format_package(pkg)
    finally:
        pkg.functions.remove(wrapper)
        pkg.top = saved_top
    out = parse_package(run_opt(text))
    top = out.function("__x0_eval")
    for n in top.nodes:
        # A failing assert survives folding with a literal-0 condition.
        if n.op == "assert" and n.operands[1].op == "literal" \
                and n.operands[1].attrs["value"] == 0:
            raise XlsToolError(f"assertion failed in XLS: {n.attrs.get('message')}")
    if top.ret is None or top.ret.op != "literal":
        raise XlsToolError(f"opt_main did not fold the calls to a literal:\n{text}")
    return list(top.ret.attrs["value"])


@dc.dataclass
class CodegenResult:
    verilog: str
    signature: ir.BlockSignature
    signature_text: str
    #: The optimized IR that codegen consumed.
    opt_ir: str


def run_codegen(text: str, opts: Optional[CodegenOptions] = None, stages: int = 1,
                extra: Sequence[str] = ()) -> CodegenResult:
    """``opt_main`` then ``codegen_main`` on the package *text*.

    Scheduling uses the ``unit`` delay model and a fixed stage count, so the
    result does not depend on a process library. The output is
    SystemVerilog."""
    tool = codegen_main()
    if tool is None:
        raise XlsToolError("codegen_main not found (set XLS_BIN or add it to PATH)")
    opts = opts or CodegenOptions()
    opt_ir = run_opt(text)
    with tempfile.TemporaryDirectory() as d:
        ir_path = os.path.join(d, "in.ir")
        sv_path = os.path.join(d, "out.sv")
        sig_path = os.path.join(d, "sig.textproto")
        with open(ir_path, "w") as f:
            f.write(opt_ir)
        cmd = [tool, ir_path, "--generator=pipeline", f"--pipeline_stages={stages}",
               "--delay_model=unit", "--use_system_verilog=true",
               f"--output_verilog_path={sv_path}", f"--output_signature_path={sig_path}",
               *opts.flags(), *extra]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise XlsToolError(f"codegen_main failed:\n{r.stderr}\n--- input ---\n{opt_ir}")
        with open(sv_path) as f:
            verilog = f.read()
        with open(sig_path) as f:
            sig_text = f.read()
    return CodegenResult(verilog, signature_from_codegen(sig_text), sig_text, opt_ir)
