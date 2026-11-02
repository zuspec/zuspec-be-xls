"""dv-flow task implementations for the zuspec.xls package."""
from __future__ import annotations

import os
from typing import Dict, List

from dv_flow.mgr import (FileSet, SeverityE, TaskDataInput, TaskDataResult, TaskMarker,
                         TaskRunCtxt)

import zuspec.ir.core as ir

from ..signature import SignatureMismatch, actual_signature, signature_from_codegen


def _files(input: TaskDataInput, filetype: str) -> List[str]:
    out = []
    for fs in getattr(input, "inputs", []) or []:
        if getattr(fs, "filetype", None) == filetype:
            out += [os.path.join(fs.basedir, f) for f in fs.files]
    return out


def _err(msg: str) -> TaskMarker:
    return TaskMarker(msg=msg, severity=SeverityE.Error)


async def Signature(ctxt: TaskRunCtxt, input: TaskDataInput) -> TaskDataResult:
    preds: Dict[str, ir.BlockSignature] = {}
    for p in _files(input, "blockSignaturePrediction"):
        s = ir.load_signature(p)
        preds[s.name] = s
    markers: List[TaskMarker] = []
    written: List[str] = []
    os.makedirs(input.rundir, exist_ok=True)
    for path in _files(input, "xlsModuleSignature"):
        with open(path) as f:
            cg = signature_from_codegen(f.read())
        pred = preds.pop(cg.name, None)
        if pred is None:
            markers.append(_err(f"no blockSignaturePrediction for module {cg.name!r}"))
            continue
        sv = os.path.join(os.path.dirname(path), f"{cg.name}.sv")
        try:
            sig = actual_signature(pred, cg, sources=[sv])
        except SignatureMismatch as e:
            markers.append(_err(str(e)))
            continue
        out = os.path.join(input.rundir, f"{cg.name}.sig.json")
        ir.save_signature(sig, out)
        written.append(os.path.basename(out))
    for name in preds:
        markers.append(_err(f"module {name!r} was predicted but XLS produced no signature "
                            f"for it"))
    status = 1 if markers else 0
    return TaskDataResult(status=status, changed=True, markers=markers, output=[
        FileSet(src=input.name, filetype="blockSignature", basedir=input.rundir,
                files=written)])


def _by_module(input: TaskDataInput, filetype: str) -> Dict[str, str]:
    """Module name -> file, for files whose FileSet names its module (one per
    file), else by the file's leading name (`<module>.<ext>`)."""
    out: Dict[str, str] = {}
    for fs in getattr(input, "inputs", []) or []:
        if getattr(fs, "filetype", None) != filetype:
            continue
        mod = (getattr(fs, "params", None) or {}).get("module")
        for f in fs.files:
            out[mod if mod and len(fs.files) == 1 else f.split(".")[0]] = \
                os.path.join(fs.basedir, f)
    return out


async def Overlay(ctxt: TaskRunCtxt, input: TaskDataInput) -> TaskDataResult:
    from zuspec.ir.core.anchor import AnchorTable
    from zuspec.ir.core.results import write_bundle

    from ..engine import XlsEngine
    from ..overlay import xls_overlay

    def read(path):
        with open(path) as f:
            return f.read()

    anchors = _by_module(input, "zuspecAnchorTable")
    sched_ir = _by_module(input, "xlsScheduledIR")
    opt_ir = _by_module(input, "xlsOptIR")
    sched = _by_module(input, "xlsSchedule")
    sigs = _by_module(input, "xlsModuleSignature")
    blocks = _by_module(input, "xlsBlockIR")
    markers: List[TaskMarker] = []
    written: List[str] = []
    os.makedirs(input.rundir, exist_ok=True)
    version = XlsEngine().version()
    for module, sig_path in sorted(sigs.items()):
        if module not in anchors:
            markers.append(_err(f"no zuspecAnchorTable for module {module!r}"))
            continue
        ir_path = sched_ir.get(module) or opt_ir.get(module)
        if ir_path is None:
            markers.append(_err(f"no xlsScheduledIR or xlsOptIR for module {module!r}"))
            continue
        table = AnchorTable.load(anchors[module])
        o = xls_overlay(table, ir_text=read(ir_path),
                        schedule_text=read(sched[module]) if module in sched else None,
                        signature_text=read(sig_path),
                        block_ir_text=read(blocks[module]) if module in blocks else None,
                        engine_version=version,
                        artifacts={"ir": ir_path, "signature": sig_path,
                                   **({"schedule": sched[module]} if module in sched else {}),
                                   **({"block_ir": blocks[module]} if module in blocks else {})})
        bundle = os.path.join(input.rundir, f"{module}.zir")
        write_bundle(bundle, table, [o])
        written.append(os.path.relpath(os.path.join(bundle, "manifest.json"), input.rundir))
    return TaskDataResult(status=1 if markers else 0, changed=True, markers=markers, output=[
        FileSet(src=input.name, filetype="zuspecIRBundle", basedir=input.rundir,
                files=written)])
