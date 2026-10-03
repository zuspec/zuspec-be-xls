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
