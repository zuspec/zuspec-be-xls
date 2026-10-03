"""Lower hand-built IR and run it through every checker we have.

``run_fn`` lowers, type-checks, prints, re-parses and interprets. When
``opt_main`` is available it also evaluates the same vectors with XLS and
requires the same answers (the X1-2 check, functions only).
"""
from typing import List, Sequence

from zuspec.be.xls import xir
from zuspec.be.xls.lower_fn import lower_functions
from zuspec.be.xls.lower_proc import lower_component
from zuspec.be.xls.printer import format_package
from zuspec.be.xls.testing.interp import Interpreter, run_proc
from zuspec.be.xls.testing.parser import parse_package
from zuspec.be.xls.testing.typecheck import check_package
from zuspec.be.xls.testing.xlstools import opt_main, run_opt, xls_eval_fn_batch


def roundtrip(pkg: xir.Package) -> xir.Package:
    check_package(pkg)
    text = format_package(pkg)
    pkg2 = parse_package(text)
    check_package(pkg2)
    assert format_package(pkg2) == text
    if opt_main() is not None:
        run_opt(text)          # XLS parses and verifies it (X1-1)
    return pkg2


def lower_fns(fns, top=None) -> xir.Package:
    pkg = xir.Package(name="t")
    lower_functions(pkg, fns)
    pkg.top = top or fns[-1].name
    return pkg


def run_fn(fns, name: str, vectors: Sequence[Sequence[xir.Value]],
           xls: bool = True) -> List[xir.Value]:
    pkg = lower_fns(fns, name)
    pkg2 = roundtrip(pkg)
    it = Interpreter(pkg2)
    got = [it.call(name, list(v)) for v in vectors]
    if xls and opt_main() is not None:
        want = xls_eval_fn_batch(pkg, name, vectors)
        for v, g, x in zip(vectors, got, want):
            assert g == x, f"{name}{tuple(v)}: interp={g} xls={x}"
    return got


def run_comp(comp, inputs, **kw):
    pkg = lower_component(comp)
    pkg2 = roundtrip(pkg)
    return pkg, run_proc(pkg2, pkg2.top, inputs, **kw)
