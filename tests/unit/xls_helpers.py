"""Helpers for building one-function packages in tests."""
from typing import Callable, List, Sequence

from zuspec.be.xls import xir


def one_fn(param_types: Sequence[xir.Type],
           body: Callable[[xir.Builder, List[xir.Node]], xir.Node],
           name: str = "f") -> xir.Package:
    pkg = xir.Package(name="t")
    f = xir.Function(name=name)
    b = xir.Builder(pkg, f)
    ps = [b.param(f"p{i}", t) for i, t in enumerate(param_types)]
    f.ret = body(b, ps)
    if f.ret in f.params:
        f.ret = b.unop("identity", f.ret)
    pkg.functions.append(f)
    pkg.top = name
    return pkg
