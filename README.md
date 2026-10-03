# zuspec-be-xls

Lowers the zuspec IR *static subset* to [XLS](https://github.com/google/xls) IR. Pure
functions become XLS `fn`s; a component's `run()` (`<prologue>; forever <activation>`)
becomes an XLS `proc` with channels, state elements and predicated effects.

The SV meaning of the source is normative. Where an XLS op differs, the lowering adds
the difference explicitly: an out-of-range array read gives 0 (XLS clamps), a negative
signed select base reads 0 below bit 0, a divide must be guarded. The semantics are in
`fw-hdl/zuspec-semantics.md` §6.3.1, and the construct table is in
`fw-hdl/xls-phase0.md` §3.

```python
from zuspec.be.xls.lower_proc import lower_component
from zuspec.be.xls.lower_fn import lower_functions
from zuspec.be.xls.printer import format_package

pkg = lower_component(component_ir)        # an xir.Package with one top proc
print(format_package(pkg))                 # XLS IR text (xlsynth v0.59.0 syntax)
```

## Layout

| Module | Role |
|---|---|
| `xir.py` | The XLS IR object model and a `Builder` with the result-type rules. |
| `printer.py` | Prints `.ir` text in the syntax of the pinned release (`XLS_SYNTAX_VERSION`). |
| `width.py` | The static subset's bottom-up width and signedness rules (`infer`), and `StaticSubsetError`. |
| `lower_fn.py`, `lower_proc.py` | The IR → XLS lowering. Out-of-subset constructs are rejected here, with the source location. |
| `signature.py` | The predicted block signature and the codegen naming flags it assumes. |
| `testing/` | A stand-in for XLS: an IR parser written separately from the printer, a type checker and an interpreter. Plus `xlstools.py`, which uses a real `opt_main` when one is available. |

## Tests

Run `pytest tests`. Tests marked `xls` need `opt_main`, either at `$XLS_BIN/opt_main` or
on `PATH`; without it they skip. They do two things:

- XLS parses and verifies every package the tests emit;
- XLS evaluates functions by constant folding (`xlstools.xls_eval_calls`), and the
  results are compared with our interpreter.

Tests marked `xls_codegen` also need `codegen_main`. They run `xlstools.run_codegen`
(`opt_main`, then `codegen_main` with the `unit` delay model) and compare XLS's module
signature with `predict_signature`.

The SV-level conformance tests (Verilator vs this lowering) live in fw-hdl, in
`tests/xls`. With `codegen_main` present they also simulate XLS's generated RTL for
every proc.
