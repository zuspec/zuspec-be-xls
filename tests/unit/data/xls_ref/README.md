# XLS IR syntax references

`*.ir` here is printed by XLS itself (xlsynth v0.59.0): `ir_converter_main` on the
DSLX sources next to it, and `*.opt.ir` by `opt_main`. They are the syntax
reference for `printer.py` and the input for the parser tests. Regenerate with:

    ir_converter_main --dslx_stdlib_path=<stdlib> --top=acc acc.x > acc_proc.ir
    ir_converter_main --dslx_stdlib_path=<stdlib> --top=f acc.x > acc_fn.ir
    ir_converter_main --dslx_stdlib_path=<stdlib> --top=<t> p2.x > p2_<t>.ir   # t = twostate, nostate, g
    opt_main p2_twostate.ir > p2_twostate.opt.ir
