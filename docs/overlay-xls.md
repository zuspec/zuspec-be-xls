# XLS results as a Zuspec overlay

*Phase 0 item C. Code: `zuspec/be/xls/overlay.py` (`xls_overlay`) and `read_schedule.py`.
The model is described in zuspec-ir-core `docs/results.md`; anchors in `docs/anchors.md`.*

`xls_overlay(anchors, ir_text=…, schedule_text=…, signature_text=…, block_ir_text=…)`
turns what `codegen_main` writes for one block into facts on the anchors of the Zuspec
model the block was lowered from.

## Inputs (`codegen_main` outputs)

| Input | Flag | Gives |
|---|---|---|
| scheduled IR (or optimized IR) | `--output_schedule_ir_path` | nodes, operands, `pos` lists |
| schedule | `--output_schedule_path` | stage, `node_delay_ps`, `path_delay_ps` per node; `length`; clock period |
| module signature | `--output_signature_path` | the block interface; per-channel `flop_kind` |
| block IR (optional) | `--output_block_ir_path` | pipeline registers `p<s>_<node>` |

dv-flow-libsynth's `synth.xls.Codegen` writes all four. It also writes
`--output_verilog_line_map_path` (filetype `xlsVerilogLineMap`), which is not read yet.

## Mapping XLS nodes to anchors

The lowering stamps each XLS node with `pos` from its IR node's `Loc`
(`lower_fn.pos_of`). XLS keeps positions through optimization: the **first** entry is
where the operation is written, and later entries are the call sites it was inlined
through.

| First position names… | Anchor | Quality |
|---|---|---|
| an expression starting there | the outermost such expression | `exact` |
| a statement starting there | the innermost such statement | `exact` |
| a declaration (function, member) | it | `exact` |
| nothing there, but a statement on that line | that statement | `partial` |
| nothing | the block | `unmapped` |

- **`merged`:** later positions that name further non-call expressions in the live model
  (operations XLS merged) add anchors, and the quality becomes `merged`.
- **Inlining is not merging:** call-site positions never make a node `merged`.
- **Lossy by design:** XLS may drop or merge positions. That only limits what reports can
  attribute.

## Facts written

| Kind | On |
|---|---|
| `xls.node_name` | every mapped node's anchor (engine extra) |
| `xls.node_stage` | `{node, stage, order}` on each scheduled node's anchor, parameters on the block anchor (engine extra; `order` is the node's position in the schedule): the schedule by XLS node, since codegen adds nodes the schedule lacks and so `xls.node_name` and `sched.stage` rows do not pair up |
| `sched.stage`, `timing.node_delay_ps`, `timing.path_delay_ps` | each node's anchor(s) |
| `sched.stage_count`, `timing.clock_period_ps`, `timing.critical_path` (per stage) | the block |
| `iface.block` | the block (the `BlockSignature`, with channel stages filled from the schedule where the signature omits them) |
| `sched.ii` | the block; `reliable: false` for procs (XLS reports `value_or(1)`) |
| `iface.channel_stage`, `iface.channel_flop` | channel member anchors (`<block>.<channel>`), when anchored |
| `rtl.register` | `<anchor>@reg:<s>` for each `p<s>_<node>` register |

**Critical path.** Per stage, the path is walked back from the node with the largest
`path_delay_ps` through operands whose `path_delay_ps` equals
`path_delay_ps − node_delay_ps`.

## Coverage (overlay header)

The header records:
- `ops_total` and `ops_by_quality`
- `pct_ops_attributed`: `exact` + `merged`, as a share of operations
- `pct_crit_delay_attributed`: the same, as a share of critical-path delay

fw-hdl's `tests/xls/test_xls_overlay.py` keeps a golden copy
(`tests/xls/golden/mapping_coverage.json`) and fails on a drop of more than 2 points. This
catches regressions in our lowering; it is not a bar XLS must meet.

| Design | Ops attributed | Critical-path delay attributed |
|---|---|---|
| gcd_binary8 (function) | 100.0% | 100.0% |
| rle_enc8 (proc) | 62.5% | 44.4% |

**Why rle maps less.** It is a proc: flow-control and state plumbing that the proc
lowering creates without expression positions maps to statements (`partial`) or to the
block. This is the place to improve.

**Lowering oracle.** `tests/unit/test_overlay.py::test_every_lowered_node_maps_back_exactly`
checks that before optimization every node maps `exact`, so any loss is XLS's.
