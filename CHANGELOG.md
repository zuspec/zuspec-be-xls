# Changelog

## Unreleased (Phase 0: results as IR)

### Added
- **`xls_overlay`** (`overlay.py`, `docs/overlay-xls.md`): XLS's schedule, timing, interface and registers as an overlay on the IR's anchors.
  - Positions are resolved in this order: exact, merged, partial, then unmapped.
  - Per-stage critical paths.
  - Coverage in the header.
  - The `xls.node_stage` fact gives the schedule node by node.
- **`XlsEngine`** (`engine.py`), registered as the `zuspec.engines` entry point `xls`.
- **`read_schedule`**, a `PackageScheduleProto` reader.
- **dv-flow task `zuspec.xls.Overlay`**, which writes `<module>.zir/` bundles.

### Changed
- `signature_from_codegen` fills in channel `stage` and `flop_kind`.

### Fixed
- A division in the `else` branch of `if (d == 0)` is now accepted as guarded.
