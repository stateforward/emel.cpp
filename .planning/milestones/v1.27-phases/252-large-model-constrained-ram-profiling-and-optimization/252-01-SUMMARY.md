---
phase: 252
plan: 01
status: complete
requirements-completed:
  - PERF-02
---

# Phase 252 Summary

## Completed

- Added maintained generation load-profile reporting for `cooperative_async` runs on the current
  publication fixture.
- Reported model file bytes, tensor data bytes, effective constrained-RAM window, async chunk
  bytes, peak resident bytes, load dispatch count, progress dispatch count, and load elapsed time.
- Kept evidence truthful: the profile explicitly reports
  `source_residency=setup_time_fixture_file_image_and_full_target_storage`, so this is constrained
  RAM emulation through the maintained public path rather than a claim of true larger-than-RAM
  source residency.
- Wired `EMEL_MODEL_LOAD_CHUNK_BYTES` and `EMEL_MODEL_LOAD_CONSTRAINED_RAM_BYTES` into the
  maintained model-loader benchmark path. The effective constrained RAM value now caps the actual
  async chunk/window used by dispatch.
- Applied the scoped optimization identified by profiling: increasing the constrained async
  window from 64 KiB to 1 MiB reduces public load dispatches from 11,165 to 780 on the same
  maintained generation fixture.
- Updated roadmap, requirements, state, project, milestone, and README evidence to mark `PERF-02`
  satisfied without claiming broader async inference or device-specific loading.

## Evidence

The maintained path is:

`tools/bench/generation_bench.cpp` -> public model-loader event -> `io/loader`
`cooperative_async` strategy -> tensor residency contracts.

64 KiB constrained window:

- `effective_ram_constraint_bytes=65536`
- `async_chunk_bytes=65536`
- `model_file_bytes=730895360`
- `tensor_data_bytes=728509440`
- `peak_resident_bytes=1499006862`
- `load_dispatches=11165`
- `progress_dispatches=11164`
- `load_elapsed_ns=26901458`
- generation compare: `439443041.000 ns/op` EMEL vs `344750959.000 ns/op` reference,
  `ratio=1.275x`

1 MiB constrained window:

- `effective_ram_constraint_bytes=1048576`
- `async_chunk_bytes=1048576`
- `model_file_bytes=730895360`
- `tensor_data_bytes=728509440`
- `peak_resident_bytes=1499006862`
- `load_dispatches=780`
- `progress_dispatches=779`
- `load_elapsed_ns=23317750`
- generation compare: `417970667.000 ns/op` EMEL vs `331386166.000 ns/op` reference,
  `ratio=1.261x`

## Remaining Bottleneck

The load-dispatch loop is no longer the dominant measured bottleneck for this benchmark after the
1 MiB window. The remaining generation-stage evidence still shows most EMEL time unattributed to
load profiling and outside Phase 252 scope.
