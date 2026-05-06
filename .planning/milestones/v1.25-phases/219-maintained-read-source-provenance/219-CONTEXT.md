---
phase: 219-maintained-read-source-provenance
status: ready
created: 2026-05-05T20:50:00Z
requirements:
  - PLAT-01
  - TIO-03
  - VAL-04
depends_on:
  - 218-publication-and-maintained-artifact-updates
---

# Phase 219 Context

## Goal

Close the maintained-path provenance gap: generation benchmark, Sortformer
diarization benchmark, embedded probe, and paritychecker lanes must no longer
own tool-local full-file read scaffolds that feed `read_copy` evidence. Their
source bytes must come from a maintained `src`-owned read/source contract before
the public `model/loader -> model/tensor -> io/loader -> io/read` path reports
read/copy strategy evidence.

## Ordering Reconciliation

The autonomous workflow order was interrupted and corrected here. Source edits
for the Phase 219 implementation were started before this `CONTEXT.md` and
`PLAN.md` existed. This artifact records that mistake explicitly and folds the
already-started work into the phase boundary instead of hiding it.

Already-started Phase 219 files:

- `src/emel/io/read/detail.hpp`
- `tools/bench/generation_bench.cpp`
- `tools/bench/diarization/sortformer_fixture.hpp`
- `tools/embedded_size/emel_probe/main.cpp`
- `tools/paritychecker/parity_assets.hpp`
- `tools/paritychecker/parity_assets.cpp`
- `tools/paritychecker/parity_engines.cpp`
- `tools/paritychecker/paritychecker_tests.cpp`
- `tests/model/loader/lifecycle_tests.cpp`

## Source-Backed Truth

- `src/emel/io/read` is the maintained read/copy component. It owns the
  read-source contract used by maintained tool lanes and the dispatch-time
  read/copy actor that copies from source spans into caller-owned target
  buffers.
- Tool lanes may perform setup-time file loading for fixtures, but the helper
  must live in the maintained `src/emel/io/read` surface, not as duplicated
  `read_file_bytes` helpers under `tools/**`.
- Read/copy benchmark, parity, and probe evidence is valid only when the EMEL
  lane goes through public runtime surfaces and does not report unsupported or
  fallback behavior as read strategy support.
- Existing dirty-tree work from earlier v1.25 phases must be preserved.

## Decisions

- Superseded by Phase 222: use public `emel::io::source::load_file_bytes` as
  the maintained source-byte helper for setup-time fixture loading in the
  maintained tool lanes.
- Remove tool-local `read_file_bytes` helpers from generation, Sortformer,
  embedded probe, and paritychecker maintained EMEL lanes.
- Keep the helper outside dispatch-critical actor actions. It performs
  filesystem work during tool setup only; dispatch continues to consume
  immutable event-provided source spans.
- Add source guardrails proving maintained tool lanes do not reintroduce
  tool-local read scaffolds.
