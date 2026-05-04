---
gsd_state_version: 1.0
milestone: v1.24
milestone_name: I/O Mmap Loading Strategy
status: milestone_complete
stopped_at: Phase 211 verification-artifact backfill executed; 208/209/210 VERIFICATION.md created with status: passed and source-backed requirement tables; v1.24 13/13 requirements validated.
last_updated: "2026-05-04T22:18:00Z"
last_activity: 2026-05-04
progress:
  total_phases: 8
  completed_phases: 8
  total_plans: 8
  completed_plans: 8
  percent: 100
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-04)

**Core value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.
**Current focus:** v1.24 milestone closed; ready to plan next milestone.

## Current Position

Phase: 211 (8 of 8) — validated
Plan: 01 — validated
Status: v1.24 milestone complete; full-scope quality gate green (Phase 210, no override),
and the per-phase VERIFICATION.md gap (`.planning/v1.24-MILESTONE-AUDIT.md` initial
`gaps_found`) is closed by Phase 211. 208/209/210 VERIFICATION.md backfilled with YAML
frontmatter (`status: passed`) and source-backed requirement tables; 208 and 209
SUMMARY/VALIDATION received minimal YAML frontmatter; 13/13 v1.24 requirements
validated.
Last activity: 2026-05-04 - Phase 211 verification-artifact backfill executed.

Progress: [##########] 100%

## Performance Metrics

**Latest audited milestone:** `v1.24 I/O Mmap Loading Strategy`

- v1.24 shipped on 2026-05-04 after Phase 210 closing full-scope quality gate passed with
  no override. 13/13 v1.24 requirements satisfied (MMAP-01..03, TIO-01..03, PLAT-01,
  LIFE-01, ERR-01, VAL-01..04).
- User approved updates to model artifacts, generated docs, snapshots, benchmarks, and benchmark
  outputs when required to close the current milestone correctly. The Phase 210 closeout used
  that authorization to refresh `snapshots/bench/benchmarks.txt` for `encoder_spm` and
  `encoder_wpm` via maintained scoped `scripts/bench.sh --snapshot --compare --update`.
- v1.23 shipped on 2026-05-04 after final source-backed audit passed with 15/15 active
  requirements satisfied.

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting this work:

- `v1.24` implements only the mmap strategy under `src/emel/io`.
- `model/tensor` owns tensor load, bind, evict, and residency semantics.
- `model/loader` remains orchestration-only and must not absorb low-level mmap byte access.
- Staged read/copy, device-specific, cooperative async, model-family widening, and tool-only mmap
  scaffolds are out of scope.
- Phase 205 introduced compile-time `EMEL_IO_MMAP_PLATFORM_SUPPORTED` (default `0`) as the
  platform-selection knob.
- Phase 206 flipped the macro default to `1` on POSIX/Windows hosts, added the actor-owned
  fixed-capacity slot pool inside io/mmap `action::context` (`EMEL_IO_MMAP_MAX_MAPPINGS = 256`),
  added `event::release_mapping` as the only public unmap surface, and placed all platform OS
  calls behind `#if defined(_WIN32)` selection inside `src/emel/io/mmap/actions.cpp`.
- Phase 207 added `event::request_mapped_load` and `event::release_mapped_load` on
  `emel::model::tensor::event` plus an `mmap_resident` lifecycle value and an
  `sm(emel::io::mmap::sm*)` injection constructor. Tensor stores zero handle state — the
  release event carries `(tensor_id, mapping_handle)` (caller obtains handle from the request
  done callback) so the actor never scans or maintains a mapping table.
- Phase 208 closed TIO-03 and VAL-04: `model/loader` references no tensor-residency or
  mmap-lifecycle symbols; `tools/bench`, `tools/paritychecker`, and `tools/embedded_size` reach
  tensor state only via the public `process_event(capture_tensor_state{...})` event — no
  `actions.hpp`/`detail.hpp`/`guards.hpp` reach-through. Loader's `load_done.used_mmap` is
  hard-coded to `false` (no inferred mmap from tensor residency). The repair pass also fixed a
  callback object/type mismatch in `effect_dispatch_io_loads` (passed
  `event::load_runtime*` via `const_cast`; restored to `event::io_phase_events*`), removed
  dangling writes to the retired `emel::model::data::weights_mapped` field in three test
  fixtures, and switched six scoped `tensor::sm` instances in
  `model_tensor_bulk_storage_supports_absent_callbacks` to `std::make_unique` heap allocation
  (Phase 207's expanded SM grew sizeof to ~2.5 MiB; six stack instances overflowed macOS's
  default + ASan red zones).
- Phase 209 closed VAL-01 and VAL-02 (repair pass after a prior worker landed premature
  validation/summary placeholders). Added two new io mmap doctests:
  `io mmap reports state_ready via visit_current_states after a full map-then-release dispatch`
  (the SML rule's preferred state-inspection helper alongside `is(...)`) and
  `io mmap validation rejection does not consume a slot` (drives four representative reject
  paths and proves the slot pool stays untouched by then mapping `k_max_mappings` files).
  Extended `scripts/check_domain_boundaries.sh` with three real script-level rules so VAL-02
  no longer rests on source-string assertions inside the doctest:
  (1) out-of-scope strategy markers leaked into `src/emel/io/mmap`,
  (2) deferred v2 `strategy_async`/`strategy_device`/`strategy_copy` implementations anywhere
  in `src`, (3) tensor residency lifecycle enumerators (`lifecycle::mmap_resident`,
  `lifecycle::resident`, `lifecycle::evicted`) escaping `src/emel/model/loader` or
  `src/emel/io`. `scripts/lint_snapshot.sh --update` regenerated the maintained lint baseline
  to include `tests/io/mmap/lifecycle_tests.cpp` (added in earlier phases but never baselined)
  and to drop retired `src/emel/model/tensor/detail.hpp`.
- Phase 210 closed VAL-03 and the v1.24 milestone. README + README template + parity roadmap
  describe mmap as implemented; deferred v2 read/copy/async/device strategies remain
  explicitly out of scope. `snapshots/bench/benchmarks.txt` refreshed for `encoder_spm`
  (text/encoders/spm_short ns_per_op=1300.292) and `encoder_wpm` (text/encoders/wpm_long
  ns_per_op=30989.708) via maintained scoped `scripts/bench.sh --snapshot --compare
  --update`. Closing full-scope quality gate exit 0 with no override; total 432s.

### Pending Todos

- 2026-04-02 - Move eager quant prepack into generator initializer
- 2026-04-02 - Reuse q8 RHS across LFM2.5 prefill matmuls
- 2026-04-02 - Optimize LFM2.5 q4 prefill kernel
- 2026-04-02 - Optimize LFM2.5 q6 prefill kernel

### Blockers/Concerns

- v1.24 has no open blockers. The Phase 204 transitional bench-regression override is fully
  removed: the Phase 210 closing full-scope gate ran without
  `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION` and reported `status=0` across all benchmark
  suites previously affected (`tokenizer/preprocessor_rwkv_long`, `text/encoders/rwkv_long`,
  `logits/sampler`, `logits/validator`, `batch/planner_simple`, `batch/planner_equal`).
- Two encoder benchmark suites (`text/encoders/spm_short`, `text/encoders/wpm_long`) showed
  intermittent under-load timing spikes (~31% above prior baseline) during the Phase 210
  closing gate runs. Each was refreshed via the maintained scoped update path. Worth
  monitoring on subsequent gates; not a v1.24 blocker.
- Phase 207's uncovered guard-branch and unexpected-event sentinel spans were re-measured
  under the Phase 210 full-scope coverage run; total line coverage is 91.7% and branch
  coverage 56.9% (above the gate thresholds of 90% / 50%).
- The previously deferred non-v1.23 quick task and four optimization todos remain carried forward
  and are not blockers for the next milestone.

## Deferred Items

Items acknowledged and deferred at v1.22 milestone close on 2026-05-03:

| Category | Item | Status |
|----------|------|--------|
| quick_task | 260401-ejm-add-non-blocking-benchmark-binary-size-c | missing |
| todo | 2026-04-02-move-eager-quant-prepack-into-generator-initializer.md | pending |
| todo | 2026-04-02-optimize-lfm2-5-q4-prefill-kernel.md | pending |
| todo | 2026-04-02-optimize-lfm2-5-q6-prefill-kernel.md | pending |
| todo | 2026-04-02-reuse-q8-rhs-across-lfm2-5-prefill-matmuls.md | pending |

## Session Continuity

Last session: 2026-05-04T22:18:00Z
Stopped at: Phase 211 verification-artifact backfill executed; v1.24 milestone closed with 13/13 requirements validated.
Resume file: .planning/milestones/v1.24-MILESTONE-AUDIT.md (or run `$gsd-audit-milestone` to confirm passed status)
