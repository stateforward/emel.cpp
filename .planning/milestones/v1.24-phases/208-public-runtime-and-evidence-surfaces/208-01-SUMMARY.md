---
phase: 208-public-runtime-and-evidence-surfaces
plan: 01
status: implemented
requirements:
  - TIO-03
  - VAL-04
created: 2026-05-04T19:00:00Z
last_updated: 2026-05-04T20:05:00Z
---

# Phase 208 Plan 01 Summary

## Outcome

`model/loader` and the maintained tool lanes (benchmarks, paritychecker,
embedded probes) interact with mmap behavior only through public runtime
surfaces. Loader's `events::load_done.used_mmap` defaults `false` and is
never derived from tensor residency or low-level mapping flags inside
loader actions. Tools that need to report mmap usage do so through
`process_event(emel::model::tensor::event::capture_tensor_state{...})` —
the same public state inspection surface tensor exposes — without
including `actions.hpp`, `detail.hpp`, or `guards.hpp` from any actor.
Phase 208 inherited the io/mmap actor (Phase 205+206) and the tensor
mapping events (Phase 207) and wires them through the public boundary
only.

## Repair Pass

This plan also closed two prior in-progress workers' unsafe edits in the
shared worktree:

| File | Repair |
|------|--------|
| `src/emel/model/loader/actions.hpp` | Restored io callback object to `event::io_phase_events*` matching the recording helpers' `static_cast<event::io_phase_events*>(object)`. The interim regression passed `const_cast<event::load_runtime *>(&ev)`, which would have aliased the wrong type at the void-pointer boundary. Also clang-formatted the file. |
| `tests/embeddings/te_fixture_data.hpp`, `tests/speech/encoder/whisper/lifecycle_tests.cpp`, `tests/speech/decoder/whisper/lifecycle_tests.cpp` | Removed leftover writes to the deleted `emel::model::data::weights_mapped` field (data field was retired in v1.23 cutover). |
| `tests/model/loader/lifecycle_tests.cpp` | Replaced sed-empty-line residue with explicit `CHECK_FALSE(owner.used_mmap)` assertions on the three load paths (full file load, model-path load, vocab-only load) — the loader's `used_mmap` is now always `false`, and the tests now assert that contract directly. |
| `tests/model/tensor/lifecycle_tests.cpp::model_tensor_bulk_storage_supports_absent_callbacks` | Phase 207 grew `emel::model::tensor::sm` to ~2.5 MiB (added 18+ states for `request_mapped_load`/`release_mapped_load`). Six scoped sm instances in this test caused a ~15 MiB stack frame, overflowing macOS's 8 MiB default + ASan red zones, producing SIGSEGV at the test entry. Switched to `std::make_unique<emel::model::tensor::sm>()` heap allocs (one-time non-hot-path allocation, AGENTS.md compliant). |

## Public Surface Audit

- `src/emel/model/loader/`: no references to
  `model::tensor::event::tensor_state`, `capture_tensor_state`,
  `lifecycle_state`, or `lifecycle::mmap_resident`. Loader actions
  contain no runtime branching on io strategy or tensor lifecycle —
  branching lives in `guards.hpp` and the `sm.hpp` transition table
  exclusively. (`io_strategy_none`, `io_strategy_present`, and
  `tensor_plan_done_with_io_strategy_*` are guards, not action-level
  ifs.)
- `tools/bench/`, `tools/paritychecker/`, `tools/embedded_size/`: no
  includes of `actions.hpp`, `detail.hpp`, or `guards.hpp` from any
  actor. Tool-side mmap reporting calls
  `process_event(capture_tensor_state{...})` and reads the public
  `state.lifecycle_state == emel::model::tensor::event::lifecycle::mmap_resident`
  predicate — the same publicly observable lifecycle Phase 207 added.
- Loader's `effect_publish_tensor_load_done_from_file_image` and
  `effect_publish_tensor_load_done_from_model_data` no longer assign
  `used_mmap` from runtime data. The only writer of `ctx.used_mmap` is
  `begin_load`, which sets it to `false`.

## Guardrails

```
$ rg -n 'model::tensor::event::tensor_state|capture_tensor_state|lifecycle_state|lifecycle::mmap_resident' src/emel/model/loader src/emel/io
(no matches; exit 1)

$ rg -n 'emel/(io/loader|model/tensor|model/loader)/(actions|detail|guards)\.hpp|emel::io::loader::(action|detail|guard)::|emel::model::tensor::(action|detail|guard)::|emel::model::loader::(action|detail|guard)::' tools/bench tools/paritychecker tools/embedded_size
(no matches; exit 1)

$ scripts/check_domain_boundaries.sh
(exit 0)
```

## Scoped Quality Gate

```
$ EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/model/loader/actions.hpp:tests/model/loader/lifecycle_tests.cpp:tests/model/tensor/lifecycle_tests.cpp:tests/embeddings/te_fixture_data.hpp:tests/speech/decoder/whisper/lifecycle_tests.cpp:tests/speech/encoder/whisper/lifecycle_tests.cpp" \
  EMEL_QUALITY_GATES_PARALLEL=0 \
  scripts/quality_gates.sh
(exit 0)
```

Lane-by-lane:

| Lane | Result |
|------|--------|
| `domain_boundaries` | passed |
| `legacy_sml_surface` | passed |
| `build_with_zig` (model_and_batch shard) | passed |
| `test_with_sanitizers` (asan, model_and_batch shard) | passed |
| `benchmarks` (full manifest expansion: gbnf, jinja, logits, batch_planner, kernel_aarch64, memory_kv, memory_recurrent, memory_hybrid, generation, diarization_sortformer, flash_attention, all tokenizer preprocessors, encoders) | no regression |
| `coverage` (changed-file scope) | line 90.2% (229/254), function 82.9% (34/41), branch 61.5% (16/26) — above thresholds |
| `paritychecker` | 1/1 paritychecker_tests passed (9.34s) |
| `fuzz_smoke` | skipped — no fuzz-affecting changed files |
| `generate_docs` | skipped — no docsgen-affecting changed files |
| `lint_snapshot` | passed |

No `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION=1` override consumed.

## Out of Scope

- Behaviour-test sweep covering tensor-driven flows beyond Phase 207's
  focused cases (Phase 209 / VAL-01).
- Domain and source guardrail tests (Phase 209 / VAL-02).
- Public docs and snapshot publication (Phase 210 / VAL-03).
- Phase 204 transitional bench override cleanup (Phase 210).

## Notes

- **No mmap inferred in loader.** Loader's `load_done.used_mmap` is
  hard-coded to `false`. Any future "true" value must come from a
  source-backed public evidence path; today there is none. Tool-side
  mmap reporting reads tensor residency via `capture_tensor_state` only
  when the tool has bound an `io::mmap::sm*` and dispatched a
  `request_mapped_load` (none of the maintained benchmarks exercise this
  path yet, so they correctly report `used_mmap = false`).
- **Stack-frame regression flagged.** The 2.5 MiB tensor sm was
  uncovered by Phase 207's added states — the SM size ballooned from a
  much smaller value because each new state contributes to compile-time
  transition lookup tables that materialize as runtime SM members. The
  bulk-storage test was the only existing test stressing six instances
  in one frame; heap allocation is the minimum-blast-radius fix. A
  follow-up to right-size the SM is an open consideration but is
  outside Phase 208 scope.
- **No git stash, reset, or checkout consumed during repair.** Stash
  `915fc599` (left by a prior worker on this branch) was retained as
  backup per shared-worktree directive.
