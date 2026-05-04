---
phase: 208-public-runtime-and-evidence-surfaces
status: validated
requirements:
  - TIO-03
  - VAL-04
created: 2026-05-04T19:00:00Z
last_updated: 2026-05-04T22:15:00Z
---

# Phase 208 Validation Evidence

## Repair Summary
Repair pass after two prior workers left unsafe edits in the shared worktree.

### Code Fixes
- `src/emel/model/loader/actions.hpp` (loader callback object mismatch):
  IO loader callbacks restored to pass `event::io_phase_events*` (the
  `record_*` helpers static_cast their `void *object` to that type). Removed
  the `const_cast<event::load_runtime *>(&ev)` regression. Also clang-formatted
  the file to match `snapshots/lint/clang_format.txt`.
- `src/emel/model/loader/actions.hpp`: deletion of the now-removed
  `weights_mapped` mirror writes left intact; loader's `load_done.used_mmap`
  remains `false` per directive (no mmap inferred from tensor residency in
  loader).
- Tests cleaned of dangling references to deleted `emel::model::data::weights_mapped`:
  `tests/embeddings/te_fixture_data.hpp`,
  `tests/speech/encoder/whisper/lifecycle_tests.cpp`,
  `tests/speech/decoder/whisper/lifecycle_tests.cpp`.
- `tests/model/loader/lifecycle_tests.cpp`: replaced empty-line residue from
  prior sed edits with explicit `CHECK_FALSE(owner.used_mmap)` assertions on
  the three load paths that previously asserted on `used_mmap`.
- `tests/model/tensor/lifecycle_tests.cpp` (`model_tensor_bulk_storage_supports_absent_callbacks`):
  the prior worker grew `emel::model::tensor::sm` to ~2.5 MiB (added 18+ states
  for `request_mapped_load`/`release_mapped_load`). With six scoped sm
  instances in the test, the compiler reserved ~15 MiB of stack frame, which
  ASan inflated past macOS's 8 MiB default — SIGSEGV at the test's first line.
  Replaced the six scoped `tensor::sm machine{}` allocations with
  `auto machine_ptr = std::make_unique<emel::model::tensor::sm>();` heap
  allocations (one-time non-hot-path allocation, AGENTS.md compliant).

### Guardrails (all clean)
1. `rg -n 'model::tensor::event::tensor_state|capture_tensor_state|lifecycle_state|lifecycle::mmap_resident' src/emel/model/loader src/emel/io` → no matches (exit 1).
2. `rg -n 'emel/(io/loader|model/tensor|model/loader)/(actions|detail|guards)\.hpp|emel::io::loader::(action|detail|guard)::|emel::model::tensor::(action|detail|guard)::|emel::model::loader::(action|detail|guard)::' tools/bench tools/paritychecker tools/embedded_size` → no matches (exit 1).
3. `scripts/check_domain_boundaries.sh` → exit 0.

### Scoped Quality Gate
Command:
```
EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/model/loader/actions.hpp:tests/model/loader/lifecycle_tests.cpp:tests/model/tensor/lifecycle_tests.cpp:tests/embeddings/te_fixture_data.hpp:tests/speech/decoder/whisper/lifecycle_tests.cpp:tests/speech/encoder/whisper/lifecycle_tests.cpp" \
EMEL_QUALITY_GATES_PARALLEL=0 \
scripts/quality_gates.sh
```
Result: **exit 0**.

Lane evidence:
- `domain_boundaries`: passed.
- `legacy_sml_surface`: passed.
- `build_with_zig` (model_and_batch shard): passed.
- `test_with_sanitizers` (asan, model_and_batch shard): passed.
- `benchmarks`: full manifest expansion ran; no regression. (Earlier transient
  `tokenizer/full_plamo2_long` blip on a prior run did not reproduce on rerun.)
- `coverage` (changed-file scope, model_and_batch shard): line 90.2%
  (229/254), function 82.9% (34/41), branch 61.5% (16/26). Threshold ≥ 90%
  line / ≥ 50% branch satisfied.
- `paritychecker`: 1/1 paritychecker_tests passed (9.34s).
- `fuzz_smoke`: skipped — no fuzz-affecting changed files.
- `generate_docs`: skipped — no docsgen-affecting changed files.
- `lint_snapshot`: passed after clang-format on the touched header.

No `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION=1` override used.

## Requirement Status
- **TIO-03**: model/loader, benchmark lanes, paritychecker lanes, and embedded
  probes use only public runtime surfaces. Loader actions never mention
  `tensor_state`, `capture_tensor_state`, `lifecycle_state`, or
  `lifecycle::mmap_resident`. Tools that need to inspect tensor state do so
  via `process_event(capture_tensor_state{...})` only — no `actions.hpp`,
  `detail.hpp`, or `guards.hpp` includes from `tools/`.
- **VAL-04**: Loader's `load_done.used_mmap` defaults `false` and is never
  promoted to `true` in any loader action; tools report mmap usage via the
  public `capture_tensor_state` event. There is no fake/derived mmap claim.
