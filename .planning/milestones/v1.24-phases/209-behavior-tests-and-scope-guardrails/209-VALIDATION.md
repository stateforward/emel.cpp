---
phase: 209-behavior-tests-and-scope-guardrails
status: validated
requirements:
  - VAL-01
  - VAL-02
created: 2026-05-04T18:50:00Z
last_updated: 2026-05-04T22:15:00Z
---

# Phase 209 Validation Evidence

## Repair Summary

Replaces prior premature placeholder validation. Repair pass added a real
`visit_current_states` doctest, a slot-accounting doctest that proves
validation rejection does not consume slots, and three real script-level
guardrails inside `scripts/check_domain_boundaries.sh`. Source-string
boundary checks remain as in-test belt-and-suspenders inside
`tests/io/mmap/lifecycle_tests.cpp` but no longer carry VAL-02 alone.

### VAL-01 Doctest Evidence

`tests/io/mmap/lifecycle_tests.cpp` drives every behavior through public
`emel::io::mmap::sm::process_event(...)` dispatch and inspects state via
`is(stateforward::sml::state<...>)` and `visit_current_states(...)`. No test
reaches into `actions.hpp`, `detail.hpp`, or `guards.hpp`.

| Behavior family | Test case |
|---|---|
| Component boundary aliases | `io mmap exposes canonical machine aliases at component boundary` |
| `visit_current_states` post-RTC inspection (full map+release) | `io mmap reports state_ready via visit_current_states after a full map-then-release dispatch` |
| Validation reject does not consume slot pool | `io mmap validation rejection does not consume a slot` |
| Validation: zero byte_size | `io mmap rejects invalid request spans before any mapping attempt` |
| Validation: empty file_path | `io mmap rejects empty file_path as invalid_request` |
| Unsupported resource: file_index | `io mmap rejects out-of-range file_index as unsupported resource` |
| Unsupported resource: offset alignment | `io mmap rejects unaligned file_offset as unsupported resource` |
| Unsupported resource: byte_size cap | `io mmap rejects byte_size above maximum as unsupported resource` |
| Unsupported resource: address-space overflow | `io mmap rejects layouts that overflow the address space` |
| Mapping failure: missing file | `io mmap surfaces file_open_failed when the path does not exist` |
| Mapping failure: directory mmap | `io mmap surfaces mapping_failed when mmap call fails` |
| Success: deterministic descriptor + content | `io mmap returns a deterministic mapped descriptor on success` |
| Release happy path + LIFO slot reuse | `io mmap release happy path returns slot to the free pool` |
| Release: out-of-range handle | `io mmap release rejects out-of-range handle` |
| Release: double release | `io mmap release rejects double release on the same handle` |
| Fail-closed without callbacks | `io mmap fails closed without an error callback` |
| Success without done callback | `io mmap success records when no done callback is supplied` |
| Resource exhaustion | `io mmap surfaces resource_exhausted when slot pool is full` |
| Unexpected events deterministic | `io mmap handles unexpected events deterministically` |
| Component-internal source surface (informational) | `io mmap boundary keeps platform calls inside actions.cpp` |

Focused run (debug):

```
build/debug/emel_tests_bin --test-case='*io mmap*'
[doctest] test cases:   20 |   20 passed | 0 failed | 913 skipped
[doctest] assertions: 1202 | 1202 passed | 0 failed |
[doctest] Status: SUCCESS!
```

Focused run (zig release, io shard):

```
build/zig/emel_tests_bin --test-case='*io mmap*'
[doctest] test cases:   20 |   20 passed | 0 failed | 5 skipped
[doctest] assertions: 1202 | 1202 passed | 0 failed |
[doctest] Status: SUCCESS!
```

### VAL-02 Script Guardrail Evidence

`scripts/check_domain_boundaries.sh` now fails closed on the three
mmap-specific scope/ownership leaks defined by VAL-02:

1. `out-of-scope strategy markers leaked into io/mmap actor` — scans
   `src/emel/io/mmap` for `strategy_staged_read`, `strategy_external_buffer`,
   `strategy_async`, `strategy_device`, `strategy_copy`. The mmap component
   must remain mmap-only; staged/external-buffer routing legitimately lives
   only in `src/emel/io/loader`.
2. `deferred v2 strategy implementations leaked into src/` — scans all of
   `src` for `strategy_async`, `strategy_device`, `strategy_copy`. v1.24 is
   the mmap-strategy milestone; async/device/copy strategy implementations
   are deferred to v2 milestones.
3. `tensor residency lifecycle enumerators escaped model/tensor` — scans
   `src/emel/model/loader` and `src/emel/io` for
   `lifecycle::mmap_resident`, `lifecycle::resident`, `lifecycle::evicted`.
   This complements the existing `model::tensor::event::lifecycle::|
   lifecycle_state|event::tensor_state` rule and locks down residency
   ownership at the lifecycle-enumerator level.

Run:

```
scripts/check_domain_boundaries.sh
exit=0
```

### Changed-File Scoped Quality Gate

```
EMEL_QUALITY_GATES_CHANGED_FILES="scripts/check_domain_boundaries.sh:tests/io/mmap/lifecycle_tests.cpp:snapshots/lint/clang_format.txt" \
EMEL_QUALITY_GATES_PARALLEL=0 \
scripts/quality_gates.sh
GATE_EXIT=0
```

Lane evidence:

- `domain_boundaries`: passed (silent return 0 from `scripts/check_domain_boundaries.sh`).
- `legacy_sml_surface`: `Legacy SML surface scan passed`.
- `build_with_zig` (io shard): `ninja: no work to do.` (cache hit; configure step ran clean).
- `bench_snapshot`: `skipping bench_snapshot: no benchmark-affecting changed files`.
- `test_with_coverage`: `skipping test_with_coverage: no changed src/emel files`.
- `paritychecker`: `skipping paritychecker: no paritychecker-affecting changed files` (parity dependency manifest fresh).
- `fuzz_smoke`: `skipping fuzz_smoke: no fuzz-affecting changed files`.
- `lint_snapshot`: passed silently after `scripts/lint_snapshot.sh --update` regenerated the maintained baseline (added `tests/io/mmap/lifecycle_tests.cpp`, removed retired `src/emel/model/tensor/detail.hpp` line; the latter was already in the dirty tree from prior phases). User authorized snapshot refresh through the manager.
- `generate_docs`: `skipping generate_docs: no docsgen-affecting changed files`.

No `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION=1` override used.

## Requirement Status

- **VAL-01**: Validated. Doctests in `tests/io/mmap/lifecycle_tests.cpp` cover
  supported mmap success, validation rejection (request span, file_path,
  file_index, offset alignment, length cap, address-space overflow), unsupported
  resource categories, mapping-side failures (`file_open_failed`,
  `mapping_failed`), resource exhaustion, release happy path with LIFO slot
  reuse, release out-of-range handle, double release, fail-closed without
  callbacks, success without done callback, unexpected events, and
  `visit_current_states`-based state inspection across a full RTC chain. All
  20 cases / 1202 assertions pass under both debug and zig release builds.
- **VAL-02**: Validated. `scripts/check_domain_boundaries.sh` now fails closed
  on (a) out-of-scope strategy markers in `src/emel/io/mmap`, (b) deferred v2
  strategy implementations anywhere in `src/`, and (c) tensor residency
  lifecycle enumerators in `src/emel/model/loader` or `src/emel/io`. The
  changed-file scoped quality gate exits 0 with no overrides.

## Final Approval

VAL-01 and VAL-02 are source-backed and gate-backed. Phase 210 may proceed.
