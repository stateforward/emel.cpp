---
phase: 209-behavior-tests-and-scope-guardrails
plan: 01
status: implemented
requirements:
  - VAL-01
  - VAL-02
created: 2026-05-04T18:50:00Z
last_updated: 2026-05-04T22:15:00Z
---

# Phase 209 Summary: Behavior Tests and Scope Guardrails

## Summary of Work

Phase 209 closes VAL-01 and VAL-02. Repair pass replaced earlier premature
artifacts that claimed validation without source-backed evidence.

### Tests (`tests/io/mmap/lifecycle_tests.cpp`)

Added two doctests on top of the existing 18:

- `io mmap reports state_ready via visit_current_states after a full
  map-then-release dispatch` — uses `sm::visit_current_states` (the SML
  rule's preferred state-inspection helper, complementing the existing
  `is(state)` checks) to confirm the actor returns to `state_ready` with no
  residual decision-state regions after a complete map+release RTC chain.
- `io mmap validation rejection does not consume a slot` — drives four
  representative validation rejections (zero byte_size, empty file_path,
  out-of-range file_index, unaligned offset) and then proves the slot pool is
  still untouched by mapping `k_max_mappings` files successfully and
  releasing them. Locks down "fail closed without resource leak" behavior on
  the rejection paths.

All 20 test cases drive the actor through public `process_event(...)` only.
Result: 20 cases / 1202 assertions pass under debug and zig release builds.

### Guardrails (`scripts/check_domain_boundaries.sh`)

Added three real script-level checks for VAL-02 (source-string assertions
inside the doctest are retained as informative belt-and-suspenders but no
longer carry VAL-02 alone):

1. `out-of-scope strategy markers leaked into io/mmap actor` — scans
   `src/emel/io/mmap` for `strategy_{staged_read,external_buffer,async,device,copy}`.
2. `deferred v2 strategy implementations leaked into src/` — scans `src` for
   `strategy_{async,device,copy}`. Staged/external-buffer routing
   legitimately exists only inside `src/emel/io/loader`, so it is excluded.
3. `tensor residency lifecycle enumerators escaped model/tensor` — scans
   `src/emel/model/loader` and `src/emel/io` for
   `lifecycle::{mmap_resident,resident,evicted}`.

`scripts/check_domain_boundaries.sh` exits 0 against the current tree.

### Snapshot

`scripts/lint_snapshot.sh --update` regenerated
`snapshots/lint/clang_format.txt` to include the existing
`tests/io/mmap/lifecycle_tests.cpp` (added in earlier phases but never
baselined) and to drop the retired `src/emel/model/tensor/detail.hpp`.
Manager-authorized via the milestone-wide user approval for snapshot/model/
benchmark regeneration. No clang-format style rules changed; only file
enumeration.

### Quality Gate

```
EMEL_QUALITY_GATES_CHANGED_FILES="scripts/check_domain_boundaries.sh:tests/io/mmap/lifecycle_tests.cpp:snapshots/lint/clang_format.txt" \
EMEL_QUALITY_GATES_PARALLEL=0 \
scripts/quality_gates.sh
GATE_EXIT=0
```

No `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION=1` override used. Bench,
coverage, parity, and fuzz lanes were skipped because no
benchmark/src/parity/fuzz files changed. Domain boundaries, legacy SML
surface, build, and lint snapshot lanes ran and passed. See
`209-VALIDATION.md` for the full lane summary.

## Next Steps

Phase 210 — Publication and Maintained Artifact Updates — must close VAL-03
(public docs, generated architecture docs, lint snapshots, benchmark
snapshots, benchmark outputs, and model artifacts truth) and remove the
transitional Phase 204 `EMEL_QUALITY_GATES_ALLOW_BENCH_REGRESSION=1`
override before milestone closeout.
