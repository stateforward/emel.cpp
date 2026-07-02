---
phase: 227-staged-read-strategy-component-boundary
plan: "01"
subsystem: io-staged-read
tags: [stateforward-sml, io-staged-read, stg-01, scaffolding]
requires:
  - phase: 227-staged-read-strategy-component-boundary
    provides: "227-CONTEXT.md, 227-01-PLAN.md with rule traceability."
provides:
  - Canonical `src/emel/io/staged_read` header bundle (seven files mirroring sibling I/O layout).
  - Minimal model with `*` initial source on idle `state_ready` plus unexpected-event no-op boundary.
  - Top-level `emel::IoStagedRead` alias in `machines.hpp`.
  - CMake registration and doctest smoke for default `state_ready` after construction.
affects: [phase-227, milestone-v1.26-issue-63]
tech-stack:
  added: []
  patterns:
    - STG-01 boundary only; no synthetic INTERNAL runtime carriers until STG-02+.
requirements-completed: [STG-01]
duration: unknown
completed: 2026-05-07
---

# Phase 227 Plan 01: staged_read component boundary Summary

Shipped STG-01 with a compilable staged-read strategy scaffold: packaged headers, trivial SML idle
graph, aggregates alias wiring, and an `io` shard doctest proving default `state_ready`.

## Performance

- **Tasks:** 2 (per `227-01-PLAN.md`)
- **Files touched:** 7 new headers plus `CMakeLists.txt`, `machines.hpp`, new lifecycle test runner.

## Accomplishments

- Added empty `context`, placeholder `detail`/`event` namespaces, trivial `effect_on_unexpected`, and one
  transition row anchored on initial `*` within `state_ready` for deterministic unexpected handling.
- Registered `tests/io/staged_read/lifecycle_tests.cpp` beside read path sources in `CMakeLists.txt`.
- Accepted plan constraints met (no fabricated `*_runtime` tokens under `src/emel/io/staged_read`; no mmap
  / device / syscall / coroutine code in scaffold).

## Build and test evidence (closeout session)

Captured 2026-05-07 in worktree `/Users/gabrielwillen/.atmux/teams/emel_cpp/milestone63/worktree`:

```text
$ ninja -C build emel_tests_bin
ninja: no work to do.

$ ctest --test-dir build --output-on-failure -R emel_tests_io
100% tests passed, 0 tests failed out of 1
Total Test time (real) = 0.45 sec (emel_tests_io shard)
```

## Changed-file quality gate (required; 2026-05-07)

Command (colon-separated `EMEL_QUALITY_GATES_CHANGED_FILES` scoped to Phase 227 implementation surface):

```bash
export EMEL_QUALITY_GATES_CHANGED_FILES='CMakeLists.txt:src/emel/machines.hpp:src/emel/io/staged_read/actions.hpp:src/emel/io/staged_read/context.hpp:src/emel/io/staged_read/detail.hpp:src/emel/io/staged_read/errors.hpp:src/emel/io/staged_read/events.hpp:src/emel/io/staged_read/guards.hpp:src/emel/io/staged_read/sm.hpp:tests/io/staged_read/lifecycle_tests.cpp'
scripts/quality_gates.sh
```

Full transcript: `/tmp/emel_qg227_final.log` (5618 lines at capture time).

Outcome: **did not pass**. `quality_gates: log end name=bench_snapshot status=0 duration=847s` succeeded, then
`test_with_coverage` reported `failed minimum line coverage (got 0.0%, minimum 90.0%)` with an empty TOTAL
instrumentation table (0 lines instrumented under the scoped `src/emel/...hpp` set). Runner line:
`quality_gates: log end name=test_with_coverage status=2 duration=1s`. Subsequent `paritychecker` and `fuzz_smoke`
steps logged `status=0` (skipped or no-op for this changed-file surface).

Transient side effect: `scripts/quality_gates.sh` rewrote `snapshots/quality_gates/timing.txt` during the run.
Per main guidance (not consent to commit timing), immediately ran:

```bash
git restore --source=HEAD --worktree --staged snapshots/quality_gates/timing.txt
```

and verified `git diff -- snapshots/quality_gates/timing.txt` is empty.

## Supplemental full-repo coverage (maintained lane; Du2aBZQSjMF7Pt_O / Mk-SA8M2iQLyM-Gp; 2026-05-07)

Per main direction (**no dummy `.cpp`**, **no weakened thresholds**, changed-only scoped lane **still not** a semantic
pass for header-only/zero-instrumented-line sets):

```bash
EMEL_COVERAGE_CHANGED_ONLY=0 scripts/test_with_coverage.sh
```

**First attempt** (`/tmp/emel_phase227_supplemental_coverage.log`): `ctest` failed before `gcovr` —
`encoder_detail_trie_basic` tripped UB in `detail::naive_trie::insert`: a `node &` into `nodes` was used across
`nodes.emplace_back()`, so `emplace_back()` could **reinvalidate** that reference (**manager root-cause**).

**Narrow fix (non–Phase 227):** `src/emel/text/encoders/types.hpp` — rewrote `insert` to index into
`nodes` only (no references across `emplace_back()`); **did not** touch `src/emel/io/staged_read/**`.

Focused repro / regression on the **`build/coverage`** tree after rebuild:

```text
$ ctest --test-dir build/coverage --output-on-failure -R '^(emel_tests_text|emel_tests_text_encoders)$'
100% tests passed, 0 tests failed out of 2
Total Test time (real) ~121 sec
```

**Green supplemental run:**

- Transcript **`/tmp/emel_phase227_supplemental_coverage_v2.log`**
- **Exit status `0`** (recorded **`/tmp/emel_phase227_supplemental_coverage_v2.exit`**)
- `ctest`: **`100% tests passed, 0 tests failed out of 13`**
- `gcovr`: **`enforcing coverage thresholds: line >= 90%, branch >= 50%`** then summary **TOTAL** lines **92.0%**,
  branches **57.1%** (defaults satisfied). Example touched file line: `src/emel/text/encoders/types.hpp`
  **`100%`** line coverage.

`snapshots/quality_gates/timing.txt`: **still clean** (`git diff` empty; this lane does not require committing timing).

Navigator **final Phase 227 approval before Phase 228** requested via `atmux send` to **milestone63-navigator**
(reply-required, message id **`XpdMe5gHRffYT_Oe`**).

## Task Commits

No git commits bundled with this atmux-driver session (`git status` shows local edits only). Landing
these changes should squash or split per milestone PR conventions before merging.

## Files Created / Modified

- Created `src/emel/io/staged_read/{actions,context,detail,errors,events,guards,sm}.hpp`.
- Modified `src/emel/machines.hpp`, `CMakeLists.txt`, `src/emel/text/encoders/types.hpp` (`naive_trie::insert` UB fix
  for coverage validation; see **Supplemental full-repo coverage**).
- Added `tests/io/staged_read/lifecycle_tests.cpp`.

## Next Phase Readiness

STG-01 scaffold is sound; **changed-file scoped quality gate** still records **`test_with_coverage` `status=2`**
for **0 instrumented lines** on the header-only change set (expected semantics; **not** treated as pass).
**Supplemental** `EMEL_COVERAGE_CHANGED_ONLY=0 scripts/test_with_coverage.sh` **passed** after the `types.hpp`
fix (see above). **Phase 228 is not started** until **milestone63-navigator** returns final Phase 227 approval
(main policy).

