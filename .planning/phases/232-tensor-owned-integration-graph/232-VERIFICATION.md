# 232 Verification

## Commands run

### 1) Build

`ninja -C build emel_tests_bin`

Exit: **0**

### 2) Focused tests

`ctest --test-dir build --output-on-failure -R 'emel_tests_(io|model)'`

Exit: **0**

### 3) Scoped quality gate

Command:

`EMEL_QUALITY_GATES_CHANGED_FILES="src/emel/model/tensor/events.hpp src/emel/model/tensor/context.hpp src/emel/model/tensor/errors.hpp src/emel/model/tensor/detail.hpp src/emel/model/tensor/guards.hpp src/emel/model/tensor/actions.hpp src/emel/model/tensor/sm.hpp tests/model/tensor/lifecycle_tests.cpp" scripts/quality_gates.sh`

Exit: **2** (red — gate did not pass)

Observed failures (both unrelated to Phase 232 staged-tensor integration files):

- **bench_snapshot**: expanded benchmark regression suite reported regressions outside staged-read or tensor-integration changed files.
- **paritychecker**: existing test `paritychecker matches llama tokens across tiny models` → model `tests/models/flan-t5-small.Q2_K.gguf` → subcase `long` failed. Failure is pre-existing and outside Phase 232 scope.

Assessment:

- Both failures are outside staged tensor integration files and the new staged-load tests.
- Quality gate is **not** claimed as passing for this phase.

### 4) Snapshot restoration

`snapshots/quality_gates/timing.txt` restored to `HEAD` after each gate run to avoid snapshot churn.

## Requirement status (phase-local)

- Phase 232 integration uses `io::staged_read::event::staged_window` source-span
  contract (not OS file-backed staged I/O widening).
- TNX-01: Implemented in source (public staged_read event dispatch only).
- TNX-02: Implemented in source/tests (tensor lifecycle remains owner).
- TNX-03: Implemented in source/tests (explicit staged done event path).
- TNX-04: Implemented in source/tests (explicit staged error event path).

## Residual risk / blockers

- Scoped quality gate exits 2: bench_snapshot regressions (unrelated expanded
  suite) and paritychecker flan-t5-small long failure (pre-existing, outside
  Phase 232 scope) remain unresolved. Gate passage is not claimed for this
  phase.
