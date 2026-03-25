---
phase: 12-parity-and-verification-closure
plan: 01
subsystem: paritychecker-reference-proof
tags: [paritychecker, llama.cpp, flash-attention, cmake, proof-surface]
requires:
  - phase: 11-generator-flash-adoption
    provides: generator-owned flash dispatch counters and canonical flash execution truth
provides:
  - paritychecker reference selection from CMake-fetched upstream `llama.cpp`
  - normal generation output that publishes fetched reference identity
  - automatic failure diagnostics for flash and parity regressions without `--dump`
affects: [13-benchmark-evidence]
tech-stack:
  added: []
  patterns: [fetched-reference identity publication, always-on normal proof surface]
key-files:
  created: []
  modified:
    [tools/paritychecker/CMakeLists.txt, tools/paritychecker/parity_runner.cpp]
key-decisions:
  - "Force `REF_IMPL_REF` to `master` in the CMake cache so existing build directories cannot silently keep an older pinned reference."
  - "Publish fetched reference identity and flash proof on the normal success surface instead of hiding them behind `--dump`."
patterns-established:
  - "Parity proof surfaces must state which fetched upstream reference they validated against."
  - "Generation failure paths dump the same proof seam automatically so one run is enough to diagnose a regression."
requirements-completed: [PAR-01, PAR-02]
duration: 0min
completed: 2026-03-21
---

# Phase 12 Plan 1 Summary

**Paritychecker now fetches the upstream reference through CMake and publishes flash-proof metadata on the normal generation surface**

## Accomplishments

- Removed the repo-local `tmp/llama.cpp` preference from
  [CMakeLists.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/CMakeLists.txt)
  so paritychecker always resolves its reference implementation through the CMake fetch path and
  records the fetched commit hash for runtime publication.
- Added reference-source and reference-ref compile definitions in
  [CMakeLists.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/CMakeLists.txt)
  so the runtime proof surface can tell users exactly which upstream revision the parity run
  validated against.
- Promoted the generation proof seam in
  [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp)
  onto the normal `--generation` success path by printing `reference_impl`, decode seam counters,
  kernel-dispatch counters, and flash-dispatch counters without requiring `--dump`.
- Added automatic failure-surface dumping in
  [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp)
  so generation init failures, missing flash proof, reference failures, and output mismatches emit
  enough diagnostics in a single run.

## Task Commits

None - execution is staying local and `commit_docs` is `false`.

## Verification

- `ctest --test-dir build/paritychecker_zig_latest --output-on-failure -R paritychecker_tests`
- `./build/paritychecker_zig_latest/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`
- `./build/paritychecker_zig_latest/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 8`
- `rg 'tmp/llama.cpp|reference_impl: source=|flash_dispatch:' tools/paritychecker/CMakeLists.txt tools/paritychecker/parity_runner.cpp`

## Deviations from Plan

- The first default-build verification still reused a stale cached `REF_IMPL_REF` from
  `build/paritychecker_zig`. The fix stayed within scope:
  [CMakeLists.txt](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/CMakeLists.txt)
  now forces the cache entry to `master`, so existing build directories cannot silently validate
  against an older reference.

## Next Readiness

- Phase 12 Plan 2 can now lock this proof surface in subprocess tests and run the repo-level
  regression gates.
- Phase 13 should explicitly revisit benchmark-tool reference selection, because
  `tools/bench` still uses the local `tmp/llama.cpp` checkout even though paritychecker no longer
  does.

---
*Phase: 12-parity-and-verification-closure*
*Completed: 2026-03-21*
