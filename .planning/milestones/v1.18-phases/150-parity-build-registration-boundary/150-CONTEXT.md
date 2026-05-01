# Phase 150: Parity Build Registration Boundary - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning
**Mode:** Autonomous infrastructure phase

<domain>
## Phase Boundary

Make the Phase 149 adapter split auditable from build and registration structure. This phase should
not change parity behavior; it should reduce duplicated CMake source lists, name runner versus
engine source groups explicitly, and add source tests that prove future engine additions have a
localized registration/build surface.

</domain>

<decisions>
## Implementation Decisions

### Build Boundary
- Keep the existing executable and test targets; do not introduce separate installed libraries in
  this milestone.
- Factor CMake source lists into explicit runner, engine, tokenizer-engine, and reference-support
  groups shared by both `paritychecker` and `paritychecker_tests`.
- Treat `parity_engine.cpp` / `parity_engines.hpp` as the registration surface for now; Phase 151
  can consume this structure for dependency manifests.

### Registration Boundary
- Keep invalid modes fail-closed through `find_engine(...) == nullptr`.
- Add tests proving all current engine names are registered and that no default fallback silently
  maps unknown modes to tokenizer.

### The Agent's Discretion
- Use source-level tests for CMake shape; they are appropriate for this tool boundary and avoid
  adding runtime-only scaffolding.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 149 added `parity_engine.hpp`, `parity_engine.cpp`, `parity_engines.hpp`, and
  `parity_engines.cpp`.
- `tools/paritychecker/CMakeLists.txt` still lists every common source twice across executable and
  test targets.
- `paritychecker_tests.cpp` already reads source files for boundary regressions.

### Established Patterns
- Tool tests can assert source organization through file reads.
- CMake hard-fails missing required dependencies such as doctest.
- The repo prefers conservative, explicit boundaries over hidden fallback behavior.

### Integration Points
- `PARITYCHECKER_BINARY_PATH` still needs to point tests at the built `paritychecker` executable.
- Both `paritychecker` and `paritychecker_tests` must link the same common runner/engine sources.

</code_context>

<specifics>
## Specific Ideas

- Add CMake variables such as `PARITYCHECKER_RUNNER_SOURCES`,
  `PARITYCHECKER_ENGINE_REGISTRATION_SOURCES`, `PARITYCHECKER_ENGINE_IMPLEMENTATION_SOURCES`,
  `PARITYCHECKER_TOKENIZER_ENGINE_SOURCES`, and `PARITYCHECKER_REFERENCE_SUPPORT_SOURCES`.
- Build `PARITYCHECKER_COMMON_SOURCES` once from those groups and use it in both targets.
- Add source tests that fail if the shared source groups disappear or if the executable and test
  targets return to duplicate direct source lists.

</specifics>

<deferred>
## Deferred Ideas

- Dependency manifest emission belongs to Phase 151.
- Full closeout behavior and lane-isolation proof belongs to Phase 152.

</deferred>
