# Phase 149: Parity Engine Adapter Split - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning
**Mode:** Autonomous infrastructure phase

<domain>
## Phase Boundary

Move existing parity modes behind explicit runner-facing engine adapter entrypoints while
preserving the current implementations and output behavior. Phase 149 should make
`parity_runner.cpp` a small orchestration surface, not the file that owns bulk tokenizer, GBNF,
kernel, Jinja, and generation parity logic.

</domain>

<decisions>
## Implementation Decisions

### Engine Adapter Boundary
- Keep existing mode implementations behaviorally intact and move them as a bundle behind named
  adapter entrypoints.
- Introduce a shared `engine_adapter` contract that has mode, name, and run function identity.
- Keep the first adapter split source-local and conservative; Phase 150 will make CMake and engine
  registration more modular.
- Do not split generation internals into many files in this phase; the generation path is large and
  behavior-sensitive, so Phase 149 should establish the boundary before finer decomposition.

### Lane Isolation
- Preserve separate EMEL and reference lane objects inside the existing implementation bundle.
- Keep tests proving paritychecker source does not reach into actor `detail`, `action`, or `guard`
  internals directly.

### The Agent's Discretion
- Use a file move/rename if it makes `parity_runner.cpp` truthfully become the runner boundary.
- Add source regressions that fail if bulk mode implementation returns to `parity_runner.cpp`.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 148 added `tools/paritychecker/parity_assets.hpp` and `.cpp`.
- `parity_runner.cpp` still contains the bulk mode implementations plus a final `run_parity`
  switch.
- Tokenizer variant implementation is already partially split into `tokenizer_*_parity.cpp`
  files.

### Established Patterns
- Paritychecker tests use source scans to enforce ownership boundaries.
- The paritychecker CMake target explicitly lists every source file for both the tool and tests.
- Existing process-level tests are the behavior safety net; source-level tests protect the new
  boundary.

### Integration Points
- `parity_runner.hpp` is the public tool runner API used by `parity_main.cpp`.
- `paritychecker_tests.cpp` can include a new engine header and verify adapter lookup without
  invoking all heavyweight mode paths.
- `tools/paritychecker/CMakeLists.txt` must build the moved implementation file into both
  `paritychecker` and `paritychecker_tests`.

</code_context>

<specifics>
## Specific Ideas

- Add `parity_engine.hpp` / `.cpp` for the adapter contract and mode lookup.
- Move the existing bulk implementation from `parity_runner.cpp` into `parity_engines.cpp`.
- Expose `engines::run_tokenizer`, `engines::run_gbnf_parser`, `engines::run_kernel`,
  `engines::run_jinja`, and `engines::run_generation` adapter functions from the moved file.
- Recreate `parity_runner.cpp` as orchestration only: find the adapter for `opts.mode` and invoke
  it.

</specifics>

<deferred>
## Deferred Ideas

- Per-engine CMake libraries and localized registration belong to Phase 150.
- Per-runner dependency manifests belong to Phase 151.
- End-to-end closeout behavior and lane-isolation audit belongs to Phase 152.

</deferred>
