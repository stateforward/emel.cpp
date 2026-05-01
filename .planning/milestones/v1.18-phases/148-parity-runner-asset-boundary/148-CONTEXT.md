# Phase 148: Parity Runner Asset Boundary - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning
**Mode:** Autonomous infrastructure phase

<domain>
## Phase Boundary

Establish the first shared `tools/paritychecker` runner boundary by centralizing common asset,
fixture, and file-loading helpers that are currently embedded in the monolithic runner. This phase
does not split parity engines yet; existing tokenizer, GBNF, kernel, Jinja, and generation behavior
must remain delegated to the current implementation paths until Phase 149.

</domain>

<decisions>
## Implementation Decisions

### Runner Boundary
- Put common file existence, byte loading, repo-root path, normalized path, and maintained
  generation fixture resolution behind a paritychecker-owned helper boundary.
- Keep CLI parsing in `parity_main.cpp` for this phase; Phase 148 is the runner asset boundary,
  not a user-facing CLI behavior change.
- Keep generation fixture validation exact: the maintained fixture must still resolve to the
  checked-in path, and same-basename impostor files outside `tests/models` must still fail.
- Preserve separate EMEL and reference lane ownership; common asset helpers may pass paths and
  bytes, but must not share model, tokenizer, runtime, cache, or output objects across lanes.

### The Agent's Discretion
- Choose the smallest helper API that removes duplicated runner-local asset logic and can support
  later engine adapters.
- Add focused source/unit regressions around the new helper boundary before broader engine
  decomposition.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/paritychecker/parity_runner.cpp` currently owns `file_exists`, `read_file_bytes`,
  generation fixture path normalization, maintained fixture lookup, and repo-root snapshot paths in
  one anonymous namespace.
- `tools/paritychecker/paritychecker_tests.cpp` already has source-level and process-level
  regressions for maintained generation fixture behavior, missing models, and actor-internal
  boundary checks.
- `tools/generation_fixture_registry.hpp` provides the maintained generation fixture list and
  fixture-relative paths used by the paritychecker.

### Established Patterns
- Paritychecker tests build as a separate CMake executable linked against the same runner source as
  the tool.
- Source-boundary regressions use doctest and plain file reads instead of adding test-only knobs to
  production code.
- Tool code may use heap allocation and filesystem helpers outside runtime dispatch; the EMEL lane
  still has to drive public runtime actors and cannot depend on reference-created objects.

### Integration Points
- `tools/paritychecker/CMakeLists.txt` owns the tool/test source list and must include any new
  helper implementation in both `paritychecker` and `paritychecker_tests`.
- `run_generation_parity(...)` is the maintained path that validates model existence, maintained
  fixture identity, file bytes, EMEL model loading, reference model loading, EMEL generation, and
  baseline comparison.
- Tokenizer parity still loads reference and EMEL vocabulary separately inside the current runner;
  this phase should not move that behavior behind an engine boundary yet.

</code_context>

<specifics>
## Specific Ideas

- Add `tools/paritychecker/parity_assets.hpp` and `.cpp` under
  `emel::paritychecker::assets`.
- Replace runner-local helpers with calls to the asset boundary.
- Add paritychecker tests proving the helper normalizes maintained fixture paths, rejects impostor
  paths, reads bytes through the shared helper, and leaves the existing process behavior intact.

</specifics>

<deferred>
## Deferred Ideas

- Engine adapter interfaces and mode registration belong to Phase 149 and Phase 150.
- Dependency manifest emission belongs to Phase 151.
- Full behavior and lane-isolation closeout proof belongs to Phase 152.

</deferred>
