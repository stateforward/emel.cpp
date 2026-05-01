# Phase 153: Parity Runner Config Ownership Closure - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Close `PARITY-01` by moving paritychecker CLI/config parsing, usage publication, request
normalization, and validation ownership behind the shared runner boundary while preserving existing
CLI behavior.

</domain>

<decisions>
## Implementation Decisions

### Runner Boundary
- Keep `parity_main.cpp` as a process entrypoint only; it should delegate to a runner-owned CLI
  function.
- Runner-owned parsing must preserve existing options, exit codes, and usage text unless a change is
  explicitly documented.
- Text-file loading should reuse the existing parity asset byte-loading boundary instead of adding a
  second file helper.
- Add source checks that fail if parsing/usage implementation drifts back into `parity_main.cpp`.

### Verification Shape
- Cover no-args/`--help` usage behavior through the existing process-capture helper.
- Cover invalid flag and invalid option-combination behavior as runner-owned validation behavior.
- Keep existing tokenizer, GBNF, kernel, Jinja, and generation parity smoke behavior unchanged.
- Run focused paritychecker build/tests and changed-file scoped quality gates.

### the agent's Discretion
All implementation details are at the agent's discretion inside the phase boundary, with preference
for small, local changes to `tools/paritychecker`.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/paritychecker/parity_runner.hpp` owns `parity_options` and `run_parity(...)`.
- `tools/paritychecker/parity_assets.*` already owns byte-loading and maintained fixture helpers.
- `tools/paritychecker/paritychecker_tests.cpp` has process-capture helpers and source-scan tests.

### Established Patterns
- `parity_runner.cpp` is the shared orchestration boundary and currently resolves engines through
  `find_engine(opts.mode)`.
- `parity_main.cpp` currently owns `parse_args(...)`, `print_usage(...)`, and text-file loading.
- Source checks are already used to enforce runner/engine ownership contracts.

### Integration Points
- `parity_main.cpp` should call a runner-owned CLI entrypoint.
- `paritychecker_tests.cpp` should guard both CLI behavior and source ownership.
- `tools/paritychecker/CMakeLists.txt` already compiles runner sources into both executable and
  tests.

</code_context>

<specifics>
## Specific Ideas

No extra user-facing behavior changes are requested. Preserve existing CLI behavior.

</specifics>

<deferred>
## Deferred Ideas

Live generation reference truth, actor-helper boundary tightening, and manifest gate consumption are
owned by Phases 154-156.

</deferred>
