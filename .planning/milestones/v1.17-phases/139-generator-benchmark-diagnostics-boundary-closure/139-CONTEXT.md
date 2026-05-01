# Phase 139: Generator Benchmark Diagnostics Boundary Closure - Context

**Gathered:** 2026-04-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Close the v1.17 source-backed audit blocker for `TEXTGEN-07`: generation benchmark publication
proof must stop including text-generator actor internals and stop reading generator context through
`emel::text::generator::sm` member diagnostics. The phase also fixes Phase 138 summary frontmatter
so requirement extraction sees `TEXTGEN-07`.

</domain>

<decisions>
## Implementation Decisions

### Public Diagnostics Contract
- Replace direct `sm` diagnostic getters with a public generator event that captures bounded
  runtime metrics through normal actor dispatch.
- Keep the event read-only and synchronous; it must not allocate, block, or mutate actor state.
- Capture only the existing diagnostic counters needed by benchmark, paritychecker, and tests.
- Do not introduce a new model family, fixture, sampling policy, or performance claim.

### Boundary Enforcement
- Remove `emel/text/generator/detail.hpp`, text-generator guard/action internals, and prefill guard
  internals from `tools/bench/generation_bench.cpp`.
- Remove or replace the context-reading `sm` diagnostic member functions so production callers
  cannot bypass the public event contract.
- Update tests and tools to use the public event contract rather than actor internals.
- Preserve existing generation behavior and output semantics.

### Verification Scope
- Add source regression checks for the benchmark path, not only the stage-probe substring.
- Rerun focused generator/runtime tests, paritychecker tests, benchmark tests, domain-boundary
  checks, and the scoped generation quality gate when feasible.
- Rerun milestone audit after the phase completes.
- Fix Phase 138 `requirements:` frontmatter to `requirements-completed:`.

### the agent's Discretion
Implementation details are at the agent's discretion as long as they comply with the SML actor
rules, avoid context access from `sm` member functions, and keep diagnostics source-backed through
a public contract.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/text/generator/events.hpp` owns public generator request and outcome event contracts.
- `src/emel/text/generator/actions.hpp` owns bounded actions that can read actor context.
- `src/emel/text/generator/sm.hpp` owns explicit SML transition rows and public wrappers.
- `tools/bench/bench_runner_tests.cpp` already contains a benchmark source-boundary regression.
- `tools/paritychecker/paritychecker_tests.cpp` already scans paritychecker sources for hidden
  generator actor internal bridges.

### Established Patterns
- Public generator behavior is driven by `process_event(...)` with immutable request events.
- Cross-phase runtime choices belong in explicit SML transitions and guards, not helper-side
  branching.
- Benchmark/parity EMEL lanes must use EMEL-owned code and public event surfaces for maintained
  proof.

### Integration Points
- `tools/bench/generation_bench.cpp` currently reads `generation_*` diagnostic methods around
  `run_emel_generate(...)`.
- `tools/paritychecker/parity_runner.cpp` uses the same `sm` diagnostics for generation baseline
  attribution and kernel metadata.
- `tests/text/generator/lifecycle_tests.cpp` asserts the diagnostic counters directly through
  `sm` methods.
- `src/emel/text/generator/sm.hpp` currently exposes those diagnostics by reading `this->context_`.

</code_context>

<specifics>
## Specific Ideas

Prefer a `event::capture_diagnostics` request carrying a required reference to a public
diagnostics struct. Add explicit `ready` and `uninitialized` transition rows that copy current
diagnostic values in an action.

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope.

</deferred>
