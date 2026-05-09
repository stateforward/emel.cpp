# Phase 250: Public Loader Resumable Async Progress - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Preserve bounded cooperative async partial progress across the maintained `io/loader` and
model-loader boundary instead of draining all async chunks inside one local action. This
phase closes `AIO-04`, `TNX-03`, `PERF-01`, and `INT-01`.

</domain>

<decisions>
## Implementation Decisions

### Progress Ownership
- Public loader contracts should expose partial progress, terminal success, and terminal
  failure honestly through explicit events/states.
- `model/tensor` remains the sole owner of tensor residency and observes async outcomes
  through public contracts only.
- Do not let `io/loader` or benchmark entrypoints reach into async actor internals,
  actions, guards, or detail helpers.
- Benchmark/probe output must distinguish partial progress, success, unsupported, and
  fallback paths without presenting fallback behavior as cooperative async evidence.

### the agent's Discretion
All technical choices are at the agent's discretion. Prefer a bounded, public, resumable
contract that composes with existing Phase 249 scheduler/error repairs and keeps existing
synchronous strategies unchanged.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/io/loader/*` owns the public loading strategy policy and maintained loader
  dispatch surface.
- `src/emel/model/tensor/*` owns tensor load, bind, evict, and residency lifecycle
  semantics.
- `src/emel/io/async/*` already has one-chunk progress and terminal callbacks at the
  strategy level.
- `tools/bench/generation_bench.cpp` and `tools/bench/model_load_strategy.hpp` hold
  maintained benchmark/reporting hooks for loading strategy evidence.

### Established Patterns
- Higher layers wire and validate; strategy-specific byte movement stays in owning I/O
  components.
- Tensor integration must dispatch through public event interfaces, not mutate another
  machine's context or call private helpers.
- Changed-file scoped quality gates should use `EMEL_QUALITY_GATES_CHANGED_FILES` because
  the worktree already contains unrelated milestone artifacts.

### Integration Points
- `io/loader` cooperative async strategy policy must surface progress to its caller without
  draining all async progress inside one hidden action loop.
- `model/tensor` must consume async progress/success/failure through explicit public
  events or states.
- Generation benchmark evidence must run through maintained public model-loader/loader
  contracts.

</code_context>

<specifics>
## Specific Ideas

No specific user-facing requirements. Preserve current sync strategy behavior and avoid any
tool-only fallback evidence.

</specifics>

<deferred>
## Deferred Ideas

Broader decode scheduling, tokenizer overlap, platform-specific async I/O, and accelerator
completion remain deferred.

</deferred>
