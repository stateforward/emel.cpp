# Phase 252: Large-Model Constrained-RAM Profiling And Optimization - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Profile and optimize the maintained cooperative async loading path on a model larger than
available RAM, or a maintained constrained-RAM emulation when a real model is unavailable,
without substituting tool-only fallbacks for runtime behavior. This phase closes `PERF-02`.

</domain>

<decisions>
## Implementation Decisions

### Profiling Contract
- Profiling must exercise `cooperative_async` through public model-loader, `io/loader`, and
  tensor contracts.
- The profile loop should record bottleneck evidence, apply scoped optimizations, and rerun
  the same maintained path until remaining bottlenecks are documented.
- Do not report whole-model in-RAM shortcuts, unsupported strategy results, or tool-only
  compute fallbacks as cooperative async performance evidence.
- Output must state model size, effective RAM constraint, chunk/window behavior, peak
  memory, throughput/latency, and whether the run used a real model or constrained-RAM
  emulation.

### the agent's Discretion
All profiling and optimization choices are at the agent's discretion. If a real
larger-than-RAM model is unavailable locally, use a maintained constrained-RAM emulation
that still drives the public cooperative async path.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/bench/generation_bench.cpp` and `tools/bench/model_load_strategy.hpp` provide the
  maintained generation benchmark and strategy-reporting surface.
- `src/emel/io/loader`, `src/emel/io/async`, and `src/emel/model/tensor` are the runtime
  path that profiling must exercise.
- Existing benchmark infrastructure under `tools/bench` supports source-backed evidence and
  must keep EMEL-owned lanes separated from reference lanes.

### Established Patterns
- Benchmark claims must name the actual maintained runtime path and avoid tool-only
  substitutes.
- Performance work should optimize the code path under test, not bypass it.
- Scoped quality gates and relevant benchmark/profiling commands must pass without
  benchmark-regression override.

### Integration Points
- Phase 252 depends on Phases 249-251 so error semantics, progress visibility, and evidence
  metadata are already truthful before profiling.
- Any new profiling command or benchmark report should integrate with existing bench
  manifests and publication evidence rather than inventing a parallel reporting path.

</code_context>

<specifics>
## Specific Ideas

No real model path was provided in the request. Use a constrained-RAM maintained emulation
if local artifacts do not include a suitable large model.

</specifics>

<deferred>
## Deferred Ideas

Device-specific async completion, decode overlap, and accelerator scheduling remain outside
this milestone.

</deferred>
