# Phase 248: Maintained Cooperative Async E2E Execution Path - Context

**Gathered:** 2026-05-09
**Status:** Ready for planning

<domain>
## Phase Boundary

This phase closes the v1.27 `PERF-01` gap by making the maintained
model-loader/generation benchmark path execute `cooperative_async` end to end instead
of reporting `io_strategy_unavailable`. The path must enter the async I/O actor through
public loader/tensor contracts and publish real measured evidence.

The user-facing point is constrained-RAM loading for large models: cooperative async
must demonstrate bounded progress over tensor windows so the maintained path does not
need to stage the whole model payload in memory before tensor residency is committed.
</domain>

<decisions>
## Implementation Decisions

### Closeout Semantics
- `PERF-01` is not satisfied by truthful unsupported reporting; it requires a maintained
  `cooperative_async` benchmark run that succeeds and records timing.
- Any fallback or unsupported path must remain labeled as non-async evidence.
- The maintained path must still avoid actor-internal async headers in tools, benchmarks,
  probes, and model-loader code.

### Constrained-RAM Value
- The implementation must explain and preserve the large-model benefit: bounded
  cooperative progress copies the requested logical tensor windows into tensor-owned
  storage without requiring a full model file/materialized byte buffer in the maintained
  loading path.
- Evidence should distinguish entrypoint-level generation benchmark timing from I/O-only
  throughput and should state what peak-memory behavior the path is intended to improve.

### Architecture Boundaries
- `model/tensor` remains the residency owner.
- `io/loader` remains the public strategy boundary.
- `src/emel/io/async` remains the async actor owner.
- Runtime behavior choice stays in guards/transitions; actions/detail helpers must not
  choose fallback or strategy paths.

### Claude's Discretion
- Choose the smallest source-backed maintained path that executes the existing async
  actor through public contracts and passes scoped quality gates.
</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 244 added tensor-owned async request/progress/done/error events and public
  dispatch into `emel::io::async::sm`.
- Phase 245 added the public `cooperative_async` strategy token and maintained strategy
  parsing/reporting helpers.
- Phase 247 documented that the maintained generation benchmark currently reports
  `cooperative_async` as `io_strategy_unavailable`.

### Established Patterns
- Loading strategies are additive under `src/emel/io`.
- Maintained tools and benchmarks use public helpers/entrypoints and must not include
  actor-internal async headers.
- Evidence files distinguish accepted maintained paths from unsupported paths.

### Integration Points
- `tools/bench` generation runner via `EMEL_MODEL_LOAD_IO_STRATEGY`.
- `model/loader` / `io/loader` strategy selection.
- `model/tensor` async public dispatch.
- `247-PERFORMANCE.md`, `REQUIREMENTS.md`, `ROADMAP.md`, `STATE.md`, and the v1.27 audit.
</code_context>

<specifics>
## Specific Ideas

The benchmark command that must stop failing unsupported is:

```sh
EMEL_MODEL_LOAD_IO_STRATEGY=cooperative_async scripts/bench.sh --snapshot --compare --suite=generation
```

The final user explanation must answer: for a large model in constrained RAM, this helps
by loading bounded windows into tensor-owned residency through cooperative progress
rather than forcing the maintained path to allocate or stage the full source payload
before model execution can proceed.
</specifics>

<deferred>
## Deferred Ideas

Continuous decode batching, tokenizer overlap, OS-specific async APIs, DMA/NPU/GPU async
completion, and kernel-internal `co_await` remain out of scope for v1.27.
</deferred>
