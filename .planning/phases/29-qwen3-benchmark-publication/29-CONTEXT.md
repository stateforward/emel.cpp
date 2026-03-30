---
phase: 29
slug: qwen3-benchmark-publication
created: 2026-03-28
status: ready
---

# Phase 29 Context

## Phase Boundary

Publish one truthful maintained compare/docs path for the canonical Qwen3 benchmark slice already
proved by Phase 28 parity. This phase does not widen model support or add any benchmark-only
fallback path.

## Implementation Decisions

### Bench Contract
- The maintained benchmark path must explicitly accept the canonical `qwen3` fixture instead of
  failing behind a stale Llama-only tool gate.
- Unsupported architectures must still fail explicitly; the maintained set stays `llama` and
  canonical `qwen3` only.
- Reference tokenization on the benchmark compare path must use the same GGUF-derived formatter
  contract already proved on parity; no raw-prompt fallback on the maintained path.

### Publication Identity
- Generation compare rows, metadata, and docs must identify the canonical Qwen3 fixture
  explicitly instead of silently reusing the old Llama case names.
- Benchmark publication must expose the canonical Qwen native `q8_0` evidence explicitly rather
  than pretending the old q2/q3/q6 generation evidence still applies.
- Historical preserved ARM flash baseline publication must be handled explicitly if the current
  canonical workload no longer matches that archived Llama artifact.

### Approval And Workflow
- User approval to update benchmark snapshots has been granted in-session on 2026-03-28.
- Snapshot/docs refresh must still run through the maintained workflow only:
  `scripts/bench.sh --compare-update` and `scripts/generate_docs.sh`.
- Benchmark-warning gate policy stays unchanged in this phase.

### the agent's Discretion
- Low-iteration compare commands may be used for quick proof while wiring the maintained path, but
  the checked-in snapshot refresh must come from the maintained compare-update workflow.

## Existing Code Insights

### Reusable Assets
- `tools/generation_formatter_contract.hpp` already models the supported primary GGUF chat
  template contract for both parity and bench surfaces.
- `tools/paritychecker/parity_runner.cpp` already contains the explicit Qwen metadata loading and
  architecture validation rules needed by the maintained tool path.
- `tools/docsgen/docsgen.cpp` already generates benchmark publication from
  `snapshots/bench/benchmarks_compare.txt`.

### Established Patterns
- Maintained tool-specific logic stays in `tools/`, not `src/`, unless runtime support is missing.
- Snapshot refresh is approval-gated and performed through the maintained compare/docs workflow.
- Operator-facing benchmark evidence is printed as explicit metadata lines that docsgen parses
  deterministically.

### Integration Points
- `tools/bench/generation_bench.cpp` owns the canonical benchmark fixture loading, EMEL/reference
  generation setup, and evidence capture.
- `tools/bench/bench_main.cpp` owns compare-mode publication and validation of maintained
  generation evidence.
- `tools/docsgen/docsgen.cpp` and `docs/benchmarking.md` own the operator-facing benchmark docs.

## Specific Ideas

- Mirror the maintained parity phase: explicit Qwen acceptance, explicit formatter contract, and
  explicit case naming instead of any hidden compatibility path.
- Add a narrow `bench_runner` subprocess regression target so the maintained benchmark contract can
  fail in tests before it regresses in docs publication.

## Deferred Ideas

- Broader multi-fixture benchmark publication remains out of scope for v1.6.
- Benchmark gate hardening beyond the current warning-only policy remains deferred by roadmap.
