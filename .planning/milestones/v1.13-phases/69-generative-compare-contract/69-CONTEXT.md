---
phase: 69
slug: generative-compare-contract
created: 2026-04-20
status: in_progress
---

# Phase 69 Context

## Phase Boundary

Phase 69 defines the shared `generation_compare/v1` contract and the first lane-local JSONL
emission seam for maintained generation workloads. It stays in `tools/bench`, scripts, tests, and
planning artifacts; it does not widen `src/` generator runtime scope or introduce manifest-driven
reference backend selection yet.

## Implementation Decisions

### Scope
- Introduce one canonical `generation_compare/v1` record format for generation lanes.
- Reuse the maintained generation benchmark surfaces instead of creating a parallel runtime path.
- Make both EMEL and the existing C++ reference lane emit the same machine-readable contract in a
  dedicated JSONL mode.
- Keep compare output dumps and lane identity outside `src/` runtime ownership.

### Constraints
- Preserve strict EMEL/reference lane isolation.
- Preserve the existing human-readable generation benchmark and compare output mode.
- Keep the first contract grounded in the current maintained generation fixtures and formatter
  contracts.
- Defer manifest-driven workload selection and pluggable backend wiring to later phases.

## Existing Code Insights

- `tools/bench/generation_bench.cpp` already owns the maintained EMEL and reference generation
  fixtures plus the timed generation request path.
- `tools/bench/bench_main.cpp` already exposes lane-specific `--mode=emel` and `--mode=reference`
  entrypoints, but it still prints only human-readable benchmark text.
- `tools/bench/embedding_compare_contract.hpp` provides the closest existing contract pattern for
  schema/versioning, JSONL emission, and optional output dumping.

## Deferred Ideas

- Manifest-driven prompt, sampler, and stop configuration
- Python or external reference backend adapters
- Unified generative compare driver and publication workflow
