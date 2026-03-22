---
phase: 09-benchmark-integration-hardening
plan: 01
subsystem: benchmark-docs
tags: [benchmark, docs, generation, compare]
requires:
  - phase: 08-generation-compare-output
    provides: canonical generation compare row on the normal `scripts/bench.sh --compare` surface
provides:
  - a published runbook for the canonical generation compare workflow
  - stable case-name filtering guidance for the llama-68m generation row
  - documented generation-specific local override knobs
affects: [09-benchmark-integration-hardening]
tech-stack:
  added: []
  patterns: [docs-first benchmark workflow, compare-row interpretation]
key-files:
  created: []
  modified: [docs/benchmarking.md, README.md]
key-decisions:
  - "Document the generation benchmark through the existing compare workflow instead of inventing a second command path."
  - "Keep README changes to a single discovery pointer and leave the operational details in `docs/benchmarking.md`."
  - "Document generation-only env overrides and the existing case-index/audit knobs as local validation helpers, not a new workflow."
patterns-established:
  - "The canonical Llama-68M generation benchmark is operated and interpreted through `scripts/bench.sh --compare`."
requirements-completed: [VBEN-01]
duration: 0min
completed: 2026-03-10
---

# Phase 09 Plan 1 Summary

**The benchmark docs now publish the canonical Llama-68M generation compare workflow**

## Accomplishments

- Updated `docs/benchmarking.md` with the normal `scripts/bench.sh --compare` flow for the
  canonical generation case `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`.
- Added row-interpretation guidance for the `emel.cpp`, `llama.cpp`, and `ratio` fields and
  documented the generation-specific local override env vars.
- Added a minimal benchmark-workflow pointer in `README.md` so the compare runbook is reachable
  from the main project surface.

## Task Commits

None. Execution stayed local with `commit_docs` disabled.

## Verification

- `rg 'scripts/bench.sh --compare|generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1|EMEL_BENCH_GENERATION_' docs/benchmarking.md`
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare | rg '^generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 .* ratio='`

## Deviations from Plan

None - plan executed exactly as written.

## Next Readiness

- Wave 1 is complete and `VBEN-01` is satisfied.
- Wave 2 remains blocked on explicit user approval before any benchmark snapshot or generated-doc
  refresh.

---
*Phase: 09-benchmark-integration-hardening*
*Completed: 2026-03-10*
