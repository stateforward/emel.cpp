# Phase 09: Benchmark Integration Hardening Research

**Phase:** 09
**Date:** 2026-03-10
**Requirement IDs:** VBEN-01, VBEN-02
**Source:** `.planning/STATE.md`, `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`,
`08-VERIFICATION.md`, `08-02-SUMMARY.md`, `CLAUDE.md`, `scripts/bench.sh`,
`scripts/quality_gates.sh`, `docs/benchmarking.md`, `docs/benchmarks.md`,
`tools/docsgen/docsgen.cpp`, `tools/bench/bench_main.cpp`, `tools/bench/bench_cases.hpp`,
`tools/bench/generation_bench.cpp`, `snapshots/bench/benchmarks.txt`, and
`snapshots/bench/benchmarks_compare.txt`

## Goal

Close the integration gap for the canonical generation benchmark that Phase 8 already published
through the normal compare surface.

Phase 9 does not own benchmark truth, backend wiring, case pairing, or compare-row formatting.
Those were closed in Phase 07.1 and Phase 8. Phase 9 owns only:

- carrying the existing generation compare row through the existing snapshot/update workflow
- documenting how users run and interpret that existing surface

## What Exists Today

- `tools/bench/bench_cases.hpp` defines one canonical generation case name:
  `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`
- `tools/bench/generation_bench.cpp` already registers both EMEL and reference generation cases
  under that shared name and already supports generation-specific env overrides:
  `EMEL_BENCH_GENERATION_ITERS`, `EMEL_BENCH_GENERATION_RUNS`,
  `EMEL_BENCH_GENERATION_WARMUP_ITERS`, and `EMEL_BENCH_GENERATION_WARMUP_RUNS`
- `tools/bench/bench_main.cpp` already includes the generation case in the default case list and
  already prints compare rows in the stable form
  `case emel.cpp <ns/op>, llama.cpp <ns/op>, ratio=<x>x`
- `scripts/bench.sh --compare` already builds `bench_runner`, runs `--mode=compare`, and prints
  the canonical generation compare row through the normal script surface
- `scripts/bench.sh --compare-update` already writes `snapshots/bench/benchmarks_compare.txt`
- `scripts/bench.sh --snapshot --compare` already derives `snapshots/bench/benchmarks.txt` input
  from compare output, so the generation case can flow into the EMEL snapshot without a second
  runner or a generation-only snapshot mode
- `tools/docsgen/docsgen.cpp` already generates `docs/benchmarks.md` from
  `snapshots/bench/benchmarks_compare.txt`

The important consequence is that the code path Phase 9 needs already exists. The remaining work
is integration closure, not new benchmark behavior.

## Current Gap

The live compare surface and the durable repo surfaces are out of sync.

- Phase 8 verification proved that `scripts/bench.sh --compare` prints the canonical generation
  row.
- `snapshots/bench/benchmarks.txt` does not contain that generation entry yet.
- `snapshots/bench/benchmarks_compare.txt` does not contain that generation row yet.
- `docs/benchmarks.md` therefore also omits the generation row because it is generated from the
  compare snapshot.
- `scripts/quality_gates.sh` still soft-fails benchmark drift, so the missing generation baseline
  is currently visible only as a warning instead of a hard stop.

This is exactly the Phase 9 problem statement recorded in `.planning/STATE.md`.

## Honest Scope

Phase 9 should stay narrow.

- Do not reopen `tools/bench/generation_bench.cpp` backend truth or fixture semantics.
- Do not change the compare row shape established in Phase 8.
- Do not add a generation-only script, snapshot file, JSON report, or alternate docs pipeline.
- Do not strengthen repo-wide benchmark policy beyond what is needed to close the missing
  generation integration gap.
- Do not update snapshots implicitly during normal verification; snapshot refresh remains an
  explicit, user-approved baseline action per `AGENTS.md`.

## Standard Stack

- `scripts/bench.sh`
  Single operator entrypoint for `--compare`, `--compare-update`, `--snapshot`, and
  `--snapshot --update`.
- `tools/bench/bench_main.cpp`
  Single benchmark runner surface for EMEL, reference, and compare modes.
- `snapshots/bench/benchmarks.txt`
  EMEL snapshot baseline used by snapshot gating.
- `snapshots/bench/benchmarks_compare.txt`
  Compare snapshot baseline used for documentation and human review.
- `tools/docsgen/docsgen.cpp` plus `scripts/generate_docs.sh`
  Existing generated-doc pipeline for `docs/benchmarks.md`.
- `docs/benchmarking.md`
  User-facing operator guidance for benchmark workflow policy.

## Architecture Patterns

- Keep one authoritative runtime surface.
  The generation benchmark must continue to ride the existing `bench_runner --mode=compare` and
  `scripts/bench.sh --compare` path.
- Keep one authoritative compare snapshot.
  `snapshots/bench/benchmarks_compare.txt` remains the source for generated benchmark docs.
- Keep one authoritative EMEL snapshot.
  `snapshots/bench/benchmarks.txt` remains the source for regression gating.
- Derive, do not duplicate.
  The combined `--snapshot --compare` path already extracts EMEL `ns_per_op` values from compare
  output. Reuse that behavior instead of adding a second generation snapshot flow.
- Document by stable case name, not by case index.
  `EMEL_BENCH_CASE_INDEX=7` is useful for focused verification, but user-facing docs should key on
  `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` because case ordering can
  change over time.
- Treat generated docs as downstream evidence, not the source of truth.
  The durable source remains the benchmark snapshot files under `snapshots/bench/`.

## Don't Hand-Roll

- Do not add a generation-only benchmark script.
- Do not add a second compare output format just for docs or snapshots.
- Do not hand-edit `docs/benchmarks.md`; regenerate it from `benchmarks_compare.txt`.
- Do not create a separate generation snapshot file.
- Do not introduce new benchmark-selection CLI flags if the existing env and case-name filtering
  are sufficient.
- Do not reopen Phase 8 case-pairing logic unless a real integration defect proves it necessary.

## Common Pitfalls

- Updating only `benchmarks.txt` leaves the repo inconsistent because compare docs still omit the
  generation case.
- Updating only `benchmarks_compare.txt` leaves snapshot gating inconsistent because the EMEL
  baseline still treats the generation case as a new unbaselined entry.
- Documenting `EMEL_BENCH_CASE_INDEX=7` as the public workflow would create fragile instructions
  tied to current registration order.
- Letting seam-audit diagnostics leak to stdout would break the compare snapshot and docs parser.
  Phase 8 already kept that output on stderr; preserve it.
- Treating the quality-gates warning as sufficient proof would leave VBEN-02 only partially
  closed. Phase 9 should prove that the intentional baseline refresh removes the known missing-row
  warning.
- Rewriting benchmark policy in this phase would exceed the roadmap. The hardening target is one
  truthful generation benchmark, not a repo-wide benchmark governance overhaul.

## Code Examples

**Run the normal compare flow and isolate the generation row by name**

```bash
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
scripts/bench.sh --compare | \
  rg '^generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 '
```

**Run the combined gate surface that Phase 9 must keep working**

```bash
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
scripts/bench.sh --snapshot --compare
```

**Intentional baseline refresh after explicit user approval**

```bash
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
scripts/bench.sh --snapshot --update

EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
scripts/bench.sh --compare-update

scripts/generate_docs.sh
```

**Optional focused local proof while keeping docs index-free**

```bash
EMEL_BENCH_CASE_INDEX=7 \
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
scripts/bench.sh --compare
```

## Smallest Honest Phase Split

Two plans are enough, and anything larger would be artificial.

### Plan 09-01: Existing Compare Flow Runbook

Primary requirement: `VBEN-01`

Scope:

- document the exact normal command path for the canonical generation compare run
- document how to isolate and interpret the generation row by stable case name
- document the generation-specific env overrides without creating new CLI surface
- keep the docs anchored to `scripts/bench.sh --compare`, not to a new helper workflow

Why this is honest:

- Phase 8 already delivered the runtime capability.
- What is still missing for `VBEN-01` is durable operator guidance that makes the capability
  discoverable and repeatable without tribal knowledge.

Likely edit surface:

- `docs/benchmarking.md`
- possibly a small README/docs index pointer if the generation workflow is too hidden today

### Plan 09-02: Snapshot And Docs Closure

Primary requirement: `VBEN-02`

Scope:

- intentionally refresh `snapshots/bench/benchmarks.txt` so the EMEL snapshot baseline contains
  the generation benchmark entry
- intentionally refresh `snapshots/bench/benchmarks_compare.txt` so the compare snapshot contains
  the published generation row
- regenerate `docs/benchmarks.md` from the compare snapshot
- prove the existing `scripts/bench.sh --snapshot --compare` path recognizes the generation entry
  without warning about a new unbaselined benchmark

Why this is honest:

- the code already has the plumbing
- the missing baseline rows are the real durable gap blocking `VBEN-02`
- this closes the repo surfaces without inventing any new benchmark machinery

Likely edit surface:

- `snapshots/bench/benchmarks.txt`
- `snapshots/bench/benchmarks_compare.txt`
- `docs/benchmarks.md`
- `docs/benchmarking.md` for the intentional update workflow

## Validation Architecture

Validation should prove the existing workflow end to end, with one focused proof for each durable
surface.

### Runtime Surface Proof

- build `bench_runner`
- run `scripts/bench.sh --compare` with low iteration counts
- assert that stdout contains the canonical generation case name and the existing compare row shape

Recommended command:

```bash
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
scripts/bench.sh --compare | \
  rg '^generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1 .* ratio='
```

### Snapshot Surface Proof

- run `scripts/bench.sh --snapshot --compare`
- confirm the generation benchmark is no longer reported as a new benchmark entry without a
  baseline after the intentional baseline update

Recommended command:

```bash
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
scripts/bench.sh --snapshot --compare
```

### Compare Snapshot And Docs Proof

- run `scripts/bench.sh --compare-update` only during the intentional baseline-refresh step
- run `scripts/generate_docs.sh`
- assert that `snapshots/bench/benchmarks_compare.txt` and `docs/benchmarks.md` both contain the
  canonical generation case name

Recommended commands:

```bash
EMEL_BENCH_ITERS=1 \
EMEL_BENCH_RUNS=1 \
EMEL_BENCH_WARMUP_ITERS=0 \
EMEL_BENCH_WARMUP_RUNS=0 \
scripts/bench.sh --compare-update

scripts/generate_docs.sh

rg 'generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1' \
  snapshots/bench/benchmarks_compare.txt \
  docs/benchmarks.md
```

### Aggregate Gate Proof

- run `scripts/quality_gates.sh`
- confirm benchmark integration no longer depends on the known missing-generation-baseline warning
- do not change the current soft-fail benchmark policy in this phase unless a real integration
  defect forces it

## Recommended Execution Notes

- Keep snapshot updates explicit and user-approved.
- Treat `docs/benchmarks.md` as generated output coupled to `benchmarks_compare.txt`.
- Use case-name matching in docs and verification text; reserve `EMEL_BENCH_CASE_INDEX` for local
  debugging and focused validation.
- If a code change is needed at all, it should be small and script/docs-oriented. The most likely
  successful Phase 9 is mostly baseline and documentation work.
