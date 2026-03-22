# Phase 08: Generation Compare Output Research

**Phase:** 08
**Date:** 2026-03-10
**Requirement IDs:** COMP-01, COMP-02
**Source:** `.planning/STATE.md`, `.planning/ROADMAP.md`, `.planning/REQUIREMENTS.md`,
`07.1-04-SUMMARY.md`, `07.1-VERIFICATION.md`, `tools/bench/*`, and `scripts/bench.sh`

## Goal

Finish the truthful compare surface for the canonical Llama-68M generation benchmark now that
Phase 07.1 replaced the EMEL-side reference-backed decode seam with the shared native EMEL decode
backend.

Phase 8 is not backend work anymore. Its job is to make the existing compare flow honest,
explicit, and inspectable for the one canonical generation case.

## What Exists Today

- `tools/bench/bench_main.cpp` already supports `--mode=compare` and already prints compare rows in
  the form:
  `case emel.cpp <ns/op>, llama.cpp <ns/op>, ratio=<x>x`.
- `tools/bench/bench_cases.hpp` and `tools/bench/generation_bench.cpp` already register both
  `append_emel_generation_cases(...)` and `append_reference_generation_cases(...)` under the same
  canonical case name:
  `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1`.
- `scripts/bench.sh --compare` already builds `bench_runner`, runs `--mode=compare`, prints the
  compare output, and can update `snapshots/bench/benchmarks_compare.txt`.
- Phase 07.1 proved that the EMEL benchmark path now uses the shared native backend while the
  explicit reference path remains separate, with seam-audit evidence kept behind
  `EMEL_BENCH_AUDIT_GENERATION_SEAMS`.

The key consequence is that Phase 8 should not reopen case registration, fixture wiring, or decode
truth work unless a small compare-surface defect is discovered while hardening the output.

## Honest Scope After Phase 07.1

The roadmap phrasing for Phase 8 still sounds larger than the code reality. In practice,
`COMP-01` is almost structurally satisfied already because compare mode runs both EMEL and
reference cases through the same `bench_runner --mode=compare` surface.

What remains is the narrow part of the requirement that users actually care about:

1. The canonical generation compare row must remain present and stable in normal compare output.
2. The compare row must clearly refer to the real canonical workload, not an alternate benchmark
   shortcut or audit-only path.
3. Any supporting diagnostics needed for truth confidence must stay opt-in so normal compare output
   remains compatible with the existing bench scripts and snapshots.

That means the smallest honest Phase 8 split is a surface-hardening phase, not another
implementation-heavy milestone.

## Existing Compare Contract

The current compare contract is already close to the desired outcome:

- `bench_main.cpp` sorts EMEL and reference results by case name.
- It exits on case-name mismatch or case-count mismatch.
- It prints both timings and the ratio on one line.
- `generation_bench.cpp` publishes the EMEL and reference generation cases under the same case
  name, using the same pinned fixture, prompt, and max-token contract.

This is a good baseline because it means Phase 8 can be limited to making the compare surface
explicit and resilient rather than inventing a new output format.

## Smallest Honest Plan Split

Two plans are enough.

### Plan 08-01: Compare Contract Hardening

Scope:
- Confirm the canonical generation case is visible and stable in the normal compare path.
- Tighten any compare-mode behavior that could let non-generation cases mask a generation mismatch,
  ambiguous row identity, or accidental drift in the canonical case naming/ordering contract.
- Add focused compare-surface tests around the existing `bench_runner --mode=compare` output for
  the generation row.

Why this is its own plan:
- It closes `COMP-01` honestly with the smallest possible edit surface.
- It keeps the work inside `tools/bench` and test coverage, with no new backend scope.

### Plan 08-02: Published Compare Evidence

Scope:
- Lock the user-facing compare row shape for the canonical generation case so `COMP-02` is proven
  through the normal script surface.
- Preserve the existing bench style instead of inventing a second report format.
- Keep seam-audit output opt-in and stderr-only so normal compare snapshots remain clean.

Why this is separate:
- It is mostly output-contract proof and script-surface verification.
- It naturally hands off to Phase 9, which owns broader snapshot and documentation integration.

## Likely Edit Surface

- `tools/bench/bench_main.cpp`
  If any compare-surface hardening is needed, it should happen here first because this is where
  paired result validation and printed compare rows already live.
- `tools/bench/generation_bench.cpp`
  Only for narrow fixes tied to case naming, generation-case registration, or audit gating. Phase 8
  should not reopen backend logic here.
- `tools/bench/bench_cases.hpp`
  Only if a declaration-level cleanup is needed to support testability or keep the generation case
  contract explicit.
- `scripts/bench.sh`
  Only for minimal compare-surface alignment. Broader snapshot/update workflow work still belongs
  in Phase 9.
- benchmark-focused tests under `tools/bench` or existing tool tests
  The phase needs output-contract coverage, not more decode-path coverage.

## Constraints And Non-Goals

- Do not change state-machine structure in `src/`; this phase should stay tool-surface-focused.
- Do not reintroduce `llama_decode` or `llama_get_logits_ith` into the EMEL benchmark path.
- Do not create a benchmark-only compare workflow outside `bench_runner --mode=compare` and
  `scripts/bench.sh --compare`.
- Do not broaden into multiple prompts, multiple token budgets, or multiple fixtures.
- Do not update snapshots automatically; Phase 9 owns the durable snapshot/tooling closure.

## Risks And Pitfalls

- The biggest planning risk is pretending Phase 8 still needs backend work. That would duplicate
  Phase 07.1 and blur the truth boundary that was just corrected.
- If Phase 8 changes compare output more than necessary, it will create avoidable snapshot churn
  and bleed Phase 9 work backward.
- If the generation compare row is only observable through audit env vars, `COMP-01` and `COMP-02`
  remain weak because the normal user flow is still underspecified.
- If seam-audit diagnostics leak into normal stdout, they will destabilize the compare snapshot
  surface and mix proof-of-truth concerns with the user-facing benchmark report.

## Validation Architecture

Phase 8 validation should prove the normal compare surface, with seam-audit checks used only as
supporting evidence that the truthful backend boundary from Phase 07.1 remains intact.

**Build loop**
- `cmake --build build/bench_tools_ninja --parallel --target bench_runner`

**Primary compare-flow proof**
- `EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare`

**Focused generation-row inspection**
- `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=compare`
- `EMEL_BENCH_CASE_INDEX=7 build/bench_tools_ninja/bench_runner --mode=compare | rg "^generation/"`

**Truth-supporting audit check**
- `EMEL_BENCH_CASE_INDEX=7 EMEL_BENCH_AUDIT_GENERATION_SEAMS=1 EMEL_BENCH_ITERS=1 EMEL_BENCH_RUNS=1 EMEL_BENCH_WARMUP_ITERS=0 EMEL_BENCH_WARMUP_RUNS=0 scripts/bench.sh --compare`

The validation bar for this phase should be:
- the canonical generation compare row appears in the standard compare surface
- that row includes EMEL timing, `llama.cpp` timing, and ratio
- the row is produced from the same native EMEL case already proven in Phase 07.1
- optional seam auditing still shows zero reference-wrapper hits on the EMEL path

## Planning Recommendation

Keep Phase 8 to two plans:

1. **Compare contract hardening**
   Lock the canonical generation row into the existing compare path and add focused output-contract
   verification.
2. **Published compare evidence**
   Prove the compare row through the normal script surface while keeping truth-support diagnostics
   opt-in and compatible with the current benchmark tooling.

Anything broader than that belongs in Phase 9 or a later milestone.
