# Phase 236 Context: Publication and Evidence Truthfulness

## Scope

Phase 236 closes v1.26 with source-backed truth reconciliation for:

- `DOC-01`: docs truth about staged constrained-memory loading and the maintained runtime path.
- `LNT-01`: lint snapshot/maintained lint gate truth.
- `BNH-01`: benchmark snapshot update workflow truth.
- `EVI-01`: parity/benchmark evidence labels must match executed runtime path.

This phase is publication/evidence audit work, not runtime behavior expansion.

## Operating Constraints

- Work from milestone worktree only:
  `/Users/gabrielwillen/.atmux/teams/emel_cpp/milestone63/worktree`.
- Do not use the root checkout as evidence; root contains stale accidental artifacts.
- Do not update snapshots casually.
- Do not claim `scripts/quality_gates.sh` pass unless run and exit `0` is recorded.
- Do not claim benchmark snapshot refresh unless maintained workflow was actually run.
- Do not claim staged-read-backed evidence unless staged runtime path actually executed.

## Prior Phase Truth Carried Forward

From phase verification artifacts (`232`-`235`):

- Phase 232 scoped quality gate is recorded as exit `2`; pass is not claimed.
- Phase 233 and 234 do not claim scoped quality gate pass in their closeout slice.
- Phase 235 records focused doctest/ctest pass evidence and explicitly states:
  scoped quality gate not attempted/no pass claimed.

Phase 236 must preserve these truths when updating milestone docs/state.

## Source-Backed Audit Targets

Primary maintained entrypoints and evidence surfaces to inspect:

- `tools/bench/generation_bench.cpp`
- `tools/bench/diarization/sortformer_fixture.hpp`
- `tools/bench/model_load_strategy.hpp`
- `tools/paritychecker/parity_engines.cpp`
- `tools/embedded_size/emel_probe/main.cpp`
- loader and staged-read runtime surfaces under `src/emel/model/loader`, `src/emel/model/tensor`,
  `src/emel/io/loader`, and `src/emel/io/staged_read`
- maintained docs and planning truth files (`ROADMAP.md`, `REQUIREMENTS.md`, `STATE.md`)

## Expected Deliverables

- `236-CONTEXT.md`
- `236-01-PLAN.md`
- `236-01-SUMMARY.md`
- `236-VERIFICATION.md`
- source-backed updates to milestone docs/state only where claims are inaccurate
