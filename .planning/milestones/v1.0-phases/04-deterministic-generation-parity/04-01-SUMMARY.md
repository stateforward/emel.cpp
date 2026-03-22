---
phase: 04-deterministic-generation-parity
plan: 01
subsystem: paritychecker
tags: [generation, generator, paritychecker, deterministic]
requires: []
provides:
  - Real bounded `emel::generator::event::generate` dispatch from paritychecker
  - Deterministic request sizing derived from the CLI prompt and max-token limit
  - Generate-phase success output that replaces the old initialize-only seam
affects: [paritychecker, generator]
tech-stack:
  added: []
  patterns: [Paritychecker-owned bounded generate harness over existing EMEL generator actors]
key-files:
  created: []
  modified:
    - tools/paritychecker/parity_runner.cpp
key-decisions:
  - "Kept `src/emel/generator/*` as the orchestration source of truth and only widened the tool-local harness around it."
  - "Sized generator initialization capacities from the actual CLI prompt and `--max-tokens` request so the bounded Phase 4 slice does not fail on valid default inputs."
patterns-established:
  - "Pattern: generation mode now reports only after `generation_done` rather than treating initialize readiness as success."
requirements-completed: [GEN-01, GEN-02]
duration: 12min
completed: 2026-03-08
---

# Phase 4 Plan 01 Summary

**Paritychecker generation mode now executes the real bounded EMEL generate path**

## Accomplishments
- Updated [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp) so generation mode no longer stops at the Phase 3 initialize seam; it now runs through the existing EMEL generator graph and only reports success after bounded generation completes.
- Tightened the request shaping so generator initialization capacities are derived from the actual prompt bytes and requested decode budget instead of a smaller hard-coded placeholder.
- Replaced the old initialize-only success contract with generate-phase output that carries `generated_tokens` and output-byte evidence for the bounded Llama-68M slice.

## Task Commits
- No commit created during this execution. The plan was completed locally on `next` while preserving unrelated workspace changes.

## Deviations from Plan
- The plan expected a new focused generator lifecycle edit, but the existing generator suite already covered the relevant contract well enough to verify the widened tool seam. I kept the machine coverage unchanged and verified it instead of duplicating assertions.

## Verification
- `build/zig/emel_tests_bin --dt-test-case="*generator*"`
- `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1`

## Next Readiness
- Wave 2 could assume a real EMEL generate seam exists and focus only on replacing the placeholder decode callbacks with a truthful reference-backed parity path.
