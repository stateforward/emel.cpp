---
phase: 03-generator-initialization-wiring
plan: 01
subsystem: paritychecker
tags: [generation, generator, initialization, paritychecker]
requires: []
provides:
  - Real paritychecker-to-generator initialize wiring for the pinned Llama-68M fixture
  - A paritychecker-owned actor harness with tokenizer, conditioner, and generator ownership
  - A tool-level regression test that fails on the old load-only success message
affects: [paritychecker, generator, paritychecker-tests]
tech-stack:
  added: []
  patterns: [Paritychecker-local harness over existing EMEL generator actors]
key-files:
  created: []
  modified:
    - tools/paritychecker/parity_runner.cpp
    - tools/paritychecker/paritychecker_tests.cpp
key-decisions:
  - "Kept generator orchestration inside `src/emel/generator::sm` and only supplied the missing paritychecker-owned actors and init request wiring."
  - "Used bounded opaque topology/plan handles plus deterministic backend callbacks for initialization only, explicitly deferring decode execution to Phase 4."
patterns-established:
  - "Pattern: paritychecker owns the outer harness state while generator, conditioner, memory, graph, sampler, and renderer stay the orchestration source of truth."
  - "Pattern: tool-level regression coverage asserts the generation path no longer stops at the Phase 2 load-only contract."
requirements-completed: [INIT-01]
duration: 39min
completed: 2026-03-08
---

# Phase 3 Plan 01 Summary

**Paritychecker generation mode now reaches the real EMEL generator initialize path**

## Accomplishments
- Updated [parity_runner.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.cpp) so generation mode no longer stops at `model::loader::events::load_done`; it now constructs tokenizer, conditioner, and generator actors around the loaded `emel::model::data` and dispatches `emel::generator::event::initialize`.
- Added bounded init-only scaffolding for the opaque topology/plan/backend inputs required by the initialize contract while keeping actual decode execution out of Phase 3 scope.
- Added [paritychecker_tests.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/paritychecker_tests.cpp) coverage that reproduces the original tool seam by asserting generation mode now prints `generation initialize ok` and no longer emits the old `generator initialization reserved for later phases` success message.

## Task Commits
- No commit created during this execution. The plan was completed locally on `next` while preserving unrelated workspace changes.

## Deviations from Plan
- The plan originally called for only generator lifecycle coverage, but I added tool-level paritychecker output capture as the first failing assertion because the reported gap was a tool contract issue, not a generator machine bug.

## Verification
- `scripts/paritychecker.sh`
- `build/paritychecker_zig/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello`

## Next Readiness
- Wave 2 could rely on a real initialize bridge and focus only on deterministic initialize success/error publication.
