---
phase: 01-generation-harness-contract
plan: 01
subsystem: infra
tags: [paritychecker, cli, generation, harness]
requires: []
provides:
  - Generation mode enum and option schema in paritychecker
  - CLI parsing and usage text for the Phase 1 generation contract
  - Deterministic malformed-input rejection for generation mode
affects: [generation-harness-contract, emel-model-loading-path]
tech-stack:
  added: []
  patterns: [Extend existing paritychecker mode dispatch before wiring runtime behavior]
key-files:
  created: []
  modified:
    - tools/paritychecker/parity_runner.hpp
    - tools/paritychecker/parity_main.cpp
key-decisions:
  - "Added only one bounded generation setting (`max_tokens`) and reused existing prompt text input to keep the contract narrow."
  - "Kept Wave 1 out of `parity_runner.cpp` so the new mode cannot fake generation parity before Wave 2 adds an explicit dispatch branch."
patterns-established:
  - "Pattern: introduce new parity modes through `parity_mode` and `parity_options` first, then extend CLI parsing."
  - "Pattern: generation mode must reject incomplete invocations at the CLI boundary before any runtime dispatch occurs."
requirements-completed: [HARN-01]
duration: 15min
completed: 2026-03-07
---

# Phase 01: Generation Harness Contract Summary

**Generation mode contract added to paritychecker with explicit CLI parsing and malformed-input rejection**

## Performance

- **Duration:** 15 min
- **Started:** 2026-03-07T22:00:00-06:00
- **Completed:** 2026-03-07T22:15:00-06:00
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Added `generation` to `parity_mode` and extended `parity_options` with a bounded
  `max_tokens` field in [parity_runner.hpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_runner.hpp).
- Updated [parity_main.cpp](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/tools/paritychecker/parity_main.cpp)
  so `--generation` and `--max-tokens` are parsed and documented.
- Made incomplete generation invocations fail deterministically at the CLI boundary instead of
  falling into undefined runtime behavior.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend paritychecker mode and options schema** - `99d0ff6` (feat)
2. **Task 2: Update CLI parsing and usage text for generation mode** - `f62ee21` (feat)

## Files Created/Modified
- `tools/paritychecker/parity_runner.hpp` - Adds the generation mode enum entry and bounded option
  field for later phases.
- `tools/paritychecker/parity_main.cpp` - Adds generation-mode CLI parsing, usage text, and
  positive-integer parsing for `--max-tokens`.

## Decisions Made
- Kept the Wave 1 contract narrow by reusing `--text` / `--text-file` for the prompt and adding
  only one bounded generation-specific knob.
- Deliberately avoided adding a `generation` dispatch branch in `parity_runner.cpp` during Wave 1
  so the tool cannot accidentally claim end-to-end generation before the harness envelope is wired.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- I initially checked the stale `build/paritychecker_zig/paritychecker` binary before the new
  paritychecker build completed, which made the help text look unchanged. Re-running the probe after
  `scripts/paritychecker.sh` finished confirmed the intended CLI contract.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Wave 2 can now add the explicit generation dispatch branch in `parity_runner.cpp` against a
  stable CLI and options contract.
- The positive `--generation --model ... --text ...` invocation still falls through to tokenizer
  behavior because Phase 1 has not yet wired generation dispatch; that is the intended next step.

---
*Phase: 01-generation-harness-contract*
*Completed: 2026-03-07*
