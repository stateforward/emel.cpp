---
phase: 26-canonical-qwen3-fixture-and-conditioning-contract
plan: 03
subsystem: generator
tags: [generator, formatter, conditioner, structured-chat]
requires:
  - phase: 26-canonical-qwen3-fixture-and-conditioning-contract
    provides: structured formatter and conditioner request contract
provides:
  - generator structured-message generate request
  - direct generator-to-conditioner formatter option forwarding
  - regression coverage for structured generator request propagation
affects: [tools/paritychecker, tools/bench, phase-26-04, phase-27]
tech-stack:
  added: []
  patterns: [non-owning chat-message spans, explicit formatter-option request flags]
key-files:
  created:
    - .planning/phases/26-canonical-qwen3-fixture-and-conditioning-contract/26-03-SUMMARY.md
  modified:
    - src/emel/generator/events.hpp
    - src/emel/generator/actions.hpp
    - src/emel/generator/guards.hpp
    - tests/generator/action_guard_tests.cpp
    - tests/generator/lifecycle_tests.cpp
key-decisions:
  - "Kept generator generate requests non-owning with std::span<const chat_message>."
  - "Removed the temporary single-user prompt bridge and forwarded request messages/options directly into conditioner preparation."
patterns-established:
  - "Generator request contracts model chat messages explicitly instead of reconstructing prompt text."
  - "Formatter option booleans travel with the request and are pinned by focused generator regression tests."
requirements-completed: [COND-01]
duration: 7min
completed: 2026-03-28
---

# Phase 26: Generator Structured-Message Boundary Summary

**Generator generate requests now carry explicit chat-message spans and formatter flags through
the conditioner boundary without reconstructing a flat prompt**

## Performance

- **Duration:** 7 min
- **Started:** 2026-03-28T03:25:10Z
- **Completed:** 2026-03-28T03:31:49Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Replaced `generator::event::generate.prompt` with non-owning structured chat-message spans plus
  explicit formatter option booleans.
- Removed the temporary generator bridge that fabricated a single `"user"` message from a flat
  prompt string.
- Added focused generator regression coverage that pins the structured request contract and proves
  formatter-request propagation into conditioner formatting.

## Task Commits

Each task was committed atomically:

1. **Task 1 + Task 2: Structured generator request cutover and regression coverage** - `4ee7af1`
   (feat)

**Plan metadata:** `4ee7af1` (feat: wave summary/state checkpoint)

## Files Created/Modified
- `.planning/phases/26-canonical-qwen3-fixture-and-conditioning-contract/26-03-SUMMARY.md` -
  Captures Wave 3 outcomes and follow-on context.
- `src/emel/generator/events.hpp` - Defines the structured-message generate request contract.
- `src/emel/generator/actions.hpp` - Forwards structured messages and explicit formatter flags into
  conditioner preparation.
- `src/emel/generator/guards.hpp` - Validates non-empty message spans instead of flat prompt text.
- `tests/generator/action_guard_tests.cpp` - Pins structured-message request guard behavior.
- `tests/generator/lifecycle_tests.cpp` - Verifies formatter propagation and the structured request
  contract through generator lifecycle tests.

## Decisions Made
- Kept the generator boundary additive and non-owning with `std::span<const chat_message>`.
- Proved request propagation with a checked formatter hook instead of reintroducing prompt
  reconstruction logic in tests.

## Deviations from Plan

### Auto-fixed Issues

**1. [Blocking bridge] Removed the Wave 26-02 temporary prompt shim during implementation**
- **Found during:** Task 2 (structured generator request cutover)
- **Issue:** `generator::action::request_conditioning` was still fabricating a single `"user"`
  message from `ev.request.prompt`.
- **Fix:** Replaced the shim with direct forwarding of `messages`, `add_generation_prompt`, and
  `enable_thinking`.
- **Files modified:** `src/emel/generator/actions.hpp`
- **Verification:** `./build/zig/emel_tests_bin --test-case='*generator*structured*,*generator*message*' --no-breaks`
- **Committed in:** `4ee7af1` (feat)

---

**Total deviations:** 1 auto-fixed (blocking bridge removal)
**Impact on plan:** Required for correctness. No scope expansion beyond the approved structured
request boundary.

## Issues Encountered
- None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Generator request propagation is ready for maintained paritychecker and benchmark formatter
  binding work in `26-04`.
- Tool surfaces still need their `generator::event::generate` call sites updated before full
  paritychecker/bench builds will be green again.

---
*Phase: 26-canonical-qwen3-fixture-and-conditioning-contract*
*Completed: 2026-03-28*
