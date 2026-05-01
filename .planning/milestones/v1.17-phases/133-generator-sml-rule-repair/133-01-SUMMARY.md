---
phase: 133
plan: 01
status: complete
requirements-completed:
  - TEXTGEN-04
---

# Phase 133 Summary: Generator SML Rule Repair

## Completed

- Added a generator lifecycle regression that reads `src/emel/text/generator/sm.hpp` and fails if
  the public initialize wrapper contains runtime branch syntax.
- Added a behavioral regression for the removed model/conditioner wrapper branch so missing
  injected dependencies are rejected through the SML transition path.
- Removed the wrapper-level model/conditioner readiness branch from
  `emel::text::generator::sm::process_event(const event::initialize &)`.
- Preserved initialize invalid-request behavior by letting the existing `valid_initialize` and
  `invalid_initialize` transition rows route missing dependencies through SML guards, transitions,
  and initialize publication actions.

## Behavior

No generation request, callback, tokenizer, formatter, sampler, initializer, or prefill behavior
was intentionally changed. Missing initialize dependencies still report
`emel::text::generator::error::invalid_request`; the decision now flows through the transition
table instead of a state-machine member branch.
