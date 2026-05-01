---
phase: 139
plan: 1
title: Generator Benchmark Diagnostics Boundary Closure
status: complete
requirements-completed:
  - TEXTGEN-07
completed: 2026-04-29
---

# Summary

Closed the remaining benchmark diagnostics boundary gap for `TEXTGEN-07`.

## Changes

- Added `emel::text::generator::diagnostics` and
  `emel::text::generator::event::capture_diagnostics` as the public generator diagnostics
  contract.
- Routed diagnostics capture through explicit generator SML transitions in `sm.hpp` and a bounded
  action in `actions.hpp`.
- Removed the context-reading generator `sm` diagnostic getters.
- Updated generator tests, the benchmark generation lane, and the paritychecker generation lane to
  capture diagnostics via `process_event(event::capture_diagnostics{...})`.
- Removed benchmark includes of text-generator actor `detail`, `action`, and `guard` internals.
- Strengthened benchmark and paritychecker source-boundary tests to reject actor-internal bridges
  and the removed `->generation_` getter pattern.
- Fixed Phase 138 frontmatter from `requirements:` to `requirements-completed:`.

## Outcome

`TEXTGEN-07` is source-backed from live code: maintained generation parity and benchmark proof now
use public actor event surfaces for generator diagnostics and no longer read text-generator actor
context through `sm` member functions.
