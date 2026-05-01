---
phase: 138
plan: 1
title: Paritychecker Diagnostic Boundary Closure
status: complete
requirements-completed:
  - TEXTGEN-07
completed: 2026-04-29
---

# Summary

Removed the remaining paritychecker diagnostic bridge into text-generator actor internals.

## Changes

- Deleted `tools/paritychecker/generation_internal_diagnostics.hpp`.
- Removed the live include and namespace alias from `tools/paritychecker/parity_runner.cpp`.
- Disabled legacy generation diagnostic implementations that depended on generator actor
  internals.
- Changed `--attribution` for generation mode to fail explicitly until a public generator
  diagnostics contract exists.
- Kept maintained generation parity on the public actor path:
  `emel::text::generator::sm::process_event(...)` for initialize and generate.
- Strengthened `tools/paritychecker/paritychecker_tests.cpp` to scan paritychecker sources and
  bridge headers, not only `parity_runner.cpp`.

## Outcome

`TEXTGEN-07` is closed from source-backed evidence: maintained paritychecker and benchmark proof no
longer include, re-export, or call text-generator actor `detail`, `action`, or `guard` internals.
