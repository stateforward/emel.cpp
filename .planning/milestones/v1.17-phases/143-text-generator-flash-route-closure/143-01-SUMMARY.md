---
phase: 143
plan: 01
status: complete
requirements-completed:
  - TEXTGEN-04
completed: 2026-04-29T19:11:22Z
---

# Phase 143 Summary: Text Generator Flash Route Closure

## Outcome

Closed the remaining source-backed `TEXTGEN-04` blocker by moving flash/nonflash runtime route
support out of `emel::text::generator::detail` predicate ownership and into guard-owned
predicates consumed by the existing parent and prefill SML routes.

## Changes

- Added guard-owned `guard_first_flash_attention_block(...)` and
  `guard_flash_attention_supported(...)` helpers in `src/emel/text/generator/guards.hpp`.
- Rewired parent decode and prefill flash route guards to call the guard-owned predicate.
- Removed the behavior-selecting `detail::flash_attention_supported(...)` helper while keeping
  already-selected flash execution and request-binding helpers in `detail.hpp`.
- Extended the route-ownership regression so guard sources fail on the flash support predicate
  as well as the previously covered detail route helpers.
- Added focused detail coverage for missing-plan kernel wrapper rejection, fallback dimension
  helpers, packed matrix layout validation, shortconv cache reset, output projection selection,
  and invalid headwise RMS normalization.
- Repaired Phase 142 summary frontmatter to use `requirements-completed:`.

## Verification

- Reproduced the blocker with the broadened route-ownership regression before production fixes.
- `cmake --build build/debug --target emel_tests_bin -j2` passed.
- `ctest --test-dir build/debug -R emel_tests_generator_and_runtime --output-on-failure` passed
  after the fix and after the final guard-helper rename.
- `scripts/check_domain_boundaries.sh` passed.
- Focused source scans found no forbidden parent/prefill guard dependency on behavior-selecting
  `emel::text::generator::detail::*supported` or backend-ready route predicates.
- The scoped generation quality gate passed with changed-file coverage at 90.0% line coverage and
  50.3% branch coverage, paritychecker tests passing, and the generation benchmark lane
  completing successfully.

## Notes

The final benchmark run preserved the maintained generation proof lane, including flash evidence
for the LFM2 workload (`flash_dispatch_calls=174`, `optimized_flash_dispatch_calls=174`) and
native quantized runtime evidence with `disallowed_fallback=0`.
