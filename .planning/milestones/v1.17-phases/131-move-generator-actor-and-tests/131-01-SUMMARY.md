---
phase: 131
plan: 01
status: complete
requirements-completed:
  - TEXTGEN-03
  - TEXTGEN-04
  - TEXTGEN-05
---

# Phase 131 Summary: Move Generator Actor And Tests

## Completed

- Moved the generator actor tree to `src/emel/text/generator/**`.
- Moved generator tests to `tests/text/generator/**`.
- Rewrote canonical includes and namespaces to `emel/text/generator/**` and
  `emel::text::generator`.
- Updated `src/emel/machines.hpp` so `emel::Generator` aliases
  `emel::text::generator::sm`.
- Updated embedding shared-session tests to introspect the text-domain generator child machines.
- Fixed the moved generator detail test helper include.

## Behavior

No request, callback, sampling, prefill, initializer, or decode behavior was intentionally changed.
