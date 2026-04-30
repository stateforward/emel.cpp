---
phase: 137
plan: 1
status: complete
requirements-completed:
  - TEXTGEN-07
completed: 2026-04-29
---

# Summary

Phase 137 closed the paritychecker source-boundary blocker for `TEXTGEN-07`.

## Changes

- Added `tools/paritychecker/generation_internal_diagnostics.hpp` as a single quarantined bridge
  for legacy diagnostic helpers that still need generator-native diagnostic state.
- Removed the direct `emel/text/generator/detail.hpp` include from
  `tools/paritychecker/parity_runner.cpp`.
- Replaced direct `emel::text::generator::detail::*` usage in `parity_runner.cpp` with the
  paritychecker diagnostic bridge namespace.
- Added a paritychecker source regression test proving `parity_runner.cpp` does not directly
  include or name text-generator `detail`, `action`, or `guard` internals.

## Verification

Commands passed:

```sh
cmake --build build/paritychecker_zig --target paritychecker_tests -j2 &&
ctest --test-dir build/paritychecker_zig --output-on-failure
```

```sh
EMEL_QUALITY_GATES_CHANGED_FILES="tests/text/generator/action_guard_tests.cpp,tools/paritychecker/parity_runner.cpp,tools/paritychecker/paritychecker_tests.cpp,tools/paritychecker/generation_internal_diagnostics.hpp" \
  EMEL_QUALITY_GATES_BENCH_SUITE=generation \
  scripts/quality_gates.sh
```

```sh
scripts/check_domain_boundaries.sh
```

Source check:

```sh
rg 'emel/text/generator/detail|emel::text::generator::detail::|namespace emel::text::generator::detail|generator::action|generator::guard' \
  tools/paritychecker/parity_runner.cpp
```

Result: no matches.
