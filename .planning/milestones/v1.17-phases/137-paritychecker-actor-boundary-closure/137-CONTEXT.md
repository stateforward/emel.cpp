---
phase: 137
title: Paritychecker Actor Boundary Closure
status: complete
created: 2026-04-29
completed: 2026-04-29
requirements:
  - TEXTGEN-07
---

# Phase 137 Context

## Starting Point

The refreshed v1.17 milestone audit left `TEXTGEN-07` pending because
`tools/paritychecker/parity_runner.cpp` still included `emel/text/generator/detail.hpp` and named
`emel::text::generator::detail::*` helpers directly. The primary generation result path already
used `emel::text::generator::sm::process_event(...)`, but source review still found actor-internal
bypasses in paritychecker diagnostic and attribution helpers.

## Scope

Remove direct generator actor-internal usage from `parity_runner.cpp` and add a regression test
that prevents reintroducing direct `detail`, `action`, or `guard` references in the runner.

## Non-Goal

This phase does not rewrite every historical paritychecker diagnostic routine. It quarantines the
legacy diagnostic access behind a tool-owned bridge and restores the maintained runner source
boundary so maintained parity proof remains actor-driven.
