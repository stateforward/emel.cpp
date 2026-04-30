---
phase: 144
status: ready
requirements:
  - TEXTGEN-04
  - TEXTGEN-07
source: .planning/milestones/v1.17-MILESTONE-AUDIT.md
---

# Phase 144 Context: Text Generator Runtime Route Ownership Closure

## Goal

Close the reopened v1.17 source-backed audit gap by moving remaining maintained generator runtime
path choices out of action-called `detail.hpp` helpers and into explicit guard-owned predicates and
`sm.hpp` transitions.

## Audit Findings To Close

- `TEXTGEN-04` is partial because child actors moved and flash route predicates are guard-owned,
  but maintained compute still performs runtime path selection inside action-called `detail.hpp`
  helpers.
- `TEXTGEN-07` is partial because parity and benchmark entrypoints use public generator events,
  but those maintained lanes still execute the same rule-conflicting detail runtime path.

## Required Source-Backed Targets

- `src/emel/text/generator/actions.hpp:310` currently wires graph compute through `detail::*`
  callbacks.
- `src/emel/text/generator/actions.hpp:473` and nearby actions pass detail-owned kernel functions
  into the maintained decode path.
- `src/emel/text/generator/detail.hpp:2091` currently chooses packed-q8, q8, or fallback matmul
  behavior from runtime backend/matrix state.
- `src/emel/text/generator/detail.hpp:4752` currently chooses prefill versus decode inside an
  action-called detail helper.

## Constraints

- Preserve maintained generation behavior, fixture scope, sampling policy, and public API surface.
- Keep EMEL and reference parity/benchmark lanes isolated.
- Runtime behavior choice belongs in `src/emel/text/generator/guards.hpp` and explicit
  `src/emel/text/generator/sm.hpp` transitions.
- `detail.hpp` may keep already-chosen numeric/data-plane work, bounds handling, and monotonic
  iteration, but helper output must not decide which route runs next.
- User has approved updates to models, snapshots, and benchmarks for truthful maintained-path
  proof.

## Expected Verification

- Focused generator/runtime tests.
- Source regression proving the audited detail route choices no longer live in action-called
  helpers.
- `scripts/check_domain_boundaries.sh`.
- Scoped `scripts/quality_gates.sh` with `EMEL_QUALITY_GATES_BENCH_SUITE=generation`.
- Maintained parity/benchmark evidence proving the explicit routes are the paths being run.
