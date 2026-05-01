# Phase 130 Context: Text Generator Contract And Boundary

## Goal

Move the canonical generative text actor to the text domain without changing runtime behavior.

## Contract

- Canonical source root: `src/emel/text/generator/**`
- Canonical include root: `emel/text/generator/**`
- Canonical namespace: `emel::text::generator`
- Canonical machine type: `emel::text::generator::sm`
- Top-level additive alias remains `emel::Generator`.

## Boundary

No compatibility wrapper is introduced for the old top-level generator ownership. Stale
`src/emel/generator`, `emel/generator`, and `emel::generator` references are treated as domain
boundary violations.

## Call Sites

The move affects CMake test lists, generator tests, embedding shared-session introspection tests,
top-level machine aliases, generation benchmarks, paritychecker, embedded-size probe, coverage
scripts, quality gates, and compliance documentation.
