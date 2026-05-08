# Project Research Summary

**Project:** EMEL (emel.cpp)
**Domain:** Constrained-memory tensor loading via staged read I/O strategy
**Researched:** 2026-05-07
**Confidence:** HIGH for process constraints; MEDIUM for final public event names and chunk policy parameters (plan-phase work)

## Executive Summary

v1.26 adds a third concrete loading strategy under `src/emel/io`, focused on
chunked or windowed reads when hosts cannot materialize a full tensor byte span
in one shot. The work must stay consistent with shipped `io/mmap` and `io/read`
actors: tensor residency remains with `model/tensor`, filesystem work stays out
of SML dispatch per established patterns, and runtime routing stays in guards
and transition tables. Cooperative coroutine scheduling is explicitly excluded
unless approved as a separate effort.

The milestone should follow the same macro-phase pattern as v1.24/v1.25:
component boundary, validation gates, execution and lifetime, tensor
integration, public runtime evidence, tests/guardrails, then publication and
artifact refresh when implementation lands.

## Key Findings

### Recommended stack

No new third-party dependencies. Reuse SML, existing I/O module layout, doctest,
and quality-gate workflow. Add only the `staged_read` component and minimal
public wiring.

### Expected features

**Table stakes:** Chunked copy with deterministic ordering; explicit errors;
tensor-owned target buffer; fail-closed validation.

**Defer:** Coroutine/async scheduling, device transfer strategies, automatic
buffer pool management hidden from callers.

### Architecture approach

Parallel `io/staged_read` actor; tensor requests staged loads through the same
class of public I/O boundary events used for mmap/read; loader and tools
observe strategy only through public surfaces.

### Critical pitfalls

Coroutines-as-control-flow, context misuse for stage indices, behavior selection
in `detail`, cross-strategy regressions, and misleading benchmark reporting.

## Implications for roadmap

| Phase | Name | Rationale |
|-------|------|-----------|
| 227 | Component boundary | Canonical `io/staged_read` scaffold |
| 228 | Span/target/platform gating | Preconditions entirely guard-modeled |
| 229 | Copy progress + completion | Deterministic staged byte semantics |
| 230 | Context + handle lifetime + strategy non-residency | EMEL dispatch-local rules |
| 231 | Error taxonomy | Observable exception-free categories |
| 232 | Tensor integration graph | Public tensor↔IO + explicit terminals |
| 233 | Public entrypoints | Loader/bench/parity/probe isolation |
| 234 | Dispatch tests | Doctest success/failure coverage |
| 235 | Guardrails | Loader/tensor/coroutine + mmap/read non-regression |
| 236 | Publication truth | Docs, lint snapshots, bench snapshots, evidence labels |

## Confidence assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Brownfield patterns exist |
| Features | MEDIUM | Exact chunk API to be specified in early execution phases |
| Architecture | HIGH | Matches #60/#62/#63 narrative |
| Pitfalls | HIGH | SML rules are explicit |

### Gaps to address in plan-phase

- Exact event naming and tensor load effect shape for staged requests.
- Whether staging reuses `io/read` primitives internally or duplicates minimal read steps (design choice with review impact).

## Sources

- `.planning/research/STACK.md`
- `.planning/research/FEATURES.md`
- `.planning/research/ARCHITECTURE.md`
- `.planning/research/PITFALLS.md`
- GitHub issue #63 (manager brief)

---
*Research completed: 2026-05-07*
*Ready for roadmap: yes*
