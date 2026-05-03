---
phase: 193
slug: retired-path-public-docs-guardrail
status: planned
gathered: 2026-05-03
---

# Phase 193 Context

## Audit Gap

The milestone audit found stale public roadmap prose that still described the retired
weight-loader callback path, backend-specific direct I/O, and upload callbacks as maintained truth.
The existing domain-boundary guardrail caught exact retired paths and identifiers but did not catch
semantic stale public documentation.

## Source Context

- `docs/roadmap.md` is public parity roadmap prose and must reflect the maintained tensor-owned
  model loading path.
- `scripts/check_domain_boundaries.sh` is the maintained guardrail for forbidden ownership leaks and
  retired path regressions.

## Constraints

- Public docs must not present a parallel retired residency owner.
- This milestone must continue to state that concrete I/O strategy work is deferred below the
  future `emel/io` seam.
- The guardrail should catch semantic stale retired-owner prose, not only exact source paths.
