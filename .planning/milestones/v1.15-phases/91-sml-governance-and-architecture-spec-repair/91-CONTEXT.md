# Phase 91: SML Governance And Architecture Spec Repair - Context

**Gathered:** 2026-04-23
**Status:** Ready for planning

<domain>
## Phase Boundary

Repair the remaining milestone audit governance issues:

- remove action-side runtime branching in `diarization/request` and
  `diarization/sortformer/executor`
- reconcile generated machine documentation with the rule that forbids maintained parallel
  machine-definition specs under `docs/architecture/*`

</domain>

<decisions>
## Implementation Decisions

- Keep optional `error_out` behavior explicit by routing it through guarded publication states
  rather than branching inside actions.
- Replace executor transformer buffer-lane selection with compile-time-selected layer helpers so
  runtime lane choice is no longer hidden in the action body.
- Move docsgen machine outputs to `.planning/architecture/` and remove generated
  `docs/architecture/` files so `src/` remains the public source of truth.

</decisions>
