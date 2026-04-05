# Requirements: EMEL

**Defined:** 2026-04-04
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented verification before widening API surface or model scope.

## v1 Requirements

### Planner Surface

- [ ] **PLAN-01**: Maintainer can locate the planner machine under `src/emel/batch/planner/`
  using only canonical component base files allowed by `AGENTS.md`.
- [ ] **PLAN-02**: Maintainer can use a canonical planner machine type and wrapper naming that
  follow the exported and internal naming rules in `AGENTS.md`.
- [ ] **PLAN-03**: Maintainer can trace planner-owned orchestration logic inside planner component
  files rather than mixed helper surfaces outside the component boundary.

### Planner Modes

- [ ] **MODE-01**: Maintainer can locate the `simple`, `sequential`, and `equal` planner child
  machines under planner-owned component paths that match the `AGENTS.md` naming and layout
  contract.
- [ ] **MODE-02**: Planner child machines communicate through explicit machine interfaces and typed
  events rather than direct cross-machine action calls or context mutation.
- [ ] **MODE-03**: Planner-mode machine files expose only the canonical machine, data, guard,
  action, event, error, and detail surfaces needed by the contract.

### Rule Compliance

- [ ] **RULE-01**: Planner-family transition tables use destination-first row style with readable
  phase sections and no new source-first rows.
- [ ] **RULE-02**: Planner-family events follow the `AGENTS.md` naming contract, including intent
  events in `event` namespaces and outcome events in `events` namespaces with explicit `_done` and
  `_error` suffixes where applicable.
- [ ] **RULE-03**: Planner-family context stores only persistent actor-owned state and does not
  mirror per-dispatch request, phase, status, or output data.

### Proof

- [ ] **PROOF-01**: The planner-family hard cutover is covered by focused tests that prove the
  maintained batching behavior is preserved after the structural changes.
- [ ] **PROOF-02**: Required validation for this milestone runs successfully on the current `x86`
  development environment without claiming ARM performance parity or benchmark publication.

## v2 Requirements

### Architecture Follow-Ons

- **ARCH-01**: Additional machine families beyond `src/emel/batch/planner` can be hard-cut over to
  the same `AGENTS.md` contract once the planner family proves the pattern.
- **ARCH-02**: Generator child machines such as `prefill` and `initializer` can adopt the same
  planner-family naming and proof pattern in a later milestone with explicit scope approval.

### ARM Validation

- **ARM-01**: Planner-adjacent performance or benchmark claims can be reintroduced when an ARM host
  is in scope for truthful validation.

## Out of Scope

| Feature | Reason |
|---------|--------|
| ARM benchmark publication or optimization claims | Current development environment is `x86`, so this milestone cannot make truthful ARM benchmark claims |
| Generator child machines such as `prefill` or `initializer` | User scoped the milestone to the planner family only |
| Broad repository-wide naming cleanup outside `src/emel/batch/planner` | The milestone is a bounded planner-family cutover, not a whole-repo rewrite |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| PLAN-01 | Phase 40 | Pending |
| PLAN-02 | Phase 40 | Pending |
| PLAN-03 | Phase 40 | Pending |
| MODE-01 | Phase 41 | Pending |
| MODE-02 | Phase 42 | Pending |
| MODE-03 | Phase 41 | Pending |
| RULE-01 | Phase 43 | Pending |
| RULE-02 | Phase 42 | Pending |
| RULE-03 | Phase 43 | Pending |
| PROOF-01 | Phase 44 | Pending |
| PROOF-02 | Phase 44 | Pending |

**Coverage:**
- v1 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0

---
*Requirements defined: 2026-04-04*
*Last updated: 2026-04-04 after v1.10 roadmap creation*
