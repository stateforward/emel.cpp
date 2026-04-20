# Roadmap

## Archived Milestones

- [x] [v1.0: EMEL Llama-68M Generation Slice](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.0-ROADMAP.md)
- [x] [v1.1: EMEL Llama-68M Generation Benchmark](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.1-ROADMAP.md)
- [x] [v1.2: Flash Attention](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.2-ROADMAP.md)
- [x] [v1.3: ARM Flash Optimizations](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.3-ROADMAP.md)
- [x] [v1.4: Full Vectorized Quantized Kernels](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.4-ROADMAP.md)
- [x] [v1.5: Full ARM Quantized Path](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.5-ROADMAP.md)
- [x] [v1.6: Qwen3-0.6B Parity And Benchmark](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.6-ROADMAP.md)
- [x] [v1.7: Generator Prefill Submachine Decomposition](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.7-ROADMAP.md)
- [x] [v1.8: Truthful Qwen3 E2E Embedded Size](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.8-ROADMAP.md)
- [x] [v1.9: Liquid LFM2.5-1.2B Thinking ARM Slice](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.9-ROADMAP.md)
- [x] [v1.11: TE-75M GGUF Trimodal Embedding Runtime](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.11-ROADMAP.md) - shipped 2026-04-15 with maintained TE trimodal embedding runtime support, refreshed closeout evidence, and a passing milestone audit.

## Current Milestone

### v1.12 Pluggable Reference Parity Bench Architecture (Reopened For Archived Closeout Proof Repair)

This milestone reopened on 2026-04-19 after the post-archive rerun audit found one narrow
closeout-proof contradiction: archived Phase `67` still references pre-archive planning paths in
its validation and verification commands. The reopen stays bounded to planning artifacts, archived
evidence, and rerun-audit truth. No runtime scope, backend scope, or public-surface scope is
changing.

## Phases

**Phase Numbering:**
- `v1.12` originally ran from Phase `62` through Phase `67`.
- The archival-proof repair continues from the next free phase number, so the reopen starts at
  Phase `68`.

- [x] **Phase 62: Reference Backend Contract** - Define the canonical pluggable compare contract
  while preserving strict EMEL/reference lane isolation.
- [x] **Phase 63: Python Reference Backend** - Bring maintained Python backends onto the shared
  compare contract with deterministic and explicit failure reporting.
- [x] **Phase 64: C++ Reference Backend Integration** - Move the maintained C++ reference lane onto
  the same manifest-driven compare contract.
- [x] **Phase 65: Unified Compare Workflow And Publication** - Publish one operator-facing compare
  workflow and artifact surface across backend languages.
- [x] **Phase 66: Repair Unified Compare Publication** - Preserve all maintained baseline records
  in shared-group compare publication.
- [x] **Phase 67: v1.12 Traceability And Nyquist Closeout** - Backfill requirement-traceability
  and Nyquist evidence for the reopened closeout sweep.
- [ ] **Phase 68: Refresh Archived v1.12 Closeout Proof Paths** - Repair archived Phase `67`
  proof paths and rerun the milestone audit against the archived workspace.

## Phase Details

### Phase 68: Refresh Archived v1.12 Closeout Proof Paths
**Goal**: Make the archived `v1.12` closeout proof self-consistent by updating Phase `67`
verification and validation evidence to reference archived `v1.12` planning paths, then rerun the
milestone audit from the reopened live ledger.
**Depends on**: Phase 67
**Requirements**: none - this phase repairs archived rerun evidence only; all `v1.12` product
requirements remain satisfied.
**Gap Closure**: Closes the broken "archived closeout evidence rerun readiness" flow and the
invalid archived Phase `67` validation evidence called out by the latest `v1.12` milestone audit.
**Success Criteria** (what must be TRUE):
  1. Archived Phase `67` verification and validation artifacts reference
     `.planning/milestones/v1.12-phases/...` and `.planning/milestones/v1.12-REQUIREMENTS.md`
     instead of removed live-root planning paths.
  2. The reopened live ledger under `.planning/` reflects `v1.12` as the current milestone,
     including a current audit file and requirements working copy suitable for rerun workflows.
  3. `$gsd-audit-milestone` reruns against reopened `v1.12` and reports no remaining archived
     closeout-proof contradiction.
**Plans**: none yet

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 62. Reference Backend Contract | 1/1 | Complete | 2026-04-17 |
| 63. Python Reference Backend | 1/1 | Complete | 2026-04-17 |
| 64. C++ Reference Backend Integration | 1/1 | Complete | 2026-04-17 |
| 65. Unified Compare Workflow And Publication | 1/1 | Complete | 2026-04-17 |
| 66. Repair Unified Compare Publication | 1/1 | Complete | 2026-04-18 |
| 67. v1.12 Traceability And Nyquist Closeout | 1/1 | Complete | 2026-04-18 |
| 68. Refresh Archived v1.12 Closeout Proof Paths | 0/0 | Planned | - |
