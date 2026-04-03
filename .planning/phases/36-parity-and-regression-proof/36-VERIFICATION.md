---
phase: 36-parity-and-regression-proof
verified: 2026-04-02T22:30:00Z
status: passed
score: 3/3 phase truths verified
---

# Phase 36 Verification Report

**Phase Goal:** Prove the executable-size publication plumbing exists for the maintained Qwen3
slice and stays aligned to the generated README path.
**Verified:** 2026-04-02T22:30:00Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | The executable-size script emits the maintained workload/build/smoke metadata schema used by publication. | ✓ VERIFIED | `./scripts/embedded_size.sh --json` returns `mode`, `scope`, `workload`, `backend`, `model_fixture`, `prompt`, `max_tokens`, `runtime_smoke`, and per-row size fields. |
| 2 | The generated README publication path is wired through docsgen and validates cleanly. | ✓ VERIFIED | `./build/docsgen/docsgen --root /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp --check` exited `0`. |
| 3 | The publication wording remains on a matched Qwen3 E2E executable comparison rather than a whole-product claim. | ✓ VERIFIED | The active README/docsgen/planning surfaces consistently describe a matched executable comparison on the canonical Qwen3 slice. |

**Score:** 3/3 truths verified

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| PUB-01 | ✓ PHASE-ALIGNED | Claimed formally in Phase 39 publication refresh |
| PUB-02 | ✓ PHASE-ALIGNED | Claimed formally in Phase 39 publication refresh |

## Automated Checks

- `./scripts/embedded_size.sh --json`
- `./build/docsgen/docsgen --root /Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp --check`

## Verification Notes

- This phase verifies publication plumbing and wording.
- The stale checked-in snapshot is intentionally deferred to Phase 39, which is the closeout
  refresh phase for publication evidence.

---
*Verified: 2026-04-02T22:30:00Z*
*Verifier: the agent*
