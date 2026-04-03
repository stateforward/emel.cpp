---
phase: 35-maintained-runtime-execution-on-arm
verified: 2026-04-02T22:25:00Z
status: passed
score: 3/3 phase truths verified
---

# Phase 35 Verification Report

**Phase Goal:** Prove the matched `llama.cpp` reference executable and runtime smoke behavior for
every published row on the maintained executable-size slice.
**Verified:** 2026-04-02T22:25:00Z
**Status:** passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | The maintained reference row builds as a final linked executable for the same canonical Qwen3 fixture and request slice. | ✓ VERIFIED | `./scripts/embedded_size.sh --json` produced the reference row at `build/embedded_size/reference_probe_build/reference_qwen3_e2e_probe`. |
| 2 | The maintained comparison remains limited to EMEL and one matched `llama.cpp` reference executable. | ✓ VERIFIED | The active planning docs and current harness output include only `emel` and `reference` rows with no additional comparator runtimes in scope. |
| 3 | The current local executable-size flow records a shared passing smoke result for the maintained workload. | ✓ VERIFIED | The latest JSON output records `"runtime_smoke": "passed"` for the maintained `hello` -> first-token workload. |

**Score:** 3/3 truths verified

## Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| REF-01 | ✓ PHASE-ALIGNED | Claimed formally in Phase 38 traceability backfill |
| SMOKE-01 | ✓ PHASE-ALIGNED | Claimed formally in Phase 38 traceability backfill |

## Automated Checks

- `./scripts/embedded_size.sh --json`

## Verification Notes

- This phase verifies comparator existence and smoke behavior. Final published snapshot freshness is
  deferred to the closeout phase.

---
*Verified: 2026-04-02T22:25:00Z*
*Verifier: the agent*
