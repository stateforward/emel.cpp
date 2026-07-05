# Phase 246 Validation

**Date:** 2026-07-04

- `ctest emel_tests` + `lint_snapshot`: 14/14 pass (first run after the change —
  the live prefill/decode flows already produce coherent snapshots at compute
  time, which is the KVM-03 claim, now guard-enforced).
- Scoped gates: `test_with_coverage` PASS (293s), `paritychecker` PASS (81s),
  `fuzz_smoke` skipped; `bench_snapshot` FAIL 138 = pre-existing reference-lane
  ggml SIGBUS reproduced on pristine main (standing disposition, see Phase 245
  validation; chip task_48a05fc3).

## What the phase enforces

`guard_snapshot_geometry_coherent` (snapshot block_tokens == backend
kv_block_tokens, maintained sequence active) and `guard_snapshot_covers_tokens`
(sequence length exactly accounts for the tokens the compute phase addresses,
with a valid block mapping for the last token) are folded into
`guard_prefill_request_ready` (prompt_token_count) and
`guard_decode_request_ready` (kv_tokens + 1). Their negations route through the
existing explicit invalid transitions in both transition tables. No token
accounting was added to machine context; the backend write cursor comparison
remains as cross-validation.

Guard-level tests cover geometry drift, inactive sequence, length drift, and
missing block mapping (`tests/text/generator/action_guard_tests.cpp`).
