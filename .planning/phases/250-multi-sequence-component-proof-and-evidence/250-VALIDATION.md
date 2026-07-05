# Phase 250 Validation

**Date:** 2026-07-04
**Status:** KVP-01 satisfied; KVE-01 and KVD-01 closeout BLOCKED on pre-existing
main-branch bench breakage (see below).

## KVP-01 — multi-sequence component proof (DONE)

- `memory_hybrid_interleaved_sequences_isolate_and_recycle_blocks`
  (tests/memory/hybrid/lifecycle_tests.cpp): interleaved growth across two
  sequences with disjoint block sets and distinct recurrent slots; free +
  reallocate reuses reclaimed ids (LIFO, permuted) without disturbing the
  surviving sequence.
- `generator_detail_kv_physical_map_isolates_interleaved_sequences`
  (tests/text/generator/detail_tests.cpp): two sequences bound from one real
  hybrid-machine snapshot produce disjoint physical position sets through the
  addressing helpers; freed-and-reused mappings are non-identity (proving the
  Phase 248 flash guard is load-bearing) and stay disjoint from the survivor.
- Full suite 14/14 with both proofs.

## KVE-01 — benchmark evidence (BLOCKED, upstream)

The bench_snapshot lane cannot run to completion on this host: the reference
comparison lane SIGBUSes inside the pinned ggml (`ggml_is_quantized`) on
pristine origin/main, and the PR #89 suites have no baselines. Both are
tracked (chip task_48a05fc3) and need owner action (reference-pin fix;
consented baseline addition). Runtime-only generation benches run clean and
the parity lane passes on the block-mapped path, but honest KVE-01 closure
requires the maintained bench commands end to end.

## KVD-01 — docs and full-scope gate (PARTIAL)

`scripts/generate_docs.sh` is idempotent after the milestone changes (no
generated-doc drift; the transition-table shape is unchanged, only guards were
strengthened). Milestone documentation lives in .planning (ROADMAP, phase
docs) and code comments. The full-scope quality gate at closeout is blocked by
the same bench lane breakage as KVE-01.
