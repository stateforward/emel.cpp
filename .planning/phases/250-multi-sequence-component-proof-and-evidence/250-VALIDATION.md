# Phase 250 Validation

**Date:** 2026-07-04
**Status:** KVP-01 satisfied; KVE-01 and KVD-01 closeout BLOCKED on full
bench-lane repair (see below). PR #94 benchmark snapshots were refreshed after
explicit 2026-07-05 consent for performance visibility only.

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

The full bench_snapshot lane still is not a closeout-proof command on this host
and remains tracked under chip task_48a05fc3. With explicit consent on
2026-07-05, PR #94 refreshed:

- `snapshots/bench/benchmarks.txt` via suite-scoped EMEL snapshot updates so
  arm64-supported suites moved without deleting the committed x86_64 rows;
  the preserved x86_64 block is marked with its previous `/shared/zig/zig`
  provenance because this arm64 host cannot rerun those rows.
- `snapshots/bench/benchmarks_compare.txt` via compare update with
  `EMEL_GENERATION_WORKLOAD_ID=all` so available Liquid LFM2 scaling rows are
  visible. Qwen rows from the previous compare snapshot are preserved with a
  fixture-absence note; generated zero-reference Sortformer/Mimi placeholders
  are not accepted as comparison evidence.

This gives performance visibility for review, including the new graph processor
and decode wavefront rows, but honest KVE-01 closure still requires the
maintained bench commands to pass end to end. Locally absent Qwen/Gemma
generation fixtures are not re-created by the snapshot refresh.

## KVD-01 — docs and full-scope gate (PARTIAL)

`scripts/generate_docs.sh` is idempotent after the milestone changes (no
generated-doc drift; the transition-table shape is unchanged, only guards were
strengthened). Milestone documentation lives in .planning (ROADMAP, phase
docs) and code comments. The full-scope quality gate at closeout is blocked by
the same full bench-lane issue as KVE-01.
