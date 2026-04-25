# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.15 - ARM Sortformer Diarization GGUF Slice

**Shipped:** 2026-04-25
**Phases:** 24 | **Plans:** 24 | **Sessions:** autonomous execution and closeout

### What Was Built

- One maintained `openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf` ARM diarization
  slice with pinned fixture provenance and explicit loader/model acceptance.
- A diarization-owned mono 16 kHz PCM request contract, native feature preparation, native
  Sortformer encoder/executor path, and deterministic probability/segment publication.
- Lane-isolated PyTorch/NeMo parity and ONNX CPU single-thread benchmark reference lanes.
- A strict generated benchmark record where EMEL exact-matches PyTorch/NeMo and ONNX and beats
  ONNX CPU single-thread.
- Source-backed closeout artifacts tracing claims from fixture through loader, runtime, parity,
  benchmark, docs, and milestone audit.

### What Worked

- The milestone did not close until synthetic/tool-only evidence was replaced by maintained runtime
  evidence.
- Treating PyTorch/NeMo as the parity reference and ONNX as the benchmark reference made the proof
  roles explicit and reviewable.
- Recursive profiling focused optimization on measured hotspots instead of guessing.
- Kernel/runtime ownership rules prevented a second ad hoc diarization compute lane from becoming
  permanent.

### What Was Inefficient

- Early closeout evidence overstated benchmark/parity truth and forced multiple inserted repair
  phases.
- Full quality gates and benchmark regeneration consumed too much wall-clock time during the final
  loop.
- The `audit-open` CLI had a wrapper bug (`output` instead of `core.output`) that blocked closeout
  until repaired.
- Summary one-liner extraction produced low-quality `Completed:` accomplishments, requiring manual
  MILESTONES cleanup.

### Patterns Established

- Milestone benchmark claims must be verified by tracing the live maintained codepath, not by
  trusting ROADMAP, SUMMARY, VERIFICATION, or generated docs alone.
- Reference lanes should have named roles: parity reference, benchmark reference, or diagnostic
  reference.
- Performance closeout records must include provider/thread contract, output identity, checksum,
  and generated record paths.
- If optimized kernel work is required for truth, add phases until it is in the kernel/runtime
  domain rather than keeping parallel helpers.

### Key Lessons

1. Source-backed audit rules need to run before milestone closeout, not after the first closeout
   attempt.
2. Benchmark gates need deterministic generated reports, but the slowest docs/gate steps should be
   profiled and optimized as tooling debt.
3. Summary frontmatter should contain useful one-liners so archive automation does not need manual
   repair.

### Cost Observations

- Model mix: not measured.
- Sessions: autonomous execution plus repeated scheduled Phase 93 optimization ticks.
- Notable: final scoped quality gate timing was `246s`, with docs generation dominating at `211s`.

---

## Milestone: v1.13 - Pluggable Generative Parity Bench

**Shipped:** 2026-04-21
**Phases:** 8 | **Plans:** 8 | **Sessions:** current autonomous closeout

### What Was Built

- One canonical `generation_compare/v1` contract for EMEL and reference generation lanes.
- Manifest-pinned maintained generation workloads with prompt, formatter, sampling, stop, seed,
  and comparability metadata.
- A maintained `llama_cpp_generation` reference backend behind operator-facing wrapper tooling.
- Truthful compare publication for exact matches, bounded drift, non-comparable runs, missing
  lanes, and errors.
- Regression coverage for both comparable LFM2 workflow publication and selected single-lane
  non-comparable publication.

### What Worked

- The audit gap workflow was useful: the initial audit exposed real behavior and evidence gaps,
  then Phases 74 through 76 closed them without broadening the milestone.
- Keeping reference backend setup in wrapper/tooling space preserved the EMEL runtime boundary.
- Requirement and Nyquist backfills made the final audit mechanically checkable instead of relying
  on narrative closeout.

### What Was Inefficient

- Early closeout summaries did not expose `requirements-completed` consistently, so audit tooling
  could not infer accomplishment and requirement coverage cleanly.
- The `audit-open` CLI path has a tooling bug in its direct command wrapper, requiring a manual
  call into the audit library during closeout.
- Metadata mismatch tests cover representative fields directly while implementation checks the full
  tuple; this is acceptable but leaves coverage debt.

### Patterns Established

- Comparable and non-comparable generative workloads should both flow through the same
  operator-facing wrapper, with the verdict carrying the truth rather than the entrypoint changing.
- Material comparability metadata must be checked before output comparison.
- Single-lane publication should leave the absent lane's raw record file empty and explain the
  non-comparable reason explicitly.

### Key Lessons

1. Audit evidence should be written at the same time as behavior work, not reconstructed at the end.
2. Wrapper-level tests catch integration claims that JSONL-only tests can miss.
3. Closeout artifacts need stable frontmatter keys so milestone tooling can summarize without
   hand repair.

### Cost Observations

- Model mix: not measured.
- Sessions: current autonomous closeout plus phase execution sessions.
- Notable: final audit did not rerun the full quality gate; it relied on the Phase 75 post-review
  full gate pass.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Sessions | Phases | Key Change |
|-----------|----------|--------|------------|
| v1.15 | autonomous execution/closeout | 24 | Source-backed audit and recursive profiling became mandatory before milestone archive. |
| v1.13 | current closeout | 8 | Audit gaps became explicit repair phases before archive. |

### Cumulative Quality

| Milestone | Tests | Coverage | Zero-Dep Additions |
|-----------|-------|----------|-------------------|
| v1.15 | Sortformer runtime/parity/benchmark tests plus final scoped quality gate | Line coverage gate preserved; scoped gate timing `246s` | No external runtime dependency in the EMEL lane. |
| v1.13 | `generation_compare_tests`, focused `bench_runner_tests`, quality gates | 90.4% line coverage at last full gate | No new runtime dependency in `src/`. |

### Top Lessons

1. Maintain lane isolation as a testable contract, not just a design statement.
2. Publish non-comparable outcomes explicitly when workload or metadata contracts diverge.
3. Keep milestone closeout files machine-readable enough for audit tooling.
4. Treat benchmark performance claims as source-backed claims that must trace through the live
   maintained runtime path.
