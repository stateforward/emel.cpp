# Roadmap

## Archived Milestones

- [x] [v1.0: EMEL Llama-68M Generation Slice](.planning/milestones/v1.0-ROADMAP.md)
- [x] [v1.1: EMEL Llama-68M Generation Benchmark](.planning/milestones/v1.1-ROADMAP.md)
- [x] [v1.2: Flash Attention](.planning/milestones/v1.2-ROADMAP.md)
- [x] [v1.3: ARM Flash Optimizations](.planning/milestones/v1.3-ROADMAP.md)
- [x] [v1.4: Full Vectorized Quantized Kernels](.planning/milestones/v1.4-ROADMAP.md)
- [x] [v1.5: Full ARM Quantized Path](.planning/milestones/v1.5-ROADMAP.md)
- [x] [v1.6: Qwen3-0.6B Parity And Benchmark](.planning/milestones/v1.6-ROADMAP.md)
- [x] [v1.7: Generator Prefill Submachine Decomposition](.planning/milestones/v1.7-ROADMAP.md)
- [x] [v1.8: Truthful Qwen3 E2E Embedded Size](.planning/milestones/v1.8-ROADMAP.md)
- [x] [v1.9: Liquid LFM2.5-1.2B Thinking ARM Slice](.planning/milestones/v1.9-ROADMAP.md)
- [x] [v1.11: TE-75M GGUF Trimodal Embedding Runtime](.planning/milestones/v1.11-ROADMAP.md) - shipped 2026-04-15 with maintained TE trimodal embedding runtime support, refreshed closeout evidence, and a passing milestone audit.
- [x] [v1.12: Pluggable Reference Parity Bench Architecture](.planning/milestones/v1.12-ROADMAP.md) - shipped 2026-04-18, reopened narrowly for archived closeout-proof repair on 2026-04-19, and returned to a passing rerun audit on 2026-04-20.
- [x] [v1.13: Pluggable Generative Parity Bench](.planning/milestones/v1.13-ROADMAP.md) - shipped 2026-04-21 with a maintained generative compare contract, workload manifests, `llama_cpp_generation` reference lane, truthful comparable/non-comparable publication, and a no-blocker audit.

## Current Milestone

### v1.14 Benchmark Variant Organization

**Goal:** Make maintained generation and embedding benchmark variants data/registry-owned so adding
a new deterministic variant does not require modifying unrelated benchmark code.

**Status:** Ready for Phase 77 planning.

## Phases

### Phase 77: Benchmark Variant Registry Contract

**Goal:** Define the shared variant registry shape, validation semantics, and deterministic
ordering guarantees used by both generation and embedding benchmarks.

**Requirements:** REG-01, REG-02, CMP-01

**Success criteria:**
1. A benchmark-owned registry contract describes required identity, fixture, workload, backend,
   comparability, and ordering metadata for both benchmark families.
2. Duplicate IDs, missing required fields, invalid schemas, and nondeterministic ordering fail in
   focused tests before any benchmark run publishes results.
3. Generation and embedding compare wrappers expose compatible selected-variant semantics.

### Phase 78: Generation Workload Discovery Cutover

**Goal:** Replace generation workload hard-coded manifest arrays with deterministic manifest
discovery and prove that a new generation workload can be added by adding local manifest data.

**Requirements:** GEN-01, GEN-02, CMP-03

**Success criteria:**
1. `bench_runner --mode=emel`, `--mode=reference`, and `--mode=compare` enumerate generation
   workloads from checked-in manifests in stable order.
2. Workload filters continue to accept workload ID, case name, and compare group.
3. Comparable and single-lane workload truth remains explicit in emitted `generation_compare/v1`
   records and summaries.

### Phase 79: Embedding Variant Discovery Cutover

**Goal:** Move embedding benchmark case identity and maintained variant metadata out of code-owned
case lists into deterministic registry/discovery data.

**Requirements:** EMB-01, EMB-02, CMP-02

**Success criteria:**
1. The EMEL embedding benchmark lane discovers maintained text, image, and audio cases from a
   registry-owned source in stable order.
2. Reference embedding lanes preserve backend identity, fixture identity, and output metadata
   while consuming the same selected-variant semantics.
3. Existing Python-golden and Liquid C++ compare tests still pass through the operator-facing
   wrapper without hard-coded per-variant test edits.

### Phase 80: Variant Addition Proof And Docs

**Goal:** Prove and document the data-only add path for both benchmark families, then close the
milestone with traceability and quality-gate evidence.

**Requirements:** ADD-01, ADD-02, ADD-03

**Success criteria:**
1. A focused generation regression demonstrates adding a sample workload without editing unrelated
   runner, compare, or test enumeration code.
2. A focused embedding regression demonstrates adding a sample variant without editing unrelated
   runner, compare, or test enumeration code.
3. Developer documentation lists the exact files required for new variants and the files that must
   remain untouched during ordinary variant additions.
4. Requirement traceability, validation, and milestone audit artifacts are ready for closeout.

## Progress

| Milestone | Phases | Plans Complete | Status | Completed |
|-----------|--------|----------------|--------|-----------|
| v1.14 Benchmark Variant Organization | 77-80 | 0/4 | Active | - |
| v1.13 Pluggable Generative Parity Bench | 69-76 | 8/8 | Shipped | 2026-04-21 |

## Next Up

Start Phase 77 with `$gsd-discuss-phase 77`.
