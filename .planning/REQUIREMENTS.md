# Requirements: EMEL v1.14 Benchmark Variant Organization

**Defined:** 2026-04-21
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.

## v1 Requirements

### Registry Contract

- [ ] **REG-01**: Developer can inspect one benchmark-owned registry contract that defines the
  required identity, fixture, workload, backend, comparability, and ordering metadata for generation
  and embedding variants.
- [ ] **REG-02**: Developer receives deterministic hard failures for duplicate variant IDs, missing
  required fields, invalid schemas, broken fixture/backend references, or nondeterministic registry
  ordering.

### Generation Variants

- [ ] **GEN-01**: Developer can add a maintained generation workload by adding prompt/workload
  manifest data without editing unrelated benchmark runner, compare, or test enumeration code.
- [ ] **GEN-02**: Operator can run generation benchmarks and compare workflows with deterministic
  workload ordering and the existing workload ID, case name, and compare-group filters.

### Embedding Variants

- [ ] **EMB-01**: Developer can add a maintained embedding benchmark variant through a registry-owned
  source without editing unrelated benchmark runner, compare, or test enumeration code.
- [ ] **EMB-02**: Operator can run embedding benchmarks and compare workflows with deterministic
  variant ordering across EMEL, Python-golden, and Liquid C++ reference lanes.

### Compare Workflow

- [ ] **CMP-01**: Operator can select generation and embedding variants through compatible
  command-line and environment semantics across the compare wrappers.
- [ ] **CMP-02**: Compare summaries preserve selected-variant metadata and reference-backend
  provenance for embedding workflows after registry discovery.
- [ ] **CMP-03**: Compare summaries preserve comparable, non-comparable, and single-lane publication
  truth for generation workflows after registry discovery.

### Additive Proof

- [ ] **ADD-01**: Repository tests prove a sample generation workload can be added data-only while
  preserving deterministic discovery and compare metadata.
- [ ] **ADD-02**: Repository tests prove a sample embedding variant can be added data-only while
  preserving deterministic discovery and compare metadata.
- [ ] **ADD-03**: Developer documentation lists the exact files required for new benchmark variants
  and identifies the runner, compare, and test files that ordinary variant additions must not touch.

## v2 Requirements

Deferred to future milestones.

### Benchmark Extensions

- **BEXT-01**: Developer can package third-party benchmark variants outside the repository.
- **BEXT-02**: Operator can run remote or service-hosted reference engines through the compare
  wrappers.
- **BEXT-03**: Developer can add new public embedding or generation APIs specifically for benchmark
  automation.

## Out of Scope

| Feature | Reason |
|---------|--------|
| New public embedding C ABI or broad CLI API commitments | This milestone only reduces benchmark variant-addition surface. |
| Remote HTTP or service-hosted reference engines | The current compare architecture is local and repo-owned. |
| Public plugin SDK or third-party backend distribution | Internal registry organization should land before public extension points. |
| New EMEL runtime/model support | Variant registry proof should not add runtime behavior solely for demonstration. |
| Performance tuning or benchmark-result optimization | The milestone preserves deterministic evidence; it does not optimize kernels or throughput. |
| Shared model/tokenizer/cache/runtime objects between lanes | Lane isolation remains a hard invariant from v1.12 and v1.13. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REG-01 | Phase 77 | Pending |
| REG-02 | Phase 77 | Pending |
| CMP-01 | Phase 77 | Pending |
| GEN-01 | Phase 78 | Pending |
| GEN-02 | Phase 78 | Pending |
| CMP-03 | Phase 78 | Pending |
| EMB-01 | Phase 79 | Pending |
| EMB-02 | Phase 79 | Pending |
| CMP-02 | Phase 79 | Pending |
| ADD-01 | Phase 80 | Pending |
| ADD-02 | Phase 80 | Pending |
| ADD-03 | Phase 80 | Pending |

**Coverage:**
- v1 requirements: 12 total
- Mapped to phases: 12
- Unmapped: 0

---
*Requirements defined: 2026-04-21*
*Last updated: 2026-04-21 after milestone initialization*
