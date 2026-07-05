# Requirements: EMEL v1.28 Memory-Owned KV Block Addressing Cutover

**Defined:** 2026-07-04
**Status:** Active
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and
parity-oriented verification before widening API surface or model scope.
**Source:** Source-backed audit (2026-07-04): `memory::hybrid` performs real block-pool
accounting on the generate path, but physical K/V storage in the generator backend is
linearly addressed (`backend.kv_cache_tokens = position + 1`), the captured
`memory::view::snapshot` is threaded into graph/processor events yet never dereferenced
in `src/`, and `sequence_recurrent_slot` is unused. The memory domain is the KV cache's
accountant, not its landlord.

## v1.28 Requirements

Each requirement is one independently testable obligation and maps to exactly one
roadmap phase. This milestone cuts physical KV-cache addressing over to the memory
machine's block map on the maintained single-sequence generation path, and proves the
block map with component-level multi-sequence coverage. It does NOT widen the public
generation API to multi-sequence serving, does not add eviction/prefix reuse, and does
not change quantization or kernel arithmetic.

### Block geometry and slot allocation

- [x] **KVM-01**: KV cache physical geometry (block size in tokens, per-layer block
  capacity) is derived from one shared block-geometry contract owned by the memory
  domain, consumed by the generator backend at prepare/initialize time, with no
  dispatch-time allocation introduced.

- [x] **KVM-02**: The maintained generate path dispatches `allocate_slots` into
  `memory::hybrid` so the block map tracks real token growth for prefill and decode,
  with block-exhaustion and invalid-request outcomes modeled as explicit `_error`
  events and guard-selected transitions (no silent clamping, no helper-selected
  fallback).

### Snapshot addressing contract

- [x] **KVM-03**: The per-step captured `memory::view::snapshot` is the single
  addressing truth for the compute path: guards validate snapshot/sequence coherence
  before compute phases, stale or incoherent snapshots route to explicit error
  transitions, and no parallel token accounting is mirrored into machine context.

### Physical addressing cutover

- [x] **KVW-01**: The K/V write path (`store_attention_kv_cache`, both linear and
  flash layouts) stores at snapshot-mapped physical block positions, bit-exact with
  the pre-cutover layout for the maintained contiguous single-sequence workload.

- [x] **KVR-01**: The non-flash attention read path walks snapshot-mapped physical
  spans with bounded, monotonic, allocation-free data-plane iteration, and maintained
  generation parity is unchanged.

- [ ] **KVR-02**: The flash attention route is guarded by an explicit
  contiguous-mapped-span predicate: contiguous mappings execute the existing optimized
  flash kernels bit-identically; non-contiguous mappings route via explicit
  transitions to the span-walking path (behavior selection in guards/transitions
  only, never in action/detail helpers).

### Recurrent state ownership

- [ ] **KVS-01**: Recurrent (shortconv) state is addressed through the snapshot's
  `sequence_recurrent_slot` mapping instead of an implicit flat layer-only offset,
  bit-exact for the maintained single-slot workload.

### Proof, performance, and publication

- [ ] **KVP-01**: Component-level tests prove block-map addressing isolation with at
  least two interleaved sequences (allocate, grow across block boundaries, free,
  reallocate) through the owning machines' `process_event` interfaces and the
  addressing helpers, including reclaimed-block reuse without cross-sequence bleed.

- [ ] **KVE-01**: Maintained generation benchmarks (LFM2 10/100/1000-token baselines
  and the kernel suites) show no regression beyond accepted thresholds after the
  cutover, dispatch remains allocation-free, and benchmark evidence is captured from
  maintained commands without snapshot updates unless explicitly approved.

- [ ] **KVD-01**: Architecture docs and mermaid diagrams reflect memory-owned KV
  addressing, domain-boundary checks pass, and the full-scope quality gate passes at
  milestone closeout.

## Out of Scope

| Item | Reason |
|------|--------|
| Public multi-sequence generation API | Next milestone candidate; this milestone proves the block map at component level only |
| KV eviction, prefix reuse, paging to disk | Requires multi-sequence serving surface first |
| Removing the dual linear+flash K/V storage (2x KV memory) | Pre-existing cost; changing it alters the performance contract and deserves its own decision |
| Quantized KV cache formats | Orthogonal to addressing ownership |
| Whisper decoder cross-KV and sortformer caches | Fixed-size domain-owned caches, intentionally outside the memory domain |

## Requirement-to-Phase Map

| Requirement | Phase |
|-------------|-------|
| KVM-01 | Phase 245 |
| KVM-02 | Phase 245 |
| KVM-03 | Phase 246 |
| KVW-01 | Phase 247 |
| KVR-01 | Phase 247 |
| KVR-02 | Phase 248 |
| KVS-01 | Phase 249 |
| KVP-01 | Phase 250 |
| KVE-01 | Phase 250 |
| KVD-01 | Phase 250 |

Mapped: 10/10 v1.28 requirements; satisfied 5, pending 5.
