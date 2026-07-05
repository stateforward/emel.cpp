# Roadmap: EMEL

## Milestones

- [x] **v1.0 EMEL Llama-68M Generation Slice** - shipped 2026-03-08
- [x] **v1.1 EMEL Llama-68M Generation Benchmark** - shipped 2026-03-11
- [x] **v1.2 Flash Attention** - shipped 2026-03-22
- [x] **v1.3 ARM Flash Optimizations** - shipped 2026-03-22
- [x] **v1.4 Full Vectorized Quantized Kernels** - shipped 2026-03-25
- [x] **v1.5 Full ARM Quantized Path** - shipped 2026-03-27
- [x] **v1.6 Qwen3-0.6B Parity And Benchmark** - shipped 2026-03-30
- [x] **v1.7 Generator Prefill Submachine Decomposition** - shipped 2026-03-30
- [x] **v1.8 Truthful Qwen3 E2E Embedded Size** - shipped 2026-04-02
- [x] **v1.9 Liquid LFM2.5-1.2B Thinking ARM Slice** - shipped 2026-04-02
- [x] **v1.11 TE-75M GGUF Trimodal Embedding Runtime** - shipped 2026-04-15
- [x] **v1.12 Pluggable Reference Parity Bench Architecture** - shipped 2026-04-18
- [x] **v1.13 Pluggable Generative Parity Bench** - shipped 2026-04-21
- [x] **v1.14 Benchmark Variant Organization** - shipped 2026-04-21
- [x] **v1.15 ARM Sortformer Diarization GGUF Slice** - shipped 2026-04-25
- [x] **v1.16 ARM Whisper GGUF Parity And Performance** - shipped 2026-04-28
- [x] **v1.17 Text Generator Domain Alignment** - shipped 2026-04-30
- [x] **v1.18 Parity Tool Boundary Refactor** - shipped 2026-05-01
- [x] **v1.19 Benchmark Tool Pluggable Runner Refactor** - shipped 2026-05-01
- [x] **v1.20 SML Dependency And Namespace Migration** - shipped 2026-05-02
- [x] **v1.21 Quality Gate Selective Runner Optimization** - shipped 2026-05-02
- [x] **v1.22 Weight Loading Ownership Cutover** - shipped 2026-05-03
- [x] **v1.23 I/O Loading Strategy Boundary** - shipped 2026-05-04
- [x] **v1.24 I/O Mmap Loading Strategy** - shipped 2026-05-04
- [x] **v1.25 I/O Read Loading Strategy** - shipped 2026-05-06
- [x] **v1.26 I/O Staged Read Loading Strategy** - completed 2026-05-08
- [x] **v1.27 Ryzen AVX2/FMA Kernel Support** - shipped 2026-06-25
- [ ] **v1.28 Memory-Owned KV Block Addressing Cutover** - planned 2026-07-04

## Current Milestone

### v1.28 Memory-Owned KV Block Addressing Cutover

**Milestone Goal:** Make the memory domain the landlord of the KV cache, not just its
accountant. Physical K/V storage addressing in the maintained generation path derives
from the `memory::hybrid` block map (`memory::view::snapshot`), replacing the
generator-local linear position accounting, with bit-exact parity on the maintained
single-sequence workload and component-level multi-sequence proof.

**Source-backed gap (2026-07-04 audit):** the generator dispatches real
`allocate_sequence`/`free_sequence`/`capture_view` events, but K/V bytes live in
generator-backend vectors addressed by `backend.kv_cache_tokens = position + 1`; the
captured snapshot is threaded into graph/processor events and never dereferenced in
`src/`; `sequence_recurrent_slot` is unused.

**Scope Guardrails:**
- Keep the public generation API single-sequence; multi-sequence serving is a later
  milestone. Component-level multi-sequence proof only.
- Preserve the generator -> graph -> processor -> kernel chain and current
  Stateforward.SML orchestration structure.
- Behavior selection (flash vs span-walk route, error routing) lives in guards and
  `sm.hpp` transitions only — never in `actions.hpp`/`detail.hpp` helper returns.
- Addressing math (logical position -> physical block offset) is data-plane index
  computation and belongs in detail kernels; route choice does not.
- No dispatch-time allocation; block-map structures are fixed-capacity arrays sized at
  construction/initialize time.
- Bit-exact parity required for the maintained contiguous single-sequence workload;
  no benchmark regression beyond accepted thresholds; no snapshot updates without
  explicit consent.
- Do not remove or restructure the dual linear+flash K/V storage in this milestone.

Execution order: 245, 246, 247, 248, 249, 250.

**Milestone progress (v1.28):** **5 / 6** phases complete.

- [x] Phase 245: Block Geometry Ownership and Slot Allocation (KVM-01, KVM-02)
- [x] Phase 246: Snapshot Addressing Contract and Coherence Guards (KVM-03)
- [x] Phase 247: Write and Scalar Read Block-Mapped Cutover (KVW-01, KVR-01)
- [x] Phase 248: Flash Route Contiguity Guards (KVR-02)
- [x] Phase 249: Recurrent Slot Wiring (KVS-01)
- [ ] Phase 250: Multi-Sequence Component Proof and Evidence (KVP-01, KVE-01, KVD-01)

## Phase Details

### Phase 245: Block Geometry Ownership and Slot Allocation

**Goal:** One shared block-geometry contract owned by the memory domain sizes the
physical cache, and the generate path keeps the block map tracking real token growth.

**Depends on:** Phase 244
**Requirements:** KVM-01, KVM-02

**Success Criteria:**

1. Block size (tokens) and per-layer block capacity come from a memory-domain contract
   consumed by the generator backend at prepare time; cache vectors are sized from it.
2. Prefill dispatches `allocate_slots` for the prompt length; decode dispatches slot
   growth at block boundaries; both through `memory::hybrid::process_event` only.
3. Block exhaustion and invalid requests surface as explicit `_error` events routed by
   guards to error states — no clamping in helpers, no context error flags.
4. No dispatch-time allocation; machine-assertion tests cover accepted and rejected
   slot-growth outcomes through the public machine surface.

### Phase 246: Snapshot Addressing Contract and Coherence Guards

**Goal:** The per-step snapshot becomes the validated, single source of addressing
truth for compute phases.

**Depends on:** Phase 245
**Requirements:** KVM-03

**Success Criteria:**

1. Guards validate snapshot/sequence coherence (active sequence, expected length)
   before compute phases; incoherent snapshots route to explicit error transitions.
2. The compute path receives the snapshot through existing typed events; no parallel
   token counters are mirrored into machine context.
3. Machine-assertion tests drive coherent and incoherent snapshot outcomes through
   `process_event`.

### Phase 247: Write and Scalar Read Block-Mapped Cutover

**Goal:** K/V writes and the non-flash attention read path address physical storage
through the snapshot block map.

**Depends on:** Phase 246
**Requirements:** KVW-01, KVR-01

**Success Criteria:**

1. `store_attention_kv_cache` writes both layouts at snapshot-mapped physical
   positions via a data-plane mapping helper.
2. The scalar attention read path walks snapshot-mapped spans with bounded, monotonic,
   allocation-free iteration.
3. Maintained single-sequence generation output is bit-exact with pre-cutover
   behavior; focused tests cover block-boundary crossings.

### Phase 248: Flash Route Contiguity Guards

**Goal:** Flash attention executes only under an explicit contiguous-mapped-span
guard; non-contiguous mappings route to the span-walking path via transitions.

**Depends on:** Phase 247
**Requirements:** KVR-02

**Success Criteria:**

1. A pure guard predicate decides flash eligibility from (event, snapshot) contiguity;
   the transition table routes the alternative explicitly.
2. Contiguous mappings produce bit-identical flash kernel views and results.
3. No helper-return route selection; machine-assertion tests cover both routes.

### Phase 249: Recurrent Slot Wiring

**Goal:** Shortconv recurrent state is addressed through the snapshot's recurrent-slot
mapping.

**Depends on:** Phase 248
**Requirements:** KVS-01

**Success Criteria:**

1. Shortconv state offsets include the snapshot-resolved slot; single-slot workload is
   bit-exact with the flat layout.
2. Slot lookup failures route through explicit error transitions.

### Phase 250: Multi-Sequence Component Proof and Evidence

**Goal:** Prove block-map isolation with interleaved sequences at component level and
close the milestone with truthful evidence.

**Depends on:** Phase 249
**Requirements:** KVP-01, KVE-01, KVD-01

**Success Criteria:**

1. Tests interleave >= 2 sequences (allocate, grow across boundaries, free,
   reallocate) through owning machines' `process_event`, asserting reclaimed-block
   reuse without cross-sequence bleed at the addressing-helper level.
2. Maintained generation benchmarks (LFM2 10/100/1000 and kernel suites) show no
   regression beyond accepted thresholds; dispatch remains allocation-free.
3. Architecture docs/mermaid updated; domain-boundary checks and the full-scope
   quality gate pass.

## Progress

**Execution Order:** 245 -> 246 -> 247 -> 248 -> 249 -> 250

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 245. Block Geometry Ownership and Slot Allocation | v1.28 | 1/1 | Complete | 2026-07-04 |
| 246. Snapshot Addressing Contract and Coherence Guards | v1.28 | 1/1 | Complete | 2026-07-04 |
| 247. Write and Scalar Read Block-Mapped Cutover | v1.28 | 1/1 | Complete | 2026-07-04 |
| 248. Flash Route Contiguity Guards | v1.28 | 1/1 | Complete | 2026-07-04 |
| 249. Recurrent Slot Wiring | v1.28 | 1/1 | Complete | 2026-07-04 |
| 250. Multi-Sequence Component Proof and Evidence | v1.28 | 0/1 | Planned | — |

## Recently Shipped

### v1.27 Ryzen AVX2/FMA Kernel Support

**Shipped:** 2026-06-25
**Archive:** `.planning/milestones/v1.27-ROADMAP.md`
**Requirements:** `.planning/milestones/v1.27-REQUIREMENTS.md`
**Audit:** `.planning/milestones/v1.27-MILESTONE-AUDIT.md`

Delivered native x86_64 AVX2/FMA support for the AMD Ryzen 9 5950X maintained
runtime slice: host feature contract, optimized flash attention, q2_K/q3_K/q6_K
x q8_K kernels, maintained generator parity attribution, and truthful
`kernel_x86_64` benchmark publication. The source-backed audit passed after
repairing the optimized benchmark attribution gap and removing the x86_64 unary
SML rule debt.

Next step: run `$gsd-new-milestone` to define the next milestone.
