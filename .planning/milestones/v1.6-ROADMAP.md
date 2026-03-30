# Roadmap

## Archived Milestones

- [x] [v1.0: EMEL Llama-68M Generation Slice](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.0-ROADMAP.md) - shipped 2026-03-08 with 7 phases and 15 plans; proved one canonical Llama-68M generation parity slice in `tools/paritychecker/`.
- [x] [v1.1: EMEL Llama-68M Generation Benchmark](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.1-ROADMAP.md) - shipped 2026-03-11 with 4 phases and 10 plans; added one truthful canonical Llama-68M generation benchmark in `tools/bench`, native EMEL decode benchmarking, compare output, and snapshot/docs integration.
- [x] [v1.2: Flash Attention](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.2-ROADMAP.md) - shipped 2026-03-22 with 5 phases and 13 plans; added an EMEL-owned flash-attention path to the canonical Llama-68M slice, hard-cut runtime tensor lifecycle through `emel::tensor::sm`, and published maintained benchmark evidence over a preserved pre-flash baseline.
- [x] [v1.3: ARM Flash Optimizations](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.3-ROADMAP.md) - shipped 2026-03-22 with 3 phases and 7 plans; delivered optimized AArch64 flash execution, maintained runtime/parity attribution, and preserved-baseline benchmark publication for the canonical ARM Llama-68M slice.
- [x] [v1.4: Full Vectorized Quantized Kernels](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.4-ROADMAP.md) - shipped 2026-03-25 with 5 phases and 11 plans; delivered EMEL-owned vectorized q2/q3/q6 kernels, full maintained `1/10/100/1000` parity proof, and quantized benchmark attribution on the canonical ARM slice.
- [x] [v1.5: Full ARM Quantized Path](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/.planning/milestones/v1.5-ROADMAP.md) - shipped 2026-03-27 with 5 phases and 10 plans; closed the maintained ARM quantized-path contract and restored canonical flash publication.

## Current Milestone

### v1.6 Qwen3-0.6B Parity And Benchmark

**Milestone Goal:** Prove one truthful canonical Qwen3-0.6B GGUF slice through the maintained
EMEL generator, paritychecker, and benchmark workflow without widening the acceptance boundary
beyond that slice.

**Scope Guardrails:**
- Keep the milestone narrow to one official `Qwen3-0.6B-Q8_0.gguf` fixture.
- Keep `tools/paritychecker` and `tools/bench` as the only maintained acceptance surfaces.
- Preserve the existing generator -> graph -> processor -> kernel actor chain and current public
  API boundaries unless explicitly approved otherwise.
- Do not claim broader Qwen-family, multi-quant, or benchmark-only support from this milestone.

## Phases

- [x] **Phase 26: Canonical Qwen3 Fixture And Conditioning Contract** - Lock the official Qwen3
  truth anchor and define one explicit request-conditioning contract for the maintained slice.
- [x] **Phase 26.1: Native q8_0 Projection And Output Runtime Support For Canonical Qwen3** (INSERTED) -
  Add the missing native `q8_0` hot-path support required before the maintained Qwen3 generator
  path can initialize truthfully.
- [x] **Phase 27: Qwen3 Runtime Architecture Bring-Up** - Bring one canonical `qwen3` slice up
  through the shipped EMEL generator path without pretending broad family support.
- [x] **Phase 28: Qwen3 Parity And Regression** - Prove the canonical Qwen3 slice on the
  maintained parity surface and add regression coverage that protects both Qwen and the prior
  Llama anchor.
- [x] **Phase 29: Qwen3 Benchmark Publication** - Publish one truthful maintained compare/docs
  path for the same parity-backed canonical Qwen3 slice.

## Phase Details

### Phase 26: Canonical Qwen3 Fixture And Conditioning Contract
**Goal**: Lock the official Qwen3 truth anchor and one explicit request-conditioning contract so
the maintained surfaces refer to the same operator-facing slice and fail truthfully before runtime
bring-up exists.
**Depends on**: Phase 25.1
**Requirements**: FIX-01, COND-01
**Success Criteria** (what must be TRUE):
  1. `tests/models/README.md` and maintained tool constants identify one official
     `Qwen3-0.6B-Q8_0.gguf` fixture with reproducible provenance.
  2. One canonical request-conditioning contract is documented for parity and benchmark use, and
     it is explicit that the maintained slice derives formatting from the primary GGUF
     `tokenizer.chat_template`, uses a structured-message formatter request shape, and exposes the
     resolved formatter contract on maintained setup/failure surfaces before downstream proof
     phases.
  3. Fixture or conditioning drift fails clearly instead of silently reusing the old Llama anchor,
     and pre-runtime Qwen setup failures are explicit until Phase 27 brings the runtime up.
**Plans**: 4 plans

Plans:
- [x] 26-01: Add official Qwen3 fixture provenance and maintained tool anchors.
- [x] 26-02: Widen the formatter and conditioner contract to structured chat messages.
- [x] 26-03: Carry the structured-message contract through the generator boundary.
- [x] 26-04: Bind and publish the primary-template formatter contract on parity and bench surfaces.

### Phase 26.1: Native q8_0 projection and output runtime support for canonical Qwen3 (INSERTED)
**Goal**: Add the missing native `q8_0` runtime/kernel support required for the canonical Qwen3
generator hot path to initialize truthfully in `src/emel`.
**Requirements**: RUN-00
**Depends on:** Phase 26
**Plans:** 2 plans

**Success Criteria** (what must be TRUE):
  1. The shipped EMEL runtime can consume the canonical Qwen3 slice's `q8_0` projection/output
     tensors natively in `src/emel` without tool-only or whole-tensor dequantize fallback.
  2. The required support stays narrow to the hot-path operations needed to unblock canonical Qwen3
     generation and does not over-claim broader `q8_0` model support.
  3. Phase 27 can resume from the blocked `27-02` runtime/parity bring-up against the new native
     support instead of failing at backend initialization.

Plans:
- [x] 26.1-01: Add explicit native `q8_0` kernel support and generator row access for the canonical
  blocker path.
- [x] 26.1-02: Prove the canonical Qwen3 bring-up path advances beyond the old backend failure
  using the new native `q8_0` support.

### Phase 27: Qwen3 Runtime Architecture Bring-Up
**Goal**: Bring one truthful canonical `qwen3` slice up through the shipped EMEL generator path.
**Depends on**: Phase 26.1
**Requirements**: RUN-01, RUN-02
**Success Criteria** (what must be TRUE):
  1. EMEL can initialize and generate on the canonical Qwen3-0.6B fixture through the maintained
     generator path in `src/emel`.
  2. Runtime support explicitly handles the canonical slice's required `qwen3` topology instead of
     only renaming the architecture.
  3. The runtime work stays narrow to the canonical Qwen3-0.6B fixture and does not over-claim
     broader Qwen support.
**Plans**: 2 plans

Plans:
- [x] 27-01: Extend model and execution-view support for the canonical Qwen3 slice.
- [x] 27-02: Bring the shipped generator path up on the canonical Qwen3 slice without broad family
  claims.

### Phase 28: Qwen3 Parity And Regression
**Goal**: Prove the canonical Qwen3 slice on maintained parity and regression surfaces.
**Depends on**: Phase 27
**Requirements**: PAR-01, VER-01
**Success Criteria** (what must be TRUE):
  1. `tools/paritychecker --generation` proves EMEL against `llama.cpp` on the canonical Qwen3
     slice using the same fixture and conditioning contract.
  2. Regression tests cover the canonical Qwen3 runtime and conditioning assumptions and fail on
     drift.
  3. The prior maintained Llama slice remains protected while support is widened.
**Plans**: 2 plans

Plans:
- [x] 28-01: Extend paritychecker generation and reference alignment for the canonical Qwen3
  slice.
- [x] 28-02: Add regression coverage for the Qwen slice and prior Llama anchor protection.

### Phase 29: Qwen3 Benchmark Publication
**Goal**: Publish one truthful maintained compare/docs path for the same parity-backed canonical
Qwen3 slice.
**Depends on**: Phase 28
**Requirements**: BENCH-01
**Success Criteria** (what must be TRUE):
  1. `tools/bench` compare output uses the same canonical Qwen3 fixture and conditioning contract
     that parity already proved.
  2. Stored benchmark evidence and generated docs publish one truthful Qwen3 row without
     overstating broader Qwen-family support.
  3. Benchmark publication reuses the existing compare/docs workflow instead of inventing a Qwen-
     only harness.
**Plans**: 2 plans

Plans:
- [x] 29-01: Add maintained compare publication for the canonical Qwen3 slice.
- [x] 29-02: Refresh stored benchmark/docs evidence for the same slice after parity-backed
  verification.

## Progress

**Execution Order:**
Phases execute in numeric order: 26 -> 26.1 -> 27 -> 28 -> 29

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 26. Canonical Qwen3 Fixture And Conditioning Contract | 4/4 | Complete | 2026-03-28 |
| 26.1. Native q8_0 Projection And Output Runtime Support For Canonical Qwen3 | 2/2 | Complete | 2026-03-28 |
| 27. Qwen3 Runtime Architecture Bring-Up | 2/2 | Complete | 2026-03-28 |
| 28. Qwen3 Parity And Regression | 2/2 | Complete | 2026-03-28 |
| 29. Qwen3 Benchmark Publication | 2/2 | Complete | 2026-03-28 |
