# Roadmap: EMEL

## Overview

v1.9 delivers one truthful maintained LiquidAI `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` ARM slice
through the existing generator, paritychecker, and benchmark workflow. The milestone stays narrow:
one official fixture, one explicit Liquid conditioning contract, one explicit `lfm2` runtime
path, then maintained parity and benchmark publication for that exact slice.

## Phases

**Phase Numbering:**
- Integer phases continue across milestones.
- v1.9 starts at Phase 33 because v1.7 ended at Phase 32.

- [ ] **Phase 33: Fixture, Metadata, And Contract Lock** - Lock the exact Liquid fixture,
      executable metadata truth, and maintained conditioning contract.
- [ ] **Phase 34: `lfm2` Model Contract Bring-Up** - Make EMEL truthfully recognize the canonical
      Liquid slice as `lfm2` with an explicit maintained model contract.
- [ ] **Phase 35: Maintained Runtime Execution On ARM** - Bring the shipped generator path up on
      the canonical Liquid slice with a truthful `Q4_K_M` runtime contract.
- [ ] **Phase 36: Parity And Regression Proof** - Prove the maintained Liquid slice against
      `llama.cpp` and protect the prior maintained anchors.
- [ ] **Phase 37: Benchmark And Docs Publication** - Publish one parity-backed Liquid benchmark
      path and maintained docs for the same slice.

## Phase Details

### Phase 33: Fixture, Metadata, And Contract Lock
**Goal**: The repo names one exact maintained Liquid fixture and one exact maintained request
contract before runtime work starts.
**Depends on**: Shipped Phase 32
**Requirements**: FIX-02, META-01, COND-03
**Success Criteria** (what must be TRUE):
  1. The repo documents exactly one official maintained fixture at
     `tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf` with checksum, source, stable path, and
     download URL.
  2. Maintained repo evidence records executable model truth for the slice from official
     GGUF/config metadata, including `architecture=lfm2`, instead of stale prose-only metadata.
  3. The maintained Liquid request path exposes one explicit structured chat-message contract with
     `tools=none`, `add_generation_prompt=true`, and no implicit raw fallback.
**Plans**: TBD

### Phase 34: `lfm2` Model Contract Bring-Up
**Goal**: EMEL-owned model-loading surfaces truthfully accept the canonical Liquid fixture as
`lfm2` and expose its maintained topology contract.
**Depends on**: Phase 33
**Requirements**: RUN-03, RUN-05
**Success Criteria** (what must be TRUE):
  1. EMEL-owned model metadata/loading surfaces identify the maintained fixture as `lfm2` instead
     of rejecting it or aliasing it to `llama` or `qwen3`.
  2. The canonical Liquid slice's required metadata, tensor naming, and hybrid block contract are
     represented explicitly in `src/emel` for the maintained fixture.
  3. Repo-visible model contract evidence makes the maintained Liquid path auditable as a
     dedicated `lfm2` architecture slice rather than a generic Llama-family claim.
**Plans**: TBD

### Phase 35: Maintained Runtime Execution On ARM
**Goal**: The shipped generator path can initialize and generate on the canonical Liquid slice on
ARM without widening beyond the maintained `Q4_K_M` truth anchor.
**Depends on**: Phase 34
**Requirements**: RUN-04, RUN-06
**Success Criteria** (what must be TRUE):
  1. The maintained generator path initializes the official
     `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture and produces bounded generation on ARM.
  2. Runtime execution for the maintained Liquid slice uses the explicit `lfm2` path and the
     maintained `Q4_K_M` surface rather than broad sibling-quant or generic Liquid claims.
  3. Maintained runtime evidence publishes a truthful quantized-path contract for the official
     `Q4_K_M` fixture only.
**Plans**: TBD

### Phase 36: Parity And Regression Proof
**Goal**: The exact maintained Liquid slice is proven correct against the reference and does not
break the existing maintained anchors.
**Depends on**: Phase 35
**Requirements**: PAR-02, VER-02
**Success Criteria** (what must be TRUE):
  1. `tools/paritychecker --generation` proves EMEL against `llama.cpp` on the canonical Liquid
     fixture using the same maintained conditioning contract from Phase 33.
  2. Stored regression evidence covers the Liquid slice and the prior maintained Llama and Qwen
     anchors.
  3. Maintained parity evidence makes the exact fixture and contract auditable for reviewers.
**Plans**: TBD

### Phase 37: Benchmark And Docs Publication
**Goal**: The repo publishes one benchmark/docs path for the same parity-backed maintained Liquid
slice and nothing broader.
**Depends on**: Phase 36
**Requirements**: BENCH-08
**Success Criteria** (what must be TRUE):
  1. `tools/bench` compare output includes one maintained Liquid benchmark case for the official
     `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` slice.
  2. Stored benchmark evidence and generated docs identify the exact fixture and maintained
     conditioning contract used for the Liquid slice.
  3. Published Liquid benchmark evidence stays aligned with the parity-backed maintained slice and
     does not broaden into generic Liquid or sibling-quant claims.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 33. Fixture, Metadata, And Contract Lock | 0/TBD | Not started | - |
| 34. `lfm2` Model Contract Bring-Up | 0/TBD | Not started | - |
| 35. Maintained Runtime Execution On ARM | 0/TBD | Not started | - |
| 36. Parity And Regression Proof | 0/TBD | Not started | - |
| 37. Benchmark And Docs Publication | 0/TBD | Not started | - |
