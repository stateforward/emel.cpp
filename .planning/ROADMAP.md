# Roadmap: EMEL

## Overview

`v1.11` delivers one truthful maintained text-generation slice for the brand new official
`ggml-org/gemma-4-E2B-it-GGUF` release through the existing generator, paritychecker, and
benchmark workflow. The milestone stays narrow: one official text fixture, one explicit Gemma 4
text-only conditioning contract, one explicit `gemma4` runtime path, one explicit reference-lane
readiness step, then maintained parity and benchmark publication for that exact slice.

This milestone explicitly does **not** include `mmproj`, image/audio/video inputs, tool-use
surfaces, or the `F16` sibling file.

## Phases

**Phase Numbering:**
- Integer phases continue across milestones.
- `v1.11` starts at Phase 38 because the current visible roadmap ceiling in `.planning/phases/` is
  Phase 37.

- [ ] **Phase 38: Fixture, Metadata, And Text-Only Contract Lock** - Lock the exact Gemma 4
      fixture, executable metadata truth, and maintained text-only conditioning contract.
- [ ] **Phase 39: `gemma4` Model Contract Bring-Up** - Make EMEL truthfully recognize the
      canonical Gemma 4 fixture as `gemma4` with an explicit text-only model contract.
- [ ] **Phase 40: Maintained Text Runtime Execution On ARM** - Bring the shipped generator path up
      on the canonical Gemma 4 text slice on ARM.
- [ ] **Phase 41: Reference, Parity, And Regression Proof** - Make the reference lane Gemma
      4-capable, then prove the maintained slice against `llama.cpp` and protect the prior
      maintained anchors.
- [ ] **Phase 42: Benchmark And Docs Publication** - Publish one parity-backed Gemma 4 benchmark
      path and maintained docs for the same text slice.

## Phase Details

### Phase 38: Fixture, Metadata, And Text-Only Contract Lock
**Goal**: The repo names one exact maintained Gemma 4 text fixture and one exact text-only request
contract before runtime work starts.
**Depends on**: Phase 37
**Requirements**: FIX-03, META-02, COND-04
**Success Criteria** (what must be TRUE):
  1. The repo documents exactly one official maintained fixture at
     `tests/models/gemma-4-e2b-it-Q8_0.gguf` with checksum, source, stable path, and download URL.
  2. Maintained repo evidence records executable model truth for the slice from official
     GGUF/config metadata, including `architecture=gemma4`, `context_length=131072`, the
     text-layer schedule, and the separate `mmproj` companion file.
  3. The maintained request path exposes one explicit structured text chat contract with
     `add_generation_prompt=true` and no implicit raw fallback.
**Plans**: TBD

### Phase 39: `gemma4` Model Contract Bring-Up
**Goal**: EMEL-owned model-loading surfaces truthfully accept the canonical Gemma 4 fixture as
`gemma4` and expose its maintained text-only topology contract.
**Depends on**: Phase 38
**Requirements**: COND-05, RUN-07, RUN-09
**Success Criteria** (what must be TRUE):
  1. EMEL-owned model metadata/loading surfaces identify the maintained fixture as `gemma4`
     instead of rejecting it or aliasing it to `llama`, `qwen3`, or `lfm2`.
  2. The canonical Gemma 4 text slice's required metadata and topology, including alternating
     sliding/full attention, shared-KV layers, rope handling, and tied embeddings, are represented
     explicitly in `src/emel` for the maintained fixture.
  3. Repo-visible contract evidence makes the maintained Gemma 4 path auditable as text-only and
     rejects `mmproj`, media, and tool-call request shapes explicitly.
**Plans**: TBD

### Phase 40: Maintained Text Runtime Execution On ARM
**Goal**: The shipped generator path can initialize and generate on the canonical Gemma 4 text
slice on ARM without widening beyond the maintained `Q8_0` truth anchor.
**Depends on**: Phase 39
**Requirements**: RUN-08
**Success Criteria** (what must be TRUE):
  1. The maintained generator path initializes the official `gemma-4-e2b-it-Q8_0.gguf` fixture and
     produces bounded text generation on ARM.
  2. Runtime execution for the maintained Gemma 4 slice uses the explicit `gemma4` path and the
     maintained text-only acceptance surface rather than broad multimodal or family claims.
  3. Maintained runtime evidence publishes a truthful quantized-path contract for the official
     `Q8_0` fixture only.
**Plans**: TBD

### Phase 41: Reference, Parity, And Regression Proof
**Goal**: The reference lane is Gemma 4-capable, the exact maintained text slice is proven
correct against the reference, and the existing maintained anchors stay green.
**Depends on**: Phase 40
**Requirements**: REF-01, PAR-03, VER-03
**Success Criteria** (what must be TRUE):
  1. The pinned `llama.cpp` reference lane used by `tools/paritychecker` and `tools/bench` is
     upgraded or confirmed so the canonical Gemma 4 fixture can be loaded for maintained
     comparison.
  2. `tools/paritychecker --generation` proves EMEL against `llama.cpp` on the canonical Gemma 4
     fixture using the same maintained text-only conditioning contract from Phase 38.
  3. Stored regression evidence covers the Gemma 4 slice and the prior maintained Llama, Qwen, and
     Liquid anchors.
**Plans**: TBD

### Phase 42: Benchmark And Docs Publication
**Goal**: The repo publishes one benchmark/docs path for the same parity-backed maintained Gemma 4
text slice and nothing broader.
**Depends on**: Phase 41
**Requirements**: BENCH-09
**Success Criteria** (what must be TRUE):
  1. `tools/bench` compare output includes one maintained Gemma 4 benchmark case for the official
     `gemma-4-e2b-it-Q8_0.gguf` text slice.
  2. Stored benchmark evidence and generated docs identify the exact fixture and maintained
     text-only conditioning contract used for the Gemma 4 slice.
  3. Published Gemma 4 benchmark evidence stays aligned with the parity-backed maintained slice and
     does not broaden into `mmproj`, multimodal, or `F16` claims.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 38. Fixture, Metadata, And Text-Only Contract Lock | 0/TBD | Not started | - |
| 39. `gemma4` Model Contract Bring-Up | 0/TBD | Not started | - |
| 40. Maintained Text Runtime Execution On ARM | 0/TBD | Not started | - |
| 41. Reference, Parity, And Regression Proof | 0/TBD | Not started | - |
| 42. Benchmark And Docs Publication | 0/TBD | Not started | - |
