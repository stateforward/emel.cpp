# Roadmap: EMEL

## Overview

v1.10 delivers one truthful maintained Prism ML `Bonsai-1.7B.gguf` slice through the existing EMEL
generator, paritychecker, and benchmark workflow. The milestone stays narrow and honest to the
research: freeze one maintained fixture, bind one explicit Bonsai conditioning contract, land the
native `Q1_0_g128` runtime path on the existing `qwen3` lane, then prove parity/regression and
publish one benchmark/docs path for that exact slice.

## Phases

**Phase Numbering:**
- Integer phases continue across milestones.
- v1.10 starts at Phase 38 because the prior in-flight roadmap ended at Phase 37.

- [ ] **Phase 38: Fixture Provenance And Metadata Truth** - Freeze the maintained Bonsai fixture,
      executable GGUF truth, and exact file-versus-format naming.
- [ ] **Phase 39: Bonsai Conditioning Contract** - Bind one explicit Bonsai formatter contract and
      reject unsupported request shapes.
- [ ] **Phase 40: Native `Q1_0_g128` Runtime Bring-Up** - Add truthful EMEL-owned runtime support
      for the maintained Bonsai slice on the existing `qwen3` path.
- [ ] **Phase 41: Parity And Regression Proof** - Prove the maintained Bonsai slice against the
      Prism reference lane and protect prior maintained anchors.
- [ ] **Phase 42: Benchmark And Publication** - Publish one truthful Bonsai benchmark/docs path
      for the same parity-backed maintained slice.

## Phase Details

### Phase 38: Fixture Provenance And Metadata Truth
**Goal**: The repo freezes one maintained Bonsai artifact identity and one executable truth set
before contract or runtime work starts.
**Depends on**: Nothing (new milestone starts at continued numbering)
**Requirements**: FIX-03, META-02, META-03
**Success Criteria** (what must be TRUE):
  1. Repo-visible fixture docs and registries name exactly one maintained file at
     `tests/models/Bonsai-1.7B.gguf` with checksum, size, source repo, and direct download URL.
  2. Maintained model-truth evidence records the Bonsai slice as `general.architecture=qwen3`
     with the required GGUF metadata values from the executable artifact, not stale prose.
  3. Operator-facing docs and publication surfaces distinguish the fixture filename
     `Bonsai-1.7B.gguf` from the weight format `Q1_0_g128` and do not rely on the stale quickstart
     filename.
**Plans**: TBD

### Phase 39: Bonsai Conditioning Contract
**Goal**: The maintained Bonsai slice exposes one explicit structured request contract before
runtime and parity claims are made.
**Depends on**: Phase 38
**Requirements**: COND-04, COND-05, COND-06
**Success Criteria** (what must be TRUE):
  1. The maintained Bonsai request path formats chat input from the embedded
     `tokenizer.chat_template` using one explicit structured message contract.
  2. Supported Bonsai requests are limited to `system`, `user`, and `assistant` messages with
     `add_generation_prompt=true`, `tools=none`, and `enable_thinking=false`.
  3. Unsupported Bonsai request shapes, including tool calls, tool responses, thinking replay,
     named template variants, and raw-prompt fallback, fail explicitly on repo-visible surfaces.
**Plans**: TBD

### Phase 40: Native `Q1_0_g128` Runtime Bring-Up
**Goal**: EMEL truthfully runs the maintained Bonsai slice on the shipped generator path using a
native `Q1_0_g128` operand path on the existing `qwen3` architecture lane.
**Depends on**: Phase 39
**Requirements**: RUN-07, RUN-08, RUN-09
**Success Criteria** (what must be TRUE):
  1. EMEL-owned model loading and metadata surfaces accept `Bonsai-1.7B.gguf` as a `qwen3`
     fixture instead of inventing a new `bonsai` or `prismml` execution family.
  2. `src/emel` provides a native `Q1_0_g128` dtype/layout and hot-path execution path for the
     maintained Bonsai tensors without a whole-tensor dequantize-to-f32 fallback.
  3. The shipped generator path initializes and produces bounded generation on
     `tests/models/Bonsai-1.7B.gguf` using the maintained Bonsai formatter contract and native
     `Q1_0_g128` runtime path.
**Plans**: TBD

### Phase 41: Parity And Regression Proof
**Goal**: Correctness claims for the maintained Bonsai slice are proven against the truthful
reference lane and do not break the prior maintained anchors.
**Depends on**: Phase 40
**Requirements**: PAR-03, VER-03
**Success Criteria** (what must be TRUE):
  1. `tools/paritychecker --generation` proves EMEL against a pinned `PrismML-Eng/llama.cpp`
     reference lane on `Bonsai-1.7B.gguf` using the same maintained formatter contract.
  2. Regression coverage protects the maintained Bonsai slice and the prior maintained Llama and
     Qwen anchors from fixture, formatter, architecture, or quant-path drift.
  3. Repo-visible parity evidence makes the exact fixture, formatter contract, and Prism reference
     identity auditable for reviewers.
**Plans**: TBD

### Phase 42: Benchmark And Publication
**Goal**: The repo publishes one truthful benchmark/docs path for the same parity-backed maintained
Bonsai slice and nothing broader.
**Depends on**: Phase 41
**Requirements**: BENCH-09
**Success Criteria** (what must be TRUE):
  1. `tools/bench` compare output includes one maintained Bonsai benchmark case for
     `tests/models/Bonsai-1.7B.gguf` aligned with the parity-backed formatter and runtime contract.
  2. Stored benchmark evidence and generated docs identify the exact fixture, Bonsai contract, and
     Prism reference lane used for the maintained slice.
  3. Published Bonsai benchmark surfaces stay scoped to the single maintained fixture and do not
     overclaim generic 1-bit support, tool calling, thinking replay, or raw fallback support.
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 38. Fixture Provenance And Metadata Truth | 0/TBD | Not started | - |
| 39. Bonsai Conditioning Contract | 0/TBD | Not started | - |
| 40. Native `Q1_0_g128` Runtime Bring-Up | 0/TBD | Not started | - |
| 41. Parity And Regression Proof | 0/TBD | Not started | - |
| 42. Benchmark And Publication | 0/TBD | Not started | - |
