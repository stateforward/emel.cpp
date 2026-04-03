# Requirements: EMEL

**Defined:** 2026-04-02
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.

## v1 Requirements

### Fixture And Metadata

- [x] **FIX-03**: The repo documents one official maintained Bonsai fixture at
  `tests/models/Bonsai-1.7B.gguf` with source repo, direct download URL, size `248302272` bytes,
  and SHA256 `0ae245fc08236af7cb64caff164937e53a8d54af611b0f398cc992c0a5ba70c4`.
- [x] **META-02**: The maintained Bonsai slice records executable GGUF truth from the live
  artifact, including `general.architecture=qwen3`, `tokenizer.ggml.model=gpt2`,
  `tokenizer.ggml.pre=qwen2`, `qwen3.context_length=32768`, `qwen3.block_count=28`,
  `qwen3.embedding_length=2048`, and `qwen3.attention.head_count/head_count_kv=16/8`.
- [x] **META-03**: Maintained docs, fixture registries, and publication surfaces distinguish the
  actual fixture filename `Bonsai-1.7B.gguf` from the weight format `Q1_0_g128` and do not rely on
  the stale quickstart filename `Bonsai-1.7B-Q1_0_g128.gguf`.

### Conditioning

- [x] **COND-04**: The maintained Bonsai slice uses one explicit structured chat-message contract
  derived from the embedded `tokenizer.chat_template` in the official GGUF.
- [x] **COND-05**: The maintained Bonsai contract supports only the approved request surface for the
  first slice: `system,user,assistant` messages, `add_generation_prompt=true`, `tools=none`, and
  `enable_thinking=false`.
- [x] **COND-06**: Unsupported Bonsai request shapes, including tool calls, tool responses,
  thinking replay/preservation, named template variants, and raw-prompt fallback, fail explicitly
  instead of silently widening the maintained contract.

### Runtime Support

- [x] **RUN-07**: EMEL-owned model metadata/loading surfaces truthfully accept the maintained
  Bonsai fixture on the existing `qwen3` architecture path without inventing a new `bonsai` or
  `prismml` execution family.
- [x] **RUN-08**: EMEL-owned `src/emel` runtime code provides a native `Q1_0_g128` operand path for
  the maintained Bonsai slice, including dtype/layout support and hot-path execution without a
  whole-tensor dequantize-to-f32 fallback.
- [x] **RUN-09**: The shipped generator path initializes and generates on `Bonsai-1.7B.gguf` using
  the maintained Bonsai formatter contract and the native `Q1_0_g128` runtime path.
- [x] **RUN-10**: EMEL-owned model metadata and generator runtime expose an explicit Qwen-family
  RoPE domain, including rope type and YaRN scaling semantics needed by the maintained Bonsai
  slice, without adding a Bonsai-only behavior lane.
- [x] **RUN-11**: The shipped generator path consumes quantized Bonsai token embeddings through a
  native runtime operand path and no longer classifies that stage as
  `approved_dense_f32_by_contract` on the maintained slice.

### Verification

- [x] **PAR-03**: `tools/paritychecker --generation` proves EMEL against a pinned
  `PrismML-Eng/llama.cpp` reference lane on the maintained Bonsai fixture using the same formatter
  contract.
- [x] **VER-03**: Regression coverage protects the maintained Bonsai slice and the prior maintained
  Llama and Qwen anchors from accidental fixture, formatter, architecture, or quant-path drift.

### Benchmarking

- [x] **PERF-01**: EMEL materially reduces the maintained Bonsai implementation-mode benchmark gap
  against the pinned Prism lane before benchmark publication.
- [ ] **BENCH-09**: `tools/bench` compare output, stored benchmark evidence, and generated docs
  publish one truthful Bonsai benchmark path aligned with the same parity-backed fixture, formatter
  contract, and Prism reference lane.

## v2 Requirements

### Broader Bonsai Coverage

- **MODEL-04**: Broaden beyond `Bonsai-1.7B.gguf` to sibling Bonsai checkpoints or future official
  variants after the first maintained slice is proven and benchmarked.
- **MODEL-05**: Generalize `Q1_0_g128` support into broader custom 1-bit or third-party GGUF
  compatibility only after the maintained Bonsai slice is stable and explicitly approved.

### Richer Request Surfaces

- **COND-07**: Add Bonsai tool calling / function calling support from the embedded template only
  after the first maintained slice has a stable parity-backed request contract.
- **COND-08**: Add Bonsai thinking preservation, replay, or richer assistant-history semantics only
  after the first maintained slice is correct and benchmarked.

### Reference And Performance

- **REF-01**: Revisit whether upstream `ggml-org/llama.cpp` can replace Prism's fork as the
  maintained comparator after upstream lands truthful Bonsai `Q1_0_g128` support.
- **GEN-05**: Optimize Bonsai-specific runtime hot spots only after the maintained slice is correct,
  parity-backed, and benchmarked.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Generic "Bonsai support" or generic "1-bit GGUF support" | v1.10 is one exact maintained fixture, not a family claim. |
| A new `bonsai` or `prismml` execution architecture | The live GGUF already reports `general.architecture=qwen3`; the widening is the quant path, not a new topology family. |
| Tool calling / function calling | The embedded template supports tools, but first-slice proof should stay on `tools=none`. |
| Thinking replay / preservation | The embedded template exposes `<think>` handling, but that widens formatter semantics beyond the first maintained slice. |
| Raw prompt fallback on the maintained Bonsai path | Maintained proof should follow one explicit structured contract, not silent raw formatting. |
| Sibling Bonsai checkpoints or additional official variants | The first milestone needs one exact proven artifact, not a matrix. |
| MLX, Swift, Android, ONNX, vLLM, or Transformers integration | Those are separate deployment stacks and outside the current EMEL generator/parity/bench acceptance boundary. |
| Broad new public API or CLI surfaces for Bonsai workflows | v1.10 is scoped to the existing maintained EMEL seams only. |
| Whole-tensor dequantize-to-f32 or tool-only compute fallback in the shipped hot path | That would violate the repo's quantized-path and performance contract unless explicitly approved as interim. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FIX-03 | Phase 38 | Complete |
| META-02 | Phase 38 | Complete |
| META-03 | Phase 38 | Complete |
| COND-04 | Phase 39 | Complete |
| COND-05 | Phase 39 | Complete |
| COND-06 | Phase 39 | Complete |
| RUN-07 | Phase 40 | Complete |
| RUN-08 | Phase 40 | Complete |
| RUN-09 | Phase 40 | Complete |
| RUN-10 | Phase 40.1 | Complete |
| PAR-03 | Phase 41 | Complete |
| VER-03 | Phase 41 | Complete |
| RUN-11 | Phase 41.1 | Complete |
| PERF-01 | Phase 41.1 | Complete |
| BENCH-09 | Phase 42 | Pending |

**Coverage:**
- v1 requirements: 15 total
- Mapped to phases: 15
- Unmapped: 0

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-04-03 after completing Phase 41.1 full-quantized performance closure*
