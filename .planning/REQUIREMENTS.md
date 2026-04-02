# Requirements: EMEL

**Defined:** 2026-04-02
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.

## v1 Requirements

### Fixture And Metadata

- [ ] **FIX-03**: The repo documents one official `gemma-4-e2b-it-Q8_0.gguf` fixture with
  checksum, source, stable maintained path under `tests/models/`, and download URL.
- [ ] **META-02**: The maintained Gemma 4 slice records executable model truth from official
  GGUF/config metadata, including `architecture=gemma4`, `context_length=131072`, the text-layer
  schedule, and the separate `mmproj` companion file, instead of relying on prose summaries.

### Conditioning

- [ ] **COND-04**: The maintained Gemma 4 slice uses one explicit canonical text-only
  request-conditioning contract derived from the official `chat_template`, with structured text
  chat-message input, `add_generation_prompt=true`, and no implicit raw fallback.
- [ ] **COND-05**: The maintained Gemma 4 text slice rejects `mmproj`, image, audio, video, and
  tool-call request shapes explicitly instead of implying multimodal or tool-use support.

### Runtime Support

- [ ] **RUN-07**: EMEL-owned `src/emel` runtime code truthfully accepts the canonical Gemma 4
  fixture as `gemma4` instead of rejecting it or aliasing it to existing `llama`, `qwen3`, or
  `lfm2` paths.
- [ ] **RUN-08**: EMEL can initialize and generate on the canonical Gemma 4 fixture through the
  maintained generator path in `src/emel`.
- [ ] **RUN-09**: The maintained Gemma 4 runtime explicitly handles the required `gemma4`
  metadata, alternating sliding/full attention schedule, shared-KV layers, rope contracts, and
  tied embeddings instead of pretending broad compatibility with an existing family.

### Reference Boundary

- [ ] **REF-01**: The pinned `llama.cpp` reference lane used by `tools/paritychecker` and
  `tools/bench` is upgraded or confirmed so the canonical Gemma 4 fixture can be loaded for
  maintained comparison.

### Verification

- [ ] **PAR-03**: `tools/paritychecker --generation` proves EMEL against `llama.cpp` on the
  canonical Gemma 4 text slice using the same fixture and conditioning contract.
- [ ] **VER-03**: Regression tests cover the canonical Gemma 4 text slice and protect the prior
  maintained Llama, Qwen, and Liquid anchors from accidental breakage while support is widened.

### Benchmarking

- [ ] **BENCH-09**: `tools/bench` compare output, stored benchmark evidence, and generated docs
  publish one truthful canonical Gemma 4 benchmark path aligned with the parity-checked text-only
  slice.

## v2 Requirements

### Multimodal Support

- **MODAL-01**: Add real `mmproj` support plus image input handling for the official Gemma 4 E2B
  path once the text slice is proven.
- **MODAL-02**: Add audio/video request support only after the repo has an explicit media encoder
  pipeline and maintained multimodal verification.

### Prompt Richness

- **COND-06**: Add tool use or broader official template features only after the canonical text
  slice has a stable maintained contract.

### Broader Model Coverage

- **MODEL-04**: Broaden beyond the canonical `gemma-4-e2b-it-Q8_0.gguf` fixture to `F16` or later
  official Gemma 4 variants only after the first text slice is proven and benchmarked.

### Performance

- **PERF-01**: Optimize Gemma 4-specific runtime hot spots after the canonical text slice is
  correct, parity-backed, and benchmarked.

## Out of Scope

| Feature | Reason |
|---------|--------|
| `mmproj-gemma-4-e2b-it-f16.gguf` | Keep `v1.11` on one truthful text-generation slice first. |
| Image, audio, or video inputs | The upstream model is multimodal, but EMEL does not yet have a maintained multimodal acceptance surface. |
| Tool use or function calling | The official template supports tools, but that widens request shape and API scope beyond the first maintained text slice. |
| `gemma-4-e2b-it-f16.gguf` as a maintained slice | The first milestone needs one exact proven text fixture, not a mixed precision matrix. |
| Raw prompt fallback on the maintained Gemma 4 path | Maintained proof should follow one explicit structured contract, not silent raw formatting. |
| Re-planning adjacent `v1.8` size-benchmark work | Gemma 4 should integrate with existing maintained benchmark seams, not absorb unrelated benchmark-surface planning. |
| Inflight `v2.0` Bonsai `Q1` kernel work | That kernel work is separate adjacent scope and is not part of Gemma 4 milestone acceptance. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FIX-03 | Phase 38 | Pending |
| META-02 | Phase 38 | Pending |
| COND-04 | Phase 38 | Pending |
| COND-05 | Phase 39 | Pending |
| RUN-07 | Phase 39 | Pending |
| RUN-08 | Phase 40 | Pending |
| RUN-09 | Phase 39 | Pending |
| REF-01 | Phase 41 | Pending |
| PAR-03 | Phase 41 | Pending |
| VER-03 | Phase 41 | Pending |
| BENCH-09 | Phase 42 | Pending |

**Coverage:**
- v1 requirements: 11 total
- Mapped to phases: 11
- Unmapped: 0

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-04-02 after roadmap creation*
