# Requirements: EMEL

**Defined:** 2026-03-31
**Core Value:** Prove real end-to-end behavior with explicit SML orchestration and parity-oriented
verification before widening API surface or model scope.

## v1 Requirements

### Fixture And Metadata

- [ ] **FIX-02**: The repo documents one official `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture with
  checksum, source, stable maintained path under `tests/models/`, and download URL.
- [ ] **META-01**: The maintained Liquid slice records executable model truth from official
  GGUF/config metadata, including `architecture=lfm2` and the official long-context setting,
  instead of relying on stale prose-only metadata.

### Conditioning

- [ ] **COND-03**: The maintained Liquid slice uses one explicit canonical request-conditioning
  contract derived from the primary GGUF `tokenizer.chat_template`, with structured chat-message
  input, `tools=none`, `add_generation_prompt=true`, and no implicit raw fallback.

### Runtime Support

- [ ] **RUN-03**: EMEL-owned `src/emel` runtime code truthfully accepts the canonical Liquid
  fixture as `lfm2` instead of rejecting it or aliasing it to existing `llama`/`qwen3` paths.
- [ ] **RUN-04**: EMEL can initialize and generate on the canonical Liquid fixture through the
  maintained generator path in `src/emel`.
- [ ] **RUN-05**: The maintained Liquid runtime explicitly handles the canonical slice's required
  `lfm2` metadata, tensor names, and hybrid block contract instead of pretending broad
  Llama-family compatibility.
- [ ] **RUN-06**: The maintained Liquid runtime publishes a truthful quantized-path contract for
  the chosen official fixture and does not claim sibling Liquid quant support that is not proven.

### Verification

- [ ] **PAR-02**: `tools/paritychecker --generation` proves EMEL against `llama.cpp` on the
  canonical Liquid slice using the same fixture and conditioning contract.
- [ ] **VER-02**: Regression tests cover the canonical Liquid slice and protect the prior
  maintained Llama and Qwen anchors from accidental breakage while support is widened.

### Benchmarking

- [ ] **BENCH-08**: `tools/bench` compare output, stored benchmark evidence, and generated docs
  publish one truthful canonical Liquid benchmark path aligned with the parity-checked slice.

## v2 Requirements

### Broader Liquid Coverage

- **MODEL-03**: Broaden beyond the canonical `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture to
  additional Liquid checkpoints or sibling quantizations once the first slice is proven and
  benchmarked.

### Prompt Richness

- **COND-04**: Add richer Liquid system-message, multi-turn thinking-history, or tool-calling
  request surfaces only after the canonical maintained slice has an explicit and stable
  conditioning contract.

### Performance

- **GEN-04**: Optimize Liquid-specific runtime hot spots after the canonical Liquid slice is
  correct, parity-backed, and benchmarked.

### Benchmark Policy

- **BENCH-07**: Revisit whether noisy benchmark drift should become a blocking repo gate once the
  maintained compare surfaces are stable enough to justify it.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Broad Liquid-family or multi-model support | Keep v1.9 fixed to one maintained Liquid Thinking GGUF slice first. |
| Sibling Liquid quantizations such as `Q4_0`, `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16`, or `F16` | The first milestone needs one exact proven fixture, not a quant matrix. |
| Tool use or function calling | Liquid documents tool use, but that widens request shape and API scope beyond the first maintained slice. |
| Multi-turn thinking-history replay | The official template supports `keep_past_thinking`, but v1.9 should stay on one narrow maintained request contract. |
| Raw prompt fallback on the maintained Liquid path | Maintained proof should follow one explicit structured contract, not silent raw formatting. |
| Decode extraction or broader generator decomposition | That work is being pursued separately and is not part of this branch's milestone. |
| Benchmark gate hardening | Warning-only benchmark debt is separate from proving the first Liquid slice. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FIX-02 | Phase 33 | Pending |
| META-01 | Phase 33 | Pending |
| COND-03 | Phase 33 | Pending |
| RUN-03 | Phase 34 | Pending |
| RUN-04 | Phase 35 | Pending |
| RUN-05 | Phase 34 | Pending |
| RUN-06 | Phase 35 | Pending |
| PAR-02 | Phase 36 | Pending |
| VER-02 | Phase 36 | Pending |
| BENCH-08 | Phase 37 | Pending |

**Coverage:**
- v1 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0

---
*Requirements defined: 2026-03-31*
*Last updated: 2026-03-31 after roadmap creation*
