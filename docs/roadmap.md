# Parity roadmap

Last updated: 2026-02-19.

This roadmap tracks parity status against reference implementations and highlights what remains.
Scaffolding decisions live in [scaffold.plan.md](plans/scaffold.plan.md).

**Completed parity (audited)**
- [x] Buffer allocator cluster: allocator, planner, chunk allocator, realloc analyzer aligned with
  `ggml-alloc.c` (parity tests included).
- [x] Encoder/tokenizer: BPE, WPM, UGM, RWKV, PLaMo-2, pretokenizer, byte fallback aligned with
  reference vocab behavior.
- [x] Model parser: GGUF metadata mapping and orchestration aligned with reference parser behavior.
- [x] Model loader: orchestration and GGUF callbacks aligned with reference behavior.
- [x] Weight loader: loader callback parity implemented; backend-specific direct I/O and async
  upload exposed via callbacks.

**Partial parity (gaps remaining)**
- [ ] KV cache: slot planning/apply/rollback modeled. Missing stream-aware tracking, stream
  selection, shift/defrag, sliding-window behavior, and sequence ops scheduling parity.
- [ ] Decoder: batch splitting/output selection aligned. Missing batch auto-generation/validation,
  sequence continuity rules, ubatch metadata/output reordering, sampling integration, output buffer
  orchestration, memory context semantics, graph reuse/scheduling, and cross-attention metadata.

**Unvalidated**
- [ ] Generator (`src/emel/generator/sm.hpp`).
- [ ] Sampler pipeline/candidate builder/token selector.
- [ ] Tensor allocator/lifetime analyzer.
- [ ] Telemetry provider/exporter.
- [ ] Top-level `src/emel/sm.hpp`.

**Next milestones**
Next: decoder parity.
1. [ ] Close decoder gaps per [decoder.plan.md](plans/decoder.plan.md).
2. [ ] Expand KV cache parity to stream-aware behavior and sliding-window handling.
3. [ ] Optimize allocator cached-guard checks with a tensor-id lookup cache to avoid
   repeated graph scans in `graph_needs_realloc`.
4. [ ] Audit generator + sampler pipeline parity against reference sampling paths.
5. [ ] Audit tensor allocator/lifetime analyzer parity against `ggml` alloc paths.
6. [ ] Add public C API parity for loader/parser/weight loader paths.
