# parity roadmap

last updated: 2026-02-19.

this roadmap tracks parity status against reference implementations and highlights what remains.
scaffolding decisions live in [scaffold.plan.md](plans/scaffold.plan.md).

**completed parity (audited)**
- [x] buffer allocator cluster: allocator, planner, chunk allocator, realloc analyzer aligned with
  `ggml-alloc.c` (parity tests included).
- [x] encoder/tokenizer: BPE, WPM, UGM, RWKV, PLaMo-2, preprocessor, byte fallback aligned with
  reference vocab behavior.
- [x] model parser: GGUF metadata mapping and orchestration aligned with reference parser behavior.
- [x] model loader: orchestration and GGUF callbacks aligned with reference behavior.
- [x] weight loader: loader callback parity implemented; backend-specific direct I/O and async
  upload exposed via callbacks.

**partial parity (gaps remaining)**
- [ ] KV cache: slot planning/apply/rollback modeled. missing stream-aware tracking, stream
  selection, shift/defrag, sliding-window behavior, and sequence ops scheduling parity.
- [ ] decoder: batch splitting/output selection aligned. missing batch auto-generation/validation,
  sequence continuity rules, ubatch metadata/output reordering, sampling integration, output buffer
  orchestration, memory context semantics, graph reuse/scheduling, and cross-attention metadata.

**unvalidated**
- [ ] generator (`src/emel/generator/sm.hpp`).
- [ ] sampler pipeline/candidate builder/token selector.
- [ ] tensor allocator/lifetime analyzer.
- [ ] telemetry provider/exporter.
- [ ] top-level `src/emel/sm.hpp`.

**next milestones**
next: decoder parity.
1. [ ] close decoder gaps per [decoder.plan.md](plans/decoder.plan.md).
2. [ ] expand KV cache parity to stream-aware behavior and sliding-window handling.
3. [ ] optimize allocator cached-guard checks with a tensor-id lookup cache to avoid
   repeated graph scans in `graph_needs_realloc`.
4. [ ] audit generator + sampler pipeline parity against reference sampling paths.
5. [ ] audit tensor allocator/lifetime analyzer parity against `ggml` alloc paths.
6. [ ] add public C API parity for loader/parser/weight loader paths.
