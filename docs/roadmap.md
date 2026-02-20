# Parity roadmap

Last updated: 2026-02-20.

This roadmap tracks parity status against `tmp/llama.cpp` and `ggml` references and highlights
what remains. Scaffolding decisions live in `docs/scaffold.plan.md`.

**Status legend**
- Complete: audited and aligned with reference behavior.
- Partial: implemented but missing known reference behaviors.
- Unvalidated: no parity audit performed yet.

**Complete**
- Buffer allocator cluster: allocator, planner, chunk allocator, realloc analyzer aligned with
  `ggml-alloc.c` (parity tests included).
- Encoder/tokenizer: BPE, WPM, UGM, RWKV, PLaMo-2, pretokenizer, byte fallback aligned with
  `llama-vocab.cpp`.
- Model parser: GGUF metadata mapping and orchestration aligned with `llama-model.cpp`.
- Model loader: orchestration and GGUF callbacks aligned; pending public C API mapping.
- Weight loader: loader callback parity implemented; backend-specific direct I/O and async upload
  exposed via callbacks.

**Partial**
- KV cache: slot planning/apply/rollback modeled. Missing stream-aware tracking, stream selection,
  shift/defrag, sliding-window behavior, and sequence ops scheduling parity.
- Decoder: batch splitting/output selection aligned. Missing batch auto-generation/validation,
  sequence continuity rules, ubatch metadata/output reordering, sampling integration, output buffer
  orchestration, memory context semantics, graph reuse/scheduling, and cross-attention metadata.

**Unvalidated**
- Generator (`src/emel/generator/sm.hpp`).
- Sampler pipeline/candidate builder/token selector.
- Tensor allocator/lifetime analyzer.
- Telemetry provider/exporter.
- Top-level `src/emel/sm.hpp`.

**Next milestones**
1. Close decoder gaps per `docs/decoder.plan.md` and `docs/scaffold.plan.md`.
2. Expand KV cache parity to stream-aware behavior and sliding-window handling.
3. Audit generator + sampler pipeline parity against `tmp/llama.cpp` sampling paths.
4. Audit tensor allocator/lifetime analyzer parity against `ggml` alloc paths.
5. Add public C API parity for loader/parser/weight loader paths.
