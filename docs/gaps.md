# Parity gaps and validation status

Scope of this audit
- Reference source: `tmp/llama.cpp/ggml/src/ggml-alloc.c`.
- Target machines reviewed: `src/emel/buffer/allocator/sm.hpp`, `src/emel/buffer/planner/sm.hpp`,
  `src/emel/buffer/chunk_allocator/sm.hpp`, `src/emel/buffer/realloc_analyzer/sm.hpp`.
- Date: 2026-02-16.
- All other machines are not yet validated against `tmp/llama.cpp` behavior.

Allocator cluster status (ggml-alloc parity)
- Unexpected-event handling is now explicit for allocator cluster machines via wildcard transitions
  to error states (`buffer::allocator`, `buffer::planner`, `buffer::chunk_allocator`,
  `buffer::realloc_analyzer`).
- In-place reuse is modeled via `tensor_desc.can_inplace` and enforced in the planner reuse path,
  including output tensor guards in the default strategy.
- Alignment is now a per-buffer input (initialize/plan) and is used in planner sizing and chunk
  allocator alignment (no longer hardcoded to 16).
- Max chunk size is now a per-buffer input and is used by the planner + chunk allocator with
  multi-chunk split plans when limits are exceeded.
- Overflow/limit hardening is enforced in planner + allocator size/count paths.
- Allocator parity scenarios from the reference test suite are now ported into:
  `tests/buffer/allocator_parity_tests.cpp` and `tests/buffer/chunk_allocator_parity_tests.cpp`.
- Public C API allocator-path tests for exact error/status mapping are implemented.
- C API equivalents of `ggml_backend_alloc_ctx_tensors_from_buft[_size]` and
  `ggml_backend_alloc_ctx_tensors` are available via EMEL allocator wrappers (without `ctx`).
- Allocator cluster audit is complete against `ggml-alloc.c`.

Model loader audit (llama.cpp parity)
- Loader, parser, and weight loader orchestration is now implemented with explicit actions, guards,
  and error propagation via `events::*_error` / `events::*_done`.
- Loader dispatches parsing and weight loading through parser/weight_loader state machines and
  supports `vocab_only`, `check_tensors`, `no_alloc`, and optional architecture validation.
- Status: complete for GGUF loader callback parity and loader orchestration.
- Implemented: GGUF header/metadata validation (including split metadata), split-file parsing and
  cross-file consistency checks, mmap/stream/direct-IO selection, tensor mapping/range checks, and
  progress/upload callbacks.
- Note: public C API entrypoints and C-boundary status mapping remain pending as a separate task.

Parser audit (llama.cpp parity)
- Reference source: `tmp/llama.cpp/src/llama-model.cpp` and `tmp/llama.cpp/src/llama-vocab.cpp`.
- Status: complete for GGUF metadata mapping to `emel::model::data` fields and parser orchestration.
- Implemented: tokenizer IDs and flags, token arrays, merges, and vocabulary metadata required by EMEL.

Weight loader audit (llama.cpp parity)
- Reference source: `tmp/llama.cpp/src/llama-model-loader.cpp` (weight mapping + data load).
- Status: implemented for EMEL loader callbacks (strategy selection, mappings init, mmap/stream
  load, validation, and cleanup).
- Notes:
  - Direct I/O handling and async upload are backend-specific and are exposed through
    loader callbacks rather than embedded in the state machine.

KV cache audit (llama.cpp parity)
- Reference source: `tmp/llama.cpp/src/llama-kv-cache.cpp`.
- Status: partial. Slot planning, apply, and rollback are modeled, but multi-stream support and
  sequence-aware operations are not yet represented in the state machine.
- Gaps to close:
  - Stream-aware cell tracking (`n_stream`, `seq_to_stream`, per-stream heads) and stream selection.
  - Sequence operations (`seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div`) and copy scheduling.
  - Shift/defrag handling and sliding-window behaviors.

Decoder audit (llama.cpp parity)
- Reference sources: `tmp/llama.cpp/src/llama-context.cpp`, `tmp/llama.cpp/src/llama-batch.cpp`,
  `tmp/llama.cpp/src/llama-batch.h`, `tmp/llama.cpp/src/llama-kv-cache.cpp`,
  `tmp/llama.cpp/src/llama-memory-*.cpp`.
- Reference commit: `abb9f3c42b5e6acee9e8e37836ef691d1a41bdb8`.
- Date: 2026-02-19.
- Status: partial. Batch splitting, output selection (`output_all`, `output_mask`, last-token),
  seq masks/primary ids, and 1D/3D position handling are aligned for decode execution.
- Gaps to close:
  - Embedding inputs and pooled embedding outputs (pooling modes, per-sequence embeddings).
  - Auto-generation and validation of batch fields (`n_seq_id`, `seq_id`, `pos`, `logits` masks).
  - Sequence coupling, continuity checks, and disallowing partial sequence subsets.
  - Ubatch metadata parity (`n_seqs`, `n_seq_tokens`, `n_seqs_unq`, `seq_id_unq`, `seq_idx`) and
    output ordering/reordering (`out_ids`, swap tracking).
  - Backend sampling integration (samplers, sampled/logits/probs/candidates buffers).
  - Output buffer reservation/resizing and host-buffer transfer orchestration.
  - Memory context semantics for decode (NO_UPDATE/FAILED_* status handling and rollback).
  - Graph reuse/scheduling parity for decode execution.
  - Encoder-decoder cross-attention metadata (e.g. T5-style cross state).

Encoder audit (llama.cpp parity)
- Reference sources: `tmp/llama.cpp/src/llama-vocab.cpp`, `tmp/llama.cpp/src/llama-vocab.h`.
- Reference commit: `abb9f3c42b5e6acee9e8e37836ef691d1a41bdb8`.
- Date: 2026-02-20.
- Status: complete. Encoder/tokenizer behavior is aligned with llama.cpp for BPE, WPM, UGM, RWKV,
  and PLaMo-2, including pre-tokenizer regex mapping, word-level splitting, defaults, and
  byte-fallback handling.

Unvalidated machines (no parity audit performed yet)
- `src/emel/model/weight_loader/sm.hpp`
- `src/emel/generator/sm.hpp`
- `src/emel/sampler/pipeline/sm.hpp`
- `src/emel/sampler/candidate_builder/sm.hpp`
- `src/emel/sampler/token_selector/sm.hpp`
- `src/emel/tensor/allocator/sm.hpp`
- `src/emel/tensor/lifetime_analyzer/sm.hpp`
- `src/emel/telemetry/provider/sm.hpp`
- `src/emel/telemetry/exporter/sm.hpp`
- `src/emel/sm.hpp`

Recommended next steps
- Decide which component to audit next against `tmp/llama.cpp` and identify the exact reference
  files and functions to map.
