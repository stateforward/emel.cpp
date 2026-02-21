# parity gaps and validation status

scope of this audit
- reference source: `ggml-alloc.c`.
- target machines reviewed: `src/emel/buffer/allocator/sm.hpp`, `src/emel/buffer/planner/sm.hpp`,
  `src/emel/buffer/chunk_allocator/sm.hpp`, `src/emel/buffer/realloc_analyzer/sm.hpp`.
- date: 2026-02-16.
- all other machines are not yet validated against reference behavior.

allocator cluster status (ggml-alloc parity)
- unexpected-event handling is now explicit for allocator cluster machines via wildcard transitions
  to error states (`buffer::allocator`, `buffer::planner`, `buffer::chunk_allocator`,
  `buffer::realloc_analyzer`).
- in-place reuse is modeled via `tensor_desc.can_inplace` and enforced in the planner reuse path,
  including output tensor guards in the default strategy.
- alignment is now a per-buffer input (initialize/plan) and is used in planner sizing and chunk
  allocator alignment (no longer hardcoded to 16).
- max chunk size is now a per-buffer input and is used by the planner + chunk allocator with
  multi-chunk split plans when limits are exceeded.
- overflow/limit hardening is enforced in planner + allocator size/count paths.
- allocator parity scenarios from the reference test suite are now ported into:
  `tests/buffer/allocator_parity_tests.cpp` and `tests/buffer/chunk_allocator_parity_tests.cpp`.
- public C API allocator-path tests for exact error/status mapping are implemented.
- C API equivalents of `ggml_backend_alloc_ctx_tensors_from_buft[_size]` and
  `ggml_backend_alloc_ctx_tensors` are available via EMEL allocator wrappers (without `ctx`).
- allocator cluster audit is complete against `ggml-alloc.c`.

model loader audit (llama.cpp parity)
- loader, parser, and weight loader orchestration is now implemented with explicit actions, guards,
  and error propagation via `events::*_error` / `events::*_done`.
- loader dispatches parsing and weight loading through parser/weight_loader state machines and
  supports `vocab_only`, `check_tensors`, `no_alloc`, and optional architecture validation.
- status: complete for GGUF loader callback parity and loader orchestration.
- implemented: GGUF header/metadata validation (including split metadata), split-file parsing and
  cross-file consistency checks, mmap/stream/direct-IO selection, tensor mapping/range checks, and
  progress/upload callbacks.
- note: public C API entrypoints and C-boundary status mapping remain pending as a separate task.

parser audit (llama.cpp parity)
- reference source: `llama-model.cpp` and `llama-vocab.cpp`.
- status: complete for GGUF metadata mapping to `emel::model::data` fields and parser orchestration.
- implemented: tokenizer IDs and flags, token arrays, merges, and vocabulary metadata required by EMEL.

weight loader audit (llama.cpp parity)
- reference source: `llama-model-loader.cpp` (weight mapping + data load).
- status: implemented for EMEL loader callbacks (strategy selection, mappings init, mmap/stream
  load, validation, and cleanup).
- notes:
  - direct I/O handling and async upload are backend-specific and are exposed through
    loader callbacks rather than embedded in the state machine.

KV cache audit (llama.cpp parity)
- reference source: `llama-kv-cache.cpp`.
- status: partial. slot planning, apply, and rollback are modeled, but multi-stream support and
  sequence-aware operations are not yet represented in the state machine.
- gaps to close:
  - stream-aware cell tracking (`n_stream`, `seq_to_stream`, per-stream heads) and stream selection.
  - sequence operations (`seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div`) and copy scheduling.
  - shift/defrag handling and sliding-window behaviors.

decoder audit (llama.cpp parity)
- reference sources: `llama-context.cpp`, `llama-batch.cpp`, `llama-batch.h`,
  `llama-kv-cache.cpp`, `llama-memory-*.cpp`.
- reference commit: `abb9f3c42b5e6acee9e8e37836ef691d1a41bdb8`.
- date: 2026-02-19.
- status: partial. batch splitting, output selection (`output_all`, `output_mask`, last-token),
  seq masks/primary ids, and 1D/3D position handling are aligned for decode execution.
- gaps to close:
  - embedding inputs and pooled embedding outputs (pooling modes, per-sequence embeddings).
  - auto-generation and validation of batch fields (`n_seq_id`, `seq_id`, `pos`, `logits` masks).
  - sequence coupling, continuity checks, and disallowing partial sequence subsets.
  - ubatch metadata parity (`n_seqs`, `n_seq_tokens`, `n_seqs_unq`, `seq_id_unq`, `seq_idx`) and
    output ordering/reordering (`out_ids`, swap tracking).
  - backend sampling integration (samplers, sampled/logits/probs/candidates buffers).
  - output buffer reservation/resizing and host-buffer transfer orchestration.
  - memory context semantics for decode (NO_UPDATE/FAILED_* status handling and rollback).
  - graph reuse/scheduling parity for decode execution.
  - encoder-decoder cross-attention metadata (e.g. t5-style cross state).

encoder audit (llama.cpp parity)
- reference sources: `llama-vocab.cpp`, `llama-vocab.h`.
- reference commit: `abb9f3c42b5e6acee9e8e37836ef691d1a41bdb8`.
- date: 2026-02-20.
- status: complete. encoder/tokenizer behavior is aligned with llama.cpp for BPE, WPM, UGM, RWKV,
  and PLaMo-2, including pre-tokenizer regex mapping, word-level splitting, defaults, and
  byte-fallback handling.

unvalidated machines (no parity audit performed yet)
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

recommended next steps
- decide which component to audit next against the reference implementation and identify the exact
  files and functions to map.
