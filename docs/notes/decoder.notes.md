# decoder parity notes (llama.cpp)

## reference
- upstream reference commit: `abb9f3c42b5e6acee9e8e37836ef691d1a41bdb8`.
- primary files: `src/llama-context.cpp`, `src/llama-batch.cpp`, `src/llama-batch.h`,
  `src/llama-kv-cache.cpp`, `src/llama-memory-*.cpp`.
- date: 2026-02-19.

## current parity snapshot (EMEL)
- batch splitting/output selection aligned for decode (`output_all`, `output_mask`, last-token only).
- seq masks / primary ids are propagated into ubatch execution and KV cache apply.
- per-ubatch 1D/3D position handling aligned.
- tests and gates passing as of 2026-02-19.

## known gaps vs llama.cpp decode
- embedding inputs and pooled embedding outputs (pooling modes, per-sequence embeddings).
- auto-generation and validation of batch fields (`n_seq_id`, `seq_id`, `pos`, `logits` masks).
- sequence coupling, continuity checks, and disallowing partial sequence subsets.
- ubatch metadata parity (`n_seqs`, `n_seq_tokens`, `n_seqs_unq`, `seq_id_unq`, `seq_idx`) and
  output ordering/reordering (`out_ids`, swap tracking).
- backend sampling integration (samplers, sampled/logits/probs/candidates buffers).
- output buffer reservation/resizing and host-buffer transfer orchestration.
- memory context semantics for decode (NO_UPDATE / FAILED_* status handling and rollback).
- graph reuse/scheduling parity for decode execution.
- encoder-decoder cross-attention metadata (e.g. t5-style cross state).

## decomposition proposal (pending approval)
- `emel/batch/sanitizer`: batch validation + auto-generation (mirrors `llama_batch_allocr::init`).
- additional decoder components suggested (names to be finalized):
  - output buffer management (reserve / reorder).
  - backend sampling output handling.
  - decode-time memory session wrapper (llama memory context semantics).
    - preferred structure: `memory/coordinator/{recurrent,kv,hybrid}` with shared coordinator events
      and a common dispatch interface.
  - optional encoder-decoder cross-attention bridge.

## open questions
- final naming for non-batch components.
- whether to keep batch sanitizer decoder-only or shared under `emel/batch`.
