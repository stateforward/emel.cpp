# Decoder parity notes (llama.cpp)

## Reference
- Upstream: `tmp/llama.cpp` at commit `abb9f3c42b5e6acee9e8e37836ef691d1a41bdb8`.
- Primary files: `src/llama-context.cpp`, `src/llama-batch.cpp`, `src/llama-batch.h`,
  `src/llama-kv-cache.cpp`, `src/llama-memory-*.cpp`.
- Date: 2026-02-19.

## Current parity snapshot (EMEL)
- Batch splitting/output selection aligned for decode (`output_all`, `output_mask`, last-token only).
- Seq masks / primary ids are propagated into ubatch execution and KV cache apply.
- Per-ubatch 1D/3D position handling aligned.
- Tests and gates passing as of 2026-02-19.

## Known gaps vs llama.cpp decode
- Embedding inputs and pooled embedding outputs (pooling modes, per-sequence embeddings).
- Auto-generation and validation of batch fields (`n_seq_id`, `seq_id`, `pos`, `logits` masks).
- Sequence coupling, continuity checks, and disallowing partial sequence subsets.
- Ubatch metadata parity (`n_seqs`, `n_seq_tokens`, `n_seqs_unq`, `seq_id_unq`, `seq_idx`) and
  output ordering/reordering (`out_ids`, swap tracking).
- Backend sampling integration (samplers, sampled/logits/probs/candidates buffers).
- Output buffer reservation/resizing and host-buffer transfer orchestration.
- Memory context semantics for decode (NO_UPDATE / FAILED_* status handling and rollback).
- Graph reuse/scheduling parity for decode execution.
- Encoder-decoder cross-attention metadata (e.g. T5-style cross state).

## Decomposition proposal (pending approval)
- `emel/batch/sanitizer`: batch validation + auto-generation (mirrors `llama_batch_allocr::init`).
- Additional decoder components suggested (names to be finalized):
  - Output buffer management (reserve / reorder).
  - Backend sampling output handling.
  - Decode-time memory session wrapper (llama memory context semantics).
    - Preferred structure: `memory/coordinator/{recurrent,kv,hybrid}` with shared coordinator events
      and a common dispatch interface.
  - Optional encoder-decoder cross-attention bridge.

## Open questions
- Final naming for non-batch components.
- Whether to keep batch sanitizer decoder-only or shared under `emel/batch`.
