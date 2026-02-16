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
- Status: partially implemented, behavior parity not yet achieved.
- Remaining gaps to close for loader parity:
- GGUF header and metadata validation: magic/version, endian, KV/tensor counts, alignment rules,
  and file offsets need concrete parsing and validation paths.
- Split-file discovery and validation: sharded file detection, shard order, and cross-file metadata
  consistency are not implemented.
- I/O strategy parity: mmap vs stream selection, sequential read paths, and error handling must be
  ported into parser/weight-loader callbacks.
- Tensor data mapping lifecycle: tensor index lookup, range checks, offset mapping, and
  buffer-to-tensor binding are incomplete.
- Progress callbacks and reporting: load progress, warnings, and detailed error classification are
  not wired to the public API yet.
- C boundary validation: loader/parse/load error mapping and status return paths are not yet
  validated because the public C API entrypoints are still pending.

Unvalidated machines (no parity audit performed yet)
- `src/emel/model/weight_loader/sm.hpp`
- `src/emel/model/parser/sm.hpp`
- `src/emel/tokenizer/sm.hpp`
- `src/emel/encoder/sm.hpp`
- `src/emel/decoder/sm.hpp`
- `src/emel/decoder/ubatch_executor/sm.hpp`
- `src/emel/decoder/compute_executor/sm.hpp`
- `src/emel/generator/sm.hpp`
- `src/emel/kv/cache/sm.hpp`
- `src/emel/memory/coordinator/sm.hpp`
- `src/emel/sampler/pipeline/sm.hpp`
- `src/emel/sampler/candidate_builder/sm.hpp`
- `src/emel/sampler/token_selector/sm.hpp`
- `src/emel/batch/splitter/sm.hpp`
- `src/emel/tensor/allocator/sm.hpp`
- `src/emel/tensor/lifetime_analyzer/sm.hpp`
- `src/emel/telemetry/provider/sm.hpp`
- `src/emel/telemetry/exporter/sm.hpp`
- `src/emel/sm.hpp`

Recommended next steps
- Decide which component to audit next against `tmp/llama.cpp` and identify the exact reference
  files and functions to map.
