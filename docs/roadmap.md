# parity roadmap

last updated: 2026-02-22.

this roadmap tracks parity status against reference implementations and highlights what remains.
scaffolding decisions live in [scaffold.plan.md](plans/scaffold.plan.md).

**completed parity (audited)**
- [x] buffer allocator cluster: allocator, planner, chunk allocator, realloc analyzer aligned with
  `ggml-alloc.c` (parity tests included).
- [x] encoder/tokenizer: tokenizer preprocessors and orchestration; BPE, WPM, UGM, RWKV, PLaMo-2,
  byte fallback aligned with reference vocab behavior.
- [x] model parser: GGUF metadata mapping and orchestration aligned with reference parser behavior.
- [x] model loader: orchestration and GGUF callbacks aligned with reference behavior.
- [x] weight loader: loader callback parity implemented; backend-specific direct I/O and async
  upload exposed via callbacks.
- [x] tools: `tools/bench` and `tools/paritychecker` parity harnesses implemented.
- [x] jinja: templating and orchestration implemented.

**open behavior gaps**
- [ ] loader/parser/weight-loader C API parity: public C API entrypoints and exact C-boundary
  status mapping are still pending.
- [ ] weight-loading full parity audit: callback-path parity is implemented, but full behavioral
  validation against the reference loader paths is still pending.
- [ ] KV cache: stream-aware cell tracking (`n_stream`, `seq_to_stream`, per-stream heads) and
  stream selection parity.
- [ ] KV cache: sequence operations (`seq_rm`, `seq_cp`, `seq_keep`, `seq_add`, `seq_div`) and
  copy scheduling parity.
- [ ] KV cache: shift/defrag handling and sliding-window behavior parity.
- [ ] decoder: embedding inputs and pooled embedding outputs (pooling modes, per-sequence
  embeddings).
- [ ] decoder: auto-generation and validation of batch fields (`n_seq_id`, `seq_id`, `pos`,
  logits masks).
- [ ] decoder: sequence coupling/continuity rules and disallowing partial sequence subsets.
- [ ] decoder: ubatch metadata parity (`n_seqs`, `n_seq_tokens`, `n_seqs_unq`, `seq_id_unq`,
  `seq_idx`).
- [ ] decoder: output ordering/reordering parity (`out_ids`, swap tracking).
- [ ] decoder: backend sampling integration (samplers, sampled/logits/probs/candidates buffers).
- [ ] decoder: output buffer reservation/resizing and host-buffer transfer orchestration parity.
- [ ] decoder: memory context decode semantics parity (`NO_UPDATE`, `FAILED_*`, rollback).
- [ ] decoder: graph reuse/scheduling parity for decode execution.
- [ ] decoder: encoder-decoder cross-attention metadata parity (for example t5-style cross state).
- [ ] generation/sampling behavior parity audit: end-to-end generation sequencing, candidate
  building, and token selection paths still need full reference validation.
- [ ] tensor memory behavior parity audit: tensor allocation + lifetime semantics still need full
  reference validation.
- [ ] telemetry behavior parity audit: provider/exporter semantics still need full reference
  validation.
- [ ] top-level orchestration parity audit: cross-component dispatch sequencing and error
  propagation still need full reference validation.
- [ ] GBNF parser behavior confidence gap: parser semantics need re-evaluation before parity
  sign-off.

**next milestones**
next: decoder parity.
1. [ ] close decoder gaps per [decoder.plan.md](plans/decoder.plan.md).
2. [ ] expand KV cache parity to stream-aware behavior and sliding-window handling.
3. [ ] optimize allocator cached-guard checks with a tensor-id lookup cache to avoid
   repeated graph scans in `graph_needs_realloc`.
4. [ ] audit generator + sampler pipeline parity against reference sampling paths.
5. [ ] audit tensor allocator/lifetime analyzer parity against `ggml` alloc paths.
6. [ ] add public C API parity for loader/parser/weight loader paths.
