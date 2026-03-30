# Phase 27: Qwen3 Runtime Architecture Bring-Up - Context

**Gathered:** 2026-03-27
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 27 must bring one truthful canonical `qwen3` runtime slice through the shipped EMEL
generator path in `src/emel` without pretending that `qwen3` is just a renamed Llama model. The
runtime work must stay narrow to the dense `Qwen3-0.6B-Q8_0.gguf` slice, explicitly bind the
required Qwen3 tensors and hparams, and keep the current Boost.SML actor structure intact.

This phase stays inside the model/execution-view and generator runtime helpers that feed the
existing generator state machine. It must not widen to MoE variants, broader Qwen-family coverage,
or public API changes.

</domain>

<decisions>
## Implementation Decisions

### Explicit Runtime Topology
- Treat `qwen3` as an explicit architecture branch with its own required tensor contract instead of
  a broad Llama-family alias.
- Require `blk.%d.attn_q_norm.weight` and `blk.%d.attn_k_norm.weight` for the canonical Qwen3
  slice and bind them explicitly in the execution view.
- Keep output projection explicit: use `output.weight` when present, with a narrow tied-embedding
  fallback only if the canonical runtime helper can prove that case safely.

### Attention Semantics
- Match the dense Qwen3 reference order: RMS norm hidden state, matmul Q/K/V, reshape by head,
  apply Q/K RMS normalization, then apply RoPE.
- Keep Q/K norm handling in allocation-free runtime helpers inside the existing generator path; no
  ad hoc hidden fallback and no state-machine rewrite.
- Preserve the existing flash/nonflash selection contract while making the underlying Q/K tensors
  truthful for Qwen3.

### Runtime Contract Publication
- Expand quantized-path auditing only as far as needed to account explicitly for any new dense-f32
  vector stages introduced by Qwen3.
- Preserve the prior Llama runtime path as a first-class supported slice rather than regressing it
  during the Qwen3 bring-up.
- Keep broader architecture handling deferred; this phase proves one canonical dense Qwen3 slice
  only.

### the agent's Discretion
- The exact helper split between execution-view binding, quantized-path audit accounting, and
  generator detail kernels can stay local as long as the resulting runtime behavior is explicit and
  architecture-guarded.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/model/data.cpp` already owns the maintained execution-view and quantized-path audit
  helpers used by generator and parity surfaces.
- `src/emel/generator/detail.hpp` already contains the full shipped Q/K/V matmul, RoPE, cache, and
  flash/nonflash execution path, so the Qwen3 bring-up can stay inside the existing runtime helper
  chain.
- `tests/generator/lifecycle_tests.cpp`, `tests/generator/detail_tests.cpp`, and
  `tests/model/loader/lifecycle_tests.cpp` already provide prepared-model fixtures and execution-view
  assertions that can be widened to cover Qwen3 explicitly.

### Established Patterns
- Generator runtime support is exposed through additive helper logic and wrapper accessors rather
  than new events or transition-table rows.
- Quantized-path truth is already modeled as explicit stage-family audit data, not by inference
  from benchmark output alone.
- Architecture validation is currently explicit and narrow on maintained tool surfaces, which fits
  the planned Qwen3 branching model.

### Integration Points
- `src/emel/model/llama/detail.hpp` and `src/emel/model/data.cpp` are the maintained execution-view
  contract that currently assumes Llama-shaped blocks.
- `src/emel/generator/detail.hpp` is the shipped runtime hot path where Qwen3 Q/K norm semantics
  must be introduced.
- `tests/generator/lifecycle_tests.cpp` and `tests/model/loader/lifecycle_tests.cpp` are the
  lowest-friction regression surfaces for explicit Qwen3 topology proof before parity and bench
  cutover.

</code_context>

<specifics>
## Specific Ideas

- The official dense Qwen3 reference path reshapes Q/K/V by head, applies RMS norm to Q and K
  using per-head weights, then applies RoPE. That exact ordering should become the canonical EMEL
  runtime behavior for `qwen3`.
- The official Qwen3 GGUF contains `output.weight`, so the canonical fixture does not require a
  tied-embedding fallback to succeed, but the maintained helper can still keep the fallback path
  explicit and narrow.
- Q/K norm weights are vector weights, so they should be published as explicit dense-f32 stages if
  they enter the maintained quantized contract inventory.

</specifics>

<deferred>
## Deferred Ideas

- `qwen3moe`, `qwen3next`, `qwen35`, or other broader Qwen-family topology support.
- Metadata-driven chat-template rendering and richer request surfaces.
- Performance tuning beyond the minimum truthful Qwen3 runtime bring-up.

</deferred>

---
*Phase: 27-qwen3-runtime-architecture-bring-up*
*Context gathered: 2026-03-27*
