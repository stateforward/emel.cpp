# Phase 83: Native Sortformer ARM Runtime Path - Context

**Gathered:** 2026-04-22
**Status:** Blocked for truthful full execution implementation
**Mode:** Autonomous smart discuss with live artifact metadata check

<domain>
## Phase Boundary

Phase 83 is the native EMEL-owned Sortformer execution path for the maintained
`openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf` artifact. It must run from GGUF
tensors in `src/` without NeMo, Python, ONNX, llama.cpp, ggml, or benchmark-only compute fallback.

The phase cannot honestly be completed by wiring a synthetic projection or a partial head-only
path. The maintained model has real Conformer encoder tensors, Sortformer module tensors, and
Transformer encoder tensors; those must be represented in stage-owned execution contracts before RUN-01 can
pass.

</domain>

<decisions>
## Implementation Decisions

### Maintained Artifact Truth
- Treat the Hugging Face GGUF metadata and tensor header as authoritative for Phase 83.
- Verify that metadata truth by self-converting the original NVIDIA `.nemo` checkpoint with the
  OpenResearchTools converter and comparing the generated GGUF against the maintained GGUF.
- The current artifact metadata reports:
  - `sortformer.config.preprocessor.features=128`
  - `sortformer.config.encoder.n_layers=17`
  - `sortformer.config.encoder.d_model=512`
  - `sortformer.config.encoder.subsampling_factor=8`
  - `sortformer.config.transformer_encoder.num_layers=18`
  - `sortformer.config.transformer_encoder.hidden_size=192`
  - `sortformer.config.sortformer_modules.chunk_len=188`
  - `sortformer.config.sortformer_modules.chunk_right_context=1`
  - `sortformer.config.sortformer_modules.fifo_len=0`
  - `sortformer.config.sortformer_modules.spkcache_update_period=188`
  - `sortformer.config.sortformer_modules.spkcache_len=188`
- Tensor families in the artifact are `prep.`, `enc.`, `mods.`, and `te.`. There is no top-level
  `ep.` family; encoder projection weights are under `mods.ep.*`.
- The self-conversion check matched the maintained GGUF on tensor manifest and all tensor payload
  hashes. The only normalized metadata difference was `general.name`.

### Contract Repair
- Repair the Phase 81/82 contract drift before runtime work. The prior constants rejected the real
  maintained GGUF by expecting `chunk_len=340`, `chunk_right_context=40`, `fifo_len=40`,
  `spkcache_update_period=300`, an 80-bin feature-extractor contract, and an `ep.` family.
- Keep the repair small and verified with model-loader and diarization request tests.

### Execution Scope
- Required execution work must bind real compact tensor names for:
  - feature-extractor tensors: `prep.feat.fb`, `prep.feat.win`
  - Conformer encoder layers: `enc.l{N}.*`
  - Sortformer module projection and speaker heads: `mods.ep.*`, `mods.fh2h.*`, `mods.sh2s.*`,
    `mods.h2s.*`
  - Transformer encoder layers: `te.l{N}.*`
- Execution choices for model readiness, maintained profile, cache readiness, output-capacity
  acceptance, and error outcomes belong in guards and transition rows.
- Numeric work belongs in kernel-owned or Sortformer component-owned execution helpers. It must not
  be hidden in benchmark tools or reference lanes.

</decisions>

<code_context>
## Existing Code Insights

- `src/emel/model/sortformer/detail.*` owns Sortformer GGUF metadata and execution-contract
  validation.
- `src/emel/diarization/request/*` owns the Phase 82 in-memory PCM and feature-extractor contract.
- Existing kernel and embedding generator code has f32, q8_0, and AArch64 matmul helpers, but no
  complete Conformer or Sortformer transformer execution path yet.
- `tests/models/README.md` documents the maintained fixture provenance and now needs to stay
  aligned with the real GGUF profile.

</code_context>

<specifics>
## Specific Ideas

The immediate implemented work in this Phase 83 pass is a blocker-unwinding repair: make existing
model and request contracts match the real maintained GGUF so the future native execution path can
load the target instead of rejecting it. Full RUN-01 still requires dedicated stage-owned
implementation plans for the Conformer plus Sortformer cache/transformer path.

Conversion scratch artifacts live under `/tmp/emel_sortformer_verify/` locally and are intentionally
not repo artifacts.

</specifics>

<deferred>
## Deferred Ideas

- Full Conformer encoder execution over `enc.l{N}.*` under
  `src/emel/diarization/sortformer/encoder/`.
- Sortformer projection/cache/head execution under `src/emel/diarization/sortformer/modules/`.
- Sortformer transformer execution over `te.l{N}.*` under
  `src/emel/diarization/sortformer/transformer/`.
- Speaker probability output and segment decoding remain Phase 84 after runtime logits/probability
  truth exists.
- Reference parity and ARM benchmark publication remain Phase 85.

</deferred>
