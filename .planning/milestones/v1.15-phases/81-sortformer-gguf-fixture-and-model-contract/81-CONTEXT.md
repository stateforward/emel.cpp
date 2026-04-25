# Phase 81: Sortformer GGUF Fixture And Model Contract - Context

**Gathered:** 2026-04-22
**Status:** Ready for planning
**Mode:** Autonomous smart discuss defaults

<domain>
## Phase Boundary

Phase 81 pins the exact `openresearchtools/diar_streaming_sortformer_4spk-v2.1-gguf` artifact and
adds truthful model-loader acceptance for the maintained `sortformer` GGUF contract. It does not
implement audio request handling, Sortformer execution, output decoding, parity comparison, or ARM
benchmarking.

</domain>

<decisions>
## Implementation Decisions

### Fixture Contract
- Use the Hugging Face repo metadata and resolve headers as provenance truth for the maintained
  artifact: repo SHA `e2970f92934105c6b385a047dc098aaaa593621b`, architecture `sortformer`, file
  `diar_streaming_sortformer_4spk-v2.1.gguf`, size `471107712` bytes, linked ETag
  `1b85d7bf641350d0d355e7494c4b7d92a1ff2fb2d886cd6dcc43f358a6266ff0`.
- Record the community-conversion source and NVIDIA Open Model License boundary in
  `tests/models/README.md`.
- Do not download or vendor the large model file in this phase.

### Model Acceptance
- Add `sortformer` to the architecture registry as its own model family instead of aliasing it to
  generation, embeddings, or omniembed.
- Load maintained metadata from `sortformer.*` GGUF keys produced by the OpenResearchTools
  conversion script.
- Treat the default maintained streaming profile as mono 16 kHz, four speakers, 80 ms frames,
  `chunk_len=188`, `chunk_right_context=1`, `fifo_len=0`,
  `spkcache_update_period=188`, and `spkcache_len=188`.

### Tensor Contract
- Require compact tensor-name scheme `compact_v1`, source format `nemo`, and maintained outtype
  `f32`.
- Require non-empty compact tensor families for frontend/preprocessor, encoder, Sortformer modules,
  and encoder projection.
- Keep the contract in `src/emel/model/sortformer/` so later runtime phases can build execution
  bindings without reaching into loader tests or benchmark helpers.

### the agent's Discretion
- Exact helper names and field layout may follow the existing `model/omniembed` pattern.
- Focused unit tests may use synthetic GGUF bindings and tensor records rather than the full
  471 MB artifact.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/emel/model/architecture/detail.cpp` owns model-family registration.
- `src/emel/model/loader/detail.hpp` exposes GGUF key lookup and typed value decode helpers.
- `src/emel/model/omniembed/detail.*` is the closest architecture-specific metadata and execution
  contract pattern.
- `tests/model/loader/lifecycle_tests.cpp` already builds synthetic GGUF bindings and synthetic
  tensor records for architecture contract tests.

### Established Patterns
- Architecture families expose `detail.hpp` / `detail.cpp` with `is_execution_architecture`,
  `load_hparams`, `build_execution_contract`, `validate_data`, and
  `validate_execution_contract`.
- `model::validate_execution_contract` resolves the architecture and delegates to the registered
  family validator.
- Maintained fixture provenance is documented in `tests/models/README.md`.

### Integration Points
- `CMakeLists.txt` must include any new architecture detail implementation in the `emel` static
  library.
- `src/emel/model/data.hpp` is the central storage for model metadata needed after GGUF loading.
- `src/emel/model/data.cpp` exposes architecture-family predicates used by tests and runtime.

</code_context>

<specifics>
## Specific Ideas

- Keep Phase 81 to loader/model acceptance only.
- Preserve the later phase boundary: no audio frontend, no Sortformer compute path, no segment
  decoder, no benchmark wrapper in this phase.

</specifics>

<deferred>
## Deferred Ideas

- Phase 82 owns diarization request/audio frontend behavior.
- Phase 83 owns native Sortformer runtime execution.
- Phase 84 owns probability and segment output.
- Phase 85 owns parity proof and benchmark publication.

</deferred>
