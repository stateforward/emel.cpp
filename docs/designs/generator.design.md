# generator architecture design (rolling)

this document defines the modality-agnostic generator actor. it orchestrates inference without
embedding modality-specific codecs.

## role
- generator is the top-level orchestrator for inference and owns long-lived components.
- generator receives the loaded model via dependency injection and binds downstream components
  from model metadata.
- generator does not load models internally; model loading belongs to higher-level C API
  orchestration.

## composition
- input adapter: conditioner (modality-specific).
- batching: token/batcher.
- planning: batch/planner.
- graph: graph/builder.
- compute: graph/processor.
- memory: memory/coordinator.
- sampling: sampler pipeline (candidate building + token selection).
- output adapter: renderer (modality-specific).

## ownership decisions (inference pipeline)
- generator is the least common owner for memory coordination.
- generator owns `memory/coordinator` and drives its lifecycle phases
  (`prepare_update`, `prepare_full`, `prepare_batch`).
- generator owns `token/batcher` and dispatches it before decode.

## events (draft)
- `event::bind` (model + component binding)
  - inputs: model metadata, handles to conditioner/renderer, `error_out`.
  - outputs: bound state ready to generate.
- `event::generate`
  - inputs: conditioned batch (or conditioner request), output buffers/handles, stop criteria,
    `error_out`.
  - outputs: completion status, rendered output via renderer.

## flow (draft)
1. bind model metadata and attach conditioner, planner, graph/builder, graph/processor, sampler,
   renderer, and memory coordinator.
2. on generate, obtain conditioned batch (dispatch to conditioner if needed).
3. run token/batcher and prepare memory phases.
4. plan steps with batch/planner to produce a `batch::plan`.
5. build or reuse a graph via graph/builder.
6. run graph/processor:
   - processor produces output buffers for sampling.
   - sampler pipeline selects next token(s).
   - generator updates batch positions and dispatches tokens to renderer.
7. stop on criteria (token limit, eos, user stop) and finalize outputs.

## conditioned input contract (draft)
- token ids + token count.
- sequence metadata (seq masks + primary ids) when multi-seq or interleaved.
- positions array when provided; otherwise generated downstream.
- output selection fields (output mask, output_all, enforce_single_output_per_seq).
- optional cross-attention handles for multimodal conditioning (future).

## renderer output contract (draft)
- per-step token ids for output rows selected by processor.
- optional row->seq mapping for multi-sequence rendering.
- emit policy flags (emit special tokens, streaming boundaries).

## invariants
- no allocations during dispatch; all buffers are caller-provided or pre-sized.
- no self-dispatch; internal progress uses anonymous transitions.

## open questions
- where should graph reuse/worst-case reservation live (generator vs graph/builder)?
- what is the minimal conditioned artifact contract between conditioner and generator?
