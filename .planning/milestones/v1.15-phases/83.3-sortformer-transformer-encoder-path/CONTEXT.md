# Phase 83.3 Context: Sortformer Transformer Encoder Path

## Goal

Bind and execute the maintained Sortformer `te.*` transformer encoder layer contract in
stage-owned native code under `src/emel/diarization/sortformer/transformer/`.

## Maintained Contract

The self-converted maintained artifact reports the transformer family as:

- `transformer_encoder.num_layers = 18`
- `transformer_encoder.hidden_size = 192`
- `transformer_encoder.inner_size = 768`
- `transformer_encoder.num_attention_heads = 8`
- `transformer_encoder.hidden_act = relu`

Each layer owns the same fixed tensor set:

- `te.lN.sa.q.{w,b}`
- `te.lN.sa.k.{w,b}`
- `te.lN.sa.v.{w,b}`
- `te.lN.sa.o.{w,b}`
- `te.lN.ln1.{w,b}`
- `te.lN.ln2.{w,b}`
- `te.lN.ff.di.{w,b}`
- `te.lN.ff.do.{w,b}`

## Constraints

- Do not add a generic Sortformer runtime or dispatcher.
- Keep the transformer path in its own component directory.
- Use caller-owned workspace for execution scratch so the stage can be composed without heap
  allocation in the hot path.
- Keep the reference lane and tool-only helpers out of the EMEL execution path.

## Plan

1. Add fixed-profile transformer tensor binding with exact shape checks for all 288 maintained
   `te.*` tensors.
2. Add native layer-normalization, multi-head self-attention, ReLU feed-forward, and residual
   stage kernels for `T x 192` frame sequences up to the maintained 188-frame chunk.
3. Add focused lifecycle tests for contract binding, missing tensor rejection, shape drift
   rejection, deterministic normalization, deterministic transformer execution, and invalid
   frame shape rejection.
4. Wire the source and tests into CMake and rerun focused plus full quality gates.
