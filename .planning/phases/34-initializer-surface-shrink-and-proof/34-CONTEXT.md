---
phase: 34
slug: initializer-surface-shrink-and-proof
created: 2026-04-02
status: ready
---

# Phase 34 Context

## Phase Boundary

Phase 34 proves the EMEL-owned Qwen3 E2E probe for the maintained executable-size workload. The
phase covers the final linked EMEL runner, native model/vocab/tokenizer loading, and the single
`hello` -> first-token execution slice used by the milestone.

This phase must stay EMEL-owned. It cannot depend on `llama.cpp` or `ggml` bootstrap for the
published EMEL path.

## Implementation Decisions

### EMEL-Owned Path
- The maintained EMEL probe must load GGUF, vocab, tokenizer, formatter contract, and generation
  state through EMEL-owned code.
- The maintained EMEL probe must not bootstrap vocab or generation behavior through the reference
  path.

### Executable Truth
- The proof surface is the final linked probe executable, not an intermediate archive or object.
- The phase is allowed to fix harness artifacts that distort executable size, but it must not
  present a smaller binary by weakening the E2E boundary.

### Probe Scope
- The maintained proof slice is exactly the canonical `hello` -> first-token path.
- Heap placement or harness cleanup is acceptable when it makes executable-size measurement more
  honest relative to the reference executable.

## Guardrails

- No library-size publication as a substitute for the runner executable.
- No reference-assisted bootstrap in the published EMEL path.
- No scope widening beyond the maintained Qwen3 fixture and bounded generation slice.
