---
phase: 39
slug: bonsai-conditioning-contract
created: 2026-04-02
status: ready
---

# Phase 39 Context

## Phase Boundary

Bind the maintained Bonsai request path to one explicit structured formatter contract backed by the
embedded `tokenizer.chat_template`, and reject unsupported request shapes before runtime or parity
surfaces can overclaim support.

## Implementation Decisions

### Runtime Source Of Truth
- `src/emel/text/formatter/format.hpp` becomes the source of truth for supported embedded-template
  contract detection, structured-request validation, and the runtime-owned Qwen/Bonsai formatter.
- Generator construction resolves formatter binding from `model::data.meta.tokenizer_data` when the
  loaded model carries an embedded chat template; metadata-free synthetic models keep their
  existing injected fallback path.
- Unsupported embedded-template cases resolve to an explicit `unsupported_template` contract kind
  instead of silently falling back to `format_raw`.

### Explicit Rejection Surface
- Conditioner bind now carries an explicit formatter contract kind, and conditioner prepare guards
- not formatter helpers - decide whether a request shape is accepted.
- The maintained Qwen/Bonsai contract accepts only `system`, `user`, and `assistant` messages with
  `add_generation_prompt=true` and `enable_thinking=false`.
- Named template variants, tool-role requests, missing generation prompts, and thinking-enabled
  requests fail as `invalid_argument` on repo-visible initialize/generate surfaces.

### Metadata Truth
- Tool-side GGUF loading now copies `tokenizer.chat_template` plus named template variants into
  `model_data.meta.tokenizer_data`, so the runtime resolves the contract from loaded model truth
  instead of duplicating GGUF parsing logic in generator code.
- No new model family is introduced; Bonsai continues through the existing `qwen3` lane.

## Existing Code Insights

### Reusable Assets
- `tools/generation_formatter_contract.hpp` already encoded the supported Qwen-style marker set and
  chat-message formatting shape used by parity and bench tooling.
- `src/emel/text/conditioner/sm.hpp` already had an explicit prepare rejection path, so Phase 39
  only needed contract-kind propagation and guard tightening rather than a new orchestration graph.
- `model::data.meta.tokenizer_data` already reserved storage for the embedded primary chat template
  and named template variants.

### Integration Points
- `src/emel/generator/sm.hpp` is the right place to resolve formatter binding once from loaded model
  metadata and inject the result into the conditioner path.
- `src/emel/text/conditioner/guards.hpp` is the repo-visible acceptance seam for supported versus
  unsupported structured request shapes.
- `tools/bench/generation_bench.cpp` and `tools/paritychecker/parity_runner.cpp` already populate
  loaded `model::data` instances and therefore can publish embedded-template truth into runtime
  metadata without widening `src/` loader state machines in this phase.
