# Project Research Summary

**Project:** EMEL
**Domain:** Brownfield C++ GGUF inference engine adding one maintained Gemma 4 E2B text slice
**Researched:** 2026-04-02
**Confidence:** MEDIUM

## Executive Summary

`v1.11` should not be framed as "Gemma 4 support." It should be framed as one narrow maintained
text-generation slice for the official `ggml-org/gemma-4-E2B-it-GGUF` artifact
`gemma-4-e2b-it-Q8_0.gguf`, proven through EMEL's existing generator, paritychecker, and
benchmark surfaces. Experts would build this as a truth-anchor milestone: pin one official file,
bind one explicit text-only chat contract, add the missing `gemma4` runtime path in `src/emel`,
make the pinned `llama.cpp` reference lane Gemma 4-capable, then publish parity and benchmark
evidence only for that exact slice.

The key implementation recommendation is to treat official GGUF/config metadata as the executable
truth source. For this milestone, that means `general.architecture=gemma4`, context length
`131072`, alternating sliding/full attention layers, shared-KV layers, tied embeddings, and the
fact that the upstream base model is multimodal with separate media token ids and a separate
`mmproj` file. That truth source drives the roadmap: Phase 38 should lock fixture identity,
metadata truth, and the Gemma 4 text-only formatter contract; Phases 39 and 40 should add explicit
`gemma4` model/runtime support without aliasing it to `llama`, `qwen3`, or `lfm2`; Phase 41
should make the reference lane Gemma 4-capable and prove parity/regression; Phase 42 should then
publish the same slice.

The main risks are false readiness and silent overclaiming. The repo can look Gemma-capable while
still only understanding current maintained families, it can imply multimodal support because an
official `mmproj` file exists, or it can assume parity is ready while the pinned `llama.cpp`
reference commit still lacks `gemma4`. Mitigation is straightforward: name one maintained file
everywhere, keep the formatter contract text only, require explicit `gemma4` runtime handling,
make the reference pin an explicit requirement, and keep benchmark publication behind parity and
regression proof.

## Key Findings

### Recommended Stack

No new serving framework or benchmark harness is needed. The correct stack is one official Gemma 4
GGUF fixture, the existing EMEL runtime plus a new repo-local `gemma4` execution path, the current
formatter/conditioner seams with a new Gemma 4 text-only binding, and a Gemma 4-capable
`llama.cpp` reference lane for parity and bench.

**Core technologies:**
- Official `ggml-org/gemma-4-E2B-it-GGUF` with `gemma-4-e2b-it-Q8_0.gguf`: maintained fixture truth
  anchor for `v1.11`.
- Official base model `google/gemma-4-E2B-it`: executable source truth for `model_type=gemma4`,
  `131072` context, text layer schedule, and multimodal token ids.
- New repo-local `gemma4` runtime/model contract in `src/emel`: required because the family is
  distinct from current maintained architectures.
- Explicit reference-lane readiness work: the current pinned `llama.cpp` ref appears not to
  contain `gemma4`, while current upstream master does.

### Expected Features

The must-have surface is tight: one documented fixture, one explicit text-only conditioning
contract, one truthful `gemma4` runtime slice, explicit rejection of unsupported media/tool
surfaces, one reference-lane readiness step, one parity proof, regression protection for prior
anchors, and one benchmark/docs publication path for the same slice. Everything else should be
treated as future scope.

**Must have (table stakes):**
- Document one official `tests/models/gemma-4-e2b-it-Q8_0.gguf` fixture with URL, checksum, stable
  path, and provenance.
- Record metadata truth from official GGUF/config sources, specifically `gemma4`, `131072`, layer
  schedule, and the separate `mmproj` companion file.
- Add one explicit Gemma 4 text-only request-conditioning contract with structured text chat
  messages and `add_generation_prompt=true`.
- Bring up one maintained `gemma4` runtime slice that can initialize and generate through the
  shipped generator path.
- Reject `mmproj`, media, and tool-call request shapes explicitly on the maintained path.
- Make the pinned `llama.cpp` reference lane Gemma 4-capable.
- Prove the same slice with `tools/paritychecker --generation` and protect existing Llama, Qwen,
  and Liquid anchors with regression coverage.
- Publish the parity-backed Gemma 4 slice through `tools/bench` compare/docs.

**Defer (v2+):**
- `mmproj` plus image/audio/video input support.
- Tool use or function calling.
- `F16` or broader Gemma 4 fixture coverage.
- Gemma 4-specific performance optimization beyond what is needed for truthful first-slice proof.

### Architecture Approach

The architecture should stay inside the repo's existing maintained seams. `tests/models/README.md`
and tool constants own fixture identity; `tools/generation_formatter_contract.hpp` owns the
canonical Gemma 4 text contract; `src/emel/model/data.*` and the generator runtime own explicit
`gemma4` acceptance and topology handling; `tools/paritychecker` and `tools/bench` remain the only
maintained proof/publication boundary, with the added requirement that the reference pin itself is
managed explicitly.

### Critical Pitfalls

1. **False architecture readiness** — do not widen acceptance before an explicit `gemma4` runtime
   path exists in `src/emel`.
2. **False multimodal claim** — do not imply `mmproj` or media support just because the official
   repo ships those files.
3. **Template false positive** — do not reuse a generic contract; add one Gemma 4-specific
   text-only matcher.
4. **Reference pin drift** — do not assume the current pinned `llama.cpp` ref already supports
   Gemma 4.
5. **Benchmark claims before correctness** — keep Gemma 4 benchmark/docs publication behind
   reference readiness, parity, and regression proof.

## Implications for Roadmap

Based on research, the roadmap should continue from Phase 37 with five narrow phases.

### Phase 38: Fixture, Metadata, And Text-Only Contract Lock
**Delivers:** Official fixture provenance under `tests/models/`, requirement/constant updates that
name `gemma-4-e2b-it-Q8_0.gguf`, explicit metadata truth sourced from official GGUF/config, and one
Gemma 4 formatter contract bound to structured text messages only.

### Phase 39: `gemma4` Model Contract Bring-Up
**Delivers:** Explicit `gemma4` architecture acceptance in model metadata/tensor handling, Gemma 4
topology handling, and explicit rejection of unsupported `mmproj`/media/tool request shapes.

### Phase 40: Maintained Text Runtime Execution On ARM
**Delivers:** Generator/runtime bring-up for one maintained `gemma4` text slice and a truthful
runtime contract for the official `Q8_0` fixture only.

### Phase 41: Reference, Parity, And Regression Proof
**Delivers:** A Gemma 4-capable `llama.cpp` reference lane, parity proof for the maintained slice,
and regression protection for prior maintained anchors.

### Phase 42: Benchmark And Docs Publication
**Delivers:** One Gemma 4 case family in `tools/bench`, compare output, stored evidence, and docs
that clearly identify the maintained fixture and formatter contract.

## Sources

- https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF
- https://huggingface.co/api/models/ggml-org/gemma-4-E2B-it-GGUF
- https://huggingface.co/api/models/google/gemma-4-E2B-it
- https://huggingface.co/google/gemma-4-E2B-it/resolve/main/config.json
- https://huggingface.co/google/gemma-4-E2B-it/resolve/main/processor_config.json
- https://huggingface.co/google/gemma-4-E2B-it/resolve/main/chat_template.jinja
- https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp
- `tools/paritychecker/reference_ref.txt`
- `https://raw.githubusercontent.com/ggml-org/llama.cpp/ecbcb7ea9d3303097519723b264a8b5f1e977028/src/llama-model.cpp`
