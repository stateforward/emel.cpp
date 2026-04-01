# Project Research Summary

**Project:** EMEL
**Domain:** Brownfield C++ GGUF inference engine adding one maintained LiquidAI `LFM2.5-1.2B-Thinking-GGUF` ARM slice
**Researched:** 2026-03-31
**Confidence:** MEDIUM

## Executive Summary

v1.9 is not "Liquid support." It is one narrow maintained ARM slice for the official
`LiquidAI/LFM2.5-1.2B-Thinking-GGUF` artifact
`LFM2.5-1.2B-Thinking-Q4_K_M.gguf`, proven through EMEL's existing generator, paritychecker, and
benchmark surfaces. Experts would build this as a truth-anchor milestone: pin one official file,
bind one explicit chat-format contract, add the missing `lfm2` runtime path in `src/emel`, then
publish parity and benchmark evidence only for that exact slice.

The key implementation recommendation is to treat official GGUF/config metadata as the executable
truth source. For this milestone, that means `general.architecture=lfm2` and context length
`128000` from the official GGUF/config report override stale prose that still mentions `32,768`.
That truth source drives the roadmap: Phase 33 should lock fixture identity, metadata truth, and
the Liquid-specific formatter contract; Phases 34 and 35 should add explicit `lfm2` model/runtime
support without aliasing it to `llama` or `qwen3`; Phases 36 and 37 should then prove and publish
the same slice.

The main risks are false readiness and silent scope creep. The repo can look Liquid-capable while
still only understanding `llama`/`qwen3`, it can accept the wrong template by reusing the Qwen
matcher, or it can accidentally broaden into a quant matrix because the official repo publishes
many siblings. Mitigation is straightforward: name one maintained file everywhere, keep the
formatter contract fixed to `tools=none` and `keep_past_thinking=false`, require explicit `lfm2`
runtime handling, and keep benchmark publication behind parity and regression proof.

## Key Findings

### Recommended Stack

The stack recommendation is still narrow, but it is no longer conservative on quantization. No new
serving framework, model runtime, or benchmark harness is needed. The correct stack is one
official Liquid GGUF fixture, the current pinned `llama.cpp` reference commit that already
understands `lfm2`, and one repo-local Liquid formatter binding added to the existing maintained
conditioning seam, plus whatever new quantized runtime support is required for the user-selected
`Q4_K_M` anchor.

The important constraint is still that the milestone proves one exact slice, not a quant matrix.
But the user explicitly changed that exact slice to `Q4_K_M`, so the milestone now includes the
quant-runtime work needed to make that claim truthful on ARM.

**Core technologies:**
- Official `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` with `LFM2.5-1.2B-Thinking-Q4_K_M.gguf`: maintained
  fixture truth anchor for v1.9.
- Current `ggml-org/llama.cpp` pin `ecbcb7ea9d3303097519723b264a8b5f1e977028`: parity and
  benchmark reference that already supports `LLM_ARCH_LFM2` and `tokenizer_pre == "lfm2"`.
- New v1.9 quant-runtime work for the user-selected `Q4_K_M` anchor: required to keep the
  maintained claim truthful on ARM.
- New repo-local Liquid formatter binding in `tools/generation_formatter_contract.hpp`: makes the
  official Liquid ChatML-style request contract explicit and auditable.

### Expected Features

The must-have surface is tight: one documented fixture, one explicit conditioning contract, one
truthful `lfm2` runtime slice, one parity proof, regression protection for the prior Llama/Qwen
anchors, and one benchmark/docs publication path for the same slice. Everything else should be
treated as future scope.

**Must have (table stakes):**
- Document one official `tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture with URL, checksum,
  stable path, and provenance.
- Record metadata truth from official GGUF/config sources, specifically `lfm2` and `128000`
  context, and make that override stale prose.
- Add one explicit Liquid request-conditioning contract with structured chat messages,
  `add_generation_prompt=true`, `tools=none`, and `keep_past_thinking=false`.
- Bring up one maintained `lfm2` runtime slice that can initialize and generate through the
  shipped generator path.
- Prove the same slice with `tools/paritychecker --generation` and protect existing Llama/Qwen
  anchors with regression coverage.
- Publish the parity-backed Liquid slice through `tools/bench` compare/docs.

**Should have (competitive):**
- Publish formatter-contract metadata in parity and benchmark output so reviewers can verify the
  maintained Liquid contract was actually used.
- Add explicit negative coverage for unsupported Liquid asks such as sibling quants or unsupported
  template variants.
- Keep case names and output slugs readable across the three maintained anchors: Llama, Qwen, and
  Liquid.

**Defer (v2+):**
- Any sibling Liquid quant such as `Q4_0`, `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16`, or `F16`.
- Broad Liquid-family support beyond `LFM2.5-1.2B-Thinking-Q4_K_M.gguf`.
- Tool use, function calling, or multi-turn thinking-history replay.
- Generic Jinja/template execution or new Liquid-specific public API surfaces.

### Architecture Approach

The architecture should stay inside the repo's existing maintained seams. `tests/models/README.md`
and tool constants own fixture identity; `tools/generation_formatter_contract.hpp` owns the
canonical Liquid contract; `src/emel/model/data.*` and the generator runtime own explicit `lfm2`
acceptance and tensor-topology handling; `tools/paritychecker` and `tools/bench` remain the only
maintained proof/publication boundary. The critical architectural rule is explicitness: add a
dedicated `lfm2` path for the hybrid 16-layer topology, including conv-style layers, attention
layers, `shortconv_l_cache=3`, tied embeddings, and the `token_embd_norm` output-norm mapping.

**Major components:**
1. `tests/models/README.md` plus tool constants and slugs: pin one official fixture and its
   metadata truth.
2. `tools/generation_formatter_contract.hpp` plus text conditioning/tokenizer seams: resolve one
   canonical Liquid prompt contract.
3. `src/emel/model/data.*` plus generator runtime execution path: accept `lfm2` explicitly and
   execute the maintained Liquid slice truthfully.
4. `tools/paritychecker` and `tools/bench`: prove and publish only the same fixture and contract
   after runtime support is real.

### Critical Pitfalls

1. **False architecture readiness** — do not widen acceptance before an explicit `lfm2` runtime
   path exists in `src/emel`.
2. **Template false positive** — do not reuse the Qwen matcher; add one Liquid-specific
   maintained contract only.
3. **Metadata drift** — treat official GGUF/config values `lfm2` and `128000` as truth, and
   document that they override stale prose.
4. **Silent quant scope creep** — name `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` everywhere so the
   milestone does not turn into broad quant-matrix support.
5. **Benchmark claims before correctness** — keep Liquid benchmark/docs publication behind parity
   and regression proof.

## Implications for Roadmap

Based on research, the roadmap should continue from Phase 32 with five narrow phases.

### Phase 33: Fixture, Metadata, And Contract Lock
**Rationale:** Phase 33 defines what the milestone actually means before runtime work starts.
**Delivers:** Official fixture provenance under `tests/models/`, requirement/constant updates that
name `LFM2.5-1.2B-Thinking-Q4_K_M.gguf`, explicit metadata truth sourced from official GGUF/config,
and one Liquid formatter contract bound to `tools=none`, `add_generation_prompt=true`, and
`keep_past_thinking=false`.
**Addresses:** `FIX-02`, `META-01`, `COND-03`.
**Avoids:** Metadata drift, template false positives, and silent quant scope creep.

### Phase 34: `lfm2` Model Contract Bring-Up
**Rationale:** The repo must reject false architecture aliases before it can claim Liquid runtime
   support.
**Delivers:** Explicit `lfm2` architecture acceptance in model metadata/tensor handling, Liquid
tensor-name mapping including `token_embd_norm`, and a truthful model/execution-view contract for
the hybrid block layout.
**Uses:** Current `src/emel/model/data.*` seams and the existing `llama.cpp` reference pin as the
truth reference.
**Implements:** The architecture boundary for `RUN-03` and part of `RUN-05`.
**Avoids:** False architecture readiness.

### Phase 35: Maintained Runtime Execution On ARM
**Rationale:** After the model contract is real, EMEL still needs a shipped generator path that can
initialize and decode the maintained Liquid slice on ARM using the existing native `q8_0` surface.
**Delivers:** Generator/runtime bring-up for one maintained `lfm2` slice, explicit handling of the
hybrid conv/attention topology, and a truthful quantized-path contract for the official `Q4_K_M`
fixture only.
**Addresses:** `RUN-04`, the remainder of `RUN-05`, and `RUN-06`.
**Avoids:** Hidden Llama/Qwen assumptions and accidental broad quant claims.

### Phase 36: Parity And Regression Proof
**Rationale:** The milestone is not maintainable until the exact slice is proven against
`llama.cpp` and the prior maintained anchors stay green.
**Delivers:** `tools/paritychecker --generation` coverage for the Liquid fixture with the same
formatter contract, stored decode-length proof, attribution and dump visibility, and regression
coverage for Llama, Qwen, and Liquid maintained anchors.
**Addresses:** `PAR-02` and `VER-02`.
**Avoids:** Benchmark-first publication and correctness claims without verification.

### Phase 37: Benchmark And Docs Publication
**Rationale:** Benchmarks should only publish a slice that is already parity-backed and regression-
protected.
**Delivers:** One Liquid case family in `tools/bench`, compare output, stored evidence, and docs
that clearly identify the maintained fixture and formatter contract.
**Addresses:** `BENCH-08`.
**Avoids:** Unbacked performance claims or publication drift.

### Phase Ordering Rationale

- Phase 33 comes first because fixture identity, metadata truth, and prompt contract define the
  exact maintained acceptance surface.
- Phase 34 precedes Phase 35 because the repo needs an explicit `lfm2` model/tensor contract
  before generator execution can be honest.
- Phase 35 is isolated from parity/bench so runtime bring-up can prove real ARM execution without
  conflating it with publication.
- Phase 36 must complete before Phase 37 so every published Liquid result is tied to maintained
  correctness proof.
- This grouping matches the research dependency chain: fixture and contract -> architecture ->
  runtime -> parity/regression -> benchmark/docs.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 34:** The `lfm2` hybrid topology, per-layer conv vs attention handling, and
  architecture-specific tensor naming are the least standard part of the milestone.
- **Phase 35:** Native runtime execution needs careful validation that the existing ARM `q8_0`
  path covers the required Liquid operator mix without slipping into fallback behavior.

Phases with standard patterns (skip research-phase):
- **Phase 33:** This can mostly reuse the v1.6 Qwen fixture/conditioning pattern if the scope
  stays narrow to one Liquid formatter binding and one official fixture.
- **Phase 36:** Parity and regression use established maintained repo patterns once the runtime
  slice exists.
- **Phase 37:** Benchmark compare/docs publication is already a standard repo workflow.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official Liquid/GGUF sources and the current repo/reference seams point to one clear bounded stack. |
| Features | HIGH | The milestone boundary is explicit: one maintained `Q4_K_M` Liquid Thinking slice on ARM. |
| Architecture | MEDIUM | The need for explicit `lfm2` support is clear, but the hybrid runtime contract is materially different from current `llama`/`qwen3` paths. |
| Pitfalls | HIGH | The main failure modes are concrete, repo-specific, and already reflected in the research and requirements. |

**Overall confidence:** MEDIUM

### Gaps to Address

- **Exact runtime/operator mapping for the hybrid `lfm2` block contract:** validate during Phase 34
  planning against the current pinned `llama.cpp` reference and EMEL-owned runtime code.
- **Native ARM `Q4_K_M` coverage for Liquid-specific execution paths:** verify during Phase 35 so
  the milestone does not accidentally depend on an unproven backend path.
- **Benchmark expectations after parity:** keep performance claims narrow until Phase 36 proves the
  same fixture and contract end to end.

## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md` — v1.9 goal, active requirements, and explicit out-of-scope boundary.
- `.planning/research/STACK.md` — official fixture choice, reference pin, metadata truth source,
  and stack recommendations.
- `.planning/research/FEATURES.md` — table stakes, differentiators, dependency ordering, and
  milestone scoping.
- `.planning/research/ARCHITECTURE.md` — maintained seam boundaries, explicit `lfm2` contract, and
  phase-level architecture patterns.
- `.planning/research/PITFALLS.md` — top failure modes and their phase mapping.
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF — official GGUF identity and maintained
  fixture source.
- https://huggingface.co/api/models/LiquidAI/LFM2.5-1.2B-Thinking-GGUF — official GGUF metadata
  including architecture and context reporting.
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/raw/main/config.json — official config truth
  for `model_type=lfm2`, `max_position_embeddings=128000`, and layer topology.
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/raw/main/tokenizer_config.json — official
  special-token and chat-template inputs.
- https://docs.liquid.ai/lfm/key-concepts/chat-template — official Liquid chat-format guidance.
- https://docs.liquid.ai/deployment/on-device/llama-cpp — official Liquid guidance that GGUF via
  `llama.cpp` is the supported on-device path.

### Secondary (MEDIUM confidence)
- Current repo files under `src/emel/model`, `src/emel/generator`, `tools/paritychecker`,
  `tools/bench`, and `tools/generation_formatter_contract.hpp` — confirmed the maintained seams
  the milestone should reuse rather than replace.
- Current pinned `ggml-org/llama.cpp` sources referenced in the research set — confirmed that the
  existing pin already understands `lfm2` metadata and tokenizer pre-processing.

### Tertiary (LOW confidence)
- None.

---
*Research completed: 2026-03-31*
*Ready for roadmap: yes*
