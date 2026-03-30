# Project Research Summary

**Project:** EMEL
**Domain:** Canonical Qwen3-0.6B parity and benchmark slice on maintained EMEL surfaces
**Researched:** 2026-03-27
**Confidence:** MEDIUM-HIGH

## Executive Summary

v1.6 should be treated as a narrow brownfield milestone, not as generic "Qwen support." The
official Qwen project already publishes an official `Qwen3-0.6B-Q8_0.gguf`, and official Qwen docs
document `llama.cpp` usage for Qwen3. That gives EMEL a clean truth anchor for one maintained
slice. The problem is inside EMEL: maintained generation, parity, and benchmark surfaces are still
hard-wired to the canonical Llama fixture, the runtime still depends on `model::llama::detail`,
and current tools still inject raw formatting into an argmax generation path.

The recommended approach is therefore strict sequencing. First, lock the milestone to one official
Qwen3-0.6B fixture and one explicit prompt-conditioning contract. Second, teach the runtime enough
about one `qwen3` slice to execute it honestly. Third, prove that slice through the maintained
parity surface. Only after those steps are real should `tools/bench`, compare snapshots, and docs
be refreshed.

The main risks are false readiness and false publication. The repo can easily look "Qwen-started"
while still rejecting `qwen3`, or it can benchmark a request contract that is not aligned with
official Qwen usage. Mitigation is to keep the milestone narrow, keep claims tied to one official
fixture, and keep parity ahead of publication.

## Key Findings

### Recommended Stack

Research strongly favors staying inside the existing EMEL toolchain and runtime seams. No new
runtime dependency or public API surface is needed. The right stack is one official Qwen3 GGUF
fixture, the current EMEL generator and text-conditioning stack, and the existing `llama.cpp`
reference path already used by paritychecker and bench.

**Core technologies:**
- Official `Qwen/Qwen3-0.6B-GGUF` fixture `Qwen3-0.6B-Q8_0.gguf`: canonical v1.6 truth anchor.
- Existing formatter/conditioner/Jinja seam in `src/emel/text/**`: preferred place to make the
  Qwen request contract explicit.
- Existing EMEL runtime in `src/emel/model` and `src/emel/generator`: the actual bring-up surface.
- Existing `llama.cpp` CPU reference path: maintained parity and benchmark comparison boundary.

### Expected Features

The table stakes are modest but strict: one official fixture, one explicit request-conditioning
contract, one maintained runtime slice, one parity proof, and one benchmark publication flow for
that exact slice. Anything broader should be deferred.

**Must have (table stakes):**
- Document one official Qwen3-0.6B GGUF fixture with checksum and provenance.
- Define one explicit canonical request-conditioning contract.
- Bring up one maintained `qwen3` runtime slice in EMEL.
- Prove that slice with `tools/paritychecker --generation`.
- Publish the same slice through `tools/bench` compare/docs.

**Should have (competitive):**
- Additive attribution or dump output that names the Qwen fixture and conditioning mode.
- Regression coverage that keeps the prior Llama anchor green while Qwen comes up.

**Defer (v2+):**
- Qwen3.5, Qwen3Next, MoE, or quant-matrix expansion.
- Public chat/tool-calling API work.
- Qwen-specific optimization milestones.

### Architecture Approach

The architecture recommendation is conservative: keep the generator actor graph intact, use the
existing formatter/conditioner seam for prompt shaping, and add Qwen support at the model/execution
view boundary instead of pretending `qwen3` is already Llama-compatible. Local reference source
shows `qwen3` is a distinct architecture and includes attention-normalization tensors beyond the
current Llama tensor set, so a string alias is not enough.

**Major components:**
1. Fixture and tool constants in `tests/models/README.md`, `tools/paritychecker`, and `tools/bench`.
2. Prompt conditioning via `src/emel/text/formatter`, `text/conditioner`, and optionally Jinja.
3. Runtime model/execution-view support in `src/emel/model/data.*` and `src/emel/generator`.
4. Existing paritychecker and bench surfaces as the only maintained acceptance boundary.

### Critical Pitfalls

1. **Changing only the fixture name while the repo still rejects `qwen3`** — avoid by treating
   fixture, architecture gate, and runtime support as one unit of work.
2. **Treating Qwen3 as a Llama alias** — avoid by adding explicit runtime support where
   `model::llama::detail` assumptions currently live.
3. **Shipping a misleading prompt contract** — avoid by making the request contract explicit
   instead of inheriting `format_raw`.
4. **Letting thinking-mode behavior leak into an argmax benchmark path** — avoid by choosing one
   deterministic canonical contract for both EMEL and `llama.cpp`.
5. **Refreshing publication before parity is real** — avoid by keeping parity ahead of bench/docs.

## Implications for Roadmap

Based on research, the roadmap should split into four narrow phases with dependencies made explicit.

### Phase 26: Canonical Fixture And Conditioning Contract
**Rationale:** The repo needs one official fixture and one explicit request contract before any
runtime work can be claimed honestly.
**Delivers:** Official Qwen3-0.6B fixture provenance, maintained tool constants/slugs, and a
documented canonical request-conditioning contract.
**Addresses:** Fixture truth-anchor and prompt-contract uncertainty.
**Avoids:** False readiness, raw-format drift, and thinking-mode ambiguity.

### Phase 27: Runtime Architecture Bring-Up
**Rationale:** Current EMEL runtime still depends on Llama-only model detail helpers.
**Delivers:** Enough `qwen3` architecture support to initialize and run one canonical Qwen3-0.6B
slice through the maintained generator path.
**Uses:** Existing `src/emel/model` and `src/emel/generator` seams.
**Implements:** The real runtime support boundary, not just tool acceptance.

### Phase 28: Parity And Verification
**Rationale:** Benchmark claims are meaningless until the maintained parity surface proves the same
slice.
**Delivers:** `tools/paritychecker --generation` support for the canonical Qwen3 slice, failing
tests turned green, and enough attribution/debug visibility to prove the right path is running.
**Avoids:** Runtime claims without correctness evidence.

### Phase 29: Benchmark Publication
**Rationale:** Publication should come last so compare/docs evidence represent the already-proven
slice.
**Delivers:** `tools/bench` compare/docs refresh for the canonical Qwen3 slice on the maintained
workflow.
**Addresses:** The user's benchmark request without widening the acceptance boundary.
**Avoids:** Unbacked performance claims and fixture/publication drift.

### Phase Ordering Rationale

- Phase 26 comes first because fixture identity and request contract define what the milestone even
  means.
- Phase 27 comes before tooling because EMEL runtime support must be real before the tools can say
  the model is supported.
- Phase 28 comes before benchmark publication because parity is the repo's correctness truth.
- Phase 29 is last because compare/docs are downstream evidence, not discovery tools.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 27:** Qwen3 tensor topology and attention-normalization handling may need focused runtime
  design, because local reference source shows differences from current Llama assumptions.
- **Phase 26:** the exact conditioning contract may need careful planning if chat-template metadata
  is not yet available in EMEL's model metadata path.

Phases with standard patterns (skip research-phase):
- **Phase 28:** parity and regression integration follow established repo patterns once the runtime
  slice exists.
- **Phase 29:** benchmark publication should reuse the existing compare/docs workflow.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Official Qwen sources and current repo seams point to a clear narrow approach. |
| Features | HIGH | The milestone boundary is explicit and intentionally small. |
| Architecture | MEDIUM | Runtime support is clearly needed, but exact Qwen3 topology work may be non-trivial because current EMEL is Llama-shaped. |
| Pitfalls | HIGH | Risks are concrete, repo-specific, and already visible in current maintained surfaces. |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- Whether EMEL currently loads enough chat-template metadata to support the preferred Qwen request
  contract, or whether that metadata path must be added in v1.6.
- The exact runtime handling required for Qwen3-specific attention-normalization tensors.
- The exact tokenizer-pre mapping reported by the official Qwen3 GGUF through current `llama.cpp`
  reference builds.

## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md` - v1.6 scope and acceptance boundary.
- `.planning/research/STACK.md`, `.planning/research/FEATURES.md`,
  `.planning/research/ARCHITECTURE.md`, and `.planning/research/PITFALLS.md` - milestone-specific
  research inputs.
- `tools/paritychecker/parity_runner.cpp` and `tools/bench/generation_bench.cpp` - current Llama
  fixture lock, raw formatting, and architecture gates.
- `src/emel/model/data.cpp` and `src/emel/generator/detail.hpp` - current runtime support boundary.
- `build/paritychecker/_deps/reference_impl-src/src/llama-arch.cpp` and
  `build/paritychecker/_deps/reference_impl-src/src/llama-model.cpp` - local reference evidence
  that Qwen3 is a distinct architecture with its own tensor/hparam expectations.
- https://huggingface.co/Qwen/Qwen3-0.6B - official model card.
- https://huggingface.co/Qwen/Qwen3-0.6B-GGUF - official GGUF model card.
- https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html - official Qwen `llama.cpp`
  guide.

### Secondary (MEDIUM confidence)
- `src/emel/text/formatter/sm.hpp` - design intent for chat-formatting support through the existing
  formatter seam.

### Tertiary (LOW confidence)
- None.

---
*Research completed: 2026-03-27*
*Ready for roadmap: yes*
