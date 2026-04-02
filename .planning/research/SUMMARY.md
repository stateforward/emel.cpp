# Project Research Summary

**Project:** EMEL
**Domain:** Brownfield C++ GGUF inference engine adding one maintained 1-bit model slice
**Researched:** 2026-04-02
**Confidence:** HIGH

## Executive Summary

The Bonsai milestone is narrower and more concrete than the current Liquid planning scope. The
live Hugging Face artifact is one file, `Bonsai-1.7B.gguf`, and direct GGUF inspection confirms it
is a `qwen3` model with `Q1_0_g128` tensors, `32768` context, `28` blocks, and an embedded
chat-template that includes tool and thinking branches. That means EMEL does not need a new model
family for v1.10, but it does need a new native quantized operand path and a Bonsai-specific
formatter contract.

The recommended approach is: freeze the exact fixture and metadata first, bind one narrow Bonsai
chat contract from the embedded template, add native `Q1_0_g128` runtime support on the maintained
generator path, and use Prism's `llama.cpp` fork as the parity/benchmark reference lane until
upstream support exists. The main risks are fixture drift, filename/metadata mismatch, request
contract drift, and overclaiming support before the 1-bit hot path is real.

## Key Findings

### Recommended Stack

Bonsai should be treated as standard GGUF container support plus one new operand class. Reuse the
existing EMEL GGUF loader and `qwen3` topology path, but add a project-owned `Q1_0_g128` dtype,
kernel path, and generator/runtime auditing. Keep the reference fork contained to tooling only.

**Core technologies:**
- `prism-ml/Bonsai-1.7B-gguf`: official fixture source and truth anchor.
- Existing EMEL `qwen3` model path: correct topology family for the maintained Bonsai file.
- Native EMEL `Q1_0_g128` path: required for truthful runtime support.
- `PrismML-Eng/llama.cpp`: current truthful reference lane for parity and benchmark work.

### Expected Features

The first maintained Bonsai slice should stay narrow and explicit.

**Must have (table stakes):**
- One documented official `Bonsai-1.7B.gguf` fixture with checksum, size, URL, and stable path.
- One explicit Bonsai formatter contract derived from the embedded template.
- Explicit rejection of unsupported tool/thinking/raw request shapes.
- Native `Q1_0_g128` runtime support on the shipped generator path.
- Maintained parity against Prism's reference lane for the same fixture and contract.
- Regression protection plus benchmark publication for the same parity-backed slice.

**Should have (competitive):**
- Published formatter-contract metadata in parity and benchmark output.
- Bonsai-specific recommended decode preset.
- Fixture-drift detection against live Hugging Face truth.

**Defer (v2+):**
- Tool calling and function calling.
- Thinking replay or preservation support.
- Sibling Bonsai checkpoints or generic 1-bit support.
- New server/API surfaces for Bonsai workflows.

### Architecture Approach

Keep Bonsai on the existing `qwen3` architecture lane and isolate the widening to the quant path
and request contract. The major components are:

1. Fixture truth and metadata capture in `tests/models/README.md` and tool registries.
2. Bonsai-specific formatter binding in `tools/generation_formatter_contract.hpp`.
3. Native `Q1_0_g128` kernels and generator/runtime routing in `src/emel/kernel/*` and
   `src/emel/generator/detail.hpp`.
4. Tool-only parity/benchmark reference integration with `PrismML-Eng/llama.cpp`.

### Critical Pitfalls

1. **Fixture drift**: pin one exact file, SHA256, URL, and repo commit before runtime work.
2. **Filename/metadata mismatch**: keep `fixture_file=Bonsai-1.7B.gguf` separate from
   `weight_format=Q1_0_g128`.
3. **False runtime readiness**: GGUF parse success is not proof of native `Q1_0_g128` support.
4. **Wrong reference lane**: upstream `ggml-org/llama.cpp` is not the truthful Bonsai comparator
   today.
5. **Formatter drift**: Bonsai is Qwen-like but not identical to the repo's current supported Qwen
   contract.

## Implications for Roadmap

Based on research, the milestone should stay in five phases.

### Phase 38: Fixture Provenance And Metadata Truth
**Rationale:** Every later proof surface depends on one frozen file identity and one frozen GGUF
truth set.
**Delivers:** Maintained fixture provenance, GGUF metadata capture, and exact file/format naming.
**Addresses:** fixture identity, metadata truth, and filename mismatch.
**Avoids:** provenance drift and stale model-card assumptions.

### Phase 39: Bonsai Conditioning Contract
**Rationale:** Request-shape truth must be fixed before parity or runtime debugging means anything.
**Delivers:** Explicit Bonsai formatter contract plus rejection semantics for unsupported template
branches.
**Uses:** embedded `tokenizer.chat_template` from the maintained file.
**Implements:** formatter/conditioner integration.

### Phase 40: Native `Q1_0_g128` Runtime Bring-Up
**Rationale:** The quant path is the real milestone-defining work and must land before parity.
**Delivers:** Native dtype/kernel/runtime support and truthful generator routing.
**Uses:** existing `qwen3` model topology with new quantized operand support.
**Implements:** kernel, generator, and quant-audit changes.

### Phase 41: Parity And Regression Proof
**Rationale:** Correctness claims should only follow once the native Bonsai path exists.
**Delivers:** Parity against Prism's fork on the exact fixture and contract, plus regression
coverage for prior maintained anchors.

### Phase 42: Benchmark And Publication
**Rationale:** Performance/docs should only be published for a parity-backed maintained slice.
**Delivers:** One Bonsai benchmark/docs row aligned with the same fixture and contract.

### Phase Ordering Rationale

- Fixture truth comes first because later phases cannot be honest without one frozen artifact.
- Conditioning truth comes before runtime so prompt drift does not contaminate debugging.
- Native `Q1_0_g128` support must precede parity and benchmark claims.
- Benchmark publication remains last to avoid overclaiming speed before correctness is proven.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 40:** `Q1_0_g128` kernel port shape, because Prism's fork is the main public reference.
- **Phase 41:** reference-lane pinning details for Prism's fork.

Phases with standard patterns:
- **Phase 38:** fixture provenance and metadata capture.
- **Phase 39:** formatter-contract binding and negative request-shape proof.
- **Phase 42:** benchmark/docs publication on an existing tooling surface.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified against the live GGUF, live Hugging Face repo, Prism fork, and local repo surfaces |
| Features | HIGH | Narrow first-slice scope is well supported by artifact truth and repo patterns |
| Architecture | HIGH | The live GGUF confirms `qwen3`; the widening is clearly the 1-bit quant path |
| Pitfalls | HIGH | Primary risks are already visible from live artifact and repo mismatch points |

**Overall confidence:** HIGH

### Gaps To Address

- Exact EMEL implementation plan for `Q1_0_g128` kernels and audit reporting.
- Exact Prism fork commit to pin for parity/benchmark truth.
- Whether the maintained Bonsai contract should support assistant history in addition to
  `system,user` prompts on day one.

## Sources

### Primary
- `https://huggingface.co/prism-ml/Bonsai-1.7B-gguf`
- `https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/tree/main`
- `https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B.gguf`
- `https://github.com/PrismML-Eng/llama.cpp`
- Local artifact inspection: `/tmp/Bonsai-1.7B.gguf`

### Secondary
- `.planning/research/STACK.md`
- `.planning/research/FEATURES.md`
- `.planning/research/ARCHITECTURE.md`
- `.planning/research/PITFALLS.md`
- Local repo inspection:
  - `src/emel/model/data.cpp`
  - `src/emel/generator/detail.hpp`
  - `tools/generation_formatter_contract.hpp`
  - `tools/generation_fixture_registry.hpp`

---
*Research completed: 2026-04-02*
*Ready for roadmap: yes*
