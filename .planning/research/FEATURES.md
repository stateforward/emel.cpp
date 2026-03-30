# Feature Research

**Domain:** Canonical Qwen3-0.6B parity and benchmark slice on EMEL's maintained generation path
**Researched:** 2026-03-27
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users will reasonably expect if EMEL claims a maintained Qwen3-0.6B slice.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| One documented canonical Qwen3-0.6B GGUF fixture with provenance | The repo already treats local GGUF fixtures and `tests/models/README.md` as operator truth anchors | LOW | Must record file name, source repo, checksum, and download URL for the official artifact |
| Maintained runtime support for one `qwen3` architecture slice | The user asked for parity and benchmark against Qwen3-0.6B, not just tokenizer research | HIGH | Current EMEL runtime is still Llama-specific in `model::llama::detail`, `parity_runner`, and `generation_bench` |
| One explicit prompt-conditioning contract for that slice | Qwen3 is an instruct model and official docs route local use through chat templates | HIGH | Current maintained tools still inject `format_raw` and argmax selection; v1.6 needs one honest contract instead of implicit raw-text behavior |
| `tools/paritychecker --generation` proves EMEL vs `llama.cpp` on the same fixture and request contract | This repo defines runtime truth through the maintained parity surface | HIGH | The surface currently requires the canonical Llama fixture and rejects non-Llama architectures |
| `tools/bench` compare/docs publish the same Qwen3 slice | Benchmark claims need to stay aligned with the parity-checked runtime slice | MEDIUM | Reuse the maintained compare, snapshot, and docs flow instead of adding a Qwen-only benchmark harness |

### Differentiators (Competitive Advantage)

Helpful additions that improve operator trust, but are not required to call v1.6 done.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Dump or attribution output that names the Qwen fixture and conditioning mode | Makes debugging and publication review less ambiguous | MEDIUM | Best done additively on existing dump/compare text, not by creating a new tool surface |
| Additive benchmark slugs that clearly distinguish the Qwen canonical row from the existing Llama anchor | Keeps publication readable when both slices coexist | LOW | Prefer explicit fixture naming over changing the benchmark workflow |
| Regression coverage that keeps the shipped Llama slice green while Qwen comes up | Reduces risk that v1.6 breaks the prior milestone's truth anchor | MEDIUM | Important because the maintained surfaces are being widened, not replaced |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Broad Qwen family rollout in v1.6 | It sounds stronger to claim Qwen support generally | Qwen3, Qwen3.5, Qwen3Next, and MoE variants are not the same topology or conditioning contract | Keep v1.6 to one canonical Qwen3-0.6B fixture |
| Community quant matrix for the first milestone | Smaller files or faster local iteration are attractive | It weakens provenance and makes parity/benchmark claims harder to interpret | Start with the official Qwen3-0.6B GGUF artifact only |
| Raw `hello` prompt parity presented as official Qwen behavior | It is the smallest code delta because the tools already use raw formatting | Official Qwen docs point to chat-template use, and default thinking-mode behavior is a poor fit for greedy decoding | Define one explicit canonical conditioning contract and use it consistently |
| Architecture aliasing that just treats `qwen3` as `llama` | It looks like the fastest path to get the tool past validation | Local reference source shows Qwen3 has distinct attention-normalization tensors, so a string alias can produce false readiness | Add explicit Qwen3 runtime support or keep the architecture rejected |
| Benchmark-only Qwen publication without parity | Bench output is visible and tempting to land first | It creates performance claims for a slice whose correctness and conditioning contract are still unresolved | Make parity and tests precede benchmark publication |

## Feature Dependencies

```text
[Canonical Qwen fixture provenance]
    └──requires──> [tests/models/README and maintained fixture constants]

[Runtime support for one qwen3 slice]
    ├──requires──> [Canonical Qwen fixture provenance]
    ├──requires──> [Architecture-aware model/execution-view support]
    └──requires──> [Explicit prompt-conditioning contract]

[Paritychecker generation proof]
    ├──requires──> [Runtime support for one qwen3 slice]
    └──requires──> [Reference path uses the same fixture and conditioning contract]

[Maintained benchmark publication]
    ├──requires──> [Paritychecker generation proof]
    └──requires──> [Existing compare/docs workflow updated for the same Qwen slice]

[Broad Qwen-family claims]
    ──conflicts──> [Narrow v1.6 truth anchor]
```

### Dependency Notes

- **Runtime support requires architecture-aware model support:** local `llama.cpp` reference code treats `qwen3` as a distinct architecture with extra attention-normalization tensors, so this milestone cannot be closed by changing a fixture constant alone.
- **Runtime support requires an explicit conditioning contract:** the current raw formatter path is a repo convenience, not an automatically valid Qwen3 operator contract.
- **Benchmark publication depends on parity first:** benchmark rows should represent the same model file and request-conditioning contract that already passed parity.
- **Broad Qwen-family claims conflict with the milestone shape:** each extra family member multiplies topology, tokenizer, and publication questions before the first slice is trustworthy.

## MVP Definition

### Launch With (v1)

- [ ] One official `Qwen3-0.6B-Q8_0.gguf` fixture is documented in `tests/models/README.md`
  - Essential because fixture identity is part of the maintained truth surface.
- [ ] One explicit canonical request-conditioning contract is documented and used in both EMEL and `llama.cpp`
  - Essential because Qwen3 output behavior depends on prompt formatting and thinking-mode handling.
- [ ] EMEL can initialize and generate on that one Qwen3-0.6B fixture through the maintained generator path
  - Essential because the milestone is runtime bring-up, not tool-only proof.
- [ ] `tools/paritychecker --generation` proves EMEL against `llama.cpp` for that slice
  - Essential because paritychecker is the accepted correctness gate.
- [ ] `tools/bench` compare/docs publish the same canonical Qwen3-0.6B slice
  - Essential because benchmark claims must stay aligned with the parity-checked runtime.

### Add After Validation (v1.x)

- [ ] Additive dump or attribution reporting that names the Qwen fixture and conditioning mode
  - Add after the first slice works, to reduce future debugging cost.
- [ ] Additional Qwen prompts or token budgets on the same canonical fixture
  - Add only after the base contract is stable and benchmark interpretation is clear.
- [ ] Stronger regression coverage that keeps Llama and Qwen maintained slices green together
  - Add once the repo starts treating both as first-class maintained anchors.

### Future Consideration (v2+)

- [ ] Qwen3.5, Qwen3Next, or MoE model families
  - Defer because they are not guaranteed to share the same topology or conditioning details.
- [ ] Broader quant matrices for Qwen3-0.6B
  - Defer until the official truth anchor is stable.
- [ ] Public API support for richer Qwen chat or tool-calling behavior
  - Defer because v1.6 is scoped to maintained parity and benchmark surfaces only.
- [ ] Qwen-specific performance optimization milestones
  - Defer until runtime correctness and publication are already honest.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Canonical fixture provenance | HIGH | LOW | P1 |
| One explicit conditioning contract | HIGH | HIGH | P1 |
| One maintained `qwen3` runtime slice | HIGH | HIGH | P1 |
| Paritychecker generation proof | HIGH | HIGH | P1 |
| Maintained benchmark publication | HIGH | MEDIUM | P1 |
| Fixture/mode attribution in dump or compare output | MEDIUM | MEDIUM | P2 |
| Dual-slice regression coverage (Llama + Qwen) | MEDIUM | MEDIUM | P2 |
| Broad Qwen-family rollout | MEDIUM | HIGH | P3 |
| Quant matrix expansion | LOW | HIGH | P3 |
| Public chat/tool-calling API work | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Repo Surface Analysis

| Feature | Existing Surface | Current State | Our Approach |
|---------|------------------|---------------|--------------|
| Canonical fixture provenance | `tests/models/README.md` | No Qwen3 fixture is documented today | Add one official Qwen3-0.6B fixture entry with checksum and source |
| Prompt conditioning | `emel::text::conditioner` + formatter injection | Maintained tools still use `format_raw` | Define one explicit Qwen3 conditioning contract and wire it through the existing seam |
| Runtime architecture gate | `src/emel/model/data.cpp`, `tools/paritychecker`, `tools/bench` | Runtime still validates only `llama` | Add truthful support for one `qwen3` slice or keep it rejected until ready |
| Parity acceptance | `tools/paritychecker --generation` | Hard-coded to the canonical Llama fixture and request shape | Extend it narrowly to one canonical Qwen3 slice without broadening the CLI surface |
| Benchmark publication | `tools/bench` compare/docs flow | Hard-coded to the canonical Llama fixture and request shape | Reuse the same flow for one Qwen3 slice after parity is real |

## Sources

- `.planning/PROJECT.md` - milestone goal, active scope, and out-of-scope boundaries.
- `AGENTS.md` - repo rules for parity claims, fallback restrictions, and milestone scope discipline.
- `tests/models/README.md` - existing fixture provenance pattern.
- `tools/paritychecker/parity_main.cpp` and `tools/paritychecker/parity_runner.cpp` - current generation surface, fixture lock, raw formatting, and Llama-only architecture gate.
- `tools/bench/generation_bench.cpp` - current benchmark fixture lock, raw formatting, and Llama-only architecture gate.
- `src/emel/text/formatter/format.hpp` and `src/emel/text/formatter/sm.hpp` - existing formatter seam and chat-formatting design intent.
- `build/paritychecker/_deps/reference_impl-src/src/llama-arch.cpp` - local reference source showing `qwen3` as a distinct architecture.
- https://huggingface.co/Qwen/Qwen3-0.6B - official model card.
- https://huggingface.co/Qwen/Qwen3-0.6B-GGUF - official GGUF model card.
- https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html - official Qwen `llama.cpp` guidance.

---
*Feature research for: EMEL v1.6 Qwen3-0.6B parity and benchmark*
*Researched: 2026-03-27*
