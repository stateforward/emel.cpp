# Feature Research

**Domain:** Single maintained `LiquidAI/LFM2.5-1.2B-Thinking-GGUF` ARM slice on EMEL's existing
generation, parity, and benchmark surfaces
**Researched:** 2026-03-31
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features EMEL needs if it claims one truthful maintained Liquid Thinking slice.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| One documented official Liquid GGUF fixture with provenance | EMEL already treats one stable `tests/models/` artifact as the truth anchor for each maintained slice | LOW | Lock one exact official filename, stable path, SHA256, and download URL before any runtime work. The maintained anchor is now `LFM2.5-1.2B-Thinking-Q4_K_M.gguf`, because the user explicitly reprioritized the milestone to that docs-recommended quant. |
| One explicit conditioning contract derived from Liquid's official chat format | The official model card says LFM2.5 uses a ChatML-like format, not free-form raw prompting, and separately documents tool use | MEDIUM | Keep the maintained path on one structured chat-message contract, `add_generation_prompt=true`, `tools=none`, and a fixed thinking-mode choice. Do not allow implicit raw fallback on the maintained slice. |
| Explicit `lfm2` runtime bring-up on the maintained generator path | LiquidAI publishes this GGUF as architecture `lfm2`, and the native model card describes a hybrid 16-layer topology, not a Llama/Qwen clone | HIGH | v1.9 is not done if EMEL aliases `lfm2` to an existing family. The milestone needs truthful model acceptance, loading, and generation for this exact architecture. |
| Quantized runtime contract proof for the chosen fixture only | A maintained ARM slice implies the shipped runtime can actually execute the pinned official artifact | HIGH | Publish native-vs-approved contract evidence for the chosen file. Do not imply support for sibling official quants just because they exist in the repo on Hugging Face. |
| Maintained parity proof against `llama.cpp` using the same fixture and conditioning | EMEL's correctness claims live on `tools/paritychecker --generation`, not on ad hoc manual runs | HIGH | Match the existing maintained pattern: same GGUF file, same prompt contract, and the usual stored decode-length coverage such as `1/10/100/1000` plus `--dump` and `--attribution`. |
| Regression protection for existing maintained anchors while Liquid lands | Brownfield widenings are expected to preserve the shipped Llama and canonical Qwen proof, not trade one truth anchor for another | MEDIUM | Coverage should catch fixture drift, conditioning drift, architecture misclassification, and accidental regressions on the prior maintained slices. |
| Benchmark compare/docs publication for the exact parity-backed Liquid slice | EMEL already publishes benchmark evidence only when it maps to a maintained, correctness-backed slice | MEDIUM | Add one Liquid row to the existing compare/docs flow after parity is green. Keep fixture naming, contract naming, and attribution readable in published output. |

### Differentiators (Competitive Advantage)

Helpful additions that make the Liquid bring-up more trustworthy and easier to operate, but are not
strictly required to call v1.9 complete.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Published formatter-contract metadata on parity and benchmark outputs | Makes every maintained claim auditable: reviewers can see the slice was run with the intended structured contract instead of a silent raw-text shortcut | LOW | Reuse the existing `generation_formatter_contract` publication seam and make the Liquid row explicit about `tools=none` and the chosen thinking-mode setting. |
| Explicit negative proof for unsupported Liquid asks | Truthfulness improves when the repo proves what it rejects, not only what it accepts | MEDIUM | Add failure coverage for wrong Liquid fixture paths, unsupported Liquid family members, unsupported quantizations, or unsupported template variants instead of silently broadening acceptance. |
| Stable three-anchor compare readability | Once Liquid joins Llama and Qwen, operators need publication rows that stay easy to distinguish and review over time | LOW | Use family/variant/fixture-specific slugs so docs and stored compare output do not blur the maintained anchors together. |
| Documented fixture-choice rationale | Liquid's docs generally recommend `Q4_K_M`, and the user wants that as the maintained truth anchor | LOW | Recording why v1.9 picked `Q4_K_M` makes the broadened quant-runtime scope explicit instead of accidental. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Broad "Liquid support" in v1.9 | It sounds stronger to say the repo supports Liquid generally | `LFM2.5-1.2B-Thinking`, `Instruct`, `Base`, `JP`, `VL`, `Audio`, and larger families are not the same acceptance surface, topology, or prompt contract | Keep v1.9 fixed to one maintained `LFM2.5-1.2B-Thinking-GGUF` slice only |
| All official GGUF quantizations for the first milestone | The official GGUF repo exposes `Q4_0`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `BF16`, and `F16`, so broad coverage is tempting | EMEL does not automatically inherit support for every official quant. Claiming the whole ladder would turn one slice bring-up into a quant-matrix milestone | Pin one exact official file first. v1.9 now names `Q4_K_M`; all other quants stay out of scope until separately proven |
| Tool use or function calling as part of v1.9 | The official model card documents tool use, so it feels close at hand | Tool schemas, tool-role replay, assistant call serialization, and any new CLI/API request surfaces are separate product work from first-slice truth | Keep the maintained contract at `tools=none`; defer tool use to a later milestone after the slice is already parity-backed |
| Raw prompt fallback or broad prompt-control knobs on the maintained path | Reusing a raw text lane or open-ended prompt controls looks like the fastest way to get output | The official model card points to a ChatML-like template, and broadening prompt controls makes parity and benchmark results harder to interpret | Keep one fixed structured chat-message contract and reject unsupported request shapes explicitly |
| Benchmark publication before parity and regression are green | Bench output is visible and easy to demo | It creates speed claims for a slice whose correctness or contract is still unresolved | Land parity and regression protection first, then publish benchmark evidence for the same slice |
| Aliasing `lfm2` to an existing architecture for speed | It reduces apparent bring-up work | The official model is explicitly `lfm2` and described as a hybrid architecture, so aliasing would create false readiness | Add explicit `lfm2` runtime handling or keep the architecture unsupported until it is real |

## Feature Dependencies

```text
[One official Liquid fixture with provenance]
    └──requires──> [Stable maintained path and checksum]

[One explicit Liquid conditioning contract]
    ├──requires──> [One official Liquid fixture with provenance]
    └──requires──> [Official GGUF chat-template inspection]

[Explicit lfm2 runtime bring-up]
    ├──requires──> [One official Liquid fixture with provenance]
    └──requires──> [Architecture-specific execution/load support]

[Quantized runtime contract proof]
    ├──requires──> [Explicit lfm2 runtime bring-up]
    └──requires──> [Chosen official quantized fixture]

[Maintained parity proof]
    ├──requires──> [One explicit Liquid conditioning contract]
    ├──requires──> [Explicit lfm2 runtime bring-up]
    └──requires──> [Quantized runtime contract proof]

[Regression protection]
    ├──requires──> [Maintained parity proof]
    └──requires──> [Existing Llama and Qwen maintained tests]

[Benchmark compare/docs publication]
    ├──requires──> [Maintained parity proof]
    └──requires──> [Regression protection]

[Tool use / function calling]
    ──conflicts──> [Narrow first-slice maintained contract]

[Broad Liquid-family claims]
    ──conflicts──> [Single maintained truth anchor]
```

### Dependency Notes

- **Fixture selection comes first:** the exact official file determines both runtime scope and what
  parity/benchmark claims are honest. This is the most important v1.9 scoping decision.
- **Conditioning must be shared across EMEL and `llama.cpp`:** parity is only meaningful if both
  sides see the same structured request and generation prompt behavior.
- **Parity depends on real runtime support:** benchmark publication is downstream of truthful model
  acceptance and execution, not a substitute for it.
- **Regression protection is part of the feature, not optional polish:** in this repo, a new
  maintained slice is incomplete if it can silently break prior maintained anchors.
- **Tool use conflicts with first-slice clarity:** once tools enter the contract, request shape,
  output interpretation, and docs scope all expand at once.

## MVP Definition

### Launch With (v1)

These are the features that belong in v1.9 itself.

- [ ] One official `tests/models/LFM2.5-1.2B-Thinking-Q4_K_M.gguf` fixture is documented with
      source, checksum, stable maintained path, and download URL
      Essential because the milestone needs one reproducible truth anchor. The user explicitly
      approved `LFM2.5-1.2B-Thinking-Q4_K_M.gguf` as the maintained variant.
- [ ] One explicit structured chat-message conditioning contract is documented and used in both
      EMEL and `llama.cpp`
      Essential because the official model card defines a ChatML-like format and a separate tool-use
      surface; v1.9 should choose one narrow maintained contract.
- [ ] EMEL truthfully accepts, loads, and generates on that exact official fixture through the
      maintained generator path
      Essential because this milestone is runtime bring-up, not model-catalog paperwork.
- [ ] The maintained path publishes quantized-contract evidence for that exact fixture without false
      support claims for sibling official quants
      Essential because one ARM slice must mean the shipped runtime contract is real.
- [ ] `tools/paritychecker --generation` proves EMEL against `llama.cpp` for that slice using the
      same fixture and conditioning contract
      Essential because paritychecker is the repo's maintained correctness gate.
- [ ] Regression tests keep the shipped Llama ARM and canonical Qwen slices green while Liquid is
      added
      Essential because the repo already has multiple maintained anchors.
- [ ] `tools/bench` compare output and generated docs publish the same Liquid slice after parity is
      real
      Essential because EMEL's benchmark claims must stay tied to maintained proof.

### Add After Validation (v1.x)

- [ ] A second official Liquid artifact on the same family
      Add only after the first slice is stable. Good follow-up examples: `Q6_K` or `Q8_0` after a
      truthful `Q4_K_M` launch, chosen as explicit follow-on runtime scope.
- [ ] Richer multi-turn or system-message coverage on the same template
      Add after the single-message maintained contract is stable and publication stays readable.
- [ ] Stronger benchmark drift policy for the Liquid row
      Add once the three-anchor compare surface is stable enough to justify a stricter gate.

### Future Consideration (v2+)

- [ ] Other Liquid model families such as `Instruct`, `Base`, `JP`, `VL`, `Audio`, or larger
      checkpoints
      Defer because each widens acceptance, topology, and docs scope.
- [ ] Tool use or function-calling request/response support
      Defer because it is a distinct conditioning and API-surface milestone.
- [ ] Broader official GGUF quant matrix support
      Defer because each added quant changes the runtime truth contract and publication story.
- [ ] New public CLI or C API surfaces shaped around Liquid-specific requests
      Defer because v1.9 should stay on the existing maintained generator, parity, and benchmark
      seams.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| One official Liquid fixture with provenance | HIGH | LOW | P1 |
| One explicit Liquid conditioning contract | HIGH | MEDIUM | P1 |
| Explicit `lfm2` runtime bring-up | HIGH | HIGH | P1 |
| Quantized runtime contract proof for the chosen fixture | HIGH | HIGH | P1 |
| Maintained parity proof against `llama.cpp` | HIGH | HIGH | P1 |
| Regression protection for Llama, Qwen, and Liquid anchors | HIGH | MEDIUM | P1 |
| Benchmark compare/docs publication for the same slice | HIGH | MEDIUM | P1 |
| Formatter-contract metadata publication | MEDIUM | LOW | P2 |
| Negative proof for unsupported Liquid surfaces | MEDIUM | MEDIUM | P2 |
| Stable three-anchor compare readability | MEDIUM | LOW | P2 |
| Second Liquid fixture on the same family | MEDIUM | HIGH | P3 |
| Tool use / function calling | LOW | HIGH | P3 |
| Broad Liquid-family rollout | LOW | HIGH | P3 |
| Full official GGUF quant matrix | LOW | HIGH | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Repo Surface Analysis

| Feature | Existing EMEL Surface | Current State | v1.9 Approach |
|---------|------------------------|---------------|----------------|
| Official fixture provenance | `tests/models/README.md` | Maintained fixture pattern already exists for Llama and Qwen | Add one LiquidAI-authored Thinking GGUF entry and make that path the only maintained Liquid truth anchor |
| Conditioning contract | `tools/generation_formatter_contract.hpp` | Current maintained path already prefers structured chat messages and publishes formatter-contract metadata | Reuse that seam with one Liquid-approved contract only; reject unsupported template variants instead of silently widening |
| Architecture gate | `src/emel/model/data.cpp` and model execution/load code | Current explicit maintained architectures are narrower than "all GGUF text models" | Add explicit `lfm2` handling; do not alias it to Llama or Qwen |
| Quantized runtime truth | `src/emel/kernel/*`, `src/emel/model/data.cpp`, generator attribution | EMEL already owns native `Q6_K` and `Q8_0` hot-path work; it does not visibly own the full official Liquid quant ladder | Pick the first Liquid fixture to match the runtime surface, or make any extra quant support an explicit requirement |
| Parity acceptance | `tools/paritychecker --generation` | Maintained proof already exists for one official Qwen slice and prior Llama anchor | Extend the maintained surface additively to one Liquid slice without broadening the public request surface |
| Benchmark publication | `tools/bench`, compare output, docs generation | Existing flow already publishes maintained slices with formatter metadata | Add one Liquid row only after parity and regression are real |

## Milestone Scope Guidance

### Belongs In v1.9

- One official Liquid Thinking GGUF fixture, pinned up front
- One narrow structured conditioning contract
- Explicit `lfm2` runtime bring-up for that fixture
- Quantized-runtime truth for that fixture only
- Maintained parity against `llama.cpp`
- Regression protection across all maintained anchors
- Benchmark compare/docs publication for the same slice

### Defer To Later Milestones

- A second official Liquid quant or sibling model
- Richer multi-turn/system/tool conditioning
- Benchmark policy hardening beyond the existing compare surface
- Performance optimization work beyond what is required for the first truthful slice

### Explicitly Out Of Scope

- Broad Liquid-family claims
- Tool use or function-calling support
- Unsupported request surfaces or raw fallback on the maintained path
- Full official GGUF quant-matrix support
- New public CLI or C API shaped around Liquid-specific workflows

## Sources

### Official LiquidAI Sources

- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF/tree/main
- https://docs.liquid.ai/deployment/on-device/llama-cpp

### Repo Sources

- `.planning/PROJECT.md`
- `.planning/milestones/v1.6-REQUIREMENTS.md`
- `.planning/milestones/v1.7-REQUIREMENTS.md`
- `tests/models/README.md`
- `tools/generation_formatter_contract.hpp`
- `tools/paritychecker/parity_main.cpp`
- `tools/paritychecker/parity_runner.cpp`
- `tools/bench/generation_bench.cpp`
- `src/emel/model/data.cpp`
- `src/emel/text/formatter/sm.hpp`

---
*Feature research for: EMEL v1.9 Liquid LFM2.5-1.2B-Thinking GGUF ARM slice*
*Researched: 2026-03-31*
