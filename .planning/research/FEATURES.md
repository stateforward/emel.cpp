# Feature Research

**Domain:** Single maintained `ggml-org/gemma-4-E2B-it-GGUF` text slice on EMEL's existing
generation, parity, and benchmark surfaces
**Researched:** 2026-04-02
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features EMEL needs if it claims one truthful maintained Gemma 4 text slice.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| One documented official Gemma 4 GGUF fixture with provenance | EMEL already treats one stable `tests/models/` artifact as the truth anchor for each maintained slice | LOW | Lock one exact official file name, stable path, SHA256, and download URL before any runtime work. |
| One explicit Gemma 4 text-only conditioning contract | The official template is richer than a plain text prompt and includes tools and media surfaces | MEDIUM | Keep the maintained path on one structured text chat contract with `add_generation_prompt=true`. Do not allow implicit raw fallback. |
| Explicit `gemma4` model/runtime bring-up on the maintained generator path | The GGUF metadata identifies the model as `gemma4`, not as a current maintained family | HIGH | `v1.11` is not done if EMEL aliases Gemma 4 to `llama`, `qwen3`, or `lfm2`. |
| Explicit rejection of unsupported multimodal and tool-use request shapes | The base model is any-to-any and the official GGUF repo ships a separate `mmproj` file | MEDIUM | The milestone needs truthful text-only behavior, not silent accidental acceptance. |
| Reference-lane readiness for Gemma 4 parity and bench | Maintained correctness claims live on `tools/paritychecker` and maintained publication lives on `tools/bench` | MEDIUM | The current pinned `llama.cpp` ref appears not to contain `gemma4`, so this cannot stay implicit. |
| Maintained parity proof against `llama.cpp` using the same fixture and contract | EMEL's correctness claims live on `tools/paritychecker --generation`, not ad hoc manual runs | HIGH | Match the existing maintained pattern: same GGUF file, same prompt contract, and stored coverage. |
| Regression protection for existing maintained anchors while Gemma 4 lands | Brownfield widenings are expected to preserve prior maintained proof | MEDIUM | Coverage should catch fixture drift, conditioning drift, architecture misclassification, and regressions on prior anchors. |
| Benchmark compare/docs publication for the exact parity-backed Gemma 4 slice | EMEL already publishes benchmark evidence only when it maps to a maintained, correctness-backed slice | MEDIUM | Add one Gemma 4 row to the existing compare/docs flow after parity is green. |

### Differentiators (Competitive Advantage)

Helpful additions that make the Gemma 4 bring-up more trustworthy and easier to operate, but are
not strictly required to call `v1.11` complete.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Published formatter-contract metadata on parity and benchmark outputs | Makes every maintained claim auditable: reviewers can see the slice ran with the intended structured contract | LOW | Reuse the existing formatter-contract publication seam and make the Gemma 4 row explicit about text-only scope. |
| Explicit negative proof for unsupported media and tool paths | Truthfulness improves when the repo proves what it rejects, not only what it accepts | MEDIUM | Add failure coverage for `mmproj`, image/audio/video placeholders, or tool-call surfaces instead of silently widening acceptance. |
| Stable four-anchor compare readability | Once Gemma 4 joins Llama, Qwen, and Liquid, operators need compare rows that stay easy to distinguish and review | LOW | Use family/variant/fixture-specific slugs so docs and stored compare output do not blur the maintained anchors together. |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Broad "Gemma 4 support" in `v1.11` | It sounds stronger to say the repo supports Gemma 4 generally | The upstream model is multimodal and any-to-any; full family claims would overstate what the repo can actually prove | Keep `v1.11` fixed to one maintained text slice |
| `mmproj` support in the first Gemma 4 milestone | The official repo exposes a matching `mmproj` file, so it feels adjacent | It widens the milestone into media encoder, request-shape, and verification work beyond today's acceptance boundary | Record the official `mmproj` file as source truth, but defer it |
| Tool use or function calling as part of `v1.11` | The official template supports tools, so it feels close at hand | Tool schemas, tool-call replay, and output interpretation are separate product work from first-slice text truth | Keep the maintained contract text only with no tools |
| Benchmark publication before reference parity is green | Bench output is visible and easy to demo | It creates speed claims for a slice whose correctness or reference lane is still unresolved | Land reference readiness and parity first, then publish benchmark evidence |
| Assuming the current pinned `llama.cpp` ref already supports Gemma 4 | The repo already has a reference lane, so it is tempting to treat it as a solved problem | The current pinned commit appears not to contain `gemma4`, so parity/bench can block late if this stays implicit | Make ref-pin readiness an explicit requirement |

## Feature Dependencies

```text
[One official Gemma 4 fixture with provenance]
    └──requires──> [Stable maintained path and checksum]

[One explicit Gemma 4 text-only conditioning contract]
    ├──requires──> [One official Gemma 4 fixture with provenance]
    └──requires──> [Official chat-template inspection]

[Explicit gemma4 runtime bring-up]
    ├──requires──> [One official Gemma 4 fixture with provenance]
    └──requires──> [Architecture-specific execution/load support]

[Explicit rejection of multimodal/tool surfaces]
    ├──requires──> [Official model metadata truth]
    └──requires──> [One explicit Gemma 4 text-only conditioning contract]

[Reference-lane readiness]
    ├──requires──> [Pinned reference commit audit]
    └──requires──> [Gemma 4-capable llama.cpp reference]

[Maintained parity proof]
    ├──requires──> [One explicit Gemma 4 text-only conditioning contract]
    ├──requires──> [Explicit gemma4 runtime bring-up]
    └──requires──> [Reference-lane readiness]

[Regression protection]
    ├──requires──> [Maintained parity proof]
    └──requires──> [Existing Llama, Qwen, and Liquid maintained tests]

[Benchmark compare/docs publication]
    ├──requires──> [Maintained parity proof]
    └──requires──> [Regression protection]
```

## MVP Definition

### Launch With (v1)

These are the features that belong in `v1.11` itself.

- [ ] One official `tests/models/gemma-4-e2b-it-Q8_0.gguf` fixture is documented with source,
      checksum, stable maintained path, and download URL.
- [ ] One explicit structured text chat conditioning contract is documented and used in both EMEL
      and `llama.cpp`.
- [ ] EMEL truthfully accepts, loads, and generates on that exact official fixture through the
      maintained generator path.
- [ ] The maintained path rejects `mmproj`, media, and tool-call request shapes explicitly.
- [ ] The pinned `llama.cpp` reference lane is Gemma 4-capable.
- [ ] `tools/paritychecker --generation` proves EMEL against `llama.cpp` for that slice using the
      same fixture and conditioning contract.
- [ ] Regression tests keep the shipped Llama, Qwen, and merged Liquid anchors green while Gemma 4
      is added.
- [ ] `tools/bench` compare output and generated docs publish the same Gemma 4 slice after parity
      is real.

### Future Consideration (v2+)

- [ ] `mmproj` plus image input support
- [ ] Audio/video request support
- [ ] Tool use or function calling
- [ ] `F16` or broader Gemma 4 fixture coverage
- [ ] Gemma 4-specific performance optimization after correctness is proven

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| One official Gemma 4 fixture with provenance | HIGH | LOW | P1 |
| One explicit Gemma 4 text-only conditioning contract | HIGH | MEDIUM | P1 |
| Explicit `gemma4` runtime bring-up | HIGH | HIGH | P1 |
| Explicit rejection of multimodal/tool paths | HIGH | MEDIUM | P1 |
| Reference-lane readiness for Gemma 4 | HIGH | MEDIUM | P1 |
| Maintained parity proof against `llama.cpp` | HIGH | HIGH | P1 |
| Regression protection for Llama/Qwen/Liquid/Gemma 4 anchors | HIGH | MEDIUM | P1 |
| Benchmark compare/docs publication for the same slice | HIGH | MEDIUM | P1 |
| Formatter-contract metadata publication | MEDIUM | LOW | P2 |
| Negative proof for unsupported request surfaces | MEDIUM | MEDIUM | P2 |
| Broad Gemma 4 rollout or multimodal support | LOW | HIGH | P3 |

## Sources

- https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF
- https://huggingface.co/api/models/ggml-org/gemma-4-E2B-it-GGUF
- https://huggingface.co/api/models/google/gemma-4-E2B-it
- https://huggingface.co/google/gemma-4-E2B-it/resolve/main/config.json
- https://huggingface.co/google/gemma-4-E2B-it/resolve/main/chat_template.jinja
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp
- `tools/paritychecker/reference_ref.txt`
