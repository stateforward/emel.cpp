# Architecture Research

**Domain:** Brownfield C++ GGUF inference engine adding one maintained Gemma 4 E2B text slice
**Researched:** 2026-04-02
**Confidence:** MEDIUM

## Standard Architecture

### System Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Verification Surfaces                             │
├──────────────────────────────────────────────────────────────────────────────┤
│  tools/paritychecker        tools/bench                                     │
│  generation parity          compare/snapshot/docs publication               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ one official fixture + one canonical request
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Prompt Conditioning Layer                           │
├──────────────────────────────────────────────────────────────────────────────┤
│  tools/generation_formatter_contract.hpp                                    │
│  src/emel/text/conditioner                                                  │
│  src/emel/text/tokenizer                                                    │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ prompt bytes + tokenizer settings
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Runtime Orchestration                             │
├──────────────────────────────────────────────────────────────────────────────┤
│  src/emel/generator::sm                                                     │
│  initialize -> prefill -> decode on one maintained Gemma 4 text slice      │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ model/execution view + tensor contracts
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Model And Execution View                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  src/emel/model/data.*                                                      │
│  existing llama/qwen3/lfm2 paths                                            │
│  new explicit gemma4 text-only architecture contract                        │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ GGUF metadata + tensor names + operator path
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Reference Boundary                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  official Gemma 4 GGUF + Gemma 4-capable llama.cpp reference                │
│  same fixture, same conditioning contract, same token budget                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `tests/models/README.md` + tool constants | Pin one official Gemma 4 truth anchor | Stable path, checksum, download URL, and slug for one maintained file |
| `tools/generation_formatter_contract.hpp` | Resolve one canonical Gemma 4 text contract | Match one supported template subset with no media and no tools |
| `src/emel/model/data.*` | Accept and validate `gemma4` metadata/tensor contracts | Explicit architecture gate plus Gemma 4-specific topology handling |
| `src/emel/generator/...` | Execute one maintained Gemma 4 text slice | Explicit `gemma4` runtime path, not an alias of existing families |
| `tools/paritychecker` + `tools/bench` | Publish maintained proof and evidence | Reuse existing surfaces with one new Gemma 4 fixture family and an explicit ref-pin readiness step |

## Architectural Patterns

### Pattern 1: Single-Fixture Truth Anchor

**What:** One official GGUF file defines the milestone.
**When to use:** When a repo widens support to one new model family without wanting generic
compatibility claims.
**Trade-offs:** Narrower than "Gemma 4 support," but it keeps parity and benchmark claims honest.

### Pattern 2: Explicit Architecture Contract

**What:** Add a dedicated `gemma4` architecture path instead of reusing `llama`, `qwen3`, or
`lfm2`.
**When to use:** When official metadata says the model family is distinct and current runtime gates
do not describe its topology truthfully.
**Trade-offs:** More up-front work than aliasing, but avoids false readiness and hidden operator
gaps.

### Pattern 3: Fixed Maintained Prompt Contract

**What:** Match one supported subset of the official Gemma 4 `chat_template` and publish that
contract on maintained proof surfaces.
**When to use:** When the official template includes optional capabilities such as tools and media
that the milestone does not need.
**Trade-offs:** Less flexible than generic template execution, but far easier to verify.

### Pattern 4: Explicit Reference Boundary Readiness

**What:** Treat the `llama.cpp` reference pin as a managed dependency of the milestone instead of
assuming it is always current enough.
**When to use:** When the repo's pinned reference commit may lag the new model family.
**Trade-offs:** One extra planning requirement, but it prevents late parity/bench dead ends.

## Data Flow

### Request Flow

```text
official Gemma 4 fixture path
    ↓
GGUF metadata is loaded and validated as architecture=gemma4
    ↓
formatter resolves one canonical Gemma 4 text-only contract
    ↓
conditioner/tokenizer prepare prompt bytes
    ↓
generator::sm initialize/prefill/decode on the maintained Gemma 4 text slice
    ↓
EMEL output is compared against llama.cpp using the same fixture and contract
    ↓
bench/docs publish only after parity is real
```

### Key Data Flows

1. **Fixture provenance flow:** one official Gemma 4 file identity is documented once and reused
   across parity and bench.
2. **Metadata truth flow:** `gemma4`, `131072`, layer schedule, and media token ids come from
   GGUF/config metadata rather than marketing prose.
3. **Runtime flow:** Gemma 4 metadata and tensors are turned into an explicit execution path that
   the shipped generator can use truthfully.
4. **Reference flow:** parity and benchmark surfaces only become valid after the pinned
   `llama.cpp` reference lane is Gemma 4-capable.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| One maintained Gemma 4 text fixture | Keep all work inside existing model, generator, parity, and bench seams |
| `mmproj` plus image input | Add explicit media encoder integration and new acceptance boundaries |
| Audio/video support | Add new media request/verification seams |
| More Gemma 4 fixtures or precisions | Re-evaluate runtime truth and benchmark claims per fixture before widening |

## Anti-Patterns

### Anti-Pattern 1: Treat `gemma4` As A Family Alias

**What people do:** let the model through an existing architecture gate because it is "just
another GGUF."
**Why it's wrong:** official metadata identifies this family as `gemma4`, with a different
multimodal/text contract and topology from the current maintained families.
**Do this instead:** add explicit `gemma4` handling or keep the model unsupported until it is real.

### Anti-Pattern 2: Claim Multimodal Readiness From The `mmproj` File Alone

**What people do:** see the official `mmproj` sibling and assume the repo now supports images or
other media.
**Why it's wrong:** a separate official asset is only source truth, not proof that EMEL has a
media encoder pipeline.
**Do this instead:** keep `v1.11` text-only and make multimodal support a later explicit milestone.

### Anti-Pattern 3: Assume The Current Reference Pin Is Good Enough

**What people do:** treat parity as a downstream concern and ignore the `llama.cpp` ref until late.
**Why it's wrong:** the current pinned commit appears not to contain `gemma4`, so parity/bench can
block after runtime work is already done.
**Do this instead:** make ref-pin readiness explicit in the roadmap before benchmark publication.

## Sources

- https://huggingface.co/api/models/ggml-org/gemma-4-E2B-it-GGUF
- https://huggingface.co/api/models/google/gemma-4-E2B-it
- https://huggingface.co/google/gemma-4-E2B-it/resolve/main/config.json
- https://huggingface.co/google/gemma-4-E2B-it/resolve/main/processor_config.json
- https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp
- `tools/paritychecker/reference_ref.txt`
- `https://raw.githubusercontent.com/ggml-org/llama.cpp/ecbcb7ea9d3303097519723b264a8b5f1e977028/src/llama-model.cpp`
