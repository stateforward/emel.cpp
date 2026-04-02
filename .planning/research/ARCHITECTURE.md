# Architecture Research

**Domain:** Brownfield C++ GGUF inference engine adding one maintained LiquidAI
`LFM2.5-1.2B-Thinking-GGUF` ARM slice
**Researched:** 2026-03-31
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
│  initialize -> prefill -> decode on one maintained Liquid slice             │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ model/execution view + tensor contracts
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Model And Execution View                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  src/emel/model/data.*                                                      │
│  existing llama/qwen3 paths                                                 │
│  new explicit lfm2 architecture contract                                    │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ GGUF metadata + tensor names + operator path
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Reference Boundary                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  official Liquid GGUF + official llama.cpp reference                        │
│  same fixture, same conditioning contract, same token budget                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `tests/models/README.md` + tool constants | Pin one official Liquid truth anchor | Stable path, checksum, download URL, and slug for one maintained file |
| `tools/generation_formatter_contract.hpp` | Resolve one canonical Liquid request contract | Match one supported template subset with `tools=none` and `keep_past_thinking=false` |
| `src/emel/model/data.*` | Accept and validate `lfm2` metadata/tensor contracts | Explicit architecture gate plus Liquid-specific tensor and metadata handling |
| `src/emel/generator/detail.hpp` | Execute one maintained Liquid slice | Explicit `lfm2` runtime path, not an alias of existing llama/qwen logic |
| `tools/paritychecker` + `tools/bench` | Publish maintained proof and evidence | Reuse existing surfaces with one new Liquid fixture family |

## Recommended Project Structure

```text
src/
├── emel/model/                 # GGUF-derived model metadata and execution view contracts
├── emel/generator/             # Maintained initialize/prefill/decode orchestration
├── emel/text/                  # Formatter, conditioner, tokenizer seams
└── emel/kernel/                # Native quantized execution path

tools/
├── generation_formatter_contract.hpp  # Maintained request-contract truth surface
├── paritychecker/                     # Correctness proof against llama.cpp
└── bench/                             # Compare/docs publication

tests/
└── models/                     # Official maintained fixture provenance
```

### Structure Rationale

- **`src/emel/model/`**: current architecture acceptance is explicit and narrow; `lfm2` needs to
  land here first so the repo does not pretend generic GGUF compatibility.
- **`src/emel/generator/`**: the user asked for maintained model support, so the shipped runtime
  path remains the truth source.
- **`tools/`**: parity and benchmark publication already define the maintained acceptance boundary;
  Liquid should be added here additively, not via a new harness.

## Architectural Patterns

### Pattern 1: Single-Fixture Truth Anchor

**What:** One official GGUF file defines the milestone.
**When to use:** When a repo widens support to one new model family without wanting generic
compatibility claims.
**Trade-offs:** Narrower than "Liquid support", but it keeps parity and benchmark claims honest.

### Pattern 2: Explicit Architecture Contract

**What:** Add a dedicated `lfm2` architecture path instead of reusing `llama`/`qwen3`.
**When to use:** When official metadata says the model family is distinct and current runtime gates
reject it.
**Trade-offs:** More up-front work than aliasing, but avoids false readiness and hidden operator
gaps.

### Pattern 3: Fixed Maintained Prompt Contract

**What:** Match one supported subset of the official Liquid primary template and publish that
contract on maintained proof surfaces.
**When to use:** When the official template includes optional capabilities such as tools or
thinking-history replay that the milestone does not need.
**Trade-offs:** Less flexible than generic template execution, but far easier to verify.

## Data Flow

### Request Flow

```text
official Liquid fixture path
    ↓
GGUF metadata is loaded and validated as architecture=lfm2
    ↓
formatter resolves one canonical Liquid chat contract
    ↓
conditioner/tokenizer prepare prompt bytes
    ↓
generator::sm initialize/prefill/decode on the maintained Liquid slice
    ↓
EMEL output is compared against llama.cpp using the same fixture and contract
    ↓
bench/docs publish only after parity is real
```

### State Management

```text
generator::action::context
    ↓ owns persistent runtime state
model_data + backend storage
    ↓ receives request-local prompt input through
formatter -> conditioner -> generate event
    ↓ returns generated output to paritychecker or bench
```

### Key Data Flows

1. **Fixture provenance flow:** official Liquid file identity is documented once and reused across
   parity and bench.
2. **Metadata truth flow:** `lfm2` architecture, context length, and template identity are taken
   from GGUF/config metadata rather than stale prose.
3. **Runtime flow:** Liquid metadata and tensors are turned into an explicit execution path that the
   shipped generator can use truthfully.
4. **Verification flow:** parity proves correctness before benchmark publication.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| One maintained Liquid fixture | Keep all work inside existing model, generator, parity, and bench seams |
| More Liquid quants on the same family | Re-evaluate runtime truth per quant before widening claims |
| More Liquid families | Re-evaluate topology, prompt contract, and acceptance boundary per family |

### Scaling Priorities

1. **First bottleneck:** architecture/runtime truth, because current code only admits `llama` and
   `qwen3`.
2. **Second bottleneck:** conditioning fidelity, because the current supported Qwen contract does
   not match Liquid's `keep_past_thinking` template surface.

## Anti-Patterns

### Anti-Pattern 1: Treat `lfm2` As A Llama Alias

**What people do:** let the model through an existing architecture gate because it is "just another
GGUF".
**Why it's wrong:** official Liquid and llama.cpp metadata identify this family as `lfm2` with a
hybrid block contract, not a llama/qwen clone.
**Do this instead:** add explicit `lfm2` handling or keep the model unsupported until it is real.

### Anti-Pattern 2: Use The Old Qwen Formatter Contract

**What people do:** reuse the current supported-template matcher because the visible tokens look
similar.
**Why it's wrong:** Liquid's primary template uses `keep_past_thinking` and different tool token
names; a false-positive matcher would overstate support.
**Do this instead:** add one Liquid-specific maintained contract and reject unsupported variants.

### Anti-Pattern 3: Publish Benchmarks From Prose Metadata

**What people do:** rely on the model card summary line and ignore GGUF/config truth.
**Why it's wrong:** the official prose says `32,768` context while GGUF/config metadata publish
`128000`; the maintained slice should follow executable metadata.
**Do this instead:** derive architecture/context truth from GGUF/config and document the discrepancy.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Official LiquidAI Hugging Face repos | Manual fixture acquisition plus provenance recording | Use for one official artifact only |
| Official llama.cpp reference | Tool-only parity and benchmark comparison | Current pin already supports `lfm2` metadata |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `tools/paritychecker` ↔ `src/emel/generator` | Existing initialize/generate flow | Keep CLI surface stable; change runtime truth underneath it |
| `tools/bench` ↔ `src/emel/generator` | Existing benchmark session setup | Add one Liquid case family only after parity is real |
| `tools/generation_formatter_contract.hpp` ↔ `text::conditioner` | Existing `format_fn` injection | Preferred place to make Liquid prompt conditioning explicit |
| `src/emel/model/data.*` ↔ `src/emel/generator/detail.hpp` | Execution-view and tensor-topology handoff | Core architecture gate for the milestone |

## Sources

- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking-GGUF
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/raw/main/config.json
- https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking/raw/main/tokenizer_config.json
- https://docs.liquid.ai/lfm/models/lfm25-1.2b-thinking
- https://docs.liquid.ai/deployment/on-device/llama-cpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/.planning/PROJECT.md
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/src/emel/model/data.cpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/src/emel/generator/detail.hpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/generation_formatter_contract.hpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/paritychecker/parity_runner.cpp
- /Users/gabrielwillen/.superset/worktrees/emel.cpp/feat/liquid-ai/tools/bench/generation_bench.cpp

---
*Architecture research for: EMEL Liquid LFM2.5-1.2B-Thinking GGUF ARM slice*
*Researched: 2026-03-31*
