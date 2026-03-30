# Architecture Research

**Domain:** Qwen3-0.6B bring-up on EMEL's maintained generation, parity, and benchmark surfaces
**Researched:** 2026-03-27
**Confidence:** MEDIUM-HIGH

## Standard Architecture

### System Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Verification Surfaces                             │
├──────────────────────────────────────────────────────────────────────────────┤
│  tools/paritychecker        tools/bench                                     │
│  generation parity          compare/snapshot/docs publication               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ one canonical fixture + one canonical request
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Prompt Conditioning Layer                           │
├──────────────────────────────────────────────────────────────────────────────┤
│  src/emel/text/formatter   src/emel/text/conditioner   src/emel/text/jinja │
│  choose request contract   apply formatter before      render chat template │
│                             tokenization                 if needed           │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ formatter output + tokenizer settings
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Runtime Orchestration                             │
├──────────────────────────────────────────────────────────────────────────────┤
│  src/emel/generator::sm                                                    │
│  - initialize tokenizer/conditioner                                        │
│  - prepare prompt + decode capacities                                      │
│  - run shipped generate flow                                               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ model/execution view + graph compute
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Model And Execution View                            │
├──────────────────────────────────────────────────────────────────────────────┤
│  src/emel/model/data.*                                                     │
│  src/emel/model/llama/detail.hpp (current truth)                           │
│  future qwen3-aware detail or equivalent runtime support                   │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ typed tensor views / step plans
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Reference Boundary                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  llama.cpp CPU reference                                                   │
│  - same fixture                                                            │
│  - same conditioning contract                                              │
│  - same max-token budget                                                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `tests/models/README.md` plus maintained fixture constants | Own canonical fixture identity and provenance | Documented file name, source repo, checksum, and download URL |
| `tools/paritychecker` | Own maintained Qwen parity acceptance | Existing CLI surface; should gain one canonical Qwen slice without widening into a model matrix |
| `tools/bench` | Own maintained compare publication | Existing compare/docs flow; should publish the same Qwen slice that parity already proves |
| `src/emel/text/formatter` / `text::conditioner` | Own prompt-conditioning contract | Existing seam currently uses `format_raw`; v1.6 should use it to make Qwen conditioning explicit |
| `src/emel/model/data.*` and model-detail helpers | Own architecture name, metadata, and execution-view truth | Current truth is Llama-specific; Qwen3 support must land here or remain rejected |
| `src/emel/generator` | Own shipped initialize/generate flow | Existing Boost.SML actor; v1.6 should avoid actor-graph rewrites and stay within current seams |

## Recommended Project Structure

```text
tests/
├── models/
│   └── README.md                 # Canonical Qwen3 fixture provenance
tools/
├── paritychecker/
│   ├── parity_main.cpp           # Usage/help text and canonical fixture guidance
│   └── parity_runner.cpp         # Maintained generation surface and architecture gates
└── bench/
    └── generation_bench.cpp      # Maintained compare path and fixture gates
src/
├── emel/
│   ├── model/
│   │   ├── data.hpp              # Architecture metadata and chat-template fields
│   │   ├── data.cpp              # Architecture-aware execution-view logic
│   │   └── llama/detail.hpp      # Current Llama-only detail truth
│   ├── generator/
│   │   ├── context.hpp           # Runtime-owned generation state
│   │   ├── detail.hpp            # Current execution path depends on model::llama::detail
│   │   └── sm.hpp                # Keep actor structure stable
│   └── text/
│       ├── formatter/format.hpp  # Existing formatter injection seam
│       ├── formatter/sm.hpp      # Chat-formatting design intent
│       ├── conditioner/**        # Pre-tokenization request shaping
│       └── jinja/**              # Existing parser/formatter support
```

### Structure Rationale

- **Fixture and tool files move first:** the milestone needs an explicit truth anchor before runtime claims become meaningful.
- **Prompt conditioning is a first-class layer:** Qwen3 is not just a different tensor file; its operator-facing request contract matters.
- **Model/execution-view support is the real runtime gate:** current EMEL generation depends on `model::llama::detail`, so architecture work must happen before the tools can honestly claim support.
- **Generator actor structure should stay stable:** AGENTS requires asking before state-machine structure changes, so v1.6 should prefer seam-level and data-plane changes over actor rewrites.

## Architectural Patterns

### Pattern 1: Truth Anchor First

**What:** Establish one official fixture, one canonical slug/path, and one documented request contract before widening runtime support.
**When to use:** Any time a brownfield milestone adds a new model family to maintained surfaces.
**Trade-offs:** Front-loads planning and provenance work, but it prevents later benchmark/parity drift.

### Pattern 2: Condition Through Existing Formatter Injection

**What:** Use `formatter::format_fn` plus `text::conditioner` to make Qwen request shaping explicit instead of adding ad hoc prompt logic inside generator actions.
**When to use:** When the model family needs a different prompt contract but the actor graph should stay unchanged.
**Trade-offs:** Requires a clean formatter decision up front, but avoids illegal runtime branching inside SML actions.

### Pattern 3: Add Architecture Support at the Model-View Boundary

**What:** Teach EMEL's model/view layer how one Qwen3 slice maps into runtime tensor views and step plans, instead of pretending the architecture is already Llama-compatible.
**When to use:** When the reference implementation shows distinct architecture/tensor semantics.
**Trade-offs:** More upfront runtime work, but much less risk of false readiness. Local reference code shows Qwen3 includes `attn_q_norm` and `attn_k_norm`, so a simple string alias is unsafe.

### Pattern 4: Keep Verification And Publication On Existing Surfaces

**What:** Reuse the current paritychecker generation mode and bench compare/docs flow.
**When to use:** When the goal is honest brownfield expansion, not a new product surface.
**Trade-offs:** Constrains scope to what the existing tools can publish, but that is exactly what keeps the milestone truthful.

## Data Flow

### Request Flow

```text
canonical Qwen fixture path
    ↓
tool surface loads GGUF metadata and validates architecture
    ↓
formatter chooses one canonical request contract
    ↓
conditioner prepares prompt bytes for tokenization
    ↓
generator::sm initialize/generate on the Qwen slice
    ↓
model/execution-view support binds Qwen tensors for runtime use
    ↓
EMEL output is compared against llama.cpp using the same fixture and request contract
    ↓
bench/docs publish only after parity is already real
```

### State Management

```text
generator::action::context
    ↓ owns persistent runtime state
model_data + backend/session storage
    ↓ receives request-local prompt input through
formatter → conditioner → generate event
    ↓ returns generated output to paritychecker or bench
```

### Key Data Flows

1. **Fixture provenance flow:** official model file identity is documented in `tests/models/README.md` and mirrored in maintained tool constants.
2. **Conditioning flow:** one canonical prompt contract is rendered before tokenization so EMEL and `llama.cpp` consume the same operator-facing request.
3. **Runtime flow:** Qwen3 metadata and tensors are turned into an execution view that the existing generator path can use without changing its actor graph.
4. **Verification flow:** paritychecker proves correctness before bench/docs refresh publication.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| One canonical Qwen3 fixture | Keep all work inside existing tool, formatter, model, and generator seams |
| More Qwen3 prompts on the same fixture | Add prompt-contract documentation and tests, not new runtime families |
| More Qwen-family models | Re-evaluate topology and conditioning per family before reusing v1.6 assumptions |

### Scaling Priorities

1. **First bottleneck:** model/execution-view support, because current runtime still depends on `model::llama::detail`.
2. **Second bottleneck:** conditioning fidelity, because Qwen3 benchmark/parity claims are weak if EMEL and `llama.cpp` do not use the same prompt contract.

## Anti-Patterns

### Anti-Pattern 1: Update Only The Fixture Name

**What people do:** Swap `Llama-68M-Chat-v1-Q2_K.gguf` for a Qwen path and call the milestone started.
**Why it's wrong:** The current maintained tools still validate `architecture == "llama"` and the runtime still builds a Llama-only execution view.
**Do this instead:** Change fixture constants, architecture gates, and runtime support together in a narrow, explicit Qwen slice.

### Anti-Pattern 2: Treat Qwen3 As A Llama Alias

**What people do:** Allow `qwen3` through the same runtime path without accounting for distinct tensor families.
**Why it's wrong:** Local reference source shows Qwen3 includes attention-normalization tensors beyond the current Llama tensor set.
**Do this instead:** Add explicit Qwen3-aware runtime handling or keep the model rejected until that support exists.

### Anti-Pattern 3: Keep `format_raw` And Claim Official Qwen Behavior

**What people do:** Reuse the current raw prompt path because it is already wired into generator initialization.
**Why it's wrong:** It avoids the hard prompt-contract decision and can produce misleading parity or endless-repetition behavior for an instruct model.
**Do this instead:** Make the Qwen request contract explicit through the existing formatter/conditioner seam.

### Anti-Pattern 4: Publish Benchmarks Before Parity

**What people do:** Land compare output first because it is visible.
**Why it's wrong:** It creates performance claims for a slice whose correctness and conditioning contract are still unproven.
**Do this instead:** Make parity and runtime tests precede benchmark publication.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Official Hugging Face Qwen GGUF repo | Manual fixture acquisition plus README provenance | Use for one canonical artifact only |
| `llama.cpp` CPU reference | Tool-only parity and benchmark comparison | Keep fixture, prompt contract, and token budget aligned with EMEL |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `tools/paritychecker` ↔ `src/emel/generator` | Existing initialize/generate flow | Keep CLI surface stable; change runtime truth underneath it |
| `tools/bench` ↔ `src/emel/generator` | Existing benchmark session setup | Keep compare flow stable; add one Qwen slice only after parity is real |
| `text::formatter` ↔ `text::conditioner` | Existing `format_fn` injection | Preferred place to make Qwen prompt conditioning explicit |
| `src/emel/model/data.*` ↔ `src/emel/generator/detail.hpp` | Execution-view and tensor-topology handoff | Current runtime is Llama-only here; this is the core architecture gate |

## New vs Modified Components

### Modified Runtime Components

- `src/emel/model/data.cpp`
  - Current execution-view builder rejects non-Llama architectures and assumes Llama tensor families.
  - v1.6 likely needs explicit Qwen3-aware handling here or an adjacent detail helper.
- `src/emel/generator/detail.hpp`
  - Current runtime depends on `emel::model::llama::detail::*`.
  - v1.6 needs enough runtime generalization to execute one Qwen3 slice honestly.
- `src/emel/text/formatter/format.hpp` or adjacent formatter helper
  - Likely integration point for a canonical Qwen request contract if raw formatting is insufficient.

### Modified Tooling Components

- `tests/models/README.md`
  - Add the official Qwen3-0.6B fixture provenance.
- `tools/paritychecker/parity_main.cpp` / `tools/paritychecker/parity_runner.cpp`
  - Update generation usage text, canonical fixture constants, architecture validation, and reference alignment.
- `tools/bench/generation_bench.cpp`
  - Update the maintained compare path to the same canonical Qwen request shape.

## Suggested Build Order

1. Fixture provenance and maintained tool constants.
2. Canonical prompt-conditioning contract.
3. Runtime architecture support for one Qwen3 slice.
4. Parity and regression tests.
5. Benchmark compare/docs refresh.

## Sources

- `.planning/PROJECT.md` - v1.6 scope and acceptance boundary.
- `AGENTS.md` - requirement to avoid silent fallback claims and to ask before state-machine structure changes.
- `src/emel/model/data.cpp` - current execution-view logic rejects non-Llama architectures.
- `src/emel/generator/detail.hpp` - current runtime depends on `emel::model::llama::detail`.
- `src/emel/text/formatter/format.hpp`, `src/emel/text/formatter/sm.hpp`, and `src/emel/text/conditioner/**` - existing prompt-conditioning seam.
- `tools/paritychecker/parity_runner.cpp` and `tools/bench/generation_bench.cpp` - current Llama-only fixture and architecture gates.
- `build/paritychecker/_deps/reference_impl-src/src/llama-arch.cpp` - local reference source enumerates `qwen3` as a distinct architecture.
- `build/paritychecker/_deps/reference_impl-src/src/llama-arch.cpp` and `build/paritychecker/_deps/reference_impl-src/src/llama-model.cpp` - local reference source shows Qwen3-specific tensor and hparam expectations, including attention-normalization tensors.
- https://huggingface.co/Qwen/Qwen3-0.6B - official model card.
- https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html - official Qwen `llama.cpp` guidance.

---
*Architecture research for: EMEL v1.6 Qwen3-0.6B parity and benchmark*
*Researched: 2026-03-27*
