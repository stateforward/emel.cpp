# Architecture Research

**Domain:** Bonsai-1.7B slice integration into EMEL's existing Boost.SML inference architecture
**Researched:** 2026-04-02
**Confidence:** HIGH

## Standard Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Maintained Slice / Tooling Layer                    │
├──────────────────────────────────────────────────────────────────────────────┤
│  tests/models/Bonsai-1.7B.gguf   generation_fixture_registry   parity/bench │
│  generation_formatter_contract   docs/snapshots/publication metadata        │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ same maintained fixture + same request contract
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Existing Model Ingest Layer                       │
├──────────────────────────────────────────────────────────────────────────────┤
│  gguf::loader::sm  ->  model metadata mapping  ->  model::loader::sm        │
│         MODIFIED            MODIFIED                   UNCHANGED SML shape   │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ raw tensor records + qwen3 metadata
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Existing Runtime / Actor Layer                      │
├──────────────────────────────────────────────────────────────────────────────┤
│  text::conditioner  ->  generator::initializer  ->  generator/prefill       │
│     UNCHANGED              UNCHANGED                  SAME machine,          │
│  tokenizer/formatter       same RTC                   maybe new contracts    │
│                                                        if route identity     │
│                                                        must stay explicit    │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ bound backend tensors + explicit compute route
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Kernel / Backend Layer                           │
├──────────────────────────────────────────────────────────────────────────────┤
│  kernel::detail  ->  kernel::aarch64 / shared CPU path  ->  graph/memory    │
│    MODIFIED                MODIFIED / NEW helpers              UNCHANGED     │
│  add Q1_0_g128             add native Q1_0_g128 x Q8 input                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Status | Responsibility | Bonsai impact |
|-----------|--------|----------------|---------------|
| `src/emel/gguf/loader` | Modified | Validate GGUF tensor types, compute row sizes, parse tensor metadata | Must accept raw GGUF type `41` / `Q1_0_g128` |
| `src/emel/model` | Modified | Map GGUF metadata into `model::data`, validate execution contract, audit quantized stages | Reuse `qwen3` contract; extend tensor-type recognition to `Q1_0_g128` |
| `src/emel/model/weight_loader` | Unchanged | Map already-parsed tensor offsets into live pointers | No structural change; works once loader row-size math is correct |
| `src/emel/text/conditioner` + tokenizer | Unchanged | Format and tokenize structured chat messages | Live Bonsai metadata fits existing Qwen-style contract |
| `src/emel/generator/initializer` | Unchanged | Bind conditioner, backend, graph, memory, sampler | No Bonsai-specific state-machine required |
| `src/emel/generator/prefill` | Modified if needed | Publish explicit prefill compute contract | Add explicit Bonsai contract only if route identity must be surfaced |
| `src/emel/generator/detail.hpp` | Modified | Bind runtime tensors, prepare backend, run prefill/decode kernels | Main integration hotspot for `Q1_0_g128` |
| `src/emel/kernel/detail.hpp` | Modified | Shared quant structs, row-size math, pack/dequant helpers | Add `block_q1_0_g128` and native operand helpers |
| `src/emel/kernel/aarch64` | Modified | Optimized hot-path kernel dispatch | Port native `Q1_0_g128 x Q8_*` math into EMEL-owned kernels |
| `tools/generation_formatter_contract.hpp` | Likely unchanged | Resolve maintained chat contract from GGUF template | Live template already matches existing Qwen contract markers |
| `tools/paritychecker` / `tools/bench` | Modified | Maintained fixture loading, proof, benchmark publication | Add Bonsai fixture and preserve EMEL-vs-reference split lanes |

## Recommended Project Structure

```text
src/emel/
├── gguf/loader/          # MODIFY: recognize raw GGUF type 41 and correct row layout
├── model/                # MODIFY: qwen3 Bonsai metadata + quantized audit
├── generator/
│   ├── initializer/      # KEEP: existing SML lifecycle
│   ├── prefill/          # MAY MODIFY: explicit Bonsai compute contracts if required
│   └── detail.hpp        # MODIFY: bind q1_0_g128 tensors and runtime kernels
├── kernel/
│   ├── detail.hpp        # MODIFY: q1_0_g128 storage and helpers
│   └── aarch64/          # MODIFY: native q1_0_g128 kernel routes
tools/
├── generation_fixture_registry.hpp   # MODIFY: add Bonsai maintained fixture
├── generation_formatter_contract.hpp # KEEP or narrowly modify if live template drifts
├── paritychecker/                     # MODIFY: Bonsai proof path
└── bench/                             # MODIFY: Bonsai benchmark path
tests/
├── models/              # MODIFY: fixture doc + checksum + stable path
├── gguf/loader/         # MODIFY: type-41 loader coverage
├── kernel/              # ADD TESTS: q1_0_g128 arithmetic and dispatch
└── generator/           # MODIFY: Bonsai init/prefill/decode coverage
```

### Structure Rationale

- `src/emel/model` stays the architectural owner of model-family structure. Bonsai should not introduce `src/emel/model/bonsai` because the live GGUF says `general.architecture=qwen3`.
- `src/emel/kernel` is where the milestone actually changes the runtime contract. Bonsai is primarily a new operand class, not a new actor graph.
- `tools/` remains the right place for the maintained fixture registry and chat-template contract resolution. The generator should stay formatter-agnostic.

## Architectural Patterns

### Pattern 1: Treat Bonsai as `qwen3` + new quantized operand class

**What:** Reuse the existing Qwen3 execution topology, attention-q/k norm behavior, conditioner wiring, and request contract. Only the weight operand class changes.
**When to use:** Always for `prism-ml/Bonsai-1.7B-gguf`.
**Trade-offs:** Minimal architectural churn. The downside is that kernel and dtype plumbing must be precise, because the model family and the weight format stop being coupled.

**Recommendation:** Do not add a Bonsai-specific top-level machine or model family. Keep `architecture_name=="qwen3"` as the branch point.

### Pattern 2: Separate raw GGUF dtypes from EMEL-prepared/internal dtypes

**What:** Preserve raw file types exactly as loaded from GGUF, and move EMEL-only prepared formats into a separate internal enum or a reserved private numeric range.
**When to use:** Before parsing Bonsai, because the live file uses raw tensor type `41`, while EMEL currently uses `41` for internal `q4_k_x8_bl4`.
**Trade-offs:** Some one-time plumbing churn across loader, kernel, and generator code. It prevents silent misclassification and keeps future custom GGUF slices survivable.

**Recommendation:** Keep `model::data::tensor_record.type` raw. Do not overload raw tensor metadata with prepared-runtime encodings.

### Pattern 3: Keep compute routing explicit at the SML boundary

**What:** If Bonsai needs distinct prefill or decode routes, surface them as explicit guards/contracts/events instead of hiding them in helper branching.
**When to use:** Only when the route identity affects truthfulness, parity labels, or benchmark publication.
**Trade-offs:** A few more explicit route states or contracts, but that is consistent with `AGENTS.md` and `docs/rules/sml.rules.md`.

**Recommendation:** Prefer small additive route contracts in `generator/prefill` over a hidden `if (tensor_type == q1_0_g128)` branch inside actions.

## Data Flow

### Maintained Load Flow

```
tests/models/Bonsai-1.7B.gguf
    ↓
gguf::loader::sm
    ↓ raw kv entries + raw tensor records (type 41 must survive intact)
model metadata mapping
    ↓ architecture=qwen3, tokenizer.model=gpt2, tokenizer.pre=qwen2
model::loader::sm validation
    ↓
generator::initializer
    ↓
generator backend prepare
    ↓
q1_0_g128-native kernel path
```

### Request Flow

```
structured chat messages
    ↓
generation_formatter_contract
    ↓
text::conditioner::sm
    ↓
tokenizer (existing BPE/Qwen path)
    ↓
generator/prefill + decode
    ↓
renderer/output
```

### Critical Data Flows

1. **GGUF truth to runtime contract:** The live GGUF header is the authority. It reports `general.architecture=qwen3`, `general.file_type=41`, a single `tokenizer.chat_template`, `tokenizer.ggml.model=gpt2`, `tokenizer.ggml.pre=qwen2`, and an executable vocab size of `151669` through the token array / embedding tensor shape. Architecture decisions should follow that, not prose.
2. **Raw dtype preservation:** Bonsai carries `197` tensors of raw type `41` and `113` tensors of raw type `0` in the sampled header. If EMEL collapses raw and prepared dtypes into one numeric space, the runtime will mis-route Bonsai before kernels even run.
3. **Quantized audit propagation:** Bonsai fits the existing audit model cleanly once `Q1_0_g128` is recognized as native quantized for 2D projection stages while 1D norms remain dense-by-contract.
4. **Conditioning contract:** The live template matches EMEL's current Qwen formatter markers, so the public request shape can stay `structured_chat_messages_v1` with `add_generation_prompt=true`, `enable_thinking=false`, and no tools. Bonsai changes the runtime/model contract, not the API shape.

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `gguf::loader` ↔ `model::data` | Direct tensor/kv structs | Must preserve raw type `41` |
| `model::data` ↔ `generator::detail` | Direct bound tensor records | Bonsai should reuse existing Qwen3 tensor naming and block lookup |
| `generator::initializer` ↔ `text::conditioner` | Existing synchronous event dispatch | No Bonsai-specific actor needed |
| `generator::prefill` ↔ `generator::detail` | Explicit compute contracts | Add Bonsai-specific contracts only if route identity must be published |
| `generator::detail` ↔ `kernel::*` | Kernel event structs | Main hotspot for `Q1_0_g128 x Q8_*` support |
| `tools/paritychecker` ↔ `tools/generation_formatter_contract` | Fixture-scoped contract resolution | Keep maintained Bonsai request contract anchored to live GGUF metadata |

### External Services / Truth Sources

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Hugging Face `prism-ml/Bonsai-1.7B-gguf` | Maintained fixture + metadata truth | Actual file is `Bonsai-1.7B.gguf`; quickstart prose uses a different filename |
| Prism `llama.cpp` fork | Arithmetic/kernel reference only | Use for operand semantics and block layout, not orchestration |
| Prism `Bonsai-demo` repo | Runtime support truth | Confirms required kernels are not upstream and come from Prism forks |

## Hotspots That Force Architectural Change

### Hotspot 1: Raw-vs-prepared dtype collision

**What breaks:** EMEL currently reserves internal prepared dtypes in the same numeric space where the Bonsai GGUF file uses raw type `41`.
**Impact:** Loader validation, tensor-type naming, generator routing, and kernel dispatch can all become wrong even before arithmetic is implemented.
**Action:** Fix this first. Everything else depends on it.

### Hotspot 2: Loader support for `Q1_0_g128`

**What breaks:** `gguf::loader::detail::ggml_layout()` does not currently describe raw type `41`, so Bonsai cannot be truthfully parsed.
**Impact:** The maintained fixture cannot enter the existing `model::loader` pipeline.
**Action:** Add the raw block layout for `Q1_0_g128` and make row-size math consistent with Prism's `128`-element block and `18`-byte storage.

### Hotspot 3: Native kernel path, not dense fallback

**What breaks:** The milestone is not honest if Bonsai is "supported" by whole-tensor dequantize-to-f32 in the shipped hot path.
**Impact:** Invalid parity and benchmark claims.
**Action:** Port the operand math into EMEL-owned kernels. Use Prism/ggml arithmetic as the reference, not its runtime control flow.

### Hotspot 4: Explicit compute route publication

**What breaks:** If Bonsai reuses old contract names that imply a different operand class, parity and benchmark output become misleading.
**Impact:** Roadmap and publication drift away from actual runtime behavior.
**Action:** If `Q1_0_g128` changes route identity, add explicit Bonsai route labels/contracts rather than reusing `*_packed_q8_0` or `*_q8_k` names.

## Implementation Build Order

1. **Fixture truth first**
   - Add `tests/models/Bonsai-1.7B.gguf` metadata, checksum, and maintained registry entry.
   - Lock the actual filename and executable metadata, not the prose examples.

2. **Fix dtype plumbing before any runtime work**
   - Extend `src/emel/gguf/loader/detail.hpp` for raw type `41`.
   - Separate raw GGUF dtypes from EMEL-prepared/internal dtypes.
   - Extend model tensor-type naming and quantized-audit recognition.

3. **Prove the request contract stays narrow**
   - Validate that the live `tokenizer.chat_template` continues to match the existing Qwen formatter resolver.
   - If it drifts, change `tools/generation_formatter_contract.hpp`, not the generator actor graph.

4. **Add EMEL-owned `Q1_0_g128` kernel primitives**
   - Add shared storage structs, row-size helpers, dequant helpers, and vec-dot math in `src/emel/kernel/detail.hpp`.
   - Add native backend execution in `src/emel/kernel/aarch64` and a correctness-oriented shared path for tests.
   - Do not ship a whole-tensor dense fallback.

5. **Wire generator backend preparation**
   - Teach `src/emel/generator/detail.hpp` to bind `qwen3` Bonsai tensors and to reject unsupported `Q1_0_g128` routes explicitly.
   - Keep `generator::initializer` unchanged.
   - Extend `generator/prefill` only if the compute contract must explicitly expose a Bonsai route.

6. **Extend maintained proof surfaces**
   - Add Bonsai to `tools/paritychecker`, `tools/bench`, and publication metadata.
   - Preserve the split-lane rule: EMEL result from EMEL-owned code, reference result from Prism fork only for comparison.

7. **Close with regression protection**
   - Loader tests for raw type `41`.
   - Kernel tests for `Q1_0_g128`.
   - Generator lifecycle tests on Bonsai fixture.
   - Parity and benchmark publication on the maintained slice.

## Anti-Patterns

### Anti-Pattern 1: Creating a `bonsai` architecture family in `src/emel/model`

**What people do:** Add a parallel Bonsai model family and duplicate Qwen3 block lookup.
**Why it's wrong:** The live GGUF already says Bonsai is `qwen3`.
**Do this instead:** Reuse Qwen3 topology and isolate the change to quantized operand support.

### Anti-Pattern 2: Leaving raw type `41` overloaded with EMEL prepared formats

**What people do:** Parse the file and hope internal type `41` still means what the runtime thinks it means.
**Why it's wrong:** That silently corrupts routing.
**Do this instead:** Split raw and prepared dtype spaces before enabling Bonsai loading.

### Anti-Pattern 3: Claiming support through dense fallback

**What people do:** Dequantize Bonsai weights to dense floats in the shipped hot path just to get parity green.
**Why it's wrong:** It violates the milestone's performance and honesty constraints.
**Do this instead:** Reject unsupported routes until native `Q1_0_g128` kernels exist.

### Anti-Pattern 4: Moving formatter resolution into the generator core

**What people do:** Make the generator own Bonsai-specific template parsing.
**Why it's wrong:** The generator is intentionally formatter-agnostic today.
**Do this instead:** Keep maintained contract resolution in tooling and inject the formatter function.

## Sources

- EMEL repo files inspected directly on branch `feat/bonsai`: `src/emel/gguf/loader/detail.hpp`, `src/emel/model/data.cpp`, `src/emel/kernel/detail.hpp`, `src/emel/generator/detail.hpp`, `src/emel/generator/prefill/*`, `src/emel/text/conditioner/*`, `tools/generation_formatter_contract.hpp`, `tools/generation_fixture_registry.hpp`, `tools/paritychecker/parity_runner.cpp`, `tools/bench/generation_bench.cpp`. Confidence: HIGH.
- Live Hugging Face repo tree: https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/tree/main . Confidence: HIGH.
- Live Hugging Face README: https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/raw/main/README.md . Confidence: HIGH.
- Live GGUF header sampled from `https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B.gguf` on 2026-04-02. Verified locally from the downloaded header region. Confidence: HIGH.
- Prism Bonsai demo README: https://github.com/PrismML-Eng/Bonsai-demo . Confidence: HIGH.
- Prism `llama.cpp` fork commit `1179bfc8295ed54e25f5cafa57dfecca373c61b8`, especially:
  - `include/llama.h`
  - `ggml/include/ggml.h`
  - `ggml/src/ggml-common.h`
  - `ggml/src/ggml-cpu/arch/arm/quants.c`
  Confidence: HIGH.

---
*Architecture research for: Bonsai-1.7B integration into EMEL*
*Researched: 2026-04-02*
