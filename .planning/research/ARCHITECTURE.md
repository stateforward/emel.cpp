# Architecture Research

**Domain:** Flash-attention integration for the existing EMEL canonical Llama-68M generation slice
**Researched:** 2026-03-12
**Confidence:** HIGH

## Standard Architecture

### System Overview

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Verification Surfaces                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  tools/paritychecker        tools/bench                                     │
│  generation parity          canonical compare benchmark                     │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ existing initialize/generate flow
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Orchestration Layer                                │
├──────────────────────────────────────────────────────────────────────────────┤
│  src/emel/generator::sm                                                    │
│  - conditioning                                                            │
│  - planning                                                                │
│  - memory reservation/snapshot                                             │
│  - graph compute dispatch                                                  │
│  - sampling / rendering                                                    │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ existing graph::event::compute callbacks
┌──────────────────────────────────────────────────────────────────────────────┐
│                            Compute Layer                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│  src/emel/graph::sm                                                         │
│    └── src/emel/graph::processor::sm                                        │
│         validate → prepare → alloc → bind → kernel → extract               │
│                            │                                                 │
│                            └── run_kernel callback into                      │
│                                src/emel/generator/detail.hpp                │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ existing kernel actor dispatch
┌──────────────────────────────────────────────────────────────────────────────┐
│                             Kernel Layer                                    │
├──────────────────────────────────────────────────────────────────────────────┤
│  src/emel/kernel::sm                                                        │
│    ├── x86_64::sm                                                           │
│    └── other backend actors (unchanged this milestone)                      │
│                                                                              │
│  New behavior for milestone: real op_flash_attn_ext execution in the        │
│  shared/scalar path plus x86_64 fast path                                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| `src/emel/generator` | Own end-to-end generation orchestration and persistent session/backend state | Existing Boost.SML actor plus `detail.hpp` native backend callbacks |
| `src/emel/graph` | Keep compute execution structured as reserve/assemble/execute phases | Existing Boost.SML graph actor and processor actor |
| `src/emel/generator/detail.hpp` native backend callbacks | Bind model tensors, KV cache, and request-local token/position inputs to the graph processor callback contract | Existing callback-based compute implementation; this is where flash attention should replace the current explicit score/probability path |
| `src/emel/kernel` | Own opcode-level execution and backend dispatch | Existing kernel actor with per-backend subactors and typed op events |
| `tools/paritychecker` | Verify canonical generation correctness against `llama.cpp` | Existing tool surface; should remain the acceptance boundary |
| `tools/bench` | Publish canonical EMEL vs `llama.cpp` generation timings | Existing benchmark surface; should remain the only benchmark integration point for the milestone |

## Recommended Project Structure

```text
src/
├── emel/
│   ├── generator/
│   │   ├── context.hpp      # Owns backend and generation session state
│   │   ├── actions.hpp      # Dispatches existing graph compute requests
│   │   ├── events.hpp       # Existing initialize/generate payloads
│   │   └── detail.hpp       # Native backend compute callbacks; flash-attn integration point
│   ├── graph/
│   │   ├── sm.hpp           # Reserve/compute orchestration stays unchanged
│   │   ├── events.hpp       # Existing callback-based compute contract
│   │   └── processor/
│   │       ├── sm.hpp       # Existing phase pipeline stays unchanged
│   │       └── kernel_step/ # Existing run_kernel callback phase stays unchanged
│   └── kernel/
│       ├── events.hpp       # Existing op_flash_attn_ext event surface
│       ├── detail.hpp       # Shared/scalar flash-attn validation + execution
│       ├── x86_64/
│       │   ├── actions.hpp  # x86_64 fast path for flash attention
│       │   ├── guards.hpp   # Existing valid/simd/invalid dispatch guards
│       │   └── sm.hpp       # Existing row already routes op_flash_attn_ext
│       └── [other backends]/ # Leave unchanged for this milestone
tools/
├── paritychecker/
│   └── parity_runner.cpp    # Existing generation acceptance surface
└── bench/
    └── generation_bench.cpp # Existing canonical compare surface
tests/
├── kernel/                  # Add flash-attn operator tests here
└── generator/               # Add generation integration tests here
```

### Structure Rationale

- **`src/emel/generator/`:** keep flash-attention adoption inside the shipped generation actor rather than creating a sidecar runtime path.
- **`src/emel/graph/`:** preserve the current compute pipeline contract so flash attention remains a callback implementation detail, not a graph-orchestration redesign.
- **`src/emel/kernel/`:** the opcode already exists here, so this is the correct place for the fused data-plane implementation.
- **`tools/paritychecker/` and `tools/bench/`:** reuse the existing acceptance surfaces instead of inventing milestone-specific tooling.

## Architectural Patterns

### Pattern 1: Callback-Owned Compute, Actor-Owned Orchestration

**What:** Keep Boost.SML actors responsible for phase ordering, while the heavy numeric work stays in bounded callback/detail kernels.
**When to use:** When a milestone changes math or kernels but should not change the orchestration graph.
**Trade-offs:** Preserves repo invariants and minimizes churn, but it requires discipline to keep new behavior inside existing callback boundaries.

**Example:**
```cpp
emel::graph::event::compute compute_ev{
  .step_plan = &ctx.compute.prefill_plan,
  .compute_ctx = &ev.ctx.io,
  .validate = emel::generator::detail::validate,
  .prepare_graph = emel::generator::detail::prepare_graph,
  .alloc_graph = emel::generator::detail::alloc_graph,
  .bind_inputs = emel::generator::detail::bind_inputs,
  .run_kernel = emel::generator::detail::run_kernel,
  .extract_outputs = emel::generator::detail::extract_outputs,
};
```

### Pattern 2: Replace Data-Plane Logic, Preserve Actor Graph

**What:** Swap the attention math inside `generator/detail.hpp` and `kernel/detail.hpp` while keeping the existing generator, graph, and processor states intact.
**When to use:** When the milestone is algorithmic rather than orchestration-focused.
**Trade-offs:** Safest for parity and regression control, but it intentionally avoids broader architectural cleanup.

**Example:**
```cpp
// Current architecture-safe change:
// run_layer() still computes q/k/v projections and writes KV cache,
// but attention context is produced by kernel.process_event(op_flash_attn_ext)
// instead of explicit attn_scores/attn_probs materialization.
```

### Pattern 3: Persistent Backend-Owned Workspace

**What:** Keep flash-attention scratch/workspace in the persistent backend owned by `generator::action::context::compute.backend`, not in dispatch-local SML context.
**When to use:** When a kernel needs reusable buffers across many top-level dispatches.
**Trade-offs:** Matches repo rules and avoids per-dispatch allocation, but it means buffer ownership must stay tightly scoped to the existing backend object.

## Data Flow

### Request Flow

```text
tools/paritychecker or tools/bench
    ↓
generator::sm.process_event(generate)
    ↓
generator actions build graph::event::compute
    ↓
graph::sm → graph::processor::sm
    ↓
validate → prepare → alloc → bind → kernel → extract
    ↓
generator::detail::run_kernel()
    ↓
run_prefill()/run_decode()
    ↓
run_layer()
    ↓
q/k/v projections + rope + KV cache write
    ↓
kernel.process_event(op_flash_attn_ext)
    ↓
attention output returned to generator backend
    ↓
output projection → FFN → logits extraction
    ↓
sampling / rendering / parity or benchmark reporting
```

### State Management

```text
generator::action::context
    ↓ owns
graph_binding.backend (persistent native backend)
    ↓ owns
model execution view, topology, plans, KV cache, logits/workspace, kernel actor
    ↓ receives per-dispatch input through
generator::event::generate_ctx::io + graph::event::compute payload
```

### Key Data Flows

1. **Prefill flow:** prompt tokens are conditioned and planned in the generator, then passed to graph compute, then bound into the native backend, which computes each layer and calls `op_flash_attn_ext` after Q/K/V projection and RoPE.
2. **Decode flow:** one sampled token plus current `kv_tokens` is passed through the same graph processor contract, with flash attention consuming the existing KV cache and current token position for single-token decode.
3. **Verification flow:** paritychecker and bench do not call flash attention directly; they exercise the same generator surface and observe the results through existing success/error and compare/reporting contracts.

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| Canonical single-model milestone | Keep all work inside existing generator/graph/kernel components; no new actors are needed |
| More models in same family | Add shape/feature validation around the same callback seam before broadening tool acceptance |
| Broader backend rollout | Introduce backend-specific fast paths behind the same `op_flash_attn_ext` event, not new generator architectures |

### Scaling Priorities

1. **First bottleneck:** the current generator attention implementation in `src/emel/generator/detail.hpp`; replace explicit score/probability materialization with fused flash-attn dispatch before attempting broader backend work.
2. **Second bottleneck:** backend specialization in `src/emel/kernel/x86_64/actions.hpp`; only after the shared/scalar path is correct should x86_64-specific optimization become the focus.

## Anti-Patterns

### Anti-Pattern 1: New Flash-Attention Actor or Parallel Orchestration Path

**What people do:** Add a dedicated flash-attention state machine or tool-local compute actor.
**Why it's wrong:** It duplicates orchestration, expands the call graph, and violates the repo preference that `src/` machines remain the source of truth.
**Do this instead:** Keep flash attention as an opcode implementation behind the existing generator → graph → processor → kernel chain.

### Anti-Pattern 2: Put Algorithm Selection Logic in Generator Actions

**What people do:** Branch in `generator/actions.hpp` between old attention and flash attention, or introduce per-dispatch mode flags in context.
**Why it's wrong:** The repo rules forbid runtime branching in actions and forbid storing dispatch-local phase/control flags in context.
**Do this instead:** Make the canonical generation backend use one implementation path for the milestone, with shape validation in kernel/detail and backend selection in existing kernel guards.

### Anti-Pattern 3: Change Graph/Processor Phases for a Kernel Milestone

**What people do:** Add extra graph phases like "flash_prepare" or "attention_execute."
**Why it's wrong:** The graph processor already has the necessary `bind → kernel → extract` contract, and phase churn increases risk across paritychecker and bench for no milestone benefit.
**Do this instead:** Keep flash attention inside the existing `run_kernel` callback and only extend data-plane helpers or backend-owned workspace as needed.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| `tools/paritychecker` reference `llama.cpp` context | Tool-only reference comparison | Keep it outside `src/emel`; align reference flash-attention settings only after the EMEL path is real |
| `tools/bench` reference `llama.cpp` context | Tool-only compare benchmark | Reuse the existing compare workflow and canonical case names |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `generator::sm` ↔ `graph::sm` | `graph::event::reserve` / `graph::event::compute` | No new events needed for the milestone; keep the existing callback bundle |
| `graph::processor::kernel_step` ↔ `generator::detail` | `run_kernel` callback | Safest place to swap in flash attention without changing processor states |
| `generator::detail` ↔ `kernel::sm` | `kernel.process_event(op_flash_attn_ext)` | This is the main new runtime integration point |
| `kernel::sm` ↔ `kernel/x86_64::sm` | Existing backend subactor dispatch | x86_64 specialization should remain behind the same op event |
| `generator::action::context::compute.backend` ↔ `generator::event::generate_ctx::io` | Pointer-based same-RTC handoff | Keep request-local inputs in `generate_ctx::io`; keep persistent workspace in backend-owned storage |

## New vs Modified Components

### Modified Runtime Components

- `src/emel/generator/detail.hpp`
  - Replace `compute_attention()` usage inside `run_layer()` with `op_flash_attn_ext` dispatch.
  - Remove or stop depending on `attn_scores` / `attn_probs` as the primary path.
  - Add any reusable flash-attn workspace to `native_backend`.
- `src/emel/generator/context.hpp`
  - Only if additional persistent backend workspace fields are needed.
  - Do not add dispatch-local flash mode or request fields.
- `src/emel/kernel/detail.hpp`
  - Add shared/scalar validation and execution for `op_flash_attn_ext`.
  - Keep it limited to the causal self-attention subset needed by canonical generation.
- `src/emel/kernel/x86_64/actions.hpp`
  - Add x86_64 fast-path execution for `op_flash_attn_ext` behind the existing dispatch pattern.
- `tools/paritychecker/parity_runner.cpp`
  - Keep the same generation CLI and acceptance messaging, but align the reference path once EMEL flash attention is live.
- `tools/bench/generation_bench.cpp`
  - Keep the same benchmark cases and compare flow, but align reference execution once EMEL flash attention is live.

### New Runtime Components

- **None required at the actor/component level.**
  - No new `src/emel/<component>/sm.hpp` should be added for this milestone.
  - No new public events or public API surfaces are required if the canonical causal slice uses the existing `op_flash_attn_ext` event schema.

### New Test/Verification Components

- `tests/kernel/...`
  - Add flash-attn operator tests scoped to the kernel component.
- `tests/generator/...`
  - Add or extend generation integration tests proving the canonical flash path stays inside the existing acceptance boundary.

## Safest Build Order

1. **Shared kernel contract first**
   - Implement `op_flash_attn_ext` validation and scalar/shared execution in `src/emel/kernel/detail.hpp`.
   - Reason: this creates a correct, backend-neutral operator before touching generator behavior.

2. **Generator backend integration second**
   - Modify `src/emel/generator/detail.hpp` so `run_layer()` uses the new operator while preserving existing graph/generator orchestration.
   - Reason: this is the first point where the shipped runtime actually changes.

3. **x86_64 fast path third**
   - Add `src/emel/kernel/x86_64/actions.hpp` specialization after scalar correctness is established.
   - Reason: optimize only after the canonical path is functionally correct.

4. **Runtime-focused tests fourth**
   - Add kernel and generator tests around the new operator path before changing acceptance tools.
   - Reason: paritychecker and bench failures are easier to interpret once lower-level correctness is already covered.

5. **Paritychecker integration fifth**
   - Keep `tools/paritychecker --generation` unchanged at the surface, then align the reference flash-attention path and confirm canonical parity.
   - Reason: parity is the primary milestone gate and depends on the runtime being real first.

6. **Bench integration sixth**
   - Update `tools/bench/generation_bench.cpp` compare behavior after parity is stable.
   - Reason: benchmark evidence is only meaningful once correctness is already locked.

7. **Quality gates last**
   - Run the normal repo gates after implementation and verification are wired.
   - Reason: this preserves the repo's existing acceptance order and avoids snapshot/bench churn while the runtime path is still moving.

## Sources

- `.planning/PROJECT.md` - milestone goal, active requirements, and acceptance boundaries. Confidence: HIGH.
- `AGENTS.md` - repo-specific architectural and SML integration rules. Confidence: HIGH.
- `docs/rules/sml.rules.md` - RTC/no-queue and bounded-action semantics that constrain flash-attention integration. Confidence: HIGH.
- `src/emel/generator/context.hpp` - current generator ownership boundary for backend state. Confidence: HIGH.
- `src/emel/generator/actions.hpp` - current generator → graph compute dispatch seam. Confidence: HIGH.
- `src/emel/generator/detail.hpp` - current native backend and attention implementation to be modified. Confidence: HIGH.
- `src/emel/graph/events.hpp` and `src/emel/graph/processor/events.hpp` - current callback contract for compute execution. Confidence: HIGH.
- `src/emel/graph/sm.hpp` and `src/emel/graph/processor/sm.hpp` - current orchestration phases that should remain unchanged. Confidence: HIGH.
- `src/emel/kernel/events.hpp`, `src/emel/kernel/detail.hpp`, `src/emel/kernel/context.hpp`, `src/emel/kernel/x86_64/actions.hpp`, `src/emel/kernel/x86_64/sm.hpp` - existing operator/event/backend dispatch surfaces for flash attention. Confidence: HIGH.
- `.planning/research/STACK.md` and `.planning/research/FEATURES.md` - supporting stack and milestone-scope conclusions for the same milestone. Confidence: HIGH.

---
*Architecture research for: EMEL v1.2 flash attention*
*Researched: 2026-03-12*
