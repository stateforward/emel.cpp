# Phase 11: Generator Flash Adoption - Research

**Researched:** 2026-03-21
**Domain:** EMEL generator/runtime flash-attention adoption for the canonical CPU-hosted Llama-68M
slice
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Phase Boundary

Phase 11 adopts the Phase 10 flash-attention kernel inside the shipped canonical generation path
under `src/emel/generator` without changing Boost.SML orchestration, public APIs, or tool-facing
contracts. This phase is about generator/runtime truth only: supported canonical requests must run
through the real `op_flash_attn_ext` backend route, and unsupported requests must behave
deterministically without silently claiming flash execution.

## Implementation Decisions

### Flash Selection Boundary
- Phase 11 only targets the canonical CPU-hosted Llama-68M generation slice already shipped by the
  generator.
- Flash adoption happens inside `src/emel/generator/detail.hpp` after Q/K/V matmuls, RoPE, and K/V
  cache writes; Boost.SML orchestration stays unchanged.

### Unsupported Request Behavior
- If the generator cannot form a canonical `op_flash_attn_ext` request, the request must fail
  explicitly instead of silently labeling the old materialized attention path as flash.
- Broader non-flash fallback policy is deferred.

### Proof Of Flash Adoption
- Generator-owned observability must distinguish flash-attention dispatch from generic kernel
  matmul traffic.
- Phase 11 proof stays local to generator/runtime tests and state inspection. Parity and benchmark
  publication remain deferred.

### Operand And Cache Contract
- The generator must feed `op_flash_attn_ext` the same position-major key/value cache layout it
  already owns today.
- No new public C API, CLI flag, or runtime configuration knob is introduced.

### Deferred Ideas (OUT OF SCOPE)
- User-visible flash proof in paritychecker belongs to Phase 12.
- Benchmark evidence belongs to Phase 13.
- Broader fallback policy, multi-model rollout, and backend-specific flash tuning are out of scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GEN-01 | The shipped canonical generation path under `src/emel/generator` dispatches through flash attention for supported Llama-68M requests. | The existing generator already owns a real `emel::kernel::sm` and builds kernel tensor views for matmul. The missing work is replacing the materialized `compute_attention(...)` call in `run_layer(...)` with a real `op_flash_attn_ext` dispatch built over the existing Q/K/V/cache buffers. |
| GEN-02 | Unsupported requests publish explicit deterministic behavior when flash attention is not selected or cannot run; the runtime must not silently claim flash-path execution. | Phase 10 already rejects non-canonical flash requests explicitly. Phase 11 can preserve truth by propagating that failure up the generator path instead of hiding it behind the old materialized attention implementation. |
</phase_requirements>

## Summary

The current generator still computes attention locally in
`src/emel/generator/detail.hpp::compute_attention(...)`, materializing `attn_scores`,
`attn_probs`, and `attn_ctx` from the generator-owned `key_cache` and `value_cache`. The exact
adoption seam is already isolated in `run_layer(...)`: after Q/K/V matmuls, RoPE, and K/V cache
writes, the generator currently calls `compute_attention(...)` and then dispatches the attention
output matmul through the kernel state machine. Phase 11 should replace only that attention seam,
not the surrounding SML orchestration.

The workspace already contains useful groundwork. The current `generator::sm` constructor prepares
the native generator backend eagerly and exposes `generation_kernel_kind()` plus
`generation_kernel_dispatch_calls()` from `src/emel/generator/sm.hpp`. Generator lifecycle tests
have also already moved away from a fake backend harness toward the native generator contract.
That means Phase 11 does not need a new backend abstraction or new state-machine rows; it needs the
real flash dispatch in `generator/detail.hpp`, plus flash-specific observability so tests can prove
that the adopted path executed.

The best narrow implementation is to construct `emel::kernel::event::op_flash_attn_ext` directly
from `backend.q`, `backend.key_cache`, `backend.value_cache`, and `backend.attn_ctx` using the
existing position-major cache layout and the same kernel actor already used for matmul. If the
kernel rejects the request, the generator should fail explicitly. That preserves truthful flash
claims, avoids hidden fallback branching, and keeps later parity/benchmark phases honest.

**Primary recommendation:** Replace generator-side materialized attention with a real
`op_flash_attn_ext` dispatch in `src/emel/generator/detail.hpp`, add flash-specific generator
observability for automated proof, and extend generator lifecycle coverage for canonical success
plus deterministic unsupported-request failure.

## Project Constraints (from AGENTS.md)

Phase 11 must honor the repo contract:

- Keep Boost.SML orchestration unchanged unless the user explicitly approves structural changes.
- Preserve RTC/no-queue behavior and avoid self-dispatch.
- Keep actions bounded, non-blocking, and allocation-free during dispatch.
- Do not silently claim flash-path execution on a non-flash path.
- Treat performance as a first-class constraint and avoid hot-path heap allocation.
- Use doctest and existing CTest targets for automated proof.
- Run `scripts/quality_gates.sh` after implementation changes.
- Keep runtime/API scope inside the existing generator -> graph -> processor -> kernel chain.

## Standard Stack

### Core
| Library / Component | Version | Purpose | Why Standard |
|---------------------|---------|---------|--------------|
| `src/emel/generator/detail.hpp` native backend | repo local | Generator data-plane execution and attention seam | This is where attention is still materialized today and where flash adoption must occur |
| `src/emel/kernel/detail.hpp` + backend `sm` wrappers | repo local | Canonical flash-attention contract and execution | Phase 10 already verified this path and Phase 11 must reuse it rather than duplicating math |
| Boost.SML generator/graph/kernel machines | pinned repo state | Orchestration | Milestone goal explicitly says orchestration stays unchanged |

### Supporting
| Library / Component | Version | Purpose | When to Use |
|---------------------|---------|---------|-------------|
| `tests/generator/lifecycle_tests.cpp` | repo local | Generator-runtime truth tests | Canonical generation success/failure coverage |
| `tools/paritychecker/paritychecker_tests.cpp` | repo local | Existing observability pattern reference | Useful to align future flash proof without changing Phase 11 scope |
| `scripts/quality_gates.sh` | repo local | Full-repo regression gate | Required after implementation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Direct `op_flash_attn_ext` dispatch from `run_layer(...)` | Keep `compute_attention(...)` and add a flash-specific helper beside it | Duplicates attention logic and weakens proof that the real kernel path ran |
| Explicit failure on unsupported flash formation | Silent fallback to the old materialized path | Easier short-term behavior retention, but violates truthful flash-claim intent and obscures later parity proof |
| Generator-local flash observability counters | Infer flash execution from generic kernel dispatch counts | Too weak because matmul already increments `kernel_dispatch_calls` |

## Architecture Patterns

### Recommended Project Structure
```text
src/
├── emel/generator/detail.hpp       # Replace materialized attention seam with flash dispatch
├── emel/generator/context.hpp      # Generator-owned runtime state / observability
├── emel/generator/sm.hpp           # Existing getters; likely extend with flash-specific getter
├── emel/kernel/detail.hpp          # Canonical flash request contract from Phase 10
└── emel/kernel/events.hpp          # op_flash_attn_ext event shape

tests/
├── generator/lifecycle_tests.cpp   # Canonical success/failure + flash observability proof
└── generator/detail_tests.cpp      # Optional focused request-shape or helper coverage
```

### Pattern 1: Replace The Attention Seam, Not The Surrounding Runtime
**What:** Keep `run_layer(...)` structure intact, but swap the `compute_attention(...)` call for a
kernel event dispatch that writes into `backend.attn_ctx`.

**When to use:** Canonical supported generation path.

**Planner implication:** Tasks should stay inside generator detail/context/sm/tests rather than
touching state-machine transition structure.

### Pattern 2: Reuse The Existing Cache Layout Verbatim
**What:** Build the flash request over the current generator-owned position-major cache layout
rather than inventing a new intermediate representation.

**When to use:** Every Phase 11 flash request.

**Planner implication:** Request-building helpers need exact tensor-view dimensions/strides for
`backend.q`, `backend.key_cache`, `backend.value_cache`, and `backend.attn_ctx`.

### Pattern 3: Flash Proof Needs Its Own Signal
**What:** Add a flash-specific counter or last-path field inside `native_backend` so tests can
prove flash selection separately from generic kernel traffic.

**When to use:** Automated generator proof and later parity integration.

**Planner implication:** At least one task should add generator observability and at least one test
must assert it.

### Anti-Patterns To Avoid
- **Hidden materialized fallback under a flash label:** This would make later parity/benchmark
  claims untrustworthy.
- **Bypassing the kernel actor:** Calling shared kernel helpers directly from generator code would
  fail the phase goal of using the existing generator -> graph -> processor -> kernel chain.
- **Tool-only observability:** Flash proof must live in `src/` state the runtime actually owns, not
  in a paritychecker-only shim.
- **Scope creep into parity/bench outputs:** Phase 11 should not update benchmark surfaces or claim
  end-to-end parity completion.

## Validation Architecture

- **Quick feedback target:** `ctest --output-on-failure -R generator`
- **Full regression target:** `scripts/quality_gates.sh`
- **Primary proof points:**
  - Canonical generation succeeds and produces the expected one-token result through the native
    generator contract.
  - Flash-specific generator observability reports at least one flash dispatch on the canonical
    path.
  - A deliberately non-canonical or non-selecting flash request fails deterministically and does
    not increment the flash-specific counter.
- **Likely test surface:** extend `tests/generator/lifecycle_tests.cpp`; optionally add focused
  helper tests in `tests/generator/detail_tests.cpp` if request construction becomes non-trivial.

## Suggested Plan Shape

Two plans should be enough:

1. **Generator flash request wiring and observability**
   - Add request-building/helper code in `src/emel/generator/detail.hpp`
   - Dispatch `op_flash_attn_ext` from `run_layer(...)`
   - Record flash-specific runtime observability in generator-owned state

2. **Generator adoption proof and deterministic negative behavior**
   - Extend generator lifecycle tests for canonical flash success
   - Add explicit unsupported-request failure coverage
   - Verify flash-specific observability and ensure no silent fallback claim

---
*Phase: 11-generator-flash-adoption*
*Researched: 2026-03-21*
