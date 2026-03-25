# Project Research Summary

**Project:** EMEL
**Domain:** CPU-hosted flash attention for the canonical Llama-68M generation slice
**Researched:** 2026-03-12
**Confidence:** HIGH

## Executive Summary

EMEL v1.2 is not a generic flash-attention project. It is a narrow brownfield milestone: replace
the current materialized attention path in the shipped CPU Llama-68M generation flow with a real
EMEL-owned fused flash-attention path, then prove it through the repo's existing acceptance
surfaces. The correct product shape is therefore "one real runtime path in `src/emel`, verified by
`tools/paritychecker` and measured by `tools/bench`," not a new API, a new actor graph, or a
tool-only prototype.

The recommended implementation strategy is equally narrow. Keep Boost.SML orchestration unchanged,
implement `op_flash_attn_ext` in the existing kernel layer, integrate it into
`src/emel/generator/detail.hpp`, validate correctness with a shared/scalar path before any x86_64
optimization, and only then align the `llama.cpp` reference settings in paritychecker and bench so
EMEL and the reference are executing the same algorithm class. That order matches the dependency
graph and minimizes ambiguous failures.

The main risks are false completion and misleading evidence. The milestone fails if EMEL still
ships the old attention path under a flash-attention label, if parity drifts while optimization is
in flight, or if the reference tools remain configured for non-flash attention and the repo still
claims parity. Mitigation is straightforward: make generator runtime adoption the first truth
point, lock correctness before SIMD work, reuse the canonical benchmark/parity workloads unchanged,
and add enough tests or seam visibility to prove the flash path actually ran.

## Key Findings

### Recommended Stack

Research strongly converges on an EMEL-native implementation. No new external runtime, framework,
or API surface is justified for this milestone. The right stack is the existing generator, graph,
and kernel actor chain, with `op_flash_attn_ext` implemented in `src/emel/kernel/**` and consumed
from `src/emel/generator/detail.hpp`. `llama.cpp` and `ggml` remain reference-only dependencies in
`tools/paritychecker` and `tools/bench`.

**Core technologies:**
- EMEL-owned `op_flash_attn_ext` in [`src/emel/kernel/detail.hpp`](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/detail.hpp) and [`src/emel/kernel/x86_64/actions.hpp`](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/kernel/x86_64/actions.hpp): exact fused attention operator and x86_64 fast path without widening runtime architecture.
- Existing generator backend in [`src/emel/generator/detail.hpp`](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/generator/detail.hpp): narrowest integration point because it already owns Q/K/V projection, KV cache state, and graph binding.
- Boost.SML v1.1.13: preserves RTC/no-queue orchestration and keeps flash attention as a data-plane change rather than a control-plane rewrite.
- Existing doctest, paritychecker, and bench surfaces: verification must stay on the repo's shipped acceptance boundaries rather than a custom flash-only harness.

### Expected Features

The table stakes are not broad feature coverage; they are real adoption in the canonical shipped
path and truthful verification. The milestone is done when the CPU Llama-68M generation slice
executes EMEL-owned flash attention through the normal initialize/generate flow, canonical
generation parity still passes, and the canonical compare benchmark still reports against
`llama.cpp` on the same workload.

**Must have (table stakes):**
- Real EMEL-owned flash attention in canonical generation, not a tool-only proof or hidden fallback.
- Canonical parity through `tools/paritychecker --generation` on the existing Llama-68M fixture.
- Canonical compare visibility through existing `tools/bench` and `scripts/bench.sh --compare`.
- CPU-hosted deterministic execution with no new public API or CLI surface.

**Should have (competitive):**
- Minimal seam or dump visibility that proves the flash path was exercised when parity diverges.
- Clear benchmark labeling or metadata indicating the canonical row now uses flash attention.
- Coverage of both short and longer canonical generation cases once the base path is stable.

**Defer (v2+):**
- Multi-model rollout beyond the canonical Llama-68M slice.
- Full `ggml_flash_attn_ext` feature coverage such as sinks, softcap, or non-causal variants.
- GPU or broader backend expansion.
- Public knobs for selecting flash-attention behavior.

### Architecture Approach

The architecture recommendation is conservative by design: keep the generator, graph, processor,
and kernel actors exactly where they are, and replace only the data-plane attention implementation.
`src/emel/generator` remains the owner of generation orchestration and persistent backend state,
`src/emel/graph` keeps the existing compute callback contract, and `src/emel/kernel` owns opcode
execution plus backend specialization. No new actor, no new phase family, and no new public event
surface is required for the milestone.

**Major components:**
1. `src/emel/generator` — owns end-to-end generation flow, backend state, KV cache semantics, and the runtime call site that must switch to flash attention.
2. `src/emel/graph` / `src/emel/graph/processor` — keeps reserve/bind/kernel/extract orchestration unchanged so the milestone stays a kernel/backend change.
3. `src/emel/kernel` — owns `op_flash_attn_ext`, shared correctness path, and x86_64 optimization behind existing backend dispatch.
4. `tools/paritychecker` and `tools/bench` — remain the acceptance boundary and must be aligned to the same algorithm class after runtime adoption is real.

### Critical Pitfalls

1. **Shipping the old path under a flash-attention label** — avoid this by making [`src/emel/generator/detail.hpp`](/Users/gabrielwillen/VSCode/stateforward/emel/emel.cpp/src/emel/generator/detail.hpp) the first runtime truth point and refusing tool-only completion.
2. **Breaking exact generation parity while optimizing** — avoid this by landing a shared/scalar operator first, matching the canonical causal subset exactly, and treating paritychecker as the primary gate before SIMD work.
3. **Claiming parity while EMEL and `llama.cpp` use different operand classes or reference settings** — avoid this by aligning paritychecker and bench reference contexts after the EMEL flash path is live, and narrowing claims to end-to-end generation compare if operand classes still differ.
4. **Smuggling runtime branching or per-dispatch flags into SML actions/context** — avoid this by keeping one shipped runtime path for the milestone and confining control flow to existing actor structure and kernel guards.
5. **Adding allocation churn in prefill/decode** — avoid this by placing reusable flash workspace in persistent backend-owned state and not in per-dispatch vectors or temporary context fields.

## Implications for Roadmap

Based on the combined research, the roadmap should stay narrow and sequence work by dependency,
not by visible tooling. The kernel contract must exist before generator adoption, generator
adoption must be real before parity/tool alignment, and parity must be stable before benchmark
claims mean anything.

### Phase 1: Shared Kernel Bring-Up
**Rationale:** Everything else depends on having a correct `op_flash_attn_ext` contract in EMEL-owned runtime code.
**Delivers:** Shared/scalar validation and execution for the canonical causal self-attention subset, plus persistent workspace design.
**Addresses:** The must-have "real EMEL flash-attention path" requirement at the opcode level.
**Avoids:** Shipping tool-only flash attention, SML rule violations, and allocation regressions.

### Phase 2: Generator Runtime Adoption And Correctness
**Rationale:** The milestone is not real until canonical generation in `src/emel/generator` uses the fused operator instead of materialized score/probability buffers.
**Delivers:** Generator integration over the existing KV cache and decode/prefill flow, plus kernel/generator tests that prove the flash path is active and parity-safe.
**Uses:** Existing generator backend, graph callback seam, and doctest verification stack.
**Implements:** The generator-to-kernel integration boundary identified in architecture research.
**Avoids:** Old-path shipping, KV semantic drift, silent verification blind spots, and premature x86_64 optimization.

### Phase 3: x86_64 Fast Path And Parity Stabilization
**Rationale:** Optimize only after the shared path is trustworthy, otherwise failures are ambiguous.
**Delivers:** AVX2-backed `op_flash_attn_ext` specialization behind existing backend dispatch, with scalar A/B coverage still available.
**Addresses:** Performance expectations without changing the milestone surface or architecture.
**Avoids:** SIMD-only correctness drift and host-specific ambiguity.

### Phase 4: Reference Alignment And Benchmark Evidence
**Rationale:** Parity and benchmark surfaces must measure the same algorithm class after EMEL adopts flash attention.
**Delivers:** `tools/paritychecker` and `tools/bench` updated to align `llama.cpp` reference settings, with canonical workloads and row names unchanged.
**Addresses:** Must-have parity and benchmark visibility on the accepted repo surfaces.
**Avoids:** Operand-class mismatch, benchmark workload drift, and misleading parity claims.

### Phase Ordering Rationale

- Phase 1 precedes everything because the kernel/operator contract is the narrowest correctness foundation and the cleanest place to handle shape validation and scratch ownership.
- Phase 2 comes before tooling because the repo's definition of done is a real shipped runtime path in `src/emel`, not a convincing demo in `tools/`.
- Phase 3 is intentionally after Phase 2 because optimization before scalar correctness produces ambiguous bugs and slows parity work.
- Phase 4 is last because benchmark and parity evidence are only truthful once runtime adoption and correctness are already stable.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3:** x86_64 AVX2 implementation details may need targeted kernel-level research if the current repo primitives are insufficient for a clean fast path.
- **Phase 4:** reference-claim wording may need explicit planning if EMEL and `llama.cpp` still differ in effective operand format after flash-attention adoption.

Phases with standard patterns (skip research-phase):
- **Phase 1:** well-bounded kernel bring-up using existing EMEL opcode surfaces and established flash-attention references.
- **Phase 2:** well-defined brownfield integration through existing generator/graph/kernel seams; architectural uncertainty is low.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Strong agreement between repo constraints and external flash-attention references; no new runtime stack is needed. |
| Features | HIGH | The milestone boundary is explicit in `.planning/PROJECT.md`, existing tool surfaces, and AGENTS rules. |
| Architecture | HIGH | Existing component boundaries and callback seams are already present and map cleanly to the required change. |
| Pitfalls | HIGH | Risks are concrete, repo-specific, and repeated across stack, features, and architecture research. |

**Overall confidence:** HIGH

### Gaps to Address

- Exact operand-class equivalence between EMEL and `llama.cpp` after flash-attention adoption is still a validation task, not a completed fact; planning should state claim boundaries precisely.
- x86_64 optimization details are less certain than the shared/scalar path and should not be allowed to block correctness milestones.
- Minimal observability for "flash path definitely executed" is recommended, but the exact mechanism can stay lightweight and should be decided during phase planning rather than broadening scope now.

## Sources

### Primary (HIGH confidence)
- `.planning/PROJECT.md` — active milestone goal, scope, acceptance boundary, and out-of-scope constraints.
- `AGENTS.md` — repo-specific contract for SML structure, parity claims, hot-path behavior, and interim-fallback restrictions.
- `docs/rules/sml.rules.md` — RTC/no-queue and bounded-action semantics that constrain integration.
- `.planning/research/STACK.md` — recommended technologies, integration points, and non-goals.
- `.planning/research/FEATURES.md` — table stakes, differentiators, and anti-features for the milestone.
- `.planning/research/ARCHITECTURE.md` — component boundaries, patterns, and safe build order.
- `.planning/research/PITFALLS.md` — repo-specific failure modes and phase-by-phase prevention guidance.
- `src/emel/generator/detail.hpp`, `src/emel/kernel/detail.hpp`, `src/emel/kernel/x86_64/actions.hpp`, `tools/paritychecker/parity_runner.cpp`, and `tools/bench/generation_bench.cpp` — concrete runtime and verification touchpoints named consistently across the research.
- FlashAttention (Tri Dao et al., 2022), FlashAttention-2 (Tri Dao, 2023), and current `llama.cpp` / `ggml` flash-attention APIs and CPU implementation — external algorithm and reference behavior sources cited in stack research.

### Secondary (MEDIUM confidence)
- Dao-AILab `flash-attention` repository — useful only as negative evidence that the common external package is CUDA/ROCm-oriented and inappropriate for this CPU-hosted milestone.

### Tertiary (LOW confidence)
- None.

---
*Research completed: 2026-03-12*
*Ready for roadmap: yes*
