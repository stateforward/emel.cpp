# Pitfalls Research

**Domain:** Flash attention added to an existing CPU-hosted inference path with parity and benchmark
obligations
**Researched:** 2026-03-12
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Shipping a "flash attention" milestone that still runs the old attention path

**What goes wrong:**
The repo reports flash attention as delivered, but the canonical generation path still materializes
attention scores/probabilities in `src/emel/generator/detail.hpp`, or only tool-local code uses a
fused path.

**Why it happens:**
It is faster to add a benchmark demo, a tool-only experiment, or a partial kernel stub than to
replace the real runtime path. In a brownfield system this can look complete because the outputs
still match.

**How to avoid:**
Make `src/emel/generator/detail.hpp` the first runtime truth point. The milestone is not done until
`run_layer()` stops depending on the old `compute_attention()` materialization as the shipped path
and the fused operator is invoked from the real generation flow.

**Warning signs:**
- `tools/paritychecker` or `tools/bench` mention flash attention, but `generator/detail.hpp` still
  owns the old score/probability path.
- Only `tools/` files changed.
- The kernel event `op_flash_attn_ext` exists, but no real generator call site uses it.

**Phase to address:**
Phase 1: Kernel bring-up and runtime truth wiring.

---

### Pitfall 2: Breaking exact generation parity while chasing a faster kernel

**What goes wrong:**
The flash-attention path changes generated text, token counts, or logits enough that canonical
generation parity fails on the Llama-68M fixture.

**Why it happens:**
Online softmax, causal masking, head-grouping, RoPE positioning, or accumulation precision is
implemented slightly differently than the reference path. These errors often only show up at decode
time or after several layers.

**How to avoid:**
Bring up the shared/scalar operator first, match the canonical causal self-attention subset
exactly, and add focused kernel tests before x86_64 optimization. Treat paritychecker as the
primary acceptance gate, not an optional final check.

**Warning signs:**
- `generation parity mismatch` appears only for generated tokens, not initial prompt processing.
- Short runs pass but the 8-token case diverges.
- Max-logit or seam comparisons drift after the first decode step.

**Phase to address:**
Phase 2: Generator integration and parity stabilization.

---

### Pitfall 3: Claiming parity while EMEL and the reference path use different operand classes

**What goes wrong:**
EMEL is benchmarked or parity-checked against `llama.cpp`, but one side still uses non-flash
attention, a different KV layout, or a different effective operand path.

**Why it happens:**
The existing tool surfaces currently disable reference flash attention, and it is easy to forget to
realign them once the EMEL path changes.

**How to avoid:**
After the EMEL runtime path is real, explicitly align `tools/paritychecker` and `tools/bench`
reference contexts to the flash-attention algorithm class. If operand classes still differ, report
results as end-to-end generation comparison, not kernel parity.

**Warning signs:**
- `LLAMA_FLASH_ATTN_TYPE_DISABLED` remains in reference generation paths after EMEL flash-attn
  lands.
- Bench numbers improve dramatically but parity language still says "same path" without proof.
- Compare output is discussed as kernel parity even though the underlying operand pipelines differ.

**Phase to address:**
Phase 3: Reference alignment in paritychecker and bench.

---

### Pitfall 4: Smuggling orchestration branching into actions or context flags

**What goes wrong:**
Flash-attention enable/disable logic is added as runtime branching in actions, or per-dispatch
phase/mode flags are stored in SML context to steer control flow.

**Why it happens:**
When retrofitting a new algorithm into an existing actor, it is tempting to add `if` branches or a
`flash_enabled` scratch flag instead of changing the data-plane implementation directly.

**How to avoid:**
Keep the milestone narrow enough that the canonical path has one runtime implementation. Do not add
per-dispatch flash mode flags to context. Keep runtime control in the existing transition graph and
data-plane work inside bounded detail/kernels.

**Warning signs:**
- New action code branches between old attention and flash attention.
- Context gains fields that look like `mode`, `phase`, `flag`, `step`, or request mirrors.
- A new SML state is added only to choose between two kernel implementations.

**Phase to address:**
Phase 1: Runtime design review before implementation.

---

### Pitfall 5: Allocation regressions in the hot path

**What goes wrong:**
Flash attention works functionally, but allocates during prefill or decode because scratch buffers,
partials, or views are created on each dispatch.

**Why it happens:**
Fused attention typically needs tile scratch and temporary reductions, and it is easy to bolt those
onto request-local execution with `std::vector` growth.

**How to avoid:**
Allocate reusable workspace once in backend-owned persistent state during initialize/prepare. Reuse
it across dispatches. Keep request-local payloads in events and persistent buffers in
`generator::action::context::compute.backend`.

**Warning signs:**
- `std::vector::resize` or `assign` appears on every prefill/decode call in new code.
- Dispatch-time allocation tests or instrumentation start failing.
- Benchmark variance increases unexpectedly after enabling flash attention.

**Phase to address:**
Phase 1: Kernel and backend workspace design.

---

### Pitfall 6: Preserving the wrong cache semantics

**What goes wrong:**
KV cache writes, positions, grouped-query indexing, or decode `kv_tokens` handling no longer match
the pre-existing generator assumptions, causing subtle wrong-token bugs.

**Why it happens:**
Flash attention changes how attention is computed, but it should not silently change cache meaning.
Implementers often optimize the access pattern and accidentally change layout assumptions.

**How to avoid:**
Keep the existing cache ownership and sequence semantics in the generator backend. Replace only the
attention reduction, not the meaning of `key_cache`, `value_cache`, positions, or `kv_tokens`
without an explicit milestone decision.

**Warning signs:**
- Prefill passes but first decode token fails.
- Decode only works when `kv_tokens == 0`.
- Head/group indexing fixes appear repeatedly during debugging.

**Phase to address:**
Phase 2: Generator backend integration tests.

---

### Pitfall 7: Optimizing x86_64 before the scalar path is trustworthy

**What goes wrong:**
AVX2 code lands early, bugs are hard to localize, and parity mismatches become ambiguous between
algorithm errors and SIMD-specific errors.

**Why it happens:**
Performance pressure makes optimization feel urgent, especially in a flash-attention milestone.

**How to avoid:**
Bring up the shared/scalar implementation first, prove it through kernel tests and paritychecker,
then add x86_64 specialization behind the same operator contract.

**Warning signs:**
- The first working implementation exists only in `kernel/x86_64/actions.hpp`.
- Failures reproduce only on some hosts or only with AVX2 available.
- No scalar/shared reference path exists for A/B comparison.

**Phase to address:**
Phase 2: Correctness first, Phase 3: host-specific optimization.

---

### Pitfall 8: Turning the benchmark into a different workload than the parity slice

**What goes wrong:**
The benchmark compares a flash-attention path, but on a different prompt length, token budget,
initialization mode, or reference setting than the canonical parity slice.

**Why it happens:**
Bench work often drifts toward "interesting numbers" rather than "same accepted workload."

**How to avoid:**
Keep the existing canonical bench case names and workload shape. Only change the internal algorithm
class once parity is already stable. Bench should measure the shipped canonical slice, not a
special-case demo path.

**Warning signs:**
- New benchmark-only prompt or token-budget constants appear for flash attention.
- The canonical row disappears or is renamed.
- Bench must be interpreted with a long explanation about how it differs from parity mode.

**Phase to address:**
Phase 3: Benchmark alignment and reporting.

---

### Pitfall 9: Silent verification blind spots

**What goes wrong:**
The milestone "passes," but no one has actually verified that flash attention is active in both the
short and long canonical cases, or that the tool surfaces are exercising the intended path.

**Why it happens:**
Existing parity and bench flows already pass today, so it is easy to assume unchanged green checks
mean the new path was covered.

**How to avoid:**
Add targeted tests and, where useful, seam/dump visibility that confirms the flash-attention path
was executed. Coverage should include at least canonical parity and both canonical benchmark cases.

**Warning signs:**
- No new kernel or generator tests accompany the change.
- Debug output cannot tell whether the fused path ran.
- The long benchmark case is ignored during validation.

**Phase to address:**
Phase 2: Verification instrumentation and tests.

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Keep old `compute_attention()` path behind a hidden fallback | Easier bring-up when flash attention is unstable | Two runtime truths, misleading parity/bench interpretation, harder maintenance | Only if explicitly approved as `interim`; otherwise never |
| Tool-only flash-attention prototype in `tools/` | Fastest way to show numbers | Does not satisfy milestone architecture and drifts from shipped runtime | Never as the landed milestone solution |
| Dequantize or repack into a simpler operand class in the hot path | Simplifies early kernel code | Breaks parity claims about the effective operand path and can hide real costs | Only with explicit user approval and clear interim labeling |
| Bench first, parity later | Gives early performance feedback | Encourages optimizing a path that may still be wrong | Acceptable only for local exploration, not for milestone completion |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| `src/emel/generator` ↔ `src/emel/kernel` | Calling a new helper path directly instead of going through the kernel actor event surface | Dispatch `op_flash_attn_ext` through the existing kernel actor boundary |
| `tools/paritychecker` ↔ reference `llama.cpp` | Leaving reference flash attention disabled after EMEL switches to flash attention | Align reference context settings once the runtime path is real |
| `tools/bench` ↔ canonical generation case | Adding flash-attention-only benchmark cases instead of reusing the canonical rows | Keep the canonical rows and compare workflow, then interpret results on that basis |
| `generator` context ↔ per-dispatch runtime data | Storing request-local flash control or temporary counts in context | Keep dispatch-local data in events / `generate_ctx::io`; keep only persistent workspace in backend state |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Materializing full score/probability buffers after adding a fused operator name | Memory traffic stays high, benchmark improvement is weak or noisy | Use online softmax and bounded tile scratch in the real operator path | Breaks immediately on prefill-heavy cases and becomes more obvious as context length grows |
| Per-dispatch scratch allocation | Extra variance, slower decode, allocator noise | Preallocate persistent workspace during initialize/prepare | Breaks immediately in the canonical benchmark and worsens with longer runs |
| SIMD-only implementation with no scalar reference | Host-specific correctness drift | Keep a shared/scalar implementation for correctness and non-AVX2 hosts | Breaks as soon as validation runs on a different host or compiler setup |
| Benchmarking setup cost instead of steady-state generation path changes | Bench deltas do not reflect the new kernel | Keep benchmark contract consistent with existing preloaded request flow | Breaks immediately when trying to interpret benchmark results |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Letting mutable internal-event pointers outlive the RTC chain | Use-after-scope or undefined behavior in debug or future refactors | Keep same-RTC pointer handoff internal only and never retain it beyond dispatch |
| Exposing flash-attention internal buffers or mutable payloads through a public API | Boundary violations and unstable ABI expectations | Keep all mutable flash-attention wiring inside `src/emel` internals and existing tool-only integration surfaces |
| Copying unchecked tensor/view metadata from tool/reference code into runtime | Invalid memory access or silent wrong-shape execution | Validate shapes and dispatch requests in kernel/detail and existing guards before execution |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Benchmark output does not make it clear the canonical row changed algorithm class | Operators misread trend changes across milestones | Add minimal labeling or release-note context while keeping the same compare workflow |
| Parity failures provide no clue whether the flash path ran | Debugging slows down and confidence drops | Reuse existing dump/seam reporting to surface path identity when needed |
| A new CLI or knob is required to access flash attention | Users/operators now have two ways to run "canonical generation" | Keep flash attention behind the existing accepted surfaces for this milestone |

## "Looks Done But Isn't" Checklist

- [ ] **Kernel operator:** Often missing exact causal masking or online softmax behavior — verify the scalar/shared operator matches canonical parity before SIMD work
- [ ] **Generator integration:** Often missing real runtime adoption — verify `generator/detail.hpp` no longer relies on the old materialized attention path as the shipped implementation
- [ ] **Paritychecker:** Often missing reference alignment — verify the reference generation context is configured for the same algorithm class
- [ ] **Bench:** Often missing workload consistency — verify the canonical benchmark row and preloaded request shape are unchanged
- [ ] **Verification:** Often missing long-case coverage — verify both the 1-token and 8-token canonical generation cases exercise the new path

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Old path still shipped under a flash-attention label | MEDIUM | Revert tool-surface claims, wire the real runtime path first, then re-run parity and bench |
| Exact parity regressions | HIGH | Freeze optimization work, diff kernel outputs or seam dumps, restore scalar correctness, then reintroduce optimized paths incrementally |
| Operand-class mismatch between EMEL and reference | MEDIUM | Correct tool reference settings or narrow the claim language to end-to-end generation compare until operand alignment is real |
| SML rule violations from branching/context abuse | MEDIUM | Refactor branching back into existing orchestration decisions and move data-plane work into bounded detail/kernels |
| Per-dispatch allocation or workspace churn | LOW | Move scratch into persistent backend-owned buffers and add allocation checks in tests |
| Benchmark workload drift | LOW | Re-pin benchmark to canonical case names and settings, regenerate truthful measurements, and avoid snapshot churn until stable |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Shipping the old path under a flash-attention label | Phase 1: kernel bring-up and runtime truth wiring | Verify `src/emel/generator/detail.hpp` calls the fused operator in the shipped generation flow |
| Breaking exact generation parity | Phase 2: generator integration and parity stabilization | `tools/paritychecker --generation` passes on the canonical fixture |
| Operand-class mismatch in reference comparison | Phase 3: paritychecker and bench reference alignment | Reference contexts are aligned and any remaining claim scope is stated precisely |
| SML branching/context abuse | Phase 1: design review and integration planning | Review new actions/context fields against `AGENTS.md` and `docs/rules/sml.rules.md` |
| Allocation regressions | Phase 1: workspace design | Allocation-sensitive tests and code inspection show no dispatch-time growth |
| KV/cache semantic drift | Phase 2: generator backend integration | Prefill and decode both pass canonical parity cases |
| Premature x86_64 optimization | Phase 3: host-specific optimization | Scalar path is already green before AVX2 specialization is enabled |
| Benchmark workload drift | Phase 3: benchmark integration | Canonical bench row names and workload settings remain intact |
| Silent verification blind spots | Phase 2: tests and observability | New kernel/generator coverage exists and long-case validation is exercised |

## Sources

- `.planning/PROJECT.md` - milestone goal, current acceptance boundary, and explicit out-of-scope constraints. Confidence: HIGH.
- `AGENTS.md` - repo-specific rules for runtime truth, parity claims, no-queue actor semantics, and interim fallback restrictions. Confidence: HIGH.
- `docs/rules/sml.rules.md` - bounded-work, no-allocation, and no-runtime-branching constraints that create flash-attention integration risk. Confidence: HIGH.
- `.planning/research/STACK.md` - stack-level decisions and non-goals for this milestone. Confidence: HIGH.
- `.planning/research/FEATURES.md` - table-stakes verification and milestone scope expectations. Confidence: HIGH.
- `.planning/research/ARCHITECTURE.md` - integration points and build-order assumptions for the same milestone. Confidence: HIGH.
- `tools/paritychecker/parity_runner.cpp` and `tools/bench/generation_bench.cpp` - current reference flash-attention disable points and canonical surface assumptions. Confidence: HIGH.
- `src/emel/generator/detail.hpp` and `src/emel/kernel/**` - current runtime integration seam where most flash-attention mistakes will occur. Confidence: HIGH.

---
*Pitfalls research for: EMEL v1.2 flash attention*
*Researched: 2026-03-12*
