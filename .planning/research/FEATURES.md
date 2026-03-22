# Feature Research

**Domain:** Flash attention for the existing EMEL canonical Llama-68M generation slice
**Researched:** 2026-03-12
**Confidence:** HIGH

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist for this milestone. Missing these means flash attention is not actually
shipped in the repo's accepted boundary.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Real EMEL-owned flash-attention path in canonical generation | The milestone goal in `.planning/PROJECT.md` is not "study flash attention"; it is a shipped EMEL generation path using it | HIGH | Must execute inside `src/emel/generator` and `src/emel/kernel`, not in tool-only code and not through a dequantize-to-f32 surrogate hot path |
| Canonical generation parity still passes through `tools/paritychecker --generation` | This repo defines correctness by paritychecker on the Llama-68M fixture, not by internal unit math alone | HIGH | Must keep the same fixture, prompt-driven CLI, bounded generation contract, and success/failure reporting shape |
| Canonical compare benchmark still works through `scripts/bench.sh --compare` | The existing benchmark surface is already the accepted place to publish EMEL vs `llama.cpp` timings | MEDIUM | Must preserve `generation/preloaded_request/llama_68m_prompt_hello_max_tokens_1` and the existing compare row expectations |
| Flash attention remains CPU-hosted and deterministic | The milestone is scoped to the current CPU-hosted path and the repo values deterministic replayable behavior | MEDIUM | Must stay in the existing host backend flow and preserve identical output for the canonical workload |
| No API-surface expansion to use flash attention | The repo already proved the generation slice without broadening the public API, and this milestone says to reuse those surfaces | LOW | The behavior should be reachable via the existing generator initialize/generate flow and current tool CLIs, not a new public switch surface |
| Bounded, allocation-aware execution that respects SML rules | This repo treats bounded RTC execution and no-queue semantics as architecture, not style | HIGH | Attention computation can be large data-plane work, but orchestration must remain a bounded phase action with no mailbox/deferred dispatch |

### Differentiators (Competitive Advantage)

These are valuable in this repo, but not required to call the first flash-attention milestone done.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Debug/dump visibility that proves the flash path is active | Makes parity triage faster when outputs diverge and reduces risk of "benchmarked the old path by mistake" | MEDIUM | Best done by extending existing `--dump` / seam reporting rather than inventing a new diagnostic executable |
| Both short and long canonical generation cases show the flash path in bench | Gives a clearer picture of whether the new path helps decode-only and slightly longer runs | MEDIUM | The current bench already has 1-token and 8-token generation cases; reuse those rather than adding a matrix |
| Explicit benchmark labeling or documentation that the canonical row now uses flash attention | Prevents operator confusion when reviewing compare output across milestones | LOW | Prefer additive naming/metadata around the existing compare flow, not a second benchmark workflow |

### Anti-Features (Commonly Requested, Often Problematic)

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Multi-model flash-attention rollout in the first milestone | It seems more impressive to claim broad model support immediately | It broadens validation, parity, and debugging far beyond the audited Llama-68M slice | Ship the canonical Llama-68M path first, then expand under a later milestone |
| New flash-attention-specific CLI flags or public API toggles | It sounds useful to force old/new paths for experimentation | It widens the accepted surface and shifts the milestone from implementation to product/API design | Keep the current CLI/API surfaces and make flash attention the internal implementation of the canonical path |
| GPU backends, backend matrices, or cross-platform rollout as part of the first flash milestone | "Flash attention" is often associated with GPU acceleration | The active milestone is CPU-hosted only, and a backend matrix would swamp parity and bench work | Keep CPU first; treat other backends as later follow-on work |
| Full `ggml_flash_attn_ext` feature coverage on day one | It is tempting to port every sink, ALiBi, non-causal, or softcap feature with the operator name | The canonical Llama-68M generation slice does not need all of those, and generic parity would become the milestone instead of generation correctness | Implement only the causal self-attention subset needed by the canonical generation path |
| Tool-only flash-attention proof without `src/emel` runtime integration | It is faster to show a benchmark or parity experiment in tool code first | The repo rules explicitly treat `src/` machines as the source of truth, and tool-only compute paths do not satisfy the milestone | Land the runtime path in `src/emel`, then verify it through the existing tools |
| Approximate attention or relaxed-output "close enough" mode | It can produce performance wins quickly | This repo's current acceptance boundary is exact parity-oriented generation on a fixed fixture | Keep exact attention semantics for the first milestone |

## Feature Dependencies

```text
[Real EMEL flash-attention generation path]
    ├──requires──> [Kernel-level fused flash-attention execution]
    ├──requires──> [Generator integration over existing KV cache and decode/prefill flow]
    └──requires──> [Canonical parity acceptance on --generation]

[Canonical bench visibility]
    ├──requires──> [Real EMEL flash-attention generation path]
    └──requires──> [Reference compare path aligned to flash-attention behavior]

[Differentiator: dump/seam visibility]
    └──enhances──> [Canonical parity acceptance on --generation]

[Multi-model rollout]
    ──conflicts──> [Narrow canonical milestone scope]

[New public flash-attention knobs]
    ──conflicts──> [Reuse existing generation/parity/bench surfaces]
```

### Dependency Notes

- **Real EMEL flash-attention generation path requires kernel-level fused flash-attention execution:** the current generator path can only claim the feature if it stops materializing the old attention flow and actually dispatches a fused operator in `src/emel`.
- **Real EMEL flash-attention generation path requires generator integration over existing KV cache and decode/prefill flow:** a standalone kernel benchmark is insufficient; the feature is only real once the shipped generation slice uses it for prompt prefill and token decode.
- **Real EMEL flash-attention generation path requires canonical parity acceptance on `--generation`:** in this repo, feature completion is not "kernel computes"; it is "canonical generation still matches the reference fixture path."
- **Canonical bench visibility requires the real runtime path first:** benchmarking a debug path or alternate tool seam would produce misleading numbers for the milestone.
- **Canonical bench visibility requires the reference compare path aligned to flash-attention behavior:** current reference contexts explicitly disable flash attention, so the bench feature is incomplete unless compare mode measures the same algorithm class.
- **Dump/seam visibility enhances parity acceptance:** it is not required to ship, but it sharply reduces debugging cost when parity mismatches appear.
- **Multi-model rollout conflicts with narrow canonical scope:** it multiplies fixtures, failure cases, and benchmark interpretation before the first path is trustworthy.
- **New public flash-attention knobs conflict with surface reuse:** they convert an implementation milestone into an API milestone and create user-visible commitments that the project has not planned yet.

## MVP Definition

### Launch With (v1)

Minimum viable flash-attention milestone for this repo.

- [ ] Canonical Llama-68M generation runs through an EMEL-owned flash-attention path in `src/emel`
  - Essential because this is the actual milestone goal.
- [ ] `tools/paritychecker --generation --model tests/models/Llama-68M-Chat-v1-Q2_K.gguf --text hello --max-tokens 1` still reports parity success
  - Essential because paritychecker is the accepted correctness surface.
- [ ] `tools/bench` compare mode still publishes the canonical generation row with the flash-attention implementation behind EMEL's path
  - Essential because the benchmark surface already defines truthful performance evidence in this repo.
- [ ] The milestone stays CPU-only, canonical-fixture-only, and uses existing CLIs/API surfaces
  - Essential because widening scope is explicitly out of bounds for v1.2.

### Add After Validation (v1.x)

- [ ] Better seam/dump reporting that identifies whether flash attention was used in EMEL and reference paths
  - Add once the base path is correct and any remaining parity mismatches need faster diagnosis.
- [ ] More robust benchmark annotations for short vs long canonical generation cases
  - Add once the repo needs clearer operator-facing reporting without changing the benchmark workflow.
- [ ] Additional kernel-focused tests for edge shapes that still map to the canonical generation family
  - Add after the first shipped path is stable enough to justify broader shape validation.

### Future Consideration (v2+)

- [ ] Multi-model flash-attention support beyond the Llama-68M canonical slice
  - Defer because it changes the validation matrix and likely reveals architecture-specific feature gaps.
- [ ] Broader `ggml_flash_attn_ext` feature parity such as sinks, softcap, or non-causal variants
  - Defer because those are not required by the current generation slice.
- [ ] Backend expansion to aarch64/GPU/Vulkan/Metal/WASM flash-attention implementations
  - Defer because the present milestone is explicitly CPU-hosted.
- [ ] Public API controls for selecting flash-attention behavior
  - Defer because the project has not broadened the API acceptance boundary yet.

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Real EMEL flash-attention canonical generation path | HIGH | HIGH | P1 |
| Canonical generation parity through existing `--generation` flow | HIGH | HIGH | P1 |
| Canonical compare benchmark visibility through existing bench flow | HIGH | MEDIUM | P1 |
| CPU-only deterministic behavior with no new public surface | HIGH | LOW | P1 |
| Dump/seam visibility for flash-attention debugging | MEDIUM | MEDIUM | P2 |
| Better benchmark labeling/annotation for flash-attention runs | MEDIUM | LOW | P2 |
| Multi-model rollout | MEDIUM | HIGH | P3 |
| Full generic `ggml_flash_attn_ext` feature coverage | LOW | HIGH | P3 |
| GPU/backend expansion | MEDIUM | HIGH | P3 |
| Public flash-attention API toggles | LOW | MEDIUM | P3 |

**Priority key:**
- P1: Must have for launch
- P2: Should have, add when possible
- P3: Nice to have, future consideration

## Repo Surface Analysis

| Feature | Existing Surface | Current State | Our Approach |
|---------|------------------|---------------|--------------|
| Canonical parity acceptance | `tools/paritychecker --generation` | Already shipped for non-flash generation on the canonical GGUF fixture | Keep the exact same CLI and success/failure contract; change only the internal implementation and reference alignment |
| Canonical compare benchmark | `scripts/bench.sh --compare` / `tools/bench` | Already shipped for canonical generation compare | Keep the same case names and compare workflow; make flash attention visible through the existing rows |
| Runtime generation path | `src/emel/generator` + `src/emel/kernel` | Current path exists but does not yet use a real flash-attention implementation | Add the feature in `src/emel`, not in tool-local code |
| Benchmark acceptance boundary | Canonical 1-token and 8-token generation rows | Already narrow and stable | Reuse the narrow cases rather than creating a new matrix |
| Product/API surface | Existing initialize/generate flow and tool CLIs | Intentionally narrow | Do not broaden it in this milestone |

## Sources

- `.planning/PROJECT.md` - milestone goal, active requirements, and out-of-scope constraints. Confidence: HIGH.
- `AGENTS.md` - engineering contract for SML, parity claims, hot-path behavior, and milestone scoping rules. Confidence: HIGH.
- `tools/paritychecker/parity_main.cpp` - existing generation CLI contract to preserve. Confidence: HIGH.
- `tools/paritychecker/parity_runner.cpp` - current generation parity acceptance behavior and failure/success reporting. Confidence: HIGH.
- `tools/bench/bench_main.cpp` - existing compare workflow and canonical generation case enforcement. Confidence: HIGH.
- `tools/bench/generation_bench.cpp` - current benchmark case behavior for the canonical generation slice. Confidence: HIGH.
- `.planning/research/STACK.md` - stack-level integration points and explicit non-goals for the same milestone. Confidence: HIGH.

---
*Feature research for: EMEL v1.2 flash attention*
*Researched: 2026-03-12*
