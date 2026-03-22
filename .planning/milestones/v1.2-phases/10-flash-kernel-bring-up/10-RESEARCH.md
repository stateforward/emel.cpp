# Phase 10: Flash Kernel Bring-Up - Research

**Researched:** 2026-03-21
**Domain:** EMEL kernel flash-attention bring-up for the canonical CPU-hosted Llama-68M slice
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Phase Boundary

Phase 10 delivers the canonical EMEL-owned `op_flash_attn_ext` path in `src/emel/kernel` for the
CPU-hosted Llama-68M slice, plus persistent workspace semantics that avoid hot-path allocation.
This phase does not widen model scope, does not adopt the operator in the shipped generator flow
yet, and does not claim user-visible parity or benchmark completion.

## Implementation Decisions

### Supported Kernel Scope
- Phase 10 stays canonical-only.
- The first supported contract is the exact causal Llama-68M attention shape needed by the shipped
  slice, not a broader family of compatible shapes.
- Generic or near-generic flash-attention coverage is explicitly deferred.

### Unsupported Request Behavior
- Unsupported or non-canonical flash-attention requests must reject explicitly.
- Phase 10 must not silently fall back to the old materialized attention path under a flash label.
- Fallback policy beyond explicit rejection is a later-phase decision.

### Proof Of Completion
- Kernel-local proof is sufficient for Phase 10.
- Required proof is shared-kernel correctness plus evidence that persistent workspace/buffers are
  reused without hot-path allocation churn.
- User-visible dump or parity evidence is deferred until generator adoption and parity phases.

## Specific Ideas

- Keep Boost.SML orchestration unchanged; this phase is a data-plane replacement only.
- Treat "flash attention" claims narrowly until the shipped generator path actually adopts the new
  kernel in Phase 11.
- Avoid proving completion through `tools/` output this early; Phase 10 should finish with kernel
  truth, not premature operator-surface claims.

### Claude's Discretion
- Exact scalar/shared implementation structure inside `src/emel/kernel`.
- Exact test shape fixtures, as long as they stay within the canonical Llama-68M contract.
- Exact persistent-workspace ownership shape, as long as it remains reusable and does not broaden
  API/runtime boundaries.

### Deferred Ideas (OUT OF SCOPE)
- Generator routing through flash attention belongs to Phase 11.
- User-visible proof that the flash path executed belongs to Phase 12.
- Reference alignment and benchmark evidence belong to Phases 12 and 13.
- Broader shape support, backend-specific optimization, and non-canonical model rollout are out of
  scope for Phase 10.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FLASH-01 | The canonical Llama-68M attention step can execute through an EMEL-owned `op_flash_attn_ext` implementation under `src/emel/kernel`. | Existing backend SM rows and kernel event scaffolding already route `op_flash_attn_ext`; Phase 10 only needs a real shared-kernel validator/executor and kernel-local tests proving canonical correctness. |
| FLASH-02 | The flash-attention operator uses persistent workspace or buffers and does not introduce allocation in the inference hot path. | Current repo patterns already prefer persistent actor/buffer reuse and fixed scratch; the canonical one-query kernel can avoid `attn_scores`/`attn_probs` materialization entirely and prove zero hot-path allocation with targeted tests. |
</phase_requirements>

## Summary

The repo already contains the SML routing needed for `op_flash_attn_ext`, but it is only
scaffolding today. `src/emel/kernel/events.hpp` declares the event, every backend state machine
already has `valid_op_flash_attn_ext` and `exec_op_flash_attn_ext` rows, and the backend action
templates already know how to mark success or rejection. The missing piece is the shared kernel
implementation: `src/emel/kernel/detail.hpp` does not currently validate or execute flash
attention, so every flash request is effectively still unsupported.

The old attention path still lives entirely in `src/emel/generator/detail.hpp`. It computes Q/K/V,
applies RoPE, copies K/V into the persistent caches, materializes `attn_scores` and `attn_probs`,
and then accumulates `attn_ctx`. Phase 10 should not touch that generator flow. Instead, it should
add a kernel-local canonical `op_flash_attn_ext` path that matches the current shipped Llama-68M
single-query causal attention slice and rejects everything else explicitly. Phase 11 can then swap
the generator over to that operator without changing Boost.SML structure.

The most practical implementation is a shared scalar kernel in `src/emel/kernel/detail.hpp` using
an online-softmax accumulation over the existing K/V cache layout, not a generic ggml-compatible
operator surface. That keeps Phase 10 inside the current repo architecture, avoids action/guard
changes, avoids hot-path allocation, and produces a truthful kernel-only proof without changing the
generator, paritychecker, or benchmark tools yet.

**Primary recommendation:** Implement canonical-only `op_flash_attn_ext` in
`src/emel/kernel/detail.hpp`, keep backend SML structure unchanged, reject all non-canonical
shapes/flags, and prove correctness plus zero hot-path allocation with new kernel tests only.

## Project Constraints (from CLAUDE.md)

`CLAUDE.md` is a symlink to `AGENTS.md`, so the engineering contract is the same file the user
explicitly provided. Phase 10 must honor these directives:

- Keep Boost.SML orchestration unchanged for this phase; this is a data-plane kernel bring-up.
- Do not add queues, deferred events, mailboxes, self-dispatch, or any other non-RTC behavior.
- Do not change state-machine structure without explicit user approval.
- Keep actions bounded, non-blocking, and allocation-free during dispatch.
- Do not put runtime branching in actions or helper functions called from actions; put runtime
  control flow in guards or explicit state transitions.
- Keep per-dispatch data out of context; only persistent actor-owned state belongs there.
- Model unsupported behavior explicitly; do not silently drop or silently fall back.
- Keep performance as a first-class goal; do not allocate in inference hot paths.
- Do not replace missing hot-path native behavior with a whole-tensor dequantize-to-f32 fallback
  unless the user explicitly approves an interim milestone.
- Keep `src/` Boost.SML machines as the source of truth; do not move orchestration into docs or
  tool-only scaffolding.
- Use doctest for unit tests.
- Keep test files scoped to one machine, one system, or one behavior.
- Run `scripts/quality_gates.sh` after implementation changes.
- Use CTest targets `emel_tests` and `lint_snapshot` for test execution.
- Enforce line coverage >= 90%.

## Standard Stack

### Core
| Library / Component | Version | Purpose | Why Standard |
|---------------------|---------|---------|--------------|
| `src/emel/kernel/detail.hpp` shared scalar kernel path | repo local | Canonical place for backend-agnostic kernel validation and execution | Existing x86_64/aarch64 backends already call into shared scalar helpers; Phase 10 can land here without changing SML structure |
| Boost.SML (`stateforward/sml.cpp`) | pinned commit `02cbea023f035185cfb400e6015c981f9b946bae` | Orchestration for kernel/backend wrappers | Already wired through every kernel backend and constrained by repo rules |
| `emel::kernel::any` / backend `sm` wrappers | repo local | Dispatches public kernel events to platform backends | Existing public kernel surface already routes `op_flash_attn_ext` |

### Supporting
| Library / Component | Version | Purpose | When to Use |
|---------------------|---------|---------|-------------|
| doctest | 2.4.11 | Unit tests | Kernel correctness, rejection behavior, and allocation checks |
| `tmp/llama.cpp` vendored reference | vendored local snapshot | Arithmetic and operand-contract reference only | Use to validate operator intent and later parity expectations, not as runtime linkage in `src/` |
| `tools/paritychecker` tensor compare dump | repo local | Later-phase evidence surface | Useful for future cross-checks, but not Phase 10 proof of completion |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Shared scalar bring-up in `src/emel/kernel/detail.hpp` | Backend-specific x86_64/aarch64 implementations first | Slower initial performance, but much lower planning risk and no SML churn; backend-specific tuning is explicitly later work |
| Canonical EMEL subset contract | Full generic ggml `ggml_flash_attn_ext` surface now | The current event type has only `src0/src1/src2/dst` and no explicit mask/sinks sources; generic parity would widen scope beyond Phase 10 |
| Kernel-local tests only | Generator/parity/bench integration now | Tool-facing proof is explicitly deferred to Phases 11-13 |
| Online-softmax accumulation | Materialized `attn_scores` + `attn_probs` buffers | Online softmax avoids unnecessary hot-path scratch and better matches the phase’s no-allocation goal |

**Build commands:**
```bash
./scripts/build_with_zig.sh
./scripts/test_with_coverage.sh
./scripts/paritychecker.sh
```

## Architecture Patterns

### Recommended Project Structure
```text
src/
├── emel/kernel/detail.hpp          # Shared scalar validator/executor for flash bring-up
├── emel/kernel/events.hpp          # Existing op event surface, including op_flash_attn_ext
├── emel/kernel/x86_64/sm.hpp       # Existing dispatch rows already route flash ext
├── emel/kernel/aarch64/sm.hpp      # Existing dispatch rows already route flash ext
├── emel/generator/detail.hpp       # Current materialized attention path; Phase 10 reference only
└── emel/model/llama/detail.hpp     # Canonical topology and step-plan metadata

tests/
├── kernel/test_helpers.hpp         # Existing tensor-view helpers
├── kernel/lifecycle_tests.cpp      # Existing broad kernel coverage
└── kernel/flash_attn_ext_tests.cpp # Recommended new focused Phase 10 coverage
```

### Pattern 1: Shared-Kernel First, Backend SML Unchanged
**What:** Add `can_run_flash_attn_ext(...)` and `run_flash_attn_ext(...)` to
`src/emel/kernel/detail.hpp`, then let the already-existing backend guards/actions use them through
their current template flow.

**When to use:** Phase 10 shared correctness bring-up for the canonical CPU path.

**Example:**
```cpp
// Source: src/emel/kernel/x86_64/actions.hpp
template <class dispatch_event_type>
struct exec_scalar_op {
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    ::emel::kernel::detail::execute_scalar_unchecked(ev.request);
    detail::mark_done(ev, ctx);
  }
};
```

**Planner implication:** Do not add new kernel states or backend-specific flash actions for Phase
10. Extend the shared helpers that the existing actions already call.

### Pattern 2: Canonical EMEL Contract Over Current Cache Layout
**What:** Validate only the current shipped Llama-68M single-query causal contract:

- `q`: one token, laid out as heads x head-dim
- `k`: current persistent key cache slice over positions `0..position`
- `v`: current persistent value cache slice over positions `0..position`
- `dst`: one-token attention context laid out as heads x head-dim
- `op_params`: only the canonical metadata actually needed for this subset

The important detail is that EMEL’s current cache layout is position-major
`[position][kv_head][dim]`, not the generic ggml `k/v` layout shown in `ggml_flash_attn_ext`.
Phase 10 should validate and execute the current EMEL layout instead of trying to generalize now.

**When to use:** Any Phase 10 flash request. Everything outside that exact layout rejects.

**Example:**
```cpp
// Source: src/emel/generator/detail.hpp
apply_rope(backend.q, backend.n_head, backend.head_dim, backend.n_rot, position, ...);
apply_rope(backend.k, backend.n_head_kv, backend.head_dim_kv, backend.n_rot, position, ...);
std::copy(backend.k.begin(), backend.k.begin() + kv_dim, backend.key_cache.begin() + cache_offset);
std::copy(backend.v.begin(), backend.v.begin() + kv_dim, backend.value_cache.begin() + cache_offset);
compute_attention(backend, layer_index, position + 1, backend.q);
```

**Planner implication:** The canonical flash operator should consume the same logical operands the
generator already produces, not force a new cache format in Phase 10.

### Pattern 3: Online Softmax for Single-Query Causal Attention
**What:** Replace materialized score/probability buffers with a streaming per-head computation:

1. Iterate K positions once to update running max and running sum.
2. Rescale the partial output accumulator when the running max changes.
3. Accumulate weighted V rows directly into `dst`.

This keeps the kernel bounded and avoids `attn_scores` / `attn_probs` scratch for the canonical
one-query case.

**When to use:** The exact Phase 10 attention slice, where `run_layer()` computes one query token
at a time in both prefill and decode.

**Example:**
```cpp
// Source intent: replace src/emel/generator/detail.hpp::compute_attention(...)
// with a kernel-local streaming equivalent for one query token:
//
// for each head:
//   M = -inf
//   S = 0
//   out[:] = 0
//   for each cached position:
//     score = dot(q_head, k_head_pos) * scale
//     update {M, S} with online-softmax rescaling
//     out += normalized_weight * v_head_pos
```

### Anti-Patterns to Avoid
- **Generator-side hidden fallback:** Do not keep `generator::detail::compute_attention(...)` as a
  silent fallback under a flash label. Unsupported requests must reject explicitly.
- **State-machine edits for data-plane math:** Do not add new SML states, queues, or internal
  self-dispatch to model numeric work.
- **Per-call heap growth:** Do not allocate `std::vector` scratch inside flash execution. The
  shared kernel must stay allocation-free during dispatch.
- **Generic-contract theater:** Do not claim generic ggml flash support when the current event
  surface has no explicit mask/sinks sources and the phase is canonical-only by decision.
- **Tool-surface claims too early:** Do not use paritychecker or benchmark output as Phase 10 proof.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Phase 10 routing | New flash-specific SML graphs | Existing backend `dispatch_op_flash_attn_ext` rows | The routing scaffold already exists in every backend SM |
| Score/probability storage | Full materialized `attn_scores` / `attn_probs` arrays inside the kernel | Online softmax over current K/V slices | Avoids hot-path allocation and matches the canonical one-query workload |
| Generic feature coverage | Full ggml mask/sinks/bias feature matrix now | Canonical Llama-68M subset validator | The current event surface cannot express the full ggml contract cleanly without scope expansion |
| Proof of completion | Early generator/parity/bench rewiring | Kernel-local doctest coverage | Phase 10 proof is explicitly kernel-local |
| Cache repacking | New per-call transpose/copy buffers for K/V | Existing tensor-view strides over persistent caches | Copying or repacking would add work and risk allocations in the hot path |

**Key insight:** The repo already has enough public kernel plumbing. Phase 10 should not invent a
new operator surface; it should give the existing `op_flash_attn_ext` rows a real canonical
executor and a real canonical validator.

## Common Pitfalls

### Pitfall 1: Mistaking Scaffolding for Support
**What goes wrong:** The planner assumes flash attention is partly implemented because the event and
transition rows already exist.
**Why it happens:** `src/emel/kernel/events.hpp` and each backend `sm.hpp` already mention
`op_flash_attn_ext`, but `src/emel/kernel/detail.hpp` does not currently validate or execute it.
**How to avoid:** Treat the current flash rows as empty routing scaffolding only.
**Warning signs:** `process_event(op_flash_attn_ext)` returns false immediately and lands on the
backend’s invalid-op path.

### Pitfall 2: Bringing Phase 11 Work into Phase 10
**What goes wrong:** The implementation rewires `src/emel/generator/detail.hpp` or tool surfaces to
prove flash ran.
**Why it happens:** The old attention path is easy to see and tempting to replace in place.
**How to avoid:** Keep generator adoption, parity proof, and benchmark claims deferred exactly as
the roadmap says.
**Warning signs:** Edits appear in `src/emel/generator/*`, `tools/paritychecker/*`, or
`tools/bench/generation_bench.cpp`.

### Pitfall 3: Validating Against the Wrong Tensor Layout
**What goes wrong:** The kernel is written against the generic ggml `k/v` tensor ordering and does
not match EMEL’s current cache storage.
**Why it happens:** `ggml_flash_attn_ext` documents `k` and `v` as `[dim, kv, head_kv, ...]`,
while EMEL’s current caches are position-major `[position][kv_head][dim]`.
**How to avoid:** Make the Phase 10 validator explicit about the accepted `ne`/`nb` layout.
**Warning signs:** Correct arithmetic on small copied fixtures, but wrong answers when pointing
directly at the real generator caches.

### Pitfall 4: Reintroducing Materialized Buffers
**What goes wrong:** The flash kernel still allocates or still needs score/probability arrays the
size of `n_ctx`.
**Why it happens:** The current generator path uses `attn_scores` and `attn_probs`, so copying that
shape feels natural.
**How to avoid:** Use single-query online softmax and accumulate directly into `dst`.
**Warning signs:** New `std::vector` or `resize()` calls appear in kernel execution code.

### Pitfall 5: Silent Non-Canonical Success
**What goes wrong:** Unsupported shapes or flags quietly compute through the old materialized path
or through a partial flash implementation.
**Why it happens:** It is tempting to widen support opportunistically while the generator still owns
the truth path.
**How to avoid:** Keep `can_run_flash_attn_ext(...)` narrow and make rejection tests first-class.
**Warning signs:** Flash requests with non-canonical shapes return true without a dedicated test.

## Code Examples

Verified patterns from local primary sources:

### Existing Shared-Kernel Dispatch Flow
```cpp
// Source: src/emel/kernel/x86_64/actions.hpp
template <class dispatch_event_type>
struct exec_scalar_op {
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    ::emel::kernel::detail::execute_scalar_unchecked(ev.request);
    detail::mark_done(ev, ctx);
  }
};
```

### Current Generator Materialized Attention Path
```cpp
// Source: src/emel/generator/detail.hpp
std::fill(backend.attn_ctx.begin(), backend.attn_ctx.end(), 0.0f);
for (int32_t head = 0; head < head_count; ++head) {
  // materialize scores into backend.attn_scores
  // materialize probabilities into backend.attn_probs
  // accumulate weighted values into backend.attn_ctx
}
```

### Vendored ggml Flash-Attention Contract
```cpp
// Source: tmp/llama.cpp/ggml/include/ggml.h
// q:    [n_embd_k, n_batch, n_head,    ne3 ]
// k:    [n_embd_k, n_kv,    n_head_kv, ne3 ]
// v:    [n_embd_v, n_kv,    n_head_kv, ne3 ]
// res:  [n_embd_v, n_head,  n_batch,   ne3 ]
ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap);
```

### Upstream Reference Selection Site
```cpp
// Source: tmp/llama.cpp/src/llama-graph.cpp
const bool use_flash_attn = cparams.flash_attn && kq_b == nullptr;
if (use_flash_attn) {
  cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, ...);
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Repo-local materialized attention in `src/emel/generator/detail.hpp` using `attn_scores` and `attn_probs` | Vendored upstream uses `ggml_flash_attn_ext` for supported flash-attn builds | Current vendored `tmp/llama.cpp` snapshot | EMEL still needs its own native kernel bring-up before it can claim flash execution |
| Generator-owned attention truth path | Kernel-owned flash operator path | Phase 11 in this roadmap, not Phase 10 | Phase 10 should stop at kernel truth only |
| Generic tool output as proof | Kernel-local correctness and allocation proof | This roadmap’s phase ordering | Keeps Phase 10 scoped and truthful |

**Deprecated/outdated:**
- Claiming “flash attention” through `tools/paritychecker` or `tools/bench/generation_bench.cpp`
  right now is incorrect; both still force `flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED`.
- Treating the existing backend flash rows as partial support is incorrect; they are routing only.

## Open Questions

1. **What exact `op_params` payload should the canonical EMEL flash event use?**
   - What we know: ggml stores `scale`, `max_bias`, and `logit_softcap` in op params and later
     uses slot 3 for precision. EMEL’s `op_flash_attn_ext` event only exposes a generic
     `op_params[64]` blob.
   - What's unclear: whether Phase 10 should pack only `scale` plus minimal canonical metadata
     (for example `position_limit` and flags), or mirror ggml’s first four slots even though
     `mask` and `sinks` are out of scope.
   - Recommendation: Keep the canonical EMEL payload minimal and explicit. Derive as much as
     possible from `tensor_view.ne/nb`, then reserve packed params only for values that cannot be
     inferred cleanly.

2. **Does Phase 10 need backend-specific SIMD immediately?**
   - What we know: Existing x86_64 and aarch64 backends already route flash events through the
     shared scalar path when there is no backend-specific override.
   - What's unclear: none for Phase 10 scope.
   - Recommendation: No. Land shared scalar correctness first; leave backend-specific tuning to a
     later phase backed by benchmark evidence.

3. **Where should “persistent workspace” live for the Phase 10 proof?**
   - What we know: The canonical one-query algorithm can avoid large scratch entirely, and the repo
     already reuses persistent caller buffers (`key_cache`, `value_cache`, `attn_ctx`) plus fixed
     thread-local scratch in some kernels.
   - What's unclear: whether the planner wants a dedicated flash workspace object now or a simpler
     no-allocation proof using existing caller-owned buffers plus fixed local scratch.
   - Recommendation: Prefer the simpler no-new-API route for Phase 10. Prove zero hot-path
     allocation and direct buffer reuse in tests; only add dedicated workspace ownership if the
     implementation truly needs it.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | doctest 2.4.11 |
| Config file | none - CMake + CTest only |
| Quick run command | `build/coverage/emel_tests_bin --test-case=*flash_attn_ext*` |
| Full suite command | `ctest --test-dir build/coverage --output-on-failure -R emel_tests -j 1` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FLASH-01 | Canonical one-query Llama attention executes through `kernel.process_event(op_flash_attn_ext)` and matches the current materialized `compute_attention(...)` result on fixture-derived shapes | unit | `build/coverage/emel_tests_bin --test-case=*flash_attn_ext*canonical*` | ❌ Wave 0 |
| FLASH-01 | Non-canonical flash requests reject explicitly and do not route through an old materialized fallback | unit | `build/coverage/emel_tests_bin --test-case=*flash_attn_ext*reject*` | ❌ Wave 0 |
| FLASH-02 | Repeated canonical flash calls do not allocate during dispatch and reuse the same caller-owned buffers / fixed scratch | unit | `build/coverage/emel_tests_bin --test-case=*flash_attn_ext*alloc*` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `build/coverage/emel_tests_bin --test-case=*flash_attn_ext*`
- **Per wave merge:** `ctest --test-dir build/coverage --output-on-failure -R emel_tests -j 1`
- **Phase gate:** `./scripts/quality_gates.sh`

### Wave 0 Gaps
- [ ] `tests/kernel/flash_attn_ext_tests.cpp` - canonical correctness, explicit rejection, and
      no-allocation coverage for `FLASH-01` / `FLASH-02`
- [ ] `CMakeLists.txt` - add the new kernel test file to `EMEL_TEST_SOURCES`
- [ ] `tests/kernel/test_helpers.hpp` - optional helper expansion for canonical flash tensor views
      and packed `op_params`
- [ ] Allocation guard helper for kernel dispatch - either extend the new flash test file or add a
      tiny local helper so the phase can prove zero hot-path allocation

## Sources

### Primary (HIGH confidence)
- `.planning/phases/10-flash-kernel-bring-up/10-CONTEXT.md` - locked scope, deferred work, and
  proof boundary for Phase 10
- `.planning/REQUIREMENTS.md` - `FLASH-01` and `FLASH-02`
- `AGENTS.md` / `CLAUDE.md` symlink - repo engineering contract and SML/performance rules
- `docs/rules/sml.rules.md` - RTC/no-queue/bounded-action semantics
- `src/emel/kernel/events.hpp` - current `op_flash_attn_ext` event surface
- `src/emel/kernel/detail.hpp` - current shared scalar validation/execution surface
- `src/emel/kernel/x86_64/actions.hpp` - existing backend action template flow
- `src/emel/kernel/x86_64/sm.hpp` - existing flash transition rows
- `src/emel/generator/detail.hpp` - current materialized attention path and persistent generator
  buffers
- `src/emel/generator/context.hpp` / `src/emel/generator/actions.hpp` - current persistent graph
  reservation and graph execution wiring
- `src/emel/graph/events.hpp`, `src/emel/graph/actions.hpp`,
  `src/emel/graph/assembler/context.hpp`, `src/emel/graph/assembler/reuse_decision_pass/guards.hpp`
  - existing repo patterns for persistent reuse semantics
- `src/emel/model/llama/detail.hpp`, `src/emel/model/data.cpp` - canonical topology/step-plan and
  workspace-capacity metadata
- `tests/kernel/*.cpp`, `tests/kernel/test_helpers.hpp` - existing kernel test patterns
- `tools/paritychecker/parity_runner.cpp`, `tools/paritychecker/paritychecker_tests.cpp` - current
  parity surfaces and explicit flash-disabled state
- `tools/bench/generation_bench.cpp`, `tools/bench/kernel/*.cpp` - current benchmark surfaces and
  deferral boundary
- `CMakeLists.txt`, `scripts/build_with_zig.sh`, `scripts/test_with_coverage.sh`,
  `scripts/paritychecker.sh`, `scripts/quality_gates.sh` - build/test/gate contract
- `tmp/llama.cpp/ggml/include/ggml.h`, `tmp/llama.cpp/ggml/src/ggml.c`,
  `tmp/llama.cpp/ggml/src/ggml-cpu/ops.cpp`, `tmp/llama.cpp/src/llama-graph.cpp`,
  `tmp/llama.cpp/src/models/llama.cpp` - local reference contract and current upstream flash path

### Secondary (MEDIUM confidence)
- None

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - entirely derived from checked-in repo structure and build/test scripts
- Architecture: HIGH - existing backend/event scaffolding and current generator attention path are
  explicit in local code
- Pitfalls: HIGH - directly evidenced by current code layout, current flash-disabled tools, and the
  repo’s engineering contract

**Research date:** 2026-03-21
**Valid until:** 2026-03-28
