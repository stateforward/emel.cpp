# Phase 14: AArch64 Flash Kernel - Research

**Researched:** 2026-03-22
**Domain:** EMEL AArch64 backend-local flash-attention optimization for the canonical Llama-68M ARM slice
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Phase Boundary

Phase 14 delivers a backend-specific AArch64 `op_flash_attn_ext` implementation for the maintained
canonical Llama-68M flash-attention contract inside `src/emel/kernel`, preserving reusable
workspace semantics and zero-allocation dispatch. This phase does not widen workload scope, does
not change the existing Boost.SML orchestration contract, and does not yet claim shipped runtime,
paritychecker, or benchmark completion.

## Existing Code Insights

### Reusable Assets
- `src/emel/kernel/aarch64/actions.hpp` already dispatches `op_flash_attn_ext` and owns
  `flash_attn_workspace` through the backend context.
- `src/emel/kernel/detail.hpp` already contains the maintained shared scalar
  `run_flash_attn_ext_with_workspace(...)` helper and request-shape validation logic.
- `tests/kernel/aarch64_tests.cpp`, `tests/kernel/lifecycle_tests.cpp`, and
  `tests/kernel/test_helpers.hpp` already cover canonical flash-attention execution and workspace
  reuse behavior.

### Established Patterns
- Backend machines keep the existing `src/emel/kernel/*/sm.hpp` dispatch contract unchanged while
  data-plane specialization lives in backend-local helpers and actions.
- Unsupported requests are rejected explicitly instead of silently claiming optimized execution.
- Persistent workspace ownership stays in backend context and must remain allocation-free during
  dispatch.

### Integration Points
- `src/emel/kernel/aarch64/actions.hpp` is the current seam where AArch64 flash requests still
  route through the shared scalar workspace helper.
- `src/emel/kernel/aarch64/context.hpp` and `src/emel/kernel/detail.hpp` define the current
  reusable workspace contract that the optimized path must preserve.
- `tests/kernel/aarch64_tests.cpp` and `tests/kernel/lifecycle_tests.cpp` are the kernel-facing
  proof surfaces for correctness and workspace reuse in this phase.

## Specific Ideas

No specific requirements; infrastructure phase.

### Claude's Discretion
- All implementation choices are at the agent's discretion; this is a pure infrastructure phase.

### Deferred Ideas (OUT OF SCOPE)
None.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PORT-01 | The canonical Llama-68M ARM generation slice can execute `op_flash_attn_ext` through an AArch64-specific optimized implementation instead of the shared scalar workspace path. | The AArch64 backend already owns the flash dispatch row and context-local workspace; Phase 14 only needs a backend-local flash helper wired under the existing `exec_op_flash_attn_ext` action so canonical requests stop calling `kernel/detail.hpp::run_flash_attn_ext_with_workspace(...)`. |
| PORT-02 | The AArch64 flash-attention implementation preserves zero-allocation hot-path behavior and uses backend-owned reusable workspace or buffers. | `aarch64::action::context` already owns persistent flash scratch. Phase 14 should extend or reshape that scratch into fixed-capacity backend-local buffers and prove reuse plus zero-allocation dispatch through kernel-local doctest coverage. |
</phase_requirements>

## Summary

The current AArch64 flash path is structurally ready but numerically still shared and scalar.
`src/emel/kernel/aarch64/sm.hpp` already routes `dispatch_op_flash_attn_ext` through
`action::exec_op_flash_attn_ext`, but `src/emel/kernel/aarch64/actions.hpp` special-cases flash by
calling `::emel::kernel::detail::run_flash_attn_ext_with_workspace(...)`. `simd_supported_request_v`
does not include `op_flash_attn_ext`, so the backend never selects a backend-local SIMD flash path
today. That means Phase 14 is a pure data-plane replacement: keep the SML rows and public event
shape unchanged, replace only the AArch64 flash math underneath the existing action seam.

The safest narrow implementation is an AArch64-local single-query online-softmax kernel for the
already-maintained canonical contract: `query_count == 1`, F32 Q/K/V/DST, current position-major
K/V cache layout, `ith == 0`, `nth == 1`, and explicit rejection outside that contract. ggml’s CPU
flash path is useful as arithmetic and structure reference, but not as a direct port target:
ggml supports mask, sinks, softcap, broader dtype combinations, and split-KV threadpool reduction
that EMEL’s current event and dispatch contract cannot express without widening scope.

**Primary recommendation:** Keep `sm.hpp` and guards structurally unchanged, add a backend-local
AArch64 flash helper under the existing action seam, preserve a fixed reusable workspace in
`aarch64::action::context`, and prove Phase 14 only through kernel-local AArch64 tests.

## Likely File Changes

| File | Why |
|------|-----|
| `src/emel/kernel/aarch64/actions.hpp` | Current flash branch is the exact seam that still delegates to the shared scalar helper. |
| `src/emel/kernel/aarch64/context.hpp` | Workspace shape may need fixed-capacity accumulators or packed-tile scratch beyond the current `score_buffer`. |
| `src/emel/kernel/aarch64/detail.hpp` | Best place to host backend-local flash helpers without changing SML structure. |
| `src/emel/kernel/detail.hpp` | Keep shared canonical validator as source of truth; only touch if a tiny helper extraction is needed to share non-optimized validation or scalar reference code. |
| `tests/kernel/aarch64_tests.cpp` | Primary proof surface for optimized-path correctness, path-selection truth, and zero-allocation behavior. |
| `tests/kernel/lifecycle_tests.cpp` | Keep backend-route acceptance/rejection proof aligned with the canonical contract. |
| `tests/kernel/test_helpers.hpp` | Likely place for larger canonical fixtures or an allocation-guard helper reused by AArch64 tests. |

## Standard Stack

### Core
| Library / Component | Version | Purpose | Why Standard |
|---------------------|---------|---------|--------------|
| `src/emel/kernel/aarch64/actions.hpp` + `src/emel/kernel/aarch64/detail.hpp` | repo local | Backend-local AArch64 flash execution seam | Existing backend specialization pattern already lives here; Phase 14 can swap math without touching orchestration. |
| `src/emel/kernel/aarch64/context.hpp` | repo local | Persistent backend-owned flash scratch | Already owns reusable flash workspace and is the correct no-allocation ownership boundary. |
| `src/emel/kernel/detail.hpp` | repo local | Canonical request validation and scalar reference behavior | Shared validator already defines the maintained flash contract consumed by generator and tests. |
| Arm ACLE Advanced SIMD / NEON intrinsics | current toolchain / AArch64 target | FMA, loads, lane reductions, and vector math for the optimized kernel | Official Arm SIMD guidance and the repo’s existing AArch64 numeric code both assume intrinsics-based specialization on AArch64. |

### Supporting
| Library / Component | Version | Purpose | When to Use |
|---------------------|---------|---------|-------------|
| doctest | repo-pinned local header | Unit proof surface | Phase 14 kernel-local correctness, path-selection truth, workspace reuse, and allocation checks. |
| `tmp/llama.cpp/ggml/src/ggml-cpu/ops.cpp` | vendored local snapshot | Reference arithmetic and CPU flash structure | Borrow online-softmax recurrence and tiling ideas, not full API semantics. |
| `scripts/quality_gates.sh` | repo local | Required repo verification gate | Run after implementation changes even though Phase 14 proof itself stays kernel-local. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Backend-local AArch64 helper under existing action seam | New flash-specific SML guards or states | Violates the phase constraint and adds architecture churn where the problem is purely numeric. |
| Fixed-capacity backend-owned scratch | Per-call `std::vector` tile packing | Easier to prototype, but violates the hot-path allocation contract. |
| Canonical F32 single-query kernel | Broader ggml-like feature or dtype parity now | Would widen scope beyond Phase 14 and dilute truthful claims. |
| Single-dispatch SIMD kernel | Split-KV threaded reduction like ggml | EMEL dispatch validation currently forces `ith == 0` and `nth == 1`, so threading changes would widen the contract. |

## Architecture Patterns

### Recommended Project Structure
```text
src/emel/kernel/
├── detail.hpp                # shared canonical flash validator and scalar reference
└── aarch64/
    ├── actions.hpp           # existing action seam; call backend-local flash helper here
    ├── context.hpp           # persistent reusable flash scratch
    ├── detail.hpp            # backend-local flash kernels and packing helpers
    ├── guards.hpp            # keep structural contract unchanged unless validator split is required
    └── sm.hpp                # leave rows unchanged

tests/kernel/
├── aarch64_tests.cpp         # optimized-path proof surface
├── lifecycle_tests.cpp       # backend route acceptance/rejection
└── test_helpers.hpp          # canonical flash fixtures and shared test utilities
```

### Pattern 1: Keep the Existing AArch64 SM Contract Intact
**What:** Leave `valid_op_flash_attn_ext` / `exec_op_flash_attn_ext` rows in
`src/emel/kernel/aarch64/sm.hpp` unchanged and replace only the math reached by
`action::exec_op_flash_attn_ext`.

**When to use:** Entire Phase 14.

**Why:** The current machine already expresses the correct orchestration boundary. The missing piece
is backend-local optimized execution, not routing.

### Pattern 2: Shared Validator, Backend-Local Executor
**What:** Continue using `kernel/detail.hpp::can_run_flash_attn_ext(...)` as the canonical request
contract, but move AArch64 flash execution into backend-local helpers.

**When to use:** Supported canonical requests.

**Why:** The validator already matches the Phase 11 generator request shape over position-major K/V
cache views. Replacing only the executor preserves the maintained contract and avoids broadening
backend-only rules into public behavior.

### Pattern 3: Single-Query Online Softmax With Fixed Scratch
**What:** Use a per-head online-softmax recurrence with fixed reusable scratch:

1. Load one Q head.
2. Stream over K/V tokens for the mapped KV head.
3. Update running `M` and `S`.
4. Rescale the output accumulator when `M` increases.
5. Accumulate weighted V directly into fixed backend-owned buffers.

**When to use:** The maintained canonical `query_count == 1` flash path.

**Why:** ggml’s reference CPU flash path uses the same recurrence, and it removes the current
shared scalar helper’s extra score-buffer pass from the hot loop while staying bounded and
allocation-free.

### Anti-Patterns to Avoid
- **Do not change AArch64 SM structure:** no new states, queues, deferred work, or event widening.
- **Do not silently keep the shared scalar helper on the supported ARM path:** PORT-01 is specifically about replacing it for canonical AArch64 requests.
- **Do not port ggml’s split-KV threadpool path:** EMEL explicitly validates `ith == 0` and `nth == 1`.
- **Do not widen the flash contract in this phase:** no mask, sinks, softcap, ALiBi, broader dtype claims, or shipped runtime adoption.
- **Do not store per-dispatch request metadata in context:** only persistent reusable buffers and durable observability fields belong there.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Flash routing | New flash-specific actor flow | Existing `dispatch_op_flash_attn_ext` rows | The actor contract is already correct. |
| Path-selection proof | Tool-local counters or benchmark evidence | Kernel-local doctest proof in `tests/kernel/aarch64_tests.cpp` | Phase 14 is kernel-local only. |
| Temporary scratch | Per-dispatch heap buffers | Fixed backend-owned workspace in `aarch64::action::context` | Required for PORT-02. |
| Generic ggml parity | Support for mask/sinks/softcap/multi-thread chunk reduction | Canonical EMEL flash subset only | Current event shape and dispatch contract do not express the full ggml feature set. |
| Scalar reference duplication | A second independent scalar implementation in AArch64 files | Existing shared scalar helper as correctness oracle in tests only | Reduces drift while keeping the shipped ARM path optimized. |

**Key insight:** Phase 14 should add an AArch64-native executor, not a second architecture.

## Common Pitfalls

### Pitfall 1: Assuming the Current AArch64 Flash Path Is Already Specialized
**What goes wrong:** Planning treats the current ARM flash route as partly optimized.
**Why it happens:** The backend owns `flash_attn_workspace`, but `exec_op_flash_attn_ext` still
calls `kernel/detail.hpp::run_flash_attn_ext_with_workspace(...)`.
**How to avoid:** Make the first implementation task replace that call site with a backend-local
helper and add a test that would fail if the shared helper were still used.
**Warning signs:** `simd_supported_request_v` still excludes `op_flash_attn_ext`, or the only
flash math remains in `kernel/detail.hpp`.

### Pitfall 2: Accidentally Broadening to ggml’s Full Contract
**What goes wrong:** The implementation starts carrying mask, sinks, softcap, or dtype-general
logic and Phase 14 scope expands.
**Why it happens:** ggml’s operator name matches EMEL’s, but the semantics do not.
**How to avoid:** Treat ggml as arithmetic reference only. Keep the maintained EMEL contract equal
to the current validator and generator request builder.
**Warning signs:** New meaning assigned to extra `op_params` slots or new assumptions about
`src3`-like tensors that EMEL does not have.

### Pitfall 3: Importing ggml’s Threading Model
**What goes wrong:** The planner tries to port chunked KV reduction or `ith`/`nth` behavior.
**Why it happens:** ggml’s CPU flash path gets much of its structure from threadpool reduction.
**How to avoid:** Keep Phase 14 single-dispatch and single-threaded inside the existing event
contract. Optimize within one request only.
**Warning signs:** Proposed changes to `validate_dispatch_request(...)`, threadpool helpers, or
request threading fields.

### Pitfall 4: Breaking the Workspace Proof While Improving the Kernel
**What goes wrong:** The math gets faster but tests can no longer prove reuse or no-allocation.
**Why it happens:** Current proof only checks `prepared_tokens` and `reuse_count`; richer scratch
needs may tempt ad hoc buffers or untracked state.
**How to avoid:** Keep reusable scratch inside `flash_attn_workspace` and preserve explicit reuse
observability or replace it with equally direct kernel-local proof in the same phase.
**Warning signs:** New `std::vector`, `resize`, or request-derived state stored in context.

## Code Examples

Verified local patterns to preserve:

### Existing Action Seam
```cpp
template <class dispatch_event_type>
struct exec_scalar_op {
  void operator()(const dispatch_event_type & ev, context & ctx) const noexcept {
    using request_type = std::remove_cvref_t<decltype(ev.request)>;
    if constexpr (std::is_same_v<request_type, ::emel::kernel::event::op_flash_attn_ext>) {
      // Phase 14 replacement seam: backend-local AArch64 flash helper belongs here.
    }
  }
};
```

### ggml Reference Recurrence Worth Reusing
```cpp
// Per query head:
// M = running max
// S = running softmax sum
// VKQ = running weighted V accumulator
//
// for each token:
//   score = dot(q, k[token]) * scale
//   if (score > M) rescale VKQ and S
//   accumulate V[token] with exp(score - M)
// normalize once at the end
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Shared scalar two-pass flash helper in `kernel/detail.hpp` | Backend-local SIMD flash helper under the same action seam | Phase 14 target | Satisfies PORT-01 without SML churn. |
| Materialized score/probability thinking inherited from old attention code | Online-softmax accumulation with fixed scratch | ggml CPU reference already uses this | Better fit for zero-allocation AArch64 optimization. |
| Threadpool chunk/reduce flash execution | Single-dispatch backend-local SIMD within `ith == 0`, `nth == 1` | EMEL current contract | Keeps Phase 14 truthful and architecture-safe. |

**Deprecated/outdated for this phase:**
- Treating the current AArch64 flash route as “already specialized” because it has backend-local
  workspace ownership.
- Using paritychecker or bench as completion proof for Phase 14.

## Open Questions

1. **Should the AArch64 optimized path preserve the current `prepared_tokens` / `reuse_count` contract exactly?**
   - What we know: current AArch64 and x86_64 tests assert those fields directly.
   - What's unclear: whether Phase 14 wants richer workspace observability than token-count reuse.
   - Recommendation: preserve the current fields in Phase 14 and add any new scratch metadata additively.

2. **Does the optimized kernel need packed K/V tile scratch or only accumulator scratch?**
   - What we know: the canonical generator layout already stores each token-head vector contiguously in head dimension.
   - What's unclear: whether repacking across tokens is still worth it for the maintained head sizes.
   - Recommendation: start with direct contiguous loads over each token-head row plus fixed accumulators; only add packed-token scratch if tests and local benchmarking inside the phase show a clear need.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | doctest (repo local via `third_party/doctest`) |
| Config file | `CMakeLists.txt` |
| Quick run command | `./build/debug/emel_tests_bin --test-case='*aarch64*flash_attn*' --no-breaks --force-colors=0` |
| Full suite command | `ctest --test-dir build/debug --output-on-failure -R emel_tests` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PORT-01 | Canonical AArch64 `op_flash_attn_ext` requests execute through backend-local optimized code, not the shared scalar workspace helper | unit | `./build/debug/emel_tests_bin --test-case='*aarch64*flash_attn*optimized*' --no-breaks --force-colors=0` | ✅ |
| PORT-01 | Canonical backend route still accepts supported requests and rejects out-of-contract shape mutations | unit | `./build/debug/emel_tests_bin --test-case='*flash_attn_ext*canonical*' --no-breaks --force-colors=0` | ✅ |
| PORT-02 | Repeated canonical AArch64 flash dispatches reuse backend-owned workspace | unit | `./build/debug/emel_tests_bin --test-case='*aarch64*flash_attn*workspace*' --no-breaks --force-colors=0` | ✅ |
| PORT-02 | Optimized AArch64 flash dispatch performs zero heap allocations | unit | `./build/debug/emel_tests_bin --test-case='*aarch64*flash_attn*alloc*' --no-breaks --force-colors=0` | ✅ |

### Sampling Rate
- **Per task commit:** `./build/debug/emel_tests_bin --test-case='*aarch64*flash_attn*' --no-breaks --force-colors=0`
- **Per wave merge:** `ctest --test-dir build/debug --output-on-failure -R emel_tests`
- **Phase gate:** `scripts/quality_gates.sh`

### Wave 0 Gaps
- [ ] `tests/kernel/aarch64_tests.cpp` needs a direct proof that canonical AArch64 flash no longer calls the shared scalar helper.
- [ ] `tests/kernel/aarch64_tests.cpp` needs a zero-allocation flash dispatch test; current repo allocation trapping exists only in `tests/graph/graph_tests.cpp`.
- [ ] `tests/kernel/test_helpers.hpp` likely needs a larger canonical fixture to exercise tile or loop tails beyond the current 2-token smoke case.

## Sources

### Primary (HIGH confidence)
- `docs/rules/sml.rules.md` - locked RTC/no-queue/bounded-work contract for all machine work.
- `AGENTS.md` - project-specific constraints on SML structure, context rules, hot-path allocation, and backend specialization.
- `src/emel/kernel/detail.hpp` - current shared flash validator and scalar/workspace executor.
- `src/emel/kernel/aarch64/actions.hpp` - current AArch64 flash insertion point and existing NEON specialization style.
- `src/emel/kernel/aarch64/context.hpp` - current backend-owned reusable workspace contract.
- `src/emel/kernel/aarch64/sm.hpp` - existing unchanged Phase 14 dispatch structure.
- `tests/kernel/aarch64_tests.cpp` - current AArch64 flash workspace proof surface.
- `tests/kernel/lifecycle_tests.cpp` - current canonical acceptance/rejection proof surface.
- `tests/kernel/test_helpers.hpp` - canonical flash fixtures.
- `tests/generator/detail_tests.cpp` and `src/emel/generator/detail.hpp` - current generator request shape and position-major K/V layout the validator already accepts.
- `tmp/llama.cpp/ggml/src/ggml-cpu/ops.cpp` and `tmp/llama.cpp/ggml/src/ggml-cpu/ops.h` - current vendored CPU flash-attention reference structure.
- Arm official SIMD/ACLE guidance: https://developer.arm.com/servers-and-cloud-computing/arm-simd

### Secondary (MEDIUM confidence)
- Arm official learning path on function multiversioning: https://learn.arm.com/learning-paths/cross-platform/function-multiversioning/

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - nearly all guidance comes from current repo seams plus official Arm SIMD guidance.
- Architecture: HIGH - the existing AArch64 SM/action/context structure makes the allowed insertion points explicit.
- Pitfalls: HIGH - they are directly grounded in current repo code, current validator behavior, and the vendored ggml reference mismatch.

**Research date:** 2026-03-22
**Valid until:** 2026-04-21
