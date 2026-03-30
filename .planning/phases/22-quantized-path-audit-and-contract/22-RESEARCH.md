# Phase 22: Quantized Path Audit And Contract - Research

**Researched:** 2026-03-25
**Domain:** Canonical ARM Llama-68M quantized operand-path contract
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
### Audit Scope And Truth Source
- Audit the shipped generator -> graph -> processor -> kernel runtime chain, not a tool-only or
  synthetic kernel path.
- Ground every contract claim in EMEL-owned `src/` behavior first, then reflect that contract out
  through maintained proof surfaces.
- Keep the audit anchored to the canonical ARM `tests/models/Llama-68M-Chat-v1-Q2_K.gguf`
  workload already used by paritychecker and bench.

### Contract Classification
- Classify stages into exactly three buckets for this milestone: native quantized, approved
  dense-f32-by-contract, and disallowed fallback.
- Treat the existing quantized `op_mul_mat` path that repacks dense rhs activations into `q8_K`
  blocks as an approved dense-f32-by-contract stage unless Phase 22 evidence proves otherwise.
- Any unsupported or not-yet-ported quantized branch must become an explicit no-claim path rather
  than silently taking a misleading f32 or dequantize-to-f32 fallback.

### Guardrails
- Do not change Boost.SML transition tables or actor ownership in this phase without explicit user
  approval.
- Prefer additive observability and operator-inventory reporting over hidden fallback behavior or
  broad runtime rewrites.
- Keep proof and inventory surfaces deterministic, maintained, and aligned with current repo gates.

### Claude's Discretion
The agent can choose the exact artifact shape for the audit and inventory surfaces, provided the
output stays narrow, machine-readable enough for later proof phases, and clearly maps each stage to
one of the approved contract buckets above.

### Deferred Ideas (OUT OF SCOPE)
- Removing approved dense-f32-by-contract stages is Phase 23 only if the milestone decides those
  stages are truly disallowed for supported canonical requests.
- Full parity/regression enforcement remains Phase 24.
- Benchmark publication refresh remains Phase 25.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| AUD-01 | The canonical Llama-68M ARM generation slice has a maintained operand-path audit that classifies each quantized stage as native quantized, approved dense-f32-by-contract, or disallowed fallback. | This research identifies the exact fixture tensor dtypes, maps them onto the shipped generator/kernel stage order, and separates kernel-native quantized matmuls from generator-side dense stages and unsupported branches. |
| PATH-02 | Unsupported or not-yet-ported quantized cases publish explicit no-claim behavior instead of silently routing through a misleading f32 fallback path. | This research shows the existing kernel rejection contract for unsupported quantized requests and recommends additive parity/bench inventory rows so those paths are published as `no-claim`, not inferred from silent absence. |
</phase_requirements>

## Summary

The current shipped canonical ARM generation slice is narrower and more concrete than the existing
aggregate counters suggest. On the maintained `tests/models/Llama-68M-Chat-v1-Q2_K.gguf` fixture,
the generator-relevant learned tensors are: `token_embd.weight=q2_k`, `output.weight=q6_k`,
`output_norm.weight=f32`, per layer `attn_q/attn_k/ffn_gate/ffn_up=q2_k`,
`attn_v/attn_output/ffn_down=q3_k`, and `attn_norm/ffn_norm=f32`. Because the fixture has two
blocks, the shipped runtime executes exactly `8` q2 matmuls, `6` q3 matmuls, and `1` q6 logits
matmul on the maintained short generation case. That matches the current benchmark snapshot.

Those matmuls are genuinely native quantized in the shipped kernel contract: `src0` remains
quantized `q2_k/q3_k/q6_k`, `src1` is required to be dense `f32`, and the backend repacks that
dense rhs into bounded `q8_K` scratch inside dispatch before executing `q*_K x q8_K` row-dot work.
That is not a dequantize-to-f32 fallback. The important dense seams are elsewhere: the token
embedding row is dequantized into `hidden`, the rest of generator math is dense `f32`, and the
current `run_layer(...)` path never dispatches `op_flash_attn_ext`; it calls `compute_attention(...)`
directly, which is why the maintained `hello / max_tokens=1` snapshot reports zero flash dispatches.

Today’s proof/publication surfaces are incomplete for Phase 22. Paritychecker and bench only publish
aggregate q2/q3/q6 dispatch counts. Those counters cannot name which stage used which dtype, cannot
surface the `token_embd.weight` q2 row materialization at all, and cannot publish explicit no-claim
semantics for unsupported quantized branches. Phase 22 should therefore split cleanly into
`22-01 audit/inventory` and `22-02 no-claim/publication`, with both implemented as additive helper
logic and maintained metadata lines, not actor changes.

**Primary recommendation:** Build one source-of-truth operator inventory from the existing
`execution_view` plus shipped generator/kernel flow, then publish that inventory and explicit
unsupported-path `no-claim` rows through paritychecker and bench without changing SML structure.

## Planner Notes

Use exactly three audit buckets: `native_quantized`, `approved_dense_f32_by_contract`, and
`disallowed_fallback`. Publish unsupported/not-yet-ported branches as explicit `no-claim`
behavior, not as a fourth bucket. Keep all observability additive and wrapper/tool-local.

## Project Constraints (from CLAUDE.md)

- Keep Boost.SML actor structure unchanged unless the user explicitly approves a machine change.
- Treat `src/` machines and helpers as the architecture source of truth; do not ground the phase in
  tool-only assumptions.
- Do not widen public APIs for this phase.
- Do not introduce misleading dequantize-to-f32 fallback in the shipped canonical hot path.
- Keep dispatch deterministic, bounded, allocation-free, and same-RTC.
- Keep reference-linked behavior inside `tools/paritychecker` and `tools/bench` only.
- Use additive wrapper accessors, helpers, tests, and publication surfaces instead of hidden
  behavior changes.
- Use doctest/CTest for tests, and `scripts/quality_gates.sh` remains the implementation gate after
  code changes.
- Do not update snapshots without explicit user consent.

## Standard Stack

### Core
| Library / Module | Version | Purpose | Why Standard |
|------------------|---------|---------|--------------|
| Boost.SML | Project-pinned `v1.1.13` semantics | Actor orchestration contract | Required by repo architecture and phase guardrails. |
| `src/emel/generator/detail.hpp` | Workspace HEAD | Canonical generation-stage truth | This file defines which learned tensors are consumed and how quantized operands enter dense compute. |
| `src/emel/kernel/detail.hpp` + `src/emel/kernel/aarch64/actions.hpp` | Workspace HEAD | Canonical quantized matmul contract | These files define supported quantized request shapes, rhs repack, and rejection behavior. |

### Supporting
| Library / Module | Version | Purpose | When to Use |
|------------------|---------|---------|-------------|
| `tools/paritychecker/parity_runner.cpp` | Workspace HEAD | Maintained proof surface | Use to publish contract rows and failure surfaces grounded in shipped runtime behavior. |
| `tools/bench/generation_bench.cpp` + `tools/bench/bench_main.cpp` | Workspace HEAD | Maintained benchmark/publication surface | Use to emit machine-readable inventory metadata that docs/snapshots already consume. |
| doctest + CTest | Bundled / CMake-managed | Unit and integration coverage | Use for fixture inventory tests, rejection tests, and publication-format tests. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Source-of-truth inventory derived from `execution_view` + generator flow | Heuristic mapping from dispatch counters alone | Counter-only mapping misses `token_embd.weight` and cannot express no-claim branches. |
| Additive wrapper/tool metadata | New events, new states, or new actor | Violates the phase guardrail and adds orchestration churn for a documentation/audit problem. |
| Maintained paritychecker/bench rows | One-off markdown or ad hoc scripts | Manual outputs are not machine-readable enough for later proof phases. |

**Version verification:** External package registry verification is not applicable here. The phase
is grounded in repository-pinned C++ modules and the project-pinned Boost.SML semantics documented
in `docs/rules/sml.rules.md`.

## Architecture Patterns

### Recommended Project Structure
```text
src/
├── emel/generator/detail.hpp      # Stage inventory and bucket classification helper
├── emel/generator/sm.hpp          # Additive wrapper accessors only if tools need them
├── emel/kernel/detail.hpp         # Supported versus disallowed quantized request rules
└── emel/kernel/aarch64/context.hpp # Existing dispatch counters; additive only
tools/
├── paritychecker/parity_runner.cpp # Canonical proof output
├── bench/generation_bench.cpp      # Canonical benchmark evidence capture
├── bench/bench_main.cpp            # Snapshot metadata emission
└── docsgen/docsgen.cpp             # Optional docs projection of new metadata rows
```

### Pattern 1: Inventory From Fixture Tensor Types Plus Shipped Stage Order
**What:** Build the audit from two facts together: the canonical fixture’s tensor dtypes and the
generator’s fixed stage order in `run_layer(...)` / `compute_logits(...)`.

**When to use:** `22-01` audit/inventory plan.

**Canonical stage inventory for the maintained fixture:**

| Stage | Tensor(s) | Current handling | Bucket |
|-------|-----------|------------------|--------|
| Token embedding load | `token_embd.weight=q2_k` | `copy_tensor_row(...)` dequantizes one q2 row into dense `hidden` | `approved_dense_f32_by_contract` |
| Attention Q matmul | `blk.{0,1}.attn_q.weight=q2_k` | kernel `op_mul_mat`, quantized lhs + dense rhs repacked to `q8_K` | `native_quantized` |
| Attention K matmul | `blk.{0,1}.attn_k.weight=q2_k` | same as above | `native_quantized` |
| Attention V matmul | `blk.{0,1}.attn_v.weight=q3_k` | same as above | `native_quantized` |
| Attention output matmul | `blk.{0,1}.attn_output.weight=q3_k` | same as above | `native_quantized` |
| FFN gate matmul | `blk.{0,1}.ffn_gate.weight=q2_k` | same as above | `native_quantized` |
| FFN up matmul | `blk.{0,1}.ffn_up.weight=q2_k` | same as above | `native_quantized` |
| FFN down matmul | `blk.{0,1}.ffn_down.weight=q3_k` | same as above | `native_quantized` |
| Logits matmul | `output.weight=q6_k` | same as above | `native_quantized` |
| Norm / rope / residual / softmax / cache math | `attn_norm/ffn_norm/output_norm=f32`, transient buffers | dense `f32` generator math | `approved_dense_f32_by_contract` |

**Important count reconciliation:** the maintained benchmark row
`optimized_q2_dispatch_calls=8 optimized_q3_dispatch_calls=6 optimized_q6_dispatch_calls=1`
matches the eight q2 matmul stages, six q3 matmul stages, and one q6 logits stage above. It does
not count the q2 token embedding row because that path never dispatches `op_mul_mat`.

**Example:**
```cpp
// Source: src/emel/generator/detail.hpp
if (!rms_norm(backend.hidden, block.attention_norm, backend.rms_epsilon, backend.norm) ||
    !matmul_vector(backend, block.attention_q, backend.norm, backend.q) ||
    !matmul_vector(backend, block.attention_k, backend.norm, backend.k) ||
    !matmul_vector(backend, block.attention_v, backend.norm, backend.v)) {
  return false;
}
```

### Pattern 2: Quantized Matmul Means Quantized LHS + Dense RHS Repacked To `q8_K`
**What:** The shipped supported quantized kernel contract is narrowly defined. `src0` may be
`q2_k/q3_k/q6_k`; `src1` must be dense `f32`; `dst` must be dense `f32`; and the backend repacks
rhs activations into `q8_K` scratch before row-dot execution.

**When to use:** `22-01` classification and `22-02` no-claim publication.

**Example:**
```cpp
// Source: src/emel/kernel/detail.hpp
const bool quantized_path = is_quantized_k_dtype(src0_type) &&
    src1_type == dtype_f32 &&
    dst_type == dtype_f32 &&
    (k % quant::QK_K) == 0u &&
    (k / quant::QK_K) <= quant::MAX_Q8_K_BLOCKS &&
    is_dense_contiguous(request.src1) &&
    is_dense_contiguous(request.dst);
```

```cpp
// Source: src/emel/kernel/aarch64/actions.hpp
::emel::kernel::detail::quant::quantize_row_q8_k_strided(
    b + block * ::emel::kernel::detail::quant::QK_K * n + j,
    n,
    &q8_blocks[block],
    ::emel::kernel::detail::quant::QK_K);
```

### Pattern 3: Publish Contract Rows Through Existing Metadata Surfaces
**What:** Extend the maintained paritychecker/bench metadata style with additive inventory rows,
not new channels. The current bench/docsgen flow already parses `# key=value` metadata lines.

**When to use:** `22-02` publication/no-claim plan.

**Recommended artifact shape:** repeated narrow rows, one logical fact per line.

```text
# generation_operand_contract: case=... stage=token_embedding tensor=token_embd.weight dtype=q2_k bucket=approved_dense_f32_by_contract supported=true note=row_copy_to_hidden
# generation_operand_contract: case=... stage=blk.0.attn_q tensor=blk.0.attn_q.weight dtype=q2_k bucket=native_quantized supported=true note=op_mul_mat_q2_k_x_q8_k
# generation_operand_no_claim: op=op_add src0=q4_0 src1=q4_0 dst=q4_0 behavior=reject_invalid_request claim=none
```

### Anti-Patterns to Avoid
- **Counter-only contract claims:** aggregate q2/q3/q6 counts cannot see token embedding row
  materialization or unsupported-branch rejection semantics.
- **Tool-only proof logic:** `matmul_vector_dequantized(...)` and similar paritychecker helpers are
  reference aids, not shipped-runtime truth.
- **Actor rewrites for observability:** new transitions, queues, or ownership changes are not needed
  for this phase.
- **Calling the current slice “flash-backed” in Phase 22 outputs:** `run_layer(...)` currently uses
  `compute_attention(...)` directly, and the maintained short-case flash counters are zero.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Stage-to-dtype mapping | Manual docs table inferred from snapshots | Shared inventory helper over `execution_view` + current generator stage order | Keeps the audit grounded in the shipped runtime chain. |
| Unsupported quantized-path publication | Silent omission when a path rejects | Explicit `no-claim` metadata rows backed by existing reject semantics | PATH-02 needs published truth, not inference from missing counters. |
| New audit actor or event stream | Extra SML machine or event family | Additive wrapper/helper accessors plus tool metadata | Preserves actor structure and keeps the phase narrow. |
| Separate benchmark-only audit format | One-off markdown or CSV outside maintained flow | Existing bench snapshot metadata + docsgen parser style | Reuses maintained repo gates and docs surfaces. |

**Key insight:** Phase 22 is an inventory/publication problem, not a new compute-path problem. The
planner should exploit the existing generator/kernel truth and only add the missing mapping layer.

## Common Pitfalls

### Pitfall 1: Mistaking Tool-Only Exact Modes For Shipped Runtime
**What goes wrong:** The audit starts from paritychecker helpers like
`matmul_vector_dequantized(...)` and overstates dequantized behavior in the shipped path.
**Why it happens:** The parity tool contains reference and exact modes beside runtime execution.
**How to avoid:** Derive contract rows from `src/emel/generator/detail.hpp` and
`src/emel/kernel/**` first, then use paritychecker only as a publication surface.
**Warning signs:** Findings cite tool-only helpers without any corresponding `src/` path.

### Pitfall 2: Treating Dispatch Counts As A Complete Contract
**What goes wrong:** The audit sees `8/6/1` q2/q3/q6 counts and concludes every quantized operand
in the slice is native quantized.
**Why it happens:** The current counters only cover `op_mul_mat` dispatches.
**How to avoid:** Include non-dispatched quantized stages, especially `token_embd.weight=q2_k`.
**Warning signs:** Token embedding is missing from the inventory.

### Pitfall 3: Confusing Dense RHS Repack With Disallowed Dequant Fallback
**What goes wrong:** The audit labels the supported `q*_K x q8_K` path as a dense-f32 fallback.
**Why it happens:** `src1` arrives as dense `f32`, so the distinction between repack and full
dequant widening gets blurred.
**How to avoid:** Reserve `disallowed_fallback` for paths that leave the supported contract, not
for the existing rhs-to-`q8_K` repack approved by the phase decisions.
**Warning signs:** Recommendations try to “remove fallback” from the already-approved q8 repack.

### Pitfall 4: Missing Existing Reject-Based No-Claim Behavior
**What goes wrong:** The planner assumes unsupported quantized branches still need runtime repair.
**Why it happens:** The rejection contract is visible in kernel guards/tests, not in published
generation outputs.
**How to avoid:** Separate “runtime already rejects” from “publication does not expose that truth.”
**Warning signs:** Proposed work changes kernel behavior instead of publishing it.

### Pitfall 5: Carrying Forward Old Flash Claims Without Checking Current Code
**What goes wrong:** Phase 22 output implies the current maintained slice still proves flash use.
**Why it happens:** Older milestone memory overrides present code and current snapshot evidence.
**How to avoid:** State plainly that current `run_layer(...)` uses `compute_attention(...)` and the
maintained short-case flash counters are zero.
**Warning signs:** Audit text says “flash-backed canonical path” without citing current source.

## Code Examples

Verified patterns from local source:

### Canonical Generator Matmul Dispatch
```cpp
// Source: src/emel/generator/detail.hpp
emel::kernel::event::op_mul_mat ev{
    .src0 = make_src_view(matrix),
    .src1 = make_src_view(
        input.data(), static_cast<uint64_t>(1u), static_cast<uint64_t>(input.size())),
    .dst = make_dst_view(
        output.data(), static_cast<uint64_t>(1u), static_cast<uint64_t>(output.size())),
    .nth = 1,
};
backend.kernel.set_kind(backend.kernel_kind);
const bool ok = backend.kernel.process_event(ev);
```

### Canonical Quantized Matmul Branch
```cpp
// Source: src/emel/kernel/detail.hpp
if (valid && quantized_src0) {
  for (uint64_t block = 0; block < block_count; ++block) {
    quant::quantize_row_q8_k_strided(
        b_dense + block * quant::QK_K * n + j, n, &q8_blocks[block], quant::QK_K);
  }
  if (src0_type == dtype_q2_k) {
    c_dense[i * n + j] = dot_q2_k_q8_k_row_scalar(...);
  } else if (src0_type == dtype_q3_k) {
    c_dense[i * n + j] = dot_q3_k_q8_k_row_scalar(...);
  } else {
    c_dense[i * n + j] = dot_q6_k_q8_k_row_scalar(...);
  }
}
```

### Existing Aggregate Publication Surface
```cpp
// Source: tools/bench/bench_main.cpp
std::printf("# generation_quantized_evidence: case=%.*s optimized_q2_dispatch_calls=%" PRIu64
            " shared_q2_dispatch_calls=%" PRIu64
            " optimized_q3_dispatch_calls=%" PRIu64
            " shared_q3_dispatch_calls=%" PRIu64
            " optimized_q6_dispatch_calls=%" PRIu64
            " shared_q6_dispatch_calls=%" PRIu64 "\n", ...);
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Aggregate q2/q3/q6 counters only | Stage-level inventory plus explicit no-claim rows | Needed in Phase 22 | Lets the repo publish contract truthfully instead of implying it from totals. |
| Tool-local exact/dequant helpers used for understanding | `src/` runtime path is the truth source; tools only reflect it | Existing repo rule, enforced here | Prevents false “fallback” claims. |
| Implicit unsupported-branch behavior | Published reject-based no-claim behavior | Needed in Phase 22 | Satisfies PATH-02 without changing actor structure. |

**Deprecated/outdated:**
- “Quantized proof equals optimized_q2/q3/q6 counts only”: outdated for Phase 22 because it omits
  token embedding and cannot describe unsupported branches.
- “Current canonical slice uses flash dispatch in the maintained short case”: outdated for the
  current code and benchmark snapshot.

## Open Questions

1. **Should `token_embd.weight=q2_k` row materialization stay approved or become a Phase 23 closure target?**
   - What we know: it is a whole-row q2-to-f32 materialization in the shipped runtime path.
   - What's unclear: whether milestone leadership wants that stage treated as permanently approved or
     merely temporarily approved.
   - Recommendation: in Phase 22, classify it as `approved_dense_f32_by_contract` and surface it
     explicitly so Phase 23 can make an informed closure decision.

2. **How much Phase 22 publication should reach docsgen now?**
   - What we know: bench snapshots and docs already parse metadata rows, and paritychecker already
     prints failure-surface lines.
   - What's unclear: whether docs/benchmarks publication is required in this phase or can wait for
     Phase 25.
   - Recommendation: implement the metadata rows in bench/parity now; wire docsgen only if the
     incremental parser change is small and snapshot updates are approved.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | doctest via `emel_tests_bin` under CTest |
| Config file | `CMakeLists.txt` |
| Quick run command | `ctest --test-dir build --output-on-failure -R emel_tests` |
| Full suite command | `scripts/quality_gates.sh` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AUD-01 | Canonical fixture inventory maps each quantized stage to the correct bucket and stage name | unit + integration | `ctest --test-dir build --output-on-failure -R emel_tests` | ✅ |
| PATH-02 | Unsupported quantized branches publish explicit no-claim / rejection truth | unit + integration | `ctest --test-dir build --output-on-failure -R emel_tests` | ✅ |

### Sampling Rate
- **Per task commit:** `ctest --test-dir build --output-on-failure -R emel_tests`
- **Per wave merge:** `scripts/quality_gates.sh`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] Extend `tests/generator/detail_tests.cpp` or `tests/generator/lifecycle_tests.cpp` with a
  canonical fixture inventory assertion that proves the `q2=8 / q3=6 / q6=1` matmul mapping and the
  `token_embd.weight=q2_k` dense-row materialization classification.
- [ ] Extend `tests/kernel/lifecycle_tests.cpp` with explicit coverage for published no-claim
  branches that already reject at the kernel boundary.
- [ ] Extend `tools/paritychecker/paritychecker_tests.cpp` and the bench/docs metadata tests so new
  contract rows are parse-checked and failure messages stay deterministic.

## Sources

### Primary (HIGH confidence)
- Local source: `src/emel/generator/detail.hpp` - shipped generator stage order, token embedding row
  materialization, dense attention path, and kernel dispatch seam
- Local source: `src/emel/generator/sm.hpp` - additive dispatch counters exposed to paritychecker and
  bench
- Local source: `src/emel/kernel/detail.hpp` - supported quantized request contract and reject logic
- Local source: `src/emel/kernel/aarch64/actions.hpp`, `src/emel/kernel/aarch64/context.hpp`,
  `src/emel/kernel/aarch64/sm.hpp`, `src/emel/kernel/aarch64/guards.hpp` - optimized/shared
  quantized dispatch path and existing observability pattern
- Local source: `tools/paritychecker/parity_runner.cpp` - maintained proof surface and current
  aggregate publication
- Local source: `tools/bench/generation_bench.cpp`, `tools/bench/bench_main.cpp`,
  `tools/docsgen/docsgen.cpp` - maintained benchmark metadata and docs projection surfaces
- Local source: `tests/kernel/aarch64_tests.cpp`, `tests/kernel/lifecycle_tests.cpp`,
  `tests/generator/lifecycle_tests.cpp` - existing proof of dispatch counters and rejection behavior
- Fixture metadata audit: direct GGUF header parse of `tests/models/Llama-68M-Chat-v1-Q2_K.gguf`
  confirming generator-relevant tensor dtypes for the canonical fixture

### Secondary (MEDIUM confidence)
- `snapshots/bench/benchmarks_compare.txt` and `docs/benchmarks.md` - current published aggregate
  evidence showing the maintained short-case `8/6/1` q2/q3/q6 totals and zero flash dispatches

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - entirely repository-pinned and directly inspected
- Architecture: HIGH - derived from shipped generator/kernel source and maintained tools
- Pitfalls: HIGH - grounded in direct source/test/publication mismatches found during audit

**Research date:** 2026-03-25
**Valid until:** 2026-04-24
