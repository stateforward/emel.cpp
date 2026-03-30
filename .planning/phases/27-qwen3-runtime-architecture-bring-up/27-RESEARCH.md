# Phase 27: Qwen3 Runtime Architecture Bring-Up - Research

**Researched:** 2026-03-27
**Domain:** Qwen3 execution-view binding, generator runtime ordering, and truthful maintained architecture support
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
### Explicit Runtime Topology
- Treat `qwen3` as an explicit architecture branch with its own required tensor contract instead of
  a broad Llama-family alias.
- Require `blk.%d.attn_q_norm.weight` and `blk.%d.attn_k_norm.weight` for the canonical Qwen3
  slice and bind them explicitly in the execution view.
- Keep output projection explicit: use `output.weight` when present, with a narrow tied-embedding
  fallback only if the canonical runtime helper can prove that case safely.

### Attention Semantics
- Match the dense Qwen3 reference order: RMS norm hidden state, matmul Q/K/V, reshape by head,
  apply Q/K RMS normalization, then apply RoPE.
- Keep Q/K norm handling in allocation-free runtime helpers inside the existing generator path; no
  ad hoc hidden fallback and no state-machine rewrite.
- Preserve the existing flash/nonflash selection contract while making the underlying Q/K tensors
  truthful for Qwen3.

### Runtime Contract Publication
- Expand quantized-path auditing only as far as needed to account explicitly for any new dense-f32
  vector stages introduced by Qwen3.
- Preserve the prior Llama runtime path as a first-class supported slice rather than regressing it
  during the Qwen3 bring-up.
- Keep broader architecture handling deferred; this phase proves one canonical dense Qwen3 slice
  only.

### the agent's Discretion
- The exact helper split between execution-view binding, quantized-path audit accounting, and
  generator detail kernels can stay local as long as the resulting runtime behavior is explicit and
  architecture-guarded.

### Deferred Ideas (OUT OF SCOPE)
- `qwen3moe`, `qwen3next`, `qwen35`, or other broader Qwen-family topology support.
- Metadata-driven chat-template rendering and richer request surfaces.
- Performance tuning beyond the minimum truthful Qwen3 runtime bring-up.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| RUN-01 | EMEL can initialize and generate on the canonical Qwen3-0.6B fixture through the maintained generator path in `src/emel`. | Existing `generator::sm` can stay unchanged if `prepare()` accepts a truthful Qwen3 execution view, binds the extra Q/K norm tensors, and runs the correct per-layer ordering. |
| RUN-02 | The maintained Qwen3 runtime support explicitly handles the canonical slice's required `qwen3` topology instead of pretending broad Llama-family compatibility. | Local GGUF tensor scan, current `build_execution_view()` rejection, and upstream Qwen3/llama.cpp references all show the maintained path needs an explicit architecture gate and Qwen3-only tensor contract. |
</phase_requirements>

## Summary

The current maintained runtime is still Llama-shaped in exactly the places that matter for this
phase. `src/emel/model/data.cpp` rejects any architecture other than `"llama"`, its block view has
no slots for `attn_q_norm` / `attn_k_norm`, and `src/emel/generator/detail.hpp` applies RoPE
immediately after Q/K projection with no per-head Q/K RMS normalization stage. The canonical local
fixture contradicts all of that: its GGUF metadata advertises `general.architecture=qwen3`, its KV
space uses `qwen3.*` keys, and its tensor set includes `output.weight`,
`blk.%d.attn_q_norm.weight`, and `blk.%d.attn_k_norm.weight`.

Upstream Qwen3 references agree on the required runtime order. Hugging Face Qwen3 applies input RMS
norm, projects Q/K/V, reshapes by head, applies head-dimension RMSNorm to Q and K, then applies
RoPE. Current llama.cpp `qwen3.cpp` does the same and keeps `output.weight` explicit with a tied
embedding fallback only when the tensor is absent. That gives this phase a narrow, concrete target:
add an explicit Qwen3 execution-view contract and route the existing generator hot path through a
Qwen3-specific per-layer helper without touching the Boost.SML topology.

One planning caveat: if the phase gate really means "initialize and generate from the real
`tests/models/Qwen3-0.6B-Q8_0.gguf` file", there is a hidden metadata-normalization dependency.
The existing maintained GGUF metadata population in `tools/paritychecker/parity_runner.cpp` maps
only `llama.*` keys into `model_data.params`, while the real fixture exposes `qwen3.*`. That does
not block the execution-view/generator work itself, but it can block an end-to-end real-fixture
acceptance test unless the plan includes a narrow `qwen3.* -> model_data.params` normalization seam.

**Primary recommendation:** Plan Phase 27 around three code changes and no state-machine rewrite:
publish an explicit Qwen3 execution view with required Q/K norm tensors, run a Qwen3-only layer
kernel order in `generator/detail.hpp`, and extend quantized-path audit/tests so the new dense-f32
vector stages are explicit and Llama remains protected.

## Project Constraints (from CLAUDE.md)

- Preserve the RTC/no-queue Boost.SML actor model; no `process_queue`, `defer_queue`, self-dispatch,
  or hidden async fallback.
- Do not change state-machine structure without asking the user; Phase 27 should stay in existing
  helper/runtime code, not transition-table rewrites.
- Keep runtime control flow explicit and out of SML actions; actions/guards must stay bounded,
  non-blocking, and allocation-free during dispatch.
- Do not mirror dispatch-local request data into machine context; use typed events for phase
  handoff.
- Keep public event payloads small and immutable; prefer non-owning spans/views over owning
  containers in events.
- Keep performance as a first-class constraint; do not replace missing hot-path support with broad
  dequantize-to-f32 fallback or other contract-widening shortcuts.
- Use doctest and `ctest`; run `scripts/quality_gates.sh` after implementation changes.
- Treat `src/` Boost.SML machines as the source of truth and keep reference behavior ports native to
  EMEL-owned code.

## Standard Stack

### Core
| Library / Module | Version | Purpose | Why Standard |
|------------------|---------|---------|--------------|
| `src/emel/model/data.cpp` + `src/emel/model/llama/detail.hpp` | repo HEAD | Maintained execution-view, block lookup, and quantized audit publication | This is the narrowest maintained contract boundary already feeding generator |
| `src/emel/generator/detail.hpp` | repo HEAD | Shipped native runtime hot path for Q/K/V, RoPE, KV cache, flash/nonflash attention, and logits | Phase 27 can stay inside this file and keep the existing generator SML intact |
| `stateforward/sml.cpp` | `02cbea023f035185cfb400e6015c981f9b946bae` | Repo-pinned Boost.SML implementation | Required orchestration stack; no new state-machine library is needed |
| `tests/models/Qwen3-0.6B-Q8_0.gguf` | SHA256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | Canonical truth anchor for architecture name, tensor names, and maintained runtime proof | The phase should follow this file, not broad Qwen-family assumptions |

### Supporting
| Library / Module | Version | Purpose | When to Use |
|------------------|---------|---------|-------------|
| doctest | 2.4.11 | Unit/integration regression coverage | Existing repo test framework for generator/model behavior |
| `tmp/llama.cpp` reference | local commit `3306dbaef` | Current reference tensor contract and Qwen3 layer order | Use as arithmetic/runtime-shape reference only, not architecture source of truth |
| Hugging Face `transformers` Qwen3 model | upstream main as researched 2026-03-27 | Confirms Qwen3 attention ordering and explicit LM head behavior | Use for architecture verification and negative-claim checking |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Explicit Qwen3 execution-view branch | Continue widening `llama`-only helpers implicitly | Smaller diff today, but it keeps the exact over-claim this phase is supposed to remove |
| Qwen3-specific layer helper in `generator/detail.hpp` | Broader generator/orchestrator rewrite | Unnecessary scope expansion and conflicts with the locked "no state-machine rewrite" decision |
| Explicit `output.weight` first, guarded fallback second | Always tie output to embeddings | Simpler code, but wrong for the canonical fixture and hides tensor-contract truth |

**Installation:** None. Use the repo-pinned stack already in `CMakeLists.txt`.

**Version verification:** Boost.SML is pinned by commit in `cmake/sml_version.cmake`, doctest
version `2.4.11` is vendored in `third_party/doctest/doctest/doctest.h`, and no new third-party
dependency is required for Phase 27.

## Architecture Patterns

### Recommended Project Structure
```text
src/emel/model/data.cpp                    # architecture gate + execution-view binding
src/emel/model/llama/detail.hpp            # current maintained execution-view types
src/emel/generator/detail.hpp              # Qwen3-specific per-layer runtime order
tests/model/loader/lifecycle_tests.cpp     # execution-view / required-tensor coverage
tests/generator/detail_tests.cpp           # Q/K norm + RoPE ordering coverage
tests/generator/lifecycle_tests.cpp        # initialize/generate regression for Qwen3 and Llama
```

### Pattern 1: Architecture-gated execution view
**What:** Route maintained execution-view construction by `model_data.architecture_name` and give
Qwen3 its own required tensor contract, including `attn_q_norm` and `attn_k_norm`.
**When to use:** `build_execution_view()`, `lookup_block_view()`, `prepare()`, and any maintained
quantized audit reporting for generator.
**Why:** The local canonical fixture advertises `qwen3`, not `llama`, and its block tensor set is
materially different.
**Example:**
```cpp
if (architecture_name_view(model_data) == "qwen3") {
  bind("attn_q_norm.weight", block_out.attention_q_norm);
  bind("attn_k_norm.weight", block_out.attention_k_norm);
  bind_output_weight_with_guarded_fallback(model_data, view_out);
  return error::none;
}
```

### Pattern 2: Qwen3 attention order lives in a dedicated runtime helper, not in SML
**What:** Keep the generator state machine unchanged and add a Qwen3-only per-layer helper that
implements the reference order inside allocation-free numeric kernels.
**When to use:** `run_layer_*()` / `prepare()` in `src/emel/generator/detail.hpp`.
**Why:** Repo rules require data-plane iteration to stay in helper kernels, and the phase only needs
one truthful runtime slice.
**Example:**
```cpp
rms_norm(hidden, block.attention_norm, backend.rms_epsilon, backend.norm);
matmul_vector(backend, block.attention_q, backend.norm, backend.q);
matmul_vector(backend, block.attention_k, backend.norm, backend.k);
matmul_vector(backend, block.attention_v, backend.norm, backend.v);
apply_qk_norm_per_head(backend.q, block.attention_q_norm, backend.n_head, backend.head_dim);
apply_qk_norm_per_head(backend.k, block.attention_k_norm, backend.n_head_kv, backend.head_dim_kv);
apply_rope(backend.q, backend.n_head, backend.head_dim, backend.n_rot, position, backend.rope_freq_base);
apply_rope(backend.k, backend.n_head_kv, backend.head_dim_kv, backend.n_rot, position, backend.rope_freq_base);
```

### Pattern 3: Quantized audit stays truthful by adding explicit Q/K norm stage families
**What:** Extend the existing stage-family audit to publish Qwen3's new dense vector stages rather
than silently folding them into unrelated categories.
**When to use:** `k_quantized_stage_family_count`, `k_stage_families`, `stage_tensor()`,
`is_vector_dequant_stage()`, and generator stage-count accessors/tests.
**Why:** The phase decision explicitly requires new dense-f32 vector stages to be accounted for, but
only as far as needed for the canonical slice.
**Example:**
```cpp
enum class quantized_stage_family : uint8_t {
  // ...
  attention_q_norm,
  attention_k_norm,
};
```

### Anti-Patterns to Avoid
- **Qwen3-through-Llama aliasing:** Do not just relax the architecture check and hope the current
  Llama block contract is close enough.
- **Wrong Q/K transform order:** Do not apply RoPE before Q/K RMS normalization or before reshaping
  by head.
- **Implicit output fallback:** Do not hide `output.weight` behind a default tied-embedding path;
  the canonical fixture already carries `output.weight`.
- **State-machine churn:** Do not add events, choice states, or transition rows for this phase; the
  work belongs in execution-view and runtime helpers.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Architecture detection | Tensor-name guessing or "Llama-family enough" heuristics | `general.architecture` plus explicit required tensor contract | Removes over-claiming and makes failure modes deterministic |
| New orchestration path | A second generator machine or special Qwen actor | Existing `generator::sm` plus architecture-specific helper routing in `prepare()` | Keeps Phase 27 narrow and aligned with repo SML rules |
| Hot-path math fallback | Whole-matrix dequantize-to-f32 or generic slow-path attention rewrite | Existing `matmul_vector`, `rms_norm`, `apply_rope`, cache, and flash/nonflash helpers | Preserves performance contract and localizes the change |
| Output projection policy | Always reuse token embeddings | Explicit `output.weight` with narrow, provable fallback | Matches canonical fixture and llama.cpp reference behavior |

**Key insight:** Phase 27 is not a "new model family" project. It is a truth-restoration phase for
one concrete tensor contract and one concrete layer order on the maintained generator path.

## Common Pitfalls

### Pitfall 1: Relaxing the architecture guard without widening the tensor contract
**What goes wrong:** `build_execution_view()` accepts `qwen3`, but later block lookup or generator
binding still assumes Llama-only tensors.
**Why it happens:** The current maintained types have only Llama-shaped members.
**How to avoid:** Treat architecture gating and required-tensor publication as one change set.
**Warning signs:** `architecture_name_view(model_data)` accepts `"qwen3"` but no code binds
`attn_q_norm` / `attn_k_norm`.

### Pitfall 2: Applying RoPE before Q/K RMS normalization
**What goes wrong:** Generation initializes, but logits drift because Q/K normalization is in the
wrong place.
**Why it happens:** Current EMEL Llama path is `Q/K/V -> RoPE`, which is not Qwen3's order.
**How to avoid:** Add a Qwen3-only helper and test the exact sequence independently of flash mode.
**Warning signs:** Qwen3 uses the old `run_layer()` body with only extra tensors bolted on.

### Pitfall 3: Forgetting to audit Q/K norm as dense vector stages
**What goes wrong:** Runtime becomes more truthful, but quantized-contract reporting still claims
the old stage counts and hides the new dense-f32 work.
**Why it happens:** The current audit inventory has only 12 stage families and no Q/K norm slots.
**How to avoid:** Extend the audit enum/table and regression counts alongside the runtime change.
**Warning signs:** Qwen3 initializes successfully while stage counts remain `8 native / 4 dense`.

### Pitfall 4: Treating `output.weight` as absent by default
**What goes wrong:** Planner over-schedules tied-embedding fallback work or under-tests explicit
output binding.
**Why it happens:** Some architectures fall back to tied embeddings, and current code already has
that concept in references.
**How to avoid:** Make `output.weight` the primary canonical path and keep fallback guarded.
**Warning signs:** Tests never assert that the canonical fixture or canonical synthetic Qwen3 model
binds `output.weight`.

### Pitfall 5: Assuming real-fixture acceptance will work once runtime math is fixed
**What goes wrong:** Runtime helpers are correct, but real GGUF initialization still fails because
metadata population only reads `llama.*`.
**Why it happens:** The canonical fixture exposes `qwen3.*` metadata keys.
**How to avoid:** Decide early whether the phase gate is synthetic-model truth only or real-fixture
truth; if real fixture is required, plan the metadata-normalization seam explicitly.
**Warning signs:** The phase plan never mentions how `model_data.params` gets populated for a real
Qwen3 GGUF.

## Code Examples

Verified patterns to follow:

### Architecture gate before execution-view publication
```cpp
const auto arch = emel::model::architecture_name_view(model_data);
if (arch == "llama") {
  return build_llama_execution_view(model_data, view_out);
}
if (arch == "qwen3") {
  return build_qwen3_execution_view(model_data, view_out);
}
return emel::error::cast(emel::model::loader::error::model_invalid);
```

### Qwen3 block contract with explicit Q/K norm tensors
```cpp
struct qwen3_block_view {
  tensor_view attention_norm = {};
  tensor_view attention_q = {};
  tensor_view attention_k = {};
  tensor_view attention_v = {};
  tensor_view attention_q_norm = {};
  tensor_view attention_k_norm = {};
  tensor_view attention_output = {};
  tensor_view feed_forward_norm = {};
  tensor_view feed_forward_gate = {};
  tensor_view feed_forward_down = {};
  tensor_view feed_forward_up = {};
};
```

### Quantized audit classification for Q/K norm vectors
```cpp
if (family == quantized_stage_family::attention_q_norm ||
    family == quantized_stage_family::attention_k_norm) {
  return is_f32_type(tensor_type) || is_supported_quantized_type(tensor_type)
             ? quantized_contract_kind::approved_dense_f32_by_contract
             : quantized_contract_kind::explicit_no_claim;
}
```

## State of the Art

| Old Approach | Current Recommended Approach | When Changed | Impact |
|--------------|------------------------------|--------------|--------|
| `build_execution_view()` accepts only `"llama"` and publishes only Llama-shaped block tensors | Execution-view builder branches explicitly by architecture and publishes a Qwen3 block contract with Q/K norm tensors | 2025 upstream Qwen3 support; Phase 27 locally | Prevents false family support claims |
| Attention helper does `RMSNorm -> Q/K/V matmul -> RoPE -> attention` for all maintained dense runtimes | Qwen3 helper does `RMSNorm -> Q/K/V matmul -> reshape -> Q/K RMSNorm -> RoPE -> attention` | 2025 upstream Qwen3 references | Aligns generator math with the canonical architecture |
| Quantized audit inventory exposes 12 Llama-oriented stage families | Audit inventory grows only enough to publish Qwen3's Q/K norm vector stages explicitly | Phase 27 | Keeps runtime-contract reporting truthful |
| Output projection may be reasoned about as tied by default | Canonical path binds `output.weight`; tied embedding remains only a guarded fallback | Existing llama.cpp behavior and local fixture truth | Avoids under-testing the real slice |

**Deprecated/outdated:**
- Treating `qwen3` support as "llama with different prompting".
- Using current `run_layer()` ordering for a Qwen3 slice without Q/K RMS norm.
- Publishing unchanged quantized stage counts after adding Qwen3-only dense vector work.

## Open Questions

1. **Does the Phase 27 acceptance gate require real-GGUF load, or is a truthful in-memory Qwen3 fixture sufficient until parity work in Phase 28?**
   - What we know: the local canonical GGUF exposes `qwen3.*` metadata, and the current maintained metadata population path only maps `llama.*` keys into `model_data.params`.
   - What's unclear: whether Phase 27 will prove RUN-01 through a synthetic prepared model plus generator runtime truth, or through an actual file-backed initialization path.
   - Recommendation: decide this in Wave 0. If the phase gate is real-fixture generation, add a narrow metadata-normalization task explicitly; do not discover it late.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | doctest 2.4.11 |
| Config file | `CMakeLists.txt` |
| Quick run command | `build/zig/emel_tests_bin --source-file=tests/model/loader/lifecycle_tests.cpp --source-file=tests/generator/detail_tests.cpp --source-file=tests/generator/lifecycle_tests.cpp` |
| Full suite command | `scripts/quality_gates.sh` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RUN-01 | Canonical Qwen3 slice initializes and generates through the maintained generator runtime without regressing the current Llama slice | integration | `build/zig/emel_tests_bin --source-file=tests/generator/lifecycle_tests.cpp` | ✅ |
| RUN-02 | Execution view, required tensors, Q/K norm order, and quantized audit expose explicit Qwen3 topology rather than Llama aliasing | unit + integration | `build/zig/emel_tests_bin --source-file=tests/model/loader/lifecycle_tests.cpp --source-file=tests/generator/detail_tests.cpp --source-file=tests/generator/lifecycle_tests.cpp` | ✅ |

### Sampling Rate
- **Per task commit:** `build/zig/emel_tests_bin --source-file=tests/model/loader/lifecycle_tests.cpp --source-file=tests/generator/detail_tests.cpp --source-file=tests/generator/lifecycle_tests.cpp`
- **Per wave merge:** `ctest --test-dir build/zig --output-on-failure -R emel_tests`
- **Phase gate:** `scripts/quality_gates.sh`

### Wave 0 Gaps
None — existing doctest/CTest infrastructure and relevant generator/model test files already exist.
Phase 27 should add Qwen3-specific cases inside those files before implementation changes.

## Sources

### Primary (HIGH confidence)
- Local canonical fixture `tests/models/Qwen3-0.6B-Q8_0.gguf` - verified `general.architecture=qwen3`,
  `qwen3.*` metadata keys, `output.weight`, and per-layer `attn_q_norm` / `attn_k_norm` tensor
  names directly from the file.
- `src/emel/model/data.cpp` - verified current maintained execution-view is Llama-only and lacks
  Q/K norm binding.
- `src/emel/generator/detail.hpp` - verified current maintained layer order and quantized audit
  integration points.
- `https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3/modeling_qwen3.py`
  - verified Qwen3 attention order and explicit LM head behavior.
- `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/models/qwen3.cpp` - verified
  reference Qwen3 graph order in llama.cpp.
- `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp` - verified
  Qwen3 tensor contract and `output.weight` fallback behavior.
- `https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-arch.cpp` - verified
  maintained Qwen3 tensor naming for `attn_q_norm` and `attn_k_norm`.
- `CMakeLists.txt`, `cmake/sml_version.cmake`, `scripts/quality_gates.sh` - verified pinned stack,
  test framework, and required gate command.

### Secondary (MEDIUM confidence)
- `tools/paritychecker/parity_runner.cpp` - verified current maintained real-GGUF metadata
  population is `llama.*`-only, which is highly relevant if Phase 27 includes file-backed
  acceptance.

### Tertiary (LOW confidence)
- None.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - repo-pinned dependencies and local fixture were verified directly.
- Architecture: HIGH - local code, local GGUF tensor scan, Hugging Face Qwen3, and llama.cpp all
  agree on the critical runtime differences.
- Pitfalls: HIGH - each pitfall maps to a verified current-code mismatch or a verified upstream
  contract requirement.

**Research date:** 2026-03-27
**Valid until:** 2026-04-26
