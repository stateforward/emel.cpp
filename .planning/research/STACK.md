# Stack Research

**Domain:** CPU-hosted flash attention for the existing EMEL Llama-68M generation slice
**Researched:** 2026-03-12
**Confidence:** HIGH

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| EMEL-owned `op_flash_attn_ext` kernel in `src/emel/kernel/detail.hpp` with `src/emel/kernel/x86_64/actions.hpp` specialization | Current repo | Fused exact attention operator for the shipped CPU generation path | The kernel event surface already exists in EMEL. Finishing it keeps the milestone inside `src/emel`, preserves the current actor model, and avoids adding a foreign runtime or GPU-only dependency. |
| Existing generator native backend in `src/emel/generator/detail.hpp` | Current repo | Replace score/probability materialization with fused flash-attention dispatch | The current generation path already owns Q/K/V projection, KV cache state, and graph binding. This is the narrowest place to swap in flash attention without inventing a new public API or tool-only compute path. |
| Boost.SML orchestration | v1.1.13 pinned by repo rules | Keep phase control inside existing generator/kernel actors | Flash attention is a data-plane kernel change, not a reason to add queues, worker actors, or asynchronous orchestration. The milestone should preserve RTC and no-queue invariants. |
| `llama.cpp`/`ggml` flash-attention reference in `tools/paritychecker` and `tools/bench` only | Current upstream API shape, March 2026 | Reference correctness and benchmark comparison | `llama.cpp` already exposes `llama_context_params.flash_attn_type` and `ggml_flash_attn_ext`, and its CPU backend implements flash attention. That makes it the right reference surface for parity and compare mode without linking new code into EMEL runtime paths. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| AVX2 intrinsics (`<immintrin.h>`) through the existing `kernel/x86_64` backend | Existing toolchain support | x86_64 fast path for the canonical CPU benchmark target | Use for the hot path in `src/emel/kernel/x86_64/actions.hpp`; keep the scalar/shared implementation as the correctness fallback. |
| Existing `doctest` test stack | Current repo | Kernel and generation verification | Use for a dedicated flash-attention kernel test file and generation-path tests that prove the canonical slice still matches the reference model. |
| Existing `llama.cpp` and `ggml` linkage in `tools/paritychecker` and `tools/bench` | Current repo | Reference-only flash-attention comparison | Use only in tool code. Do not pull these libraries into `src/emel` or any public/runtime EMEL surface. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| Zig C/C++ toolchain | Default build path | No new compiler dependency is needed. Keep the repo-standard Zig build for implementation and performance checks. |
| `ctest` targets `emel_tests` and `lint_snapshot` | Required verification surfaces | Flash-attention changes should remain inside existing CI gates; do not add a parallel ad hoc verification workflow. |
| Existing `tools/paritychecker` and `tools/bench` executables | Acceptance boundary | Reuse the existing canonical generation CLI surfaces. The milestone should not create a new flash-attention-only executable. |

## Installation

```bash
# No new third-party packages are recommended for this milestone.
# Reuse the existing repo toolchain and targets.
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER="zig cc" -DCMAKE_CXX_COMPILER="zig c++"
cmake --build build --target emel_tests bench_runner paritychecker
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| EMEL-native fused flash-attention operator | Dao-AILab `flash-attention` package | Use the external package only in a CUDA/ROCm + PyTorch project. It is the wrong stack for this CPU-hosted C++ milestone. |
| Replace `compute_attention()` with fused streaming attention | Keep the current `attn_scores` + `attn_probs` materialization as the shipped path | Keep the old path only as a temporary debug oracle while bringing up the new kernel. It should not remain the benchmarked implementation. |
| Reuse existing paritychecker and bench surfaces | Add a separate flash-attention benchmark/parity tool | Only do that in a later milestone if the acceptance boundary intentionally expands beyond the existing canonical generation flow. |
| x86_64 AVX2 specialization on top of a shared scalar implementation | Multi-backend rollout across aarch64, CUDA, Metal, Vulkan, and WASM | Only after the canonical CPU Llama-68M path is correct and benchmarked. The current milestone should not broaden into a backend matrix. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Dao-AILab `flash-attention` or any PyTorch/CUDA runtime package in EMEL runtime code | It adds the wrong runtime model and does not fit the existing CPU-hosted C++ generation slice | Implement the operator natively in `src/emel/kernel/**` and keep external references tool-only |
| Whole-score and whole-probability materialization as the final hot path | That defeats the memory-traffic win of flash attention and leaves the milestone benchmarking the old algorithm under a new name | Use online softmax with running max/sum and bounded tile scratch |
| `op_flash_attn_back` for this milestone | Backward pass is irrelevant to canonical generation and would broaden the implementation surface materially | Implement `op_flash_attn_ext` only |
| General `ggml_flash_attn_ext` feature parity for sinks, ALiBi/KQ bias, logit softcap, MLA special V views, or non-causal attention | None of those are required by the current Llama-68M causal generation slice, and they would turn a narrow milestone into generic backend work | Scope the kernel to the exact canonical generation requirements first |
| KV-cache quantization, dual-layout KV caches, or a transposed-V cache migration | That changes the effective operand path and risks violating the repo rule against claiming parity when operand classes diverge | Keep the current EMEL-owned cache representation for this milestone and label results as generation compare, not kernel operand parity |
| New public C API knobs or a new flash-attention-specific CLI | The milestone acceptance boundary is the existing generation, paritychecker, and bench surfaces | Reuse the current initialize/generate flow and existing compare commands |
| Internal threadpools, async work queues, or mailbox-style follow-up dispatch inside SML actors | That conflicts with the repo's RTC/no-queue rules and would add scheduling complexity unrelated to the narrow milestone | Keep one bounded synchronous kernel dispatch per phase; future parallel scheduling must be explicit and outside actor semantics |

## Stack Patterns by Variant

**If the request is prompt prefill (`token_count > 1`):**
- Use a Q-tiled, KV-streaming flash-attention kernel.
- Because the current generator path is materializing `attn_scores` and `attn_probs` across context length, and prefill is where flash attention most clearly removes that memory traffic.

**If the request is single-token decode (`token_count == 1`):**
- Use a single-query KV-streaming reduction with online max/sum accumulation.
- Because the canonical generation benchmark is decode-centric, and the reference CPU path uses chunked partial reduction for long KV ranges in this regime.

**If the host is x86_64 with AVX2 available:**
- Use the existing `kernel/x86_64` runtime dispatch to select an AVX2-specialized flash-attention implementation.
- Because the repo already detects AVX2 at runtime and the canonical compare surface is CPU-hosted.

**If the host is not x86_64 or AVX2 is unavailable:**
- Keep a shared scalar flash-attention implementation as fallback.
- Because correctness and bounded behavior matter more than broad backend rollout in this milestone.

## Integration Points

- `src/emel/kernel/detail.hpp`
  Add `can_run_flash_attn_ext()` and `run_flash_attn_ext()` for the scalar/shared path. This is where the exact attention math, causal masking, running max/sum, and bounded tile scratch should live.
- `src/emel/kernel/x86_64/actions.hpp`
  Add `execute_avx2_flash_attn_ext()` using the repo's existing AVX2 selection pattern. Keep the interface identical to the scalar path so generator code does not branch by backend.
- `src/emel/generator/detail.hpp`
  Replace the current `compute_attention()` implementation, which writes `attn_scores` and `attn_probs`, with a kernel dispatch to `op_flash_attn_ext` over the existing projected Q vector and current KV cache views. Any scratch buffers required by the new operator should be allocated once in `prepare()` and reused.
- `src/emel/generator/context.hpp`
  Keep flash-attention workspace in persistent backend-owned storage, not dispatch-local SML context fields. The current `graph_binding.backend` is the right owner.
- `tools/paritychecker/parity_runner.cpp`
  Keep the same `--generation` CLI, but change the reference context from `LLAMA_FLASH_ATTN_TYPE_DISABLED` to the flash-attention path for the canonical generation comparison. Keep `n_gpu_layers = 0` and the same fixture so the compare stays CPU-only.
- `tools/bench/generation_bench.cpp`
  Keep the same benchmark case names and compare flow, but run the reference path with flash attention enabled so the bench reflects the same algorithm class. If EMEL retains a different KV operand format, report the result as end-to-end generation compare, not kernel parity.

## Relevant Algorithm References

- **FlashAttention (Tri Dao et al., 2022):** use the IO-aware exact attention algorithm and online softmax formulation as the core math reference.
- **FlashAttention-2 (Tri Dao, 2023):** useful for work partitioning ideas, but this milestone only needs the exact CPU-hosted subset, not the full FA-2 feature/performance envelope.
- **Current `llama.cpp` CPU flash-attention implementation:** use it as the practical reference for shape constraints, tiled prefill, single-query chunk reduction, mask handling, and F32 accumulation.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| Boost.SML v1.1.13 | Existing EMEL actor wrappers | No new queueing or thread-safe policy should be introduced for flash attention work. |
| `llama.cpp` `llama_context_params.flash_attn_type` | `tools/paritychecker` and `tools/bench` only | Use it to align the reference generation path with flash attention while keeping the benchmark CPU-only. |
| `ggml_flash_attn_ext` | EMEL `op_flash_attn_ext` milestone subset | The milestone should match the canonical causal self-attention subset, not every optional `ggml` feature. |
| x86_64 AVX2 specialization | Shared scalar fallback | Keep identical semantics across both so parity tests do not depend on CPU feature flags. |

## Sources

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) - core exact-attention algorithm and online softmax reference. Confidence: HIGH.
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) - work partitioning guidance for a performant fused operator. Confidence: HIGH.
- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) - confirms the popular external package is a CUDA/ROCm-oriented stack, which is out of scope for this CPU milestone. Confidence: MEDIUM.
- [ggml `ggml_flash_attn_ext` API](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h) - current reference operator surface. Confidence: HIGH.
- [llama.cpp `flash_attn_type` API](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h) - current reference context toggle for parity/bench surfaces. Confidence: HIGH.
- [llama.cpp graph integration](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp) - current use of `ggml_flash_attn_ext` in model graph construction. Confidence: HIGH.
- [llama.cpp CPU flash-attention implementation](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ops.cpp) - current reference for tiled and chunk-reduction CPU techniques. Confidence: HIGH.
- `src/emel/generator/detail.hpp` - current EMEL generation path still materializes attention scores/probabilities, so this is the main integration point. Confidence: HIGH.
- `src/emel/kernel/events.hpp` and `src/emel/kernel/x86_64/sm.hpp` - EMEL already exposes `op_flash_attn_ext`, so the stack addition is implementation, not interface invention. Confidence: HIGH.
- `tools/paritychecker/parity_runner.cpp` and `tools/bench/generation_bench.cpp` - current reference contexts explicitly disable flash attention today, so these are the required tool-surface touchpoints. Confidence: HIGH.

---
*Stack research for: EMEL v1.2 flash attention*
*Researched: 2026-03-12*
