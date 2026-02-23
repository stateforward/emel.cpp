---
title: kernel/dispatch architecture design
status: draft
---

# kernel/dispatch architecture design

this document defines how the kernel domain handles runtime hardware detection and ISA
(Instruction Set Architecture) dispatch without compromising the inference hot path.

## the llama.cpp legacy (procedural function pointers)
in `llama.cpp`, runtime ISA dispatch is handled procedurally:
1. **hardware detection:** at startup, the system executes `CPUID` (x86) or `getauxval` (ARM) to detect features like AVX2, AVX512, or NEON.
2. **global trait tables:** it populates global `type_traits` arrays with function pointers to specifically compiled functions (e.g., `ggml_vec_dot_q8_K_avx512`).
3. **dispatch:** during the hot loop, it performs a function pointer lookup: `vec_dot(...)`.

this provides extremely fast execution (a single pointer dereference) but relies on heavy procedural initialization and global mutable state, which violates emel's strict functional and architectural constraints.

## the emel approach: zero-overhead sml dispatch
`emel` achieves the exact same zero-overhead hot path as `llama.cpp`'s function pointers, but does so entirely declaratively using the `boost::sml` state machine boundary.

### 1. hardware detection at construction
when `kernel::any` is instantiated, it performs the same hardware detection (e.g., checking `CPUID` for `AVX512F`).
it uses these results to construct the active hardware variant (e.g., `kernel::x86_64`) with a specific ISA tier injected into its context.

### 2. compile-time transition tables
unlike traditional dynamic actors, `boost::sml` transition tables are heavily template-metaprogrammed.
when the `graph/processor` dispatches an operation:
`kernel.process_event(op::mul_mat{...})`
the compiler statically resolves the active state and generates an optimized `switch` or jump table. if the action is a simple math kernel call, it is often fully inlined by the compiler.
**the overhead is practically zero**, structurally identical to a direct virtual function call or function pointer dereference.

### 3. declarative fallbacks (`unexpected_event`)
this architecture shines brightest when dealing with unsupported hardware operations.
if the active hardware variant (e.g., `kernel::metal`) does not support an operation (e.g., `op::rwkv_wkv6`), it simply omits that operation from its SML transition table.

when `graph/processor` dispatches that unsupported operation:
- `boost::sml` safely drops it into `sml::unexpected_event`.
- the `kernel::any` wrapper catches the `unexpected_event` and seamlessly routes the payload to a CPU fallback variant (e.g., `kernel::aarch64`).
- no procedural `if (unsupported) { run_cpu(); }` spaghetti code is required. fallbacks are declarative features of the state machine topology.

**preventing fallback loops:** to prevent an infinite SML loop (e.g., if a brand new custom kernel `op::map_custom1` is unsupported by both Metal *and* the CPU fallback), the fallback chain is strictly acyclic (Device -> CPU -> `EMEL_ERR_UNSUPPORTED_OP`). the `kernel::any` wrapper explicitly guards against re-routing unexpected events from the final fallback tier.

## summary
by pushing the hardware dispatch into a flat `sm_any` hierarchy and leveraging compile-time transition tables, `emel` maintains absolute `llama.cpp` performance parity while securing the execution pipeline behind an elegant, type-safe, and declarative Actor Model boundary.
