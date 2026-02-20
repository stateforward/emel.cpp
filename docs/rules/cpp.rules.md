---
trigger: always_on
globs: src/**/*, include/**/*, tests/**/*
---

# C/C++ Real-Time Governance Rules (2025-2026)

## 1. Scope and assumptions

- **CPPRT-001** MUST comply with the **RTC**, **no-queue**, **determinism**, **single-writer**, **no-heap-during-dispatch**, and **bounded-work** invariants defined in `sml.rules.md` for any Boost.SML-based actor/state-machine dispatch chain.

- **CPPRT-002** MUST treat the following as **dispatch-critical** (real-time constrained) code paths:
  - `boost::sml::sm<...>::process_event(...)` execution, including guards, actions, entry/exit actions, and anonymous/internal transitions.
  - Any synchronous cross-actor call that occurs inside the above chain.

- **CPPRT-003** MUST separate execution into **Initialization/Configuration phase** (not time critical) and **Dispatch/RTC phase** (time critical).
  MUST document (in code comments or module docs) which functions are allowed in the Dispatch/RTC phase.

- **CPPRT-004** MUST define “deterministic” as: for a fixed build (compiler + flags), fixed hardware/OS configuration, and identical initial state and input event sequence (including payloads), the observed state transitions and side effects are identical.

- **CPPRT-005** MUST NOT read time, randomness, filesystem state, network state, environment variables, or global mutable process state directly from the Dispatch/RTC phase.
  MUST inject such information **only** via explicit event payloads or precomputed immutable configuration captured during Initialization.

- **CPPRT-006** MUST assume **hard real-time constraints** for rules that affect determinism and bounded latency (unless a rule explicitly says MAY/SHOULD for soft real-time).

- **CPPRT-007** MUST treat any operation with unbounded or input-dependent worst-case latency as forbidden in Dispatch/RTC (including blocking I/O, paging, locks, and dynamic allocation).

- **CPPRT-008** MUST NOT create background worker threads, task pools, or asynchronous jobs from within an actor’s Dispatch/RTC execution (aligns with “no message queue” and RTC invariants).

- **CPPRT-009** MUST use the terms **actor**, **dispatch**, **event**, **RTC chain**, and **dispatch-critical** consistently across the codebase and in reviews.

- **CPPRT-010** If a rule is violated for a specific subsystem, the violation MUST be:
  - explicitly documented near the code,
  - covered by a dedicated test that bounds the risk,
  - and tracked by an issue with an owner and removal plan.

## 2. Target standards (C17/C23, C++20/C++23) and compiler assumptions

- **CPPRT-011** MUST compile all C code as **C17** or newer (`-std=c17` or `-std=c23`).
  MUST compile all C++ code as **C++20** or newer (`-std=c++20` or `-std=c++23`).  
  MUST NOT rely on compiler extensions unless wrapped behind a portability layer and feature-tested.

- **CPPRT-012** MUST declare whether each target is **freestanding** or **hosted** (C/C++ standard terms) and MUST gate library usage accordingly (e.g., no `<iostream>` in freestanding builds).

- **CPPRT-013** MUST require a toolchain that correctly implements:
  - C11/C17 atomics (`<stdatomic.h>`) for C targets that use concurrency, or an explicit platform atomic layer.
  - C++11 atomics (`<atomic>`) for all C++ targets that use concurrency.

- **CPPRT-014** MUST compile with warnings enabled and treated as errors for production builds (`-Werror` or equivalent).
  MUST explicitly document any warning suppressions and keep them narrowly scoped.
  (Ref: GCC “-pedantic / -pedantic-errors” behavior: https://gcc.gnu.org/onlinedocs/gcc/Warnings-and-Errors.html)

- **CPPRT-015** SHOULD enable at least: `-Wall -Wextra -Wpedantic` (or MSVC equivalents) on all builds.
  SHOULD additionally enable hardening warnings appropriate for the codebase (e.g., `-Wconversion`, `-Wshadow`, `-Wdouble-promotion`) and fix violations rather than suppressing them.

- **CPPRT-016** MUST pin and version the compiler and standard library used in CI for each target triple.
  MUST record compiler version and key flags in build artifacts (e.g., `--version` output embedded into `--build-info`).

- **CPPRT-017** MUST define at least two build profiles:
  - **rt-debug** (instrumentation, sanitizers allowed, deterministic flags preserved)
  - **rt-release** (optimized, still deterministic, no sanitizer overhead)

- **CPPRT-018** MUST NOT use `-Ofast` or `-ffast-math` in Dispatch/RTC binaries that require deterministic numeric behavior, because they may change IEEE/ISO semantics.
  (Ref: GCC Optimize Options `-ffast-math`: https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)

- **CPPRT-019** SHOULD use link-time dead stripping for embedded/code-size-constrained targets (`-ffunction-sections -fdata-sections` + linker GC) when supported.
  MUST verify that dead stripping does not remove required registration/entry points (no reliance on static initialization side effects).

- **CPPRT-020** MUST use compile-time feature detection (e.g., `__has_cpp_attribute`, `__cpp_lib_*`) for optional C++23 features (such as `<expected>`), and provide fallbacks.

- **CPPRT-021** SHOULD maintain a baseline flag set similar to:
  
  ```sh
  # C++ (rt-release)
  clang++ -std=c++20 -O2   -fno-exceptions -fno-rtti   -Wall -Wextra -Wpedantic -Werror   -Wconversion -Wshadow   -fvisibility=hidden
  ```
  
  Flags MUST be tailored per platform/toolchain and validated for jitter/throughput.

## 3. Determinism and undefined behavior rules

- **CPPRT-022** MUST treat **undefined behavior (UB)** as a release-blocking defect in all targets, including embedded and “performance-only” builds.

- **CPPRT-023** MUST run a CI configuration that executes tests with UB detection enabled (e.g., Clang/GCC UBSan).
  (Ref: UBSan overview: https://www.chromium.org/developers/testing/undefinedbehaviorsanitizer/ )

- **CPPRT-024** MUST NOT allow signed integer overflow in either C or C++ (it is UB).
  MUST use checked arithmetic, saturation arithmetic, or unsigned/explicitly widened arithmetic when overflow is possible.
  (Ref: SEI CERT C INT32-C: https://wiki.sei.cmu.edu/confluence/pages/viewpage.action?pageId=87152052)

- **CPPRT-025** MUST guard against UB in shifts and bit operations:
  - shift count MUST be in `[0, bit_width-1]`
  - left-shift on signed values that overflows MUST NOT occur
  - negative shift counts MUST NOT occur

- **CPPRT-026** MUST NOT write expressions whose correctness depends on unspecified/undefined order of evaluation or side effects (common in C/C++).
  (Ref: SEI CERT C EXP30-C “Do not depend on the order of evaluation for side effects”: https://abougouffa.github.io/awesome-coding-standards/sei-cert-c-2016.pdf)

- **CPPRT-027** MUST obey strict aliasing rules; code MUST be correct under `-O2` with strict aliasing enabled.
  MUST NOT “fix” aliasing violations by globally disabling aliasing (`-fno-strict-aliasing`) except as a temporary containment measure with a tracked bug.
  (Ref: GCC on strict-aliasing type punning warning: https://gcc.gnu.org/gcc-4.4/porting_to.html)

- **CPPRT-028** MUST NOT type-pun by `reinterpret_cast`/pointer-cast and dereference.
  MUST use `std::bit_cast` (C++20+) or `memcpy` for bit-level reinterpretation.
  (Ref: `std::bit_cast`: https://en.cppreference.com/w/cpp/numeric/bit_cast.html)

- **CPPRT-029** SHOULD use this pattern for bit reinterpretation:
  
  ```cpp
  #include <bit>
  #include <cstdint>
  
  std::uint32_t f32_bits(float x) noexcept {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    return std::bit_cast<std::uint32_t>(x);
  }
  ```

- **CPPRT-030** MUST NOT use `volatile` for inter-thread synchronization or atomicity.
  MUST use atomics or platform synchronization primitives instead.
  (Ref: C++ Core Guidelines CP.8: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#cp8-dont-try-to-use-volatile-for-synchronization ; C volatile note: https://en.cppreference.com/w/c/language/volatile.html)

- **CPPRT-031** MAY use `volatile` only for:
  - memory-mapped I/O registers,
  - signal/ISR shared flags when paired with the platform’s required barriers,
  - and compiler-visibility constraints for special memory.
  MUST document the hardware contract at the declaration site.

- **CPPRT-032** MUST NOT access objects outside their lifetime (including after placement-new reuse) without following the C++ object lifetime rules.
  (Ref: lifetime rules: https://en.cppreference.com/w/cpp/language/lifetime.html)

- **CPPRT-033** If storage is reused via placement new for a different object lifetime, MUST use `std::launder` as required by the standard when reusing prior pointers.
  (Ref: `std::launder`: https://en.cppreference.com/w/cpp/utility/launder.html)

- **CPPRT-034** MUST treat data races as UB in C/C++ and as release-blocking defects.
  MUST ensure that any shared writable state is protected by the single-writer invariant or by correctly ordered atomics.

- **CPPRT-035** MUST NOT serialize/deserialize by `reinterpret_cast`ing structs or relying on padding/endianness.
  MUST use explicit serialization routines that define byte order and field widths.

- **CPPRT-036** MUST restrict `reinterpret_cast` (C++) and dangerous pointer casts (C/C++) to:
  - low-level boundary code (HAL, serialization, SIMD intrinsics wrappers),
  - with documented justification,
  - and with tests/sanitizer coverage.
  MUST NOT use casts to bypass the type system in general application logic.

- **CPPRT-037** When accessing object representations, MUST use `unsigned char`/`std::byte` pointers or `memcpy`, not incompatible typed pointers.

- **CPPRT-038** MUST NOT read uninitialized variables, padding, or storage.
  MUST initialize all fields explicitly, especially in structs that cross module boundaries.

- **CPPRT-039** MUST NOT perform pointer arithmetic that produces a pointer outside the bounds of the same object (except one-past-the-end as permitted).
  MUST avoid “pointer provenance” violations by using indices/offsets and bounds checks.

## 4. Memory model discipline (stack vs heap, static storage, alignment)

- **CPPRT-040** On hosted OS targets with virtual memory, MUST prevent page faults during Dispatch/RTC by:
  - locking memory where appropriate (`mlockall`/equivalent),
  - and pre-faulting stack/working set during Initialization.
  (Ref: `mlockall` note about reserving locked stack pages: https://man7.org/linux/man-pages/man2/mlock.2.html)

- **CPPRT-041** MUST set and enforce a per-thread stack budget for Dispatch/RTC threads (via linker script, RTOS config, or OS thread attributes).
  MUST measure worst-case stack usage for the dispatch-critical call chain and keep a safety margin.

- **CPPRT-042** MUST store dispatch-critical working state in:
  - actor-owned structs (typically embedded in the actor object),
  - stack allocations with statically known size,
  - or static storage with immutable or single-writer discipline.
  MUST NOT allocate from the heap in dispatch-critical code.

- **CPPRT-043** Any global/static object reachable from Dispatch/RTC MUST be **constant-initialized** (no dynamic initialization).
  MUST NOT rely on C++ dynamic initialization order across translation units.

- **CPPRT-044** MUST NOT use function-local `static` variables that require runtime initialization or guard checks inside Dispatch/RTC paths (may introduce hidden locks and nondeterministic first-use latency).

- **CPPRT-045** MUST explicitly align hot data structures (e.g., ring buffers, tensor tiles, atomic flags) to at least cache-line boundaries when false sharing or unaligned access can affect latency (`alignas(64)` or `std::hardware_destructive_interference_size` where supported).
  (Ref: `std::hardware_destructive_interference_size`: https://en.cppreference.com/w/cpp/thread/hardware_destructive_interference_s