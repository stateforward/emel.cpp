#pragma once

#include <cstdint>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) &&                             \
    (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

#include "emel/kernel/detail.hpp"

namespace emel::kernel::x86_64::detail {

struct cpuid_registers {
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
};

inline cpuid_registers read_cpuid(const uint32_t leaf,
                                  const uint32_t subleaf) noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  int regs[4] = {};
  __cpuidex(regs, static_cast<int>(leaf), static_cast<int>(subleaf));
  return cpuid_registers{
      .eax = static_cast<uint32_t>(regs[0]),
      .ebx = static_cast<uint32_t>(regs[1]),
      .ecx = static_cast<uint32_t>(regs[2]),
      .edx = static_cast<uint32_t>(regs[3]),
  };
#elif (defined(__GNUC__) || defined(__clang__)) &&                             \
    (defined(__x86_64__) || defined(__i386__))
  uint32_t eax = 0;
  uint32_t ebx = 0;
  uint32_t ecx = 0;
  uint32_t edx = 0;
  __cpuid_count(leaf, subleaf, eax, ebx, ecx, edx);
  return cpuid_registers{.eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx};
#else
  (void)leaf;
  (void)subleaf;
  return {};
#endif
}

inline uint64_t read_xcr0() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  return _xgetbv(0);
#elif (defined(__GNUC__) || defined(__clang__)) &&                             \
    (defined(__x86_64__) || defined(__i386__))
  uint32_t eax = 0;
  uint32_t edx = 0;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<uint64_t>(edx) << 32u) | eax;
#else
  return 0u;
#endif
}

inline bool os_supports_avx_state() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
  constexpr uint32_t XSAVE_BIT = 1u << 26u;
  constexpr uint32_t OSXSAVE_BIT = 1u << 27u;
  constexpr uint64_t XMM_YMM_STATE_BITS = 0x6u;
  const cpuid_registers leaf1 = read_cpuid(1u, 0u);
  if ((leaf1.ecx & XSAVE_BIT) == 0u || (leaf1.ecx & OSXSAVE_BIT) == 0u) {
    return false;
  }
  return (read_xcr0() & XMM_YMM_STATE_BITS) == XMM_YMM_STATE_BITS;
#else
  return false;
#endif
}

inline bool detect_avx2() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
  constexpr uint32_t AVX_BIT = 1u << 28u;
  constexpr uint32_t AVX2_BIT = 1u << 5u;
  const cpuid_registers max_leaf = read_cpuid(0u, 0u);
  if (max_leaf.eax < 7u || !os_supports_avx_state()) {
    return false;
  }
  const cpuid_registers leaf1 = read_cpuid(1u, 0u);
  const cpuid_registers leaf7 = read_cpuid(7u, 0u);
  return (leaf1.ecx & AVX_BIT) != 0u && (leaf7.ebx & AVX2_BIT) != 0u;
#else
  return false;
#endif
}

inline bool detect_fma() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
  constexpr uint32_t AVX_BIT = 1u << 28u;
  constexpr uint32_t FMA_BIT = 1u << 12u;
  if (!os_supports_avx_state()) {
    return false;
  }
  const cpuid_registers leaf1 = read_cpuid(1u, 0u);
  return (leaf1.ecx & AVX_BIT) != 0u && (leaf1.ecx & FMA_BIT) != 0u;
#else
  return false;
#endif
}

inline bool detect_f16c() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
  constexpr uint32_t AVX_BIT = 1u << 28u;
  constexpr uint32_t F16C_BIT = 1u << 29u;
  if (!os_supports_avx_state()) {
    return false;
  }
  const cpuid_registers leaf1 = read_cpuid(1u, 0u);
  return (leaf1.ecx & AVX_BIT) != 0u && (leaf1.ecx & F16C_BIT) != 0u;
#else
  return false;
#endif
}

struct host_feature_contract {
  bool avx2_available = false;
  bool fma_available = false;
  bool f16c_available = false;
  bool avx512_claimed = false;
  bool avx_vnni_claimed = false;
  bool amx_claimed = false;
  bool bf16_claimed = false;
  bool native_fp16_claimed = false;

  bool avx2_fma_f16c_available() const noexcept {
    return avx2_available && fma_available && f16c_available;
  }
};

inline host_feature_contract detect_host_feature_contract() noexcept {
  return host_feature_contract{
      .avx2_available = detect_avx2(),
      .fma_available = detect_fma(),
      .f16c_available = detect_f16c(),
      .avx512_claimed = false,
      .avx_vnni_claimed = false,
      .amx_claimed = false,
      .bf16_claimed = false,
      .native_fp16_claimed = false,
  };
}

} // namespace emel::kernel::x86_64::detail

namespace emel::kernel::x86_64::action {

namespace detail {

inline bool detect_avx2() noexcept {
  return ::emel::kernel::x86_64::detail::detect_avx2();
}

inline bool detect_fma() noexcept {
  return ::emel::kernel::x86_64::detail::detect_fma();
}

inline bool detect_f16c() noexcept {
  return ::emel::kernel::x86_64::detail::detect_f16c();
}

} // namespace detail

struct context {
  using host_feature_contract =::emel::kernel::x86_64::detail::host_feature_contract;

  context() noexcept
      : context(::emel::kernel::x86_64::detail::detect_host_feature_contract(),
                {}, 0) {}

  context(const bool avx2,
          const ::emel::kernel::detail::flash_attn_workspace &workspace,
          const uint64_t generation) noexcept
      : context(
            host_feature_contract{
                .avx2_available = avx2,
                .fma_available = detail::detect_fma(),
                .f16c_available = detail::detect_f16c(),
                .avx512_claimed = false,
                .avx_vnni_claimed = false,
                .amx_claimed = false,
                .bf16_claimed = false,
                .native_fp16_claimed = false,
            },
            workspace, generation) {}

  context(const host_feature_contract &contract,
          const ::emel::kernel::detail::flash_attn_workspace &workspace,
          const uint64_t generation) noexcept
      : host_features(contract), avx2_available(contract.avx2_available),
        fma_available(contract.fma_available),
        f16c_available(contract.f16c_available),
        avx512_claimed(contract.avx512_claimed),
        avx_vnni_claimed(contract.avx_vnni_claimed),
        amx_claimed(contract.amx_claimed), bf16_claimed(contract.bf16_claimed),
        native_fp16_claimed(contract.native_fp16_claimed),
        flash_attn_workspace(workspace), dispatch_generation(generation) {}

  const host_feature_contract host_features;
  const bool avx2_available;
  const bool fma_available;
  const bool f16c_available;
  const bool avx512_claimed;
  const bool avx_vnni_claimed;
  const bool amx_claimed;
  const bool bf16_claimed;
  const bool native_fp16_claimed;
  ::emel::kernel::detail::flash_attn_workspace flash_attn_workspace;
  uint64_t optimized_flash_dispatch_count = 0;
  uint64_t shared_flash_dispatch_count = 0;
  uint64_t optimized_q2_dispatch_count = 0;
  uint64_t shared_q2_dispatch_count = 0;
  uint64_t optimized_q3_dispatch_count = 0;
  uint64_t shared_q3_dispatch_count = 0;
  uint64_t optimized_f32_fma_dispatch_count = 0;
  uint64_t optimized_f32_fma_vector_dispatch_count = 0;
  uint64_t optimized_q4_dispatch_count = 0;
  uint64_t shared_q4_dispatch_count = 0;
  uint64_t optimized_q6_dispatch_count = 0;
  uint64_t shared_q6_dispatch_count = 0;
  uint64_t optimized_q4_0_dispatch_count = 0;
  uint64_t shared_q4_0_dispatch_count = 0;
  uint64_t optimized_q4_1_dispatch_count = 0;
  uint64_t shared_q4_1_dispatch_count = 0;
  uint64_t optimized_q5_0_dispatch_count = 0;
  uint64_t shared_q5_0_dispatch_count = 0;
  uint64_t optimized_q8_0_dispatch_count = 0;
  uint64_t shared_q8_0_dispatch_count = 0;
  // TODO(emel): remove once dispatch observability no longer relies on this
  // counter.
  uint64_t dispatch_generation = 0;
};

} // namespace emel::kernel::x86_64::action
