#pragma once

#include <cstdint>

namespace emel::kernel::x86_64::action {

namespace detail {

inline bool detect_avx2() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2");
#else
  return false;
#endif
#else
  return false;
#endif
}

}  // namespace detail

struct context {
  const bool avx2_available = detail::detect_avx2();
  // TODO(emel): remove once dispatch observability no longer relies on this counter.
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::kernel::x86_64::action
