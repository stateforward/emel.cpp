#pragma once

#include <cstdint>

namespace emel::kernel::aarch64::action {

namespace detail {

inline bool detect_neon() noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  return true;
#else
  return false;
#endif
}

}  // namespace detail

struct context {
  const bool neon_available = detail::detect_neon();
  // TODO(emel): remove once dispatch observability no longer relies on this counter.
  uint64_t dispatch_generation = 0;
};

}  // namespace emel::kernel::aarch64::action
