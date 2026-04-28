#pragma once

#include <cstdint>

#include "emel/kernel/detail.hpp"

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
  ::emel::kernel::detail::flash_attn_workspace flash_attn_workspace = {};
  // TODO(emel): remove once dispatch observability no longer relies on this counter.
  uint64_t dispatch_generation = 0;
  uint64_t optimized_q5_0_dispatch_count = 0;
  uint64_t optimized_q5_0_vector_dispatch_count = 0;
  uint64_t optimized_q8_0_dispatch_count = 0;
  uint64_t optimized_q8_0_vector_dispatch_count = 0;
  uint64_t optimized_q8_0_packed_dispatch_count = 0;
  uint64_t optimized_q8_0_packed_bl4_dispatch_count = 0;
  uint64_t optimized_q8_0_packed_bl8_dispatch_count = 0;
  uint64_t optimized_q8_0_packed_bl8_full_groups_dispatch_count = 0;
  uint64_t optimized_q8_0_packed_bl8_matrix_x4_dispatch_count = 0;
  uint64_t shared_q5_0_dispatch_count = 0;
  uint64_t shared_q8_0_dispatch_count = 0;
  uint64_t optimized_q2_dispatch_count = 0;
  uint64_t shared_q2_dispatch_count = 0;
  uint64_t optimized_q3_dispatch_count = 0;
  uint64_t shared_q3_dispatch_count = 0;
  uint64_t optimized_q4_dispatch_count = 0;
  uint64_t optimized_q4_vector_dispatch_count = 0;
  uint64_t optimized_q4_vector_packed_dispatch_count = 0;
  uint64_t optimized_q4_vector_packed_q8_rhs_dispatch_count = 0;
  uint64_t optimized_q4_0_dispatch_count = 0;
  uint64_t optimized_q4_0_vector_dispatch_count = 0;
  uint64_t optimized_q4_1_dispatch_count = 0;
  uint64_t optimized_q4_1_vector_dispatch_count = 0;
  uint64_t shared_q4_dispatch_count = 0;
  uint64_t optimized_q6_dispatch_count = 0;
  uint64_t optimized_q6_vector_dispatch_count = 0;
  uint64_t optimized_q6_vector_argmax_dispatch_count = 0;
  uint64_t optimized_q6_vector_packed_dispatch_count = 0;
  uint64_t optimized_q6_vector_packed_q8_rhs_dispatch_count = 0;
  uint64_t optimized_q6_vector_packed_q8_rhs_argmax_dispatch_count = 0;
  uint64_t optimized_q6_vector_prepared_q8_rhs_dispatch_count = 0;
  uint64_t optimized_q6_vector_prepared_q8_rhs_i8mm_dispatch_count = 0;
  uint64_t optimized_q6_vector_prepared_q8_rhs_argmax_i8mm_dispatch_count = 0;
  uint64_t optimized_q6_vector_q8_argmax_prepared_i8mm_dispatch_count = 0;
  uint64_t shared_q6_dispatch_count = 0;
  uint64_t optimized_flash_dispatch_count = 0;
  uint64_t shared_flash_dispatch_count = 0;
};

}  // namespace emel::kernel::aarch64::action
