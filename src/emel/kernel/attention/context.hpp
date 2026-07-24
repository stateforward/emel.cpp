#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace emel::kernel::attention::action {

inline constexpr std::size_t k_max_head_dim = 8192u;
inline constexpr std::size_t k_max_context = 8192u;

// This persistent actor-owned workspace is constructed once and reused by
// same-RTC head-range dispatches. It is intentionally not dispatch-local state.
struct context {
  alignas(64) std::array<uint16_t, k_max_head_dim> q_bf16 = {};
  alignas(64) std::array<float, k_max_context> scores = {};
  alignas(64) std::array<uint16_t, k_max_context> weights_bf16 = {};
  alignas(64) std::array<double, k_max_head_dim> output_accumulators = {};
};

} // namespace emel::kernel::attention::action
