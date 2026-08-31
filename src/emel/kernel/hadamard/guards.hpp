#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "emel/kernel/hadamard/actions.hpp"
#include "emel/kernel/attention/guards.hpp"
#include "emel/kernel/x86_64/context.hpp"

namespace emel::kernel::hadamard::guard {

inline bool power_of_two(const uint32_t n) noexcept {
  return n != 0u && (n & (n - 1u)) == 0u;
}

inline bool spans_valid(const event::mlp_row_request &request) noexcept {
  const uint64_t d_bytes64 = static_cast<uint64_t>(request.hada_n) * 2u;
  if (request.d_model == 0u || !power_of_two(request.hada_n) ||
      request.hada_n < request.d_model ||
      request.hada_n > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      request.d_model > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      d_bytes64 > std::numeric_limits<std::size_t>::max())
    return false;
  const auto d_bytes = static_cast<std::size_t>(d_bytes64);
  const auto model_bytes = static_cast<std::size_t>(request.d_model) * sizeof(float);
  const auto workspace_bytes =
      static_cast<std::size_t>(request.hada_n) * sizeof(float);
  if (request.input.data() == nullptr || request.skip.data() == nullptr ||
      request.d1.data() == nullptr || request.d2.data() == nullptr ||
      request.d3.data() == nullptr || request.workspace.data() == nullptr ||
      request.output.data() == nullptr ||
      request.input.size() < request.d_model ||
      request.skip.size() < request.d_model ||
      request.output.size() < request.d_model ||
      request.workspace.size() < request.hada_n ||
      request.d1.size() < d_bytes || request.d2.size() < d_bytes ||
      request.d3.size() < d_bytes ||
      reinterpret_cast<std::uintptr_t>(request.input.data()) % alignof(float) !=
          0u ||
      reinterpret_cast<std::uintptr_t>(request.skip.data()) % alignof(float) !=
          0u ||
      reinterpret_cast<std::uintptr_t>(request.workspace.data()) %
              alignof(float) !=
          0u ||
      reinterpret_cast<std::uintptr_t>(request.output.data()) % alignof(float) !=
          0u)
    return false;
  const auto disjoint = [](const void *lhs, const std::size_t lhs_bytes,
                           const void *rhs, const std::size_t rhs_bytes) noexcept {
    return emel::kernel::attention::guard::guard_ranges_disjoint(
        lhs, lhs_bytes, rhs, rhs_bytes);
  };
  return disjoint(request.workspace.data(), workspace_bytes,
                  request.output.data(), model_bytes) &&
         disjoint(request.workspace.data(), workspace_bytes,
                  request.input.data(), model_bytes) &&
         disjoint(request.workspace.data(), workspace_bytes,
                  request.skip.data(), model_bytes) &&
         disjoint(request.workspace.data(), workspace_bytes, request.d1.data(),
                  d_bytes) &&
         disjoint(request.workspace.data(), workspace_bytes, request.d2.data(),
                  d_bytes) &&
         disjoint(request.workspace.data(), workspace_bytes, request.d3.data(),
                  d_bytes) &&
         disjoint(request.output.data(), model_bytes, request.input.data(),
                  model_bytes) &&
         disjoint(request.output.data(), model_bytes, request.skip.data(),
                  model_bytes) &&
         disjoint(request.output.data(), model_bytes, request.d1.data(),
                  d_bytes) &&
         disjoint(request.output.data(), model_bytes, request.d2.data(),
                  d_bytes) &&
         disjoint(request.output.data(), model_bytes, request.d3.data(),
                  d_bytes);
}

struct guard_execute_mlp_row {
  bool operator()(const event::execute_mlp_row &ev,
                  const action::context &) const noexcept {
    return spans_valid(ev.request);
  }
};

// Process-wide host capability is immutable after startup; cache the pure
// query so repeated dispatch guards do not execute CPUID/XGETBV in the hot path.
inline bool avx2_fma_f16c_available() noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) &&                               \
    ((defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)) ||           \
     defined(__GNUC__) || defined(__clang__))
  static const bool available =
      emel::kernel::x86_64::detail::detect_avx2() &&
      emel::kernel::x86_64::detail::detect_fma() &&
      emel::kernel::x86_64::detail::detect_f16c();
  return available;
#else
  return false;
#endif
}

struct guard_execute_mlp_row_avx2 {
  bool operator()(const event::execute_mlp_row_avx2 &ev,
                  const action::context &) const noexcept {
    return avx2_fma_f16c_available() && ev.request.d_model == 512u &&
           ev.request.hada_n == 512u && spans_valid(ev.request);
  }
};

} // namespace emel::kernel::hadamard::guard
