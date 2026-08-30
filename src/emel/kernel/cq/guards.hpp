#pragma once

#include "emel/kernel/cq/actions.hpp"

namespace emel::kernel::cq::guard {

template <uint32_t Bits>
inline bool supported(const event::gemv_request &request) noexcept {
  const uint32_t in_pad =
      (request.weights.shape[1] + request.weights.group - 1u) /
      request.weights.group * request.weights.group;
  return detail::valid_view<Bits>(request.weights, request.codebook,
                                  request.activation, request.output) &&
         request.workspace.size() >= in_pad;
}

template <uint32_t Bits>
inline bool avx2_supported(const event::gemv_request &request) noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__) && defined(__FMA__)
  return supported<Bits>(request);
#else
  (void)request;
  return false;
#endif
#else
  (void)request;
  return false;
#endif
}

template <uint32_t Bits>
inline bool rows_supported(const event::gemv_rows_request &request) noexcept {
  const uint32_t in_pad =
      (request.weights.shape[1] + request.weights.group - 1u) /
      request.weights.group * request.weights.group;
  return detail::valid_packed_view<Bits>(request.weights, request.codebook) &&
         request.row_count > 0u &&
         static_cast<uint64_t>(request.row_begin) + request.row_count <=
             request.weights.shape[0] &&
         request.activation.size() >= request.weights.shape[1] &&
         request.output.size() >= request.row_count &&
         request.workspace.size() >= in_pad;
}

template <uint32_t Bits>
inline bool
dequant_rows_supported(const event::dequant_rows_request &request) noexcept {
  return detail::valid_packed_view<Bits>(request.weights, request.codebook) &&
         request.row_count > 0u &&
         static_cast<uint64_t>(request.row_begin) + request.row_count <=
             request.weights.shape[0] &&
         request.output.size() >= static_cast<uint64_t>(request.row_count) *
                                      request.weights.shape[1];
}

template <uint32_t Bits> struct guard_execute_scalar {
  bool operator()(const event::execute_scalar<Bits> &ev,
                  const action::context &) const noexcept {
    return supported<Bits>(ev.request);
  }
};

template <uint32_t Bits> struct guard_execute_avx2 {
  bool operator()(const event::execute_avx2<Bits> &ev,
                  const action::context &) const noexcept {
    return avx2_supported<Bits>(ev.request);
  }
};

template <uint32_t Bits> struct guard_execute_scalar_rows {
  bool operator()(const event::execute_scalar_rows<Bits> &ev,
                  const action::context &) const noexcept {
    return rows_supported<Bits>(ev.request);
  }
};

template <uint32_t Bits> struct guard_execute_scalar_dequant {
  bool operator()(const event::execute_scalar_dequant<Bits> &ev,
                  const action::context &) const noexcept {
    return dequant_rows_supported<Bits>(ev.request);
  }
};

} // namespace emel::kernel::cq::guard
