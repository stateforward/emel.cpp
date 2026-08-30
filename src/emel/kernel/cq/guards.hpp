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

inline bool
prepare_supported(const event::prepare_q4_request &request) noexcept {
  const auto &view = request.weights;
  if (view.data == nullptr || view.bits != 4u || view.shape[0] == 0u ||
      view.shape[1] == 0u || view.group == 0u ||
      view.group > detail::k_max_group ||
      !detail::is_power_of_two(view.group) || (view.group & 1u) != 0u)
    return false;
  const uint64_t in_pad =
      (static_cast<uint64_t>(view.shape[1]) + view.group - 1u) / view.group *
      view.group;
  const uint64_t packed_row = in_pad / 2u;
  const uint64_t packed_bytes =
      static_cast<uint64_t>(view.shape[0]) * packed_row;
  const uint64_t norm_count =
      static_cast<uint64_t>(view.shape[0]) * in_pad / view.group;
  const uint64_t blocked_bytes =
      static_cast<uint64_t>(view.shape[0] / 8u * 8u) * packed_row;
  return packed_bytes + norm_count * 2u <= view.nbytes &&
         request.packed_by_pair8.size() >= blocked_bytes;
}
inline bool prepared_supported(const event::prepared_q4_view &view,
                               const std::span<const float> codebook) noexcept {
  const uint64_t packed_row = view.in_pad / 2u;
  const uint64_t blocked_bytes =
      static_cast<uint64_t>(view.out / 8u * 8u) * packed_row;
  return view.source != nullptr && view.out > 0u && view.in > 0u &&
         view.group > 0u && view.group <= detail::k_max_group &&
         detail::is_power_of_two(view.group) && (view.group & 1u) == 0u &&
         view.in_pad >= view.in && view.in_pad % view.group == 0u &&
         view.packed_by_pair8.size() >= blocked_bytes && codebook.size() >= 28u;
}

struct guard_prepare_q4 {
  bool operator()(const event::prepare_q4 &ev,
                  const action::context &) const noexcept {
    return prepare_supported(ev.request);
  }
};

struct guard_execute_prepared_pair_lut_q4 {
  bool operator()(const event::execute_prepared_pair_lut_q4 &ev,
                  const action::context &) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    const auto &request = ev.request;
    return prepared_supported(request.weights, request.codebook) &&
           request.activation.size() >= request.weights.in &&
           request.output.size() >= request.weights.out &&
           request.workspace.size() >= request.weights.in_pad &&
           request.pair_lut.size() >= 256u &&
           request.pair_scratch.size() >=
               static_cast<uint64_t>(request.weights.out) *
                   (request.weights.in_pad / 2u);
#else
    (void)ev;
    return false;
#endif
  }
};

struct guard_execute_prepared_pair_lut_batch4_q4 {
  bool operator()(const event::execute_prepared_pair_lut_batch4_q4 &ev,
                  const action::context &) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    const auto &request = ev.request;
    const auto *first = request.targets[0].weights;
    uint64_t scratch_count = 0u;
    for (const auto &target : request.targets) {
      if (target.weights != nullptr)
        scratch_count += static_cast<uint64_t>(target.weights->out) *
                         (first != nullptr ? first->in_pad / 2u : 0u);
    }
    if (first == nullptr || request.activation.size() < first->in ||
        request.workspace.size() < first->in_pad ||
        request.pair_lut.size() < 256u ||
        request.pair_scratch.size() < scratch_count)
      return false;
    for (const auto &target : request.targets)
      if (target.weights == nullptr ||
          !prepared_supported(*target.weights, request.codebook) ||
          target.weights->in != first->in ||
          target.weights->group != first->group ||
          target.weights->in_pad != first->in_pad ||
          target.output.size() < target.weights->out)
        return false;
    return true;
#else
    (void)ev;
    return false;
#endif
  }
};

struct guard_execute_prepared_pair_lut_rows_q4 {
  bool operator()(const event::execute_prepared_pair_lut_rows_q4 &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    return prepared_supported(request.weights, request.codebook) &&
           request.row_count > 0u &&
           static_cast<uint64_t>(request.row_begin) + request.row_count <=
               request.weights.out &&
           request.activation.size() >= request.weights.in &&
           request.output.size() >= request.row_count &&
           request.workspace.size() >= request.weights.in_pad &&
           request.pair_lut.size() >= 256u &&
           request.pair_scratch.size() >=
               static_cast<uint64_t>(request.row_count) *
                   (request.weights.in_pad / 2u);
  }
};

struct guard_execute_prepared_dequant_q4 {
  bool operator()(const event::execute_prepared_dequant_q4 &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    return prepared_supported(request.weights, request.codebook) &&
           request.row_count > 0u &&
           static_cast<uint64_t>(request.row_begin) + request.row_count <=
               request.weights.out &&
           request.output.size() >=
               static_cast<uint64_t>(request.row_count) * request.weights.in;
  }
};

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
