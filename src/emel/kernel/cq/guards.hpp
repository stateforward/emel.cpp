#pragma once

#include <limits>

#include "emel/kernel/cq/actions.hpp"

namespace emel::kernel::cq::guard {

inline bool a8_supported(const event::quantize_a8_request &request) noexcept {
  if (request.input.empty() ||
      request.quantized.size() < request.input.size() ||
      request.integer_values.size() < request.input.size())
    return false;
  for (const float value : request.input)
    if (!std::isfinite(value))
      return false;
  return true;
}

struct guard_quantize_a8 {
  bool operator()(const event::quantize_a8 &ev,
                  const action::context &) const noexcept {
    return a8_supported(ev.request);
  }
};

struct guard_execute_fwht_avx2 {
  bool operator()(const event::execute_fwht_avx2 &ev,
                  const action::context &) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    return ev.request.values.size() == 128u;
#else
    (void)ev;
    return false;
#endif
  }
};

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
      view.group > detail::k_max_group || !detail::is_power_of_two(view.group))
    return false;
  const uint64_t in_pad =
      (static_cast<uint64_t>(view.shape[1]) + view.group - 1u) / view.group *
      view.group;
  const uint64_t index_count = static_cast<uint64_t>(view.shape[0]) * in_pad;
  const uint64_t norm_count = index_count / view.group;
  const uint64_t blocked_norm_count =
      static_cast<uint64_t>(view.shape[0] / 32u * 32u) *
      (in_pad / view.group);
  const uint64_t packed_bytes = index_count / 2u;
  return packed_bytes + norm_count * 2u <= view.nbytes &&
         request.indices.size() >= index_count &&
         request.indices_by_input32.size() >= index_count &&
         request.norms.size() >= norm_count &&
         request.norms_by_group32.size() >= blocked_norm_count;
}
inline bool
prepared_codebook_supported(const event::prepared_codebook_q4 &codebook) noexcept {
  return codebook.values.size() >= emel::cact::loader::k_codebook_len;
}

struct guard_prepare_codebook_q4 {
  bool operator()(const event::prepare_codebook_q4 &ev,
                  const action::context &) const noexcept {
    return ev.request.codebook.size() >= emel::cact::loader::k_codebook_len;
  }
};
inline bool prepared_supported(const event::prepared_q4_view &view,
                               const std::span<const float> codebook) noexcept {
  if (view.source == nullptr || view.out == 0u || view.in == 0u ||
      view.group == 0u || view.group > detail::k_max_group ||
      !detail::is_power_of_two(view.group) || view.in_pad < view.in ||
      view.in_pad % view.group != 0u)
    return false;
  const uint64_t index_count = static_cast<uint64_t>(view.out) * view.in_pad;
  const uint64_t norm_count = index_count / view.group;
  const uint64_t blocked_count =
      static_cast<uint64_t>(view.out / 32u * 32u) * view.in_pad;
  const uint64_t blocked_norm_count =
      static_cast<uint64_t>(view.out / 32u * 32u) *
      (view.in_pad / view.group);
  return view.indices.size() >= index_count &&
         view.indices_by_input32.size() >= blocked_count &&
         view.norms.size() >= norm_count &&
         (blocked_norm_count == 0u ||
          view.norms_by_group32.size() >= blocked_norm_count) &&
         codebook.size() >= 28u;
}

struct guard_prepare_q4 {
  bool operator()(const event::prepare_q4 &ev,
                  const action::context &) const noexcept {
    return prepare_supported(ev.request);
  }
};

struct guard_execute_prepared_avx2_q4 {
  bool operator()(const event::execute_prepared_avx2_q4 &ev,
                  const action::context &) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    const auto &request = ev.request;
    return prepared_supported(request.weights, request.codebook.values) &&
           prepared_codebook_supported(request.codebook) &&
           request.weights.group == 128u &&
           request.activation.size() >= request.weights.in &&
           request.output.size() >= request.weights.out &&
           request.workspace.size() >= request.weights.in_pad;
#else
    (void)ev;
    return false;
#endif
  }
};

inline bool ranges_disjoint(const float *const a, const size_t a_count,
                            const float *const b,
                            const size_t b_count) noexcept {
  constexpr uintptr_t max_address = std::numeric_limits<uintptr_t>::max();
  if (a_count > max_address / sizeof(float) ||
      b_count > max_address / sizeof(float))
    return false;
  const auto a_begin = reinterpret_cast<uintptr_t>(a);
  const auto b_begin = reinterpret_cast<uintptr_t>(b);
  const size_t a_bytes = a_count * sizeof(float);
  const size_t b_bytes = b_count * sizeof(float);
  if (a_begin > max_address - a_bytes || b_begin > max_address - b_bytes)
    return false;
  return a_begin + a_bytes <= b_begin || b_begin + b_bytes <= a_begin;
}

inline bool output_ranges_disjoint(const event::prepared_gemv_target &a,
                                   const event::prepared_gemv_target &b) noexcept {
  return ranges_disjoint(a.output.data(), a.weights->out, b.output.data(),
                         b.weights->out);
}

inline bool prepared_pair_matches(const event::prepared_gemv_target &a,
                                  const event::prepared_gemv_target &b,
                                  const std::span<const float> codebook) noexcept {
  return a.weights != nullptr && b.weights != nullptr &&
         prepared_supported(*a.weights, codebook) &&
         prepared_supported(*b.weights, codebook) &&
         a.weights->out == b.weights->out && a.weights->in == b.weights->in &&
         a.weights->group == b.weights->group &&
         a.weights->in_pad == b.weights->in_pad &&
         a.weights->out % 32u == 0u && a.output.size() >= a.weights->out &&
         b.output.size() >= b.weights->out;
}

struct guard_execute_prepared_avx2_batch4_pair_q4 {
  bool operator()(const event::execute_prepared_avx2_batch4_pair_q4 &ev,
                  const action::context &) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    const auto &request = ev.request;
    const auto *first = request.q.weights;
    return first != nullptr && first->group == 128u &&
           prepared_codebook_supported(request.codebook) &&
           request.activation.size() >= first->in &&
           request.workspace.size() >= first->in_pad &&
           prepared_pair_matches(request.q, request.gate,
                                 request.codebook.values) &&
           output_ranges_disjoint(request.q, request.gate) &&
           prepared_pair_matches(request.k, request.v,
                                 request.codebook.values) &&
           output_ranges_disjoint(request.k, request.v) &&
           output_ranges_disjoint(request.q, request.k) &&
           output_ranges_disjoint(request.q, request.v) &&
           output_ranges_disjoint(request.gate, request.k) &&
           output_ranges_disjoint(request.gate, request.v) &&
           ranges_disjoint(request.workspace.data(), first->in_pad,
                           request.q.output.data(), request.q.weights->out) &&
           ranges_disjoint(request.workspace.data(), first->in_pad,
                           request.gate.output.data(),
                           request.gate.weights->out) &&
           ranges_disjoint(request.workspace.data(), first->in_pad,
                           request.k.output.data(), request.k.weights->out) &&
           ranges_disjoint(request.workspace.data(), first->in_pad,
                           request.v.output.data(), request.v.weights->out) &&
           request.k.weights->in == first->in &&
           request.k.weights->group == first->group &&
           request.k.weights->in_pad == first->in_pad;
#else
    (void)ev;
    return false;
#endif
  }
};

struct guard_execute_prepared_avx2_batch4_q4 {
  bool operator()(const event::execute_prepared_avx2_batch4_q4 &ev,
                  const action::context &) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    const auto &request = ev.request;
    const auto *first = request.targets[0].weights;
    if (first == nullptr || first->group != 128u ||
        request.activation.size() < first->in ||
        request.workspace.size() < first->in_pad)
      return false;
    for (const auto &target : request.targets)
      if (target.weights == nullptr ||
          !prepared_supported(*target.weights, request.codebook.values) ||
          !prepared_codebook_supported(request.codebook) ||
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

struct guard_execute_prepared_avx2_rows_q4 {
  bool operator()(const event::execute_prepared_avx2_rows_q4 &ev,
                  const action::context &) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    const auto &request = ev.request;
    return prepared_supported(request.weights, request.codebook.values) &&
           prepared_codebook_supported(request.codebook) &&
           request.weights.group == 128u && request.row_count > 0u &&
           static_cast<uint64_t>(request.row_begin) + request.row_count <=
               request.weights.out &&
           request.activation.size() >= request.weights.in &&
           request.output.size() >= request.row_count &&
           request.workspace.size() >= request.weights.in_pad;
#else
    (void)ev;
    return false;
#endif
  }
};

struct guard_execute_prepared_dequant_q4 {
  bool operator()(const event::execute_prepared_dequant_q4 &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    return prepared_supported(request.weights, request.codebook.values) &&
           prepared_codebook_supported(request.codebook) &&
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
