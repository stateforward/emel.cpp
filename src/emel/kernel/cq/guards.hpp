#pragma once

#include <limits>
#include <cstddef>
#include <cstdint>

#include "emel/kernel/cq/actions.hpp"
#include "emel/kernel/x86_64/context.hpp"

namespace emel::kernel::cq::guard {
inline bool checked_bytes(const uint64_t count, const uint64_t element_bytes,
                          size_t &bytes) noexcept {
  uint64_t product = 0u;
  if (!detail::checked_multiply_u64(count, element_bytes, product) ||
      product > std::numeric_limits<size_t>::max())
    return false;
  bytes = static_cast<size_t>(product);
  return true;
}

inline bool ranges_disjoint(const void *first, const size_t first_bytes,
                            const void *second,
                            const size_t second_bytes) noexcept {
  if ((first_bytes != 0u && first == nullptr) ||
      (second_bytes != 0u && second == nullptr))
    return false;
  const auto first_begin = reinterpret_cast<uintptr_t>(first);
  const auto second_begin = reinterpret_cast<uintptr_t>(second);
  const auto max_address = std::numeric_limits<uintptr_t>::max();
  if (first_begin > max_address - first_bytes ||
      second_begin > max_address - second_bytes)
    return false;
  return first_begin + first_bytes <= second_begin ||
         second_begin + second_bytes <= first_begin;
}
template <typename T>
inline bool span_has_data(const std::span<T> values) noexcept {
  return values.empty() || values.data() != nullptr;
}


inline bool finite_values(const std::span<const float> values,
                          const size_t count) noexcept {
  if (count > values.size() || !span_has_data(values))
    return false;
  for (size_t i = 0u; i < count; ++i)
    if (!std::isfinite(values[i]))
      return false;
  return true;
}

inline bool selectors_supported(const std::span<const uint8_t> selectors,
                                const size_t count) noexcept {
  if (count > selectors.size() || !span_has_data(selectors))
    return false;
  for (size_t i = 0u; i < count; ++i)
    if (selectors[i] >= 16u)
      return false;
  return true;
}

template <uint32_t Bits>
inline bool generic_ranges_supported(const event::gemv_request &request,
                                     const detail::layout &layout) noexcept {
  size_t weight_bytes = 0u;
  size_t activation_bytes = 0u;
  size_t output_bytes = 0u;
  size_t workspace_bytes = 0u;
  size_t codebook_bytes = 0u;
  if (!span_has_data(request.codebook) ||
      !span_has_data(request.activation) || !span_has_data(request.output) ||
      !span_has_data(request.workspace) ||
      layout.total_bytes > std::numeric_limits<size_t>::max() ||
      !checked_bytes(request.weights.shape[1], sizeof(float),
                     activation_bytes) ||
      !checked_bytes(request.weights.shape[0], sizeof(float), output_bytes) ||
      !checked_bytes(layout.in_pad, sizeof(float), workspace_bytes) ||
      !checked_bytes(Bits == detail::k_ternary_record_bits ? 0u : 28u,
                     sizeof(float), codebook_bytes))
    return false;
  weight_bytes = static_cast<size_t>(layout.total_bytes);
  const void *weights = request.weights.data;
  const void *codebook = request.codebook.data();
  const void *activation = request.activation.data();
  void *output = request.output.data();
  void *workspace = request.workspace.data();
  return ranges_disjoint(output, output_bytes, weights, weight_bytes) &&
         ranges_disjoint(output, output_bytes, codebook, codebook_bytes) &&
         ranges_disjoint(output, output_bytes, activation, activation_bytes) &&
         ranges_disjoint(output, output_bytes, workspace, workspace_bytes) &&
         ranges_disjoint(workspace, workspace_bytes, weights, weight_bytes) &&
         ranges_disjoint(workspace, workspace_bytes, codebook,
                         codebook_bytes) &&
         ranges_disjoint(workspace, workspace_bytes, activation,
                         activation_bytes);
}

inline bool prepared_read_ranges_disjoint(
    const event::prepared_q4_view &view,
    const event::prepared_codebook_q4 &codebook, const void *write,
    const size_t write_bytes) noexcept {
  size_t indices_bytes = 0u;
  size_t blocked_indices_bytes = 0u;
  size_t norms_bytes = 0u;
  size_t blocked_norms_bytes = 0u;
  size_t codebook_bytes = 0u;
  if (!checked_bytes(view.indices().size(), sizeof(uint8_t), indices_bytes) ||
      !checked_bytes(view.indices_by_input32().size(), sizeof(uint8_t),
                     blocked_indices_bytes) ||
      !checked_bytes(view.norms().size(), sizeof(float), norms_bytes) ||
      !checked_bytes(view.norms_by_group32().size(), sizeof(float),
                     blocked_norms_bytes) ||
      !checked_bytes(codebook.values().size(), sizeof(float), codebook_bytes))
    return false;
  return ranges_disjoint(write, write_bytes, view.indices().data(),
                         indices_bytes) &&
         ranges_disjoint(write, write_bytes, view.indices_by_input32().data(),
                         blocked_indices_bytes) &&
         ranges_disjoint(write, write_bytes, view.norms().data(), norms_bytes) &&
         ranges_disjoint(write, write_bytes, view.norms_by_group32().data(),
                         blocked_norms_bytes) &&
         ranges_disjoint(write, write_bytes, codebook.values().data(),
                         codebook_bytes) &&
         ranges_disjoint(write, write_bytes, codebook.byte_planes().data(),
                         sizeof(event::prepared_codebook_q4::byte_planes_type));
}

inline bool a8_supported(const event::quantize_a8_request &request) noexcept {
  if (request.input.empty() || !span_has_data(request.input) ||
      !span_has_data(request.quantized) ||
      !span_has_data(request.integer_values) ||
      request.quantized.size() < request.input.size() ||
      request.integer_values.size() < request.input.size())
    return false;
  size_t input_bytes = 0u;
  size_t quantized_bytes = 0u;
  if (!checked_bytes(request.input.size(), sizeof(float), input_bytes) ||
      !checked_bytes(request.input.size(), sizeof(int8_t), quantized_bytes) ||
      !finite_values(request.input, request.input.size()))
    return false;
  return ranges_disjoint(request.quantized.data(), quantized_bytes,
                         request.input.data(), input_bytes) &&
         ranges_disjoint(request.integer_values.data(), input_bytes,
                         request.input.data(), input_bytes) &&
         ranges_disjoint(request.quantized.data(), quantized_bytes,
                         request.integer_values.data(), input_bytes);
}

struct guard_quantize_a8 {
  bool operator()(const event::quantize_a8 &ev,
                  const action::context &) const noexcept {
    return a8_supported(ev.request);
  }
};

struct guard_execute_fwht_avx2 {
  bool operator()(const event::execute_fwht_avx2 &ev,
                  const action::context &ctx) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    return ctx.avx2_fma_available && ev.request.values.size() == 128u &&
           span_has_data(ev.request.values);
#else
    (void)ctx;
    (void)ev;
    return false;
#endif
  }
};

template <uint32_t Bits>
inline bool supported(const event::gemv_request &request) noexcept {
  detail::layout layout{};
  if (!detail::valid_view<Bits>(request.weights, request.codebook,
                                request.activation, request.output) ||
      !detail::checked_layout<Bits>(request.weights.shape[0],
                                    request.weights.shape[1],
                                    request.weights.group, layout))
    return false;
  return request.workspace.size() >= layout.in_pad &&
         std::isfinite(request.output_scale) &&
         finite_values(request.codebook,
                       Bits == detail::k_ternary_record_bits ? 0u : 28u) &&
         generic_ranges_supported<Bits>(request, layout);
}

template <uint32_t Bits>
inline bool avx2_supported(const event::gemv_request &request,
                           const action::context &ctx) noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
  return ctx.avx2_fma_available && supported<Bits>(request);
#else
  (void)request;
  (void)ctx;
  return false;
#endif
}

template <uint32_t Bits>
inline bool rows_supported(const event::gemv_rows_request &request) noexcept {
  detail::layout layout{};
  size_t activation_bytes = 0u;
  size_t output_bytes = 0u;
  size_t workspace_bytes = 0u;
  size_t codebook_bytes = 0u;
  if (!span_has_data(request.codebook) ||
      !span_has_data(request.activation) || !span_has_data(request.output) ||
      !span_has_data(request.workspace) ||
      !detail::valid_packed_view<Bits>(request.weights, request.codebook) ||
      !detail::checked_layout<Bits>(request.weights.shape[0],
                                    request.weights.shape[1],
                                    request.weights.group, layout) ||
      request.row_count == 0u ||
      static_cast<uint64_t>(request.row_begin) + request.row_count >
          request.weights.shape[0] ||
      request.activation.size() < request.weights.shape[1] ||
      request.output.size() < request.row_count ||
      request.workspace.size() < layout.in_pad ||
      !std::isfinite(request.output_scale) ||
      !finite_values(request.codebook,
                     Bits == detail::k_ternary_record_bits ? 0u : 28u) ||
      layout.total_bytes > std::numeric_limits<size_t>::max() ||
      !checked_bytes(request.weights.shape[1], sizeof(float),
                     activation_bytes) ||
      !checked_bytes(request.row_count, sizeof(float), output_bytes) ||
      !checked_bytes(layout.in_pad, sizeof(float), workspace_bytes) ||
      !checked_bytes(Bits == detail::k_ternary_record_bits ? 0u : 28u,
                     sizeof(float), codebook_bytes))
    return false;
  const size_t weight_bytes = static_cast<size_t>(layout.total_bytes);
  return ranges_disjoint(request.output.data(), output_bytes,
                         request.weights.data, weight_bytes) &&
         ranges_disjoint(request.output.data(), output_bytes,
                         request.codebook.data(), codebook_bytes) &&
         ranges_disjoint(request.output.data(), output_bytes,
                         request.activation.data(), activation_bytes) &&
         ranges_disjoint(request.output.data(), output_bytes,
                         request.workspace.data(), workspace_bytes) &&
         ranges_disjoint(request.workspace.data(), workspace_bytes,
                         request.weights.data, weight_bytes) &&
         ranges_disjoint(request.workspace.data(), workspace_bytes,
                         request.codebook.data(), codebook_bytes) &&
         ranges_disjoint(request.workspace.data(), workspace_bytes,
                         request.activation.data(), activation_bytes);
}

template <uint32_t Bits>
inline bool
dequant_rows_supported(const event::dequant_rows_request &request) noexcept {
  detail::layout layout{};
  size_t output_bytes = 0u;
  size_t codebook_bytes = 0u;
  uint64_t output_count = 0u;
  if (!span_has_data(request.codebook) || !span_has_data(request.output) ||
      !detail::valid_packed_view<Bits>(request.weights, request.codebook) ||
      !detail::checked_layout<Bits>(request.weights.shape[0],
                                    request.weights.shape[1],
                                    request.weights.group, layout) ||
      request.row_count == 0u ||
      static_cast<uint64_t>(request.row_begin) + request.row_count >
          request.weights.shape[0] ||
      !detail::checked_multiply_u64(request.row_count,
                                    request.weights.shape[1], output_count) ||
      request.output.size() < output_count || !std::isfinite(request.scale) ||
      !finite_values(request.codebook,
                     Bits == detail::k_ternary_record_bits ? 0u : 28u) ||
      layout.total_bytes > std::numeric_limits<size_t>::max() ||
      !checked_bytes(output_count, sizeof(float), output_bytes) ||
      !checked_bytes(Bits == detail::k_ternary_record_bits ? 0u : 28u,
                     sizeof(float), codebook_bytes))
    return false;
  const size_t weight_bytes = static_cast<size_t>(layout.total_bytes);
  return ranges_disjoint(request.output.data(), output_bytes,
                         request.weights.data, weight_bytes) &&
         ranges_disjoint(request.output.data(), output_bytes,
                         request.codebook.data(), codebook_bytes);
}


inline bool
prepare_supported(const event::prepare_q4_request &request) noexcept {
  const auto &view = request.weights;
  detail::layout layout{};
  uint64_t index_count = 0u;
  uint64_t blocked_count = 0u;
  uint64_t blocked_norm_count = 0u;
  const uint64_t blocked_rows =
      static_cast<uint64_t>(view.shape[0] / 32u * 32u);
  if (view.data == nullptr || view.bits != 4u ||
      view.group > detail::k_max_group ||
      !detail::is_power_of_two(view.group) ||
      !detail::checked_layout<4u>(view.shape[0], view.shape[1], view.group,
                                  layout) ||
      !detail::checked_multiply_u64(view.shape[0], layout.in_pad,
                                    index_count) ||
      !detail::checked_multiply_u64(blocked_rows, layout.in_pad,
                                    blocked_count) ||
      !detail::checked_multiply_u64(blocked_rows,
                                    layout.in_pad / view.group,
                                    blocked_norm_count) ||
      layout.total_bytes > view.nbytes ||
      layout.total_bytes > std::numeric_limits<size_t>::max())
    return false;
  const uint64_t norm_count = index_count / view.group;
  const auto &prepared = request.prepared;
  if (!prepared.capacity_valid() ||
      prepared.out() != view.shape[0] || prepared.in() != view.shape[1] ||
      prepared.group() != view.group || prepared.in_pad() != layout.in_pad ||
      prepared.index_capacity() != index_count ||
      prepared.input32_capacity() != blocked_count ||
      prepared.norm_capacity() != norm_count ||
      prepared.group32_norm_capacity() != blocked_norm_count)
    return false;
  const auto *source_data = static_cast<const uint8_t *>(view.data);
  const uint8_t *source_norms =
      source_data + static_cast<size_t>(layout.packed_bytes);
  for (size_t i = 0u; i < static_cast<size_t>(norm_count); ++i)
    if (!std::isfinite(
            detail::fp16_to_fp32(detail::load_u16(source_norms + i * 2u))))
      return false;
  return true;
}
inline bool prepared_codebook_structure_supported(
    const event::prepared_codebook_q4 &codebook) noexcept {
  return codebook.published() && span_has_data(codebook.values()) &&
         codebook.values().size() >= emel::cact::loader::k_codebook_len;
}
struct guard_prepare_codebook_q4 {
  bool operator()(const event::prepare_codebook_q4 &ev,
                  const action::context &) const noexcept {
    return span_has_data(ev.request.codebook) &&
           ev.request.codebook.size() >= emel::cact::loader::k_codebook_len &&
           finite_values(ev.request.codebook,
                         emel::cact::loader::k_codebook_len);
  }
};
inline bool prepared_structure_supported(
    const event::prepared_q4_view &view) noexcept {
  detail::layout layout{};
  return view.published() && view.capacity_valid() &&
         view.group() <= detail::k_max_group &&
         detail::is_power_of_two(view.group()) &&
         detail::checked_layout<4u>(view.out(), view.in(), view.group(),
                                    layout) &&
         view.in_pad() == layout.in_pad;
}

struct guard_prepare_q4 {
  bool operator()(const event::prepare_q4 &ev,
                  const action::context &) const noexcept {
    return prepare_supported(ev.request);
  }
};

struct guard_execute_prepared_avx2_q4 {
  bool operator()(const event::execute_prepared_avx2_q4 &ev,
                  const action::context &ctx) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    const auto &request = ev.request;
    size_t activation_bytes = 0u;
    size_t output_bytes = 0u;
    size_t workspace_bytes = 0u;
    if (!ctx.avx2_fma_available || request.weights.group() != 128u ||
        !prepared_structure_supported(request.weights) ||
        !prepared_codebook_structure_supported(request.codebook) ||
        request.activation.size() < request.weights.in() ||
        request.output.size() < request.weights.out() ||
        request.workspace.size() < request.weights.in_pad() ||
        !std::isfinite(request.output_scale) ||
        !checked_bytes(request.weights.in(), sizeof(float), activation_bytes) ||
        !checked_bytes(request.weights.out(), sizeof(float), output_bytes) ||
        !checked_bytes(request.weights.in_pad(), sizeof(float), workspace_bytes))
      return false;
    return prepared_read_ranges_disjoint(request.weights, request.codebook,
                                          request.output.data(), output_bytes) &&
           prepared_read_ranges_disjoint(request.weights, request.codebook,
                                          request.workspace.data(),
                                          workspace_bytes) &&
           ranges_disjoint(request.output.data(), output_bytes,
                           request.activation.data(), activation_bytes) &&
           ranges_disjoint(request.output.data(), output_bytes,
                           request.workspace.data(), workspace_bytes) &&
           ranges_disjoint(request.workspace.data(), workspace_bytes,
                           request.activation.data(), activation_bytes);
#else
    (void)ev;
    (void)ctx;
    return false;
#endif
  }
};

struct guard_execute_prepared_avx2_dot_q4 {
  bool operator()(const event::execute_prepared_avx2_dot_q4 &ev,
                  const action::context &ctx) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    const auto &request = ev.request;
    size_t activation_bytes = 0u;
    size_t output_bytes = 0u;
    if (!ctx.avx2_fma_available || request.weights.group() != 128u ||
        !prepared_structure_supported(request.weights) ||
        !prepared_codebook_structure_supported(request.codebook) ||
        request.activation_fwht.size() < request.weights.in_pad() ||
        request.output.size() < request.weights.out() ||
        !std::isfinite(request.output_scale) ||
        !checked_bytes(request.weights.in_pad(), sizeof(float),
                       activation_bytes) ||
        !checked_bytes(request.weights.out(), sizeof(float), output_bytes))
      return false;
    return prepared_read_ranges_disjoint(request.weights, request.codebook,
                                          request.output.data(), output_bytes) &&
           ranges_disjoint(request.output.data(), output_bytes,
                           request.activation_fwht.data(), activation_bytes);
#else
    (void)ev;
    (void)ctx;
    return false;
#endif
  }
};

struct guard_execute_prepared_avx2_batch4_q4 {
  bool operator()(const event::execute_prepared_avx2_batch4_q4 &ev,
                  const action::context &ctx) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    const auto &request = ev.request;
    const auto *first = request.targets[0].weights;
    size_t activation_bytes = 0u;
    size_t workspace_bytes = 0u;
    if (!ctx.avx2_fma_available || first == nullptr || first->group() != 128u ||
        request.activation.size() < first->in() ||
        request.workspace.size() < first->in_pad() ||
        !std::isfinite(request.output_scale) ||
        !checked_bytes(first->in(), sizeof(float), activation_bytes) ||
        !checked_bytes(first->in_pad(), sizeof(float), workspace_bytes) ||
        !ranges_disjoint(request.workspace.data(), workspace_bytes,
                         request.activation.data(), activation_bytes))
      return false;
    for (size_t i = 0u; i < request.targets.size(); ++i) {
      const auto &target = request.targets[i];
      size_t output_bytes = 0u;
      if (target.weights == nullptr ||
          !prepared_structure_supported(*target.weights) ||
          !prepared_codebook_structure_supported(request.codebook) ||
          target.weights->in() != first->in() ||
          target.weights->group() != first->group() ||
          target.weights->in_pad() != first->in_pad() ||
          target.output.size() < target.weights->out() ||
          !checked_bytes(target.weights->out(), sizeof(float), output_bytes) ||
          !prepared_read_ranges_disjoint(*target.weights, request.codebook,
                                         target.output.data(), output_bytes) ||
          !prepared_read_ranges_disjoint(*target.weights, request.codebook,
                                         request.workspace.data(),
                                         workspace_bytes) ||
          !ranges_disjoint(target.output.data(), output_bytes,
                           request.activation.data(), activation_bytes) ||
          !ranges_disjoint(target.output.data(), output_bytes,
                           request.workspace.data(), workspace_bytes))
        return false;
      for (size_t j = 0u; j < i; ++j) {
        size_t other_bytes = 0u;
        if (!checked_bytes(request.targets[j].weights->out(), sizeof(float),
                           other_bytes) ||
            !ranges_disjoint(target.output.data(), output_bytes,
                             request.targets[j].output.data(), other_bytes))
          return false;
      }
    }
    return true;
#else
    (void)ev;
    (void)ctx;
    return false;
#endif
  }
};

struct guard_execute_prepared_avx2_rows_q4 {
  bool operator()(const event::execute_prepared_avx2_rows_q4 &ev,
                  const action::context &ctx) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    const auto &request = ev.request;
    size_t activation_bytes = 0u;
    size_t output_bytes = 0u;
    size_t workspace_bytes = 0u;
    if (!ctx.avx2_fma_available || request.weights.group() != 128u ||
        !prepared_structure_supported(request.weights) ||
        !prepared_codebook_structure_supported(request.codebook) ||
        request.row_count == 0u ||
        static_cast<uint64_t>(request.row_begin) + request.row_count >
            request.weights.out() ||
        request.activation.size() < request.weights.in() ||
        request.output.size() < request.row_count ||
        request.workspace.size() < request.weights.in_pad() ||
        !std::isfinite(request.output_scale) ||
        !checked_bytes(request.weights.in(), sizeof(float), activation_bytes) ||
        !checked_bytes(request.row_count, sizeof(float), output_bytes) ||
        !checked_bytes(request.weights.in_pad(), sizeof(float), workspace_bytes))
      return false;
    return prepared_read_ranges_disjoint(request.weights, request.codebook,
                                          request.output.data(), output_bytes) &&
           prepared_read_ranges_disjoint(request.weights, request.codebook,
                                          request.workspace.data(),
                                          workspace_bytes) &&
           ranges_disjoint(request.output.data(), output_bytes,
                           request.activation.data(), activation_bytes) &&
           ranges_disjoint(request.output.data(), output_bytes,
                           request.workspace.data(), workspace_bytes) &&
           ranges_disjoint(request.workspace.data(), workspace_bytes,
                           request.activation.data(), activation_bytes);
#else
    (void)ev;
    (void)ctx;
    return false;
#endif
  }
};

struct guard_execute_prepared_dequant_q4 {
  bool operator()(const event::execute_prepared_dequant_q4 &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    uint64_t output_count = 0u;
    size_t output_bytes = 0u;
    return prepared_structure_supported(request.weights) &&
           prepared_codebook_structure_supported(request.codebook) &&
           request.row_count > 0u &&
           static_cast<uint64_t>(request.row_begin) + request.row_count <=
               request.weights.out() &&
           detail::checked_multiply_u64(request.row_count, request.weights.in(),
                                        output_count) &&
           request.output.size() >= output_count &&
           checked_bytes(output_count, sizeof(float), output_bytes) &&
           std::isfinite(request.scale) &&
           prepared_read_ranges_disjoint(request.weights, request.codebook,
                                          request.output.data(), output_bytes);
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
                  const action::context &ctx) const noexcept {
    return avx2_supported<Bits>(ev.request, ctx);
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
