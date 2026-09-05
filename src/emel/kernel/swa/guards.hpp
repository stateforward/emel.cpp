#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "emel/kernel/attention/guards.hpp"
#include "emel/kernel/swa/actions.hpp"

namespace emel::kernel::swa::guard {

namespace detail {

struct attend_lengths {
  std::size_t span = 0u;
  std::size_t workspace = 0u;
  std::size_t query = 0u;
  std::size_t cache = 0u;
};

struct cache_write_lengths {
  std::size_t row_elements = 0u;
  std::size_t cache_elements = 0u;
};

inline bool checked_multiply(const uint64_t lhs, const uint64_t rhs,
                             uint64_t &product) noexcept {
  if (lhs != 0u && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  product = lhs * rhs;
  return true;
}

inline bool checked_float_elements(const uint64_t elements,
                                   std::size_t &narrowed) noexcept {
  if (elements == 0u || elements > std::numeric_limits<std::size_t>::max() ||
      elements > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      elements > std::numeric_limits<std::uintptr_t>::max() / sizeof(float))
    return false;
  narrowed = static_cast<std::size_t>(elements);
  return true;
}

inline bool validate_attend_lengths(const event::attend_request &request,
                                    const uint64_t workspace_reps,
                                    attend_lengths &lengths) noexcept {
  if (request.heads == 0u || request.kv_heads == 0u || request.capacity == 0u ||
      request.head_dim == 0u || request.window_begin > request.position)
    return false;

  const uint64_t span =
      static_cast<uint64_t>(request.position) - request.window_begin + 1u;
  if (span == 0u || span > request.capacity ||
      span > std::numeric_limits<uint32_t>::max() ||
      span > std::numeric_limits<std::size_t>::max())
    return false;

  uint64_t workspace = 0u;
  uint64_t query = 0u;
  uint64_t cache_rows = 0u;
  uint64_t cache = 0u;
  if (!checked_multiply(span, workspace_reps, workspace) ||
      !checked_multiply(request.heads, request.head_dim, query) ||
      !checked_multiply(request.kv_heads, request.capacity, cache_rows) ||
      !checked_multiply(cache_rows, request.head_dim, cache) ||
      !checked_float_elements(workspace, lengths.workspace) ||
      !checked_float_elements(query, lengths.query) ||
      !checked_float_elements(cache, lengths.cache))
    return false;

  lengths.span = static_cast<std::size_t>(span);
  return true;
}

inline bool validate_attend_spans(const event::attend_request &request,
                                  const attend_lengths &lengths) noexcept {
  if (request.query.data() == nullptr || request.key_cache.data() == nullptr ||
      request.value_cache.data() == nullptr ||
      request.workspace.data() == nullptr || request.output.data() == nullptr ||
      request.query.size() < lengths.query ||
      request.key_cache.size() < lengths.cache ||
      request.value_cache.size() < lengths.cache ||
      request.workspace.size() < lengths.workspace ||
      request.output.size() < lengths.query)
    return false;

  const std::size_t workspace_bytes = lengths.workspace * sizeof(float);
  const std::size_t query_bytes = lengths.query * sizeof(float);
  const std::size_t cache_bytes = lengths.cache * sizeof(float);
  return attention::guard::guard_ranges_disjoint(
             request.workspace.data(), workspace_bytes, request.output.data(),
             query_bytes) &&
         attention::guard::guard_ranges_disjoint(
             request.workspace.data(), workspace_bytes, request.query.data(),
             query_bytes) &&
         attention::guard::guard_ranges_disjoint(
             request.workspace.data(), workspace_bytes,
             request.key_cache.data(), cache_bytes) &&
         attention::guard::guard_ranges_disjoint(
             request.workspace.data(), workspace_bytes,
             request.value_cache.data(), cache_bytes) &&
         attention::guard::guard_ranges_disjoint(
             request.output.data(), query_bytes, request.query.data(),
             query_bytes) &&
         attention::guard::guard_ranges_disjoint(
             request.output.data(), query_bytes, request.key_cache.data(),
             cache_bytes) &&
         attention::guard::guard_ranges_disjoint(
             request.output.data(), query_bytes, request.value_cache.data(),
             cache_bytes);
}

inline bool
validate_cache_write_lengths(const event::cache_write_request &request,
                             cache_write_lengths &lengths) noexcept {
  if (request.kv_heads == 0u || request.head_dim == 0u ||
      request.capacity == 0u)
    return false;

  uint64_t rows = 0u;
  uint64_t cache_rows = 0u;
  uint64_t cache = 0u;
  if (!checked_multiply(request.kv_heads, request.head_dim, rows) ||
      !checked_multiply(request.kv_heads, request.capacity, cache_rows) ||
      !checked_multiply(cache_rows, request.head_dim, cache) ||
      !checked_float_elements(rows, lengths.row_elements) ||
      !checked_float_elements(cache, lengths.cache_elements))
    return false;

  return true;
}

inline bool
validate_cache_write_spans(const event::cache_write_request &request,
                           const cache_write_lengths &lengths) noexcept {
  if (request.key_rows.data() == nullptr ||
      request.value_rows.data() == nullptr ||
      request.key_cache.data() == nullptr ||
      request.value_cache.data() == nullptr ||
      request.key_rows.size() < lengths.row_elements ||
      request.value_rows.size() < lengths.row_elements ||
      request.key_cache.size() < lengths.cache_elements ||
      request.value_cache.size() < lengths.cache_elements)
    return false;

  const std::size_t rows_bytes = lengths.row_elements * sizeof(float);
  const std::size_t cache_bytes = lengths.cache_elements * sizeof(float);
  return attention::guard::guard_ranges_disjoint(
             request.key_cache.data(), cache_bytes, request.value_cache.data(),
             cache_bytes) &&
         attention::guard::guard_ranges_disjoint(
             request.key_cache.data(), cache_bytes, request.key_rows.data(),
             rows_bytes) &&
         attention::guard::guard_ranges_disjoint(
             request.key_cache.data(), cache_bytes, request.value_rows.data(),
             rows_bytes) &&
         attention::guard::guard_ranges_disjoint(
             request.value_cache.data(), cache_bytes, request.key_rows.data(),
             rows_bytes) &&
         attention::guard::guard_ranges_disjoint(
             request.value_cache.data(), cache_bytes, request.value_rows.data(),
             rows_bytes);
}

} // namespace detail

struct guard_execute_attend {
  bool operator()(const event::execute_attend &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    detail::attend_lengths lengths{};
    return detail::validate_attend_lengths(request, 1u, lengths) &&
           (request.heads % request.kv_heads) == 0u &&
           detail::validate_attend_spans(request, lengths);
  }
};

struct guard_execute_attend_gqa2_avx2 {
  bool operator()(const event::execute_attend_gqa2_avx2 &ev,
                  const action::context &ctx) const noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__AVX2__) &&           \
    defined(__FMA__)
    const auto &request = ev.request;
    detail::attend_lengths lengths{};
    return ctx.avx2_fma_available &&
           static_cast<uint64_t>(request.heads) ==
               uint64_t{2} * request.kv_heads &&
           detail::validate_attend_lengths(request, 2u, lengths) &&
           detail::validate_attend_spans(request, lengths);
#else
    (void)ev;
    return false;
#endif
  }
};

struct guard_execute_attend_gqa2_avx2_vector_exp {
  bool operator()(const event::execute_attend_gqa2_avx2_vector_exp &ev,
                  const action::context &ctx) const noexcept {
    return guard_execute_attend_gqa2_avx2{}(
        event::execute_attend_gqa2_avx2{ev.request, ev.result}, ctx);
  }
};

struct guard_execute_cache_write {
  bool operator()(const event::execute_cache_write &ev,
                  const action::context &) const noexcept {
    const auto &request = ev.request;
    detail::cache_write_lengths lengths{};
    return detail::validate_cache_write_lengths(request, lengths) &&
           detail::validate_cache_write_spans(request, lengths);
  }
};

struct guard_execute_gate_mul {
  bool operator()(const event::execute_gate_mul &ev,
                  const action::context &) const noexcept {
    return ev.request.dim > 0u && ev.request.values.size() >= ev.request.dim &&
           ev.request.gate_logits.size() >= ev.request.dim;
  }
};

struct guard_execute_residual_gate {
  bool operator()(const event::execute_residual_gate &ev,
                  const action::context &) const noexcept {
    return ev.request.dim > 0u && ev.request.skip.size() >= ev.request.dim &&
           ev.request.values.size() >= ev.request.dim &&
           ev.request.output.size() >= ev.request.dim;
  }
};

} // namespace emel::kernel::swa::guard
