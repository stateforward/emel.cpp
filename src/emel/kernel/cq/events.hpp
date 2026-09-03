#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

#include "emel/kernel/cq/detail.hpp"

namespace emel::kernel::cq {
struct model;
}

namespace emel::kernel::cq::action {
struct effect_prepare_codebook_q4;
struct effect_prepare_q4;
} // namespace emel::kernel::cq::action

namespace emel::kernel::cq::event {

struct dispatch_result {
  bool accepted = false;
};

// Construction/init-owned exact CQ4 lookup representation. Only the
// preparation action may publish an execution-ready instance; all execution
// payload is copied into this object before publication.
class alignas(32) prepared_codebook_q4 {
public:
  using values_type = std::array<float, emel::cact::loader::k_codebook_len>;
  using byte_planes_type = std::array<std::array<uint8_t, 32u>, 4u>;

  prepared_codebook_q4() noexcept = default;
  prepared_codebook_q4(const prepared_codebook_q4 &) = delete;
  prepared_codebook_q4 &operator=(const prepared_codebook_q4 &) = delete;
  prepared_codebook_q4(prepared_codebook_q4 &&other) noexcept
      : values_(std::move(other.values_)),
        byte_planes_(std::move(other.byte_planes_)),
        published_(other.published_) {
    other.reset();
  }
  prepared_codebook_q4 &operator=(prepared_codebook_q4 &&other) noexcept {
    if (this == &other || published_)
      return *this;
    values_ = std::move(other.values_);
    byte_planes_ = std::move(other.byte_planes_);
    published_ = other.published_;
    other.reset();
    return *this;
  }

  [[nodiscard]] bool published() const noexcept { return published_; }
  [[nodiscard]] std::span<const float> values() const noexcept {
    return published_ ? std::span<const float>{values_}
                      : std::span<const float>{};
  }
  [[nodiscard]] const byte_planes_type &byte_planes() const noexcept {
    return byte_planes_;
  }

private:
  friend struct action::effect_prepare_codebook_q4;

  void publish(const std::span<const float> values,
               const byte_planes_type &byte_planes) noexcept {
    for (size_t i = 0u; i < values_.size(); ++i)
      values_[i] = values[i];
    byte_planes_ = byte_planes;
    published_ = true;
  }
  void reset() noexcept {
    values_ = {};
    byte_planes_ = {};
    published_ = false;
  }

  values_type values_ = {};
  byte_planes_type byte_planes_ = {};
  bool published_ = false;
};

struct prepare_codebook_q4_request {
  std::span<const float> codebook;
  prepared_codebook_q4 &prepared;
};

struct prepare_codebook_q4 {
  const prepare_codebook_q4_request &request;
  dispatch_result &result;
};

// JAX-compatible signed A8 fake quantization over one full activation vector.
// `quantized` keeps the exact integer operand and `integer_values` keeps the
// same integers widened to f32 for the linear FWHT/CQ projection. The caller
// applies `scale` once to each projection output.
struct quantize_a8_request {
  std::span<const float> input;
  std::span<int8_t> quantized;
  std::span<float> integer_values;
  float &scale;
};

struct quantize_a8 {
  const quantize_a8_request &request;
  dispatch_result &result;
};
struct fwht_request {
  std::span<float> values;
};

struct execute_fwht_avx2 {
  const fwht_request &request;
  dispatch_result &result;
};

struct gemv_request {
  const emel::cact::loader::tensor_view &weights;
  std::span<const float> codebook;
  std::span<const float> activation;
  std::span<float> output;
  std::span<float> workspace;
  float output_scale = 1.0f;
};
// Construction/init-owned CQ4 representation, published only by the
// preparation action after the complete source payload has been validated and
// materialized into private storage. Construction fixes exact capacity from
// geometry; preparation and execution are allocation-free.
class prepared_q4_view {
public:
  prepared_q4_view() noexcept = default;
  prepared_q4_view(const uint32_t out, const uint32_t in,
                   const uint32_t group) {
    detail::layout layout{};
    if (group > detail::k_max_group || !detail::is_power_of_two(group) ||
        !detail::checked_layout<4u>(out, in, group, layout))
      return;
    uint64_t index_count = 0u;
    if (!detail::checked_multiply_u64(out, layout.in_pad, index_count) ||
        index_count > std::numeric_limits<size_t>::max())
      return;
    const uint64_t blocked_rows = static_cast<uint64_t>(out / 32u) * 32u;
    uint64_t blocked_count = 0u;
    if (!detail::checked_multiply_u64(blocked_rows, layout.in_pad,
                                      blocked_count) ||
        blocked_count > std::numeric_limits<size_t>::max())
      return;
    const uint64_t norm_count = index_count / group;
    const uint64_t blocked_norm_count = blocked_count / group;
    if (norm_count > std::numeric_limits<size_t>::max() ||
        blocked_norm_count > std::numeric_limits<size_t>::max())
      return;
    out_ = out;
    in_ = in;
    group_ = group;
    in_pad_ = layout.in_pad;
    indices_.resize(static_cast<size_t>(index_count));
    indices_by_input32_.resize(static_cast<size_t>(blocked_count));
    norms_.resize(static_cast<size_t>(norm_count));
    norms_by_group32_.resize(static_cast<size_t>(blocked_norm_count));
    capacity_valid_ = true;
  }

  prepared_q4_view(const prepared_q4_view &) = delete;
  prepared_q4_view &operator=(const prepared_q4_view &) = delete;
  prepared_q4_view(prepared_q4_view &&other) noexcept {
    move_from(std::move(other));
  }
  prepared_q4_view &operator=(prepared_q4_view &&other) noexcept {
    if (this != &other && !published_)
      move_from(std::move(other));
    return *this;
  }

  [[nodiscard]] bool capacity_valid() const noexcept { return capacity_valid_; }
  [[nodiscard]] bool published() const noexcept { return published_; }
  [[nodiscard]] uint32_t out() const noexcept { return out_; }
  [[nodiscard]] uint32_t in() const noexcept { return in_; }
  [[nodiscard]] uint32_t group() const noexcept { return group_; }
  [[nodiscard]] uint32_t in_pad() const noexcept { return in_pad_; }
  [[nodiscard]] size_t index_capacity() const noexcept {
    return indices_.size();
  }
  [[nodiscard]] size_t input32_capacity() const noexcept {
    return indices_by_input32_.size();
  }
  [[nodiscard]] size_t norm_capacity() const noexcept { return norms_.size(); }
  [[nodiscard]] size_t group32_norm_capacity() const noexcept {
    return norms_by_group32_.size();
  }
  [[nodiscard]] std::span<const uint8_t> indices() const noexcept {
    return published_ ? std::span<const uint8_t>{indices_}
                      : std::span<const uint8_t>{};
  }
  // 32-row output blocks, input-major within each block. Tail rows remain in
  // row-major indices(); the blocked layout exists only for hot full GEMV.
  [[nodiscard]] std::span<const uint8_t> indices_by_input32() const noexcept {
    return published_ ? std::span<const uint8_t>{indices_by_input32_}
                      : std::span<const uint8_t>{};
  }
  [[nodiscard]] std::span<const float> norms() const noexcept {
    return published_ ? std::span<const float>{norms_}
                      : std::span<const float>{};
  }
  // Complete 32-row output blocks, group-major within each block and ordered
  // like lookup_codebook32_raw. Tail rows continue to use norms().
  [[nodiscard]] std::span<const float> norms_by_group32() const noexcept {
    return published_ ? std::span<const float>{norms_by_group32_}
                      : std::span<const float>{};
  }

private:
  friend struct action::effect_prepare_q4;

  void publish() noexcept { published_ = true; }
  void move_from(prepared_q4_view &&other) noexcept {
    out_ = other.out_;
    in_ = other.in_;
    group_ = other.group_;
    in_pad_ = other.in_pad_;
    indices_ = std::move(other.indices_);
    indices_by_input32_ = std::move(other.indices_by_input32_);
    norms_ = std::move(other.norms_);
    norms_by_group32_ = std::move(other.norms_by_group32_);
    capacity_valid_ = other.capacity_valid_;
    published_ = other.published_;
    other.out_ = 0u;
    other.in_ = 0u;
    other.group_ = 0u;
    other.in_pad_ = 0u;
    other.indices_.clear();
    other.indices_by_input32_.clear();
    other.norms_.clear();
    other.norms_by_group32_.clear();
    other.capacity_valid_ = false;
    other.published_ = false;
  }

  uint32_t out_ = 0u;
  uint32_t in_ = 0u;
  uint32_t group_ = 0u;
  uint32_t in_pad_ = 0u;
  std::vector<uint8_t> indices_ = {};
  std::vector<uint8_t> indices_by_input32_ = {};
  std::vector<float> norms_ = {};
  std::vector<float> norms_by_group32_ = {};
  bool capacity_valid_ = false;
  bool published_ = false;
};

struct prepare_q4_request {
  const emel::cact::loader::tensor_view &weights;
  prepared_q4_view &prepared;
};

struct prepare_q4 {
  const prepare_q4_request &request;
  dispatch_result &result;
};

struct prepared_gemv_request {
  const prepared_q4_view &weights;
  const prepared_codebook_q4 &codebook;
  std::span<const float> activation;
  std::span<float> output;
  std::span<float> workspace;
  float output_scale = 1.0f;
};

// Dot-only prepared CQ4 request over an activation already padded and
// transformed in 128-value FWHT groups by its owner.
struct prepared_dot_q4_request {
  const prepared_q4_view &weights;
  const prepared_codebook_q4 &codebook;
  std::span<const float> activation_fwht;
  std::span<float> output;
  float output_scale = 1.0f;
};

struct execute_prepared_avx2_dot_q4 {
  const prepared_dot_q4_request &request;
  dispatch_result &result;
};

struct prepared_gemv_target {
  const prepared_q4_view *weights = nullptr;
  std::span<float> output = {};
};

// Four projections sharing one activation transform. The fixed arity keeps
// dispatch allocation-free and matches the graph's q/k/v/gate hot path.
struct prepared_gemv_batch4_request {
  std::array<prepared_gemv_target, 4u> targets = {};
  const prepared_codebook_q4 &codebook;
  std::span<const float> activation;
  std::span<float> workspace;
  float output_scale = 1.0f;
};

struct execute_prepared_avx2_batch4_q4 {
  const prepared_gemv_batch4_request &request;
  dispatch_result &result;
};

struct execute_prepared_avx2_q4 {
  const prepared_gemv_request &request;
  dispatch_result &result;
};

template <uint32_t Bits> struct execute_scalar {
  const gemv_request &request;
  dispatch_result &result;
};
template <uint32_t Bits> struct execute_avx2 {
  const gemv_request &request;
  dispatch_result &result;
};

// Row-range GEMV over a packed CQ view: fills output[0..row_count) from packed
// rows [row_begin, row_begin + row_count). Weights stay packed; the fp16 norm
// table is addressed against the full view row count (shape[0]).
struct gemv_rows_request {
  const emel::cact::loader::tensor_view &weights;
  std::span<const float> codebook;
  std::span<const float> activation;
  uint32_t row_begin = 0u;
  uint32_t row_count = 0u;
  std::span<float> output;
  std::span<float> workspace;
  float output_scale = 1.0f;
};

template <uint32_t Bits> struct execute_scalar_rows {
  const gemv_rows_request &request;
  dispatch_result &result;
};

struct prepared_gemv_rows_request {
  const prepared_q4_view &weights;
  const prepared_codebook_q4 &codebook;
  std::span<const float> activation;
  uint32_t row_begin = 0u;
  uint32_t row_count = 0u;
  std::span<float> output;
  std::span<float> workspace;
  float output_scale = 1.0f;
};

struct execute_prepared_avx2_rows_q4 {
  const prepared_gemv_rows_request &request;
  dispatch_result &result;
};

// Dequantizes packed CQ rows [row_begin, row_begin + row_count) to f32 exactly
// like the exporter's `_cq_unpack` (codebook value scaled by the group norm,
// then the normalized Walsh-Hadamard rotation), truncated to shape[1] columns
// and scaled by `scale`. Intended for per-row gathers (embedding rows, engram
// table rows); never a whole-tensor dequant fallback.
struct dequant_rows_request {
  const emel::cact::loader::tensor_view &weights;
  std::span<const float> codebook;
  uint32_t row_begin = 0u;
  uint32_t row_count = 0u;
  float scale = 1.0f;
  std::span<float> output;
};

template <uint32_t Bits> struct execute_scalar_dequant {
  const dequant_rows_request &request;
  dispatch_result &result;
};

struct prepared_dequant_rows_request {
  const prepared_q4_view &weights;
  const prepared_codebook_q4 &codebook;
  uint32_t row_begin = 0u;
  uint32_t row_count = 0u;
  float scale = 1.0f;
  std::span<float> output;
};

struct execute_prepared_dequant_q4 {
  const prepared_dequant_rows_request &request;
  dispatch_result &result;
};

using execute_scalar_q2 = execute_scalar<2u>;
using execute_scalar_q3 = execute_scalar<3u>;
using execute_scalar_q4 = execute_scalar<4u>;
using execute_scalar_ternary = execute_scalar<5u>;
using execute_avx2_q2 = execute_avx2<2u>;
using execute_avx2_q3 = execute_avx2<3u>;
using execute_avx2_q4 = execute_avx2<4u>;
using execute_scalar_rows_q2 = execute_scalar_rows<2u>;
using execute_scalar_rows_q3 = execute_scalar_rows<3u>;
using execute_scalar_rows_q4 = execute_scalar_rows<4u>;
using execute_scalar_dequant_q2 = execute_scalar_dequant<2u>;
using execute_scalar_dequant_q3 = execute_scalar_dequant<3u>;
using execute_scalar_dequant_q4 = execute_scalar_dequant<4u>;

struct capture_diagnostics {
  uint64_t &scalar_calls;
  uint64_t &avx2_calls;
};

struct capture_prepared_diagnostics {
  uint64_t &prepare_calls;
  uint64_t &prepared_calls;
};

struct capture_a8_diagnostics {
  uint64_t &quantize_calls;
};

using timestamp_now_fn = uint64_t (*)() noexcept;

struct configure_timing {
  bool enabled = false;
  timestamp_now_fn now = nullptr;
};

struct timing_breakdown {
  uint64_t quantize_nanoseconds = 0u;
  uint64_t fwht_nanoseconds = 0u;
  uint64_t dot_full_nanoseconds = 0u;
  uint64_t dot_batch_nanoseconds = 0u;
  uint64_t dot_rows_nanoseconds = 0u;
  uint64_t dequant_nanoseconds = 0u;
};

struct capture_timing {
  timing_breakdown &breakdown;
};
} // namespace emel::kernel::cq::event
