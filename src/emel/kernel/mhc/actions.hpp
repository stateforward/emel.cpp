#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/mhc/events.hpp"

namespace emel::kernel::mhc::action {

// The mhc kernel holds no persistent actor state.
struct context {};

struct effect_execute_pre_mix {
  void operator()(const event::execute_pre_mix &ev, context &) const noexcept {
    namespace quant = emel::kernel::detail::quant;
    const auto &request = ev.request;
    const uint16_t *a_bits =
        reinterpret_cast<const uint16_t *>(request.a.data());
    const uint16_t *b_bits =
        reinterpret_cast<const uint16_t *>(request.b.data());
    const float a = quant::fp16_to_fp32(a_bits[0]);
    for (uint32_t i = 0u; i < request.dim; ++i)
      request.output[i] = 0.0f;
    for (uint32_t lane = 0u; lane < request.lane_count; ++lane) {
      const float pre_off =
          8.0f * static_cast<float>(lane == request.lane_index) - 4.0f;
      const float logit = a * request.phi_dots[lane] +
                          quant::fp16_to_fp32(b_bits[lane]) + pre_off;
      const float weight = 1.0f / (1.0f + std::exp(-logit));
      const float *lane_row =
          request.lanes.data() + static_cast<size_t>(lane) * request.dim;
      for (uint32_t i = 0u; i < request.dim; ++i)
        request.output[i] += weight * lane_row[i];
    }
    ev.result.accepted = true;
  }
};

struct effect_execute_post_mix {
  void operator()(const event::execute_post_mix &ev, context &) const noexcept {
    namespace quant = emel::kernel::detail::quant;
    const auto &request = ev.request;
    const uint32_t n = request.lane_count;
    const uint16_t *b_post_bits =
        reinterpret_cast<const uint16_t *>(request.b_post.data());
    const uint16_t *b_res_bits =
        reinterpret_cast<const uint16_t *>(request.b_res.data());
    const float a_post = quant::fp16_to_fp32(
        reinterpret_cast<const uint16_t *>(request.a_post.data())[0]);
    const float a_res = quant::fp16_to_fp32(
        reinterpret_cast<const uint16_t *>(request.a_res.data())[0]);

    // Log-domain sinkhorn over the residual routing logits (20 iterations of
    // row then column log-sum-exp normalization), matching `_sinkhorn`.
    float log_kernel[event::k_max_lanes * event::k_max_lanes];
    for (uint32_t row = 0u; row < n; ++row)
      for (uint32_t col = 0u; col < n; ++col) {
        const uint32_t at = row * n + col;
        log_kernel[at] =
            a_res * request.res_dots[at] + quant::fp16_to_fp32(b_res_bits[at]);
      }
    for (uint32_t iteration = 0u; iteration < event::k_sinkhorn_iterations;
         ++iteration) {
      for (uint32_t row = 0u; row < n; ++row) {
        float max_value = log_kernel[row * n];
        for (uint32_t col = 1u; col < n; ++col)
          max_value = std::max(max_value, log_kernel[row * n + col]);
        float sum = 0.0f;
        for (uint32_t col = 0u; col < n; ++col)
          sum += std::exp(log_kernel[row * n + col] - max_value);
        const float log_sum = max_value + std::log(sum);
        for (uint32_t col = 0u; col < n; ++col)
          log_kernel[row * n + col] -= log_sum;
      }
      for (uint32_t col = 0u; col < n; ++col) {
        float max_value = log_kernel[col];
        for (uint32_t row = 1u; row < n; ++row)
          max_value = std::max(max_value, log_kernel[row * n + col]);
        float sum = 0.0f;
        for (uint32_t row = 0u; row < n; ++row)
          sum += std::exp(log_kernel[row * n + col] - max_value);
        const float log_sum = max_value + std::log(sum);
        for (uint32_t row = 0u; row < n; ++row)
          log_kernel[row * n + col] -= log_sum;
      }
    }

    for (uint32_t row = 0u; row < n; ++row) {
      const float post_off =
          -4.0f * (1.0f - static_cast<float>(row == request.lane_index));
      const float logit = a_post * request.post_dots[row] +
                          quant::fp16_to_fp32(b_post_bits[row]) + post_off;
      const float hpost = 2.0f / (1.0f + std::exp(-logit));
      float *output_row =
          request.output.data() + static_cast<size_t>(row) * request.dim;
      for (uint32_t i = 0u; i < request.dim; ++i)
        output_row[i] = hpost * (request.block_out[i] - request.u[i]);
      for (uint32_t col = 0u; col < n; ++col) {
        const float weight = std::exp(log_kernel[row * n + col]);
        const float *lane_row =
            request.lanes.data() + static_cast<size_t>(col) * request.dim;
        for (uint32_t i = 0u; i < request.dim; ++i)
          output_row[i] += weight * lane_row[i];
      }
    }
    ev.result.accepted = true;
  }
};

struct effect_execute_mean_lanes {
  void operator()(const event::execute_mean_lanes &ev,
                  context &) const noexcept {
    const auto &request = ev.request;
    const float inv_count = 1.0f / static_cast<float>(request.lane_count);
    for (uint32_t i = 0u; i < request.dim; ++i)
      request.output[i] = 0.0f;
    for (uint32_t lane = 0u; lane < request.lane_count; ++lane) {
      const float *lane_row =
          request.lanes.data() + static_cast<size_t>(lane) * request.dim;
      for (uint32_t i = 0u; i < request.dim; ++i)
        request.output[i] += lane_row[i] * inv_count;
    }
    ev.result.accepted = true;
  }
};

struct effect_on_unexpected {
  template <class event_type>
  void operator()(const event_type &, context &) const noexcept {}
};

} // namespace emel::kernel::mhc::action
