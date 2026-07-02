#include "emel/diarization/sortformer/detail.hpp"

#include <cstddef>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include "emel/kernel/aarch64/actions.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"

namespace emel::diarization::sortformer::detail {

namespace {

#if defined(__aarch64__) || defined(__ARM_NEON)
float compute_neon_horizontal_sum(const float32x4_t value) noexcept {
#if defined(__aarch64__)
  return vaddvq_f32(value);
#else
  const float32x2_t lanes = vadd_f32(vget_low_f32(value), vget_high_f32(value));
  const float32x2_t sum = vpadd_f32(lanes, lanes);
  return vget_lane_f32(sum, 0);
#endif
}

float32x4_t compute_neon_fma(const float32x4_t acc,
                             const float32x4_t lhs,
                             const float32x4_t rhs) noexcept {
#if defined(__aarch64__)
  return vfmaq_f32(acc, lhs, rhs);
#else
  return vmlaq_f32(acc, lhs, rhs);
#endif
}
#endif

bool run_dense_matmul(std::span<const float> input,
                      std::span<const float> weights,
                      std::span<float> output) noexcept {
  const uint64_t row_bytes = sizeof(float) * static_cast<uint64_t>(input.size());

  emel::kernel::event::op_mul_mat request{
      .src0 = {
          .data = weights.data(),
          .type = emel::kernel::event::dtype::f32,
          .ne = {
              static_cast<uint64_t>(input.size()),
              static_cast<uint64_t>(output.size()),
              1u,
              1u,
          },
          .nb = {
              sizeof(float),
              row_bytes,
              row_bytes * static_cast<uint64_t>(output.size()),
              row_bytes * static_cast<uint64_t>(output.size()),
          },
      },
      .src1 = {
          .data = input.data(),
          .type = emel::kernel::event::dtype::f32,
          .ne = {
              1u,
              static_cast<uint64_t>(input.size()),
              1u,
              1u,
          },
          .nb = {
              sizeof(float),
              sizeof(float),
              sizeof(float) * static_cast<uint64_t>(input.size()),
              sizeof(float) * static_cast<uint64_t>(input.size()),
          },
      },
      .dst = {
          .data = output.data(),
          .type = emel::kernel::event::dtype::f32,
          .ne = {
              1u,
              static_cast<uint64_t>(output.size()),
              1u,
              1u,
          },
          .nb = {
              sizeof(float),
              sizeof(float),
              sizeof(float) * static_cast<uint64_t>(output.size()),
              sizeof(float) * static_cast<uint64_t>(output.size()),
          },
      },
  };

#if defined(__aarch64__) || defined(__ARM_NEON)
  if (!emel::kernel::aarch64::detail::execute_neon_mul_mat(request)) {
    return false;
  }
#else
  if (!emel::kernel::detail::run_mul_mat(request)) {
    return false;
  }
#endif
  return true;
}

bool run_dense_batch_matmul_from_transposed(std::span<const float> transposed_input,
                                            const size_t row_count,
                                            const size_t input_dim,
                                            std::span<const float> weights,
                                            const size_t output_dim,
                                            std::span<float> transposed_output) noexcept {
  const uint64_t input_row_bytes = sizeof(float) * static_cast<uint64_t>(input_dim);
  const uint64_t frame_row_bytes = sizeof(float) * static_cast<uint64_t>(row_count);

  emel::kernel::event::op_mul_mat request{
      .src0 = {
          .data = weights.data(),
          .type = emel::kernel::event::dtype::f32,
          .ne = {
              static_cast<uint64_t>(input_dim),
              static_cast<uint64_t>(output_dim),
              1u,
              1u,
          },
          .nb = {
              sizeof(float),
              input_row_bytes,
              input_row_bytes * static_cast<uint64_t>(output_dim),
              input_row_bytes * static_cast<uint64_t>(output_dim),
          },
      },
      .src1 = {
          .data = transposed_input.data(),
          .type = emel::kernel::event::dtype::f32,
          .ne = {
              static_cast<uint64_t>(row_count),
              static_cast<uint64_t>(input_dim),
              1u,
              1u,
          },
          .nb = {
              sizeof(float),
              frame_row_bytes,
              frame_row_bytes * static_cast<uint64_t>(input_dim),
              frame_row_bytes * static_cast<uint64_t>(input_dim),
          },
      },
      .dst = {
          .data = transposed_output.data(),
          .type = emel::kernel::event::dtype::f32,
          .ne = {
              static_cast<uint64_t>(row_count),
              static_cast<uint64_t>(output_dim),
              1u,
              1u,
          },
          .nb = {
              sizeof(float),
              frame_row_bytes,
              frame_row_bytes * static_cast<uint64_t>(output_dim),
              frame_row_bytes * static_cast<uint64_t>(output_dim),
          },
      },
  };

#if defined(__aarch64__) || defined(__ARM_NEON)
  if (!emel::kernel::aarch64::detail::execute_neon_mul_mat(request)) {
    return false;
  }
#else
  if (!emel::kernel::detail::run_mul_mat(request)) {
    return false;
  }
#endif

  return true;
}

bool run_dense_batch_matmul_from_transposed_prepared(
    std::span<const float> transposed_input,
    const size_t row_count,
    const size_t input_dim,
    std::span<const float> weights,
    const dense_weight_cache & cache,
    const size_t output_dim,
    std::span<float> transposed_output) noexcept {
  const uint64_t input_row_bytes = sizeof(float) * static_cast<uint64_t>(input_dim);
  const uint64_t frame_row_bytes = sizeof(float) * static_cast<uint64_t>(row_count);

  emel::kernel::event::op_mul_mat request{
      .src0 = {
          .data = weights.data(),
          .type = emel::kernel::event::dtype::f32,
          .ne = {
              static_cast<uint64_t>(input_dim),
              static_cast<uint64_t>(output_dim),
              1u,
              1u,
          },
          .nb = {
              sizeof(float),
              input_row_bytes,
              input_row_bytes * static_cast<uint64_t>(output_dim),
              input_row_bytes * static_cast<uint64_t>(output_dim),
          },
      },
      .src1 = {
          .data = transposed_input.data(),
          .type = emel::kernel::event::dtype::f32,
          .ne = {
              static_cast<uint64_t>(row_count),
              static_cast<uint64_t>(input_dim),
              1u,
              1u,
          },
          .nb = {
              sizeof(float),
              frame_row_bytes,
              frame_row_bytes * static_cast<uint64_t>(input_dim),
              frame_row_bytes * static_cast<uint64_t>(input_dim),
          },
      },
      .dst = {
          .data = transposed_output.data(),
          .type = emel::kernel::event::dtype::f32,
          .ne = {
              static_cast<uint64_t>(row_count),
              static_cast<uint64_t>(output_dim),
              1u,
              1u,
          },
          .nb = {
              sizeof(float),
              frame_row_bytes,
              frame_row_bytes * static_cast<uint64_t>(output_dim),
              frame_row_bytes * static_cast<uint64_t>(output_dim),
          },
      },
  };

#if defined(__aarch64__) || defined(__ARM_NEON)
  if (!emel::kernel::aarch64::detail::execute_neon_mul_mat_prepared_f32_lhs_4row(
          request,
          cache.lhs_4row.data(),
          static_cast<uint64_t>(cache.lhs_4row.size()))) {
    return false;
  }
#else
  (void)cache;
  if (!emel::kernel::detail::run_mul_mat(request)) {
    return false;
  }
#endif

  return true;
}

}  // namespace

float compute_dot_64(const float *lhs, const float *rhs) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  for (size_t dim = 0u; dim < 64u; dim += 16u) {
    acc0 = compute_neon_fma(acc0, vld1q_f32(lhs + dim), vld1q_f32(rhs + dim));
    acc1 = compute_neon_fma(acc1, vld1q_f32(lhs + dim + 4u),
                            vld1q_f32(rhs + dim + 4u));
    acc2 = compute_neon_fma(acc2, vld1q_f32(lhs + dim + 8u),
                            vld1q_f32(rhs + dim + 8u));
    acc3 = compute_neon_fma(acc3, vld1q_f32(lhs + dim + 12u),
                            vld1q_f32(rhs + dim + 12u));
  }

  return compute_neon_horizontal_sum(vaddq_f32(vaddq_f32(acc0, acc1),
                                               vaddq_f32(acc2, acc3)));
#else
  float acc = 0.0f;
  for (size_t dim = 0u; dim < 64u; ++dim) {
    acc += lhs[dim] * rhs[dim];
  }
  return acc;
#endif
}

float compute_dot_24(const float *lhs, const float *rhs) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  for (size_t dim = 0u; dim < 24u; dim += 12u) {
    acc0 = compute_neon_fma(acc0, vld1q_f32(lhs + dim), vld1q_f32(rhs + dim));
    acc1 = compute_neon_fma(acc1, vld1q_f32(lhs + dim + 4u),
                            vld1q_f32(rhs + dim + 4u));
    acc2 = compute_neon_fma(acc2, vld1q_f32(lhs + dim + 8u),
                            vld1q_f32(rhs + dim + 8u));
  }

  return compute_neon_horizontal_sum(vaddq_f32(vaddq_f32(acc0, acc1), acc2));
#else
  float acc = 0.0f;
  for (size_t dim = 0u; dim < 24u; ++dim) {
    acc += lhs[dim] * rhs[dim];
  }
  return acc;
#endif
}

void compute_weighted_sum_64(const float *weights,
                             const float *values,
                             const size_t value_stride,
                             const size_t value_count,
                             float *output) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  float32x4_t acc4 = vdupq_n_f32(0.0f);
  float32x4_t acc5 = vdupq_n_f32(0.0f);
  float32x4_t acc6 = vdupq_n_f32(0.0f);
  float32x4_t acc7 = vdupq_n_f32(0.0f);
  float32x4_t acc8 = vdupq_n_f32(0.0f);
  float32x4_t acc9 = vdupq_n_f32(0.0f);
  float32x4_t acc10 = vdupq_n_f32(0.0f);
  float32x4_t acc11 = vdupq_n_f32(0.0f);
  float32x4_t acc12 = vdupq_n_f32(0.0f);
  float32x4_t acc13 = vdupq_n_f32(0.0f);
  float32x4_t acc14 = vdupq_n_f32(0.0f);
  float32x4_t acc15 = vdupq_n_f32(0.0f);
  for (size_t index = 0u; index < value_count; ++index) {
    const float32x4_t weight = vdupq_n_f32(weights[index]);
    const float *row = values + (index * value_stride);
    acc0 = compute_neon_fma(acc0, weight, vld1q_f32(row));
    acc1 = compute_neon_fma(acc1, weight, vld1q_f32(row + 4u));
    acc2 = compute_neon_fma(acc2, weight, vld1q_f32(row + 8u));
    acc3 = compute_neon_fma(acc3, weight, vld1q_f32(row + 12u));
    acc4 = compute_neon_fma(acc4, weight, vld1q_f32(row + 16u));
    acc5 = compute_neon_fma(acc5, weight, vld1q_f32(row + 20u));
    acc6 = compute_neon_fma(acc6, weight, vld1q_f32(row + 24u));
    acc7 = compute_neon_fma(acc7, weight, vld1q_f32(row + 28u));
    acc8 = compute_neon_fma(acc8, weight, vld1q_f32(row + 32u));
    acc9 = compute_neon_fma(acc9, weight, vld1q_f32(row + 36u));
    acc10 = compute_neon_fma(acc10, weight, vld1q_f32(row + 40u));
    acc11 = compute_neon_fma(acc11, weight, vld1q_f32(row + 44u));
    acc12 = compute_neon_fma(acc12, weight, vld1q_f32(row + 48u));
    acc13 = compute_neon_fma(acc13, weight, vld1q_f32(row + 52u));
    acc14 = compute_neon_fma(acc14, weight, vld1q_f32(row + 56u));
    acc15 = compute_neon_fma(acc15, weight, vld1q_f32(row + 60u));
  }

  vst1q_f32(output, acc0);
  vst1q_f32(output + 4u, acc1);
  vst1q_f32(output + 8u, acc2);
  vst1q_f32(output + 12u, acc3);
  vst1q_f32(output + 16u, acc4);
  vst1q_f32(output + 20u, acc5);
  vst1q_f32(output + 24u, acc6);
  vst1q_f32(output + 28u, acc7);
  vst1q_f32(output + 32u, acc8);
  vst1q_f32(output + 36u, acc9);
  vst1q_f32(output + 40u, acc10);
  vst1q_f32(output + 44u, acc11);
  vst1q_f32(output + 48u, acc12);
  vst1q_f32(output + 52u, acc13);
  vst1q_f32(output + 56u, acc14);
  vst1q_f32(output + 60u, acc15);
#else
  for (size_t dim = 0u; dim < 64u; ++dim) {
    float acc = 0.0f;
    for (size_t index = 0u; index < value_count; ++index) {
      acc += weights[index] * values[(index * value_stride) + dim];
    }
    output[dim] = acc;
  }
#endif
}

void compute_weighted_sum_24(const float *weights,
                             const float *values,
                             const size_t value_stride,
                             const size_t value_count,
                             float *output) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  float32x4_t acc2 = vdupq_n_f32(0.0f);
  float32x4_t acc3 = vdupq_n_f32(0.0f);
  float32x4_t acc4 = vdupq_n_f32(0.0f);
  float32x4_t acc5 = vdupq_n_f32(0.0f);
  for (size_t index = 0u; index < value_count; ++index) {
    const float32x4_t weight = vdupq_n_f32(weights[index]);
    const float *row = values + (index * value_stride);
    acc0 = compute_neon_fma(acc0, weight, vld1q_f32(row));
    acc1 = compute_neon_fma(acc1, weight, vld1q_f32(row + 4u));
    acc2 = compute_neon_fma(acc2, weight, vld1q_f32(row + 8u));
    acc3 = compute_neon_fma(acc3, weight, vld1q_f32(row + 12u));
    acc4 = compute_neon_fma(acc4, weight, vld1q_f32(row + 16u));
    acc5 = compute_neon_fma(acc5, weight, vld1q_f32(row + 20u));
  }

  vst1q_f32(output, acc0);
  vst1q_f32(output + 4u, acc1);
  vst1q_f32(output + 8u, acc2);
  vst1q_f32(output + 12u, acc3);
  vst1q_f32(output + 16u, acc4);
  vst1q_f32(output + 20u, acc5);
#else
  for (size_t dim = 0u; dim < 24u; ++dim) {
    float acc = 0.0f;
    for (size_t index = 0u; index < value_count; ++index) {
      acc += weights[index] * values[(index * value_stride) + dim];
    }
    output[dim] = acc;
  }
#endif
}

bool prepare_dense_weight_cache(std::span<const float> weights,
                                const size_t input_dim,
                                const size_t output_dim,
                                dense_weight_cache & cache) noexcept {
  if (input_dim == 0u || output_dim == 0u || weights.size() != input_dim * output_dim) {
    return false;
  }

  if (cache.source == weights.data() &&
      cache.input_dim == input_dim &&
      cache.output_dim == output_dim) {
    return true;
  }

#if defined(__aarch64__) || defined(__ARM_NEON)
  const uint64_t prepared_count =
      emel::kernel::aarch64::detail::prepared_f32_lhs_4row_value_count(
          static_cast<uint64_t>(input_dim),
          static_cast<uint64_t>(output_dim));
  if (cache.lhs_4row.size() < prepared_count) {
    return false;
  }
  if (!emel::kernel::aarch64::detail::prepare_neon_mul_mat_f32_lhs_4row(
          weights.data(),
          static_cast<uint64_t>(input_dim),
          static_cast<uint64_t>(output_dim),
          cache.lhs_4row.data(),
          static_cast<uint64_t>(cache.lhs_4row.size()))) {
    return false;
  }
#endif

  cache.source = weights.data();
  cache.input_dim = input_dim;
  cache.output_dim = output_dim;
  return true;
}

bool compute_dense(std::span<const float> input,
                   std::span<const float> weights,
                   std::span<const float> bias,
                   std::span<float> output) noexcept {
  if (input.empty() || output.empty() || bias.size() != output.size() ||
      weights.size() != input.size() * output.size()) {
    return false;
  }

  if (!run_dense_matmul(input, weights, output)) {
    return false;
  }

  for (size_t row = 0u; row < output.size(); ++row) {
    output[row] += bias[row];
  }

  return true;
}

bool compute_dense_without_bias(std::span<const float> input,
                                std::span<const float> weights,
                                std::span<float> output) noexcept {
  if (input.empty() || output.empty() || weights.size() != input.size() * output.size()) {
    return false;
  }

  return run_dense_matmul(input, weights, output);
}

bool transpose_dense_input(std::span<const float> input_rows,
                           const size_t row_count,
                           const size_t input_dim,
                           std::span<float> transposed_input) noexcept {
  if (row_count == 0u || input_dim == 0u ||
      input_rows.size() != row_count * input_dim ||
      transposed_input.size() < input_dim * row_count) {
    return false;
  }

  for (size_t input_index = 0u; input_index < input_dim; ++input_index) {
    for (size_t row = 0u; row < row_count; ++row) {
      transposed_input[(input_index * row_count) + row] =
          input_rows[(row * input_dim) + input_index];
    }
  }

  return true;
}

bool compute_dense_batch(std::span<const float> input_rows,
                         const size_t row_count,
                         const size_t input_dim,
                         std::span<const float> weights,
                         std::span<const float> bias,
                         const size_t output_dim,
                         std::span<float> transposed_input,
                         std::span<float> transposed_output,
                         std::span<float> output_rows) noexcept {
  if (row_count == 0u || input_dim == 0u || output_dim == 0u ||
      input_rows.size() != row_count * input_dim ||
      weights.size() != input_dim * output_dim ||
      bias.size() != output_dim ||
      transposed_input.size() < input_dim * row_count ||
      transposed_output.size() < output_dim * row_count ||
      output_rows.size() != row_count * output_dim) {
    return false;
  }

  for (size_t input_index = 0u; input_index < input_dim; ++input_index) {
    for (size_t row = 0u; row < row_count; ++row) {
      transposed_input[(input_index * row_count) + row] =
          input_rows[(row * input_dim) + input_index];
    }
  }

  if (!run_dense_batch_matmul_from_transposed(transposed_input,
                                              row_count,
                                              input_dim,
                                              weights,
                                              output_dim,
                                              transposed_output)) {
    return false;
  }

  for (size_t row = 0u; row < row_count; ++row) {
    for (size_t output_index = 0u; output_index < output_dim; ++output_index) {
      output_rows[(row * output_dim) + output_index] =
          transposed_output[(output_index * row_count) + row] + bias[output_index];
    }
  }

  return true;
}

bool compute_dense_batch_prepared(std::span<const float> input_rows,
                                  const size_t row_count,
                                  const size_t input_dim,
                                  std::span<const float> weights,
                                  const dense_weight_cache & cache,
                                  std::span<const float> bias,
                                  const size_t output_dim,
                                  std::span<float> transposed_input,
                                  std::span<float> transposed_output,
                                  std::span<float> output_rows) noexcept {
  if (row_count == 0u || input_dim == 0u || output_dim == 0u ||
      input_rows.size() != row_count * input_dim ||
      weights.size() != input_dim * output_dim ||
      cache.source != weights.data() ||
      cache.input_dim != input_dim ||
      cache.output_dim != output_dim ||
      bias.size() != output_dim ||
      transposed_input.size() < input_dim * row_count ||
      transposed_output.size() < output_dim * row_count ||
      output_rows.size() != row_count * output_dim) {
    return false;
  }

  for (size_t input_index = 0u; input_index < input_dim; ++input_index) {
    for (size_t row = 0u; row < row_count; ++row) {
      transposed_input[(input_index * row_count) + row] =
          input_rows[(row * input_dim) + input_index];
    }
  }

  if (!run_dense_batch_matmul_from_transposed_prepared(transposed_input,
                                                       row_count,
                                                       input_dim,
                                                       weights,
                                                       cache,
                                                       output_dim,
                                                       transposed_output)) {
    return false;
  }

  for (size_t row = 0u; row < row_count; ++row) {
    for (size_t output_index = 0u; output_index < output_dim; ++output_index) {
      output_rows[(row * output_dim) + output_index] =
          transposed_output[(output_index * row_count) + row] + bias[output_index];
    }
  }

  return true;
}

bool compute_dense_batch_residual_prepared(std::span<const float> input_rows,
                                           const size_t row_count,
                                           const size_t input_dim,
                                           std::span<const float> weights,
                                           const dense_weight_cache & cache,
                                           std::span<const float> bias,
                                           const size_t output_dim,
                                           std::span<const float> residual_rows,
                                           std::span<float> transposed_input,
                                           std::span<float> transposed_output,
                                           std::span<float> output_rows) noexcept {
  if (row_count == 0u || input_dim == 0u || output_dim == 0u ||
      input_rows.size() != row_count * input_dim ||
      weights.size() != input_dim * output_dim ||
      cache.source != weights.data() ||
      cache.input_dim != input_dim ||
      cache.output_dim != output_dim ||
      bias.size() != output_dim ||
      residual_rows.size() != row_count * output_dim ||
      transposed_input.size() < input_dim * row_count ||
      transposed_output.size() < output_dim * row_count ||
      output_rows.size() != row_count * output_dim) {
    return false;
  }

  if (!transpose_dense_input(input_rows, row_count, input_dim, transposed_input)) {
    return false;
  }

  return compute_dense_batch_from_transposed_scaled_residual_prepared(transposed_input,
                                                                      row_count,
                                                                      input_dim,
                                                                      weights,
                                                                      cache,
                                                                      bias,
                                                                      output_dim,
                                                                      1.0f,
                                                                      residual_rows,
                                                                      transposed_output,
                                                                      output_rows);
}

bool compute_dense_batch_without_bias(std::span<const float> input_rows,
                                      const size_t row_count,
                                      const size_t input_dim,
                                      std::span<const float> weights,
                                      const size_t output_dim,
                                      std::span<float> transposed_input,
                                      std::span<float> transposed_output,
                                      std::span<float> output_rows) noexcept {
  if (row_count == 0u || input_dim == 0u || output_dim == 0u ||
      input_rows.size() != row_count * input_dim ||
      weights.size() != input_dim * output_dim ||
      transposed_input.size() < input_dim * row_count ||
      transposed_output.size() < output_dim * row_count ||
      output_rows.size() != row_count * output_dim) {
    return false;
  }

  for (size_t input_index = 0u; input_index < input_dim; ++input_index) {
    for (size_t row = 0u; row < row_count; ++row) {
      transposed_input[(input_index * row_count) + row] =
          input_rows[(row * input_dim) + input_index];
    }
  }

  if (!run_dense_batch_matmul_from_transposed(transposed_input,
                                              row_count,
                                              input_dim,
                                              weights,
                                              output_dim,
                                              transposed_output)) {
    return false;
  }

  for (size_t row = 0u; row < row_count; ++row) {
    for (size_t output_index = 0u; output_index < output_dim; ++output_index) {
      output_rows[(row * output_dim) + output_index] =
          transposed_output[(output_index * row_count) + row];
    }
  }

  return true;
}

bool compute_dense_batch_without_bias_prepared(std::span<const float> input_rows,
                                               const size_t row_count,
                                               const size_t input_dim,
                                               std::span<const float> weights,
                                               const dense_weight_cache & cache,
                                               const size_t output_dim,
                                               std::span<float> transposed_input,
                                               std::span<float> transposed_output,
                                               std::span<float> output_rows) noexcept {
  if (row_count == 0u || input_dim == 0u || output_dim == 0u ||
      input_rows.size() != row_count * input_dim ||
      weights.size() != input_dim * output_dim ||
      cache.source != weights.data() ||
      cache.input_dim != input_dim ||
      cache.output_dim != output_dim ||
      transposed_input.size() < input_dim * row_count ||
      transposed_output.size() < output_dim * row_count ||
      output_rows.size() != row_count * output_dim) {
    return false;
  }

  for (size_t input_index = 0u; input_index < input_dim; ++input_index) {
    for (size_t row = 0u; row < row_count; ++row) {
      transposed_input[(input_index * row_count) + row] =
          input_rows[(row * input_dim) + input_index];
    }
  }

  if (!run_dense_batch_matmul_from_transposed_prepared(transposed_input,
                                                       row_count,
                                                       input_dim,
                                                       weights,
                                                       cache,
                                                       output_dim,
                                                       transposed_output)) {
    return false;
  }

  for (size_t row = 0u; row < row_count; ++row) {
    for (size_t output_index = 0u; output_index < output_dim; ++output_index) {
      output_rows[(row * output_dim) + output_index] =
          transposed_output[(output_index * row_count) + row];
    }
  }

  return true;
}

bool compute_dense_batch_to_transposed(std::span<const float> input_rows,
                                       const size_t row_count,
                                       const size_t input_dim,
                                       std::span<const float> weights,
                                       std::span<const float> bias,
                                       const size_t output_dim,
                                       std::span<float> transposed_input,
                                       std::span<float> transposed_output) noexcept {
  if (row_count == 0u || input_dim == 0u || output_dim == 0u ||
      input_rows.size() != row_count * input_dim ||
      weights.size() != input_dim * output_dim ||
      bias.size() != output_dim ||
      transposed_input.size() < input_dim * row_count ||
      transposed_output.size() < output_dim * row_count) {
    return false;
  }

  for (size_t input_index = 0u; input_index < input_dim; ++input_index) {
    for (size_t row = 0u; row < row_count; ++row) {
      transposed_input[(input_index * row_count) + row] =
          input_rows[(row * input_dim) + input_index];
    }
  }

  if (!run_dense_batch_matmul_from_transposed(transposed_input,
                                              row_count,
                                              input_dim,
                                              weights,
                                              output_dim,
                                              transposed_output)) {
    return false;
  }

  for (size_t output_index = 0u; output_index < output_dim; ++output_index) {
    const float bias_value = bias[output_index];
    const size_t output_base = output_index * row_count;
    for (size_t row = 0u; row < row_count; ++row) {
      transposed_output[output_base + row] += bias_value;
    }
  }

  return true;
}

bool compute_dense_batch_to_transposed_prepared(std::span<const float> input_rows,
                                                const size_t row_count,
                                                const size_t input_dim,
                                                std::span<const float> weights,
                                                const dense_weight_cache & cache,
                                                std::span<const float> bias,
                                                const size_t output_dim,
                                                std::span<float> transposed_input,
                                                std::span<float> transposed_output) noexcept {
  if (row_count == 0u || input_dim == 0u || output_dim == 0u ||
      input_rows.size() != row_count * input_dim ||
      weights.size() != input_dim * output_dim ||
      cache.source != weights.data() ||
      cache.input_dim != input_dim ||
      cache.output_dim != output_dim ||
      bias.size() != output_dim ||
      transposed_input.size() < input_dim * row_count ||
      transposed_output.size() < output_dim * row_count) {
    return false;
  }

  for (size_t input_index = 0u; input_index < input_dim; ++input_index) {
    for (size_t row = 0u; row < row_count; ++row) {
      transposed_input[(input_index * row_count) + row] =
          input_rows[(row * input_dim) + input_index];
    }
  }

  if (!run_dense_batch_matmul_from_transposed_prepared(transposed_input,
                                                       row_count,
                                                       input_dim,
                                                       weights,
                                                       cache,
                                                       output_dim,
                                                       transposed_output)) {
    return false;
  }

  for (size_t output_index = 0u; output_index < output_dim; ++output_index) {
    const float bias_value = bias[output_index];
    const size_t output_base = output_index * row_count;
    for (size_t row = 0u; row < row_count; ++row) {
      transposed_output[output_base + row] += bias_value;
    }
  }

  return true;
}

bool compute_dense_batch_from_transposed(std::span<const float> transposed_input,
                                         const size_t row_count,
                                         const size_t input_dim,
                                         std::span<const float> weights,
                                         std::span<const float> bias,
                                         const size_t output_dim,
                                         std::span<float> transposed_output,
                                         std::span<float> output_rows) noexcept {
  if (row_count == 0u || input_dim == 0u || output_dim == 0u ||
      transposed_input.size() < input_dim * row_count ||
      weights.size() != input_dim * output_dim ||
      bias.size() != output_dim ||
      transposed_output.size() < output_dim * row_count ||
      output_rows.size() != row_count * output_dim) {
    return false;
  }

  if (!run_dense_batch_matmul_from_transposed(transposed_input,
                                              row_count,
                                              input_dim,
                                              weights,
                                              output_dim,
                                              transposed_output)) {
    return false;
  }

  for (size_t row = 0u; row < row_count; ++row) {
    for (size_t output_index = 0u; output_index < output_dim; ++output_index) {
      output_rows[(row * output_dim) + output_index] =
          transposed_output[(output_index * row_count) + row] + bias[output_index];
    }
  }

  return true;
}

bool compute_dense_batch_from_transposed_prepared(std::span<const float> transposed_input,
                                                  const size_t row_count,
                                                  const size_t input_dim,
                                                  std::span<const float> weights,
                                                  const dense_weight_cache & cache,
                                                  std::span<const float> bias,
                                                  const size_t output_dim,
                                                  std::span<float> transposed_output,
                                                  std::span<float> output_rows) noexcept {
  if (row_count == 0u || input_dim == 0u || output_dim == 0u ||
      transposed_input.size() < input_dim * row_count ||
      weights.size() != input_dim * output_dim ||
      cache.source != weights.data() ||
      cache.input_dim != input_dim ||
      cache.output_dim != output_dim ||
      bias.size() != output_dim ||
      transposed_output.size() < output_dim * row_count ||
      output_rows.size() != row_count * output_dim) {
    return false;
  }

  if (!run_dense_batch_matmul_from_transposed_prepared(transposed_input,
                                                       row_count,
                                                       input_dim,
                                                       weights,
                                                       cache,
                                                       output_dim,
                                                       transposed_output)) {
    return false;
  }

  for (size_t row = 0u; row < row_count; ++row) {
    for (size_t output_index = 0u; output_index < output_dim; ++output_index) {
      output_rows[(row * output_dim) + output_index] =
          transposed_output[(output_index * row_count) + row] + bias[output_index];
    }
  }

  return true;
}

bool compute_dense_batch_from_transposed_scaled_residual_prepared(
    std::span<const float> transposed_input,
    const size_t row_count,
    const size_t input_dim,
    std::span<const float> weights,
    const dense_weight_cache & cache,
    std::span<const float> bias,
    const size_t output_dim,
    const float dense_scale,
    std::span<const float> residual_rows,
    std::span<float> transposed_output,
    std::span<float> output_rows) noexcept {
  if (row_count == 0u || input_dim == 0u || output_dim == 0u ||
      transposed_input.size() < input_dim * row_count ||
      weights.size() != input_dim * output_dim ||
      cache.source != weights.data() ||
      cache.input_dim != input_dim ||
      cache.output_dim != output_dim ||
      bias.size() != output_dim ||
      residual_rows.size() != row_count * output_dim ||
      transposed_output.size() < output_dim * row_count ||
      output_rows.size() != row_count * output_dim) {
    return false;
  }

  if (!run_dense_batch_matmul_from_transposed_prepared(transposed_input,
                                                       row_count,
                                                       input_dim,
                                                       weights,
                                                       cache,
                                                       output_dim,
                                                       transposed_output)) {
    return false;
  }

  for (size_t row = 0u; row < row_count; ++row) {
    for (size_t output_index = 0u; output_index < output_dim; ++output_index) {
      const size_t output_offset = (row * output_dim) + output_index;
      const float dense_value =
          transposed_output[(output_index * row_count) + row] + bias[output_index];
      output_rows[output_offset] =
          residual_rows[output_offset] + (dense_value * dense_scale);
    }
  }

  return true;
}

}  // namespace emel::diarization::sortformer::detail
