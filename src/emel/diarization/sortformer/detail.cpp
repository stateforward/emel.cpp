#include "emel/diarization/sortformer/detail.hpp"

#include <cstddef>

#include "emel/kernel/aarch64/actions.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"

namespace emel::diarization::sortformer::detail {

namespace {

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
              1u,
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
      .nth = 1u,
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
              1u,
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
      .nth = 1u,
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
              1u,
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
      .nth = 1u,
  };

#if defined(__aarch64__) || defined(__ARM_NEON)
  if (!emel::kernel::aarch64::detail::execute_neon_mul_mat_prepared_f32_lhs_4row(
          request,
          cache.lhs_4row.data(),
          static_cast<uint64_t>(cache.lhs_4row.size()))) {
    return false;
  }
#else
  if (!emel::kernel::detail::run_mul_mat(request)) {
    return false;
  }
#endif

  return true;
}

}  // namespace

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
