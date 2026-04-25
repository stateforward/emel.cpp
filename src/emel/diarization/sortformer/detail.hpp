#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace emel::diarization::sortformer::detail {

struct dense_weight_cache {
  const float * source = nullptr;
  size_t input_dim = 0u;
  size_t output_dim = 0u;
  std::vector<float> lhs_4row = {};
};

bool prepare_dense_weight_cache(std::span<const float> weights,
                                size_t input_dim,
                                size_t output_dim,
                                dense_weight_cache & cache) noexcept;

bool compute_dense(std::span<const float> input,
                   std::span<const float> weights,
                   std::span<const float> bias,
                   std::span<float> output) noexcept;

bool compute_dense_without_bias(std::span<const float> input,
                                std::span<const float> weights,
                                std::span<float> output) noexcept;

bool transpose_dense_input(std::span<const float> input_rows,
                           size_t row_count,
                           size_t input_dim,
                           std::span<float> transposed_input) noexcept;

bool compute_dense_batch(std::span<const float> input_rows,
                         size_t row_count,
                         size_t input_dim,
                         std::span<const float> weights,
                         std::span<const float> bias,
                         size_t output_dim,
                         std::span<float> transposed_input,
                         std::span<float> transposed_output,
                         std::span<float> output_rows) noexcept;

bool compute_dense_batch_prepared(std::span<const float> input_rows,
                                  size_t row_count,
                                  size_t input_dim,
                                  std::span<const float> weights,
                                  const dense_weight_cache & cache,
                                  std::span<const float> bias,
                                  size_t output_dim,
                                  std::span<float> transposed_input,
                                  std::span<float> transposed_output,
                                  std::span<float> output_rows) noexcept;

bool compute_dense_batch_residual_prepared(std::span<const float> input_rows,
                                           size_t row_count,
                                           size_t input_dim,
                                           std::span<const float> weights,
                                           const dense_weight_cache & cache,
                                           std::span<const float> bias,
                                           size_t output_dim,
                                           std::span<const float> residual_rows,
                                           std::span<float> transposed_input,
                                           std::span<float> transposed_output,
                                           std::span<float> output_rows) noexcept;

bool compute_dense_batch_without_bias(std::span<const float> input_rows,
                                      size_t row_count,
                                      size_t input_dim,
                                      std::span<const float> weights,
                                      size_t output_dim,
                                      std::span<float> transposed_input,
                                      std::span<float> transposed_output,
                                      std::span<float> output_rows) noexcept;

bool compute_dense_batch_without_bias_prepared(std::span<const float> input_rows,
                                               size_t row_count,
                                               size_t input_dim,
                                               std::span<const float> weights,
                                               const dense_weight_cache & cache,
                                               size_t output_dim,
                                               std::span<float> transposed_input,
                                               std::span<float> transposed_output,
                                               std::span<float> output_rows) noexcept;

bool compute_dense_batch_to_transposed(std::span<const float> input_rows,
                                       size_t row_count,
                                       size_t input_dim,
                                       std::span<const float> weights,
                                       std::span<const float> bias,
                                       size_t output_dim,
                                       std::span<float> transposed_input,
                                       std::span<float> transposed_output) noexcept;

bool compute_dense_batch_to_transposed_prepared(std::span<const float> input_rows,
                                                size_t row_count,
                                                size_t input_dim,
                                                std::span<const float> weights,
                                                const dense_weight_cache & cache,
                                                std::span<const float> bias,
                                                size_t output_dim,
                                                std::span<float> transposed_input,
                                                std::span<float> transposed_output) noexcept;

bool compute_dense_batch_from_transposed(std::span<const float> transposed_input,
                                         size_t row_count,
                                         size_t input_dim,
                                         std::span<const float> weights,
                                         std::span<const float> bias,
                                         size_t output_dim,
                                         std::span<float> transposed_output,
                                         std::span<float> output_rows) noexcept;

bool compute_dense_batch_from_transposed_prepared(std::span<const float> transposed_input,
                                                  size_t row_count,
                                                  size_t input_dim,
                                                  std::span<const float> weights,
                                                  const dense_weight_cache & cache,
                                                  std::span<const float> bias,
                                                  size_t output_dim,
                                                  std::span<float> transposed_output,
                                                  std::span<float> output_rows) noexcept;

bool compute_dense_batch_from_transposed_scaled_residual_prepared(
    std::span<const float> transposed_input,
    size_t row_count,
    size_t input_dim,
    std::span<const float> weights,
    const dense_weight_cache & cache,
    std::span<const float> bias,
    size_t output_dim,
    float dense_scale,
    std::span<const float> residual_rows,
    std::span<float> transposed_output,
    std::span<float> output_rows) noexcept;

}  // namespace emel::diarization::sortformer::detail
