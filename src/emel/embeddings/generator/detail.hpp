#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>

#include "emel/embeddings/generator/context.hpp"
#include "emel/embeddings/generator/errors.hpp"
#include "emel/embeddings/generator/events.hpp"
#include "emel/model/data.hpp"
#include "emel/model/omniembed/detail.hpp"
#include "emel/text/conditioner/events.hpp"
#include "emel/text/conditioner/errors.hpp"

namespace emel::embeddings::generator::detail {

inline constexpr std::string_view k_word_embeddings_name =
    "text_encoder.0.auto_model.embeddings.word_embeddings.weight";
inline constexpr std::string_view k_position_embeddings_name =
    "text_encoder.0.auto_model.embeddings.position_embeddings.weight";
inline constexpr std::string_view k_token_type_embeddings_name =
    "text_encoder.0.auto_model.embeddings.token_type_embeddings.weight";
inline constexpr std::string_view k_embeddings_norm_weight_name =
    "text_encoder.0.auto_model.embeddings.LayerNorm.weight";
inline constexpr std::string_view k_embeddings_norm_bias_name =
    "text_encoder.0.auto_model.embeddings.LayerNorm.bias";
inline constexpr std::string_view k_dense_weight_name = "text_encoder.2.linear.weight";
inline constexpr std::string_view k_dense_bias_name = "text_encoder.2.linear.bias";
inline constexpr std::string_view k_projection_expand_weight_name = "text_projection.expand.0.weight";
inline constexpr std::string_view k_projection_expand_bias_name = "text_projection.expand.0.bias";
inline constexpr std::string_view k_projection_expand_norm_weight_name =
    "text_projection.expand.2.weight";
inline constexpr std::string_view k_projection_expand_norm_bias_name =
    "text_projection.expand.2.bias";
inline constexpr std::string_view k_projection_residual_weight_name =
    "text_projection.residual_blocks.0.0.weight";
inline constexpr std::string_view k_projection_residual_bias_name =
    "text_projection.residual_blocks.0.0.bias";
inline constexpr std::string_view k_projection_residual_norm_weight_name =
    "text_projection.residual_blocks.0.2.weight";
inline constexpr std::string_view k_projection_residual_norm_bias_name =
    "text_projection.residual_blocks.0.2.bias";
inline constexpr std::string_view k_projection_project_weight_name =
    "text_projection.project.weight";
inline constexpr std::string_view k_projection_project_bias_name =
    "text_projection.project.bias";
inline constexpr std::string_view k_image_projection_expand_weight_name =
    "image_projection.expand.0.weight";
inline constexpr std::string_view k_image_projection_expand_bias_name =
    "image_projection.expand.0.bias";
inline constexpr std::string_view k_image_projection_expand_norm_weight_name =
    "image_projection.expand.2.weight";
inline constexpr std::string_view k_image_projection_expand_norm_bias_name =
    "image_projection.expand.2.bias";
inline constexpr std::string_view k_image_projection_residual_weight_name =
    "image_projection.residual_blocks.0.0.weight";
inline constexpr std::string_view k_image_projection_residual_bias_name =
    "image_projection.residual_blocks.0.0.bias";
inline constexpr std::string_view k_image_projection_residual_norm_weight_name =
    "image_projection.residual_blocks.0.2.weight";
inline constexpr std::string_view k_image_projection_residual_norm_bias_name =
    "image_projection.residual_blocks.0.2.bias";
inline constexpr std::string_view k_image_projection_project_weight_name =
    "image_projection.project.weight";
inline constexpr std::string_view k_image_projection_project_bias_name =
    "image_projection.project.bias";
inline constexpr std::string_view k_audio_projection_expand_weight_name =
    "audio_projection.expand.0.weight";
inline constexpr std::string_view k_audio_projection_expand_bias_name =
    "audio_projection.expand.0.bias";
inline constexpr std::string_view k_audio_projection_expand_norm_weight_name =
    "audio_projection.expand.2.weight";
inline constexpr std::string_view k_audio_projection_expand_norm_bias_name =
    "audio_projection.expand.2.bias";
inline constexpr std::string_view k_audio_projection_residual_weight_name =
    "audio_projection.residual_blocks.0.0.weight";
inline constexpr std::string_view k_audio_projection_residual_bias_name =
    "audio_projection.residual_blocks.0.0.bias";
inline constexpr std::string_view k_audio_projection_residual_norm_weight_name =
    "audio_projection.residual_blocks.0.2.weight";
inline constexpr std::string_view k_audio_projection_residual_norm_bias_name =
    "audio_projection.residual_blocks.0.2.bias";
inline constexpr std::string_view k_audio_projection_project_weight_name =
    "audio_projection.project.weight";
inline constexpr std::string_view k_audio_projection_project_bias_name =
    "audio_projection.project.bias";
inline constexpr int32_t k_audio_input_sample_rate = 16000;
inline constexpr int32_t k_audio_input_sample_count = 4000;
inline constexpr float k_sqrt_two = 1.4142135623730950488f;
inline constexpr float k_pi = 3.14159265358979323846f;
inline constexpr float k_attention_scale_default = 0.17677669529663689318f;  // 1 / sqrt(32)

template <class runtime_event_type>
constexpr decltype(auto)
unwrap_runtime_event(const runtime_event_type & ev) noexcept {
  if constexpr (requires {
                  ev.request;
                  ev.ctx;
                }) {
    return (ev);
  } else if constexpr (requires { ev.event_; }) {
    return (ev.event_);
  } else {
    return (ev);
  }
}

inline emel::error::type to_error(const error err) noexcept {
  return emel::error::cast(err);
}

inline int32_t conditioner_error_code(const emel::text::conditioner::error err) noexcept {
  return emel::text::conditioner::detail::to_local_error_code(err);
}

inline void copy_name(std::array<char, 256> & buffer, const std::string_view text) noexcept {
  std::fill(buffer.begin(), buffer.end(), '\0');
  const size_t length = std::min(buffer.size() - 1u, text.size());
  if (length != 0u) {
    std::memcpy(buffer.data(), text.data(), length);
  }
}

inline const emel::model::data::tensor_record *
find_tensor(const emel::model::data & model, const std::string_view name) noexcept {
  for (uint32_t index = 0u; index < model.n_tensors; ++index) {
    const auto & tensor = model.tensors[index];
    if (emel::model::tensor_name_view(model, tensor) == name) {
      return &tensor;
    }
  }
  return nullptr;
}

inline bool bind_vector_f32(const emel::model::data::tensor_record & tensor,
                            const int32_t expected_size,
                            action::vector_view & view_out) noexcept {
  view_out = {};
  if (tensor.data == nullptr ||
      tensor.type != static_cast<int32_t>(emel::kernel::detail::dtype_f32) ||
      tensor.n_dims <= 0 ||
      tensor.dims[0] <= 0) {
    return false;
  }

  const int32_t size = static_cast<int32_t>(tensor.dims[0]);
  const int32_t rows = tensor.n_dims > 1 ? static_cast<int32_t>(tensor.dims[1]) : 1;
  if (rows != 1 || size != expected_size) {
    return false;
  }

  view_out.tensor = &tensor;
  view_out.data = static_cast<const float *>(tensor.data);
  view_out.size = size;
  return true;
}

inline size_t dense_row_bytes(const uint8_t dtype, const int32_t cols) noexcept {
  if (cols <= 0) {
    return 0u;
  }
  if (dtype == emel::kernel::detail::dtype_f32) {
    return static_cast<size_t>(cols) * sizeof(float);
  }
  if (dtype == emel::kernel::detail::dtype_f16) {
    return static_cast<size_t>(cols) * sizeof(uint16_t);
  }
  if (dtype == emel::kernel::detail::dtype_q8_0) {
    return emel::kernel::detail::quantized_row_storage_bytes(dtype, static_cast<uint64_t>(cols));
  }
  return 0u;
}

inline bool bind_matrix(const emel::model::data::tensor_record & tensor,
                        const int32_t expected_rows,
                        const int32_t expected_cols,
                        action::matrix_view & view_out) noexcept {
  view_out = {};
  if (tensor.data == nullptr || tensor.n_dims <= 1 || tensor.dims[0] <= 0 || tensor.dims[1] <= 0) {
    return false;
  }

  const int32_t cols = static_cast<int32_t>(tensor.dims[0]);
  const int32_t rows = static_cast<int32_t>(tensor.dims[1]);
  const uint8_t dtype = static_cast<uint8_t>(tensor.type);
  const size_t row_bytes = dense_row_bytes(dtype, cols);
  if (rows != expected_rows || cols != expected_cols || row_bytes == 0u) {
    return false;
  }

  view_out.tensor = &tensor;
  view_out.data = tensor.data;
  view_out.dtype = dtype;
  view_out.rows = rows;
  view_out.cols = cols;
  view_out.row_bytes = row_bytes;
  return true;
}

inline bool bind_running_vector_f32(const emel::model::data::tensor_record & tensor,
                                    const int32_t expected_size,
                                    action::vector_view & view_out) noexcept {
  return bind_vector_f32(tensor, expected_size, view_out);
}

inline bool bind_batch_norm(const emel::model::data::tensor_record & weight,
                            const emel::model::data::tensor_record & bias,
                            const emel::model::data::tensor_record & running_mean,
                            const emel::model::data::tensor_record & running_var,
                            const int32_t expected_channels,
                            action::batch_norm_view & view_out) noexcept {
  view_out = {};
  if (!bind_vector_f32(weight, expected_channels, view_out.weight) ||
      !bind_vector_f32(bias, expected_channels, view_out.bias) ||
      !bind_running_vector_f32(running_mean, expected_channels, view_out.running_mean) ||
      !bind_running_vector_f32(running_var, expected_channels, view_out.running_var)) {
    return false;
  }
  view_out.channels = expected_channels;
  return true;
}

inline bool bind_conv_f16_hwio(const emel::model::data::tensor_record & tensor,
                               const int32_t expected_kernel_h,
                               const int32_t expected_kernel_w,
                               const int32_t expected_input_channels,
                               const int32_t expected_output_channels,
                               action::conv2d_view & view_out) noexcept {
  view_out = {};
  if (tensor.data == nullptr ||
      tensor.type != static_cast<int32_t>(emel::kernel::detail::dtype_f16) ||
      tensor.n_dims != 4 ||
      tensor.dims[0] != expected_kernel_w ||
      tensor.dims[1] != expected_kernel_h ||
      tensor.dims[2] != expected_input_channels ||
      tensor.dims[3] != expected_output_channels) {
    return false;
  }

  view_out.tensor = &tensor;
  view_out.data = static_cast<const uint16_t *>(tensor.data);
  view_out.kernel_w = expected_kernel_w;
  view_out.kernel_h = expected_kernel_h;
  view_out.input_channels = expected_input_channels;
  view_out.output_channels = expected_output_channels;
  return true;
}

inline bool bind_pointwise_f16(const emel::model::data::tensor_record & tensor,
                               const int32_t expected_rows,
                               const int32_t expected_cols,
                               action::matrix_view & view_out) noexcept {
  view_out = {};
  if (tensor.data == nullptr ||
      tensor.type != static_cast<int32_t>(emel::kernel::detail::dtype_f16) ||
      tensor.n_dims != 4 ||
      tensor.dims[0] != 1 ||
      tensor.dims[1] != 1 ||
      tensor.dims[2] != expected_cols ||
      tensor.dims[3] != expected_rows) {
    return false;
  }

  view_out.tensor = &tensor;
  view_out.data = tensor.data;
  view_out.dtype = static_cast<uint8_t>(tensor.type);
  view_out.rows = expected_rows;
  view_out.cols = expected_cols;
  view_out.row_bytes = static_cast<size_t>(expected_cols) * sizeof(uint16_t);
  return true;
}

inline bool make_layer_name(const int32_t layer_index,
                            const std::string_view suffix,
                            std::array<char, 256> & buffer,
                            std::string_view & name_out) noexcept {
  const int written = std::snprintf(buffer.data(),
                                    buffer.size(),
                                    "text_encoder.0.auto_model.encoder.layer.%d.%.*s",
                                    layer_index,
                                    static_cast<int>(suffix.size()),
                                    suffix.data());
  if (written <= 0 || static_cast<size_t>(written) >= buffer.size()) {
    name_out = {};
    return false;
  }
  name_out = std::string_view{buffer.data(), static_cast<size_t>(written)};
  return true;
}

inline bool make_image_block_name(const int32_t stage_index,
                                  const int32_t block_index,
                                  const std::string_view suffix,
                                  std::array<char, 256> & buffer,
                                  std::string_view & name_out) noexcept {
  const int written = std::snprintf(buffer.data(),
                                    buffer.size(),
                                    "image_encoder.blocks.%d.%d.%.*s",
                                    stage_index,
                                    block_index,
                                    static_cast<int>(suffix.size()),
                                    suffix.data());
  if (written <= 0 || static_cast<size_t>(written) >= buffer.size()) {
    name_out = {};
    return false;
  }
  name_out = std::string_view{buffer.data(), static_cast<size_t>(written)};
  return true;
}

inline bool bind_image_stage0(const emel::model::data & model,
                              action::edge_residual_runtime & runtime_out) noexcept {
  runtime_out = {};

  const auto * conv_exp = find_tensor(model, "image_encoder.blocks.0.0.conv_exp.weight");
  const auto * bn1_weight = find_tensor(model, "image_encoder.blocks.0.0.bn1.weight");
  const auto * bn1_bias = find_tensor(model, "image_encoder.blocks.0.0.bn1.bias");
  const auto * bn1_mean = find_tensor(model, "image_encoder.blocks.0.0.bn1.running_mean");
  const auto * bn1_var = find_tensor(model, "image_encoder.blocks.0.0.bn1.running_var");
  const auto * conv_pwl = find_tensor(model, "image_encoder.blocks.0.0.conv_pwl.weight");
  const auto * bn2_weight = find_tensor(model, "image_encoder.blocks.0.0.bn2.weight");
  const auto * bn2_bias = find_tensor(model, "image_encoder.blocks.0.0.bn2.bias");
  const auto * bn2_mean = find_tensor(model, "image_encoder.blocks.0.0.bn2.running_mean");
  const auto * bn2_var = find_tensor(model, "image_encoder.blocks.0.0.bn2.running_var");
  if (conv_exp == nullptr ||
      bn1_weight == nullptr ||
      bn1_bias == nullptr ||
      bn1_mean == nullptr ||
      bn1_var == nullptr ||
      conv_pwl == nullptr ||
      bn2_weight == nullptr ||
      bn2_bias == nullptr ||
      bn2_mean == nullptr ||
      bn2_var == nullptr) {
    return false;
  }

  constexpr int32_t input_channels = 32;
  constexpr int32_t expanded_channels = 128;
  constexpr int32_t output_channels = 48;
  if (!bind_conv_f16_hwio(
          *conv_exp, 3, 3, input_channels, expanded_channels, runtime_out.conv_exp) ||
      !bind_batch_norm(
          *bn1_weight, *bn1_bias, *bn1_mean, *bn1_var, expanded_channels, runtime_out.bn1) ||
      !bind_pointwise_f16(*conv_pwl, output_channels, expanded_channels, runtime_out.conv_pwl) ||
      !bind_batch_norm(
          *bn2_weight, *bn2_bias, *bn2_mean, *bn2_var, output_channels, runtime_out.bn2)) {
    return false;
  }

  runtime_out.input_channels = input_channels;
  runtime_out.output_channels = output_channels;
  runtime_out.stride = 2;
  runtime_out.ready = true;
  return true;
}

inline bool bind_image_ui_block(const emel::model::data & model,
                                const int32_t stage_index,
                                const int32_t block_index,
                                const int32_t stride,
                                const int32_t input_channels,
                                action::universal_inverted_runtime & runtime_out) noexcept {
  runtime_out = {};

  std::array<char, 256> buffer = {};
  std::string_view name = {};
  const auto find_block_tensor = [&](const std::string_view suffix) noexcept
      -> const emel::model::data::tensor_record * {
    return make_image_block_name(stage_index, block_index, suffix, buffer, name) ?
        find_tensor(model, name) :
        nullptr;
  };

  const auto * pw_exp = find_block_tensor("pw_exp.conv.weight");
  const auto * pw_exp_bn_weight = find_block_tensor("pw_exp.bn.weight");
  const auto * pw_exp_bn_bias = find_block_tensor("pw_exp.bn.bias");
  const auto * pw_exp_bn_mean = find_block_tensor("pw_exp.bn.running_mean");
  const auto * pw_exp_bn_var = find_block_tensor("pw_exp.bn.running_var");
  const auto * pw_proj = find_block_tensor("pw_proj.conv.weight");
  const auto * pw_proj_bn_weight = find_block_tensor("pw_proj.bn.weight");
  const auto * pw_proj_bn_bias = find_block_tensor("pw_proj.bn.bias");
  const auto * pw_proj_bn_mean = find_block_tensor("pw_proj.bn.running_mean");
  const auto * pw_proj_bn_var = find_block_tensor("pw_proj.bn.running_var");
  if (pw_exp == nullptr ||
      pw_exp_bn_weight == nullptr ||
      pw_exp_bn_bias == nullptr ||
      pw_exp_bn_mean == nullptr ||
      pw_exp_bn_var == nullptr ||
      pw_proj == nullptr ||
      pw_proj_bn_weight == nullptr ||
      pw_proj_bn_bias == nullptr ||
      pw_proj_bn_mean == nullptr ||
      pw_proj_bn_var == nullptr ||
      pw_exp->n_dims != 4 ||
      pw_proj->n_dims != 4) {
    return false;
  }

  const int32_t expanded_channels = static_cast<int32_t>(pw_exp->dims[3]);
  const int32_t output_channels = static_cast<int32_t>(pw_proj->dims[3]);
  if (expanded_channels <= 0 || output_channels <= 0) {
    return false;
  }

  const auto * dw_start = find_block_tensor("dw_start.conv.weight");
  const auto * dw_start_bn_weight = find_block_tensor("dw_start.bn.weight");
  const auto * dw_start_bn_bias = find_block_tensor("dw_start.bn.bias");
  const auto * dw_start_bn_mean = find_block_tensor("dw_start.bn.running_mean");
  const auto * dw_start_bn_var = find_block_tensor("dw_start.bn.running_var");
  const auto * dw_mid = find_block_tensor("dw_mid.conv.weight");
  const auto * dw_mid_bn_weight = find_block_tensor("dw_mid.bn.weight");
  const auto * dw_mid_bn_bias = find_block_tensor("dw_mid.bn.bias");
  const auto * dw_mid_bn_mean = find_block_tensor("dw_mid.bn.running_mean");
  const auto * dw_mid_bn_var = find_block_tensor("dw_mid.bn.running_var");

  const bool has_dw_start = dw_start != nullptr;
  const bool has_dw_mid = dw_mid != nullptr;
  const int32_t dw_start_stride = has_dw_start && !has_dw_mid ? stride : 1;
  const int32_t dw_mid_stride = has_dw_mid ? stride : 1;

  if (has_dw_start &&
      (dw_start_bn_weight == nullptr ||
       dw_start_bn_bias == nullptr ||
       dw_start_bn_mean == nullptr ||
       dw_start_bn_var == nullptr ||
       !bind_conv_f16_hwio(*dw_start,
                           static_cast<int32_t>(dw_start->dims[1]),
                           static_cast<int32_t>(dw_start->dims[0]),
                           1,
                           input_channels,
                           runtime_out.dw_start) ||
       !bind_batch_norm(*dw_start_bn_weight,
                        *dw_start_bn_bias,
                        *dw_start_bn_mean,
                        *dw_start_bn_var,
                        input_channels,
                        runtime_out.dw_start_bn))) {
    return false;
  }

  if (has_dw_mid &&
      (dw_mid_bn_weight == nullptr ||
       dw_mid_bn_bias == nullptr ||
       dw_mid_bn_mean == nullptr ||
       dw_mid_bn_var == nullptr ||
       !bind_conv_f16_hwio(*dw_mid,
                           static_cast<int32_t>(dw_mid->dims[1]),
                           static_cast<int32_t>(dw_mid->dims[0]),
                           1,
                           expanded_channels,
                           runtime_out.dw_mid) ||
       !bind_batch_norm(*dw_mid_bn_weight,
                        *dw_mid_bn_bias,
                        *dw_mid_bn_mean,
                        *dw_mid_bn_var,
                        expanded_channels,
                        runtime_out.dw_mid_bn))) {
    return false;
  }

  if (!bind_pointwise_f16(*pw_exp, expanded_channels, input_channels, runtime_out.pw_exp) ||
      !bind_batch_norm(*pw_exp_bn_weight,
                       *pw_exp_bn_bias,
                       *pw_exp_bn_mean,
                       *pw_exp_bn_var,
                       expanded_channels,
                       runtime_out.pw_exp_bn) ||
      !bind_pointwise_f16(*pw_proj, output_channels, expanded_channels, runtime_out.pw_proj) ||
      !bind_batch_norm(*pw_proj_bn_weight,
                       *pw_proj_bn_bias,
                       *pw_proj_bn_mean,
                       *pw_proj_bn_var,
                       output_channels,
                       runtime_out.pw_proj_bn)) {
    return false;
  }

  runtime_out.has_dw_start = has_dw_start;
  runtime_out.has_dw_mid = has_dw_mid;
  runtime_out.has_skip = stride == 1 && input_channels == output_channels;
  runtime_out.input_channels = input_channels;
  runtime_out.expanded_channels = expanded_channels;
  runtime_out.output_channels = output_channels;
  runtime_out.stride = stride;
  runtime_out.ready = true;
  (void) dw_start_stride;
  (void) dw_mid_stride;
  return true;
}

inline bool bind_image_conv_norm(const emel::model::data & model,
                                 const std::string_view conv_name,
                                 const std::string_view norm_prefix,
                                 const int32_t input_channels,
                                 const int32_t output_channels,
                                 action::conv_norm_runtime & runtime_out) noexcept {
  runtime_out = {};

  const auto * conv = find_tensor(model, conv_name);
  std::array<char, 256> buffer = {};
  const auto make_norm_name = [&](const std::string_view suffix) noexcept {
    const int written = std::snprintf(buffer.data(),
                                      buffer.size(),
                                      "%.*s%.*s",
                                      static_cast<int>(norm_prefix.size()),
                                      norm_prefix.data(),
                                      static_cast<int>(suffix.size()),
                                      suffix.data());
    return written > 0 && static_cast<size_t>(written) < buffer.size() ?
        std::string_view{buffer.data(), static_cast<size_t>(written)} :
        std::string_view{};
  };

  const auto * norm_weight = find_tensor(model, make_norm_name(".weight"));
  const auto * norm_bias = find_tensor(model, make_norm_name(".bias"));
  const auto * norm_mean = find_tensor(model, make_norm_name(".running_mean"));
  const auto * norm_var = find_tensor(model, make_norm_name(".running_var"));
  if (conv == nullptr ||
      norm_weight == nullptr ||
      norm_bias == nullptr ||
      norm_mean == nullptr ||
      norm_var == nullptr) {
    return false;
  }

  if (!bind_pointwise_f16(*conv, output_channels, input_channels, runtime_out.conv) ||
      !bind_batch_norm(*norm_weight,
                       *norm_bias,
                       *norm_mean,
                       *norm_var,
                       output_channels,
                       runtime_out.norm)) {
    return false;
  }

  runtime_out.input_channels = input_channels;
  runtime_out.output_channels = output_channels;
  return true;
}

inline bool bind_layer(const emel::model::data & model,
                       const int32_t layer_index,
                       const int32_t hidden_size,
                       const int32_t intermediate_size,
                       action::layer_weights & layer_out) noexcept {
  layer_out = {};
  std::array<char, 256> buffer = {};
  std::string_view name = {};
  const auto find_layer_tensor = [&](const std::string_view suffix) noexcept
      -> const emel::model::data::tensor_record * {
    return make_layer_name(layer_index, suffix, buffer, name) ? find_tensor(model, name) : nullptr;
  };

  const auto * attention_query = find_layer_tensor("attention.self.query.weight");
  const auto * attention_query_bias = find_layer_tensor("attention.self.query.bias");
  const auto * attention_key = find_layer_tensor("attention.self.key.weight");
  const auto * attention_key_bias = find_layer_tensor("attention.self.key.bias");
  const auto * attention_value = find_layer_tensor("attention.self.value.weight");
  const auto * attention_value_bias = find_layer_tensor("attention.self.value.bias");
  const auto * attention_output = find_layer_tensor("attention.output.dense.weight");
  const auto * attention_output_bias = find_layer_tensor("attention.output.dense.bias");
  const auto * attention_output_norm_weight =
      find_layer_tensor("attention.output.LayerNorm.weight");
  const auto * attention_output_norm_bias =
      find_layer_tensor("attention.output.LayerNorm.bias");
  const auto * intermediate = find_layer_tensor("intermediate.dense.weight");
  const auto * intermediate_bias = find_layer_tensor("intermediate.dense.bias");
  const auto * output = find_layer_tensor("output.dense.weight");
  const auto * output_bias = find_layer_tensor("output.dense.bias");
  const auto * output_norm_weight = find_layer_tensor("output.LayerNorm.weight");
  const auto * output_norm_bias = find_layer_tensor("output.LayerNorm.bias");

  if (attention_query == nullptr ||
      attention_query_bias == nullptr ||
      attention_key == nullptr ||
      attention_key_bias == nullptr ||
      attention_value == nullptr ||
      attention_value_bias == nullptr ||
      attention_output == nullptr ||
      attention_output_bias == nullptr ||
      attention_output_norm_weight == nullptr ||
      attention_output_norm_bias == nullptr ||
      intermediate == nullptr ||
      intermediate_bias == nullptr ||
      output == nullptr ||
      output_bias == nullptr ||
      output_norm_weight == nullptr ||
      output_norm_bias == nullptr) {
    return false;
  }

  return bind_matrix(*attention_query, hidden_size, hidden_size, layer_out.attention_query) &&
      bind_vector_f32(*attention_query_bias, hidden_size, layer_out.attention_query_bias) &&
      bind_matrix(*attention_key, hidden_size, hidden_size, layer_out.attention_key) &&
      bind_vector_f32(*attention_key_bias, hidden_size, layer_out.attention_key_bias) &&
      bind_matrix(*attention_value, hidden_size, hidden_size, layer_out.attention_value) &&
      bind_vector_f32(*attention_value_bias, hidden_size, layer_out.attention_value_bias) &&
      bind_matrix(*attention_output, hidden_size, hidden_size, layer_out.attention_output) &&
      bind_vector_f32(*attention_output_bias, hidden_size, layer_out.attention_output_bias) &&
      bind_vector_f32(
          *attention_output_norm_weight, hidden_size, layer_out.attention_output_norm_weight) &&
      bind_vector_f32(
          *attention_output_norm_bias, hidden_size, layer_out.attention_output_norm_bias) &&
      bind_matrix(*intermediate, intermediate_size, hidden_size, layer_out.intermediate) &&
      bind_vector_f32(*intermediate_bias, intermediate_size, layer_out.intermediate_bias) &&
      bind_matrix(*output, hidden_size, intermediate_size, layer_out.output) &&
      bind_vector_f32(*output_bias, hidden_size, layer_out.output_bias) &&
      bind_vector_f32(*output_norm_weight, hidden_size, layer_out.output_norm_weight) &&
      bind_vector_f32(*output_norm_bias, hidden_size, layer_out.output_norm_bias);
}

inline bool bind_projection_runtime(
    const emel::model::data & model,
    const std::string_view expand_weight_name,
    const std::string_view expand_bias_name,
    const std::string_view expand_norm_weight_name,
    const std::string_view expand_norm_bias_name,
    const std::string_view residual_weight_name,
    const std::string_view residual_bias_name,
    const std::string_view residual_norm_weight_name,
    const std::string_view residual_norm_bias_name,
    const std::string_view project_weight_name,
    const std::string_view project_bias_name,
    const int32_t input_size,
    const int32_t hidden_size,
    const int32_t output_size,
    action::projection_runtime & runtime_out) noexcept {
  runtime_out = {};

  const auto * expand_weight = find_tensor(model, expand_weight_name);
  const auto * expand_bias = find_tensor(model, expand_bias_name);
  const auto * expand_norm_weight = find_tensor(model, expand_norm_weight_name);
  const auto * expand_norm_bias = find_tensor(model, expand_norm_bias_name);
  const auto * residual_weight = find_tensor(model, residual_weight_name);
  const auto * residual_bias = find_tensor(model, residual_bias_name);
  const auto * residual_norm_weight = find_tensor(model, residual_norm_weight_name);
  const auto * residual_norm_bias = find_tensor(model, residual_norm_bias_name);
  const auto * project_weight = find_tensor(model, project_weight_name);
  const auto * project_bias = find_tensor(model, project_bias_name);

  if (expand_weight == nullptr ||
      expand_bias == nullptr ||
      expand_norm_weight == nullptr ||
      expand_norm_bias == nullptr ||
      residual_weight == nullptr ||
      residual_bias == nullptr ||
      residual_norm_weight == nullptr ||
      residual_norm_bias == nullptr ||
      project_weight == nullptr ||
      project_bias == nullptr) {
    return false;
  }

  if (!bind_matrix(*expand_weight, hidden_size, input_size, runtime_out.expand) ||
      !bind_vector_f32(*expand_bias, hidden_size, runtime_out.expand_bias) ||
      !bind_vector_f32(*expand_norm_weight, hidden_size, runtime_out.expand_norm_weight) ||
      !bind_vector_f32(*expand_norm_bias, hidden_size, runtime_out.expand_norm_bias) ||
      !bind_matrix(*residual_weight, hidden_size, hidden_size, runtime_out.residual) ||
      !bind_vector_f32(*residual_bias, hidden_size, runtime_out.residual_bias) ||
      !bind_vector_f32(
          *residual_norm_weight, hidden_size, runtime_out.residual_norm_weight) ||
      !bind_vector_f32(*residual_norm_bias, hidden_size, runtime_out.residual_norm_bias) ||
      !bind_matrix(*project_weight, output_size, hidden_size, runtime_out.project) ||
      !bind_vector_f32(*project_bias, output_size, runtime_out.project_bias)) {
    return false;
  }

  runtime_out.input_size = input_size;
  runtime_out.hidden_size = hidden_size;
  runtime_out.output_size = output_size;
  return true;
}

inline bool build_text_runtime(const emel::model::data & model,
                               action::text_runtime & runtime_out) noexcept {
  runtime_out = {};
  if (emel::model::architecture_name_view(model) != "omniembed") {
    return false;
  }

  const auto * word_embeddings = find_tensor(model, k_word_embeddings_name);
  const auto * position_embeddings = find_tensor(model, k_position_embeddings_name);
  const auto * token_type_embeddings = find_tensor(model, k_token_type_embeddings_name);
  const auto * embeddings_norm_weight = find_tensor(model, k_embeddings_norm_weight_name);
  const auto * embeddings_norm_bias = find_tensor(model, k_embeddings_norm_bias_name);
  const auto * dense_weight = find_tensor(model, k_dense_weight_name);
  const auto * dense_bias = find_tensor(model, k_dense_bias_name);
  const auto * projection_expand_weight = find_tensor(model, k_projection_expand_weight_name);

  if (word_embeddings == nullptr ||
      position_embeddings == nullptr ||
      token_type_embeddings == nullptr ||
      embeddings_norm_weight == nullptr ||
      embeddings_norm_bias == nullptr ||
      dense_weight == nullptr ||
      dense_bias == nullptr ||
      projection_expand_weight == nullptr) {
    return false;
  }

  const int32_t hidden_size = word_embeddings->n_dims > 0 ? static_cast<int32_t>(word_embeddings->dims[0])
                                                           : 0;
  const int32_t vocab_size = word_embeddings->n_dims > 1 ? static_cast<int32_t>(word_embeddings->dims[1])
                                                         : 0;
  const int32_t max_positions =
      position_embeddings->n_dims > 1 ? static_cast<int32_t>(position_embeddings->dims[1]) : 0;
  const int32_t token_type_count =
      token_type_embeddings->n_dims > 1 ? static_cast<int32_t>(token_type_embeddings->dims[1]) : 0;
  const int32_t output_size =
      dense_weight->n_dims > 1 ? static_cast<int32_t>(dense_weight->dims[1]) : 0;
  const int32_t intermediate_size =
      projection_expand_weight->n_dims > 0 ? static_cast<int32_t>(projection_expand_weight->dims[1])
                                           : 0;

  if (hidden_size <= 0 ||
      vocab_size <= 0 ||
      max_positions <= 0 ||
      token_type_count <= 0 ||
      output_size <= 0 ||
      intermediate_size <= 0 ||
      model.params.n_embd_out <= 0) {
    return false;
  }

  runtime_out.hidden_size = hidden_size;
  runtime_out.max_positions = max_positions;
  runtime_out.output_size = output_size;
  runtime_out.projection_hidden_size = intermediate_size;
  runtime_out.shared_embedding_size = model.params.n_embd_out;
  runtime_out.attention_head_count = 12;
  runtime_out.attention_head_dim = hidden_size / runtime_out.attention_head_count;
  if (runtime_out.attention_head_count <= 0 ||
      runtime_out.attention_head_dim <= 0 ||
      runtime_out.attention_head_dim * runtime_out.attention_head_count != hidden_size) {
    return false;
  }

  if (!bind_matrix(*word_embeddings, vocab_size, hidden_size, runtime_out.word_embeddings) ||
      !bind_matrix(*position_embeddings, max_positions, hidden_size, runtime_out.position_embeddings) ||
      !bind_matrix(*token_type_embeddings,
                   token_type_count,
                   hidden_size,
                   runtime_out.token_type_embeddings) ||
      !bind_vector_f32(*embeddings_norm_weight, hidden_size, runtime_out.embeddings_norm_weight) ||
      !bind_vector_f32(*embeddings_norm_bias, hidden_size, runtime_out.embeddings_norm_bias) ||
      !bind_matrix(*dense_weight, output_size, hidden_size, runtime_out.dense) ||
      !bind_vector_f32(*dense_bias, output_size, runtime_out.dense_bias) ||
      !bind_projection_runtime(model,
                               k_projection_expand_weight_name,
                               k_projection_expand_bias_name,
                               k_projection_expand_norm_weight_name,
                               k_projection_expand_norm_bias_name,
                               k_projection_residual_weight_name,
                               k_projection_residual_bias_name,
                               k_projection_residual_norm_weight_name,
                               k_projection_residual_norm_bias_name,
                               k_projection_project_weight_name,
                               k_projection_project_bias_name,
                               output_size,
                               intermediate_size,
                               runtime_out.shared_embedding_size,
                               runtime_out.projection)) {
    return false;
  }

  int32_t layer_count = 0;
  for (; layer_count < action::k_max_text_layers; ++layer_count) {
    action::layer_weights layer = {};
    if (!bind_layer(model, layer_count, hidden_size, 4 * hidden_size, layer)) {
      break;
    }
    runtime_out.layers[static_cast<size_t>(layer_count)] = layer;
  }

  if (layer_count <= 0) {
    return false;
  }

  runtime_out.layer_count = layer_count;
  runtime_out.intermediate_size = 4 * hidden_size;
  runtime_out.ready = true;
  return true;
}

inline bool build_image_runtime(const emel::model::data & model,
                               action::image_runtime & runtime_out) noexcept {
  runtime_out = {};
  if (emel::model::architecture_name_view(model) != "omniembed") {
    return true;
  }

  if (model.meta.clip_vision_data.embedding_length <= 0 ||
      model.meta.clip_vision_data.projection_dim != model.params.n_embd_out ||
      model.meta.clip_vision_data.preproc_image_size <= 0 ||
      model.meta.clip_vision_data.image_mean_count != 3u ||
      model.meta.clip_vision_data.image_std_count != 3u ||
      emel::model::metadata_string_view(model.meta, model.meta.clip_vision_data.encoder_name) !=
          "mobilenetv4_conv_medium.e180_r384_in12k") {
    return true;
  }

  const int32_t input_size = model.meta.clip_vision_data.preproc_image_size;
  if (input_size <= 0) {
    return true;
  }

  const auto * stem = find_tensor(model, "image_encoder.conv_stem.weight");
  const auto * stem_bn_weight = find_tensor(model, "image_encoder.bn1.weight");
  const auto * stem_bn_bias = find_tensor(model, "image_encoder.bn1.bias");
  const auto * stem_bn_mean = find_tensor(model, "image_encoder.bn1.running_mean");
  const auto * stem_bn_var = find_tensor(model, "image_encoder.bn1.running_var");
  if (stem == nullptr ||
      stem_bn_weight == nullptr ||
      stem_bn_bias == nullptr ||
      stem_bn_mean == nullptr ||
      stem_bn_var == nullptr ||
      !bind_conv_f16_hwio(*stem, 3, 3, 3, 32, runtime_out.stem) ||
      !bind_batch_norm(
          *stem_bn_weight, *stem_bn_bias, *stem_bn_mean, *stem_bn_var, 32, runtime_out.stem_bn) ||
      !bind_image_stage0(model, runtime_out.stage0)) {
    return true;
  }

  constexpr std::array<int32_t, 3> k_stage_counts = {2, 8, 11};
  constexpr std::array<int32_t, 3> k_stage_strides = {2, 2, 2};
  runtime_out.block_count = 0;
  int32_t channels = runtime_out.stage0.output_channels;
  int32_t spatial = input_size / 4;
  int32_t max_feature_elements = 32 * (input_size / 2) * (input_size / 2);
  for (int32_t stage = 1; stage <= 3; ++stage) {
    const int32_t stage_block_count = k_stage_counts[static_cast<size_t>(stage - 1)];
    for (int32_t block_index = 0; block_index < stage_block_count; ++block_index) {
      const int32_t stride = block_index == 0 ? k_stage_strides[static_cast<size_t>(stage - 1)] : 1;
      auto & block = runtime_out.blocks[static_cast<size_t>(runtime_out.block_count)];
      if (!bind_image_ui_block(model, stage, block_index, stride, channels, block)) {
        return true;
      }
      ++runtime_out.block_count;

      if (block.expanded_channels > 0) {
        const int32_t mid_spatial = stride > 1 && block.has_dw_mid ? spatial / stride : spatial;
        max_feature_elements = std::max(
            max_feature_elements, block.expanded_channels * mid_spatial * mid_spatial);
      }

      spatial = stride > 1 ? spatial / stride : spatial;
      channels = block.output_channels;
      max_feature_elements = std::max(max_feature_elements, channels * spatial * spatial);
    }
  }

  if (!bind_image_conv_norm(
          model, "image_encoder.blocks.4.0.conv.weight", "image_encoder.blocks.4.0.bn1", channels, 960, runtime_out.stage4) ||
      !bind_image_conv_norm(
          model, "image_encoder.conv_head.weight", "image_encoder.norm_head", 960, 1280, runtime_out.head) ||
      !bind_projection_runtime(model,
                               k_image_projection_expand_weight_name,
                               k_image_projection_expand_bias_name,
                               k_image_projection_expand_norm_weight_name,
                               k_image_projection_expand_norm_bias_name,
                               k_image_projection_residual_weight_name,
                               k_image_projection_residual_bias_name,
                               k_image_projection_residual_norm_weight_name,
                               k_image_projection_residual_norm_bias_name,
                               k_image_projection_project_weight_name,
                               k_image_projection_project_bias_name,
                               model.meta.clip_vision_data.embedding_length,
                               1920,
                               model.params.n_embd_out,
                               runtime_out.projection)) {
    return true;
  }

  spatial = input_size / 32;
  max_feature_elements = std::max(max_feature_elements, 960 * spatial * spatial);
  runtime_out.input_size = input_size;
  runtime_out.embedding_size = model.meta.clip_vision_data.embedding_length;
  runtime_out.feature_buffer_elements = max_feature_elements;
  runtime_out.ready = runtime_out.embedding_size > 0 &&
      runtime_out.feature_buffer_elements > 0 &&
      runtime_out.block_count == 21;
  return true;
}

struct audio_block_config {
  int32_t input_channels = 0;
  int32_t expanded_channels = 0;
  int32_t output_channels = 0;
  int32_t kernel_size = 0;
  int32_t stride = 1;
  bool use_se = false;
  bool use_hardswish = false;
};

inline int32_t make_divisible_8(const int32_t value) noexcept;
inline int32_t output_dim_same(const int32_t input,
                               const int32_t kernel,
                               const int32_t stride) noexcept;

inline bool make_audio_feature_name(const int32_t feature_index,
                                    const std::string_view suffix,
                                    std::array<char, 256> & buffer,
                                    std::string_view & name_out) noexcept {
  const int written = std::snprintf(
      buffer.data(),
      buffer.size(),
      "audio_encoder.features.%d.%s",
      feature_index,
      std::string{suffix}.c_str());
  if (written <= 0 || static_cast<size_t>(written) >= buffer.size()) {
    return false;
  }
  name_out = std::string_view{buffer.data(), static_cast<size_t>(written)};
  return true;
}

inline bool bind_audio_se(const emel::model::data & model,
                          const int32_t feature_index,
                          const std::string_view prefix,
                          const int32_t input_size,
                          const int32_t hidden_size,
                          action::squeeze_excitation_runtime & runtime_out) noexcept {
  runtime_out = {};

  std::array<char, 256> buffer = {};
  std::string_view name = {};
  const auto find_feature_tensor = [&](const std::string_view suffix) noexcept
      -> const emel::model::data::tensor_record * {
    return make_audio_feature_name(feature_index, suffix, buffer, name) ? find_tensor(model, name) :
        nullptr;
  };

  const auto * fc1_weight =
      find_feature_tensor(std::string{prefix} + ".conc_se_layers.0.fc1.weight");
  const auto * fc1_bias =
      find_feature_tensor(std::string{prefix} + ".conc_se_layers.0.fc1.bias");
  const auto * fc2_weight =
      find_feature_tensor(std::string{prefix} + ".conc_se_layers.0.fc2.weight");
  const auto * fc2_bias =
      find_feature_tensor(std::string{prefix} + ".conc_se_layers.0.fc2.bias");
  if (fc1_weight == nullptr ||
      fc1_bias == nullptr ||
      fc2_weight == nullptr ||
      fc2_bias == nullptr) {
    return false;
  }

  if (!bind_matrix(*fc1_weight, hidden_size, input_size, runtime_out.fc1) ||
      !bind_vector_f32(*fc1_bias, hidden_size, runtime_out.fc1_bias) ||
      !bind_matrix(*fc2_weight, input_size, hidden_size, runtime_out.fc2) ||
      !bind_vector_f32(*fc2_bias, input_size, runtime_out.fc2_bias)) {
    return false;
  }

  runtime_out.input_size = input_size;
  runtime_out.hidden_size = hidden_size;
  runtime_out.ready = true;
  return true;
}

inline bool bind_audio_block(const emel::model::data & model,
                             const int32_t feature_index,
                             const audio_block_config & config,
                             action::audio_inverted_residual_runtime & runtime_out) noexcept {
  runtime_out = {};

  std::array<char, 256> buffer = {};
  std::string_view name = {};
  const auto find_feature_tensor = [&](const std::string_view suffix) noexcept
      -> const emel::model::data::tensor_record * {
    return make_audio_feature_name(feature_index, suffix, buffer, name) ? find_tensor(model, name) :
        nullptr;
  };
  const auto bind_feature_bn = [&](const std::string_view prefix,
                                   const int32_t channels,
                                   action::batch_norm_view & out) noexcept {
    const auto * weight = find_feature_tensor(std::string{prefix} + ".1.weight");
    const auto * bias = find_feature_tensor(std::string{prefix} + ".1.bias");
    const auto * mean = find_feature_tensor(std::string{prefix} + ".1.running_mean");
    const auto * var = find_feature_tensor(std::string{prefix} + ".1.running_var");
    if (weight == nullptr || bias == nullptr || mean == nullptr || var == nullptr) {
      return false;
    }
    return bind_batch_norm(*weight, *bias, *mean, *var, channels, out);
  };

  const bool has_expand = config.expanded_channels != config.input_channels;
  const char * depthwise_prefix = has_expand ? "block.1" : "block.0";
  const char * se_prefix = has_expand ? "block.2" : "block.1";
  const char * project_prefix = config.use_se ?
      (has_expand ? "block.3" : "block.2") :
      (has_expand ? "block.2" : "block.1");

  runtime_out.has_expand = has_expand;
  runtime_out.has_se = config.use_se;
  runtime_out.has_skip = config.stride == 1 && config.input_channels == config.output_channels;
  runtime_out.use_hardswish = config.use_hardswish;
  runtime_out.input_channels = config.input_channels;
  runtime_out.expanded_channels = config.expanded_channels;
  runtime_out.output_channels = config.output_channels;
  runtime_out.kernel_size = config.kernel_size;
  runtime_out.stride = config.stride;

  if (has_expand) {
    const auto * expand_weight = find_feature_tensor("block.0.0.weight");
    if (expand_weight == nullptr ||
        !bind_pointwise_f16(
            *expand_weight, config.expanded_channels, config.input_channels, runtime_out.expand) ||
        !bind_feature_bn("block.0", config.expanded_channels, runtime_out.expand_bn)) {
      return false;
    }
  }

  const auto * depthwise_weight =
      find_feature_tensor(std::string{depthwise_prefix} + ".0.weight");
  if (depthwise_weight == nullptr ||
      !bind_conv_f16_hwio(*depthwise_weight,
                          config.kernel_size,
                          config.kernel_size,
                          1,
                          config.expanded_channels,
                          runtime_out.depthwise) ||
      !bind_feature_bn(depthwise_prefix, config.expanded_channels, runtime_out.depthwise_bn)) {
    return false;
  }

  if (config.use_se &&
      !bind_audio_se(model,
                     feature_index,
                     se_prefix,
                     config.expanded_channels,
                     make_divisible_8(config.expanded_channels / 4),
                     runtime_out.se)) {
    return false;
  }

  const auto * project_weight = find_feature_tensor(std::string{project_prefix} + ".0.weight");
  if (project_weight == nullptr ||
      !bind_pointwise_f16(
          *project_weight, config.output_channels, config.expanded_channels, runtime_out.project) ||
      !bind_feature_bn(project_prefix, config.output_channels, runtime_out.project_bn)) {
    return false;
  }

  runtime_out.ready = true;
  return true;
}

inline float mel_scale_scalar(const float frequency) noexcept {
  return 1127.0f * std::log(1.0f + frequency / 700.0f);
}

inline float inverse_mel_scale_scalar(const float mel_frequency) noexcept {
  return 700.0f * (std::exp(mel_frequency / 1127.0f) - 1.0f);
}

inline int32_t make_divisible_8(const int32_t value) noexcept {
  return std::max(8, ((value + 7) / 8) * 8);
}

inline void build_audio_fft_window(action::audio_runtime & runtime) noexcept {
  std::fill(runtime.fft_window.get(), runtime.fft_window.get() + runtime.n_fft, 0.0f);
  const int32_t pad_left = (runtime.n_fft - runtime.win_length) / 2;
  for (int32_t index = 0; index < runtime.win_length; ++index) {
    const float angle =
        (2.0f * k_pi * static_cast<float>(index)) /
        static_cast<float>(runtime.win_length - 1);
    runtime.fft_window[static_cast<size_t>(pad_left + index)] = 0.5f - 0.5f * std::cos(angle);
  }
}

inline void build_audio_mel_filters(action::audio_runtime & runtime) noexcept {
  const int32_t fft_bin_count = runtime.n_fft / 2;
  const int32_t full_fft_bin_count = fft_bin_count + 1;
  std::fill(runtime.mel_filters.get(),
            runtime.mel_filters.get() +
                static_cast<size_t>(runtime.num_mel_bins * full_fft_bin_count),
            0.0f);

  const float fft_bin_width =
      static_cast<float>(runtime.resampled_sample_rate) / static_cast<float>(runtime.n_fft);
  const float mel_low = mel_scale_scalar(runtime.low_frequency);
  const float mel_high = mel_scale_scalar(runtime.high_frequency);
  const float mel_delta =
      (mel_high - mel_low) / static_cast<float>(runtime.num_mel_bins + 1);

  for (int32_t bin = 0; bin < runtime.num_mel_bins; ++bin) {
    const float left_mel = mel_low + static_cast<float>(bin) * mel_delta;
    const float center_mel = mel_low + static_cast<float>(bin + 1) * mel_delta;
    const float right_mel = mel_low + static_cast<float>(bin + 2) * mel_delta;
    float * bin_weights =
        runtime.mel_filters.get() +
        static_cast<size_t>(bin * full_fft_bin_count);
    for (int32_t fft_bin = 0; fft_bin < fft_bin_count; ++fft_bin) {
      const float mel = mel_scale_scalar(fft_bin_width * static_cast<float>(fft_bin));
      const float up_slope = (mel - left_mel) / (center_mel - left_mel);
      const float down_slope = (right_mel - mel) / (right_mel - center_mel);
      bin_weights[static_cast<size_t>(fft_bin)] = std::max(0.0f, std::min(up_slope, down_slope));
    }
    bin_weights[static_cast<size_t>(fft_bin_count)] = 0.0f;
  }
}

inline bool build_audio_runtime(const emel::model::data & model,
                                action::audio_runtime & runtime_out) noexcept {
  runtime_out = {};
  if (emel::model::architecture_name_view(model) != "omniembed") {
    return false;
  }
  if (emel::model::metadata_string_view(model.meta, model.meta.clip_audio_data.encoder_name) !=
      "efficientat_mn20_as") {
    return false;
  }

  static constexpr std::array<audio_block_config, 15> k_configs = {{
      {32, 32, 32, 3, 1, false, false},
      {32, 128, 48, 3, 2, false, false},
      {48, 144, 48, 3, 1, false, false},
      {48, 144, 80, 5, 2, true, false},
      {80, 240, 80, 5, 1, true, false},
      {80, 240, 80, 5, 1, true, false},
      {80, 480, 160, 3, 2, false, true},
      {160, 400, 160, 3, 1, false, true},
      {160, 368, 160, 3, 1, false, true},
      {160, 368, 160, 3, 1, false, true},
      {160, 960, 224, 3, 1, true, true},
      {224, 1344, 224, 3, 1, true, true},
      {224, 1344, 320, 5, 2, true, true},
      {320, 1920, 320, 5, 1, true, true},
      {320, 1920, 320, 5, 1, true, true},
  }};

  runtime_out.input_sample_rate = k_audio_input_sample_rate;
  runtime_out.input_sample_count = k_audio_input_sample_count;
  runtime_out.resampled_sample_rate = model.meta.clip_audio_data.sample_rate;
  runtime_out.resampled_sample_count =
      runtime_out.input_sample_count * runtime_out.resampled_sample_rate /
      runtime_out.input_sample_rate;
  runtime_out.preemphasized_sample_count = runtime_out.resampled_sample_count - 1;
  runtime_out.n_fft = model.meta.clip_audio_data.n_fft;
  runtime_out.win_length = model.meta.clip_audio_data.win_length;
  runtime_out.hop_size = model.meta.clip_audio_data.hop_size;
  runtime_out.num_mel_bins = model.meta.clip_audio_data.num_mel_bins;
  runtime_out.time_frames =
      1 + runtime_out.preemphasized_sample_count / runtime_out.hop_size;
  runtime_out.embedding_size = model.meta.clip_audio_data.embedding_length;
  runtime_out.batch_norm_epsilon = 1.0e-5f;
  runtime_out.low_frequency = model.meta.clip_audio_data.low_frequency;
  runtime_out.high_frequency = model.meta.clip_audio_data.high_frequency;
  runtime_out.preemphasis_coefficient = model.meta.clip_audio_data.preemphasis_coefficient;
  runtime_out.log_offset = model.meta.clip_audio_data.log_offset;
  runtime_out.normalize_bias = model.meta.clip_audio_data.normalize_bias;
  runtime_out.normalize_scale = model.meta.clip_audio_data.normalize_scale;
  runtime_out.max_dense_input_size = runtime_out.embedding_size;

  const auto * stem_weight = find_tensor(model, "audio_encoder.features.0.0.weight");
  const auto * stem_bn_weight = find_tensor(model, "audio_encoder.features.0.1.weight");
  const auto * stem_bn_bias = find_tensor(model, "audio_encoder.features.0.1.bias");
  const auto * stem_bn_mean = find_tensor(model, "audio_encoder.features.0.1.running_mean");
  const auto * stem_bn_var = find_tensor(model, "audio_encoder.features.0.1.running_var");
  const auto * head_weight = find_tensor(model, "audio_encoder.features.16.0.weight");
  const auto * head_bn_weight = find_tensor(model, "audio_encoder.features.16.1.weight");
  const auto * head_bn_bias = find_tensor(model, "audio_encoder.features.16.1.bias");
  const auto * head_bn_mean = find_tensor(model, "audio_encoder.features.16.1.running_mean");
  const auto * head_bn_var = find_tensor(model, "audio_encoder.features.16.1.running_var");
  if (stem_weight == nullptr ||
      stem_bn_weight == nullptr ||
      stem_bn_bias == nullptr ||
      stem_bn_mean == nullptr ||
      stem_bn_var == nullptr ||
      head_weight == nullptr ||
      head_bn_weight == nullptr ||
      head_bn_bias == nullptr ||
      head_bn_mean == nullptr ||
      head_bn_var == nullptr) {
    return false;
  }

  if (!bind_conv_f16_hwio(*stem_weight, 3, 3, 1, 32, runtime_out.stem.conv) ||
      !bind_batch_norm(*stem_bn_weight, *stem_bn_bias, *stem_bn_mean, *stem_bn_var, 32, runtime_out.stem.norm) ||
      !bind_conv_f16_hwio(*head_weight, 1, 1, 320, 1920, runtime_out.head.conv) ||
      !bind_batch_norm(*head_bn_weight, *head_bn_bias, *head_bn_mean, *head_bn_var, 1920, runtime_out.head.norm)) {
    return false;
  }
  runtime_out.stem.input_channels = 1;
  runtime_out.stem.output_channels = 32;
  runtime_out.head.input_channels = 320;
  runtime_out.head.output_channels = 1920;

  int32_t height = runtime_out.num_mel_bins;
  int32_t width = runtime_out.time_frames;
  int32_t max_feature_elements = height * width;
  int32_t max_dense_input = runtime_out.embedding_size;

  height = output_dim_same(height, 3, 2);
  width = output_dim_same(width, 3, 2);
  max_feature_elements = std::max(max_feature_elements, height * width * 32);

  for (int32_t index = 0; index < static_cast<int32_t>(k_configs.size()); ++index) {
    auto & block = runtime_out.blocks[static_cast<size_t>(index)];
    if (!bind_audio_block(model, index + 1, k_configs[static_cast<size_t>(index)], block)) {
      return false;
    }
    ++runtime_out.block_count;

    const auto & config = k_configs[static_cast<size_t>(index)];
    max_dense_input = std::max(max_dense_input, config.expanded_channels);
    const int32_t output_height = output_dim_same(height, config.kernel_size, config.stride);
    const int32_t output_width = output_dim_same(width, config.kernel_size, config.stride);
    max_feature_elements =
        std::max(max_feature_elements, output_height * output_width * config.expanded_channels);
    max_feature_elements =
        std::max(max_feature_elements, output_height * output_width * config.output_channels);
    height = output_height;
    width = output_width;
  }

  height = output_dim_same(height, 1, 1);
  width = output_dim_same(width, 1, 1);
  max_feature_elements = std::max(max_feature_elements, height * width * 1920);
  max_dense_input = std::max(max_dense_input, 1920);

  if (!bind_projection_runtime(model,
                               k_audio_projection_expand_weight_name,
                               k_audio_projection_expand_bias_name,
                               k_audio_projection_expand_norm_weight_name,
                               k_audio_projection_expand_norm_bias_name,
                               k_audio_projection_residual_weight_name,
                               k_audio_projection_residual_bias_name,
                               k_audio_projection_residual_norm_weight_name,
                               k_audio_projection_residual_norm_bias_name,
                               k_audio_projection_project_weight_name,
                               k_audio_projection_project_bias_name,
                               model.meta.clip_audio_data.embedding_length,
                               1920,
                               model.params.n_embd_out,
                               runtime_out.projection)) {
    return false;
  }

  runtime_out.feature_buffer_elements = max_feature_elements;
  runtime_out.max_dense_input_size = max_dense_input;
  runtime_out.fft_window = std::unique_ptr<float[]>{new (std::nothrow) float[runtime_out.n_fft]};
  runtime_out.mel_filters = std::unique_ptr<float[]>{new (std::nothrow) float[
      static_cast<size_t>(runtime_out.num_mel_bins) *
      static_cast<size_t>(runtime_out.n_fft / 2 + 1)]};
  if (runtime_out.fft_window == nullptr || runtime_out.mel_filters == nullptr) {
    return false;
  }

  build_audio_fft_window(runtime_out);
  build_audio_mel_filters(runtime_out);
  runtime_out.ready = runtime_out.input_sample_count > 0 &&
      runtime_out.resampled_sample_count > 0 &&
      runtime_out.preemphasized_sample_count > 0 &&
      runtime_out.time_frames > 0 &&
      runtime_out.embedding_size > 0 &&
      runtime_out.feature_buffer_elements > 0 &&
      runtime_out.block_count == static_cast<int32_t>(k_configs.size());
  return runtime_out.ready;
}

inline bool reserve_scratch(action::context & ctx, const emel::model::data & model) noexcept {
  if (!build_text_runtime(model, ctx.text)) {
    ctx.scratch = {};
    return false;
  }
  (void) build_image_runtime(model, ctx.image);
  (void) build_audio_runtime(model, ctx.audio);

  const size_t token_count = static_cast<size_t>(ctx.text.max_positions);
  const size_t hidden_size = static_cast<size_t>(ctx.text.hidden_size);
  const size_t intermediate_size = static_cast<size_t>(ctx.text.intermediate_size);
  const size_t output_size = static_cast<size_t>(ctx.text.output_size);
  const size_t projection_hidden = static_cast<size_t>(std::max(
      ctx.text.projection_hidden_size,
      std::max(ctx.image.ready ? ctx.image.projection.hidden_size : 0,
               ctx.audio.ready ? ctx.audio.projection.hidden_size : 0)));
  const size_t projection_residual = static_cast<size_t>(std::max(
      static_cast<int32_t>(projection_hidden),
      ctx.audio.ready ? ctx.audio.max_dense_input_size : 0));
  const size_t shared_embedding = static_cast<size_t>(ctx.text.shared_embedding_size);
  const size_t image_input_elements =
      ctx.image.ready ? static_cast<size_t>(3 * ctx.image.input_size * ctx.image.input_size) : 0u;
  const size_t image_feature_elements =
      ctx.image.ready ? static_cast<size_t>(ctx.image.feature_buffer_elements) : 0u;
  const size_t image_embedding_size =
      ctx.image.ready ? static_cast<size_t>(ctx.image.embedding_size) : 0u;
  const size_t audio_waveform_size =
      ctx.audio.ready ? static_cast<size_t>(ctx.audio.resampled_sample_count) : 0u;
  const size_t audio_preemphasized_size =
      ctx.audio.ready ? static_cast<size_t>(ctx.audio.preemphasized_sample_count) : 0u;
  const size_t audio_fft_frame_size =
      ctx.audio.ready ? static_cast<size_t>(ctx.audio.n_fft) : 0u;
  const size_t audio_power_size =
      ctx.audio.ready ? static_cast<size_t>(ctx.audio.n_fft / 2 + 1) : 0u;
  const size_t audio_input_elements =
      ctx.audio.ready ? static_cast<size_t>(ctx.audio.num_mel_bins * ctx.audio.time_frames) : 0u;
  const size_t audio_feature_elements =
      ctx.audio.ready ? static_cast<size_t>(ctx.audio.feature_buffer_elements) : 0u;
  const size_t audio_embedding_size =
      ctx.audio.ready ? static_cast<size_t>(ctx.audio.embedding_size) : 0u;
  const size_t max_q8_cols = static_cast<size_t>(std::max(
      ctx.text.hidden_size,
      std::max(ctx.text.intermediate_size,
               std::max(ctx.text.output_size,
                        std::max(std::max(ctx.text.projection_hidden_size,
                                          ctx.image.ready ? ctx.image.projection.hidden_size : 0),
                                 std::max(ctx.audio.ready ? ctx.audio.projection.hidden_size : 0,
                                          ctx.audio.ready ? ctx.audio.max_dense_input_size : 0))))));
  const size_t q8_block_capacity =
      max_q8_cols / static_cast<size_t>(emel::kernel::detail::quant::QK8_0);

  auto allocate_i32 = [](const size_t count) noexcept {
    return std::unique_ptr<int32_t[]>{new (std::nothrow) int32_t[count]};
  };
  auto allocate_f32 = [](const size_t count) noexcept {
    return std::unique_ptr<float[]>{new (std::nothrow) float[count]};
  };
  auto allocate_q8 = [](const size_t count) noexcept {
    return std::unique_ptr<emel::kernel::detail::quant::block_q8_0[]>(
        new (std::nothrow) emel::kernel::detail::quant::block_q8_0[count]);
  };

  action::scratch_buffers scratch = {};
  scratch.token_ids = allocate_i32(token_count);
  scratch.sequence_a = allocate_f32(token_count * hidden_size);
  scratch.sequence_b = allocate_f32(token_count * hidden_size);
  scratch.query = allocate_f32(token_count * hidden_size);
  scratch.key = allocate_f32(token_count * hidden_size);
  scratch.value = allocate_f32(token_count * hidden_size);
  scratch.attention_context = allocate_f32(token_count * hidden_size);
  scratch.attention_scores = allocate_f32(token_count);
  scratch.token_hidden = allocate_f32(hidden_size);
  scratch.feed_forward = allocate_f32(intermediate_size);
  scratch.pooled = allocate_f32(hidden_size);
  scratch.text_embedding = allocate_f32(output_size);
  scratch.projection_hidden = allocate_f32(projection_hidden);
  scratch.projection_residual = allocate_f32(projection_residual);
  scratch.full_embedding = allocate_f32(shared_embedding);
  if (ctx.image.ready) {
    scratch.image_input = allocate_f32(image_input_elements);
    scratch.image_a = allocate_f32(image_feature_elements);
    scratch.image_b = allocate_f32(image_feature_elements);
    scratch.image_c = allocate_f32(image_feature_elements);
    scratch.image_embedding = allocate_f32(image_embedding_size);
  }
  if (ctx.audio.ready) {
    scratch.audio_waveform = allocate_f32(audio_waveform_size);
    scratch.audio_preemphasized = allocate_f32(audio_preemphasized_size);
    scratch.audio_fft_frame = allocate_f32(audio_fft_frame_size);
    scratch.audio_power = allocate_f32(audio_power_size);
    scratch.audio_input = allocate_f32(audio_input_elements);
    scratch.audio_a = allocate_f32(audio_feature_elements);
    scratch.audio_b = allocate_f32(audio_feature_elements);
    scratch.audio_c = allocate_f32(audio_feature_elements);
    scratch.audio_embedding = allocate_f32(audio_embedding_size);
  }
  scratch.q8_input = allocate_q8(q8_block_capacity);
  scratch.q8_input_block_capacity = q8_block_capacity;

  scratch.ready = scratch.token_ids != nullptr &&
      scratch.sequence_a != nullptr &&
      scratch.sequence_b != nullptr &&
      scratch.query != nullptr &&
      scratch.key != nullptr &&
      scratch.value != nullptr &&
      scratch.attention_context != nullptr &&
      scratch.attention_scores != nullptr &&
      scratch.token_hidden != nullptr &&
      scratch.feed_forward != nullptr &&
      scratch.pooled != nullptr &&
      scratch.text_embedding != nullptr &&
      scratch.projection_hidden != nullptr &&
      scratch.projection_residual != nullptr &&
      scratch.full_embedding != nullptr &&
      scratch.q8_input != nullptr;
  if (ctx.image.ready) {
    scratch.ready = scratch.ready &&
        scratch.image_input != nullptr &&
        scratch.image_a != nullptr &&
        scratch.image_b != nullptr &&
        scratch.image_c != nullptr &&
        scratch.image_embedding != nullptr;
  }
  if (ctx.audio.ready) {
    scratch.ready = scratch.ready &&
        scratch.audio_waveform != nullptr &&
        scratch.audio_preemphasized != nullptr &&
        scratch.audio_fft_frame != nullptr &&
        scratch.audio_power != nullptr &&
        scratch.audio_input != nullptr &&
        scratch.audio_a != nullptr &&
        scratch.audio_b != nullptr &&
        scratch.audio_c != nullptr &&
        scratch.audio_embedding != nullptr;
  }
  ctx.scratch = std::move(scratch);
  return ctx.scratch.ready;
}

inline bool is_valid_preprocessor(
    const emel::text::tokenizer::preprocessor::preprocessor_kind value) noexcept {
  switch (value) {
    case emel::text::tokenizer::preprocessor::preprocessor_kind::spm:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::bpe:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::wpm:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::ugm:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::rwkv:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::plamo2:
    case emel::text::tokenizer::preprocessor::preprocessor_kind::fallback:
      return true;
    default:
      return false;
  }
}

inline bool is_valid_encoder(const emel::text::encoders::encoder_kind value) noexcept {
  switch (value) {
    case emel::text::encoders::encoder_kind::spm:
    case emel::text::encoders::encoder_kind::bpe:
    case emel::text::encoders::encoder_kind::wpm:
    case emel::text::encoders::encoder_kind::ugm:
    case emel::text::encoders::encoder_kind::rwkv:
    case emel::text::encoders::encoder_kind::plamo2:
    case emel::text::encoders::encoder_kind::fallback:
      return true;
    default:
      return false;
  }
}

inline int32_t shared_embedding_size(const action::context & ctx) noexcept {
  return ctx.model != nullptr ? ctx.model->params.n_embd_out : ctx.text.shared_embedding_size;
}

inline int32_t requested_output_dimension(const event::embed_text & request,
                                          const action::context & ctx) noexcept {
  const int32_t full_dimension = shared_embedding_size(ctx);
  return request.truncate_dimension > 0 ? request.truncate_dimension : full_dimension;
}

inline int32_t requested_output_dimension(const event::embed_image & request,
                                          const action::context & ctx) noexcept {
  const int32_t full_dimension = shared_embedding_size(ctx);
  return request.truncate_dimension > 0 ? request.truncate_dimension : full_dimension;
}

inline int32_t requested_output_dimension(const event::embed_audio & request,
                                          const action::context & ctx) noexcept {
  const int32_t full_dimension = shared_embedding_size(ctx);
  return request.truncate_dimension > 0 ? request.truncate_dimension : full_dimension;
}

inline bool is_supported_truncate_dimension(const action::context & ctx,
                                            const int32_t dimension) noexcept {
  if (ctx.model == nullptr || dimension <= 0) {
    return false;
  }
  if (dimension == ctx.text.shared_embedding_size) {
    return true;
  }
  for (uint32_t index = 0u; index < ctx.model->params.matryoshka_dimension_count; ++index) {
    if (ctx.model->params.matryoshka_dimensions[index] == dimension) {
      return true;
    }
  }
  return false;
}

inline bool is_valid_image_payload(std::span<const uint8_t> rgba,
                                   const int32_t width,
                                   const int32_t height) noexcept {
  if (rgba.data() == nullptr || width <= 0 || height <= 0) {
    return false;
  }
  const uint64_t pixel_count = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  const uint64_t byte_count = pixel_count * 4u;
  return byte_count == rgba.size();
}

inline bool is_valid_audio_payload(std::span<const float> pcm,
                                   const int32_t sample_rate) noexcept {
  return pcm.data() != nullptr &&
      sample_rate == k_audio_input_sample_rate &&
      pcm.size() == static_cast<size_t>(k_audio_input_sample_count);
}

inline bool has_embed_callbacks(const event::embed_text_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_done);
}

inline bool has_embed_callbacks(const event::embed_image_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_done);
}

inline bool has_embed_callbacks(const event::embed_audio_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_done);
}

inline bool has_embed_error_callback(const event::embed_text_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_error);
}

inline bool has_embed_error_callback(const event::embed_image_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_error);
}

inline bool has_embed_error_callback(const event::embed_audio_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_error);
}

inline bool has_initialize_callback(const event::initialize_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_done);
}

inline bool has_initialize_error_callback(const event::initialize_run & runtime_ev) noexcept {
  return static_cast<bool>(runtime_ev.request.on_error);
}

inline void set_error(const event::initialize_run & runtime_ev, const error err) noexcept {
  runtime_ev.ctx.err = to_error(err);
}

inline void set_error(const event::embed_text_run & runtime_ev, const error err) noexcept {
  runtime_ev.ctx.err = to_error(err);
}

inline void set_error(const event::embed_image_run & runtime_ev, const error err) noexcept {
  runtime_ev.ctx.err = to_error(err);
}

inline void set_error(const event::embed_audio_run & runtime_ev, const error err) noexcept {
  runtime_ev.ctx.err = to_error(err);
}

inline bool matmul_q8_0(const action::matrix_view & matrix,
                        std::span<const float> input,
                        std::span<emel::kernel::detail::quant::block_q8_0> q8_input,
                        std::span<float> output) noexcept {
  if (matrix.dtype != emel::kernel::detail::dtype_q8_0 ||
      matrix.cols <= 0 ||
      matrix.rows <= 0 ||
      static_cast<size_t>(matrix.cols) != input.size() ||
      static_cast<size_t>(matrix.rows) != output.size() ||
      (static_cast<size_t>(matrix.cols) % static_cast<size_t>(emel::kernel::detail::quant::QK8_0)) != 0u) {
    return false;
  }

  const size_t block_count =
      static_cast<size_t>(matrix.cols) / static_cast<size_t>(emel::kernel::detail::quant::QK8_0);
  if (block_count == 0u || block_count > q8_input.size()) {
    return false;
  }

  emel::kernel::detail::quant::quantize_row_q8_0_strided(
      input.data(), 1u, q8_input.data(), matrix.cols);
  const auto * base = static_cast<const uint8_t *>(matrix.data);
  for (int32_t row = 0; row < matrix.rows; ++row) {
    const auto * row_ptr = base + static_cast<size_t>(row) * matrix.row_bytes;
    output[static_cast<size_t>(row)] =
        emel::kernel::detail::dot_q8_0_q8_0_row_scalar(
            reinterpret_cast<const emel::kernel::detail::quant::block_q8_0 *>(row_ptr),
            q8_input.data(),
            block_count);
  }
  return true;
}

inline bool matmul_f16(const action::matrix_view & matrix,
                       std::span<const float> input,
                       std::span<float> output) noexcept {
  if (matrix.dtype != emel::kernel::detail::dtype_f16 ||
      matrix.cols <= 0 ||
      matrix.rows <= 0 ||
      static_cast<size_t>(matrix.cols) != input.size() ||
      static_cast<size_t>(matrix.rows) != output.size()) {
    return false;
  }

  const auto * base = static_cast<const uint8_t *>(matrix.data);
  for (int32_t row = 0; row < matrix.rows; ++row) {
    const auto * weights = reinterpret_cast<const uint16_t *>(
        base + static_cast<size_t>(row) * matrix.row_bytes);
    float acc = 0.0f;
    for (int32_t col = 0; col < matrix.cols; ++col) {
      acc += emel::kernel::detail::quant::fp16_to_fp32(weights[col]) *
          input[static_cast<size_t>(col)];
    }
    output[static_cast<size_t>(row)] = acc;
  }
  return true;
}

inline bool matmul_f32(const action::matrix_view & matrix,
                       std::span<const float> input,
                       std::span<float> output) noexcept {
  if (matrix.dtype != emel::kernel::detail::dtype_f32 ||
      matrix.cols <= 0 ||
      matrix.rows <= 0 ||
      static_cast<size_t>(matrix.cols) != input.size() ||
      static_cast<size_t>(matrix.rows) != output.size()) {
    return false;
  }

  const auto * base = static_cast<const uint8_t *>(matrix.data);
  for (int32_t row = 0; row < matrix.rows; ++row) {
    const auto * weights =
        reinterpret_cast<const float *>(base + static_cast<size_t>(row) * matrix.row_bytes);
    float acc = 0.0f;
    for (int32_t col = 0; col < matrix.cols; ++col) {
      acc += weights[col] * input[static_cast<size_t>(col)];
    }
    output[static_cast<size_t>(row)] = acc;
  }
  return true;
}

inline bool matmul(const action::matrix_view & matrix,
                   std::span<const float> input,
                   std::span<emel::kernel::detail::quant::block_q8_0> q8_input,
                   std::span<float> output) noexcept {
  if (matrix.dtype == emel::kernel::detail::dtype_q8_0) {
    return matmul_q8_0(matrix, input, q8_input, output);
  }
  if (matrix.dtype == emel::kernel::detail::dtype_f16) {
    return matmul_f16(matrix, input, output);
  }
  if (matrix.dtype == emel::kernel::detail::dtype_f32) {
    return matmul_f32(matrix, input, output);
  }
  return false;
}

inline bool add_bias(std::span<float> values, const action::vector_view & bias) noexcept {
  if (static_cast<int32_t>(values.size()) != bias.size || bias.data == nullptr) {
    return false;
  }
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] += bias.data[index];
  }
  return true;
}

inline bool add_in_place(std::span<float> dst, std::span<const float> src) noexcept {
  if (dst.size() != src.size()) {
    return false;
  }
  for (size_t index = 0; index < dst.size(); ++index) {
    dst[index] += src[index];
  }
  return true;
}

inline float gelu(const float value) noexcept {
  return 0.5f * value * (1.0f + std::erf(value / k_sqrt_two));
}

inline void apply_gelu(std::span<float> values) noexcept {
  for (float & value : values) {
    value = gelu(value);
  }
}

inline bool layer_norm(std::span<const float> input,
                       const action::vector_view & weight,
                       const action::vector_view & bias,
                       const float epsilon,
                       std::span<float> output) noexcept {
  if (input.empty() ||
      input.size() != output.size() ||
      static_cast<int32_t>(input.size()) != weight.size ||
      static_cast<int32_t>(input.size()) != bias.size ||
      weight.data == nullptr ||
      bias.data == nullptr) {
    return false;
  }

  double mean = 0.0;
  for (const float value : input) {
    mean += static_cast<double>(value);
  }
  mean /= static_cast<double>(input.size());

  double variance = 0.0;
  for (const float value : input) {
    const double delta = static_cast<double>(value) - mean;
    variance += delta * delta;
  }
  variance /= static_cast<double>(input.size());

  const float scale = 1.0f / std::sqrt(static_cast<float>(variance) + epsilon);
  for (size_t index = 0; index < input.size(); ++index) {
    const float centered = input[index] - static_cast<float>(mean);
    output[index] = centered * scale * weight.data[index] + bias.data[index];
  }
  return true;
}

inline bool l2_normalize(std::span<float> values) noexcept {
  if (values.empty()) {
    return false;
  }
  double sum = 0.0;
  for (const float value : values) {
    sum += static_cast<double>(value) * static_cast<double>(value);
  }
  if (sum <= 0.0) {
    return false;
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(sum));
  for (float & value : values) {
    value *= scale;
  }
  return true;
}

inline bool copy_q8_embedding_row(const action::matrix_view & embeddings,
                                  const int32_t row,
                                  std::span<float> output) noexcept {
  if (embeddings.dtype != emel::kernel::detail::dtype_q8_0 ||
      row < 0 ||
      row >= embeddings.rows ||
      static_cast<int32_t>(output.size()) != embeddings.cols) {
    return false;
  }
  const auto * row_ptr = reinterpret_cast<const emel::kernel::detail::quant::block_q8_0 *>(
      static_cast<const uint8_t *>(embeddings.data) + static_cast<size_t>(row) * embeddings.row_bytes);
  emel::kernel::detail::quant::dequantize_row_q8_0(row_ptr, output.data(), embeddings.cols);
  return true;
}

inline bool soft_max(std::span<float> values) noexcept {
  if (values.empty()) {
    return false;
  }
  float max_value = values[0];
  for (size_t index = 1; index < values.size(); ++index) {
    max_value = std::max(max_value, values[index]);
  }
  double sum = 0.0;
  for (float & value : values) {
    value = std::exp(value - max_value);
    sum += static_cast<double>(value);
  }
  if (sum <= 0.0) {
    return false;
  }
  const float inv_sum = 1.0f / static_cast<float>(sum);
  for (float & value : values) {
    value *= inv_sum;
  }
  return true;
}

inline std::span<float> token_span(float * base,
                                   const int32_t token_index,
                                   const int32_t hidden_size) noexcept {
  return std::span<float>{
      base + static_cast<size_t>(token_index) * static_cast<size_t>(hidden_size),
      static_cast<size_t>(hidden_size),
  };
}

inline std::span<const float> token_span(const float * base,
                                         const int32_t token_index,
                                         const int32_t hidden_size) noexcept {
  return std::span<const float>{
      base + static_cast<size_t>(token_index) * static_cast<size_t>(hidden_size),
      static_cast<size_t>(hidden_size),
  };
}

inline std::span<float> head_span(std::span<float> token_values,
                                  const int32_t head_index,
                                  const int32_t head_dim) noexcept {
  return token_values.subspan(
      static_cast<size_t>(head_index) * static_cast<size_t>(head_dim),
      static_cast<size_t>(head_dim));
}

inline std::span<const float> head_span(std::span<const float> token_values,
                                        const int32_t head_index,
                                        const int32_t head_dim) noexcept {
  return token_values.subspan(
      static_cast<size_t>(head_index) * static_cast<size_t>(head_dim),
      static_cast<size_t>(head_dim));
}

inline float dot(std::span<const float> lhs, std::span<const float> rhs) noexcept {
  float sum = 0.0f;
  for (size_t index = 0; index < lhs.size(); ++index) {
    sum += lhs[index] * rhs[index];
  }
  return sum;
}

inline bool embed_tokens(action::context & ctx, const int32_t token_count) noexcept {
  auto word = std::span<float>{ctx.scratch.token_hidden.get(), static_cast<size_t>(ctx.text.hidden_size)};
  auto pos = std::span<float>{ctx.scratch.projection_residual.get(), static_cast<size_t>(ctx.text.hidden_size)};
  auto type = std::span<float>{ctx.scratch.pooled.get(), static_cast<size_t>(ctx.text.hidden_size)};
  for (int32_t token_index = 0; token_index < token_count; ++token_index) {
    const int32_t token_id = ctx.scratch.token_ids[static_cast<size_t>(token_index)];
    if (token_id < 0 || token_id >= ctx.text.word_embeddings.rows) {
      return false;
    }
    if (!copy_q8_embedding_row(ctx.text.word_embeddings, token_id, word) ||
        !copy_q8_embedding_row(ctx.text.position_embeddings, token_index, pos) ||
        !copy_q8_embedding_row(ctx.text.token_type_embeddings, 0, type)) {
      return false;
    }
    auto out = token_span(ctx.scratch.sequence_a.get(), token_index, ctx.text.hidden_size);
    for (int32_t feature = 0; feature < ctx.text.hidden_size; ++feature) {
      out[static_cast<size_t>(feature)] =
          word[static_cast<size_t>(feature)] +
          pos[static_cast<size_t>(feature)] +
          type[static_cast<size_t>(feature)];
    }
    if (!layer_norm(out,
                    ctx.text.embeddings_norm_weight,
                    ctx.text.embeddings_norm_bias,
                    ctx.text.encoder_layer_norm_epsilon,
                    out)) {
      return false;
    }
  }
  return true;
}

inline bool run_attention_layer(action::context & ctx,
                                const action::layer_weights & layer,
                                const int32_t token_count,
                                float * sequence_in,
                                float * sequence_out) noexcept {
  auto q8_input = std::span<emel::kernel::detail::quant::block_q8_0>{
      ctx.scratch.q8_input.get(),
      ctx.scratch.q8_input_block_capacity,
  };
  for (int32_t token_index = 0; token_index < token_count; ++token_index) {
    const auto input = token_span(sequence_in, token_index, ctx.text.hidden_size);
    auto query = token_span(ctx.scratch.query.get(), token_index, ctx.text.hidden_size);
    auto key = token_span(ctx.scratch.key.get(), token_index, ctx.text.hidden_size);
    auto value = token_span(ctx.scratch.value.get(), token_index, ctx.text.hidden_size);
    if (!matmul(layer.attention_query, input, q8_input, query) ||
        !add_bias(query, layer.attention_query_bias) ||
        !matmul(layer.attention_key, input, q8_input, key) ||
        !add_bias(key, layer.attention_key_bias) ||
        !matmul(layer.attention_value, input, q8_input, value) ||
        !add_bias(value, layer.attention_value_bias)) {
      return false;
    }
  }

  for (int32_t token_index = 0; token_index < token_count; ++token_index) {
    auto query = token_span(ctx.scratch.query.get(), token_index, ctx.text.hidden_size);
    auto context = token_span(ctx.scratch.attention_context.get(), token_index, ctx.text.hidden_size);
    std::fill(context.begin(), context.end(), 0.0f);

    for (int32_t head_index = 0; head_index < ctx.text.attention_head_count; ++head_index) {
      const auto query_head =
          head_span(std::span<const float>{query.data(), query.size()},
                    head_index,
                    ctx.text.attention_head_dim);
      auto scores = std::span<float>{ctx.scratch.attention_scores.get(), static_cast<size_t>(token_count)};
      for (int32_t other_index = 0; other_index < token_count; ++other_index) {
        const auto key_head =
            head_span(token_span(ctx.scratch.key.get(), other_index, ctx.text.hidden_size),
                      head_index,
                      ctx.text.attention_head_dim);
        scores[static_cast<size_t>(other_index)] = dot(query_head, key_head) *
            k_attention_scale_default;
      }
      if (!soft_max(scores)) {
        return false;
      }

      auto context_head = head_span(context, head_index, ctx.text.attention_head_dim);
      for (int32_t other_index = 0; other_index < token_count; ++other_index) {
        const auto value_head =
            head_span(token_span(ctx.scratch.value.get(), other_index, ctx.text.hidden_size),
                      head_index,
                      ctx.text.attention_head_dim);
        const float weight = scores[static_cast<size_t>(other_index)];
        for (int32_t feature = 0; feature < ctx.text.attention_head_dim; ++feature) {
          context_head[static_cast<size_t>(feature)] +=
              value_head[static_cast<size_t>(feature)] * weight;
        }
      }
    }

    auto hidden = std::span<float>{ctx.scratch.token_hidden.get(), static_cast<size_t>(ctx.text.hidden_size)};
    const auto input = token_span(sequence_in, token_index, ctx.text.hidden_size);
    if (!matmul(layer.attention_output, context, q8_input, hidden) ||
        !add_bias(hidden, layer.attention_output_bias)) {
      return false;
    }
    for (int32_t feature = 0; feature < ctx.text.hidden_size; ++feature) {
      hidden[static_cast<size_t>(feature)] += input[static_cast<size_t>(feature)];
    }
    if (!layer_norm(hidden,
                    layer.attention_output_norm_weight,
                    layer.attention_output_norm_bias,
                    ctx.text.encoder_layer_norm_epsilon,
                    hidden)) {
      return false;
    }

    auto feed_forward = std::span<float>{ctx.scratch.feed_forward.get(), static_cast<size_t>(ctx.text.intermediate_size)};
    if (!matmul(layer.intermediate, hidden, q8_input, feed_forward) ||
        !add_bias(feed_forward, layer.intermediate_bias)) {
      return false;
    }
    apply_gelu(feed_forward);

    auto output = token_span(sequence_out, token_index, ctx.text.hidden_size);
    if (!matmul(layer.output, feed_forward, q8_input, output) ||
        !add_bias(output, layer.output_bias)) {
      return false;
    }
    for (int32_t feature = 0; feature < ctx.text.hidden_size; ++feature) {
      output[static_cast<size_t>(feature)] += hidden[static_cast<size_t>(feature)];
    }
    if (!layer_norm(output,
                    layer.output_norm_weight,
                    layer.output_norm_bias,
                    ctx.text.encoder_layer_norm_epsilon,
                    output)) {
      return false;
    }
  }

  return true;
}

inline bool mean_pool(action::context & ctx, const int32_t token_count, const float * sequence) noexcept {
  auto pooled = std::span<float>{ctx.scratch.pooled.get(), static_cast<size_t>(ctx.text.hidden_size)};
  std::fill(pooled.begin(), pooled.end(), 0.0f);
  const float scale = 1.0f / static_cast<float>(token_count);
  for (int32_t token_index = 0; token_index < token_count; ++token_index) {
    const auto row = token_span(sequence, token_index, ctx.text.hidden_size);
    for (int32_t feature = 0; feature < ctx.text.hidden_size; ++feature) {
      pooled[static_cast<size_t>(feature)] += row[static_cast<size_t>(feature)] * scale;
    }
  }
  return true;
}

inline bool run_projection_head(action::context & ctx,
                                const action::projection_runtime & projection,
                                std::span<const float> input_embedding) noexcept;

inline int32_t same_padding(const int32_t kernel, const int32_t stride) noexcept {
  return ((stride - 1) + (kernel - 1)) / 2;
}

inline int32_t output_dim_same(const int32_t input, const int32_t kernel, const int32_t stride) noexcept {
  return (input + 2 * same_padding(kernel, stride) - kernel) / stride + 1;
}

template <bool apply_relu>
inline void apply_batch_norm_hwc(float * values,
                                 const int32_t spatial,
                                 const action::batch_norm_view & bn,
                                 const float epsilon) noexcept {
  const int32_t channels = bn.channels;
  for (int32_t pixel_index = 0; pixel_index < spatial * spatial; ++pixel_index) {
    float * pixel = values + static_cast<size_t>(pixel_index) * static_cast<size_t>(channels);
    for (int32_t channel = 0; channel < channels; ++channel) {
      const float gamma = bn.weight.data[channel];
      const float beta = bn.bias.data[channel];
      const float mean = bn.running_mean.data[channel];
      const float variance = bn.running_var.data[channel];
      float value =
          (pixel[channel] - mean) * (gamma / std::sqrt(variance + epsilon)) + beta;
      if constexpr (apply_relu) {
        value = std::max(value, 0.0f);
      }
      pixel[channel] = value;
    }
  }
}

inline bool pointwise_conv_hwc(const action::matrix_view & matrix,
                               const float * input,
                               const int32_t pixel_count,
                               float * output) noexcept {
  if (matrix.dtype != emel::kernel::detail::dtype_f16 ||
      matrix.rows <= 0 ||
      matrix.cols <= 0 ||
      input == nullptr ||
      output == nullptr ||
      pixel_count <= 0) {
    return false;
  }

  for (int32_t pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
    const auto input_pixel = std::span<const float>{
        input + static_cast<size_t>(pixel_index) * static_cast<size_t>(matrix.cols),
        static_cast<size_t>(matrix.cols),
    };
    auto output_pixel = std::span<float>{
        output + static_cast<size_t>(pixel_index) * static_cast<size_t>(matrix.rows),
        static_cast<size_t>(matrix.rows),
    };
    if (!matmul_f16(matrix, input_pixel, output_pixel)) {
      return false;
    }
  }
  return true;
}

inline bool standard_conv_hwc(const action::conv2d_view & conv,
                              const float * input,
                              const int32_t input_spatial,
                              const int32_t stride,
                              float * output,
                              int32_t & output_spatial_out) noexcept {
  if (conv.data == nullptr ||
      input == nullptr ||
      output == nullptr ||
      conv.input_channels <= 0 ||
      conv.output_channels <= 0 ||
      input_spatial <= 0 ||
      stride <= 0) {
    return false;
  }

  const int32_t output_spatial = output_dim_same(input_spatial, conv.kernel_h, stride);
  const int32_t pad_h = same_padding(conv.kernel_h, stride);
  const int32_t pad_w = same_padding(conv.kernel_w, stride);
  const auto * weights = conv.data;
  for (int32_t oy = 0; oy < output_spatial; ++oy) {
    for (int32_t ox = 0; ox < output_spatial; ++ox) {
      float * output_pixel =
          output + static_cast<size_t>(oy * output_spatial + ox) *
              static_cast<size_t>(conv.output_channels);
      std::fill(output_pixel, output_pixel + conv.output_channels, 0.0f);
      for (int32_t ky = 0; ky < conv.kernel_h; ++ky) {
        const int32_t iy = oy * stride + ky - pad_h;
        if (iy < 0 || iy >= input_spatial) {
          continue;
        }
        for (int32_t kx = 0; kx < conv.kernel_w; ++kx) {
          const int32_t ix = ox * stride + kx - pad_w;
          if (ix < 0 || ix >= input_spatial) {
            continue;
          }
          const float * input_pixel =
              input + static_cast<size_t>(iy * input_spatial + ix) *
                  static_cast<size_t>(conv.input_channels);
          for (int32_t output_channel = 0; output_channel < conv.output_channels; ++output_channel) {
            float acc = output_pixel[output_channel];
            const size_t output_base =
                static_cast<size_t>(output_channel) *
                static_cast<size_t>(conv.input_channels * conv.kernel_h * conv.kernel_w);
            for (int32_t input_channel = 0; input_channel < conv.input_channels; ++input_channel) {
              const size_t index = output_base +
                  static_cast<size_t>(input_channel * conv.kernel_h * conv.kernel_w +
                                      ky * conv.kernel_w +
                                      kx);
              acc += emel::kernel::detail::quant::fp16_to_fp32(weights[index]) *
                  input_pixel[input_channel];
            }
            output_pixel[output_channel] = acc;
          }
        }
      }
    }
  }
  output_spatial_out = output_spatial;
  return true;
}

inline bool depthwise_conv_hwc(const action::conv2d_view & conv,
                               const float * input,
                               const int32_t input_spatial,
                               const int32_t stride,
                               float * output,
                               int32_t & output_spatial_out) noexcept {
  if (conv.data == nullptr ||
      input == nullptr ||
      output == nullptr ||
      conv.input_channels != 1 ||
      conv.output_channels <= 0 ||
      input_spatial <= 0 ||
      stride <= 0) {
    return false;
  }

  const int32_t channels = conv.output_channels;
  const int32_t output_spatial = output_dim_same(input_spatial, conv.kernel_h, stride);
  const int32_t pad_h = same_padding(conv.kernel_h, stride);
  const int32_t pad_w = same_padding(conv.kernel_w, stride);
  const auto * weights = conv.data;
  for (int32_t oy = 0; oy < output_spatial; ++oy) {
    for (int32_t ox = 0; ox < output_spatial; ++ox) {
      float * output_pixel =
          output + static_cast<size_t>(oy * output_spatial + ox) * static_cast<size_t>(channels);
      for (int32_t channel = 0; channel < channels; ++channel) {
        float acc = 0.0f;
        const size_t channel_base =
            static_cast<size_t>(channel * conv.kernel_h * conv.kernel_w);
        for (int32_t ky = 0; ky < conv.kernel_h; ++ky) {
          const int32_t iy = oy * stride + ky - pad_h;
          if (iy < 0 || iy >= input_spatial) {
            continue;
          }
          for (int32_t kx = 0; kx < conv.kernel_w; ++kx) {
            const int32_t ix = ox * stride + kx - pad_w;
            if (ix < 0 || ix >= input_spatial) {
              continue;
            }
            const float * input_pixel =
                input + static_cast<size_t>(iy * input_spatial + ix) * static_cast<size_t>(channels);
            const size_t index =
                channel_base + static_cast<size_t>(ky * conv.kernel_w + kx);
            acc += emel::kernel::detail::quant::fp16_to_fp32(weights[index]) *
                input_pixel[channel];
          }
        }
        output_pixel[channel] = acc;
      }
    }
  }
  output_spatial_out = output_spatial;
  return true;
}

inline void average_pool_hwc(const float * input,
                             const int32_t spatial,
                             const int32_t channels,
                             float * output) noexcept {
  std::fill(output, output + channels, 0.0f);
  const float scale = 1.0f / static_cast<float>(spatial * spatial);
  for (int32_t pixel_index = 0; pixel_index < spatial * spatial; ++pixel_index) {
    const float * input_pixel =
        input + static_cast<size_t>(pixel_index) * static_cast<size_t>(channels);
    for (int32_t channel = 0; channel < channels; ++channel) {
      output[channel] += input_pixel[channel] * scale;
    }
  }
}

inline float hardswish(const float value) noexcept {
  const float shifted = std::clamp(value + 3.0f, 0.0f, 6.0f);
  return value * shifted * (1.0f / 6.0f);
}

template <bool apply_relu, bool apply_hardswish>
inline void apply_activation_in_place(std::span<float> values) noexcept {
  for (float & value : values) {
    if constexpr (apply_relu) {
      value = std::max(value, 0.0f);
    } else if constexpr (apply_hardswish) {
      value = hardswish(value);
    }
  }
}

template <bool apply_relu, bool apply_hardswish>
inline void apply_batch_norm_hwc_rect(float * values,
                                      const int32_t height,
                                      const int32_t width,
                                      const action::batch_norm_view & bn,
                                      const float epsilon) noexcept {
  const int32_t channels = bn.channels;
  const int32_t pixel_count = height * width;
  for (int32_t pixel_index = 0; pixel_index < pixel_count; ++pixel_index) {
    float * pixel = values + static_cast<size_t>(pixel_index) * static_cast<size_t>(channels);
    for (int32_t channel = 0; channel < channels; ++channel) {
      const float gamma = bn.weight.data[channel];
      const float beta = bn.bias.data[channel];
      const float mean = bn.running_mean.data[channel];
      const float variance = bn.running_var.data[channel];
      float value =
          (pixel[channel] - mean) * (gamma / std::sqrt(variance + epsilon)) + beta;
      if constexpr (apply_relu) {
        value = std::max(value, 0.0f);
      } else if constexpr (apply_hardswish) {
        value = hardswish(value);
      }
      pixel[channel] = value;
    }
  }
}

inline bool standard_conv_hwc_rect(const action::conv2d_view & conv,
                                   const float * input,
                                   const int32_t input_height,
                                   const int32_t input_width,
                                   const int32_t stride_h,
                                   const int32_t stride_w,
                                   float * output,
                                   int32_t & output_height_out,
                                   int32_t & output_width_out) noexcept {
  if (conv.data == nullptr ||
      input == nullptr ||
      output == nullptr ||
      conv.input_channels <= 0 ||
      conv.output_channels <= 0 ||
      input_height <= 0 ||
      input_width <= 0 ||
      stride_h <= 0 ||
      stride_w <= 0) {
    return false;
  }

  const int32_t output_height = output_dim_same(input_height, conv.kernel_h, stride_h);
  const int32_t output_width = output_dim_same(input_width, conv.kernel_w, stride_w);
  const int32_t pad_h = same_padding(conv.kernel_h, stride_h);
  const int32_t pad_w = same_padding(conv.kernel_w, stride_w);
  const auto * weights = conv.data;
  for (int32_t oy = 0; oy < output_height; ++oy) {
    for (int32_t ox = 0; ox < output_width; ++ox) {
      float * output_pixel =
          output + static_cast<size_t>(oy * output_width + ox) *
              static_cast<size_t>(conv.output_channels);
      std::fill(output_pixel, output_pixel + conv.output_channels, 0.0f);
      for (int32_t ky = 0; ky < conv.kernel_h; ++ky) {
        const int32_t iy = oy * stride_h + ky - pad_h;
        if (iy < 0 || iy >= input_height) {
          continue;
        }
        for (int32_t kx = 0; kx < conv.kernel_w; ++kx) {
          const int32_t ix = ox * stride_w + kx - pad_w;
          if (ix < 0 || ix >= input_width) {
            continue;
          }
          const float * input_pixel =
              input + static_cast<size_t>(iy * input_width + ix) *
                  static_cast<size_t>(conv.input_channels);
          for (int32_t output_channel = 0; output_channel < conv.output_channels; ++output_channel) {
            float acc = output_pixel[output_channel];
            const size_t output_base =
                static_cast<size_t>(output_channel) *
                static_cast<size_t>(conv.input_channels * conv.kernel_h * conv.kernel_w);
            for (int32_t input_channel = 0; input_channel < conv.input_channels; ++input_channel) {
              const size_t index = output_base +
                  static_cast<size_t>(input_channel * conv.kernel_h * conv.kernel_w +
                                      ky * conv.kernel_w +
                                      kx);
              acc += emel::kernel::detail::quant::fp16_to_fp32(weights[index]) *
                  input_pixel[input_channel];
            }
            output_pixel[output_channel] = acc;
          }
        }
      }
    }
  }
  output_height_out = output_height;
  output_width_out = output_width;
  return true;
}

inline bool depthwise_conv_hwc_rect(const action::conv2d_view & conv,
                                    const float * input,
                                    const int32_t input_height,
                                    const int32_t input_width,
                                    const int32_t stride_h,
                                    const int32_t stride_w,
                                    float * output,
                                    int32_t & output_height_out,
                                    int32_t & output_width_out) noexcept {
  if (conv.data == nullptr ||
      input == nullptr ||
      output == nullptr ||
      conv.input_channels != 1 ||
      conv.output_channels <= 0 ||
      input_height <= 0 ||
      input_width <= 0 ||
      stride_h <= 0 ||
      stride_w <= 0) {
    return false;
  }

  const int32_t channels = conv.output_channels;
  const int32_t output_height = output_dim_same(input_height, conv.kernel_h, stride_h);
  const int32_t output_width = output_dim_same(input_width, conv.kernel_w, stride_w);
  const int32_t pad_h = same_padding(conv.kernel_h, stride_h);
  const int32_t pad_w = same_padding(conv.kernel_w, stride_w);
  const auto * weights = conv.data;
  for (int32_t oy = 0; oy < output_height; ++oy) {
    for (int32_t ox = 0; ox < output_width; ++ox) {
      float * output_pixel =
          output + static_cast<size_t>(oy * output_width + ox) * static_cast<size_t>(channels);
      for (int32_t channel = 0; channel < channels; ++channel) {
        float acc = 0.0f;
        const size_t channel_base =
            static_cast<size_t>(channel * conv.kernel_h * conv.kernel_w);
        for (int32_t ky = 0; ky < conv.kernel_h; ++ky) {
          const int32_t iy = oy * stride_h + ky - pad_h;
          if (iy < 0 || iy >= input_height) {
            continue;
          }
          for (int32_t kx = 0; kx < conv.kernel_w; ++kx) {
            const int32_t ix = ox * stride_w + kx - pad_w;
            if (ix < 0 || ix >= input_width) {
              continue;
            }
            const float * input_pixel =
                input + static_cast<size_t>(iy * input_width + ix) * static_cast<size_t>(channels);
            const size_t index =
                channel_base + static_cast<size_t>(ky * conv.kernel_w + kx);
            acc += emel::kernel::detail::quant::fp16_to_fp32(weights[index]) *
                input_pixel[channel];
          }
        }
        output_pixel[channel] = acc;
      }
    }
  }
  output_height_out = output_height;
  output_width_out = output_width;
  return true;
}

inline void average_pool_hwc_rect(const float * input,
                                  const int32_t height,
                                  const int32_t width,
                                  const int32_t channels,
                                  float * output) noexcept {
  std::fill(output, output + channels, 0.0f);
  const float scale = 1.0f / static_cast<float>(height * width);
  for (int32_t pixel_index = 0; pixel_index < height * width; ++pixel_index) {
    const float * input_pixel =
        input + static_cast<size_t>(pixel_index) * static_cast<size_t>(channels);
    for (int32_t channel = 0; channel < channels; ++channel) {
      output[channel] += input_pixel[channel] * scale;
    }
  }
}

inline int32_t reflect_index(int32_t index, const int32_t length) noexcept {
  while (index < 0 || index >= length) {
    if (index < 0) {
      index = -index;
    } else {
      index = 2 * length - 2 - index;
    }
  }
  return index;
}

inline float bicubic_weight(const float value) noexcept {
  const float x = std::fabs(value);
  if (x <= 1.0f) {
    return ((1.5f * x - 2.5f) * x * x) + 1.0f;
  }
  if (x < 2.0f) {
    return (((-0.5f * x + 2.5f) * x - 4.0f) * x) + 2.0f;
  }
  return 0.0f;
}

inline bool prepare_image_input(action::context & ctx,
                                std::span<const uint8_t> rgba,
                                const int32_t width,
                                const int32_t height) noexcept {
  if (ctx.model == nullptr ||
      !ctx.image.ready ||
      !is_valid_image_payload(rgba, width, height) ||
      ctx.scratch.image_input == nullptr) {
    return false;
  }

  const int32_t output_size = ctx.image.input_size;
  const auto & vision = ctx.model->meta.clip_vision_data;
  const float mean[3] = {
    vision.image_mean[0],
    vision.image_mean[1],
    vision.image_mean[2],
  };
  const float std[3] = {
    vision.image_std[0],
    vision.image_std[1],
    vision.image_std[2],
  };

  float * output = ctx.scratch.image_input.get();
  for (int32_t oy = 0; oy < output_size; ++oy) {
    const float source_y =
        ((static_cast<float>(oy) + 0.5f) * static_cast<float>(height) /
         static_cast<float>(output_size)) -
        0.5f;
    const int32_t base_y = static_cast<int32_t>(std::floor(source_y));
    for (int32_t ox = 0; ox < output_size; ++ox) {
      const float source_x =
          ((static_cast<float>(ox) + 0.5f) * static_cast<float>(width) /
           static_cast<float>(output_size)) -
          0.5f;
      const int32_t base_x = static_cast<int32_t>(std::floor(source_x));
      float accum[3] = {0.0f, 0.0f, 0.0f};
      float weight_sum = 0.0f;
      for (int32_t ky = -1; ky <= 2; ++ky) {
        const int32_t iy = std::clamp(base_y + ky, 0, height - 1);
        const float wy = bicubic_weight(source_y - static_cast<float>(base_y + ky));
        for (int32_t kx = -1; kx <= 2; ++kx) {
          const int32_t ix = std::clamp(base_x + kx, 0, width - 1);
          const float wx = bicubic_weight(source_x - static_cast<float>(base_x + kx));
          const float weight = wy * wx;
          const size_t pixel_index =
              (static_cast<size_t>(iy) * static_cast<size_t>(width) + static_cast<size_t>(ix)) * 4u;
          accum[0] += weight * (static_cast<float>(rgba[pixel_index]) / 255.0f);
          accum[1] += weight * (static_cast<float>(rgba[pixel_index + 1u]) / 255.0f);
          accum[2] += weight * (static_cast<float>(rgba[pixel_index + 2u]) / 255.0f);
          weight_sum += weight;
        }
      }

      const float inv_weight = weight_sum > 0.0f ? 1.0f / weight_sum : 1.0f;
      float * output_pixel =
          output + static_cast<size_t>(oy * output_size + ox) * 3u;
      for (int32_t channel = 0; channel < 3; ++channel) {
        const float normalized = accum[channel] * inv_weight;
        output_pixel[channel] = (normalized - mean[channel]) / std[channel];
      }
    }
  }

  return true;
}

inline bool prepare_audio_input(action::context & ctx,
                                std::span<const float> pcm,
                                const int32_t sample_rate) noexcept {
  if (ctx.model == nullptr ||
      !ctx.audio.ready ||
      !is_valid_audio_payload(pcm, sample_rate) ||
      ctx.scratch.audio_waveform == nullptr ||
      ctx.scratch.audio_preemphasized == nullptr ||
      ctx.scratch.audio_fft_frame == nullptr ||
      ctx.scratch.audio_power == nullptr ||
      ctx.scratch.audio_input == nullptr ||
      ctx.audio.fft_window == nullptr ||
      ctx.audio.mel_filters == nullptr) {
    return false;
  }

  float * resampled = ctx.scratch.audio_waveform.get();
  for (int32_t index = 0; index < ctx.audio.resampled_sample_count; ++index) {
    const float source_position =
        static_cast<float>(index) *
        static_cast<float>(sample_rate) /
        static_cast<float>(ctx.audio.resampled_sample_rate);
    const int32_t left = std::min(
        static_cast<int32_t>(source_position),
        ctx.audio.input_sample_count - 1);
    const int32_t right = std::min(left + 1, ctx.audio.input_sample_count - 1);
    const float frac = source_position - static_cast<float>(left);
    resampled[index] = pcm[static_cast<size_t>(left)] * (1.0f - frac) +
        pcm[static_cast<size_t>(right)] * frac;
  }

  float * preemphasized = ctx.scratch.audio_preemphasized.get();
  for (int32_t index = 0; index < ctx.audio.preemphasized_sample_count; ++index) {
    preemphasized[index] = resampled[index + 1] -
        ctx.audio.preemphasis_coefficient * resampled[index];
  }

  const int32_t fft_bin_count = ctx.audio.n_fft / 2 + 1;
  for (int32_t frame = 0; frame < ctx.audio.time_frames; ++frame) {
    float * fft_frame = ctx.scratch.audio_fft_frame.get();
    for (int32_t sample_index = 0; sample_index < ctx.audio.n_fft; ++sample_index) {
      const int32_t centered_index = frame * ctx.audio.hop_size + sample_index - ctx.audio.n_fft / 2;
      const int32_t reflected_index =
          reflect_index(centered_index, ctx.audio.preemphasized_sample_count);
      fft_frame[sample_index] =
          preemphasized[reflected_index] * ctx.audio.fft_window[sample_index];
    }

    float * power = ctx.scratch.audio_power.get();
    for (int32_t bin = 0; bin < fft_bin_count; ++bin) {
      const float angle_step =
          -2.0f * k_pi * static_cast<float>(bin) / static_cast<float>(ctx.audio.n_fft);
      const float cos_step = std::cos(angle_step);
      const float sin_step = std::sin(angle_step);
      float cos_value = 1.0f;
      float sin_value = 0.0f;
      float real = 0.0f;
      float imag = 0.0f;
      for (int32_t sample_index = 0; sample_index < ctx.audio.n_fft; ++sample_index) {
        const float value = fft_frame[sample_index];
        real += value * cos_value;
        imag += value * sin_value;
        const float next_cos = cos_value * cos_step - sin_value * sin_step;
        sin_value = sin_value * cos_step + cos_value * sin_step;
        cos_value = next_cos;
      }
      power[bin] = real * real + imag * imag;
    }

    for (int32_t mel_bin = 0; mel_bin < ctx.audio.num_mel_bins; ++mel_bin) {
      const float * weights =
          ctx.audio.mel_filters.get() + static_cast<size_t>(mel_bin * fft_bin_count);
      float energy = 0.0f;
      for (int32_t fft_bin = 0; fft_bin < fft_bin_count; ++fft_bin) {
        energy += weights[fft_bin] * power[fft_bin];
      }
      const float value = std::log(energy + ctx.audio.log_offset);
      ctx.scratch.audio_input[static_cast<size_t>(mel_bin * ctx.audio.time_frames + frame)] =
          (value + ctx.audio.normalize_bias) / ctx.audio.normalize_scale;
    }
  }

  return true;
}

inline bool run_audio_se(action::context & ctx,
                         const action::squeeze_excitation_runtime & se,
                         float * values,
                         const int32_t height,
                         const int32_t width) noexcept {
  if (!se.ready ||
      values == nullptr ||
      ctx.scratch.audio_embedding == nullptr ||
      ctx.scratch.projection_hidden == nullptr ||
      ctx.scratch.projection_residual == nullptr) {
    return false;
  }

  auto pooled = std::span<float>{ctx.scratch.audio_embedding.get(), static_cast<size_t>(se.input_size)};
  auto hidden = std::span<float>{ctx.scratch.projection_hidden.get(), static_cast<size_t>(se.hidden_size)};
  auto scale = std::span<float>{ctx.scratch.projection_residual.get(), static_cast<size_t>(se.input_size)};
  auto q8_input = std::span<emel::kernel::detail::quant::block_q8_0>{
      ctx.scratch.q8_input.get(),
      ctx.scratch.q8_input_block_capacity,
  };

  average_pool_hwc_rect(values, height, width, se.input_size, pooled.data());
  if (!matmul(se.fc1, pooled, q8_input, hidden) ||
      !add_bias(hidden, se.fc1_bias)) {
    return false;
  }
  apply_activation_in_place<true, false>(hidden);
  if (!matmul(se.fc2, hidden, q8_input, scale) ||
      !add_bias(scale, se.fc2_bias)) {
    return false;
  }

  const int32_t pixel_count = height * width;
  for (int32_t index = 0; index < se.input_size; ++index) {
    scale[static_cast<size_t>(index)] =
        1.0f / (1.0f + std::exp(-scale[static_cast<size_t>(index)]));
  }
  for (int32_t pixel = 0; pixel < pixel_count; ++pixel) {
    float * pixel_values =
        values + static_cast<size_t>(pixel) * static_cast<size_t>(se.input_size);
    for (int32_t channel = 0; channel < se.input_size; ++channel) {
      pixel_values[channel] *= scale[static_cast<size_t>(channel)];
    }
  }
  return true;
}

inline bool run_audio_block(action::context & ctx,
                            const action::audio_inverted_residual_runtime & block,
                            int32_t & height,
                            int32_t & width) noexcept {
  if (!block.ready ||
      ctx.scratch.audio_a == nullptr ||
      ctx.scratch.audio_b == nullptr ||
      ctx.scratch.audio_c == nullptr) {
    return false;
  }

  const size_t input_elements =
      static_cast<size_t>(height) * static_cast<size_t>(width) *
      static_cast<size_t>(block.input_channels);
  if (block.has_skip) {
    std::memcpy(
        ctx.scratch.audio_c.get(), ctx.scratch.audio_a.get(), input_elements * sizeof(float));
  }

  const float * depthwise_src = ctx.scratch.audio_a.get();
  float * depthwise_dst = ctx.scratch.audio_b.get();
  int32_t work_height = height;
  int32_t work_width = width;
  if (block.has_expand) {
    if (!pointwise_conv_hwc(
            block.expand,
            ctx.scratch.audio_a.get(),
            height * width,
            ctx.scratch.audio_b.get())) {
      return false;
    }
    if (block.use_hardswish) {
      apply_batch_norm_hwc_rect<false, true>(
          ctx.scratch.audio_b.get(),
          height,
          width,
          block.expand_bn,
          ctx.audio.batch_norm_epsilon);
    } else {
      apply_batch_norm_hwc_rect<true, false>(
          ctx.scratch.audio_b.get(),
          height,
          width,
          block.expand_bn,
          ctx.audio.batch_norm_epsilon);
    }
    depthwise_src = ctx.scratch.audio_b.get();
    depthwise_dst = ctx.scratch.audio_a.get();
  }

  int32_t output_height = 0;
  int32_t output_width = 0;
  if (!depthwise_conv_hwc_rect(block.depthwise,
                               depthwise_src,
                               work_height,
                               work_width,
                               block.stride,
                               block.stride,
                               depthwise_dst,
                               output_height,
                               output_width)) {
    return false;
  }
  if (block.use_hardswish) {
    apply_batch_norm_hwc_rect<false, true>(
        depthwise_dst,
        output_height,
        output_width,
        block.depthwise_bn,
        ctx.audio.batch_norm_epsilon);
  } else {
    apply_batch_norm_hwc_rect<true, false>(
        depthwise_dst,
        output_height,
        output_width,
        block.depthwise_bn,
        ctx.audio.batch_norm_epsilon);
  }

  if (block.has_se &&
      !run_audio_se(ctx, block.se, depthwise_dst, output_height, output_width)) {
    return false;
  }

  float * project_dst = depthwise_dst == ctx.scratch.audio_a.get() ?
      ctx.scratch.audio_b.get() :
      ctx.scratch.audio_a.get();
  if (!pointwise_conv_hwc(
          block.project,
          depthwise_dst,
          output_height * output_width,
          project_dst)) {
    return false;
  }
  apply_batch_norm_hwc_rect<false, false>(
      project_dst,
      output_height,
      output_width,
      block.project_bn,
      ctx.audio.batch_norm_epsilon);

  if (block.has_skip) {
    if (!add_in_place(
            std::span<float>{project_dst, input_elements},
            std::span<const float>{ctx.scratch.audio_c.get(), input_elements})) {
      return false;
    }
  }

  if (project_dst != ctx.scratch.audio_a.get()) {
    const size_t output_elements =
        static_cast<size_t>(output_height) * static_cast<size_t>(output_width) *
        static_cast<size_t>(block.output_channels);
    std::memcpy(
        ctx.scratch.audio_a.get(), project_dst, output_elements * sizeof(float));
  }

  height = output_height;
  width = output_width;
  return true;
}

inline bool run_audio_embedding(action::context & ctx) noexcept {
  if (!ctx.audio.ready ||
      !ctx.scratch.ready ||
      ctx.scratch.audio_input == nullptr ||
      ctx.scratch.audio_a == nullptr ||
      ctx.scratch.audio_b == nullptr ||
      ctx.scratch.audio_embedding == nullptr) {
    return false;
  }

  int32_t height = 0;
  int32_t width = 0;
  if (!standard_conv_hwc_rect(ctx.audio.stem.conv,
                              ctx.scratch.audio_input.get(),
                              ctx.audio.num_mel_bins,
                              ctx.audio.time_frames,
                              2,
                              2,
                              ctx.scratch.audio_a.get(),
                              height,
                              width)) {
    return false;
  }
  apply_batch_norm_hwc_rect<false, true>(
      ctx.scratch.audio_a.get(),
      height,
      width,
      ctx.audio.stem.norm,
      ctx.audio.batch_norm_epsilon);

  for (int32_t index = 0; index < ctx.audio.block_count; ++index) {
    if (!run_audio_block(
            ctx, ctx.audio.blocks[static_cast<size_t>(index)], height, width)) {
      return false;
    }
  }

  int32_t head_height = 0;
  int32_t head_width = 0;
  if (!standard_conv_hwc_rect(ctx.audio.head.conv,
                              ctx.scratch.audio_a.get(),
                              height,
                              width,
                              1,
                              1,
                              ctx.scratch.audio_b.get(),
                              head_height,
                              head_width)) {
    return false;
  }
  apply_batch_norm_hwc_rect<false, true>(
      ctx.scratch.audio_b.get(),
      head_height,
      head_width,
      ctx.audio.head.norm,
      ctx.audio.batch_norm_epsilon);
  average_pool_hwc_rect(
      ctx.scratch.audio_b.get(),
      head_height,
      head_width,
      ctx.audio.embedding_size,
      ctx.scratch.audio_embedding.get());
  return run_projection_head(
      ctx,
      ctx.audio.projection,
      std::span<const float>{ctx.scratch.audio_embedding.get(),
                             static_cast<size_t>(ctx.audio.embedding_size)});
}

inline bool run_edge_residual(action::context & ctx, int32_t & spatial) noexcept {
  if (!ctx.image.stage0.ready) {
    return false;
  }

  int32_t stage_spatial = 0;
  if (!standard_conv_hwc(
          ctx.image.stage0.conv_exp, ctx.scratch.image_a.get(), spatial, ctx.image.stage0.stride, ctx.scratch.image_b.get(), stage_spatial)) {
    return false;
  }
  apply_batch_norm_hwc<true>(
      ctx.scratch.image_b.get(), stage_spatial, ctx.image.stage0.bn1, ctx.image.batch_norm_epsilon);
  if (!pointwise_conv_hwc(
          ctx.image.stage0.conv_pwl, ctx.scratch.image_b.get(), stage_spatial * stage_spatial, ctx.scratch.image_a.get())) {
    return false;
  }
  apply_batch_norm_hwc<false>(
      ctx.scratch.image_a.get(), stage_spatial, ctx.image.stage0.bn2, ctx.image.batch_norm_epsilon);
  spatial = stage_spatial;
  return true;
}

inline bool run_universal_inverted_block(action::context & ctx,
                                         const action::universal_inverted_runtime & block,
                                         int32_t & spatial) noexcept {
  if (!block.ready) {
    return false;
  }

  const size_t input_elements =
      static_cast<size_t>(spatial) * static_cast<size_t>(spatial) *
      static_cast<size_t>(block.input_channels);
  if (block.has_skip) {
    std::memcpy(
        ctx.scratch.image_c.get(), ctx.scratch.image_a.get(), input_elements * sizeof(float));
  }

  const float * expansion_src = ctx.scratch.image_a.get();
  float * expansion_dst = ctx.scratch.image_b.get();
  int32_t expansion_spatial = spatial;
  if (block.has_dw_start) {
    const int32_t dw_start_stride = block.has_dw_mid ? 1 : block.stride;
    if (!depthwise_conv_hwc(
            block.dw_start,
            ctx.scratch.image_a.get(),
            spatial,
            dw_start_stride,
            ctx.scratch.image_b.get(),
            expansion_spatial)) {
      return false;
    }
    apply_batch_norm_hwc<false>(
        ctx.scratch.image_b.get(),
        expansion_spatial,
        block.dw_start_bn,
        ctx.image.batch_norm_epsilon);
    expansion_src = ctx.scratch.image_b.get();
    expansion_dst = ctx.scratch.image_a.get();
  }

  if (!pointwise_conv_hwc(
          block.pw_exp,
          expansion_src,
          expansion_spatial * expansion_spatial,
          expansion_dst)) {
    return false;
  }
  apply_batch_norm_hwc<true>(
      expansion_dst,
      expansion_spatial,
      block.pw_exp_bn,
      ctx.image.batch_norm_epsilon);

  const float * projection_src = expansion_dst;
  float * projection_dst = expansion_dst == ctx.scratch.image_a.get() ?
      ctx.scratch.image_b.get() :
      ctx.scratch.image_a.get();
  int32_t output_spatial = expansion_spatial;
  if (block.has_dw_mid) {
    if (!depthwise_conv_hwc(
            block.dw_mid,
            expansion_dst,
            expansion_spatial,
            block.stride,
            projection_dst,
            output_spatial)) {
      return false;
    }
    apply_batch_norm_hwc<true>(
        projection_dst,
        output_spatial,
        block.dw_mid_bn,
        ctx.image.batch_norm_epsilon);
    projection_src = projection_dst;
    projection_dst = projection_dst == ctx.scratch.image_a.get() ?
        ctx.scratch.image_b.get() :
        ctx.scratch.image_a.get();
  }

  if (!pointwise_conv_hwc(
          block.pw_proj,
          projection_src,
          output_spatial * output_spatial,
          projection_dst)) {
    return false;
  }
  apply_batch_norm_hwc<false>(
      projection_dst,
      output_spatial,
      block.pw_proj_bn,
      ctx.image.batch_norm_epsilon);

  if (block.has_skip) {
    if (!add_in_place(
            std::span<float>{projection_dst, input_elements},
            std::span<const float>{ctx.scratch.image_c.get(), input_elements})) {
      return false;
    }
  }

  if (projection_dst != ctx.scratch.image_a.get()) {
    const size_t output_elements =
        static_cast<size_t>(output_spatial) * static_cast<size_t>(output_spatial) *
        static_cast<size_t>(block.output_channels);
    std::memcpy(
        ctx.scratch.image_a.get(), projection_dst, output_elements * sizeof(float));
  }

  spatial = output_spatial;
  return true;
}

inline bool run_image_embedding(action::context & ctx) noexcept {
  if (!ctx.image.ready ||
      !ctx.scratch.ready ||
      ctx.scratch.image_input == nullptr ||
      ctx.scratch.image_a == nullptr ||
      ctx.scratch.image_b == nullptr ||
      ctx.scratch.image_c == nullptr ||
      ctx.scratch.image_embedding == nullptr) {
    return false;
  }

  int32_t spatial = 0;
  if (!standard_conv_hwc(
          ctx.image.stem,
          ctx.scratch.image_input.get(),
          ctx.image.input_size,
          2,
          ctx.scratch.image_a.get(),
          spatial)) {
    return false;
  }
  apply_batch_norm_hwc<true>(
      ctx.scratch.image_a.get(), spatial, ctx.image.stem_bn, ctx.image.batch_norm_epsilon);

  if (!run_edge_residual(ctx, spatial)) {
    return false;
  }

  for (int32_t block_index = 0; block_index < ctx.image.block_count; ++block_index) {
    if (!run_universal_inverted_block(
            ctx, ctx.image.blocks[static_cast<size_t>(block_index)], spatial)) {
      return false;
    }
  }

  if (!pointwise_conv_hwc(
          ctx.image.stage4.conv,
          ctx.scratch.image_a.get(),
          spatial * spatial,
          ctx.scratch.image_b.get())) {
    return false;
  }
  apply_batch_norm_hwc<true>(
      ctx.scratch.image_b.get(), spatial, ctx.image.stage4.norm, ctx.image.batch_norm_epsilon);
  const size_t stage4_elements =
      static_cast<size_t>(spatial) * static_cast<size_t>(spatial) *
      static_cast<size_t>(ctx.image.stage4.output_channels);
  std::memcpy(ctx.scratch.image_a.get(), ctx.scratch.image_b.get(), stage4_elements * sizeof(float));

  average_pool_hwc(
      ctx.scratch.image_a.get(),
      spatial,
      ctx.image.stage4.output_channels,
      ctx.scratch.projection_hidden.get());
  if (!pointwise_conv_hwc(
          ctx.image.head.conv, ctx.scratch.projection_hidden.get(), 1, ctx.scratch.image_embedding.get())) {
    return false;
  }
  apply_batch_norm_hwc<true>(
      ctx.scratch.image_embedding.get(), 1, ctx.image.head.norm, ctx.image.batch_norm_epsilon);

  return run_projection_head(
      ctx,
      ctx.image.projection,
      std::span<const float>{
          ctx.scratch.image_embedding.get(),
          static_cast<size_t>(ctx.image.embedding_size),
      });
}

inline bool run_projection_head(action::context & ctx,
                                const action::projection_runtime & projection,
                                std::span<const float> input_embedding) noexcept {
  auto q8_input = std::span<emel::kernel::detail::quant::block_q8_0>{
      ctx.scratch.q8_input.get(),
      ctx.scratch.q8_input_block_capacity,
  };
  auto projection_hidden = std::span<float>{
      ctx.scratch.projection_hidden.get(), static_cast<size_t>(projection.hidden_size)};
  auto projection_residual = std::span<float>{
      ctx.scratch.projection_residual.get(), static_cast<size_t>(projection.hidden_size)};
  auto full_embedding = std::span<float>{
      ctx.scratch.full_embedding.get(), static_cast<size_t>(projection.output_size)};

  if (!matmul(projection.expand, input_embedding, q8_input, projection_hidden) ||
      !add_bias(projection_hidden, projection.expand_bias)) {
    return false;
  }
  apply_gelu(projection_hidden);
  if (!layer_norm(projection_hidden,
                  projection.expand_norm_weight,
                  projection.expand_norm_bias,
                  projection.layer_norm_epsilon,
                  projection_hidden)) {
    return false;
  }

  std::memcpy(projection_residual.data(),
              projection_hidden.data(),
              projection_hidden.size_bytes());
  if (!matmul(projection.residual, projection_hidden, q8_input, projection_residual) ||
      !add_bias(projection_residual, projection.residual_bias)) {
    return false;
  }
  apply_gelu(projection_residual);
  if (!layer_norm(projection_residual,
                  projection.residual_norm_weight,
                  projection.residual_norm_bias,
                  projection.layer_norm_epsilon,
                  projection_residual) ||
      !add_in_place(projection_residual, projection_hidden)) {
    return false;
  }

  return matmul(projection.project, projection_residual, q8_input, full_embedding) &&
      add_bias(full_embedding, projection.project_bias) &&
      l2_normalize(full_embedding);
}

inline bool run_text_projection(action::context & ctx) noexcept {
  auto q8_input = std::span<emel::kernel::detail::quant::block_q8_0>{
      ctx.scratch.q8_input.get(),
      ctx.scratch.q8_input_block_capacity,
  };
  auto pooled = std::span<const float>{ctx.scratch.pooled.get(), static_cast<size_t>(ctx.text.hidden_size)};
  auto text_embedding =
      std::span<float>{ctx.scratch.text_embedding.get(), static_cast<size_t>(ctx.text.output_size)};
  if (!matmul(ctx.text.dense, pooled, q8_input, text_embedding) ||
      !add_bias(text_embedding, ctx.text.dense_bias) ||
      !l2_normalize(text_embedding)) {
    return false;
  }
  return run_projection_head(ctx, ctx.text.projection, text_embedding);
}

inline bool run_text_embedding(action::context & ctx, const int32_t token_count) noexcept {
  if (!ctx.text.ready ||
      !ctx.scratch.ready ||
      token_count <= 0 ||
      token_count > ctx.text.max_positions) {
    return false;
  }

  if (!embed_tokens(ctx, token_count)) {
    return false;
  }

  float * sequence_in = ctx.scratch.sequence_a.get();
  float * sequence_out = ctx.scratch.sequence_b.get();
  for (int32_t layer_index = 0; layer_index < ctx.text.layer_count; ++layer_index) {
    if (!run_attention_layer(
            ctx, ctx.text.layers[static_cast<size_t>(layer_index)], token_count, sequence_in, sequence_out)) {
      return false;
    }
    std::swap(sequence_in, sequence_out);
  }

  return mean_pool(ctx, token_count, sequence_in) && run_text_projection(ctx);
}

inline bool publish_embedding(const action::context & ctx,
                              const int32_t dimension,
                              std::span<float> output) noexcept {
  const int32_t full_dimension = shared_embedding_size(ctx);
  if (dimension <= 0 ||
      dimension > full_dimension ||
      static_cast<size_t>(dimension) > output.size()) {
    return false;
  }

  auto source = std::span<const float>{
      ctx.scratch.full_embedding.get(),
      static_cast<size_t>(full_dimension),
  };
  if (dimension == full_dimension) {
    std::memcpy(output.data(), source.data(), static_cast<size_t>(dimension) * sizeof(float));
    return true;
  }

  std::memcpy(output.data(), source.data(), static_cast<size_t>(dimension) * sizeof(float));
  auto truncated = output.first(static_cast<size_t>(dimension));
  return l2_normalize(truncated);
}

inline void write_initialize_error_out(const event::initialize_run & runtime_ev) noexcept {
  if (runtime_ev.request.error_out != nullptr) {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
}

inline void write_embed_error_out(const event::embed_text_run & runtime_ev) noexcept {
  runtime_ev.request.output_dimension_out = runtime_ev.ctx.output_dimension;
  if (runtime_ev.request.error_out != nullptr) {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
}

inline void write_embed_error_out(const event::embed_image_run & runtime_ev) noexcept {
  runtime_ev.request.output_dimension_out = runtime_ev.ctx.output_dimension;
  if (runtime_ev.request.error_out != nullptr) {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
}

inline void write_embed_error_out(const event::embed_audio_run & runtime_ev) noexcept {
  runtime_ev.request.output_dimension_out = runtime_ev.ctx.output_dimension;
  if (runtime_ev.request.error_out != nullptr) {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
}

inline void emit_initialize_done(const event::initialize_run & runtime_ev) noexcept {
  runtime_ev.request.on_done(events::initialize_done{.request = &runtime_ev.request});
}

inline void emit_initialize_error(const event::initialize_run & runtime_ev) noexcept {
  runtime_ev.request.on_error(events::initialize_error{
      .request = &runtime_ev.request,
      .err = runtime_ev.ctx.err,
  });
}

inline void emit_embed_done(const event::embed_text_run & runtime_ev) noexcept {
  runtime_ev.request.on_done(events::text_embedding_done{
      .request = &runtime_ev.request,
      .output_dimension = runtime_ev.ctx.output_dimension,
  });
}

inline void emit_embed_done(const event::embed_image_run & runtime_ev) noexcept {
  runtime_ev.request.on_done(events::image_embedding_done{
      .request = &runtime_ev.request,
      .output_dimension = runtime_ev.ctx.output_dimension,
  });
}

inline void emit_embed_done(const event::embed_audio_run & runtime_ev) noexcept {
  runtime_ev.request.on_done(events::audio_embedding_done{
      .request = &runtime_ev.request,
      .output_dimension = runtime_ev.ctx.output_dimension,
  });
}

inline void emit_embed_error(const event::embed_text_run & runtime_ev) noexcept {
  runtime_ev.request.on_error(events::text_embedding_error{
      .request = &runtime_ev.request,
      .err = runtime_ev.ctx.err,
  });
}

inline void emit_embed_error(const event::embed_image_run & runtime_ev) noexcept {
  runtime_ev.request.on_error(events::image_embedding_error{
      .request = &runtime_ev.request,
      .err = runtime_ev.ctx.err,
  });
}

inline void emit_embed_error(const event::embed_audio_run & runtime_ev) noexcept {
  runtime_ev.request.on_error(events::audio_embedding_error{
      .request = &runtime_ev.request,
      .err = runtime_ev.ctx.err,
  });
}

}  // namespace emel::embeddings::generator::detail
