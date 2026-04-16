#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/model/data.hpp"
#include "emel/model/omniembed/detail.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"

namespace emel::embeddings::generator::action {

inline constexpr int32_t k_max_text_layers = 16;

enum class text_route_kind : uint8_t {
  none = 0u,
  omniembed_bert_text = 1u,
};

enum class image_route_kind : uint8_t {
  none = 0u,
  omniembed_mobilenetv4_medium = 1u,
};

enum class audio_route_kind : uint8_t {
  none = 0u,
  omniembed_efficientat_mn20_as = 1u,
};

struct matrix_view {
  const emel::model::data::tensor_record * tensor = nullptr;
  const void * data = nullptr;
  std::unique_ptr<float[]> expanded_f32 = {};
  std::unique_ptr<float[]> transposed_f32 = {};
  std::unique_ptr<float[]> packed_rhs_f32 = {};
  uint8_t dtype = static_cast<uint8_t>(emel::kernel::event::dtype::unknown);
  int32_t rows = 0;
  int32_t cols = 0;
  int32_t packed_rhs_cols = 0;
  size_t row_bytes = 0u;
  size_t transposed_row_bytes = 0u;
};

struct vector_view {
  const emel::model::data::tensor_record * tensor = nullptr;
  const float * data = nullptr;
  int32_t size = 0;
};

struct batch_norm_view {
  vector_view weight = {};
  vector_view bias = {};
  vector_view running_mean = {};
  vector_view running_var = {};
  std::unique_ptr<float[]> scale_storage = {};
  std::unique_ptr<float[]> shift_storage = {};
  const float * scale = nullptr;
  const float * shift = nullptr;
  int32_t channels = 0;
};

struct conv2d_view {
  const emel::model::data::tensor_record * tensor = nullptr;
  const uint16_t * data = nullptr;
  const float * data_f32 = nullptr;
  std::unique_ptr<float[]> expanded_f32 = {};
  const float * kernel_major_f32 = nullptr;
  std::unique_ptr<float[]> kernel_major_storage = {};
  const float * depthwise_kernel_major_f32 = nullptr;
  std::unique_ptr<float[]> depthwise_kernel_major_storage = {};
  int32_t kernel_w = 0;
  int32_t kernel_h = 0;
  int32_t input_channels = 0;
  int32_t output_channels = 0;
};

struct layer_weights {
  matrix_view attention_query = {};
  matrix_view attention_key = {};
  matrix_view attention_value = {};
  matrix_view attention_output = {};
  matrix_view intermediate = {};
  matrix_view output = {};
  vector_view attention_query_bias = {};
  vector_view attention_key_bias = {};
  vector_view attention_value_bias = {};
  vector_view attention_output_bias = {};
  vector_view attention_output_norm_weight = {};
  vector_view attention_output_norm_bias = {};
  vector_view intermediate_bias = {};
  vector_view output_bias = {};
  vector_view output_norm_weight = {};
  vector_view output_norm_bias = {};
};

struct projection_runtime {
  int32_t input_size = 0;
  int32_t hidden_size = 0;
  int32_t output_size = 0;
  float layer_norm_epsilon = 1.0e-5f;

  matrix_view expand = {};
  vector_view expand_bias = {};
  vector_view expand_norm_weight = {};
  vector_view expand_norm_bias = {};
  matrix_view residual = {};
  vector_view residual_bias = {};
  vector_view residual_norm_weight = {};
  vector_view residual_norm_bias = {};
  matrix_view project = {};
  vector_view project_bias = {};
};

struct conv_norm_runtime {
  matrix_view conv = {};
  batch_norm_view norm = {};
  int32_t input_channels = 0;
  int32_t output_channels = 0;
};

struct edge_residual_runtime {
  bool ready = false;
  conv2d_view conv_exp = {};
  batch_norm_view bn1 = {};
  matrix_view conv_pwl = {};
  batch_norm_view bn2 = {};
  int32_t input_channels = 0;
  int32_t output_channels = 0;
  int32_t stride = 1;
};

struct universal_inverted_runtime {
  bool ready = false;
  bool has_dw_start = false;
  bool has_dw_mid = false;
  bool has_skip = false;
  int32_t input_channels = 0;
  int32_t expanded_channels = 0;
  int32_t output_channels = 0;
  int32_t stride = 1;

  conv2d_view dw_start = {};
  batch_norm_view dw_start_bn = {};
  matrix_view pw_exp = {};
  batch_norm_view pw_exp_bn = {};
  conv2d_view dw_mid = {};
  batch_norm_view dw_mid_bn = {};
  matrix_view pw_proj = {};
  batch_norm_view pw_proj_bn = {};
};

struct squeeze_excitation_runtime {
  bool ready = false;
  int32_t input_size = 0;
  int32_t hidden_size = 0;
  matrix_view fc1 = {};
  vector_view fc1_bias = {};
  matrix_view fc2 = {};
  vector_view fc2_bias = {};
};

struct audio_conv_norm_runtime {
  conv2d_view conv = {};
  batch_norm_view norm = {};
  int32_t input_channels = 0;
  int32_t output_channels = 0;
};

struct audio_inverted_residual_runtime {
  bool ready = false;
  bool has_expand = false;
  bool has_se = false;
  bool has_skip = false;
  bool use_hardswish = false;
  int32_t input_channels = 0;
  int32_t expanded_channels = 0;
  int32_t output_channels = 0;
  int32_t kernel_size = 0;
  int32_t stride = 1;
  matrix_view expand = {};
  batch_norm_view expand_bn = {};
  conv2d_view depthwise = {};
  batch_norm_view depthwise_bn = {};
  squeeze_excitation_runtime se = {};
  matrix_view project = {};
  batch_norm_view project_bn = {};
};

struct text_runtime {
  bool ready = false;
  text_route_kind route_kind = text_route_kind::none;
  int32_t layer_count = 0;
  int32_t max_positions = 0;
  int32_t hidden_size = 0;
  int32_t intermediate_size = 0;
  int32_t output_size = 0;
  int32_t shared_embedding_size = 0;
  int32_t projection_hidden_size = 0;
  int32_t attention_head_count = 0;
  int32_t attention_head_dim = 0;
  float encoder_layer_norm_epsilon = 1.0e-12f;
  float projection_layer_norm_epsilon = 1.0e-5f;

  matrix_view word_embeddings = {};
  matrix_view position_embeddings = {};
  matrix_view token_type_embeddings = {};
  vector_view embeddings_norm_weight = {};
  vector_view embeddings_norm_bias = {};

  std::array<layer_weights, k_max_text_layers> layers = {};

  matrix_view dense = {};
  vector_view dense_bias = {};
  projection_runtime projection = {};
};

struct image_runtime {
  inline static constexpr int32_t k_max_blocks = 21;

  bool ready = false;
  image_route_kind route_kind = image_route_kind::none;
  int32_t input_size = 0;
  int32_t embedding_size = 0;
  int32_t feature_buffer_elements = 0;
  float batch_norm_epsilon = 1.0e-5f;
  conv2d_view stem = {};
  batch_norm_view stem_bn = {};
  edge_residual_runtime stage0 = {};
  std::array<universal_inverted_runtime, k_max_blocks> blocks = {};
  int32_t block_count = 0;
  conv_norm_runtime stage4 = {};
  conv_norm_runtime head = {};
  projection_runtime projection = {};
};

struct audio_runtime {
  inline static constexpr int32_t k_max_blocks = 15;

  bool ready = false;
  audio_route_kind route_kind = audio_route_kind::none;
  int32_t input_sample_rate = 0;
  int32_t input_sample_count = 0;
  int32_t resampled_sample_rate = 0;
  int32_t resampled_sample_count = 0;
  int32_t preemphasized_sample_count = 0;
  int32_t n_fft = 0;
  int32_t win_length = 0;
  int32_t hop_size = 0;
  int32_t num_mel_bins = 0;
  int32_t time_frames = 0;
  int32_t embedding_size = 0;
  int32_t feature_buffer_elements = 0;
  int32_t max_dense_input_size = 0;
  float batch_norm_epsilon = 1.0e-5f;
  float low_frequency = 0.0f;
  float high_frequency = 0.0f;
  float preemphasis_coefficient = 0.0f;
  float log_offset = 0.0f;
  float normalize_bias = 0.0f;
  float normalize_scale = 0.0f;

  audio_conv_norm_runtime stem = {};
  std::array<audio_inverted_residual_runtime, k_max_blocks> blocks = {};
  int32_t block_count = 0;
  audio_conv_norm_runtime head = {};
  projection_runtime projection = {};
  std::unique_ptr<float[]> mel_filters = {};
  std::unique_ptr<float[]> fft_window = {};
  std::unique_ptr<float[]> fft_twiddle_cos = {};
  std::unique_ptr<float[]> fft_twiddle_sin = {};
  std::unique_ptr<int32_t[]> fft_bit_reverse = {};
  std::unique_ptr<int32_t[]> mel_bin_start = {};
  std::unique_ptr<int32_t[]> mel_bin_end = {};
};

struct scratch_buffers {
  std::unique_ptr<int32_t[]> token_ids = {};
  std::unique_ptr<float[]> sequence_a = {};
  std::unique_ptr<float[]> sequence_b = {};
  std::unique_ptr<float[]> query = {};
  std::unique_ptr<float[]> key = {};
  std::unique_ptr<float[]> value = {};
  std::unique_ptr<float[]> attention_context = {};
  std::unique_ptr<float[]> attention_scores = {};
  std::unique_ptr<float[]> token_hidden = {};
  std::unique_ptr<float[]> feed_forward = {};
  std::unique_ptr<float[]> pooled = {};
  std::unique_ptr<float[]> text_embedding = {};
  std::unique_ptr<float[]> projection_hidden = {};
  std::unique_ptr<float[]> projection_residual = {};
  std::unique_ptr<float[]> full_embedding = {};
  std::unique_ptr<float[]> image_input = {};
  std::unique_ptr<float[]> image_a = {};
  std::unique_ptr<float[]> image_b = {};
  std::unique_ptr<float[]> image_c = {};
  std::unique_ptr<float[]> image_embedding = {};
  std::unique_ptr<float[]> audio_waveform = {};
  std::unique_ptr<float[]> audio_preemphasized = {};
  std::unique_ptr<float[]> audio_fft_frame = {};
  std::unique_ptr<float[]> audio_power = {};
  std::unique_ptr<float[]> audio_input = {};
  std::unique_ptr<float[]> audio_a = {};
  std::unique_ptr<float[]> audio_b = {};
  std::unique_ptr<float[]> audio_c = {};
  std::unique_ptr<float[]> audio_embedding = {};
  std::unique_ptr<emel::kernel::detail::quant::block_q8_0[]> q8_input = {};
  size_t q8_input_block_capacity = 0u;
  bool ready = false;
};

struct context {
  const emel::model::data * model = nullptr;
  emel::text::conditioner::sm * conditioner = nullptr;
  void * formatter_ctx = nullptr;
  emel::text::formatter::format_fn format_prompt =
      emel::text::formatter::format_raw;
  emel::model::omniembed::detail::execution_contract execution_contract = {};

  text_runtime text = {};
  image_runtime image = {};
  audio_runtime audio = {};
  scratch_buffers scratch = {};
  bool initialized = false;
};

}  // namespace emel::embeddings::generator::action
