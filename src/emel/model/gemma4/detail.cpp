#include "emel/model/gemma4/detail.hpp"

#include <array>
#include <span>

#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"

namespace emel::model::gemma4::detail {

namespace {

constexpr std::string_view k_architecture = "gemma4";
constexpr std::string_view k_token_embedding_name = "token_embd.weight";
constexpr std::string_view k_output_norm_name = "output_norm.weight";
constexpr int32_t k_block_count = 35;
constexpr int32_t k_context_length = 131072;
constexpr int32_t k_embedding_length = 1536;
constexpr int32_t k_embedding_length_per_layer_input = 256;
constexpr int32_t k_feed_forward_length = 6144;
constexpr int32_t k_head_count = 8;
constexpr int32_t k_head_count_kv = 1;
constexpr int32_t k_key_length = 512;
constexpr int32_t k_key_length_swa = 256;
constexpr int32_t k_value_length = 512;
constexpr int32_t k_value_length_swa = 256;
constexpr int32_t k_vocab_size = 262144;
constexpr int32_t k_sliding_window = 512;
constexpr int32_t k_shared_kv_layers = 20;
constexpr int32_t k_full_attention_interval = 5;
constexpr float k_layer_norm_rms_epsilon = 1e-6f;
constexpr int32_t k_rope_dimension_count = 512;
constexpr int32_t k_rope_dimension_count_swa = 256;
constexpr float k_rope_freq_base = 1000000.0f;
constexpr float k_rope_freq_base_swa = 10000.0f;
constexpr std::array<uint8_t, 35> k_sliding_window_pattern = {
    1u, 1u, 1u, 1u, 0u,
    1u, 1u, 1u, 1u, 0u,
    1u, 1u, 1u, 1u, 0u,
    1u, 1u, 1u, 1u, 0u,
    1u, 1u, 1u, 1u, 0u,
    1u, 1u, 1u, 1u, 0u,
    1u, 1u, 1u, 1u, 0u,
};

bool is_shared_kv_layer(const int32_t block_index) noexcept {
  return block_index >= (k_block_count - k_shared_kv_layers) && block_index < k_block_count;
}

}  // namespace

bool is_execution_architecture(const std::string_view architecture) noexcept {
  return architecture == k_architecture;
}

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept {
  uint32_t pattern_count = 0u;
  if (!loader.assign_i32("gemma4.context_length", model_out.params.n_ctx) ||
      !loader.assign_i32("gemma4.embedding_length", model_out.params.n_embd) ||
      !loader.assign_i32(
          "gemma4.embedding_length_per_layer_input", model_out.params.embd_length_per_layer_input) ||
      !loader.assign_i32_or_first_array_value("gemma4.feed_forward_length", model_out.params.n_ff) ||
      !loader.assign_i32("gemma4.attention.head_count", model_out.params.n_head) ||
      !loader.assign_i32("gemma4.attention.head_count_kv", model_out.params.n_head_kv) ||
      !loader.assign_i32("gemma4.attention.key_length", model_out.params.attention_key_length) ||
      !loader.assign_i32(
          "gemma4.attention.key_length_swa", model_out.params.attention_key_length_swa) ||
      !loader.assign_i32("gemma4.attention.value_length", model_out.params.attention_value_length) ||
      !loader.assign_i32(
          "gemma4.attention.value_length_swa", model_out.params.attention_value_length_swa) ||
      !loader.assign_i32("gemma4.block_count", model_out.params.n_layer) ||
      !loader.assign_i32("gemma4.vocab_size", model_out.params.n_vocab) ||
      !loader.assign_i32("gemma4.attention.sliding_window", model_out.params.attention_sliding_window) ||
      !loader.assign_i32(
          "gemma4.attention.shared_kv_layers", model_out.params.attention_shared_kv_layers) ||
      !loader.assign_i32("gemma4.rope.dimension_count", model_out.params.n_rot) ||
      !loader.assign_i32("gemma4.rope.dimension_count_swa", model_out.params.n_rot_swa) ||
      !loader.assign_f32(
          "gemma4.attention.layer_norm_rms_epsilon", model_out.params.attention_layer_norm_rms_epsilon) ||
      !loader.assign_f32("gemma4.final_logit_softcapping", model_out.params.final_logit_softcapping) ||
      !loader.assign_f32("gemma4.rope.freq_base", model_out.params.rope_freq_base) ||
      !loader.assign_f32("gemma4.rope.freq_base_swa", model_out.params.rope_freq_base_swa) ||
      !loader.copy_flag_array(
          "gemma4.attention.sliding_window_pattern",
          std::span<uint8_t>{model_out.params.attention_sliding_window_pattern_flags},
          pattern_count)) {
    return false;
  }

  model_out.params.attention_sliding_window_pattern_count = pattern_count;
  model_out.params.full_attention_interval = 5;
  model_out.params.n_embd_out = model_out.params.n_embd;
  model_out.params.tie_word_embeddings = true;
  return true;
}

namespace {

emel::error::type validate_contract(const emel::model::data & model_data,
                                    const bool strict_metadata) noexcept {
  const bool metadata_ok = strict_metadata
      ? model_data.n_layers == k_block_count &&
            model_data.params.n_layer == k_block_count &&
            model_data.params.n_ctx == k_context_length &&
            model_data.params.n_embd == k_embedding_length &&
            model_data.params.n_embd_out == k_embedding_length &&
            model_data.params.n_ff == k_feed_forward_length &&
            model_data.params.n_head == k_head_count &&
            model_data.params.n_head_kv == k_head_count_kv &&
            model_data.params.attention_key_length == k_key_length &&
            model_data.params.attention_key_length_swa == k_key_length_swa &&
            model_data.params.attention_value_length == k_value_length &&
            model_data.params.attention_value_length_swa == k_value_length_swa &&
            model_data.params.n_vocab == k_vocab_size &&
            model_data.params.embd_length_per_layer_input == k_embedding_length_per_layer_input &&
            model_data.params.full_attention_interval == k_full_attention_interval &&
            model_data.params.attention_sliding_window == k_sliding_window &&
            model_data.params.attention_shared_kv_layers == k_shared_kv_layers &&
            model_data.params.n_rot == k_rope_dimension_count &&
            model_data.params.n_rot_swa == k_rope_dimension_count_swa &&
            model_data.params.attention_layer_norm_rms_epsilon == k_layer_norm_rms_epsilon &&
            model_data.params.rope_freq_base == k_rope_freq_base &&
            model_data.params.rope_freq_base_swa == k_rope_freq_base_swa &&
            model_data.params.tie_word_embeddings &&
            model_data.params.attention_sliding_window_pattern_count ==
                k_sliding_window_pattern.size()
      : emel::model::architecture_name_view(model_data) == "gemma4" &&
            model_data.n_layers > 0 &&
            model_data.params.n_layer == model_data.n_layers &&
            model_data.params.n_ctx > 0 &&
            model_data.params.n_embd > 0 &&
            model_data.params.n_embd_out > 0 &&
            model_data.params.n_ff > 0 &&
            model_data.params.n_head > 0 &&
            model_data.params.n_head_kv > 0 &&
            model_data.params.attention_key_length > 0 &&
            model_data.params.attention_value_length > 0 &&
            model_data.params.n_vocab > 0 &&
            model_data.params.attention_layer_norm_rms_epsilon > 0.0f &&
            model_data.params.rope_freq_base > 0.0f &&
            model_data.params.tie_word_embeddings;
  if (!metadata_ok ||
      !emel::model::builder::detail::has_tensor_named(model_data, k_token_embedding_name) ||
      !emel::model::builder::detail::has_tensor_named(model_data, k_output_norm_name)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  if (strict_metadata) {
    for (size_t idx = 0; idx < k_sliding_window_pattern.size(); ++idx) {
      if (model_data.params.attention_sliding_window_pattern_flags[idx] !=
          k_sliding_window_pattern[idx]) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
      }
    }
  }

  for (int32_t block_index = 0; block_index < model_data.n_layers; ++block_index) {
    const bool common_ok =
        emel::model::builder::detail::require_block_tensor(
            model_data, block_index, "attn_norm.weight") &&
        emel::model::builder::detail::require_block_tensor(
            model_data, block_index, "attn_q.weight") &&
        emel::model::builder::detail::require_block_tensor(
            model_data, block_index, "attn_k.weight") &&
        emel::model::builder::detail::require_block_tensor(
            model_data, block_index, "attn_q_norm.weight") &&
        emel::model::builder::detail::require_block_tensor(
            model_data, block_index, "attn_k_norm.weight") &&
        emel::model::builder::detail::require_block_tensor(
            model_data, block_index, "attn_output.weight") &&
        emel::model::builder::detail::require_block_tensor(
            model_data, block_index, "ffn_norm.weight") &&
        emel::model::builder::detail::require_block_tensor(
            model_data, block_index, "ffn_gate.weight") &&
        emel::model::builder::detail::require_block_tensor(
            model_data, block_index, "ffn_down.weight") &&
        emel::model::builder::detail::require_block_tensor(
            model_data, block_index, "ffn_up.weight");
    if (!common_ok) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    if (!is_shared_kv_layer(block_index) &&
        !emel::model::builder::detail::require_block_tensor(
            model_data, block_index, "attn_v.weight")) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
  }

  return emel::error::cast(emel::model::loader::error::none);
}

}  // namespace

emel::error::type validate_builder_contract(const emel::model::data & model_data) noexcept {
  return validate_contract(model_data, false);
}

emel::error::type validate_execution_contract(const emel::model::data & model_data) noexcept {
  return validate_contract(model_data, true);
}

}  // namespace emel::model::gemma4::detail
