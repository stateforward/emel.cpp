#include "emel/model/lfm2/detail.hpp"

#include <array>

#include "emel/model/detail.hpp"
#include "emel/model/llama/detail.hpp"
#include "emel/model/loader/errors.hpp"

namespace emel::model::lfm2::detail {

namespace {

constexpr std::string_view k_architecture = "lfm2";
constexpr std::string_view k_token_embedding_name = "token_embd.weight";
constexpr std::string_view k_output_norm_name = "token_embd_norm.weight";
constexpr int32_t k_block_count = 16;
constexpr int32_t k_context_length = 128000;
constexpr int32_t k_embedding_length = 2048;
constexpr int32_t k_head_count = 32;
constexpr int32_t k_head_count_kv = 8;
constexpr int32_t k_vocab_size = 65536;
constexpr int32_t k_shortconv_l_cache = 3;
constexpr float k_rope_freq_base = 1000000.0f;
constexpr std::array<int32_t, 6> k_attention_layers = {2, 5, 8, 10, 12, 14};

bool is_attention_layer(const int32_t block_index) noexcept {
  for (const int32_t candidate : k_attention_layers) {
    if (candidate == block_index) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool is_execution_architecture(const std::string_view architecture) noexcept {
  return architecture == k_architecture;
}

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept {
  if (!loader.assign_i32("lfm2.context_length", model_out.params.n_ctx) ||
      !loader.assign_i32("lfm2.embedding_length", model_out.params.n_embd) ||
      !loader.assign_i32("lfm2.feed_forward_length", model_out.params.n_ff) ||
      !loader.assign_i32("lfm2.attention.head_count", model_out.params.n_head) ||
      !loader.assign_i32("lfm2.block_count", model_out.params.n_layer) ||
      !loader.assign_i32("lfm2.vocab_size", model_out.params.n_vocab) ||
      !loader.assign_i32("lfm2.shortconv.l_cache", model_out.params.shortconv_l_cache) ||
      !loader.assign_f32(
          "lfm2.attention.layer_norm_rms_epsilon", model_out.params.attention_layer_norm_rms_epsilon) ||
      !loader.assign_f32("lfm2.rope.freq_base", model_out.params.rope_freq_base) ||
      !loader.assign_first_nonzero_i32_from_array(
          "lfm2.attention.head_count_kv", model_out.params.n_head_kv)) {
    return false;
  }

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
            model_data.params.n_head == k_head_count &&
            model_data.params.n_head_kv == k_head_count_kv &&
            model_data.params.n_vocab == k_vocab_size &&
            model_data.params.shortconv_l_cache == k_shortconv_l_cache &&
            model_data.params.rope_freq_base == k_rope_freq_base
      : emel::model::architecture_name_view(model_data) == "lfm2" &&
            model_data.n_layers > 0 &&
            model_data.params.n_layer == model_data.n_layers &&
            model_data.params.n_ctx > 0 &&
            model_data.params.n_embd > 0 &&
            model_data.params.n_head > 0 &&
            model_data.params.n_head_kv > 0 &&
            model_data.params.n_vocab > 0 &&
            model_data.params.shortconv_l_cache > 0 &&
            model_data.params.rope_freq_base > 0.0f;
  if (!metadata_ok ||
      !emel::model::llama::detail::has_tensor_named(model_data, k_token_embedding_name) ||
      !emel::model::llama::detail::has_tensor_named(model_data, k_output_norm_name)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  for (int32_t block_index = 0; block_index < model_data.n_layers; ++block_index) {
    const bool common_ok =
        emel::model::llama::detail::require_block_tensor(
            model_data, block_index, "attn_norm.weight") &&
        emel::model::llama::detail::require_block_tensor(
            model_data, block_index, "ffn_norm.weight") &&
        emel::model::llama::detail::require_block_tensor(
            model_data, block_index, "ffn_gate.weight") &&
        emel::model::llama::detail::require_block_tensor(
            model_data, block_index, "ffn_down.weight") &&
        emel::model::llama::detail::require_block_tensor(
            model_data, block_index, "ffn_up.weight");
    if (!common_ok) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    const bool attention_layer = is_attention_layer(block_index);
    const bool hybrid_ok =
        attention_layer
            ? emel::model::llama::detail::require_block_tensor(
                  model_data, block_index, "attn_q.weight") &&
                  emel::model::llama::detail::require_block_tensor(
                      model_data, block_index, "attn_k.weight") &&
                  emel::model::llama::detail::require_block_tensor(
                      model_data, block_index, "attn_v.weight") &&
                  emel::model::llama::detail::require_block_tensor(
                      model_data, block_index, "attn_q_norm.weight") &&
                  emel::model::llama::detail::require_block_tensor(
                      model_data, block_index, "attn_k_norm.weight") &&
                  emel::model::llama::detail::require_block_tensor(
                      model_data, block_index, "attn_output.weight") &&
                  emel::model::llama::detail::reject_block_tensor(
                      model_data, block_index, "shortconv.conv.weight") &&
                  emel::model::llama::detail::reject_block_tensor(
                      model_data, block_index, "shortconv.in_proj.weight") &&
                  emel::model::llama::detail::reject_block_tensor(
                      model_data, block_index, "shortconv.out_proj.weight")
            : emel::model::llama::detail::require_block_tensor(
                  model_data, block_index, "shortconv.conv.weight") &&
                  emel::model::llama::detail::require_block_tensor(
                      model_data, block_index, "shortconv.in_proj.weight") &&
                  emel::model::llama::detail::require_block_tensor(
                      model_data, block_index, "shortconv.out_proj.weight") &&
                  emel::model::llama::detail::reject_block_tensor(
                      model_data, block_index, "attn_q.weight") &&
                  emel::model::llama::detail::reject_block_tensor(
                      model_data, block_index, "attn_k.weight") &&
                  emel::model::llama::detail::reject_block_tensor(
                      model_data, block_index, "attn_v.weight") &&
                  emel::model::llama::detail::reject_block_tensor(
                      model_data, block_index, "attn_q_norm.weight") &&
                  emel::model::llama::detail::reject_block_tensor(
                      model_data, block_index, "attn_k_norm.weight") &&
                  emel::model::llama::detail::reject_block_tensor(
                      model_data, block_index, "attn_output.weight");
    if (!hybrid_ok) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
  }

  return emel::error::cast(emel::model::loader::error::none);
}

}  // namespace

emel::error::type validate_builder_contract(const emel::model::data & model_data) noexcept {
  return validate_contract(model_data, false);
}

emel::error::type validate_data(const emel::model::data & model_data) noexcept {
  return validate_builder_contract(model_data);
}

emel::error::type validate_execution_contract(const emel::model::data & model_data) noexcept {
  return validate_contract(model_data, true);
}

}  // namespace emel::model::lfm2::detail
