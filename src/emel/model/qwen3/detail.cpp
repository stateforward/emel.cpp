#include "emel/model/qwen3/detail.hpp"

#include "emel/model/builder/detail.hpp"

namespace emel::model::qwen3::detail {

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept {
  int32_t key_length = 0;
  int32_t value_length = 0;
  if (!loader.assign_i32("qwen3.context_length", model_out.params.n_ctx) ||
      !loader.assign_i32("qwen3.embedding_length", model_out.params.n_embd) ||
      !loader.assign_i32("qwen3.feed_forward_length", model_out.params.n_ff) ||
      !loader.assign_i32("qwen3.attention.head_count", model_out.params.n_head) ||
      !loader.assign_i32("qwen3.attention.head_count_kv", model_out.params.n_head_kv) ||
      !loader.assign_i32("qwen3.attention.key_length", key_length) ||
      !loader.assign_i32("qwen3.attention.value_length", value_length) ||
      !loader.assign_i32("qwen3.block_count", model_out.params.n_layer) ||
      !loader.assign_f32(
          "qwen3.attention.layer_norm_rms_epsilon", model_out.params.attention_layer_norm_rms_epsilon) ||
      !loader.assign_f32("qwen3.rope.freq_base", model_out.params.rope_freq_base)) {
    return false;
  }

  model_out.params.attention_key_length = key_length;
  model_out.params.attention_value_length = value_length;
  model_out.params.n_embd_out = model_out.params.n_embd;
  model_out.params.tie_word_embeddings = true;
  if (model_out.params.n_rot == 0) {
    model_out.params.n_rot = key_length;
  }

  return key_length > 0 && value_length > 0;
}

emel::error::type validate_data(const emel::model::data & model_data) noexcept {
  return emel::model::builder::detail::validate_shared_execution_contract(model_data);
}

emel::error::type build_view(const emel::model::data & model_data,
                             emel::model::builder::detail::view & view_out) noexcept {
  using block_contract_kind = emel::model::builder::detail::block_contract_kind;
  const emel::model::builder::detail::view_contract contract{
      .block_contract = block_contract_kind::qwen3,
      .ties_output = true,
      .uses_qk_norm = true,
      .output_norm_name = "output_norm.weight",
      .shared_kv_layer_begin = -1,
  };
  return emel::model::builder::detail::build_view_for_contract(model_data, contract, view_out);
}

}  // namespace emel::model::qwen3::detail
