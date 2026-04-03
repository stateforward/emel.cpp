#include "emel/model/llama/detail.hpp"

#include "emel/model/builder/detail.hpp"

namespace emel::model::llama::detail {

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept {
  return loader.assign_i32("llama.context_length", model_out.params.n_ctx) &&
         loader.assign_i32("llama.embedding_length", model_out.params.n_embd) &&
         loader.assign_i32("llama.embedding_length_out", model_out.params.n_embd_out) &&
         loader.assign_i32("llama.feed_forward_length", model_out.params.n_ff) &&
         loader.assign_i32("llama.attention.head_count", model_out.params.n_head) &&
         loader.assign_i32("llama.attention.head_count_kv", model_out.params.n_head_kv) &&
         loader.assign_i32("llama.rope.dimension_count", model_out.params.n_rot) &&
         loader.assign_i32("llama.block_count", model_out.params.n_layer) &&
         loader.assign_i32("llama.vocab_size", model_out.params.n_vocab) &&
         loader.assign_f32(
             "llama.attention.layer_norm_epsilon", model_out.params.attention_layer_norm_epsilon) &&
         loader.assign_f32(
             "llama.attention.layer_norm_rms_epsilon",
             model_out.params.attention_layer_norm_rms_epsilon) &&
         loader.assign_f32("llama.attention.clamp_kqv", model_out.params.attention_clamp_kqv) &&
         loader.assign_f32("llama.attn_logit_softcapping", model_out.params.attn_logit_softcapping) &&
         loader.assign_f32(
             "llama.final_logit_softcapping", model_out.params.final_logit_softcapping) &&
         loader.assign_f32("llama.residual_scale", model_out.params.residual_scale) &&
         loader.assign_f32("llama.embedding_scale", model_out.params.embedding_scale) &&
         loader.assign_f32("llama.rope.freq_base", model_out.params.rope_freq_base) &&
         loader.assign_f32("llama.rope.freq_base_swa", model_out.params.rope_freq_base_swa);
}

emel::error::type validate_data(const emel::model::data & model_data) noexcept {
  return emel::model::builder::detail::validate_shared_execution_contract(model_data);
}

emel::error::type build_view(const emel::model::data & model_data,
                             emel::model::builder::detail::view & view_out) noexcept {
  using block_contract_kind = emel::model::builder::detail::block_contract_kind;
  const emel::model::builder::detail::view_contract contract{
      .block_contract = block_contract_kind::llama,
      .ties_output = false,
      .uses_qk_norm = false,
      .output_norm_name = "output_norm.weight",
      .shared_kv_layer_begin = -1,
  };
  return emel::model::builder::detail::build_view_for_contract(model_data, contract, view_out);
}

emel::error::type build_execution_view(
    const emel::model::data & model_data,
    emel::model::builder::detail::execution_view & view_out) noexcept {
  return emel::model::llama::detail::build_view(model_data, view_out);
}

}  // namespace emel::model::llama::detail
