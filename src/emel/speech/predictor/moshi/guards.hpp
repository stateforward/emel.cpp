#pragma once

#include "emel/kernel/detail.hpp"
#include "emel/memory/hybrid/errors.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/moshi/detail.hpp"
#include "emel/speech/predictor/moshi/context.hpp"
#include "emel/speech/predictor/moshi/detail.hpp"
#include "emel/speech/predictor/moshi/events.hpp"

namespace emel::speech::predictor::moshi::guard {

struct guard_bind_contract_valid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const auto &model = runtime_ev.request.model;
    const auto &lm = model.moshi_lm;
    int32_t max_delay = 0;
    for (uint32_t index = 0; index < lm.delay_count; ++index) {
      if (lm.delays[index] > max_delay) {
        max_delay = lm.delays[index];
      }
    }
    const int32_t codebook_count = lm.n_q + 1;
    const bool personaplex = lm.depformer_weights_per_step;
    const int32_t delayed_dep_q = personaplex ? lm.inference_dep_q : lm.dep_q;
    const int32_t needed_tokens = codebook_count - delayed_dep_q - 1;
    const int32_t row_count = max_delay + 2 + static_cast<int32_t>(personaplex);
    const bool inference_contract =
        !personaplex ||
        (lm.inference_dep_q > 0 && lm.inference_dep_q <= lm.dep_q &&
         lm.inference_prompt_token_count ==
             static_cast<uint32_t>(codebook_count) &&
         lm.inference_pre_text_silence_frames >= 0 &&
         lm.inference_post_text_silence_frames >= 0);
    return emel::model::moshi::detail::validate_execution_contract(model) ==
               emel::error::cast(emel::model::loader::error::none) &&
           codebook_count > 0 && codebook_count <= action::k_max_codebooks &&
           lm.dep_q > 0 && lm.dep_q < codebook_count && delayed_dep_q > 0 &&
           delayed_dep_q <= lm.dep_q && needed_tokens >= 0 && row_count > 0 &&
           row_count <= action::k_max_delay_rows && inference_contract &&
           lm.delay_count >= static_cast<uint32_t>(codebook_count) &&
           runtime_ev.request.max_sequences > 0 &&
           runtime_ev.request.max_blocks > 0 &&
           runtime_ev.request.block_tokens > 0 &&
           runtime_ev.request.sequence_id >= 0 &&
           runtime_ev.request.sequence_id < runtime_ev.request.max_sequences &&
           runtime_ev.request.codebook_capacity >= codebook_count &&
           runtime_ev.request.codebook_capacity <= action::k_max_codebooks &&
           runtime_ev.request.delay_cache_row_capacity >= row_count &&
           runtime_ev.request.delay_cache_row_capacity <=
               action::k_max_delay_rows &&
           ctx.policy.token_ungenerated < ctx.policy.token_zero &&
           ctx.policy.prediction_step_size > 0 &&
           ctx.policy.prediction_output_count > 0;
  }
};

struct guard_bind_contract_invalid {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_bind_contract_valid{}(runtime_ev, ctx);
  }
};

struct guard_personaplex_lmgen {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.model.moshi_lm.depformer_weights_per_step;
  }
};

struct guard_standard_lmgen {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_personaplex_lmgen{}(runtime_ev, ctx);
  }
};

struct guard_voice_contract_valid {
  bool operator()(const event::load_voice_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    emel::model::moshi::detail::execution_contract contract = {};
    const auto &voice_model = runtime_ev.request.voice_model;
    const auto result = emel::model::moshi::detail::build_execution_contract(
        voice_model, contract);
    const auto *embeddings = contract.voice.embeddings.tensor;
    const auto *cache = contract.voice.cache.tensor;
    const auto embedding_type =
        embeddings != nullptr ? static_cast<uint8_t>(embeddings->type) : 0u;
    const auto cache_type =
        cache != nullptr ? static_cast<uint8_t>(cache->type) : 0u;
    const bool embedding_type_ok =
        embedding_type == emel::kernel::detail::dtype_f32 ||
        embedding_type == emel::kernel::detail::dtype_f16 ||
        embedding_type == emel::kernel::detail::dtype_bf16;
    return ctx.session.model != nullptr &&
           ctx.session.model->moshi_lm.depformer_weights_per_step &&
           ctx.lmgen.delayed_dep_q > 0 &&
           ctx.lmgen.delayed_dep_q ==
               ctx.session.model->moshi_lm.inference_dep_q &&
           result == emel::error::cast(emel::model::loader::error::none) &&
           contract.component == emel::model::data::moshi_component::voice &&
           embeddings != nullptr && cache != nullptr &&
           embeddings->n_dims == 4 &&
           embeddings->dims[0] == ctx.session.model->moshi_lm.dim &&
           embeddings->dims[1] == 1 && embeddings->dims[2] == 1 &&
           embeddings->dims[3] > 0 &&
           embeddings->dims[3] <=
               static_cast<int64_t>(event::k_max_voice_embedding_dim) &&
           embedding_type_ok && cache->n_dims == 2 &&
           cache->dims[0] == ctx.lmgen.cache_row_count &&
           cache->dims[1] == ctx.lmgen.codebook_count &&
           cache_type == emel::kernel::detail::dtype_i32;
  }
};

struct guard_voice_contract_invalid {
  bool operator()(const event::load_voice_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_voice_contract_valid{}(runtime_ev, ctx);
  }
};

struct guard_memory_accepted {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.memory_accepted &&
           ev.ctx.memory_error == static_cast<int32_t>(emel::error::cast(
                                      emel::memory::hybrid::error::none));
  }
};

struct guard_graph_initialize_succeeded {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.graph_accepted &&
           runtime_ev.ctx.graph_error ==
               action::detail_ns::to_error(error::none);
  }
};

struct guard_graph_initialize_failed {
  bool operator()(const event::initialize_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_graph_initialize_succeeded{}(runtime_ev, ctx);
  }
};

struct guard_memory_rejected {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_memory_accepted{}(runtime_ev, ctx);
  }
};

struct guard_predict_request_valid {
  bool operator()(const event::predict_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const bool voice_ready =
        !ctx.voice.loaded || (ctx.voice.ready && ctx.voice.prompt_ready);
    return ctx.session.model != nullptr && voice_ready &&
           runtime_ev.request.model_tokens.size() ==
               static_cast<size_t>(ctx.lmgen.codebook_count) &&
           runtime_ev.request.planned_step_size ==
               ctx.policy.prediction_step_size &&
           runtime_ev.request.planned_output_count ==
               ctx.policy.prediction_output_count;
  }
};

struct guard_execute_request_valid {
  bool operator()(const event::execute_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const bool voice_ready =
        !ctx.voice.loaded || (ctx.voice.ready && ctx.voice.prompt_ready);
    return ctx.session.model != nullptr && voice_ready &&
           runtime_ev.request.model_tokens.size() ==
               static_cast<size_t>(ctx.lmgen.codebook_count);
  }
};

struct guard_execute_request_invalid {
  bool operator()(const event::execute_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_execute_request_valid{}(runtime_ev, ctx);
  }
};

struct guard_sample_request_valid {
  bool operator()(const event::sample_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return ctx.session.model != nullptr &&
           runtime_ev.request.audio_tokens_out.size() >=
               static_cast<size_t>(ctx.lmgen.generated_dep_q);
  }
};

struct guard_sample_request_invalid {
  bool operator()(const event::sample_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_sample_request_valid{}(runtime_ev, ctx);
  }
};

struct guard_predict_blocked_by_voice_prompt {
  bool operator()(const event::predict_run &,
                  const action::context &ctx) const noexcept {
    return ctx.session.model != nullptr && ctx.voice.loaded &&
           (!ctx.voice.ready || !ctx.voice.prompt_ready);
  }
};

struct guard_predict_request_shape_invalid {
  bool operator()(const event::predict_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_predict_request_valid{}(runtime_ev, ctx) &&
           !guard_predict_blocked_by_voice_prompt{}(runtime_ev, ctx);
  }
};

struct guard_capture_tokenizer_state_valid {
  bool operator()(const event::capture_tokenizer_state &ev,
                  const action::context &ctx) const noexcept {
    return ctx.session.model != nullptr &&
           ev.cache_out.size() ==
               static_cast<size_t>(ctx.lmgen.cache_row_count *
                                   ctx.lmgen.codebook_count);
  }
};

struct guard_capture_tokenizer_state_invalid {
  bool operator()(const event::capture_tokenizer_state &ev,
                  const action::context &ctx) const noexcept {
    return !guard_capture_tokenizer_state_valid{}(ev, ctx);
  }
};

struct guard_voice_prefill_request_valid {
  bool operator()(const event::prefill_voice_run &,
                  const action::context &ctx) const noexcept {
    return ctx.session.model != nullptr && ctx.voice.loaded &&
           !ctx.voice.ready && ctx.voice.prompt_frame_index >= 0 &&
           ctx.voice.prompt_frame_index < ctx.voice.prompt_frame_count &&
           ctx.voice.embedding_dim > 0 &&
           ctx.voice.embedding_dim <=
               static_cast<int32_t>(event::k_max_voice_embedding_dim);
  }
};

struct guard_voice_prefill_request_invalid {
  bool operator()(const event::prefill_voice_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_voice_prefill_request_valid{}(runtime_ev, ctx);
  }
};

struct guard_voice_embedding_frame_f32 {
  bool operator()(const event::prefill_voice_run &,
                  const action::context &ctx) const noexcept {
    return ctx.voice.contract.voice.embeddings.tensor != nullptr &&
           static_cast<uint8_t>(
               ctx.voice.contract.voice.embeddings.tensor->type) ==
               emel::kernel::detail::dtype_f32;
  }
};

struct guard_voice_embedding_frame_f16 {
  bool operator()(const event::prefill_voice_run &,
                  const action::context &ctx) const noexcept {
    return ctx.voice.contract.voice.embeddings.tensor != nullptr &&
           static_cast<uint8_t>(
               ctx.voice.contract.voice.embeddings.tensor->type) ==
               emel::kernel::detail::dtype_f16;
  }
};

struct guard_voice_embedding_frame_bf16 {
  bool operator()(const event::prefill_voice_run &,
                  const action::context &ctx) const noexcept {
    return ctx.voice.contract.voice.embeddings.tensor != nullptr &&
           static_cast<uint8_t>(
               ctx.voice.contract.voice.embeddings.tensor->type) ==
               emel::kernel::detail::dtype_bf16;
  }
};

struct guard_voice_embedding_frame_loaded {
  bool operator()(const event::prefill_voice_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.embedding_frame_ok;
  }
};

struct guard_voice_embedding_frame_failed {
  bool operator()(const event::prefill_voice_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_voice_embedding_frame_loaded{}(runtime_ev, ctx);
  }
};

struct guard_voice_prefill_complete {
  bool operator()(const event::prefill_voice_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.ctx.complete;
  }
};

struct guard_voice_prefill_pending {
  bool operator()(const event::prefill_voice_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_voice_prefill_complete{}(runtime_ev, ctx);
  }
};

struct guard_personaplex_prompt_begin_valid {
  bool operator()(const event::begin_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const int32_t frame_count = runtime_ev.request.pre_text_silence_frames +
                                runtime_ev.request.text_token_count +
                                runtime_ev.request.post_text_silence_frames;
    return ctx.session.model != nullptr && ctx.voice.loaded &&
           ctx.voice.ready && !ctx.voice.prompt_ready &&
           !ctx.voice.prompt_started &&
           runtime_ev.request.text_token_count >= 0 &&
           runtime_ev.request.pre_text_silence_frames >= 0 &&
           runtime_ev.request.post_text_silence_frames >= 0 &&
           frame_count >= 0 && frame_count <= action::k_max_delay_rows &&
           ctx.session.model->moshi_lm.inference_prompt_token_count ==
               static_cast<uint32_t>(ctx.lmgen.codebook_count);
  }
};

struct guard_personaplex_prompt_begin_nonempty_valid {
  bool operator()(const event::begin_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const int32_t frame_count = runtime_ev.request.pre_text_silence_frames +
                                runtime_ev.request.text_token_count +
                                runtime_ev.request.post_text_silence_frames;
    return guard_personaplex_prompt_begin_valid{}(runtime_ev, ctx) &&
           frame_count > 0;
  }
};

struct guard_personaplex_prompt_begin_empty_valid {
  bool operator()(const event::begin_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    const int32_t frame_count = runtime_ev.request.pre_text_silence_frames +
                                runtime_ev.request.text_token_count +
                                runtime_ev.request.post_text_silence_frames;
    const auto *model = ctx.session.model;
    const int32_t default_frame_count =
        model == nullptr
            ? -1
            : model->moshi_lm.inference_pre_text_silence_frames +
                  model->moshi_lm.inference_post_text_silence_frames;
    return guard_personaplex_prompt_begin_valid{}(runtime_ev, ctx) &&
           frame_count == 0 && default_frame_count > 0 &&
           default_frame_count <= action::k_max_delay_rows;
  }
};

struct guard_personaplex_prompt_begin_invalid {
  bool operator()(const event::begin_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_personaplex_prompt_begin_nonempty_valid{}(runtime_ev, ctx) &&
           !guard_personaplex_prompt_begin_empty_valid{}(runtime_ev, ctx);
  }
};

struct guard_personaplex_prompt_prefill_request_valid {
  bool operator()(const event::prefill_personaplex_prompt_run &,
                  const action::context &ctx) const noexcept {
    const int32_t frame_count = ctx.voice.pre_text_silence_remaining +
                                ctx.voice.text_tokens_remaining +
                                ctx.voice.post_text_silence_remaining;
    return ctx.session.model != nullptr && ctx.voice.loaded &&
           ctx.voice.ready && ctx.voice.prompt_started &&
           !ctx.voice.prompt_ready && frame_count > 0;
  }
};

struct guard_personaplex_prompt_prefill_request_invalid {
  bool operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_personaplex_prompt_prefill_request_valid{}(runtime_ev, ctx);
  }
};

struct guard_personaplex_prompt_pre_silence_pending {
  bool operator()(const event::prefill_personaplex_prompt_run &,
                  const action::context &ctx) const noexcept {
    return ctx.voice.pre_text_silence_remaining > 0;
  }
};

struct guard_personaplex_prompt_text_pending {
  bool operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return ctx.voice.pre_text_silence_remaining == 0 &&
           ctx.voice.text_tokens_remaining > 0 &&
           runtime_ev.request.text_token >= 0 &&
           runtime_ev.request.text_token < ctx.session.text_card;
  }
};

struct guard_personaplex_prompt_post_silence_pending {
  bool operator()(const event::prefill_personaplex_prompt_run &,
                  const action::context &ctx) const noexcept {
    return ctx.voice.pre_text_silence_remaining == 0 &&
           ctx.voice.text_tokens_remaining == 0 &&
           ctx.voice.post_text_silence_remaining > 0;
  }
};

struct guard_personaplex_prompt_phase_invalid {
  bool operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_personaplex_prompt_pre_silence_pending{}(runtime_ev, ctx) &&
           !guard_personaplex_prompt_text_pending{}(runtime_ev, ctx) &&
           !guard_personaplex_prompt_post_silence_pending{}(runtime_ev, ctx);
  }
};

struct guard_personaplex_prompt_complete {
  bool operator()(const event::prefill_personaplex_prompt_run &,
                  const action::context &ctx) const noexcept {
    return ctx.voice.pre_text_silence_remaining == 0 &&
           ctx.voice.text_tokens_remaining == 0 &&
           ctx.voice.post_text_silence_remaining == 0;
  }
};

struct guard_personaplex_prompt_pending {
  bool operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_personaplex_prompt_complete{}(runtime_ev, ctx);
  }
};

struct guard_has_personaplex_prompt_complete_out {
  bool operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.complete_out != nullptr;
  }
};

struct guard_no_personaplex_prompt_complete_out {
  bool operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_personaplex_prompt_complete_out{}(runtime_ev, ctx);
  }
};

struct guard_has_personaplex_prompt_remaining_out {
  bool operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.remaining_frames_out != nullptr;
  }
};

struct guard_no_personaplex_prompt_remaining_out {
  bool operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_personaplex_prompt_remaining_out{}(runtime_ev, ctx);
  }
};

struct guard_has_voice_complete_out {
  bool operator()(const event::prefill_voice_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.complete_out != nullptr;
  }
};

struct guard_no_voice_complete_out {
  bool operator()(const event::prefill_voice_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_voice_complete_out{}(runtime_ev, ctx);
  }
};

struct guard_has_voice_remaining_out {
  bool operator()(const event::prefill_voice_run &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.remaining_frames_out != nullptr;
  }
};

struct guard_no_voice_remaining_out {
  bool operator()(const event::prefill_voice_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_voice_remaining_out{}(runtime_ev, ctx);
  }
};

struct guard_graph_step_accepted {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.graph_accepted &&
           ev.ctx.graph_error == action::detail_ns::to_error(error::none);
  }
};

struct guard_graph_step_rejected {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_graph_step_accepted{}(runtime_ev, ctx);
  }
};

struct guard_graph_step_accepted_personaplex_prompt_pre_silence {
  bool operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_graph_step_accepted{}(runtime_ev, ctx) &&
           guard_personaplex_prompt_pre_silence_pending{}(runtime_ev, ctx);
  }
};

struct guard_graph_step_accepted_personaplex_prompt_text {
  bool operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_graph_step_accepted{}(runtime_ev, ctx) &&
           guard_personaplex_prompt_text_pending{}(runtime_ev, ctx);
  }
};

struct guard_graph_step_accepted_personaplex_prompt_post_silence {
  bool operator()(const event::prefill_personaplex_prompt_run &runtime_ev,
                  const action::context &ctx) const noexcept {
    return guard_graph_step_accepted{}(runtime_ev, ctx) &&
           guard_personaplex_prompt_post_silence_pending{}(runtime_ev, ctx);
  }
};

struct guard_has_graph_error_out {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    const auto &ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.request.graph_error_out != nullptr;
  }
};

struct guard_no_graph_error_out {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_graph_error_out{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_has_error_out {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return runtime_ev.request.error_out != nullptr;
  }
};

template <class runtime_event_type> struct guard_no_error_out {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_error_out<runtime_event_type>{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_has_done_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_done);
  }
};

template <class runtime_event_type> struct guard_no_done_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_done_callback<runtime_event_type>{}(runtime_ev, ctx);
  }
};

template <class runtime_event_type> struct guard_has_error_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &) const noexcept {
    return static_cast<bool>(runtime_ev.request.on_error);
  }
};

template <class runtime_event_type> struct guard_no_error_callback {
  bool operator()(const runtime_event_type &runtime_ev,
                  const action::context &ctx) const noexcept {
    return !guard_has_error_callback<runtime_event_type>{}(runtime_ev, ctx);
  }
};

struct guard_unexpected_error_out_present {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &) const noexcept {
    if constexpr (requires {
                    ev.ctx.err;
                    ev.request.error_out;
                  }) {
      return ev.request.error_out != nullptr;
    } else {
      return false;
    }
  }
};

struct guard_unexpected_error_out_absent {
  template <class event_type>
  bool operator()(const event_type &ev,
                  const action::context &ctx) const noexcept {
    return !guard_unexpected_error_out_present{}(ev, ctx);
  }
};

} // namespace emel::speech::predictor::moshi::guard
