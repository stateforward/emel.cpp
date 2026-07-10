#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "emel/speech/transcriber/context.hpp"
#include "emel/speech/transcriber/detail.hpp"
#include "emel/speech/transcriber/events.hpp"

namespace emel::speech::transcriber::action {

struct effect_begin_initialize {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::none);
  }
};

struct effect_reject_initialize {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::invalid_request);
  }
};

struct effect_mark_tokenizer_invalid {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::tokenizer_invalid);
  }
};

struct effect_mark_unsupported_model {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::unsupported_model);
  }
};

struct effect_mark_initialize_backend_error {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::backend);
  }
};

struct effect_store_initialize_error {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_store_initialize_success {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_initialize_done {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::initialize_done{&runtime_ev.request});
  }
};

struct effect_emit_initialize_error {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(
        events::initialize_error{&runtime_ev.request, runtime_ev.ctx.err});
  }
};

struct effect_begin_recognize {
  void operator()(const event::recognize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::none);
    runtime_ev.ctx.encoder_accepted = false;
    runtime_ev.ctx.decoder_accepted = false;
    runtime_ev.ctx.detokenize_accepted = false;
    runtime_ev.ctx.encoder_frame_count = 0;
    runtime_ev.ctx.encoder_width = 0;
    runtime_ev.ctx.generated_token_count = 0;
    runtime_ev.ctx.selected_token = 0;
    runtime_ev.ctx.confidence = 0.0f;
    runtime_ev.ctx.transcript_size = 0;
    runtime_ev.ctx.encoder_digest = 0u;
    runtime_ev.ctx.decoder_digest = 0u;
    runtime_ev.request.transcript_size_out = 0;
    runtime_ev.request.generated_token_count_out = 0;
  }
};

struct effect_reject_recognize {
  void operator()(const event::recognize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::invalid_request);
    runtime_ev.request.transcript_size_out = 0;
    runtime_ev.request.generated_token_count_out = 0;
  }
};

struct effect_mark_uninitialized {
  void operator()(const event::recognize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::uninitialized);
  }
};

struct effect_mark_backend_error {
  void operator()(const event::recognize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail::to_error(error::backend);
  }
};

// Pipeline effects: dispatch into the injected component actors that the sm
// wrapper owns. Each effect drives one already-chosen phase; which phase runs
// (and whether the pipeline runs at all) is decided by guards in the transition
// table, never here.
struct effect_encode {
  void operator()(const event::recognize_run &runtime_ev,
                  context &ctx) const noexcept {
    speech::encoder::event::encode encode_ev{
        ctx.deps.encoder_contract,
        runtime_ev.request.pcm,
        runtime_ev.request.sample_rate,
        runtime_ev.request.channel_count,
        runtime_ev.request.storage.encoder_workspace,
        runtime_ev.request.storage.encoder_state,
        runtime_ev.ctx.encoder_frame_count,
        runtime_ev.ctx.encoder_width,
        runtime_ev.ctx.encoder_digest};
    runtime_ev.ctx.encoder_accepted = ctx.encoder->process_event(encode_ev);
    runtime_ev.ctx.err = detail::to_error(error::none);
  }
};

struct effect_decode {
  void operator()(const event::recognize_run &runtime_ev,
                  context &ctx) const noexcept {
    const auto encoder_state_size =
        static_cast<size_t>(runtime_ev.ctx.encoder_frame_count) *
        static_cast<size_t>(ctx.deps.decoder_contract.embedding_length);
    speech::decoder::event::decode decode_ev{
        ctx.deps.decoder_contract,
        std::span<const float>{runtime_ev.request.storage.encoder_state.data(),
                               encoder_state_size},
        runtime_ev.ctx.encoder_frame_count,
        ctx.deps.decode_policy,
        runtime_ev.request.storage.generated_tokens,
        runtime_ev.ctx.generated_token_count,
        runtime_ev.request.storage.decoder_workspace,
        runtime_ev.request.storage.logits,
        runtime_ev.ctx.selected_token,
        runtime_ev.ctx.confidence,
        runtime_ev.ctx.decoder_digest};
    runtime_ev.ctx.decoder_accepted = ctx.decoder->process_event(decode_ev);
    runtime_ev.ctx.err = detail::to_error(error::none);
  }
};

struct effect_detokenize {
  void operator()(const event::recognize_run &runtime_ev,
                  context &ctx) const noexcept {
    speech::tokenizer::event::detokenize detokenize_ev{
        runtime_ev.request.tokenizer.model_json,
        std::span<const int32_t>{
            runtime_ev.request.storage.generated_tokens.data(),
            static_cast<size_t>(runtime_ev.ctx.generated_token_count),
        },
        runtime_ev.request.transcript, runtime_ev.ctx.transcript_size};
    runtime_ev.ctx.detokenize_accepted =
        ctx.tokenizer->process_event(detokenize_ev);
    runtime_ev.ctx.err = detail::to_error(error::none);
  }
};

struct effect_publish_recognition_outputs {
  void operator()(const event::recognize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.transcript_size_out = runtime_ev.ctx.transcript_size;
    runtime_ev.request.selected_token_out = runtime_ev.ctx.selected_token;
    runtime_ev.request.confidence_out = runtime_ev.ctx.confidence;
    runtime_ev.request.encoder_frame_count_out =
        runtime_ev.ctx.encoder_frame_count;
    runtime_ev.request.encoder_width_out = runtime_ev.ctx.encoder_width;
    runtime_ev.request.encoder_digest_out = runtime_ev.ctx.encoder_digest;
    runtime_ev.request.decoder_digest_out = runtime_ev.ctx.decoder_digest;
    runtime_ev.request.generated_token_count_out =
        runtime_ev.ctx.generated_token_count;
  }
};

struct effect_store_recognize_success {
  void operator()(const event::recognize_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_store_recognize_error {
  void operator()(const event::recognize_run &runtime_ev,
                  context &) const noexcept {
    *runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_emit_recognize_done {
  void operator()(const event::recognize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_done(events::recognition_done{
        .request = &runtime_ev.request,
        .transcript_size = runtime_ev.ctx.transcript_size,
        .selected_token = runtime_ev.ctx.selected_token,
        .confidence = runtime_ev.ctx.confidence,
        .encoder_frame_count = runtime_ev.ctx.encoder_frame_count,
        .encoder_width = runtime_ev.ctx.encoder_width,
        .generated_token_count = runtime_ev.ctx.generated_token_count,
        .encoder_digest = runtime_ev.ctx.encoder_digest,
        .decoder_digest = runtime_ev.ctx.decoder_digest,
    });
  }
};

struct effect_emit_recognize_error {
  void operator()(const event::recognize_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.request.on_error(
        events::recognition_error{&runtime_ev.request, runtime_ev.ctx.err});
  }
};

struct effect_on_unexpected {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &, context &) const noexcept {}
};

inline constexpr effect_begin_initialize effect_begin_initialize{};
inline constexpr effect_reject_initialize effect_reject_initialize{};
inline constexpr effect_mark_tokenizer_invalid effect_mark_tokenizer_invalid{};
inline constexpr effect_mark_unsupported_model effect_mark_unsupported_model{};
inline constexpr effect_mark_initialize_backend_error
    effect_mark_initialize_backend_error{};
inline constexpr effect_store_initialize_error effect_store_initialize_error{};
inline constexpr effect_store_initialize_success
    effect_store_initialize_success{};
inline constexpr effect_emit_initialize_done effect_emit_initialize_done{};
inline constexpr effect_emit_initialize_error effect_emit_initialize_error{};
inline constexpr effect_begin_recognize effect_begin_recognize{};
inline constexpr effect_encode effect_encode{};
inline constexpr effect_decode effect_decode{};
inline constexpr effect_detokenize effect_detokenize{};
inline constexpr effect_reject_recognize effect_reject_recognize{};
inline constexpr effect_mark_uninitialized effect_mark_uninitialized{};
inline constexpr effect_mark_backend_error effect_mark_backend_error{};
inline constexpr effect_publish_recognition_outputs
    effect_publish_recognition_outputs{};
inline constexpr effect_store_recognize_success
    effect_store_recognize_success{};
inline constexpr effect_store_recognize_error effect_store_recognize_error{};
inline constexpr effect_emit_recognize_done effect_emit_recognize_done{};
inline constexpr effect_emit_recognize_error effect_emit_recognize_error{};
inline constexpr effect_on_unexpected effect_on_unexpected{};

} // namespace emel::speech::transcriber::action
