#pragma once

#include <algorithm>

#include "emel/speech/generator/moshi/personaplex/session/context.hpp"
#include "emel/speech/generator/moshi/personaplex/session/events.hpp"

namespace emel::speech::generator::moshi::personaplex::session::action {

namespace detail_ns {

inline emel::error::type to_error(const error::code value) noexcept {
  return emel::error::cast(value);
}

template <class runtime_event_type>
void effect_store_error(const runtime_event_type &runtime_ev,
                        const error::code value) noexcept {
  runtime_ev.ctx.err = to_error(value);
  runtime_ev.request.error_out = runtime_ev.ctx.err;
}

} // namespace detail_ns

struct effect_initialize_encoder {
  static void effect_capture_codec_initialize(
      event::initialize_ctx &runtime,
      const mimi::events::initialize_done &done) noexcept {
    runtime.frame_samples = done.frame_samples;
    runtime.mimi_n_q = done.n_q;
  }

  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    mimi::event::initialize request{
        runtime_ev.request.mimi_model,
        runtime_ev.request.encoder_storage.prepared,
        runtime_ev.request.encoder_storage.state,
        runtime_ev.request.encoder_storage.workspace,
        runtime_ev.request.encoder_storage.frame};
    request.error_out = &runtime_ev.ctx.child_err;
    request.on_done =
        emel::callback<void(const mimi::events::initialize_done &)>::from<
            event::initialize_ctx,
            &effect_initialize_encoder::effect_capture_codec_initialize>(
            runtime_ev.ctx);
    runtime_ev.ctx.child_accepted = ctx.encoder.process_event(request);
  }
};

struct effect_initialize_decoder {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    mimi::event::initialize request{
        runtime_ev.request.mimi_model,
        runtime_ev.request.decoder_storage.prepared,
        runtime_ev.request.decoder_storage.state,
        runtime_ev.request.decoder_storage.workspace,
        runtime_ev.request.decoder_storage.frame};
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted = ctx.decoder.process_event(request);
  }
};

struct effect_initialize_executor {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    executor::event::initialize request{runtime_ev.request.lm_model};
    request.sampling_enabled = runtime_ev.request.sampling.enabled;
    request.sampling_consume_forced_text =
        runtime_ev.request.sampling.consume_forced_text;
    request.sampling_audio_temperature =
        runtime_ev.request.sampling.audio_temperature;
    request.sampling_text_temperature =
        runtime_ev.request.sampling.text_temperature;
    request.sampling_audio_top_k = runtime_ev.request.sampling.audio_top_k;
    request.sampling_text_top_k = runtime_ev.request.sampling.text_top_k;
    request.sampling_seed = runtime_ev.request.sampling.seed;
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted = ctx.graph_executor.process_event(request);
  }
};

struct effect_initialize_generator {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    moshi::event::initialize request{runtime_ev.request.lm_model};
    request.max_blocks = runtime_ev.request.max_blocks;
    request.block_tokens = runtime_ev.request.block_tokens;
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted = ctx.generator.process_event(request);
  }
};

struct effect_load_voice {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    moshi::event::load_voice request{runtime_ev.request.voice_model};
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted = ctx.generator.process_event(request);
  }
};

struct effect_publish_initialize_done {
  void operator()(const event::initialize_run &runtime_ev,
                  context &ctx) const noexcept {
    ctx.frame_samples = runtime_ev.ctx.frame_samples;
    ctx.public_n_q = runtime_ev.ctx.mimi_n_q;
    runtime_ev.ctx.err = detail_ns::to_error(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_fail_initialize_invalid {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    detail_ns::effect_store_error(runtime_ev, error::invalid_request);
  }
};

template <error::code error_code> struct effect_fail_initialize_child {
  void operator()(const event::initialize_run &runtime_ev,
                  context &) const noexcept {
    detail_ns::effect_store_error(runtime_ev, error_code);
  }
};

struct effect_prefill_voice {
  void operator()(const event::advance_voice_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    runtime_ev.ctx.graph_err = 0;
    runtime_ev.ctx.complete = false;
    runtime_ev.ctx.remaining_frames = -1;
    moshi::event::prefill_voice request{};
    request.error_out = &runtime_ev.ctx.child_err;
    request.graph_error_out = &runtime_ev.ctx.graph_err;
    request.complete_out = &runtime_ev.ctx.complete;
    request.remaining_frames_out = &runtime_ev.ctx.remaining_frames;
    runtime_ev.ctx.child_accepted = ctx.generator.process_event(request);
  }
};

struct effect_begin_prompt {
  void operator()(const event::advance_voice_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    moshi::event::begin_personaplex_prompt request{};
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted = ctx.generator.process_event(request);
  }
};

struct effect_publish_advance_voice_done {
  void operator()(const event::advance_voice_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <error::code error_code> struct effect_fail_advance_voice {
  void operator()(const event::advance_voice_run &runtime_ev,
                  context &) const noexcept {
    detail_ns::effect_store_error(runtime_ev, error_code);
  }
};

struct effect_prefill_prompt {
  void operator()(const event::advance_prompt_run &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    runtime_ev.ctx.graph_err = 0;
    runtime_ev.ctx.complete = false;
    runtime_ev.ctx.remaining_frames = -1;
    moshi::event::prefill_personaplex_prompt request{};
    request.text_token = runtime_ev.request.text_token;
    request.error_out = &runtime_ev.ctx.child_err;
    request.graph_error_out = &runtime_ev.ctx.graph_err;
    request.complete_out = &runtime_ev.ctx.complete;
    request.remaining_frames_out = &runtime_ev.ctx.remaining_frames;
    runtime_ev.ctx.child_accepted = ctx.generator.process_event(request);
  }
};

struct effect_publish_advance_prompt_done {
  void operator()(const event::advance_prompt_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

template <error::code error_code> struct effect_fail_advance_prompt {
  void operator()(const event::advance_prompt_run &runtime_ev,
                  context &) const noexcept {
    detail_ns::effect_store_error(runtime_ev, error_code);
  }
};

template <class runtime_event_type> struct effect_encode_frame {
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    mimi::event::encode_frame request{
        runtime_ev.request.payload.pcm,
        runtime_ev.request.payload.input_codes_out.first(
            static_cast<size_t>(ctx.public_n_q))};
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted = ctx.encoder.process_event(request);
  }
};

template <class runtime_event_type> struct effect_generate_frame {
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    runtime_ev.ctx.graph_err = 0;
    runtime_ev.ctx.produced = false;
    std::fill_n(runtime_ev.request.payload.output_codes_out.data(),
                static_cast<size_t>(ctx.public_n_q), -1);
    moshi::event::step request{
        runtime_ev.request.payload.input_codes_out.first(
            static_cast<size_t>(ctx.public_n_q)),
        runtime_ev.request.payload.output_codes_out.first(
            static_cast<size_t>(ctx.public_n_q)),
        runtime_ev.request.payload.text_token_out};
    request.error_out = &runtime_ev.ctx.child_err;
    request.graph_error_out = &runtime_ev.ctx.graph_err;
    request.produced_out = &runtime_ev.ctx.produced;
    runtime_ev.ctx.child_accepted = ctx.generator.process_event(request);
  }
};

template <class runtime_event_type> struct effect_decode_frame {
  void operator()(const runtime_event_type &runtime_ev,
                  context &ctx) const noexcept {
    runtime_ev.ctx.child_err = 0;
    mimi::event::decode_frame request{
        runtime_ev.request.payload.output_codes_out.first(
            static_cast<size_t>(ctx.public_n_q)),
        runtime_ev.request.payload.pcm_out};
    request.error_out = &runtime_ev.ctx.child_err;
    runtime_ev.ctx.child_accepted = ctx.decoder.process_event(request);
  }
};

template <class runtime_event_type> struct effect_publish_frame_done {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
    runtime_ev.request.payload.produced_out = runtime_ev.ctx.produced;
  }
};

template <class runtime_event_type, error::code error_code>
struct effect_fail_frame {
  void operator()(const runtime_event_type &runtime_ev,
                  context &) const noexcept {
    detail_ns::effect_store_error(runtime_ev, error_code);
    runtime_ev.request.payload.produced_out = false;
  }
};

struct effect_publish_begin_flush_done {
  void operator()(const event::begin_flush_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_fail_begin_flush_invalid {
  void operator()(const event::begin_flush_run &runtime_ev,
                  context &) const noexcept {
    detail_ns::effect_store_error(runtime_ev, error::invalid_request);
  }
};

struct effect_publish_finish_done {
  void operator()(const event::finish_run &runtime_ev,
                  context &) const noexcept {
    runtime_ev.ctx.err = detail_ns::to_error(error::none);
    runtime_ev.request.error_out = runtime_ev.ctx.err;
  }
};

struct effect_fail_finish_invalid {
  void operator()(const event::finish_run &runtime_ev,
                  context &) const noexcept {
    detail_ns::effect_store_error(runtime_ev, error::invalid_request);
  }
};

struct effect_mark_unexpected {
  template <class unexpected_event_type>
  void operator()(const unexpected_event_type &runtime_ev,
                  context &) const noexcept {
    if constexpr (requires {
                    runtime_ev.ctx.err;
                    runtime_ev.request.error_out;
                  }) {
      runtime_ev.ctx.err = detail_ns::to_error(error::unexpected_event);
      runtime_ev.request.error_out = runtime_ev.ctx.err;
    }
  }
};

} // namespace emel::speech::generator::moshi::personaplex::session::action
