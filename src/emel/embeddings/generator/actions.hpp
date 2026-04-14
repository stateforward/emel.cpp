#pragma once

#include "emel/embeddings/generator/context.hpp"
#include "emel/embeddings/generator/detail.hpp"
#include "emel/embeddings/generator/events.hpp"
#include "emel/text/conditioner/errors.hpp"

namespace emel::embeddings::generator::action {

struct effect_begin_initialize {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context & ctx) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.err = detail::to_error(error::none);
    ev.ctx.bind_accepted = false;
    ev.ctx.bind_err_code =
        detail::conditioner_error_code(emel::text::conditioner::error::none);
    ctx.initialized = false;
  }
};

struct effect_reject_initialize {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    detail::set_error(ev, error::invalid_request);
  }
};

struct effect_dispatch_bind_conditioner {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context & ctx) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    int32_t err =
        detail::conditioner_error_code(emel::text::conditioner::error::none);
    emel::text::conditioner::event::bind bind_ev{ctx.model->vocab_data};
    bind_ev.preprocessor_variant = ev.request.preprocessor_variant;
    bind_ev.encoder_variant = ev.request.encoder_variant;
    bind_ev.tokenizer_sm = ev.request.tokenizer_sm;
    bind_ev.dispatch_tokenizer_bind = ev.request.dispatch_tokenizer_bind;
    bind_ev.dispatch_tokenizer_tokenize = ev.request.dispatch_tokenizer_tokenize;
    bind_ev.formatter_ctx = ctx.formatter_ctx;
    bind_ev.format_prompt = ctx.format_prompt;
    bind_ev.add_special = ev.request.add_special;
    bind_ev.parse_special = ev.request.parse_special;
    bind_ev.error_out = &err;
    ev.ctx.bind_accepted = ctx.conditioner->process_event(bind_ev);
    ev.ctx.bind_err_code = err;
  }
};

struct effect_mark_initialized {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context & ctx) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.err = detail::to_error(error::none);
    ctx.initialized = true;
    detail::write_initialize_error_out(ev);
  }
};

struct effect_set_initialize_model_invalid {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    detail::set_error(ev, error::model_invalid);
  }
};

struct effect_set_initialize_backend_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    detail::set_error(ev, error::backend);
  }
};

struct effect_write_initialize_error_out {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    detail::write_initialize_error_out(ev);
  }
};

struct effect_emit_initialize_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    detail::emit_initialize_done(ev);
  }
};

struct effect_emit_initialize_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    detail::emit_initialize_error(ev);
  }
};

struct effect_begin_embed_text {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.err = detail::to_error(error::none);
    ev.ctx.prepare_accepted = false;
    ev.ctx.prepare_err_code =
        detail::conditioner_error_code(emel::text::conditioner::error::none);
    ev.ctx.token_count = 0;
    ev.ctx.output_dimension = 0;
    ev.request.output_dimension_out = 0;
  }
};

struct effect_begin_embed_image {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.err = detail::to_error(error::none);
    ev.ctx.prepared = false;
    ev.ctx.output_dimension = 0;
    ev.request.output_dimension_out = 0;
  }
};

struct effect_begin_embed_audio {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.err = detail::to_error(error::none);
    ev.ctx.prepared = false;
    ev.ctx.output_dimension = 0;
    ev.request.output_dimension_out = 0;
  }
};

struct effect_reject_embed_text {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.output_dimension = 0;
    detail::set_error(ev, error::invalid_request);
  }
};

struct effect_reject_embed_image {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.output_dimension = 0;
    detail::set_error(ev, error::invalid_request);
  }
};

struct effect_reject_embed_audio {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.output_dimension = 0;
    detail::set_error(ev, error::invalid_request);
  }
};

struct effect_dispatch_condition_text {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context & ctx) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    int32_t token_count = 0;
    int32_t err =
        detail::conditioner_error_code(emel::text::conditioner::error::none);
    emel::text::conditioner::event::prepare prepare_ev{
        token_count,
        err,
    };
    prepare_ev.messages = ev.request.messages;
    prepare_ev.add_generation_prompt = ev.request.add_generation_prompt;
    prepare_ev.enable_thinking = ev.request.enable_thinking;
    prepare_ev.use_bind_defaults = true;
    prepare_ev.token_ids_out = ctx.scratch.token_ids.get();
    prepare_ev.token_capacity = ctx.text.max_positions;
    ev.ctx.prepare_accepted = ctx.conditioner->process_event(prepare_ev);
    ev.ctx.prepare_err_code = err;
    ev.ctx.token_count = token_count;
  }
};

struct effect_prepare_image_input {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context & ctx) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.prepared =
        detail::prepare_image_input(ctx, ev.request.rgba, ev.request.width, ev.request.height);
    if (!ev.ctx.prepared) {
      detail::set_error(ev, error::backend);
    }
  }
};

struct effect_prepare_audio_input {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context & ctx) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.prepared = detail::prepare_audio_input(ctx, ev.request.pcm, ev.request.sample_rate);
    if (!ev.ctx.prepared) {
      detail::set_error(ev, error::backend);
    }
  }
};

struct effect_set_embed_invalid_request {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.output_dimension = 0;
    detail::set_error(ev, error::invalid_request);
  }
};

struct effect_set_embed_model_invalid {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.output_dimension = 0;
    detail::set_error(ev, error::model_invalid);
  }
};

struct effect_set_embed_backend_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    ev.ctx.output_dimension = 0;
    detail::set_error(ev, error::backend);
  }
};

struct effect_run_text_embedding {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context & ctx) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    if (!detail::run_text_embedding(ctx, ev.ctx.token_count)) {
      detail::set_error(ev, error::backend);
    }
  }
};

struct effect_run_image_embedding {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context & ctx) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    if (!detail::run_image_embedding(ctx)) {
      detail::set_error(ev, error::backend);
    }
  }
};

struct effect_run_audio_embedding {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context & ctx) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    if (!detail::run_audio_embedding(ctx)) {
      detail::set_error(ev, error::backend);
    }
  }
};

struct effect_publish_full_embedding {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context & ctx) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    const int32_t dimension = detail::shared_embedding_size(ctx);
    if (!detail::publish_embedding(ctx, dimension, ev.request.output)) {
      ev.ctx.output_dimension = 0;
      detail::set_error(ev, error::backend);
      return;
    }
    ev.ctx.output_dimension = dimension;
    ev.request.output_dimension_out = dimension;
    detail::write_embed_error_out(ev);
  }
};

struct effect_publish_truncated_embedding {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context & ctx) const noexcept {
    auto & ev = detail::unwrap_runtime_event(runtime_ev);
    const int32_t dimension =
        detail::requested_output_dimension(ev.request, ctx);
    if (!detail::publish_embedding(ctx, dimension, ev.request.output)) {
      ev.ctx.output_dimension = 0;
      detail::set_error(ev, error::backend);
      return;
    }
    ev.ctx.output_dimension = dimension;
    ev.request.output_dimension_out = dimension;
    detail::write_embed_error_out(ev);
  }
};

struct effect_write_embed_error_out {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    detail::write_embed_error_out(ev);
  }
};

struct effect_emit_embed_done {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    detail::emit_embed_done(ev);
  }
};

struct effect_emit_embed_error {
  template <class runtime_event_type>
  void operator()(const runtime_event_type & runtime_ev,
                  context &) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    detail::emit_embed_error(ev);
  }
};

struct effect_ignore_unexpected {
  void operator()() const noexcept {}
};

inline constexpr effect_begin_initialize effect_begin_initialize{};
inline constexpr effect_reject_initialize effect_reject_initialize{};
inline constexpr effect_dispatch_bind_conditioner effect_dispatch_bind_conditioner{};
inline constexpr effect_mark_initialized effect_mark_initialized{};
inline constexpr effect_set_initialize_model_invalid effect_set_initialize_model_invalid{};
inline constexpr effect_set_initialize_backend_error effect_set_initialize_backend_error{};
inline constexpr effect_write_initialize_error_out effect_write_initialize_error_out{};
inline constexpr effect_emit_initialize_done effect_emit_initialize_done{};
inline constexpr effect_emit_initialize_error effect_emit_initialize_error{};
inline constexpr effect_begin_embed_text effect_begin_embed_text{};
inline constexpr effect_begin_embed_image effect_begin_embed_image{};
inline constexpr effect_begin_embed_audio effect_begin_embed_audio{};
inline constexpr effect_reject_embed_text effect_reject_embed_text{};
inline constexpr effect_reject_embed_image effect_reject_embed_image{};
inline constexpr effect_reject_embed_audio effect_reject_embed_audio{};
inline constexpr effect_dispatch_condition_text effect_dispatch_condition_text{};
inline constexpr effect_prepare_image_input effect_prepare_image_input{};
inline constexpr effect_prepare_audio_input effect_prepare_audio_input{};
inline constexpr effect_set_embed_invalid_request effect_set_embed_invalid_request{};
inline constexpr effect_set_embed_model_invalid effect_set_embed_model_invalid{};
inline constexpr effect_set_embed_backend_error effect_set_embed_backend_error{};
inline constexpr effect_run_text_embedding effect_run_text_embedding{};
inline constexpr effect_run_image_embedding effect_run_image_embedding{};
inline constexpr effect_run_audio_embedding effect_run_audio_embedding{};
inline constexpr effect_publish_full_embedding effect_publish_full_embedding{};
inline constexpr effect_publish_truncated_embedding effect_publish_truncated_embedding{};
inline constexpr effect_write_embed_error_out effect_write_embed_error_out{};
inline constexpr effect_emit_embed_done effect_emit_embed_done{};
inline constexpr effect_emit_embed_error effect_emit_embed_error{};
inline constexpr effect_ignore_unexpected effect_ignore_unexpected{};

}  // namespace emel::embeddings::generator::action
