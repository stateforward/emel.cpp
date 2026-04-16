#pragma once

#include "emel/embeddings/generator/context.hpp"
#include "emel/embeddings/generator/detail.hpp"
#include "emel/embeddings/generator/events.hpp"

namespace emel::embeddings::generator::guard {

struct guard_valid_initialize {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return ctx.model != nullptr &&
        ctx.conditioner != nullptr &&
        ev.request.tokenizer_sm != nullptr &&
        ctx.format_prompt != nullptr &&
        detail::is_valid_preprocessor(ev.request.preprocessor_variant) &&
        detail::is_valid_encoder(ev.request.encoder_variant);
  }
};

struct guard_invalid_initialize {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_valid_initialize{}(runtime_ev, ctx);
  }
};

struct guard_initialize_success {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return ctx.text.ready &&
        ctx.scratch.ready &&
        ev.ctx.bind_accepted &&
        ev.ctx.bind_err_code ==
            detail::conditioner_error_code(emel::text::conditioner::error::none) &&
        ev.ctx.err == detail::to_error(error::none);
  }
};

struct guard_initialize_model_invalid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return !ctx.text.ready ||
        ev.ctx.bind_err_code ==
        detail::conditioner_error_code(emel::text::conditioner::error::model_invalid);
  }
};

struct guard_initialize_backend_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return !guard_initialize_success{}(runtime_ev, ctx) &&
        !guard_initialize_model_invalid{}(runtime_ev, ctx) &&
        ev.ctx.err == detail::to_error(error::none);
  }
};

struct guard_has_initialize_done_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev) const noexcept {
    return detail::has_initialize_callback(detail::unwrap_runtime_event(runtime_ev));
  }
};

struct guard_no_initialize_done_callback {
  bool operator()(const event::initialize_run & runtime_ev) const noexcept {
    return !guard_has_initialize_done_callback{}(runtime_ev);
  }
};

struct guard_has_initialize_error_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev) const noexcept {
    return detail::has_initialize_error_callback(detail::unwrap_runtime_event(runtime_ev));
  }
};

struct guard_no_initialize_error_callback {
  bool operator()(const event::initialize_run & runtime_ev) const noexcept {
    return !guard_has_initialize_error_callback{}(runtime_ev);
  }
};

struct guard_valid_embed_full {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return ctx.initialized &&
        !ev.request.messages.empty() &&
        ev.request.output.data() != nullptr &&
        static_cast<int32_t>(ev.request.output.size()) >= detail::shared_embedding_size(ctx) &&
        (ev.request.truncate_dimension == 0 ||
         ev.request.truncate_dimension == detail::shared_embedding_size(ctx));
  }
};

struct guard_valid_embed_truncate {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    const int32_t requested = detail::requested_output_dimension(ev.request, ctx);
    return ctx.initialized &&
        !ev.request.messages.empty() &&
        ev.request.output.data() != nullptr &&
        requested > 0 &&
        requested != detail::shared_embedding_size(ctx) &&
        detail::is_supported_truncate_dimension(ctx, requested) &&
        static_cast<int32_t>(ev.request.output.size()) >= requested;
  }
};

struct guard_valid_embed_image_full {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return ctx.initialized &&
        ctx.image.ready &&
        detail::is_valid_image_payload(ev.request.rgba, ev.request.width, ev.request.height) &&
        ev.request.output.data() != nullptr &&
        static_cast<int32_t>(ev.request.output.size()) >= detail::shared_embedding_size(ctx) &&
        (ev.request.truncate_dimension == 0 ||
         ev.request.truncate_dimension == detail::shared_embedding_size(ctx));
  }
};

struct guard_valid_embed_image_truncate {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    const int32_t requested = detail::requested_output_dimension(ev.request, ctx);
    return ctx.initialized &&
        ctx.image.ready &&
        detail::is_valid_image_payload(ev.request.rgba, ev.request.width, ev.request.height) &&
        ev.request.output.data() != nullptr &&
        requested > 0 &&
        requested != detail::shared_embedding_size(ctx) &&
        detail::is_supported_truncate_dimension(ctx, requested) &&
        static_cast<int32_t>(ev.request.output.size()) >= requested;
  }
};

struct guard_valid_embed_audio_full {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return ctx.initialized &&
        ctx.audio.ready &&
        detail::is_valid_audio_payload(ev.request.pcm, ev.request.sample_rate) &&
        ev.request.output.data() != nullptr &&
        static_cast<int32_t>(ev.request.output.size()) >= detail::shared_embedding_size(ctx) &&
        (ev.request.truncate_dimension == 0 ||
         ev.request.truncate_dimension == detail::shared_embedding_size(ctx));
  }
};

struct guard_valid_embed_audio_truncate {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    const int32_t requested = detail::requested_output_dimension(ev.request, ctx);
    return ctx.initialized &&
        ctx.audio.ready &&
        detail::is_valid_audio_payload(ev.request.pcm, ev.request.sample_rate) &&
        ev.request.output.data() != nullptr &&
        requested > 0 &&
        requested != detail::shared_embedding_size(ctx) &&
        detail::is_supported_truncate_dimension(ctx, requested) &&
        static_cast<int32_t>(ev.request.output.size()) >= requested;
  }
};

struct guard_invalid_embed {
  bool operator()(const event::embed_text_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_valid_embed_full{}(runtime_ev, ctx) &&
        !guard_valid_embed_truncate{}(runtime_ev, ctx);
  }
};

struct guard_invalid_embed_image {
  bool operator()(const event::embed_image_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_valid_embed_image_full{}(runtime_ev, ctx) &&
        !guard_valid_embed_image_truncate{}(runtime_ev, ctx);
  }
};

struct guard_invalid_embed_audio {
  bool operator()(const event::embed_audio_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_valid_embed_audio_full{}(runtime_ev, ctx) &&
        !guard_valid_embed_audio_truncate{}(runtime_ev, ctx);
  }
};

struct guard_text_route_bert {
  bool operator()(const event::embed_text_run &,
                  const action::context & ctx) const noexcept {
    return ctx.text.route_kind == action::text_route_kind::omniembed_bert_text;
  }
};

struct guard_text_route_unsupported {
  bool operator()(const event::embed_text_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_text_route_bert{}(runtime_ev, ctx);
  }
};

struct guard_text_encode_bert_ready {
  bool operator()(const event::embed_text_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return guard_text_route_bert{}(runtime_ev, ctx) &&
        ctx.text.ready &&
        ctx.scratch.ready &&
        ev.ctx.token_count > 0 &&
        ev.ctx.token_count <= ctx.text.max_positions;
  }
};

struct guard_text_encode_bert_unready {
  bool operator()(const event::embed_text_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return guard_text_route_bert{}(runtime_ev, ctx) &&
        !guard_text_encode_bert_ready{}(runtime_ev, ctx);
  }
};

struct guard_image_route_mobilenetv4 {
  bool operator()(const event::embed_image_run &,
                  const action::context & ctx) const noexcept {
    return ctx.image.route_kind == action::image_route_kind::omniembed_mobilenetv4_medium;
  }
};

struct guard_image_route_unsupported {
  bool operator()(const event::embed_image_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_image_route_mobilenetv4{}(runtime_ev, ctx);
  }
};

struct guard_image_prepare_mobilenetv4_ready {
  bool operator()(const event::embed_image_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return guard_image_route_mobilenetv4{}(runtime_ev, ctx) &&
        ctx.model != nullptr &&
        ctx.image.ready &&
        ctx.scratch.ready;
  }
};

struct guard_image_prepare_mobilenetv4_unready {
  bool operator()(const event::embed_image_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return guard_image_route_mobilenetv4{}(runtime_ev, ctx) &&
        !guard_image_prepare_mobilenetv4_ready{}(runtime_ev, ctx);
  }
};

struct guard_image_encode_mobilenetv4_ready {
  bool operator()(const event::embed_image_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return guard_image_route_mobilenetv4{}(runtime_ev, ctx) &&
        ctx.image.ready &&
        ctx.scratch.ready;
  }
};

struct guard_image_encode_mobilenetv4_unready {
  bool operator()(const event::embed_image_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return guard_image_route_mobilenetv4{}(runtime_ev, ctx) &&
        !guard_image_encode_mobilenetv4_ready{}(runtime_ev, ctx);
  }
};

struct guard_audio_route_efficientat {
  bool operator()(const event::embed_audio_run &,
                  const action::context & ctx) const noexcept {
    return ctx.audio.route_kind == action::audio_route_kind::omniembed_efficientat_mn20_as;
  }
};

struct guard_audio_route_unsupported {
  bool operator()(const event::embed_audio_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return !guard_audio_route_efficientat{}(runtime_ev, ctx);
  }
};

struct guard_audio_prepare_efficientat_ready {
  bool operator()(const event::embed_audio_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return guard_audio_route_efficientat{}(runtime_ev, ctx) &&
        ctx.model != nullptr &&
        ctx.audio.ready &&
        ctx.scratch.ready;
  }
};

struct guard_audio_prepare_efficientat_unready {
  bool operator()(const event::embed_audio_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return guard_audio_route_efficientat{}(runtime_ev, ctx) &&
        !guard_audio_prepare_efficientat_ready{}(runtime_ev, ctx);
  }
};

struct guard_audio_encode_efficientat_ready {
  bool operator()(const event::embed_audio_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return guard_audio_route_efficientat{}(runtime_ev, ctx) &&
        ctx.audio.ready &&
        ctx.scratch.ready;
  }
};

struct guard_audio_encode_efficientat_unready {
  bool operator()(const event::embed_audio_run & runtime_ev,
                  const action::context & ctx) const noexcept {
    return guard_audio_route_efficientat{}(runtime_ev, ctx) &&
        !guard_audio_encode_efficientat_ready{}(runtime_ev, ctx);
  }
};

struct guard_prepare_success {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.prepare_accepted &&
        ev.ctx.prepare_err_code ==
            detail::conditioner_error_code(emel::text::conditioner::error::none) &&
        ev.ctx.token_count > 0 &&
        ev.ctx.err == detail::to_error(error::none);
  }
};

struct guard_prepare_invalid_request {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.prepare_err_code ==
        detail::conditioner_error_code(emel::text::conditioner::error::invalid_argument);
  }
};

struct guard_prepare_model_invalid {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return ev.ctx.prepare_err_code ==
        detail::conditioner_error_code(emel::text::conditioner::error::model_invalid);
  }
};

struct guard_prepare_backend_error {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return !guard_prepare_success{}(runtime_ev) &&
        !guard_prepare_invalid_request{}(runtime_ev) &&
        !guard_prepare_model_invalid{}(runtime_ev) &&
        ev.ctx.err == detail::to_error(error::none);
  }
};

struct guard_embedding_succeeded_full {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    return detail::requested_output_dimension(ev.request, ctx) == detail::shared_embedding_size(ctx);
  }
};

struct guard_embedding_succeeded_truncate {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev,
                  const action::context & ctx) const noexcept {
    const auto & ev = detail::unwrap_runtime_event(runtime_ev);
    const int32_t requested = detail::requested_output_dimension(ev.request, ctx);
    return requested > 0 &&
        requested < detail::shared_embedding_size(ctx);
  }
};

struct guard_has_embed_done_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev) const noexcept {
    return detail::has_embed_callbacks(detail::unwrap_runtime_event(runtime_ev));
  }
};

struct guard_no_embed_done_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev) const noexcept {
    return !guard_has_embed_done_callback{}(runtime_ev);
  }
};

struct guard_has_embed_error_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev) const noexcept {
    return detail::has_embed_error_callback(detail::unwrap_runtime_event(runtime_ev));
  }
};

struct guard_no_embed_error_callback {
  template <class runtime_event_type>
  bool operator()(const runtime_event_type & runtime_ev) const noexcept {
    return !guard_has_embed_error_callback{}(runtime_ev);
  }
};

}  // namespace emel::embeddings::generator::guard
