#pragma once
// benchmark: designed

#include <boost/sml.hpp>

#include "emel/embeddings/generator/actions.hpp"
#include "emel/embeddings/generator/context.hpp"
#include "emel/embeddings/generator/detail.hpp"
#include "emel/embeddings/generator/events.hpp"
#include "emel/embeddings/generator/guards.hpp"
#include "emel/sm.hpp"

namespace emel::embeddings::generator {

struct state_uninitialized {};
struct state_initializing {};
struct state_initialize_decision {};
struct state_initialize_publish_success {};
struct state_initialize_publish_error {};
struct state_initialize_error_channel_decision {};
struct state_idle {};
struct state_conditioning {};
struct state_conditioning_decision {};
struct state_encoding {};
struct state_image_preparing {};
struct state_image_encoding {};
struct state_audio_preparing {};
struct state_audio_encoding {};
struct state_embedding_decision {};
struct state_embed_publish_success {};
struct state_embed_publish_error {};
struct state_embed_error_channel_decision {};
struct state_done {};
struct state_errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<state_initializing> <= *sml::state<state_uninitialized>
          + sml::event<event::initialize_run> [ guard::guard_valid_initialize{} ]
          / action::effect_begin_initialize
      , sml::state<state_initialize_publish_error> <= sml::state<state_uninitialized>
          + sml::event<event::initialize_run> [ guard::guard_invalid_initialize{} ]
          / action::effect_reject_initialize

      , sml::state<state_initializing> <= sml::state<state_idle>
          + sml::event<event::initialize_run> [ guard::guard_valid_initialize{} ]
          / action::effect_begin_initialize
      , sml::state<state_initialize_publish_error> <= sml::state<state_idle>
          + sml::event<event::initialize_run> [ guard::guard_invalid_initialize{} ]
          / action::effect_reject_initialize
      , sml::state<state_conditioning> <= sml::state<state_idle>
          + sml::event<event::embed_text_run> [ guard::guard_valid_embed_full{} ]
          / action::effect_begin_embed_text
      , sml::state<state_conditioning> <= sml::state<state_idle>
          + sml::event<event::embed_text_run> [ guard::guard_valid_embed_truncate{} ]
          / action::effect_begin_embed_text
      , sml::state<state_embed_publish_error> <= sml::state<state_idle>
          + sml::event<event::embed_text_run> [ guard::guard_invalid_embed{} ]
          / action::effect_reject_embed_text
      , sml::state<state_image_preparing> <= sml::state<state_idle>
          + sml::event<event::embed_image_run> [ guard::guard_valid_embed_image_full{} ]
          / action::effect_begin_embed_image
      , sml::state<state_image_preparing> <= sml::state<state_idle>
          + sml::event<event::embed_image_run> [ guard::guard_valid_embed_image_truncate{} ]
          / action::effect_begin_embed_image
      , sml::state<state_embed_publish_error> <= sml::state<state_idle>
          + sml::event<event::embed_image_run> [ guard::guard_invalid_embed_image{} ]
          / action::effect_reject_embed_image
      , sml::state<state_audio_preparing> <= sml::state<state_idle>
          + sml::event<event::embed_audio_run> [ guard::guard_valid_embed_audio_full{} ]
          / action::effect_begin_embed_audio
      , sml::state<state_audio_preparing> <= sml::state<state_idle>
          + sml::event<event::embed_audio_run> [ guard::guard_valid_embed_audio_truncate{} ]
          / action::effect_begin_embed_audio
      , sml::state<state_embed_publish_error> <= sml::state<state_idle>
          + sml::event<event::embed_audio_run> [ guard::guard_invalid_embed_audio{} ]
          / action::effect_reject_embed_audio

      , sml::state<state_initializing> <= sml::state<state_done>
          + sml::event<event::initialize_run> [ guard::guard_valid_initialize{} ]
          / action::effect_begin_initialize
      , sml::state<state_initialize_publish_error> <= sml::state<state_done>
          + sml::event<event::initialize_run> [ guard::guard_invalid_initialize{} ]
          / action::effect_reject_initialize
      , sml::state<state_conditioning> <= sml::state<state_done>
          + sml::event<event::embed_text_run> [ guard::guard_valid_embed_full{} ]
          / action::effect_begin_embed_text
      , sml::state<state_conditioning> <= sml::state<state_done>
          + sml::event<event::embed_text_run> [ guard::guard_valid_embed_truncate{} ]
          / action::effect_begin_embed_text
      , sml::state<state_embed_publish_error> <= sml::state<state_done>
          + sml::event<event::embed_text_run> [ guard::guard_invalid_embed{} ]
          / action::effect_reject_embed_text
      , sml::state<state_image_preparing> <= sml::state<state_done>
          + sml::event<event::embed_image_run> [ guard::guard_valid_embed_image_full{} ]
          / action::effect_begin_embed_image
      , sml::state<state_image_preparing> <= sml::state<state_done>
          + sml::event<event::embed_image_run> [ guard::guard_valid_embed_image_truncate{} ]
          / action::effect_begin_embed_image
      , sml::state<state_embed_publish_error> <= sml::state<state_done>
          + sml::event<event::embed_image_run> [ guard::guard_invalid_embed_image{} ]
          / action::effect_reject_embed_image
      , sml::state<state_audio_preparing> <= sml::state<state_done>
          + sml::event<event::embed_audio_run> [ guard::guard_valid_embed_audio_full{} ]
          / action::effect_begin_embed_audio
      , sml::state<state_audio_preparing> <= sml::state<state_done>
          + sml::event<event::embed_audio_run> [ guard::guard_valid_embed_audio_truncate{} ]
          / action::effect_begin_embed_audio
      , sml::state<state_embed_publish_error> <= sml::state<state_done>
          + sml::event<event::embed_audio_run> [ guard::guard_invalid_embed_audio{} ]
          / action::effect_reject_embed_audio

      , sml::state<state_initializing> <= sml::state<state_errored>
          + sml::event<event::initialize_run> [ guard::guard_valid_initialize{} ]
          / action::effect_begin_initialize
      , sml::state<state_initialize_publish_error> <= sml::state<state_errored>
          + sml::event<event::initialize_run> [ guard::guard_invalid_initialize{} ]
          / action::effect_reject_initialize
      , sml::state<state_conditioning> <= sml::state<state_errored>
          + sml::event<event::embed_text_run> [ guard::guard_valid_embed_full{} ]
          / action::effect_begin_embed_text
      , sml::state<state_conditioning> <= sml::state<state_errored>
          + sml::event<event::embed_text_run> [ guard::guard_valid_embed_truncate{} ]
          / action::effect_begin_embed_text
      , sml::state<state_embed_publish_error> <= sml::state<state_errored>
          + sml::event<event::embed_text_run> [ guard::guard_invalid_embed{} ]
          / action::effect_reject_embed_text
      , sml::state<state_image_preparing> <= sml::state<state_errored>
          + sml::event<event::embed_image_run> [ guard::guard_valid_embed_image_full{} ]
          / action::effect_begin_embed_image
      , sml::state<state_image_preparing> <= sml::state<state_errored>
          + sml::event<event::embed_image_run> [ guard::guard_valid_embed_image_truncate{} ]
          / action::effect_begin_embed_image
      , sml::state<state_embed_publish_error> <= sml::state<state_errored>
          + sml::event<event::embed_image_run> [ guard::guard_invalid_embed_image{} ]
          / action::effect_reject_embed_image
      , sml::state<state_audio_preparing> <= sml::state<state_errored>
          + sml::event<event::embed_audio_run> [ guard::guard_valid_embed_audio_full{} ]
          / action::effect_begin_embed_audio
      , sml::state<state_audio_preparing> <= sml::state<state_errored>
          + sml::event<event::embed_audio_run> [ guard::guard_valid_embed_audio_truncate{} ]
          / action::effect_begin_embed_audio
      , sml::state<state_embed_publish_error> <= sml::state<state_errored>
          + sml::event<event::embed_audio_run> [ guard::guard_invalid_embed_audio{} ]
          / action::effect_reject_embed_audio

      //------------------------------------------------------------------------------//
      , sml::state<state_initialize_decision> <= sml::state<state_initializing>
          + sml::completion<event::initialize_run>
          / action::effect_dispatch_bind_conditioner
      , sml::state<state_initialize_publish_success> <= sml::state<state_initialize_decision>
          + sml::completion<event::initialize_run> [ guard::guard_initialize_success{} ]
          / action::effect_mark_initialized
      , sml::state<state_initialize_publish_error> <= sml::state<state_initialize_decision>
          + sml::completion<event::initialize_run> [ guard::guard_initialize_model_invalid{} ]
          / action::effect_set_initialize_model_invalid
      , sml::state<state_initialize_publish_error> <= sml::state<state_initialize_decision>
          + sml::completion<event::initialize_run> [ guard::guard_initialize_backend_error{} ]
          / action::effect_set_initialize_backend_error

      , sml::state<state_idle> <= sml::state<state_initialize_publish_success>
          + sml::completion<event::initialize_run> [ guard::guard_has_initialize_done_callback{} ]
          / action::effect_emit_initialize_done
      , sml::state<state_idle> <= sml::state<state_initialize_publish_success>
          + sml::completion<event::initialize_run> [ guard::guard_no_initialize_done_callback{} ]

      , sml::state<state_initialize_error_channel_decision> <= sml::state<state_initialize_publish_error>
          + sml::completion<event::initialize_run>
          / action::effect_write_initialize_error_out
      , sml::state<state_errored> <= sml::state<state_initialize_error_channel_decision>
          + sml::completion<event::initialize_run> [ guard::guard_has_initialize_error_callback{} ]
          / action::effect_emit_initialize_error
      , sml::state<state_errored> <= sml::state<state_initialize_error_channel_decision>
          + sml::completion<event::initialize_run> [ guard::guard_no_initialize_error_callback{} ]

      //------------------------------------------------------------------------------//
      , sml::state<state_conditioning_decision> <= sml::state<state_conditioning>
          + sml::completion<event::embed_text_run>
          / action::effect_dispatch_condition_text
      , sml::state<state_embed_publish_error> <= sml::state<state_conditioning_decision>
          + sml::completion<event::embed_text_run> [ guard::guard_prepare_invalid_request{} ]
          / action::effect_set_embed_invalid_request
      , sml::state<state_embed_publish_error> <= sml::state<state_conditioning_decision>
          + sml::completion<event::embed_text_run> [ guard::guard_prepare_model_invalid{} ]
          / action::effect_set_embed_model_invalid
      , sml::state<state_embed_publish_error> <= sml::state<state_conditioning_decision>
          + sml::completion<event::embed_text_run> [ guard::guard_prepare_backend_error{} ]
          / action::effect_set_embed_backend_error
      , sml::state<state_encoding> <= sml::state<state_conditioning_decision>
          + sml::completion<event::embed_text_run> [ guard::guard_prepare_success{} ]

      , sml::state<state_embed_publish_error> <= sml::state<state_encoding>
          + sml::completion<event::embed_text_run> [ guard::guard_text_route_unsupported{} ]
          / action::effect_set_embed_backend_error
      , sml::state<state_embed_publish_error> <= sml::state<state_encoding>
          + sml::completion<event::embed_text_run> [ guard::guard_text_encode_bert_unready{} ]
          / action::effect_set_embed_backend_error
      , sml::state<state_embedding_decision> <= sml::state<state_encoding>
          + sml::completion<event::embed_text_run> [ guard::guard_text_encode_bert_ready{} ]
          / action::effect_run_text_embedding_bert
      , sml::state<state_embed_publish_success> <= sml::state<state_embedding_decision>
          + sml::completion<event::embed_text_run> [ guard::guard_embedding_succeeded_full{} ]
          / action::effect_publish_full_embedding
      , sml::state<state_embed_publish_success> <= sml::state<state_embedding_decision>
          + sml::completion<event::embed_text_run> [ guard::guard_embedding_succeeded_truncate{} ]
          / action::effect_publish_truncated_embedding

      , sml::state<state_done> <= sml::state<state_embed_publish_success>
          + sml::completion<event::embed_text_run> [ guard::guard_has_embed_done_callback{} ]
          / action::effect_emit_embed_done
      , sml::state<state_done> <= sml::state<state_embed_publish_success>
          + sml::completion<event::embed_text_run> [ guard::guard_no_embed_done_callback{} ]

      , sml::state<state_embed_error_channel_decision> <= sml::state<state_embed_publish_error>
          + sml::completion<event::embed_text_run>
          / action::effect_write_embed_error_out
      , sml::state<state_errored> <= sml::state<state_embed_error_channel_decision>
          + sml::completion<event::embed_text_run> [ guard::guard_has_embed_error_callback{} ]
          / action::effect_emit_embed_error
      , sml::state<state_errored> <= sml::state<state_embed_error_channel_decision>
          + sml::completion<event::embed_text_run> [ guard::guard_no_embed_error_callback{} ]

      //------------------------------------------------------------------------------//
      , sml::state<state_embed_publish_error> <= sml::state<state_image_preparing>
          + sml::completion<event::embed_image_run> [ guard::guard_image_route_unsupported{} ]
          / action::effect_set_embed_backend_error
      , sml::state<state_embed_publish_error> <= sml::state<state_image_preparing>
          + sml::completion<event::embed_image_run> [ guard::guard_image_prepare_mobilenetv4_unready{} ]
          / action::effect_set_embed_backend_error
      , sml::state<state_image_encoding> <= sml::state<state_image_preparing>
          + sml::completion<event::embed_image_run> [ guard::guard_image_prepare_mobilenetv4_ready{} ]
          / action::effect_prepare_image_input_mobilenetv4

      , sml::state<state_embed_publish_error> <= sml::state<state_image_encoding>
          + sml::completion<event::embed_image_run> [ guard::guard_image_route_unsupported{} ]
          / action::effect_set_embed_backend_error
      , sml::state<state_embed_publish_error> <= sml::state<state_image_encoding>
          + sml::completion<event::embed_image_run> [ guard::guard_image_encode_mobilenetv4_unready{} ]
          / action::effect_set_embed_backend_error
      , sml::state<state_embedding_decision> <= sml::state<state_image_encoding>
          + sml::completion<event::embed_image_run> [ guard::guard_image_encode_mobilenetv4_ready{} ]
          / action::effect_run_image_embedding_mobilenetv4
      , sml::state<state_embed_publish_success> <= sml::state<state_embedding_decision>
          + sml::completion<event::embed_image_run> [ guard::guard_embedding_succeeded_full{} ]
          / action::effect_publish_full_embedding
      , sml::state<state_embed_publish_success> <= sml::state<state_embedding_decision>
          + sml::completion<event::embed_image_run> [ guard::guard_embedding_succeeded_truncate{} ]
          / action::effect_publish_truncated_embedding

      , sml::state<state_done> <= sml::state<state_embed_publish_success>
          + sml::completion<event::embed_image_run> [ guard::guard_has_embed_done_callback{} ]
          / action::effect_emit_embed_done
      , sml::state<state_done> <= sml::state<state_embed_publish_success>
          + sml::completion<event::embed_image_run> [ guard::guard_no_embed_done_callback{} ]

      , sml::state<state_embed_error_channel_decision> <= sml::state<state_embed_publish_error>
          + sml::completion<event::embed_image_run>
          / action::effect_write_embed_error_out
      , sml::state<state_errored> <= sml::state<state_embed_error_channel_decision>
          + sml::completion<event::embed_image_run> [ guard::guard_has_embed_error_callback{} ]
          / action::effect_emit_embed_error
      , sml::state<state_errored> <= sml::state<state_embed_error_channel_decision>
          + sml::completion<event::embed_image_run> [ guard::guard_no_embed_error_callback{} ]

      //------------------------------------------------------------------------------//
      , sml::state<state_embed_publish_error> <= sml::state<state_audio_preparing>
          + sml::completion<event::embed_audio_run> [ guard::guard_audio_route_unsupported{} ]
          / action::effect_set_embed_backend_error
      , sml::state<state_embed_publish_error> <= sml::state<state_audio_preparing>
          + sml::completion<event::embed_audio_run> [ guard::guard_audio_prepare_efficientat_unready{} ]
          / action::effect_set_embed_backend_error
      , sml::state<state_audio_encoding> <= sml::state<state_audio_preparing>
          + sml::completion<event::embed_audio_run> [ guard::guard_audio_prepare_efficientat_ready{} ]
          / action::effect_prepare_audio_input_efficientat

      , sml::state<state_embed_publish_error> <= sml::state<state_audio_encoding>
          + sml::completion<event::embed_audio_run> [ guard::guard_audio_route_unsupported{} ]
          / action::effect_set_embed_backend_error
      , sml::state<state_embed_publish_error> <= sml::state<state_audio_encoding>
          + sml::completion<event::embed_audio_run> [ guard::guard_audio_encode_efficientat_unready{} ]
          / action::effect_set_embed_backend_error
      , sml::state<state_embedding_decision> <= sml::state<state_audio_encoding>
          + sml::completion<event::embed_audio_run> [ guard::guard_audio_encode_efficientat_ready{} ]
          / action::effect_run_audio_embedding_efficientat
      , sml::state<state_embed_publish_success> <= sml::state<state_embedding_decision>
          + sml::completion<event::embed_audio_run> [ guard::guard_embedding_succeeded_full{} ]
          / action::effect_publish_full_embedding
      , sml::state<state_embed_publish_success> <= sml::state<state_embedding_decision>
          + sml::completion<event::embed_audio_run> [ guard::guard_embedding_succeeded_truncate{} ]
          / action::effect_publish_truncated_embedding

      , sml::state<state_done> <= sml::state<state_embed_publish_success>
          + sml::completion<event::embed_audio_run> [ guard::guard_has_embed_done_callback{} ]
          / action::effect_emit_embed_done
      , sml::state<state_done> <= sml::state<state_embed_publish_success>
          + sml::completion<event::embed_audio_run> [ guard::guard_no_embed_done_callback{} ]

      , sml::state<state_embed_error_channel_decision> <= sml::state<state_embed_publish_error>
          + sml::completion<event::embed_audio_run>
          / action::effect_write_embed_error_out
      , sml::state<state_errored> <= sml::state<state_embed_error_channel_decision>
          + sml::completion<event::embed_audio_run> [ guard::guard_has_embed_error_callback{} ]
          / action::effect_emit_embed_error
      , sml::state<state_errored> <= sml::state<state_embed_error_channel_decision>
          + sml::completion<event::embed_audio_run> [ guard::guard_no_embed_error_callback{} ]

      //------------------------------------------------------------------------------//
      , sml::state<state_uninitialized> <= sml::state<state_uninitialized> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_uninitialized> <= sml::state<state_initializing> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_uninitialized> <= sml::state<state_initialize_decision> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_idle> <= sml::state<state_initialize_publish_success> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_errored> <= sml::state<state_initialize_publish_error> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_errored> <= sml::state<state_initialize_error_channel_decision> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected

      , sml::state<state_idle> <= sml::state<state_idle> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_idle> <= sml::state<state_conditioning> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_idle> <= sml::state<state_conditioning_decision> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_idle> <= sml::state<state_encoding> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_idle> <= sml::state<state_image_preparing> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_idle> <= sml::state<state_image_encoding> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_idle> <= sml::state<state_audio_preparing> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_idle> <= sml::state<state_audio_encoding> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_idle> <= sml::state<state_embedding_decision> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_done> <= sml::state<state_embed_publish_success> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_errored> <= sml::state<state_embed_publish_error> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_errored> <= sml::state<state_embed_error_channel_decision> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_done> <= sml::state<state_done> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
      , sml::state<state_errored> <= sml::state<state_errored> + sml::unexpected_event<sml::_>
          / action::effect_reject_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  sm() : base_type() {}

  sm(const emel::model::data & model_ref,
     emel::text::conditioner::sm & conditioner_ref,
     void * formatter_ctx = nullptr,
     emel::text::formatter::format_fn format_prompt =
         emel::text::formatter::format_raw)
      : base_type() {
    this->context_.model = &model_ref;
    this->context_.conditioner = &conditioner_ref;
    this->context_.formatter_ctx = formatter_ctx;
    this->context_.format_prompt = format_prompt;
    (void) detail::reserve_scratch(this->context_, model_ref);
  }

  sm(const sm &) = delete;
  sm(sm &&) = delete;
  sm & operator=(const sm &) = delete;
  sm & operator=(sm &&) = delete;

  bool process_event(const event::initialize & ev) {
    event::initialize_ctx ctx{};
    event::initialize_run runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == detail::to_error(error::none);
  }

  bool process_event(const event::embed_text & ev) {
    event::embed_text_ctx ctx{};
    event::embed_text_run runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == detail::to_error(error::none);
  }

  bool process_event(const event::embed_image & ev) {
    event::embed_image_ctx ctx{};
    event::embed_image_run runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == detail::to_error(error::none);
  }

  bool process_event(const event::embed_audio & ev) {
    event::embed_audio_ctx ctx{};
    event::embed_audio_run runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == detail::to_error(error::none);
  }
};

}  // namespace emel::embeddings::generator
