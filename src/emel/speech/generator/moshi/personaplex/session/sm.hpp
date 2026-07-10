#pragma once
// benchmark: designed

#include <stateforward/sml.hpp>

#include "emel/sm.hpp"
#include "emel/speech/generator/moshi/personaplex/session/actions.hpp"
#include "emel/speech/generator/moshi/personaplex/session/context.hpp"
#include "emel/speech/generator/moshi/personaplex/session/events.hpp"
#include "emel/speech/generator/moshi/personaplex/session/guards.hpp"

namespace emel::speech::generator::moshi::personaplex::session {

struct state_uninitialized {};
struct state_initialize_encoder_result {};
struct state_initialize_decoder_result {};
struct state_initialize_executor_result {};
struct state_initialize_generator_result {};
struct state_load_voice_result {};
struct state_voice_prefill {};
struct state_voice_prefill_result {};
struct state_prompt_begin_result {};
struct state_prompt_prefill {};
struct state_prompt_prefill_result {};
struct state_live {};
struct state_live_encode_result {};
struct state_live_generate_result {};
struct state_live_decode_result {};
struct state_flush {};
struct state_flush_encode_result {};
struct state_flush_generate_result {};
struct state_flush_decode_result {};
struct state_done {};
struct state_failed {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    using init_run = event::initialize_run;
    using voice_run = event::advance_voice_run;
    using prompt_run = event::advance_prompt_run;
    using live_run = event::live_frame_run;
    using begin_flush_run = event::begin_flush_run;
    using flush_run = event::flush_frame_run;
    using finish_run = event::finish_run;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Initialize every child actor in deterministic RTC order.
        sml::state<state_initialize_encoder_result> <= *sml::state<state_uninitialized>
          + sml::event<init_run> [ guard::guard_initialize_request_valid{} ]
          / action::effect_initialize_encoder{}
      , sml::state<state_failed> <= sml::state<state_uninitialized>
          + sml::event<init_run> [ guard::guard_initialize_request_invalid{} ]
          / action::effect_fail_initialize_invalid{}
      , sml::state<state_initialize_decoder_result> <= sml::state<state_initialize_encoder_result>
          + sml::completion<init_run> [ guard::guard_encoder_initialize_succeeded{} ]
          / action::effect_initialize_decoder{}
      , sml::state<state_failed> <= sml::state<state_initialize_encoder_result>
          + sml::completion<init_run> [ guard::guard_encoder_initialize_failed{} ]
          / action::effect_fail_initialize_child<error::codec_initialize_failed>{}
      , sml::state<state_initialize_executor_result> <= sml::state<state_initialize_decoder_result>
          + sml::completion<init_run> [ guard::guard_child_succeeded<init_run>{} ]
          / action::effect_initialize_executor{}
      , sml::state<state_failed> <= sml::state<state_initialize_decoder_result>
          + sml::completion<init_run> [ guard::guard_child_failed<init_run>{} ]
          / action::effect_fail_initialize_child<error::codec_initialize_failed>{}
      , sml::state<state_initialize_generator_result> <= sml::state<state_initialize_executor_result>
          + sml::completion<init_run> [ guard::guard_child_succeeded<init_run>{} ]
          / action::effect_initialize_generator{}
      , sml::state<state_failed> <= sml::state<state_initialize_executor_result>
          + sml::completion<init_run> [ guard::guard_child_failed<init_run>{} ]
          / action::effect_fail_initialize_child<error::executor_initialize_failed>{}
      , sml::state<state_load_voice_result> <= sml::state<state_initialize_generator_result>
          + sml::completion<init_run> [ guard::guard_child_succeeded<init_run>{} ]
          / action::effect_load_voice{}
      , sml::state<state_failed> <= sml::state<state_initialize_generator_result>
          + sml::completion<init_run> [ guard::guard_child_failed<init_run>{} ]
          / action::effect_fail_initialize_child<error::generator_initialize_failed>{}
      , sml::state<state_voice_prefill> <= sml::state<state_load_voice_result>
          + sml::completion<init_run> [ guard::guard_child_succeeded<init_run>{} ]
          / action::effect_publish_initialize_done{}
      , sml::state<state_failed> <= sml::state<state_load_voice_result>
          + sml::completion<init_run> [ guard::guard_child_failed<init_run>{} ]
          / action::effect_fail_initialize_child<error::voice_load_failed>{}

      //------------------------------------------------------------------------------//
      // Voice and empty/system prompt prefill phases.
      , sml::state<state_voice_prefill_result> <= sml::state<state_voice_prefill>
          + sml::event<voice_run> / action::effect_prefill_voice{}
      , sml::state<state_voice_prefill> <= sml::state<state_voice_prefill_result>
          + sml::completion<voice_run> [ guard::guard_phase_succeeded_incomplete<voice_run>{} ]
          / action::effect_publish_advance_voice_done{}
      , sml::state<state_prompt_begin_result> <= sml::state<state_voice_prefill_result>
          + sml::completion<voice_run> [ guard::guard_phase_succeeded_complete<voice_run>{} ]
          / action::effect_begin_prompt{}
      , sml::state<state_failed> <= sml::state<state_voice_prefill_result>
          + sml::completion<voice_run> [ guard::guard_phase_failed<voice_run>{} ]
          / action::effect_fail_advance_voice<error::voice_prefill_failed>{}
      , sml::state<state_prompt_prefill> <= sml::state<state_prompt_begin_result>
          + sml::completion<voice_run> [ guard::guard_child_succeeded<voice_run>{} ]
          / action::effect_publish_advance_voice_done{}
      , sml::state<state_failed> <= sml::state<state_prompt_begin_result>
          + sml::completion<voice_run> [ guard::guard_child_failed<voice_run>{} ]
          / action::effect_fail_advance_voice<error::prompt_begin_failed>{}
      , sml::state<state_prompt_prefill_result> <= sml::state<state_prompt_prefill>
          + sml::event<prompt_run> / action::effect_prefill_prompt{}
      , sml::state<state_prompt_prefill> <= sml::state<state_prompt_prefill_result>
          + sml::completion<prompt_run> [ guard::guard_phase_succeeded_incomplete<prompt_run>{} ]
          / action::effect_publish_advance_prompt_done{}
      , sml::state<state_live> <= sml::state<state_prompt_prefill_result>
          + sml::completion<prompt_run> [ guard::guard_phase_succeeded_complete<prompt_run>{} ]
          / action::effect_publish_advance_prompt_done{}
      , sml::state<state_failed> <= sml::state<state_prompt_prefill_result>
          + sml::completion<prompt_run> [ guard::guard_phase_failed<prompt_run>{} ]
          / action::effect_fail_advance_prompt<error::prompt_prefill_failed>{}

      //------------------------------------------------------------------------------//
      // Live input frames. Produced/decode selection is explicit.
      , sml::state<state_live_encode_result> <= sml::state<state_live>
          + sml::event<live_run> [ guard::guard_frame_request_valid<live_run>{} ]
          / action::effect_encode_frame<live_run>{}
      , sml::state<state_failed> <= sml::state<state_live>
          + sml::event<live_run> [ guard::guard_frame_request_invalid<live_run>{} ]
          / action::effect_fail_frame<live_run, error::invalid_request>{}
      , sml::state<state_live_generate_result> <= sml::state<state_live_encode_result>
          + sml::completion<live_run> [ guard::guard_child_succeeded<live_run>{} ]
          / action::effect_generate_frame<live_run>{}
      , sml::state<state_failed> <= sml::state<state_live_encode_result>
          + sml::completion<live_run> [ guard::guard_child_failed<live_run>{} ]
          / action::effect_fail_frame<live_run, error::encode_failed>{}
      , sml::state<state_live_decode_result> <= sml::state<state_live_generate_result>
          + sml::completion<live_run> [ guard::guard_frame_generated_and_produced<live_run>{} ]
          / action::effect_decode_frame<live_run>{}
      , sml::state<state_live> <= sml::state<state_live_generate_result>
          + sml::completion<live_run> [ guard::guard_frame_generated_without_output<live_run>{} ]
          / action::effect_publish_frame_done<live_run>{}
      , sml::state<state_failed> <= sml::state<state_live_generate_result>
          + sml::completion<live_run> [ guard::guard_frame_generate_failed<live_run>{} ]
          / action::effect_fail_frame<live_run, error::generate_failed>{}
      , sml::state<state_live> <= sml::state<state_live_decode_result>
          + sml::completion<live_run> [ guard::guard_child_succeeded<live_run>{} ]
          / action::effect_publish_frame_done<live_run>{}
      , sml::state<state_failed> <= sml::state<state_live_decode_result>
          + sml::completion<live_run> [ guard::guard_child_failed<live_run>{} ]
          / action::effect_fail_frame<live_run, error::decode_failed>{}

      //------------------------------------------------------------------------------//
      // Explicit silence flush and terminal phases.
      , sml::state<state_flush> <= sml::state<state_live>
          + sml::event<begin_flush_run> / action::effect_publish_begin_flush_done{}
      , sml::state<state_flush_encode_result> <= sml::state<state_flush>
          + sml::event<flush_run> [ guard::guard_frame_request_valid<flush_run>{} ]
          / action::effect_encode_frame<flush_run>{}
      , sml::state<state_failed> <= sml::state<state_flush>
          + sml::event<flush_run> [ guard::guard_frame_request_invalid<flush_run>{} ]
          / action::effect_fail_frame<flush_run, error::invalid_request>{}
      , sml::state<state_flush_generate_result> <= sml::state<state_flush_encode_result>
          + sml::completion<flush_run> [ guard::guard_child_succeeded<flush_run>{} ]
          / action::effect_generate_frame<flush_run>{}
      , sml::state<state_failed> <= sml::state<state_flush_encode_result>
          + sml::completion<flush_run> [ guard::guard_child_failed<flush_run>{} ]
          / action::effect_fail_frame<flush_run, error::encode_failed>{}
      , sml::state<state_flush_decode_result> <= sml::state<state_flush_generate_result>
          + sml::completion<flush_run> [ guard::guard_frame_generated_and_produced<flush_run>{} ]
          / action::effect_decode_frame<flush_run>{}
      , sml::state<state_flush> <= sml::state<state_flush_generate_result>
          + sml::completion<flush_run> [ guard::guard_frame_generated_without_output<flush_run>{} ]
          / action::effect_publish_frame_done<flush_run>{}
      , sml::state<state_failed> <= sml::state<state_flush_generate_result>
          + sml::completion<flush_run> [ guard::guard_frame_generate_failed<flush_run>{} ]
          / action::effect_fail_frame<flush_run, error::generate_failed>{}
      , sml::state<state_flush> <= sml::state<state_flush_decode_result>
          + sml::completion<flush_run> [ guard::guard_child_succeeded<flush_run>{} ]
          / action::effect_publish_frame_done<flush_run>{}
      , sml::state<state_failed> <= sml::state<state_flush_decode_result>
          + sml::completion<flush_run> [ guard::guard_child_failed<flush_run>{} ]
          / action::effect_fail_frame<flush_run, error::decode_failed>{}
      , sml::state<state_done> <= sml::state<state_flush>
          + sml::event<finish_run> / action::effect_publish_finish_done{}

      //------------------------------------------------------------------------------//
      // Every external event outside its modeled phase is explicit.
      , sml::state<state_failed> <= sml::state<state_uninitialized>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_initialize_encoder_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_initialize_decoder_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_initialize_executor_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_initialize_generator_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_load_voice_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_voice_prefill>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_voice_prefill_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_prompt_begin_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_prompt_prefill>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_prompt_prefill_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_live>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_live_encode_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_live_generate_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_live_decode_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_flush>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_flush_encode_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_flush_generate_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_flush_decode_result>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_done>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
      , sml::state<state_failed> <= sml::state<state_failed>
          + sml::unexpected_event<sml::_> / action::effect_mark_unexpected{}
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  explicit sm(const action::dependencies &deps)
      : base_type(std::in_place, deps) {}

  sm(const sm &) = delete;
  sm &operator=(const sm &) = delete;
  sm(sm &&) = delete;
  sm &operator=(sm &&) = delete;

  bool process_event(const event::initialize &request) {
    event::initialize_ctx ctx{};
    event::initialize_run runtime_ev{request, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::advance_voice &request) {
    event::phase_ctx ctx{};
    event::advance_voice_run runtime_ev{request, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::advance_prompt &request) {
    event::phase_ctx ctx{};
    event::advance_prompt_run runtime_ev{request, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::live_frame &request) {
    event::frame_ctx ctx{};
    event::live_frame_run runtime_ev{request, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::begin_flush &request) {
    event::simple_ctx ctx{};
    event::begin_flush_run runtime_ev{request, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::flush_frame &request) {
    event::frame_ctx ctx{};
    event::flush_frame_run runtime_ev{request, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }

  bool process_event(const event::finish &request) {
    event::simple_ctx ctx{};
    event::finish_run runtime_ev{request, ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && ctx.err == action::detail_ns::to_error(error::none);
  }
};

using Session = sm;

} // namespace emel::speech::generator::moshi::personaplex::session
