#pragma once

// benchmark: scaffold

#include "emel/sm.hpp"
#include "emel/text/tokenizer/needle/actions.hpp"
#include "emel/text/tokenizer/needle/context.hpp"
#include "emel/text/tokenizer/needle/errors.hpp"
#include "emel/text/tokenizer/needle/events.hpp"
#include "emel/text/tokenizer/needle/guards.hpp"

namespace emel::text::tokenizer::needle {

// Loader lifecycle: unloaded until a load request parses the `.cact` RAW
// SentencePiece-BPE dump into the shared vocab; every rejection lands in
// state_errored with a typed load_error. Re-loading from loaded or errored is
// allowed (the request carries fresh blob and output storage).
struct state_unloaded {};
struct state_loaded {};
struct state_errored {};

struct state_load_request_decision {};
struct state_load_outcome_dispatch {};
struct state_load_done_callback_decision {};
struct state_load_error_callback_decision {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Load op.
        sml::state<state_load_request_decision> <= *sml::state<state_unloaded>
          + sml::event<event::load_runtime> / action::effect_begin_load
      , sml::state<state_load_request_decision> <= sml::state<state_loaded>
          + sml::event<event::load_runtime> / action::effect_begin_load
      , sml::state<state_load_request_decision> <= sml::state<state_errored>
          + sml::event<event::load_runtime> / action::effect_begin_load

      , sml::state<state_load_outcome_dispatch> <= sml::state<state_load_request_decision>
          + sml::completion<event::load_runtime> [ guard::guard_load_valid_request{} ]
          / action::effect_exec_load
      , sml::state<state_load_outcome_dispatch> <= sml::state<state_load_request_decision>
          + sml::completion<event::load_runtime> [ guard::guard_load_invalid_request{} ]
          / action::effect_mark_load_invalid_request

      , sml::state<state_load_done_callback_decision> <= sml::state<state_load_outcome_dispatch>
          + sml::completion<event::load_runtime> [ guard::guard_load_error_none{} ]
      , sml::state<state_load_error_callback_decision> <= sml::state<state_load_outcome_dispatch>
          + sml::completion<event::load_runtime> [ guard::guard_load_error_invalid_request{} ]
      , sml::state<state_load_error_callback_decision> <= sml::state<state_load_outcome_dispatch>
          + sml::completion<event::load_runtime> [ guard::guard_load_error_model_invalid{} ]
      , sml::state<state_load_error_callback_decision> <= sml::state<state_load_outcome_dispatch>
          + sml::completion<event::load_runtime> [ guard::guard_load_error_capacity{} ]
      , sml::state<state_load_error_callback_decision> <= sml::state<state_load_outcome_dispatch>
          + sml::completion<event::load_runtime> [ guard::guard_load_error_parse_failed{} ]
      , sml::state<state_load_error_callback_decision> <= sml::state<state_load_outcome_dispatch>
          + sml::completion<event::load_runtime> [ guard::guard_load_error_internal_error{} ]
      , sml::state<state_load_error_callback_decision> <= sml::state<state_load_outcome_dispatch>
          + sml::completion<event::load_runtime> [ guard::guard_load_error_unknown{} ]

      , sml::state<state_loaded> <= sml::state<state_load_done_callback_decision>
          + sml::completion<event::load_runtime> [ guard::guard_load_done_callback_present{} ]
          / action::effect_publish_load_done
      , sml::state<state_errored> <= sml::state<state_load_done_callback_decision>
          + sml::completion<event::load_runtime> [ guard::guard_load_done_callback_absent{} ]
          / action::effect_mark_load_invalid_request

      , sml::state<state_errored> <= sml::state<state_load_error_callback_decision>
          + sml::completion<event::load_runtime> [ guard::guard_load_error_callback_present{} ]
          / action::effect_publish_load_error
      , sml::state<state_errored> <= sml::state<state_load_error_callback_decision>
          + sml::completion<event::load_runtime> [ guard::guard_load_error_callback_absent{} ]

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<state_errored> <= sml::state<state_unloaded> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_loaded> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_errored> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_load_request_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_load_outcome_dispatch> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_load_done_callback_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
      , sml::state<state_errored> <= sml::state<state_load_error_callback_decision> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  sm() : base_type() {}

  bool process_event(const event::load &ev) {
    event::load_ctx ctx{};
    event::load_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.err == emel::error::cast(error::none);
  }
};

using NeedleTokenizerLoader = sm;

} // namespace emel::text::tokenizer::needle
