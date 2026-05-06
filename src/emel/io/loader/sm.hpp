#pragma once

// benchmark: designed

#include "emel/io/loader/actions.hpp"
#include "emel/io/loader/context.hpp"
#include "emel/io/loader/detail.hpp"
#include "emel/io/loader/events.hpp"
#include "emel/io/loader/guards.hpp"
#include "emel/sm.hpp"

namespace emel::io::loader {

struct state_ready {};
struct state_request_decision {};
struct state_no_strategy_error_decision {};
struct state_unsupported_strategy_error_decision {};
struct state_error_callback {};
struct state_read_dispatch_decision {};
struct state_done_decision {};
struct state_done_callback {};
struct state_batch_request_decision {};
struct state_batch_unsupported_strategy_error_decision {};
struct state_batch_error_callback {};
struct state_batch_read_dispatch_decision {};
struct state_batch_done_decision {};
struct state_batch_done_callback {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Loading strategy boundary. Concrete strategies are explicit future routes.
        sml::state<state_request_decision> <= *sml::state<state_ready>
          + sml::event<detail::load_tensor_runtime>
          [ guard::tensor_span_valid{} ]
          / action::effect_begin_load_tensor
      , sml::state<state_no_strategy_error_decision> <= sml::state<state_request_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::strategy_none{} ]
          / action::effect_mark_unsupported_strategy
      , sml::state<state_unsupported_strategy_error_decision> <=
          sml::state<state_request_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::strategy_mapped_file{} ]
          / action::effect_mark_unsupported_strategy
      , sml::state<state_read_dispatch_decision> <=
          sml::state<state_request_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::strategy_read_copy_with_actor{} ]
          / action::effect_dispatch_read_tensor
      , sml::state<state_unsupported_strategy_error_decision> <=
          sml::state<state_request_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::strategy_read_copy_without_actor{} ]
          / action::effect_mark_unsupported_strategy
      , sml::state<state_unsupported_strategy_error_decision> <=
          sml::state<state_request_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::strategy_external_buffer{} ]
          / action::effect_mark_unsupported_strategy
      , sml::state<state_unsupported_strategy_error_decision> <=
          sml::state<state_request_decision>
          + sml::completion<detail::load_tensor_runtime>
          / action::effect_mark_unsupported_strategy
      , sml::state<state_no_strategy_error_decision> <= sml::state<state_ready>
          + sml::event<detail::load_tensor_runtime>
          [ guard::tensor_span_invalid{} ]
          / action::effect_mark_invalid_request

      , sml::state<state_done_decision> <= sml::state<state_read_dispatch_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::read_load_succeeded{} ]
      , sml::state<state_unsupported_strategy_error_decision> <=
          sml::state<state_read_dispatch_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::read_load_failed{} ]

      //------------------------------------------------------------------------------//
      // Completion/error publication is explicit even before concrete strategies exist.
      , sml::state<state_done_callback> <= sml::state<state_done_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::done_callback_present{} ]
          / action::effect_publish_load_tensor_done
      , sml::state<state_ready> <= sml::state<state_done_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::done_callback_absent{} ]
          / action::effect_record_load_tensor_done
      , sml::state<state_ready> <= sml::state<state_done_callback>
          + sml::completion<detail::load_tensor_runtime>
          / action::effect_record_load_tensor_done

      , sml::state<state_error_callback> <= sml::state<state_no_strategy_error_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_load_tensor_error
      , sml::state<state_ready> <= sml::state<state_no_strategy_error_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_load_tensor_error
      , sml::state<state_error_callback> <=
          sml::state<state_unsupported_strategy_error_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::error_callback_present{} ]
          / action::effect_publish_load_tensor_error
      , sml::state<state_ready> <=
          sml::state<state_unsupported_strategy_error_decision>
          + sml::completion<detail::load_tensor_runtime>
          [ guard::error_callback_absent{} ]
          / action::effect_record_load_tensor_error
      , sml::state<state_ready> <= sml::state<state_error_callback>
          + sml::completion<detail::load_tensor_runtime>
          / action::effect_record_load_tensor_error

      //------------------------------------------------------------------------------//
      // Batch read/copy route. The selected strategy dispatches exactly one
      // public io/read batch event; source copying remains owned by io/read.
      , sml::state<state_batch_request_decision> <= *sml::state<state_ready>
          + sml::event<detail::load_tensor_batch_runtime>
          [ guard::batch_span_valid{} ]
          / action::effect_begin_load_tensor_batch
      , sml::state<state_batch_unsupported_strategy_error_decision> <=
          sml::state<state_ready>
          + sml::event<detail::load_tensor_batch_runtime>
          [ guard::batch_span_invalid{} ]
          / action::effect_mark_load_tensor_batch_invalid_request
      , sml::state<state_batch_read_dispatch_decision> <=
          sml::state<state_batch_request_decision>
          + sml::completion<detail::load_tensor_batch_runtime>
          [ guard::strategy_read_copy_batch_with_actor{} ]
          / action::effect_dispatch_read_tensor_batch
      , sml::state<state_batch_unsupported_strategy_error_decision> <=
          sml::state<state_batch_request_decision>
          + sml::completion<detail::load_tensor_batch_runtime>
          [ guard::strategy_read_copy_batch_without_actor{} ]
          / action::effect_mark_load_tensor_batch_unsupported_strategy
      , sml::state<state_batch_unsupported_strategy_error_decision> <=
          sml::state<state_batch_request_decision>
          + sml::completion<detail::load_tensor_batch_runtime>
          / action::effect_mark_load_tensor_batch_unsupported_strategy

      , sml::state<state_batch_done_decision> <=
          sml::state<state_batch_read_dispatch_decision>
          + sml::completion<detail::load_tensor_batch_runtime>
          [ guard::read_batch_succeeded{} ]
      , sml::state<state_batch_unsupported_strategy_error_decision> <=
          sml::state<state_batch_read_dispatch_decision>
          + sml::completion<detail::load_tensor_batch_runtime>
          [ guard::read_batch_failed{} ]
          / action::effect_record_read_tensor_batch_failed

      //------------------------------------------------------------------------------//
      // Batch completion/error publication.
      , sml::state<state_batch_done_callback> <=
          sml::state<state_batch_done_decision>
          + sml::completion<detail::load_tensor_batch_runtime>
          [ guard::batch_done_callback_present{} ]
          / action::effect_publish_load_tensor_batch_done
      , sml::state<state_ready> <= sml::state<state_batch_done_decision>
          + sml::completion<detail::load_tensor_batch_runtime>
          [ guard::batch_done_callback_absent{} ]
          / action::effect_record_load_tensor_batch_done
      , sml::state<state_ready> <= sml::state<state_batch_done_callback>
          + sml::completion<detail::load_tensor_batch_runtime>
          / action::effect_record_load_tensor_batch_done

      , sml::state<state_batch_error_callback> <=
          sml::state<state_batch_unsupported_strategy_error_decision>
          + sml::completion<detail::load_tensor_batch_runtime>
          [ guard::batch_error_callback_present{} ]
          / action::effect_publish_load_tensor_batch_error
      , sml::state<state_ready> <=
          sml::state<state_batch_unsupported_strategy_error_decision>
          + sml::completion<detail::load_tensor_batch_runtime>
          [ guard::batch_error_callback_absent{} ]
          / action::effect_record_load_tensor_batch_error
      , sml::state<state_ready> <= sml::state<state_batch_error_callback>
          + sml::completion<detail::load_tensor_batch_runtime>
          / action::effect_record_load_tensor_batch_error

      //------------------------------------------------------------------------------//
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_request_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_no_strategy_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_unsupported_strategy_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_error_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_read_dispatch_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_done_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_done_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_batch_request_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_batch_unsupported_strategy_error_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_batch_error_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <=
          sml::state<state_batch_read_dispatch_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_batch_done_decision>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_batch_done_callback>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::base_type;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  bool process_event(const event::load_tensor &ev) {
    detail::runtime_status ctx{};
    detail::load_tensor_runtime runtime{ev, ctx};
    const bool accepted = base_type::process_event(runtime);
    return accepted && ctx.ok;
  }

  bool process_event(const event::load_tensor_batch &ev) {
    detail::batch_runtime_status status{};
    detail::load_tensor_batch_runtime runtime{ev, status};
    const bool accepted = base_type::process_event(runtime);
    return accepted && status.ok;
  }
};

} // namespace emel::io::loader
