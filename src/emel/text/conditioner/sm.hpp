#pragma once

#include <array>
#include <cstdint>

#include "emel/sm.hpp"
#include "emel/text/conditioner/actions.hpp"
#include "emel/text/conditioner/detail.hpp"
#include "emel/text/conditioner/errors.hpp"
#include "emel/text/conditioner/events.hpp"
#include "emel/text/conditioner/guards.hpp"

namespace emel::text::conditioner {

struct uninitialized {};
struct binding {};
struct bind_decision {};
struct bind_success {};
struct bind_error {};
struct bind_publish_success {};
struct bind_publish_error {};
struct preparing {};
struct format_decision {};
struct tokenizing {};
struct tokenize_decision {};
struct prepare_success {};
struct prepare_error {};
struct prepare_publish_success_count {};
struct prepare_publish_success_error {};
struct prepare_publish_error_count {};
struct prepare_publish_error {};
struct done {};
struct errored {};
struct idle {};
struct unexpected {};

/*
conditioner architecture notes (single source of truth)

state purpose
- bind_* states model bind dispatch, decision, and publication.
- prepare_* states model format/tokenize dispatch, decision, and publication.
- idle/done/errored are quiescent externally observable outcomes.
- unexpected is explicit unexpected-event sink.

guard semantics
- request validity and all runtime branching are explicit guard predicates.
- publication guards select optional output/callback paths.

action side effects
- actions are branch-free phase kernels and output publishers.
- callbacks are emitted only from guarded transitions.
*/
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<binding> <= *sml::state<uninitialized> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::begin_bind
      , sml::state<bind_error> <= sml::state<uninitialized> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::reject_bind
      , sml::state<prepare_error> <= sml::state<uninitialized> + sml::event<event::prepare_runtime>
          / action::reject_prepare

      //------------------------------------------------------------------------------//
      , sml::state<binding> <= sml::state<idle> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::begin_bind
      , sml::state<bind_error> <= sml::state<idle> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::reject_bind
      , sml::state<preparing> <= sml::state<idle> + sml::event<event::prepare_runtime>
          [ guard::valid_prepare_with_bind_defaults{} ]
          / action::begin_prepare_bind_defaults
      , sml::state<preparing> <= sml::state<idle> + sml::event<event::prepare_runtime>
          [ guard::valid_prepare_with_request_overrides{} ]
          / action::begin_prepare_from_request
      , sml::state<prepare_error> <= sml::state<idle> + sml::event<event::prepare_runtime>
          [ guard::invalid_prepare{} ]
          / action::reject_prepare

      //------------------------------------------------------------------------------//
      , sml::state<binding> <= sml::state<done> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::begin_bind
      , sml::state<bind_error> <= sml::state<done> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::reject_bind
      , sml::state<preparing> <= sml::state<done> + sml::event<event::prepare_runtime>
          [ guard::valid_prepare_with_bind_defaults{} ]
          / action::begin_prepare_bind_defaults
      , sml::state<preparing> <= sml::state<done> + sml::event<event::prepare_runtime>
          [ guard::valid_prepare_with_request_overrides{} ]
          / action::begin_prepare_from_request
      , sml::state<prepare_error> <= sml::state<done> + sml::event<event::prepare_runtime>
          [ guard::invalid_prepare{} ]
          / action::reject_prepare

      //------------------------------------------------------------------------------//
      , sml::state<binding> <= sml::state<errored> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::begin_bind
      , sml::state<bind_error> <= sml::state<errored> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::reject_bind
      , sml::state<preparing> <= sml::state<errored> + sml::event<event::prepare_runtime>
          [ guard::valid_prepare_with_bind_defaults{} ]
          / action::begin_prepare_bind_defaults
      , sml::state<preparing> <= sml::state<errored> + sml::event<event::prepare_runtime>
          [ guard::valid_prepare_with_request_overrides{} ]
          / action::begin_prepare_from_request
      , sml::state<prepare_error> <= sml::state<errored> + sml::event<event::prepare_runtime>
          [ guard::invalid_prepare{} ]
          / action::reject_prepare

      //------------------------------------------------------------------------------//
      , sml::state<binding> <= sml::state<unexpected> + sml::event<event::bind_runtime>
          [ guard::valid_bind{} ]
          / action::begin_bind
      , sml::state<bind_error> <= sml::state<unexpected> + sml::event<event::bind_runtime>
          [ guard::invalid_bind{} ]
          / action::reject_bind
      , sml::state<preparing> <= sml::state<unexpected> + sml::event<event::prepare_runtime>
          [ guard::valid_prepare_with_bind_defaults{} ]
          / action::begin_prepare_bind_defaults
      , sml::state<preparing> <= sml::state<unexpected> + sml::event<event::prepare_runtime>
          [ guard::valid_prepare_with_request_overrides{} ]
          / action::begin_prepare_from_request
      , sml::state<prepare_error> <= sml::state<unexpected> + sml::event<event::prepare_runtime>
          [ guard::invalid_prepare{} ]
          / action::reject_prepare

      //------------------------------------------------------------------------------//
      , sml::state<bind_decision> <= sml::state<binding>
          + sml::completion<event::bind_runtime>
          / action::dispatch_bind_tokenizer
      , sml::state<bind_error> <= sml::state<bind_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_rejected_no_error{} ]
          / action::bind_error_backend
      , sml::state<bind_error> <= sml::state<bind_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_error_code_present{} ]
          / action::bind_error_from_code
      , sml::state<bind_success> <= sml::state<bind_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_successful{} ]
          / action::bind_success

      //------------------------------------------------------------------------------//
      , sml::state<bind_publish_success> <= sml::state<bind_success>
          + sml::completion<event::bind_runtime> [ guard::has_bind_error_out{} ]
          / action::write_bind_error_out
      , sml::state<bind_publish_success> <= sml::state<bind_success>
          + sml::completion<event::bind_runtime> [ guard::no_bind_error_out{} ]
      , sml::state<idle> <= sml::state<bind_publish_success>
          + sml::completion<event::bind_runtime> [ guard::has_bind_done_callback{} ]
          / action::emit_bind_done
      , sml::state<idle> <= sml::state<bind_publish_success>
          + sml::completion<event::bind_runtime> [ guard::no_bind_done_callback{} ]

      //------------------------------------------------------------------------------//
      , sml::state<bind_publish_error> <= sml::state<bind_error>
          + sml::completion<event::bind_runtime> [ guard::has_bind_error_out{} ]
          / action::write_bind_error_out
      , sml::state<bind_publish_error> <= sml::state<bind_error>
          + sml::completion<event::bind_runtime> [ guard::no_bind_error_out{} ]
      , sml::state<errored> <= sml::state<bind_publish_error>
          + sml::completion<event::bind_runtime> [ guard::has_bind_error_callback{} ]
          / action::emit_bind_error
      , sml::state<errored> <= sml::state<bind_publish_error>
          + sml::completion<event::bind_runtime> [ guard::no_bind_error_callback{} ]

      //------------------------------------------------------------------------------//
      , sml::state<format_decision> <= sml::state<preparing>
          + sml::completion<event::prepare_runtime>
          / action::dispatch_format
      , sml::state<prepare_error> <= sml::state<format_decision>
          + sml::completion<event::prepare_runtime> [ guard::format_rejected_no_error{} ]
          / action::format_error_backend
      , sml::state<prepare_error> <= sml::state<format_decision>
          + sml::completion<event::prepare_runtime> [ guard::format_error_code_present{} ]
          / action::format_error_from_code
      , sml::state<prepare_error> <= sml::state<format_decision>
          + sml::completion<event::prepare_runtime> [ guard::format_length_overflow{} ]
          / action::format_error_invalid_argument
      , sml::state<tokenizing> <= sml::state<format_decision>
          + sml::completion<event::prepare_runtime> [ guard::format_successful{} ]

      //------------------------------------------------------------------------------//
      , sml::state<tokenize_decision> <= sml::state<tokenizing>
          + sml::completion<event::prepare_runtime>
          / action::dispatch_tokenize
      , sml::state<prepare_error> <= sml::state<tokenize_decision>
          + sml::completion<event::prepare_runtime> [ guard::tokenize_rejected_no_error{} ]
          / action::tokenize_error_backend
      , sml::state<prepare_error> <= sml::state<tokenize_decision>
          + sml::completion<event::prepare_runtime> [ guard::tokenize_error_code_present{} ]
          / action::tokenize_error_from_code
      , sml::state<prepare_error> <= sml::state<tokenize_decision>
          + sml::completion<event::prepare_runtime> [ guard::tokenize_count_invalid{} ]
          / action::tokenize_error_backend
      , sml::state<prepare_success> <= sml::state<tokenize_decision>
          + sml::completion<event::prepare_runtime> [ guard::tokenize_successful{} ]
          / action::prepare_success

      //------------------------------------------------------------------------------//
      , sml::state<prepare_publish_success_count> <= sml::state<prepare_success>
          + sml::completion<event::prepare_runtime>
          / action::write_prepare_token_count
      , sml::state<prepare_publish_success_error> <= sml::state<prepare_publish_success_count>
          + sml::completion<event::prepare_runtime>
          / action::write_prepare_error_out
      , sml::state<done> <= sml::state<prepare_publish_success_error>
          + sml::completion<event::prepare_runtime> [ guard::has_prepare_done_callback{} ]
          / action::emit_prepare_done
      , sml::state<done> <= sml::state<prepare_publish_success_error>
          + sml::completion<event::prepare_runtime> [ guard::no_prepare_done_callback{} ]

      //------------------------------------------------------------------------------//
      , sml::state<prepare_publish_error_count> <= sml::state<prepare_error>
          + sml::completion<event::prepare_runtime>
          / action::write_prepare_token_count
      , sml::state<prepare_publish_error> <= sml::state<prepare_publish_error_count>
          + sml::completion<event::prepare_runtime>
          / action::write_prepare_error_out
      , sml::state<errored> <= sml::state<prepare_publish_error>
          + sml::completion<event::prepare_runtime> [ guard::has_prepare_error_callback{} ]
          / action::emit_prepare_error
      , sml::state<errored> <= sml::state<prepare_publish_error>
          + sml::completion<event::prepare_runtime> [ guard::no_prepare_error_callback{} ]

      //------------------------------------------------------------------------------//
      , sml::state<unexpected> <= sml::state<uninitialized> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<bind_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<bind_success> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<bind_error> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<bind_publish_success> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<bind_publish_error> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<preparing> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<format_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<tokenizing> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<tokenize_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<prepare_success> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<prepare_error> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<prepare_publish_success_count> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<prepare_publish_success_error> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<prepare_publish_error_count> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<prepare_publish_error> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<done> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<errored> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<idle> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<unexpected> + sml::unexpected_event<sml::_>
          / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;

  sm() : base_type() {}

  bool process_event(const event::bind & ev) {
    event::bind_ctx runtime_ctx{
      .err = error::none,
      .result = false,
      .bind_accepted = false,
      .bind_err_code = detail::to_local_error_code(error::none),
    };
    event::bind_runtime runtime_ev{ev, runtime_ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && runtime_ctx.result;
  }

  bool process_event(const event::prepare & ev) {
    std::array<char, action::k_max_formatted_bytes> formatted;
    event::prepare_ctx runtime_ctx{
      .err = error::none,
      .formatted = formatted.data(),
      .formatted_capacity = formatted.size(),
      .formatted_length = 0,
      .add_special = true,
      .parse_special = false,
      .token_count = 0,
      .result = false,
      .format_accepted = false,
      .format_err_code = detail::to_local_error_code(error::none),
      .tokenize_accepted = false,
      .tokenize_err_code = detail::to_local_error_code(error::none),
    };
    event::prepare_runtime runtime_ev{ev, runtime_ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && runtime_ctx.result;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

};

using Conditioner = sm;

}  // namespace emel::text::conditioner
