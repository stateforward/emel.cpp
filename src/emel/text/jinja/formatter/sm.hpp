#pragma once

/*
design doc: docs/designs/text/jinja/formatter.design.md
*/

#include <cstdint>

#include "emel/text/jinja/formatter/actions.hpp"
#include "emel/text/jinja/formatter/context.hpp"
#include "emel/text/jinja/formatter/detail.hpp"
#include "emel/text/jinja/formatter/events.hpp"
#include "emel/text/jinja/formatter/guards.hpp"
#include "emel/sm.hpp"

namespace emel::text::jinja::formatter {

struct initialized {};
struct request_decision {};
struct copy_exec {};
struct result_decision {};
struct done {};
struct errored {};
struct unexpected {};

/**
 * jinja renderer orchestration model.
 *
 * state purposes:
 * - `initialized`: idle state awaiting render intent.
 * - `request_decision`: route copy/empty/overflow paths for one dispatch.
 * - `copy_exec`: copy source text to caller output buffer.
 * - `result_decision`: route done/error callback dispatch for one dispatch.
 * - `done`/`errored`: terminal outcomes.
 * - `unexpected`: sequencing contract violation.
 *
 * guard semantics:
 * - `valid_render`/`invalid_render` validate request parameters.
 * - `source_*` guards model copy-path routing.
 *
 * action side effects:
 * - `begin_render` marks start of one dispatch pass.
 * - `mark_empty_output` marks zero-length success path.
 * - `copy_source_text` performs one bounded buffer copy.
 * - `mark_capacity_error` marks overflow path.
 * - `reject_invalid_render` marks invalid-request path.
 * - `dispatch_done`/`dispatch_error` send synchronous completion callbacks.
 * - `on_unexpected` reports sequencing violations.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
        sml::state<request_decision> <= *sml::state<initialized>
          + sml::event<event::render_runtime>[ guard::valid_render{} ]
          / action::begin_render
      , sml::state<result_decision> <= sml::state<initialized>
          + sml::event<event::render_runtime>[ guard::invalid_render_with_callbacks{} ]
          / action::reject_invalid_render
      , sml::state<errored> <= sml::state<initialized>
          + sml::event<event::render_runtime>[ guard::invalid_render_without_callbacks{} ]
          / action::reject_invalid_render

      , sml::state<request_decision> <= sml::state<done>
          + sml::event<event::render_runtime>[ guard::valid_render{} ]
          / action::begin_render
      , sml::state<result_decision> <= sml::state<done>
          + sml::event<event::render_runtime>[ guard::invalid_render_with_callbacks{} ]
          / action::reject_invalid_render
      , sml::state<errored> <= sml::state<done>
          + sml::event<event::render_runtime>[ guard::invalid_render_without_callbacks{} ]
          / action::reject_invalid_render

      , sml::state<request_decision> <= sml::state<errored>
          + sml::event<event::render_runtime>[ guard::valid_render{} ]
          / action::begin_render
      , sml::state<result_decision> <= sml::state<errored>
          + sml::event<event::render_runtime>[ guard::invalid_render_with_callbacks{} ]
          / action::reject_invalid_render
      , sml::state<errored> <= sml::state<errored>
          + sml::event<event::render_runtime>[ guard::invalid_render_without_callbacks{} ]
          / action::reject_invalid_render

      , sml::state<request_decision> <= sml::state<unexpected>
          + sml::event<event::render_runtime>[ guard::valid_render{} ]
          / action::begin_render
      , sml::state<result_decision> <= sml::state<unexpected>
          + sml::event<event::render_runtime>[ guard::invalid_render_with_callbacks{} ]
          / action::reject_invalid_render
      , sml::state<errored> <= sml::state<unexpected>
          + sml::event<event::render_runtime>[ guard::invalid_render_without_callbacks{} ]
          / action::reject_invalid_render

      //------------------------------------------------------------------------------//
      , sml::state<result_decision> <= sml::state<request_decision>
          + sml::completion<event::render_runtime> [ guard::source_empty{} ]
          / action::mark_empty_output
      , sml::state<copy_exec> <= sml::state<request_decision>
          + sml::completion<event::render_runtime> [ guard::copy_ready{} ]
          / action::copy_source_text
      , sml::state<result_decision> <= sml::state<request_decision>
          + sml::completion<event::render_runtime> [ guard::source_overflow{} ]
          / action::mark_capacity_error

      , sml::state<result_decision> <= sml::state<copy_exec>
          + sml::completion<event::render_runtime>
      , sml::state<done> <= sml::state<result_decision>
          + sml::completion<event::render_runtime> [ guard::request_ok{} ]
          / action::dispatch_done
      , sml::state<errored> <= sml::state<result_decision>
          + sml::completion<event::render_runtime> [ guard::request_failed{} ]
          / action::dispatch_error

      //------------------------------------------------------------------------------//
      , sml::state<unexpected> <= sml::state<initialized> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<request_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<copy_exec> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<result_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<done> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<errored> + sml::unexpected_event<sml::_>
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
  explicit sm(const action::context & ctx) : base_type(ctx) {}

  bool process_event(const event::render & ev) {
    size_t output_length_sink = 0;
    bool output_truncated_sink = false;
    int32_t error_sink = detail::to_error_code(error::none);
    size_t error_pos_sink = 0;

    event::render_ctx runtime_ctx{
        detail::bind_optional(ev.output_length, output_length_sink),
        detail::bind_optional(ev.output_truncated, output_truncated_sink),
        detail::bind_optional(ev.error_out, error_sink),
        detail::bind_optional(ev.error_pos_out, error_pos_sink),
    };
    event::render_runtime runtime_ev{ev, runtime_ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    return accepted && runtime_ctx.err == error::none;
  }

  using base_type::process_event;
  using base_type::is;
  using base_type::visit_current_states;
};

using Formatter = sm;

}  // namespace emel::text::jinja::formatter
