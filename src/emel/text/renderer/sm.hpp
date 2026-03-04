#pragma once


#include <cstdint>

#include "emel/error/error.hpp"
#include "emel/sm.hpp"
#include "emel/text/renderer/actions.hpp"
#include "emel/text/renderer/errors.hpp"
#include "emel/text/renderer/events.hpp"
#include "emel/text/renderer/guards.hpp"

namespace emel::text::renderer {

struct uninitialized {};
struct initializing {};
struct initialization_decision {};
struct initialize_publish_success {};
struct initialize_publish_error {};
struct initialized {};
struct rendering {};
struct render_dispatch_decision {};
struct render_result_decision {};
struct render_commit_output_exec {};
struct render_strip_decision {};
struct render_strip_exec {};
struct render_strip_state_exec {};
struct render_stop_match_exec {};
struct render_finalize_decision {};
struct render_publish_success {};
struct render_publish_error {};
struct flushing {};
struct flush_publish_success {};
struct flush_publish_error {};
struct done {};
struct errored {};
struct unexpected {};

/*
renderer architecture notes (single source of truth)

state purpose
- uninitialized: awaiting dependency and vocab initializing.
- initializing/initialization_decision: initialization request acceptance and detokenizer attach outcome.
- initialize_publish_*: explicit success/error publication for initialization.
- initialized: ready for render and flush requests.
- rendering/render_*: render request setup, detokenizer dispatch, explicit commit/strip/stop/finalize phases.
- render_publish_*: explicit success/error publication for render.
- flushing: emits buffered bytes (utf-8 pending + stop holdback).
- flush_publish_*: explicit success/error publication for flush.
- done/errored: terminal outcomes for the latest request.
- unexpected: sequencing contract violation.

key invariants
- per-sequence utf-8 pending bytes and holdback are stored in renderer context.
- detokenizer stays stateless and receives all pending state via event payloads.
- output bytes are written only to caller-provided buffers.

control invariants
- input validation and all branch outcomes are explicit guard predicates.
- publication is split into explicit success/error branches per request kind.
*/
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      sml::state<initializing> <= *sml::state<uninitialized>
            + sml::event<event::initialize_runtime>[ guard::valid_initialize{} ]
          / action::begin_initialize
      , sml::state<initialize_publish_error> <= sml::state<uninitialized>
            + sml::event<event::initialize_runtime>[ guard::invalid_initialize{} ]
          / action::reject_initialize
      , sml::state<render_publish_error> <= sml::state<uninitialized>
            + sml::event<event::render_runtime>
          / action::reject_render
      , sml::state<flush_publish_error> <= sml::state<uninitialized>
            + sml::event<event::flush_runtime>
          / action::reject_flush

      , sml::state<initializing> <= sml::state<initialized>
            + sml::event<event::initialize_runtime>[ guard::valid_initialize{} ]
          / action::begin_initialize
      , sml::state<initialize_publish_error> <= sml::state<initialized>
            + sml::event<event::initialize_runtime>[ guard::invalid_initialize{} ]
          / action::reject_initialize
      , sml::state<rendering> <= sml::state<initialized>
            + sml::event<event::render_runtime>[ guard::valid_render{} ]
          / action::begin_render
      , sml::state<render_publish_error> <= sml::state<initialized>
            + sml::event<event::render_runtime>[ guard::invalid_render{} ]
          / action::reject_render
      , sml::state<flushing> <= sml::state<initialized>
            + sml::event<event::flush_runtime>[ guard::valid_flush{} ]
          / action::begin_flush
      , sml::state<flush_publish_error> <= sml::state<initialized>
            + sml::event<event::flush_runtime>[ guard::invalid_flush{} ]
          / action::reject_flush

      , sml::state<initializing> <= sml::state<done>
            + sml::event<event::initialize_runtime>[ guard::valid_initialize{} ]
          / action::begin_initialize
      , sml::state<initialize_publish_error> <= sml::state<done>
            + sml::event<event::initialize_runtime>[ guard::invalid_initialize{} ]
          / action::reject_initialize
      , sml::state<rendering> <= sml::state<done>
            + sml::event<event::render_runtime>[ guard::valid_render{} ]
          / action::begin_render
      , sml::state<render_publish_error> <= sml::state<done>
            + sml::event<event::render_runtime>[ guard::invalid_render{} ]
          / action::reject_render
      , sml::state<flushing> <= sml::state<done>
            + sml::event<event::flush_runtime>[ guard::valid_flush{} ]
          / action::begin_flush
      , sml::state<flush_publish_error> <= sml::state<done>
            + sml::event<event::flush_runtime>[ guard::invalid_flush{} ]
          / action::reject_flush

      , sml::state<initializing> <= sml::state<errored>
            + sml::event<event::initialize_runtime>[ guard::valid_initialize{} ]
          / action::begin_initialize
      , sml::state<initialize_publish_error> <= sml::state<errored>
            + sml::event<event::initialize_runtime>[ guard::invalid_initialize{} ]
          / action::reject_initialize
      , sml::state<rendering> <= sml::state<errored>
            + sml::event<event::render_runtime>[ guard::valid_render{} ]
          / action::begin_render
      , sml::state<render_publish_error> <= sml::state<errored>
            + sml::event<event::render_runtime>[ guard::invalid_render{} ]
          / action::reject_render
      , sml::state<flushing> <= sml::state<errored>
            + sml::event<event::flush_runtime>[ guard::valid_flush{} ]
          / action::begin_flush
      , sml::state<flush_publish_error> <= sml::state<errored>
            + sml::event<event::flush_runtime>[ guard::invalid_flush{} ]
          / action::reject_flush

      , sml::state<initializing> <= sml::state<unexpected>
            + sml::event<event::initialize_runtime>[ guard::valid_initialize{} ]
          / action::begin_initialize
      , sml::state<unexpected> <= sml::state<unexpected>
            + sml::event<event::initialize_runtime>[ guard::invalid_initialize{} ]
          / action::reject_initialize
      , sml::state<rendering> <= sml::state<unexpected>
            + sml::event<event::render_runtime>[ guard::valid_render{} ]
          / action::begin_render
      , sml::state<unexpected> <= sml::state<unexpected>
            + sml::event<event::render_runtime>[ guard::invalid_render{} ]
          / action::reject_render
      , sml::state<flushing> <= sml::state<unexpected>
            + sml::event<event::flush_runtime>[ guard::valid_flush{} ]
          / action::begin_flush
      , sml::state<unexpected> <= sml::state<unexpected>
            + sml::event<event::flush_runtime>[ guard::invalid_flush{} ]
          / action::reject_flush

      //------------------------------------------------------------------------------//
      , sml::state<initialize_publish_success> <= sml::state<initialization_decision>
            + sml::completion<event::initialize_runtime> [ guard::initialize_dispatch_ok{} ]
          / action::commit_initialize_success
      , sml::state<initialize_publish_error> <= sml::state<initialization_decision>
            + sml::completion<event::initialize_runtime> [ guard::initialize_dispatch_backend_failure{} ]
          / action::set_backend_error
      , sml::state<initialize_publish_error> <= sml::state<initialization_decision>
            + sml::completion<event::initialize_runtime> [ guard::initialize_dispatch_reported_error{} ]
          / action::set_error_from_detokenizer
      , sml::state<initialize_publish_error> <= sml::state<initialization_decision>
            + sml::completion<event::initialize_runtime>
          / action::set_error_from_detokenizer
      , sml::state<initialized> <= sml::state<initialize_publish_success>
            + sml::completion<event::initialize_runtime>
          / action::publish_initialize_done
      , sml::state<errored> <= sml::state<initialize_publish_error>
            + sml::completion<event::initialize_runtime>
          / action::publish_initialize_error

      , sml::state<initialization_decision> <= sml::state<initializing>
            + sml::completion<event::initialize_runtime>
          / action::dispatch_initialize_detokenizer

      //------------------------------------------------------------------------------//
      , sml::state<render_publish_success> <= sml::state<rendering>
            + sml::completion<event::render_runtime> [ guard::sequence_stop_matched{} ]
          / action::render_sequence_already_stopped
      , sml::state<render_dispatch_decision> <= sml::state<rendering>
            + sml::completion<event::render_runtime> [ guard::sequence_running{} ]
          / action::dispatch_render_detokenizer
      , sml::state<render_result_decision> <= sml::state<render_dispatch_decision>
            + sml::completion<event::render_runtime> [ guard::render_dispatch_ok{} ]
      , sml::state<render_commit_output_exec> <= sml::state<render_result_decision>
            + sml::completion<event::render_runtime>
      , sml::state<render_strip_decision> <= sml::state<render_commit_output_exec>
            + sml::completion<event::render_runtime>
          / action::commit_render_detokenizer_output
      , sml::state<render_strip_exec> <= sml::state<render_strip_decision>
            + sml::completion<event::render_runtime> [ guard::strip_needed{} ]
      , sml::state<render_strip_state_exec> <= sml::state<render_strip_decision>
            + sml::completion<event::render_runtime> [ guard::strip_not_needed{} ]
      , sml::state<render_publish_error> <= sml::state<render_strip_decision>
            + sml::completion<event::render_runtime>
          / action::ensure_last_error
      , sml::state<render_strip_state_exec> <= sml::state<render_strip_exec>
            + sml::completion<event::render_runtime>
          / action::strip_render_leading_space
      , sml::state<render_stop_match_exec> <= sml::state<render_strip_state_exec>
            + sml::completion<event::render_runtime>
          / action::update_render_strip_state
      , sml::state<render_finalize_decision> <= sml::state<render_stop_match_exec>
            + sml::completion<event::render_runtime>
          / action::apply_render_stop_matching
      , sml::state<render_publish_success> <= sml::state<render_finalize_decision>
            + sml::completion<event::render_runtime> [ guard::request_ok{} ]
          / action::mark_done
      , sml::state<render_publish_error> <= sml::state<render_finalize_decision>
            + sml::completion<event::render_runtime> [ guard::request_failed{} ]
          / action::ensure_last_error
      , sml::state<render_publish_error> <= sml::state<render_finalize_decision>
            + sml::completion<event::render_runtime>
          / action::ensure_last_error
      , sml::state<render_publish_error> <= sml::state<render_dispatch_decision>
            + sml::completion<event::render_runtime> [ guard::render_dispatch_backend_failure{} ]
          / action::set_backend_error
      , sml::state<render_publish_error> <= sml::state<render_dispatch_decision>
            + sml::completion<event::render_runtime> [ guard::render_dispatch_reported_error{} ]
          / action::set_error_from_detokenizer
      , sml::state<render_publish_error> <= sml::state<render_dispatch_decision>
            + sml::completion<event::render_runtime> [ guard::render_dispatch_lengths_invalid{} ]
          / action::set_invalid_request
      , sml::state<render_publish_error> <= sml::state<render_dispatch_decision>
            + sml::completion<event::render_runtime>
          / action::ensure_last_error
      , sml::state<done> <= sml::state<render_publish_success>
            + sml::completion<event::render_runtime>
          / action::publish_render_done
      , sml::state<errored> <= sml::state<render_publish_error>
            + sml::completion<event::render_runtime>
          / action::publish_render_error

      //------------------------------------------------------------------------------//
      , sml::state<flush_publish_success> <= sml::state<flushing>
            + sml::completion<event::flush_runtime> [ guard::flush_output_fits{} ]
          / action::flush_copy_sequence_buffers
      , sml::state<flush_publish_error> <= sml::state<flushing>
            + sml::completion<event::flush_runtime> [ guard::flush_output_too_large{} ]
          / action::set_invalid_request
      , sml::state<done> <= sml::state<flush_publish_success>
            + sml::completion<event::flush_runtime>
          / action::publish_flush_done
      , sml::state<errored> <= sml::state<flush_publish_error>
            + sml::completion<event::flush_runtime>
          / action::publish_flush_error

      //------------------------------------------------------------------------------//
      , sml::state<unexpected> <= sml::state<uninitialized> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<initializing> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<initialization_decision> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<initialize_publish_success> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<initialize_publish_error> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<initialized> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<rendering> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_dispatch_decision> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_result_decision> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_commit_output_exec> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_strip_decision> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_strip_exec> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_strip_state_exec> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_stop_match_exec> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_finalize_decision> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_publish_success> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_publish_error> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<flushing> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<flush_publish_success> + sml::unexpected_event<sml::_>
            / action::on_unexpected
      , sml::state<unexpected> <= sml::state<flush_publish_error> + sml::unexpected_event<sml::_>
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

  bool process_event(const event::initialize & ev) {
    event::initialize_ctx runtime_ctx{};
    const bool accepted = base_type::process_event(event::initialize_runtime{ev,
                                                                           runtime_ctx});
    return accepted && runtime_ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::render & ev) {
    event::render_ctx runtime_ctx{};
    const bool accepted = base_type::process_event(event::render_runtime{ev,
                                                                        runtime_ctx});
    return accepted && runtime_ctx.err == emel::error::cast(error::none);
  }

  bool process_event(const event::flush & ev) {
    event::flush_ctx runtime_ctx{};
    const bool accepted = base_type::process_event(event::flush_runtime{ev,
                                                                       runtime_ctx});
    return accepted && runtime_ctx.err == emel::error::cast(error::none);
  }

  using base_type::process_event;
  using base_type::visit_current_states;
};

using Renderer = sm;

}  // namespace emel::text::renderer
