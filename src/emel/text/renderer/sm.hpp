#pragma once

/*
design doc: docs/designs/text/renderer.design.md
 ---
 title: text/renderer architecture design
 status: draft
 ---
 
 # text/renderer architecture design
 
 this document defines the text/renderer actor. it acts as a domain-specific output consumer for the
 modality-agnostic `generator`. it receives raw token IDs, translates them to text, and handles
 string-based stopping criteria and streaming formats.
 
 ## role
 - act as an injected dependency (or external consumer) for the `generator`.
 - receive raw `token_id` streams from the generator's sampling phase.
 - own the `text/detokenizer` codec to translate tokens into utf-8 bytes.
 - handle text-domain complexities: utf-8 boundary buffering, whitespace stripping, and string-based
   stop sequence matching.
 
 ## architecture shift: omnimodal decoupling
 to support omnimodal generation (text, audio, vision), the core `generator` must remain completely
 ignorant of what a `token_id` represents. it cannot own text-specific formatting or stop sequences.
 
 the `generator` is passed a `renderer` (e.g., a `text/renderer` or an `audio/renderer`). when the
 generator samples a new token, it passes it to the renderer. if the renderer detects that the newly
 emitted text matches a user-defined string stop sequence (e.g., `"\nUser:"`), the renderer signals
 back to the generator to halt that sequence.
 
 ## events
 - `event::bind`
   - inputs: `vocab`, renderer options (e.g., stop sequences, strip whitespace flags), and optional callbacks.
   - outputs: invokes callback upon successfully binding state ready to render.
 - `event::render` (called by the generator post-sampling)
   - inputs: `token_id`, sequence ID, output buffers (for streaming to the user), and optional callbacks (`dispatch_done`, `dispatch_error`).
   - outputs: populates caller-provided buffers with translated utf-8 bytes (if any) and a `sequence_status` flag,
     invoking the appropriate callback before returning to prevent context reading.
 - `event::flush`
   - inputs: sequence ID, output buffers, and optional callbacks.
   - outputs: forces the emission of any pending bytes into the output buffers and invokes the callback.
 
 ## state model
 
 ```text
 uninitialized ──► binding ──► idle
                                │
 idle ──► rendering ──► render_decision ──► (idle | errored)
   ▲                                           │
   └───────────────────────────────────────────┘
 ```
 
 - `uninitialized` — awaiting initial setup.
 - `binding` — storing vocab references and compiling stop sequence patterns.
 - `idle` — waiting for a `token_id` from the generator.
 - `rendering` — passing the token to the `text/detokenizer` and buffering the result.
 - `render_decision` — evaluating the newly translated text against the stop sequence list.
 - unexpected events route to `unexpected`.
 
 ## responsibilities
 1. **utf-8 stream management:** modern models use byte-fallback tokens. a single `token_id` might be
    a partial utf-8 character (e.g., `0xE2`). the renderer maintains a tiny pending byte buffer per
    sequence and only flushes valid, complete utf-8 strings to the user's output buffer.
 2. **stop sequence matching:** evaluate the rolling text window against user-provided stop strings.
    if a match occurs, truncate the output and return a `stop_sequence_matched` status to the generator.
 3. **formatting:** handle optional policies like stripping leading spaces from the very first token
    generated.
 
 ## error codes
 
 this actor can produce the following local error flags:
 
 - `error::invalid_request` — invalid bind/render/flush request payload.
 - `error::backend_error` — downstream detokenizer dispatch failed without an explicit code.
 - `error::model_invalid` — downstream detokenizer reported model/token validity failure.
 - `error::internal_error` — internal unexpected path.
 - `error::untracked` — downstream returned an unmapped legacy code.
*/


#include <cstdint>
#include <array>

#include "emel/error/error.hpp"
#include "emel/sm.hpp"
#include "emel/text/renderer/actions.hpp"
#include "emel/text/renderer/errors.hpp"
#include "emel/text/renderer/events.hpp"
#include "emel/text/renderer/guards.hpp"

namespace emel::text::renderer {

struct uninitialized {};
struct binding {};
struct binding_decision {};
struct idle {};
struct rendering {};
struct render_sequence_decision {};
struct render_dispatch_decision {};
struct render_strip_decision {};
struct render_apply_decision {};
struct render_stop_decision {};
struct render_decision {};
struct flushing {};
struct flush_decision {};
struct done {};
struct errored {};
struct unexpected {};

/*
renderer architecture notes (single source of truth)

state purpose
- uninitialized: awaiting dependency and vocab binding.
- binding/binding_decision: bind request acceptance and detokenizer bind outcome.
- idle: ready for render and flush requests.
- rendering/render_*: render request setup, per-sequence routing, detokenizer dispatch, strip/stop phases.
- flushing/flush_decision: emits buffered bytes (utf-8 pending + stop holdback).
- done/errored: terminal outcomes for the latest request.
- unexpected: sequencing contract violation.

key invariants
- per-sequence utf-8 pending bytes and stop holdback are stored in renderer context.
- detokenizer stays stateless and receives all pending state via event payloads.
- output bytes are written only to caller-provided buffers.

guard semantics
- valid_bind: dependency pointers and stop sequence constraints are valid.
- valid_render/valid_flush: output pointers and sequence id are valid.
- request_ok/request_failed: branch on runtime action outcomes.
- bind/render/flush decision guards model all runtime routing branches.

action side effects
- begin_bind/bind_detokenizer: configure dependencies and dispatch child bind.
- begin_render/dispatch_render_detokenizer: initialize render runtime ctx and dispatch child detokenize.
- strip/apply actions: apply leading-strip policy and stop sequence matching.
- begin_flush/flush_copy_sequence_buffers: emit pending and holdback bytes.
- mark_done/ensure_last_error: finalize terminal request result.
*/
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<binding> <= *sml::state<uninitialized>
          + sml::event<event::bind_runtime>[ guard::valid_bind{} ]
          / action::begin_bind
      , sml::state<errored> <= sml::state<uninitialized>
          + sml::event<event::bind_runtime>[ guard::invalid_bind{} ]
          / action::reject_bind
      , sml::state<errored> <= sml::state<uninitialized> + sml::event<event::render_runtime>
          / action::reject_render
      , sml::state<errored> <= sml::state<uninitialized> + sml::event<event::flush_runtime>
          / action::reject_flush

      , sml::state<binding> <= sml::state<idle>
          + sml::event<event::bind_runtime>[ guard::valid_bind{} ]
          / action::begin_bind
      , sml::state<errored> <= sml::state<idle>
          + sml::event<event::bind_runtime>[ guard::invalid_bind{} ]
          / action::reject_bind
      , sml::state<rendering> <= sml::state<idle>
          + sml::event<event::render_runtime>[ guard::valid_render{} ]
          / action::begin_render
      , sml::state<errored> <= sml::state<idle>
          + sml::event<event::render_runtime>[ guard::invalid_render{} ]
          / action::reject_render
      , sml::state<flushing> <= sml::state<idle>
          + sml::event<event::flush_runtime>[ guard::valid_flush{} ]
          / action::begin_flush
      , sml::state<errored> <= sml::state<idle>
          + sml::event<event::flush_runtime>[ guard::invalid_flush{} ]
          / action::reject_flush

      , sml::state<binding> <= sml::state<done>
          + sml::event<event::bind_runtime>[ guard::valid_bind{} ]
          / action::begin_bind
      , sml::state<errored> <= sml::state<done>
          + sml::event<event::bind_runtime>[ guard::invalid_bind{} ]
          / action::reject_bind
      , sml::state<rendering> <= sml::state<done>
          + sml::event<event::render_runtime>[ guard::valid_render{} ]
          / action::begin_render
      , sml::state<errored> <= sml::state<done>
          + sml::event<event::render_runtime>[ guard::invalid_render{} ]
          / action::reject_render
      , sml::state<flushing> <= sml::state<done>
          + sml::event<event::flush_runtime>[ guard::valid_flush{} ]
          / action::begin_flush
      , sml::state<errored> <= sml::state<done>
          + sml::event<event::flush_runtime>[ guard::invalid_flush{} ]
          / action::reject_flush

      , sml::state<binding> <= sml::state<errored>
          + sml::event<event::bind_runtime>[ guard::valid_bind{} ]
          / action::begin_bind
      , sml::state<errored> <= sml::state<errored>
          + sml::event<event::bind_runtime>[ guard::invalid_bind{} ]
          / action::reject_bind
      , sml::state<rendering> <= sml::state<errored>
          + sml::event<event::render_runtime>[ guard::valid_render{} ]
          / action::begin_render
      , sml::state<errored> <= sml::state<errored>
          + sml::event<event::render_runtime>[ guard::invalid_render{} ]
          / action::reject_render
      , sml::state<flushing> <= sml::state<errored>
          + sml::event<event::flush_runtime>[ guard::valid_flush{} ]
          / action::begin_flush
      , sml::state<errored> <= sml::state<errored>
          + sml::event<event::flush_runtime>[ guard::invalid_flush{} ]
          / action::reject_flush

      , sml::state<binding> <= sml::state<unexpected>
          + sml::event<event::bind_runtime>[ guard::valid_bind{} ]
          / action::begin_bind
      , sml::state<unexpected> <= sml::state<unexpected>
          + sml::event<event::bind_runtime>[ guard::invalid_bind{} ]
          / action::reject_bind
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
      , sml::state<binding_decision> <= sml::state<binding>
          + sml::completion<event::bind_runtime> / action::bind_detokenizer
      , sml::state<idle> <= sml::state<binding_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_dispatch_ok{} ]
          / action::commit_bind_success
      , sml::state<errored> <= sml::state<binding_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_dispatch_backend_failure{} ]
          / action::set_backend_error
      , sml::state<errored> <= sml::state<binding_decision>
          + sml::completion<event::bind_runtime> [ guard::bind_dispatch_reported_error{} ]
          / action::set_error_from_detokenizer
      , sml::state<errored> <= sml::state<binding_decision>
          + sml::completion<event::bind_runtime>
          / action::ensure_last_error

      , sml::state<render_sequence_decision> <= sml::state<rendering>
          + sml::completion<event::render_runtime>
      , sml::state<done> <= sml::state<render_sequence_decision>
          + sml::completion<event::render_runtime> [ guard::sequence_stop_matched{} ]
          / action::render_sequence_already_stopped
      , sml::state<render_dispatch_decision> <= sml::state<render_sequence_decision>
          + sml::completion<event::render_runtime> [ guard::sequence_running{} ]
          / action::dispatch_render_detokenizer
      , sml::state<errored> <= sml::state<render_dispatch_decision>
          + sml::completion<event::render_runtime> [ guard::render_dispatch_backend_failure{} ]
          / action::set_backend_error
      , sml::state<errored> <= sml::state<render_dispatch_decision>
          + sml::completion<event::render_runtime> [ guard::render_dispatch_reported_error{} ]
          / action::set_error_from_detokenizer
      , sml::state<errored> <= sml::state<render_dispatch_decision>
          + sml::completion<event::render_runtime> [ guard::render_dispatch_lengths_invalid{} ]
          / action::set_invalid_request
      , sml::state<render_strip_decision> <= sml::state<render_dispatch_decision>
          + sml::completion<event::render_runtime> [ guard::render_dispatch_ok{} ]
          / action::commit_render_detokenizer_output
      , sml::state<render_apply_decision> <= sml::state<render_strip_decision>
          + sml::completion<event::render_runtime> [ guard::strip_needed{} ]
          / action::strip_render_leading_space
      , sml::state<render_apply_decision> <= sml::state<render_strip_decision>
          + sml::completion<event::render_runtime> [ guard::strip_not_needed{} ]
      , sml::state<render_stop_decision> <= sml::state<render_apply_decision>
          + sml::completion<event::render_runtime> / action::update_render_strip_state
      , sml::state<render_decision> <= sml::state<render_stop_decision>
          + sml::completion<event::render_runtime> / action::apply_render_stop_matching
      , sml::state<done> <= sml::state<render_decision>
          + sml::completion<event::render_runtime> [ guard::request_ok{} ]
          / action::mark_done
      , sml::state<errored> <= sml::state<render_decision>
          + sml::completion<event::render_runtime> [ guard::request_failed{} ]
          / action::ensure_last_error

      , sml::state<flush_decision> <= sml::state<flushing>
          + sml::completion<event::flush_runtime>
      , sml::state<done> <= sml::state<flush_decision>
          + sml::completion<event::flush_runtime> [ guard::flush_output_fits{} ]
          / action::flush_copy_sequence_buffers
      , sml::state<errored> <= sml::state<flush_decision>
          + sml::completion<event::flush_runtime> [ guard::flush_output_too_large{} ]
          / action::set_invalid_request

      //------------------------------------------------------------------------------//
      , sml::state<unexpected> <= sml::state<uninitialized> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<binding_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<idle> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<rendering> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_sequence_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_dispatch_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_strip_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_apply_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_stop_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<render_decision> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<flushing> + sml::unexpected_event<sml::_>
          / action::on_unexpected
      , sml::state<unexpected> <= sml::state<flush_decision> + sml::unexpected_event<sml::_>
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

namespace detail {

template <class value_type>
inline void write_optional(value_type * destination,
                           value_type & sink,
                           const value_type value) noexcept {
  value_type * destinations[2] = {&sink, destination};
  value_type * const target =
      destinations[static_cast<size_t>(destination != nullptr)];
  *target = value;
}

template <class event_type>
inline bool ignore_callback(void *, const event_type &) noexcept {
  return true;
}

template <class event_type>
inline void dispatch_optional_callback(
    void * owner,
    bool (*callback)(void * owner, const event_type &),
    const event_type & payload) noexcept {
  const size_t callback_ready = static_cast<size_t>(callback != nullptr);
  const size_t owner_ready = static_cast<size_t>(owner != nullptr);
  const size_t valid = callback_ready & owner_ready;
  bool (*callbacks[2])(void *, const event_type &) = {
      ignore_callback<event_type>,
      callback};
  void * owners[2] = {nullptr, owner};
  callbacks[valid](owners[valid], payload);
}

inline emel::error::type select_error_code(
    const bool ok,
    const emel::error::type runtime_error) noexcept {
  const std::array<emel::error::type, 2> fallback_errors = {
      emel::error::cast(error::backend_error),
      runtime_error};
  const emel::error::type failure_error =
      fallback_errors[static_cast<size_t>(runtime_error != emel::error::cast(error::none))];
  const std::array<emel::error::type, 2> final_errors = {
      failure_error,
      emel::error::cast(error::none)};
  return final_errors[static_cast<size_t>(ok)];
}

inline void dispatch_bind_done(const event::bind & ev,
                               const int32_t,
                               const events::binding_done & done_ev,
                               const events::binding_error &) noexcept {
  dispatch_optional_callback(ev.owner_sm, ev.dispatch_done, done_ev);
}

inline void dispatch_bind_error(const event::bind & ev,
                                const int32_t,
                                const events::binding_done &,
                                const events::binding_error & error_ev) noexcept {
  dispatch_optional_callback(ev.owner_sm, ev.dispatch_error, error_ev);
}

inline void dispatch_render_done(const event::render & ev,
                                 const int32_t,
                                 const events::rendering_done & done_ev,
                                 const events::rendering_error &) noexcept {
  dispatch_optional_callback(ev.owner_sm, ev.dispatch_done, done_ev);
}

inline void dispatch_render_error(const event::render & ev,
                                  const int32_t,
                                  const events::rendering_done &,
                                  const events::rendering_error & error_ev) noexcept {
  dispatch_optional_callback(ev.owner_sm, ev.dispatch_error, error_ev);
}

inline void dispatch_flush_done(const event::flush & ev,
                                const int32_t,
                                const events::flush_done & done_ev,
                                const events::flush_error &) noexcept {
  dispatch_optional_callback(ev.owner_sm, ev.dispatch_done, done_ev);
}

inline void dispatch_flush_error(const event::flush & ev,
                                 const int32_t,
                                 const events::flush_done &,
                                 const events::flush_error & error_ev) noexcept {
  dispatch_optional_callback(ev.owner_sm, ev.dispatch_error, error_ev);
}

template <class request_type, class done_event_type, class error_event_type>
inline void dispatch_result_callback(
    const bool ok,
    const request_type & request,
    const int32_t err,
    const done_event_type & done_ev,
    const error_event_type & error_ev,
    void (*on_done)(
        const request_type &,
        const int32_t,
        const done_event_type &,
        const error_event_type &) noexcept,
    void (*on_error)(
        const request_type &,
        const int32_t,
        const done_event_type &,
        const error_event_type &) noexcept) noexcept {
  using dispatch_fn_type = void (*)(const request_type &,
                                    const int32_t,
                                    const done_event_type &,
                                    const error_event_type &) noexcept;
  const std::array<dispatch_fn_type, 2> dispatchers = {on_error, on_done};
  dispatchers[static_cast<size_t>(ok)](request, err, done_ev, error_ev);
}

}  // namespace detail

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;

  sm() : base_type() {}

  bool process_event(const event::bind & ev) {
    namespace sml = boost::sml;

    event::bind_ctx runtime_ctx{};
    event::bind_runtime runtime_ev{ev, runtime_ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    const bool ok = this->is(sml::state<idle>);
    const emel::error::type err_code = detail::select_error_code(ok, runtime_ctx.err);
    this->last_error_ = err_code;
    const int32_t err = static_cast<int32_t>(err_code);

    int32_t error_sink = 0;
    detail::write_optional(ev.error_out, error_sink, err);

    const events::binding_done done_ev{&ev};
    const events::binding_error error_ev{&ev, err};
    detail::dispatch_result_callback(
        ok,
        ev,
        err,
        done_ev,
        error_ev,
        detail::dispatch_bind_done,
        detail::dispatch_bind_error);

    return accepted && ok;
  }

  bool process_event(const event::render & ev) {
    namespace sml = boost::sml;

    event::render_ctx runtime_ctx{};
    event::render_runtime runtime_ev{ev, runtime_ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    const bool ok = this->is(sml::state<done>);
    const emel::error::type err_code = detail::select_error_code(ok, runtime_ctx.err);
    this->last_error_ = err_code;
    const int32_t err = static_cast<int32_t>(err_code);

    size_t output_length_sink = 0;
    detail::write_optional(ev.output_length_out, output_length_sink, runtime_ctx.output_length);
    sequence_status status_sink = sequence_status::running;
    detail::write_optional(ev.status_out, status_sink, runtime_ctx.status);
    int32_t error_sink = 0;
    detail::write_optional(ev.error_out, error_sink, err);

    const events::rendering_done done_ev{&ev, runtime_ctx.output_length, runtime_ctx.status};
    const events::rendering_error error_ev{&ev, err};
    detail::dispatch_result_callback(
        ok,
        ev,
        err,
        done_ev,
        error_ev,
        detail::dispatch_render_done,
        detail::dispatch_render_error);

    return accepted && ok;
  }

  bool process_event(const event::flush & ev) {
    namespace sml = boost::sml;

    event::flush_ctx runtime_ctx{};
    event::flush_runtime runtime_ev{ev, runtime_ctx};
    const bool accepted = base_type::process_event(runtime_ev);
    const bool ok = this->is(sml::state<done>);
    const emel::error::type err_code = detail::select_error_code(ok, runtime_ctx.err);
    this->last_error_ = err_code;
    const int32_t err = static_cast<int32_t>(err_code);

    size_t output_length_sink = 0;
    detail::write_optional(ev.output_length_out, output_length_sink, runtime_ctx.output_length);
    sequence_status status_sink = sequence_status::running;
    detail::write_optional(ev.status_out, status_sink, runtime_ctx.status);
    int32_t error_sink = 0;
    detail::write_optional(ev.error_out, error_sink, err);

    const events::flush_done done_ev{&ev, runtime_ctx.output_length, runtime_ctx.status};
    const events::flush_error error_ev{&ev, err};
    detail::dispatch_result_callback(
        ok,
        ev,
        err,
        done_ev,
        error_ev,
        detail::dispatch_flush_done,
        detail::dispatch_flush_error);

    return accepted && ok;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return static_cast<int32_t>(this->last_error_); }

 private:
  emel::error::type last_error_ = emel::error::cast(error::none);
};

}  // namespace emel::text::renderer
