#pragma once

#include <span>
#include <string_view>

#include "emel/model/needle/request/actions.hpp"
#include "emel/model/needle/request/context.hpp"
#include "emel/model/needle/request/errors.hpp"
#include "emel/model/needle/request/events.hpp"
#include "emel/model/needle/request/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::needle::request {

struct state_unconfigured {};
struct state_configure_decision {};
struct state_configure_outcome {};
struct state_ready {};
struct state_reset_decision {};
struct state_reset_outcome {};
struct state_reset_ready {};
struct state_complete_decision {};
struct state_rendering {};
struct state_tokenizing {};
struct state_prefilling {};
struct state_decoding {};
struct state_detokenizing {};
struct state_normalizing {};
struct state_complete_outcome {};
struct state_errored {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Persistent constructor configuration.
        sml::state<state_configure_decision> <= *sml::state<state_unconfigured>
          + sml::event<event::configure_run> / action::effect_begin_configure{}
      , sml::state<state_configure_decision> <= sml::state<state_ready>
          + sml::event<event::configure_run> / action::effect_begin_configure{}
      , sml::state<state_configure_decision> <= sml::state<state_reset_ready>
          + sml::event<event::configure_run> / action::effect_begin_configure{}
      , sml::state<state_configure_decision> <= sml::state<state_errored>
          + sml::event<event::configure_run> / action::effect_begin_configure{}
      , sml::state<state_configure_outcome> <= sml::state<state_configure_decision>
          + sml::completion<event::configure_run> [ guard::guard_configure_valid{} ]
          / action::effect_initialize_assets{}
      , sml::state<state_errored> <= sml::state<state_configure_decision>
          + sml::completion<event::configure_run> [ guard::guard_configure_invalid{} ]
          / action::effect_mark_invalid{}
      , sml::state<state_ready> <= sml::state<state_configure_outcome>
          + sml::completion<event::configure_run> [ guard::guard_error_none{} ]
          / action::effect_store_configuration{}
      , sml::state<state_errored> <= sml::state<state_configure_outcome>
          + sml::completion<event::configure_run> [ guard::guard_error_present{} ]

      //------------------------------------------------------------------------------//
      // Explicit reset boundary (excluded from comparable external wall time).
      , sml::state<state_reset_decision> <= sml::state<state_ready>
          + sml::event<event::reset_run> / action::effect_begin_reset{}
      , sml::state<state_reset_decision> <= sml::state<state_reset_ready>
          + sml::event<event::reset_run> / action::effect_begin_reset{}
      , sml::state<state_reset_decision> <= sml::state<state_errored>
          + sml::event<event::reset_run> / action::effect_begin_reset{}
      , sml::state<state_reset_outcome> <= sml::state<state_reset_decision>
          + sml::completion<event::reset_run> [ guard::guard_reset_valid{} ]
          / action::effect_exec_reset{}
      , sml::state<state_errored> <= sml::state<state_reset_decision>
          + sml::completion<event::reset_run> [ guard::guard_reset_invalid{} ]
          / action::effect_mark_invalid{}
      , sml::state<state_reset_ready> <= sml::state<state_reset_outcome>
          + sml::completion<event::reset_run> [ guard::guard_error_none{} ]
      , sml::state<state_errored> <= sml::state<state_reset_outcome>
          + sml::completion<event::reset_run> [ guard::guard_error_present{} ]

      //------------------------------------------------------------------------------//
      // Raw query request: render, tokenize, BOS, graph, detokenize, normalize.
      , sml::state<state_complete_decision> <= sml::state<state_reset_ready>
          + sml::event<event::complete_run> / action::effect_begin_complete{}
      , sml::state<state_rendering> <= sml::state<state_complete_decision>
          + sml::completion<event::complete_run> [ guard::guard_complete_valid{} ]
          / action::effect_render_prompt{}
      , sml::state<state_errored> <= sml::state<state_complete_decision>
          + sml::completion<event::complete_run> [ guard::guard_complete_invalid{} ]
          / action::effect_mark_invalid{}
      , sml::state<state_tokenizing> <= sml::state<state_rendering>
          + sml::completion<event::complete_run> [ guard::guard_error_none{} ]
          / action::effect_tokenize_prompt{}
      , sml::state<state_errored> <= sml::state<state_rendering>
          + sml::completion<event::complete_run> [ guard::guard_error_present{} ]
      , sml::state<state_prefilling> <= sml::state<state_tokenizing>
          + sml::completion<event::complete_run> [ guard::guard_error_none{} ]
          / action::effect_prefill{}
      , sml::state<state_errored> <= sml::state<state_tokenizing>
          + sml::completion<event::complete_run> [ guard::guard_error_present{} ]
      , sml::state<state_decoding> <= sml::state<state_prefilling>
          + sml::completion<event::complete_run> [ guard::guard_error_none{} ]
      , sml::state<state_errored> <= sml::state<state_prefilling>
          + sml::completion<event::complete_run> [ guard::guard_error_present{} ]
      , sml::state<state_decoding> <= sml::state<state_decoding>
          + sml::completion<event::complete_run> [ guard::guard_generation_continues{} ]
          / action::effect_decode_token{}
      , sml::state<state_detokenizing> <= sml::state<state_decoding>
          + sml::completion<event::complete_run> [ guard::guard_generation_stops{} ]
          / action::effect_finish_generation{}
      , sml::state<state_errored> <= sml::state<state_decoding>
          + sml::completion<event::complete_run> [ guard::guard_error_present{} ]
      , sml::state<state_normalizing> <= sml::state<state_detokenizing>
          + sml::completion<event::complete_run> [ guard::guard_error_none{} ]
          / action::effect_detokenize_generation{}
      , sml::state<state_errored> <= sml::state<state_detokenizing>
          + sml::completion<event::complete_run> [ guard::guard_error_present{} ]
      , sml::state<state_complete_outcome> <= sml::state<state_normalizing>
          + sml::completion<event::complete_run> [ guard::guard_error_none{} ]
          / action::effect_normalize_response{}
      , sml::state<state_errored> <= sml::state<state_normalizing>
          + sml::completion<event::complete_run> [ guard::guard_error_present{} ]
      , sml::state<state_ready> <= sml::state<state_complete_outcome>
          + sml::completion<event::complete_run> [ guard::guard_error_none{} ]
      , sml::state<state_errored> <= sml::state<state_complete_outcome>
          + sml::completion<event::complete_run> [ guard::guard_error_present{} ]

      //------------------------------------------------------------------------------//
      // Unexpected public events are explicit failures.
      , sml::state<state_errored> <= sml::state<state_unconfigured> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
      , sml::state<state_errored> <= sml::state<state_ready> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
      , sml::state<state_errored> <= sml::state<state_reset_ready> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
      , sml::state<state_errored> <= sml::state<state_errored> + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;

  explicit sm(const needle::contract &bound,
              action::timestamp_now_fn timestamp_now =
                  &action::steady_timestamp_now)
      : base_type(std::in_place,
                  action::dependencies{.bound = bound,
                                       .timestamp_now = timestamp_now}) {}

  sm(const sm &) = delete;
  sm &operator=(const sm &) = delete;

  bool process_event(const event::configure &ev);
  bool process_event(const event::reset &ev);
  bool process_event(const event::complete &ev);
  bool prepare(const event::prepare &ev) noexcept {
    event::prepare_ctx ctx{};
    event::prepare_run runtime{ev, ctx};
    if (!this->context_.configured || ev.query.empty() ||
        ev.query.size() > action::k_max_query_bytes ||
        ev.max_new_tokens == 0u ||
        ev.max_new_tokens >= this->context_.generated_ids.size())
      return false;
    action::effect_render_prompt{}(runtime, this->context_);
    if (ctx.err != emel::error::cast(error::none)) return false;
    action::effect_tokenize_prompt{}(runtime, this->context_);
    return ctx.err == emel::error::cast(error::none);
  }
  std::string_view rendered_prompt() const noexcept {
    return {this->context_.prompt_storage.data(), this->context_.prompt_size};
  }
  std::span<const int32_t> prompt_token_ids() const noexcept {
    return {this->context_.prompt_ids.data(), this->context_.prompt_id_count};
  }

  using base_type::is;
  using base_type::visit_current_states;

  std::string_view normalized_envelope() const noexcept {
    return {this->context_.normalized_envelope.data(),
            this->context_.normalized_envelope_size};
  }
  std::span<const int32_t> generated_token_ids() const noexcept {
    return {this->context_.generated_ids.data(),
            this->context_.generated_id_count};
  }
  uint32_t prompt_tokens() const noexcept {
    return static_cast<uint32_t>(this->context_.prompt_id_count);
  }
  uint32_t generated_tokens() const noexcept {
    return static_cast<uint32_t>(this->context_.generated_id_count);
  }
  uint64_t prefill_nanoseconds() const noexcept {
    return this->context_.prefill_nanoseconds;
  }
  uint64_t decode_nanoseconds() const noexcept {
    return this->context_.decode_nanoseconds;
  }
};

using NeedleRequest = sm;

} // namespace emel::model::needle::request
