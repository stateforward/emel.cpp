#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sampler/token_selector/actions.hpp"
#include "emel/sampler/token_selector/events.hpp"
#include "emel/sampler/token_selector/guards.hpp"
#include "emel/sm.hpp"

namespace emel::sampler::token_selector {

using Process = boost::sml::back::process<
  event::validate,
  events::validate_done,
  events::validate_error,
  event::select,
  events::select_done,
  events::select_error,
  events::token_selection_done,
  events::token_selection_error>;

// Ready state. Accepts one selection request at a time.
struct initialized {};
// Validates selection payload and candidate inputs.
struct validating {};
// Executes token selection policy over candidate set.
struct selecting_token {};
// Terminal success state.
struct done {};
// Terminal error state.
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::select_token> / action::begin_select_token =
          sml::state<validating>,
      sml::state<validating> + sml::on_entry<event::select_token> /
          [](const event::select_token & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate validate{
              .candidate_ids = ev.candidate_ids,
              .candidate_scores = ev.candidate_scores,
              .candidate_count = ev.candidate_count,
              .policy = ev.policy,
              .random_01 = ev.random_01,
              .error_out = &phase_error,
            };
            process(validate);
            if (ev.error_out != nullptr) {
              *ev.error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::validate_error{
                .err = phase_error,
                .request = &ev,
              });
              return;
            }
            process(events::validate_done{
              .request = &ev,
            });
          },

      sml::state<validating> + sml::event<event::validate> / action::run_validate =
          sml::state<validating>,
      sml::state<validating> +
          sml::event<events::validate_done>[guard::has_candidates{}] =
          sml::state<selecting_token>,
      sml::state<validating> +
          sml::event<events::validate_done>[guard::no_candidates{}] =
          sml::state<errored>,
      sml::state<validating> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<selecting_token> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::select_token * request = ev.request;
            event::select select{
              .candidate_ids = request != nullptr ? request->candidate_ids : nullptr,
              .candidate_scores = request != nullptr ? request->candidate_scores : nullptr,
              .candidate_count = request != nullptr ? request->candidate_count : 0,
              .policy = request != nullptr ? request->policy : event::selection_policy::argmax,
              .random_01 = request != nullptr ? request->random_01 : 0.0f,
              .selected_token_out = request != nullptr ? request->selected_token_out : nullptr,
              .error_out = &phase_error,
            };
            process(select);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::select_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::select_done{
              .request = request,
            });
          },
      sml::state<selecting_token> + sml::event<event::select> / action::run_select =
          sml::state<selecting_token>,
      sml::state<selecting_token> + sml::event<events::select_done> = sml::state<done>,
      sml::state<selecting_token> + sml::event<events::select_error> = sml::state<errored>,

      sml::state<done> + sml::on_entry<events::select_done> /
          [](const events::select_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::select_token * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = EMEL_OK;
            }
            process(events::token_selection_done{
              .token_id = ctx.selected_token,
              .request = request,
            });
          },
      sml::state<done> + sml::event<events::token_selection_done> /
          action::on_token_selection_done = sml::state<initialized>,
      sml::state<done> + sml::event<events::token_selection_error> /
          action::on_token_selection_error = sml::state<initialized>,

      sml::state<errored> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            int32_t err = EMEL_ERR_BACKEND;
            const event::select_token * request = nullptr;
            if constexpr (requires { ev.err; }) {
              err = ev.err;
            }
            if constexpr (requires { ev.request; }) {
              request = ev.request;
            }
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = err;
            }
            process(events::token_selection_error{
              .err = err,
              .request = request,
            });
          },
      sml::state<errored> + sml::event<events::token_selection_error> /
          action::on_token_selection_error = sml::state<initialized>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

  int32_t selected_token() const noexcept { return context_.selected_token; }

 private:
  action::context context_{};
};

}  // namespace emel::sampler::token_selector
