#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sampler/pipeline/actions.hpp"
#include "emel/sampler/pipeline/events.hpp"
#include "emel/sampler/pipeline/guards.hpp"
#include "emel/sm.hpp"

namespace emel::sampler::pipeline {

using Process = boost::sml::back::process<
  event::prepare_candidates,
  events::prepare_candidates_done,
  events::prepare_candidates_error,
  event::apply_sampling,
  events::apply_sampling_done,
  events::apply_sampling_error,
  event::select_token,
  events::select_token_done,
  events::select_token_error,
  events::sampling_done,
  events::sampling_error>;

// Ready state. Accepts one sampling request at a time.
struct initialized {};
// Validates and materializes candidate buffers from logits.
struct preparing_candidates {};
// Applies sampler function chain in-order.
struct sampling {};
// Selects the final token from candidate set.
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
      *sml::state<initialized> + sml::event<event::sample> / action::begin_sample =
          sml::state<preparing_candidates>,
      sml::state<preparing_candidates> + sml::on_entry<event::sample> /
          [](const event::sample & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::prepare_candidates prepare{
              .logits = ev.logits,
              .vocab_size = ev.vocab_size,
              .candidate_ids = ev.candidate_ids,
              .candidate_scores = ev.candidate_scores,
              .candidate_capacity = ev.candidate_capacity,
              .error_out = &phase_error,
            };
            process(prepare);
            if (ev.error_out != nullptr) {
              *ev.error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::prepare_candidates_error{
                .err = phase_error,
                .request = &ev,
              });
              return;
            }
            process(events::prepare_candidates_done{
              .request = &ev,
            });
          },

      sml::state<preparing_candidates> + sml::event<event::prepare_candidates> /
          action::run_prepare_candidates = sml::state<preparing_candidates>,
      sml::state<preparing_candidates> + sml::event<events::prepare_candidates_done> =
          sml::state<sampling>,
      sml::state<preparing_candidates> + sml::event<events::prepare_candidates_error> =
          sml::state<errored>,

      sml::state<sampling> + sml::on_entry<events::prepare_candidates_done> /
          [](const events::prepare_candidates_done & ev, action::context & ctx,
             process_t & process) noexcept {
            if (!guard::has_more_samplers{}(ev, ctx)) {
              process(events::apply_sampling_done{
                .request = ev.request,
              });
              return;
            }
            int32_t phase_error = EMEL_OK;
            const event::sample * request = ev.request;
            event::apply_sampling apply{
              .candidate_ids = request != nullptr ? request->candidate_ids : nullptr,
              .candidate_scores = request != nullptr ? request->candidate_scores : nullptr,
              .sampler_fns = request != nullptr ? request->sampler_fns : nullptr,
              .sampler_user_data = request != nullptr ? request->sampler_user_data : nullptr,
              .error_out = &phase_error,
            };
            process(apply);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::apply_sampling_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::apply_sampling_done{
              .request = request,
            });
          },
      sml::state<sampling> + sml::on_entry<events::apply_sampling_done> /
          [](const events::apply_sampling_done & ev, action::context & ctx,
             process_t & process) noexcept {
            if (!guard::has_more_samplers{}(ev, ctx)) {
              return;
            }
            int32_t phase_error = EMEL_OK;
            const event::sample * request = ev.request;
            event::apply_sampling apply{
              .candidate_ids = request != nullptr ? request->candidate_ids : nullptr,
              .candidate_scores = request != nullptr ? request->candidate_scores : nullptr,
              .sampler_fns = request != nullptr ? request->sampler_fns : nullptr,
              .sampler_user_data = request != nullptr ? request->sampler_user_data : nullptr,
              .error_out = &phase_error,
            };
            process(apply);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::apply_sampling_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::apply_sampling_done{
              .request = request,
            });
          },
      sml::state<sampling> + sml::event<event::apply_sampling> / action::run_apply_sampling =
          sml::state<sampling>,
      sml::state<sampling> +
          sml::event<events::apply_sampling_done>[guard::has_more_samplers{}] =
          sml::state<sampling>,
      sml::state<sampling> +
          sml::event<events::apply_sampling_done>[guard::no_more_samplers{}] =
          sml::state<selecting_token>,
      sml::state<sampling> + sml::event<events::apply_sampling_error> = sml::state<errored>,

      sml::state<selecting_token> + sml::on_entry<events::apply_sampling_done> /
          [](const events::apply_sampling_done & ev, action::context & ctx,
             process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::sample * request = ev.request;
            event::select_token select{
              .candidate_ids = request != nullptr ? request->candidate_ids : nullptr,
              .candidate_scores = request != nullptr ? request->candidate_scores : nullptr,
              .selected_token_out = request != nullptr ? request->selected_token_out : nullptr,
              .error_out = &phase_error,
            };
            process(select);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::select_token_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::select_token_done{
              .request = request,
            });
            (void)ctx;
          },
      sml::state<selecting_token> + sml::event<event::select_token> / action::run_select_token =
          sml::state<selecting_token>,
      sml::state<selecting_token> + sml::event<events::select_token_done> = sml::state<done>,
      sml::state<selecting_token> + sml::event<events::select_token_error> = sml::state<errored>,

      sml::state<done> + sml::on_entry<events::select_token_done> /
          [](const events::select_token_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::sample * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = EMEL_OK;
            }
            process(events::sampling_done{
              .token_id = ctx.selected_token,
              .request = request,
            });
          },
      sml::state<done> + sml::event<events::sampling_done> / action::on_sampling_done =
          sml::state<initialized>,
      sml::state<done> + sml::event<events::sampling_error> / action::on_sampling_error =
          sml::state<initialized>,

      sml::state<errored> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            int32_t err = EMEL_ERR_BACKEND;
            const event::sample * request = nullptr;
            if constexpr (requires { ev.err; }) {
              err = ev.err;
            }
            if constexpr (requires { ev.request; }) {
              request = ev.request;
            }
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = err;
            }
            process(events::sampling_error{
              .err = err,
              .request = request,
            });
          },
      sml::state<errored> + sml::event<events::sampling_error> / action::on_sampling_error =
          sml::state<initialized>
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

}  // namespace emel::sampler::pipeline
