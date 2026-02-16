#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sampler/candidate_builder/actions.hpp"
#include "emel/sampler/candidate_builder/events.hpp"
#include "emel/sampler/candidate_builder/guards.hpp"
#include "emel/sm.hpp"

namespace emel::sampler::candidate_builder {

using Process = boost::sml::back::process<
  event::validate,
  events::validate_done,
  events::validate_error,
  event::build_candidates,
  events::build_candidates_done,
  events::build_candidates_error,
  event::normalize_scores,
  events::normalize_scores_done,
  events::normalize_scores_error,
  events::build_done,
  events::build_error>;

// Ready state. Accepts one build request at a time.
struct initialized {};
// Validates build payload and output buffers.
struct validating {};
// Builds candidate id/score arrays from logits.
struct building_candidates {};
// Normalizes candidate scores for downstream sampling.
struct normalizing_scores {};
// Terminal success state.
struct done {};
// Terminal error state.
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::build> / action::begin_build =
          sml::state<validating>,
      sml::state<validating> + sml::on_entry<event::build> /
          [](const event::build & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate validate{
              .logits = ev.logits,
              .vocab_size = ev.vocab_size,
              .candidate_ids_out = ev.candidate_ids_out,
              .candidate_scores_out = ev.candidate_scores_out,
              .candidate_capacity = ev.candidate_capacity,
              .candidate_count_out = ev.candidate_count_out,
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
      sml::state<validating> + sml::event<events::validate_done> = sml::state<building_candidates>,
      sml::state<validating> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<building_candidates> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::build * request = ev.request;
            event::build_candidates build_candidates{
              .logits = request != nullptr ? request->logits : nullptr,
              .vocab_size = request != nullptr ? request->vocab_size : 0,
              .candidate_ids_out = request != nullptr ? request->candidate_ids_out : nullptr,
              .candidate_scores_out = request != nullptr ? request->candidate_scores_out : nullptr,
              .error_out = &phase_error,
            };
            process(build_candidates);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::build_candidates_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::build_candidates_done{
              .request = request,
            });
          },
      sml::state<building_candidates> + sml::event<event::build_candidates> /
          action::run_build_candidates = sml::state<building_candidates>,
      sml::state<building_candidates> + sml::event<events::build_candidates_done> =
          sml::state<normalizing_scores>,
      sml::state<building_candidates> + sml::event<events::build_candidates_error> =
          sml::state<errored>,

      sml::state<normalizing_scores> + sml::on_entry<events::build_candidates_done> /
          [](const events::build_candidates_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::build * request = ev.request;
            event::normalize_scores normalize{
              .candidate_scores_out = request != nullptr ? request->candidate_scores_out : nullptr,
              .error_out = &phase_error,
            };
            process(normalize);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::normalize_scores_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::normalize_scores_done{
              .request = request,
            });
          },
      sml::state<normalizing_scores> + sml::event<event::normalize_scores> /
          action::run_normalize_scores = sml::state<normalizing_scores>,
      sml::state<normalizing_scores> +
          sml::event<events::normalize_scores_done>[guard::has_candidates{}] =
          sml::state<done>,
      sml::state<normalizing_scores> +
          sml::event<events::normalize_scores_done>[guard::no_candidates{}] =
          sml::state<errored>,
      sml::state<normalizing_scores> + sml::event<events::normalize_scores_error> =
          sml::state<errored>,

      sml::state<done> + sml::on_entry<events::normalize_scores_done> /
          [](const events::normalize_scores_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::build * request = ev.request;
            process(events::build_done{
              .candidate_count = ctx.candidate_count,
              .candidate_count_out = request != nullptr ? request->candidate_count_out : nullptr,
              .request = request,
            });
          },
      sml::state<done> + sml::event<events::build_done> / action::on_build_done =
          sml::state<initialized>,
      sml::state<done> + sml::event<events::build_error> / action::on_build_error =
          sml::state<initialized>,

      sml::state<errored> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            int32_t err = EMEL_ERR_BACKEND;
            const event::build * request = nullptr;
            if constexpr (requires { ev.err; }) {
              err = ev.err;
            }
            if constexpr (requires { ev.request; }) {
              request = ev.request;
            }
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = err;
            }
            process(events::build_error{
              .err = err,
              .candidate_count_out = request != nullptr ? request->candidate_count_out : nullptr,
              .request = request,
            });
          },
      sml::state<errored> + sml::event<events::build_error> / action::on_build_error =
          sml::state<initialized>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

  int32_t candidate_count() const noexcept { return context_.candidate_count; }

 private:
  action::context context_{};
};

}  // namespace emel::sampler::candidate_builder
