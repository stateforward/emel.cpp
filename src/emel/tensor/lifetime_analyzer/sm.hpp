#pragma once

#include <type_traits>

#include "emel/sm.hpp"
#include "emel/tensor/lifetime_analyzer/actions.hpp"
#include "emel/tensor/lifetime_analyzer/events.hpp"
#include "emel/tensor/lifetime_analyzer/guards.hpp"

namespace emel::tensor::lifetime_analyzer {

using Process = boost::sml::back::process<
  event::validate,
  events::validate_done,
  events::validate_error,
  event::collect_ranges,
  events::collect_ranges_done,
  events::collect_ranges_error,
  event::publish,
  events::publish_done,
  events::publish_error,
  events::analyze_done,
  events::analyze_error,
  events::reset_done,
  events::reset_error>;

/**
 * Tensor lifetime analysis orchestration model.
 *
 * Runtime invariants:
 * - Inputs are accepted only through `event::analyze`.
 * - Phase outcomes route through explicit `_done` / `_error` events only.
 * - All state mutation and side effects (writing output arrays) occur in actions.
 * - Completion/error is explicit through `_done` / `_error` events.
 *
 * State purposes:
 * - `idle`: accepts `event::analyze` and `event::reset`.
 * - `validating`: validates payload pointers/counts and output contracts.
 * - `collecting_ranges`: computes first/last-use ranges per tensor id.
 * - `publishing`: writes computed ranges to output buffers.
 * - `done`: successful terminal before `events::analyze_done`.
 * - `failed`: failed terminal before `events::analyze_error`.
 * - `resetting`: clears runtime state.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;

    struct idle {};
    struct validating {};
    struct collecting_ranges {};
    struct publishing {};
    struct done {};
    struct failed {};
    struct resetting {};

    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::analyze> / action::begin_analyze =
          sml::state<validating>,
      sml::state<validating> + sml::on_entry<event::analyze> /
          [](const event::analyze & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate validate{
              .tensors = ev.tensors,
              .tensor_count = ev.tensor_count,
              .first_use_out = ev.first_use_out,
              .last_use_out = ev.last_use_out,
              .ranges_out_count = ev.ranges_out_count,
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
      sml::state<validating> + sml::event<events::validate_done> =
          sml::state<collecting_ranges>,
      sml::state<validating> + sml::event<events::validate_error> =
          sml::state<failed>,

      sml::state<collecting_ranges> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::analyze * request = ev.request;
            event::collect_ranges collect{
              .tensors = request != nullptr ? request->tensors : nullptr,
              .tensor_count = request != nullptr ? request->tensor_count : 0,
              .error_out = &phase_error,
            };
            process(collect);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::collect_ranges_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::collect_ranges_done{
              .request = request,
            });
          },
      sml::state<collecting_ranges> + sml::event<event::collect_ranges> /
          action::run_collect_ranges = sml::state<collecting_ranges>,
      sml::state<collecting_ranges> + sml::event<events::collect_ranges_done> =
          sml::state<publishing>,
      sml::state<collecting_ranges> + sml::event<events::collect_ranges_error> =
          sml::state<failed>,

      sml::state<publishing> + sml::on_entry<events::collect_ranges_done> /
          [](const events::collect_ranges_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::analyze * request = ev.request;
            event::publish publish{
              .first_use_out = request != nullptr ? request->first_use_out : nullptr,
              .last_use_out = request != nullptr ? request->last_use_out : nullptr,
              .ranges_out_count = request != nullptr ? request->ranges_out_count : 0,
              .error_out = &phase_error,
            };
            process(publish);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::publish_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::publish_done{
              .request = request,
            });
          },
      sml::state<publishing> + sml::event<event::publish> / action::run_publish =
          sml::state<publishing>,
      sml::state<publishing> + sml::event<events::publish_done> =
          sml::state<done>,
      sml::state<publishing> + sml::event<events::publish_error> =
          sml::state<failed>,

      sml::state<done> + sml::on_entry<events::publish_done> /
          [](const events::publish_done & ev, action::context &, process_t & process) noexcept {
            process(events::analyze_done{
              .request = ev.request,
            });
          },
      sml::state<done> + sml::event<events::analyze_done> / action::on_analyze_done =
          sml::state<idle>,
      sml::state<done> + sml::event<events::analyze_error> / action::on_analyze_error =
          sml::state<idle>,

      sml::state<idle> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<validating> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<collecting_ranges> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<publishing> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<done> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<failed> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<resetting> + sml::on_entry<event::reset> /
          [](const event::reset & ev, action::context &, process_t & process) noexcept {
            process(events::reset_done{
              .request = &ev,
            });
          },
      sml::state<resetting> + sml::event<events::reset_done> / action::on_reset_done =
          sml::state<idle>,
      sml::state<resetting> + sml::event<events::reset_error> / action::on_reset_error =
          sml::state<failed>,

      sml::state<failed> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            int32_t err = EMEL_ERR_INVALID_ARGUMENT;
            const event::analyze * request = nullptr;
            if constexpr (requires { ev.err; }) {
              err = ev.err;
            }
            if constexpr (requires { ev.request; }) {
              using request_type = std::decay_t<decltype(ev.request)>;
              if constexpr (std::is_same_v<request_type, const event::analyze *>) {
                request = ev.request;
              }
            }
            process(events::analyze_error{
              .err = err,
              .request = request,
            });
          },
      sml::state<failed> + sml::event<events::analyze_error> / action::on_analyze_error =
          sml::state<idle>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

  int32_t analyzed_tensor_count() const noexcept { return context_.tensor_count; }

 private:
  action::context context_{};
};

}  // namespace emel::tensor::lifetime_analyzer
