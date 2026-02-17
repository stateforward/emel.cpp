#pragma once

#include <type_traits>

#include "emel/buffer/realloc_analyzer/actions.hpp"
#include "emel/buffer/realloc_analyzer/events.hpp"
#include "emel/buffer/realloc_analyzer/guards.hpp"
#include "emel/sm.hpp"

namespace emel::buffer::realloc_analyzer {

using Process = boost::sml::back::process<
  event::validate,
  events::validate_done,
  events::validate_error,
  event::evaluate,
  events::evaluate_done,
  events::evaluate_error,
  event::publish,
  events::publish_done,
  events::publish_error,
  events::analyze_done,
  events::analyze_error,
  events::reset_done,
  events::reset_error>;

/**
 * Buffer realloc analyzer orchestration model.
 *
 * Parity reference:
 * - `ggml_gallocr_node_needs_realloc(...)`
 * - `ggml_gallocr_needs_realloc(...)`
 *
 * Runtime invariants:
 * - Inputs are accepted only through `event::analyze`.
 * - Phase outcomes route through explicit `_done` / `_error` events only.
 * - Side effects are limited to writing output pointers from actions.
 * - No cross-machine mutation; analysis is pure over payload + snapshot.
 *
 * State purposes:
 * - `idle`: accepts `event::analyze` and `event::reset`.
 * - `validating`: validates graph/snapshot payload contracts.
 * - `evaluating`: evaluates whether realloc is required.
 * - `publishing`: publishes `needs_realloc` boundary output.
 * - `done`: successful terminal before `events::analyze_done`.
 * - `failed`: failed terminal before `events::analyze_error`.
 * - `resetting`: clears runtime analysis state.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;

    struct idle {};
    struct validating {};
    struct evaluating {};
    struct publishing {};
    struct done {};
    struct failed {};
    struct resetting {};

    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::analyze> / action::begin_analyze = sml::state<validating>,
      sml::state<validating> + sml::on_entry<event::analyze> /
          [](const event::analyze & ev, action::context &, process_t & process) noexcept {
            process(event::validate{
              .graph = ev.graph,
              .node_allocs = ev.node_allocs,
              .node_alloc_count = ev.node_alloc_count,
              .leaf_allocs = ev.leaf_allocs,
              .leaf_alloc_count = ev.leaf_alloc_count,
              .error_out = ev.error_out,
              .request = &ev,
            });
          },

      sml::state<validating> + sml::event<event::validate> [guard::valid_analyze_request{}] /
          [](const event::validate & ev, action::context & ctx, process_t & process) noexcept {
            if (ev.error_out != nullptr) {
              *ev.error_out = EMEL_OK;
            }
            ctx.step += 1;
            process(events::validate_done{.request = ev.request});
          } = sml::state<validating>,
      sml::state<validating> + sml::event<event::validate> [guard::invalid_analyze_request{}] /
          [](const event::validate & ev, action::context & ctx, process_t & process) noexcept {
            if (ev.error_out != nullptr) {
              *ev.error_out = EMEL_ERR_INVALID_ARGUMENT;
            }
            ctx.step += 1;
            process(events::validate_error{
              .err = EMEL_ERR_INVALID_ARGUMENT,
              .request = ev.request,
            });
          } = sml::state<validating>,
      sml::state<validating> + sml::event<events::validate_done> = sml::state<evaluating>,
      sml::state<validating> + sml::event<events::validate_error> = sml::state<failed>,

      sml::state<evaluating> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context & ctx, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::evaluate evaluate{
              .graph = ev.request != nullptr ? ev.request->graph : event::graph_view{},
              .node_allocs = ev.request != nullptr ? ev.request->node_allocs : nullptr,
              .node_alloc_count = ev.request != nullptr ? ev.request->node_alloc_count : 0,
              .leaf_allocs = ev.request != nullptr ? ev.request->leaf_allocs : nullptr,
              .leaf_alloc_count = ev.request != nullptr ? ev.request->leaf_alloc_count : 0,
              .error_out = &phase_error,
            };
            process(evaluate);
            if (phase_error != EMEL_OK) {
              process(events::evaluate_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::evaluate_done{
              .request = ev.request,
            });
            (void)ctx;
          },
      sml::state<evaluating> + sml::event<event::evaluate> / action::run_evaluate =
          sml::state<evaluating>,
      sml::state<evaluating> + sml::event<events::evaluate_done> = sml::state<publishing>,
      sml::state<evaluating> + sml::event<events::evaluate_error> = sml::state<failed>,

      sml::state<publishing> + sml::on_entry<events::evaluate_done> /
          [](const events::evaluate_done & ev, action::context & ctx, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::publish publish{
              .needs_realloc_out = ev.request != nullptr ? ev.request->needs_realloc_out : nullptr,
              .error_out = &phase_error,
            };
            process(publish);
            if (phase_error != EMEL_OK) {
              process(events::publish_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::publish_done{
              .request = ev.request,
            });
            (void)ctx;
          },
      sml::state<publishing> + sml::event<event::publish> / action::run_publish =
          sml::state<publishing>,
      sml::state<publishing> + sml::event<events::publish_done> = sml::state<done>,
      sml::state<publishing> + sml::event<events::publish_error> = sml::state<failed>,

      sml::state<done> + sml::on_entry<events::publish_done> /
          [](const events::publish_done & ev, action::context & ctx, process_t & process) noexcept {
            const event::analyze * request = ev.request;
            process(events::analyze_done{
              .needs_realloc = ctx.needs_realloc ? 1 : 0,
              .needs_realloc_out = request != nullptr ? request->needs_realloc_out : nullptr,
              .error_out = request != nullptr ? request->error_out : nullptr,
              .request = request,
            });
          },
      sml::state<done> + sml::event<events::analyze_done> / action::on_analyze_done =
          sml::state<idle>,
      sml::state<done> + sml::event<events::analyze_error> / action::on_analyze_error =
          sml::state<idle>,

      sml::state<idle> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<validating> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<evaluating> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<publishing> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<done> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<failed> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<resetting> + sml::on_entry<event::reset> /
          [](const event::reset & ev, action::context &, process_t & process) noexcept {
            process(events::reset_done{
              .error_out = ev.error_out,
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
              .error_out = request != nullptr ? request->error_out : nullptr,
              .request = request,
            });
          },
      sml::state<failed> + sml::event<events::analyze_error> / action::on_analyze_error =
          sml::state<idle>,

      sml::state<idle> + sml::event<sml::_> / action::on_unexpected = sml::state<failed>,
      sml::state<validating> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<evaluating> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<publishing> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<done> + sml::event<sml::_> / action::on_unexpected = sml::state<failed>,
      sml::state<resetting> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>,
      sml::state<failed> + sml::event<sml::_> / action::on_unexpected =
          sml::state<failed>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

  bool needs_realloc() const noexcept { return context_.needs_realloc; }

 private:
  action::context context_{};
};

}  // namespace emel::buffer::realloc_analyzer
