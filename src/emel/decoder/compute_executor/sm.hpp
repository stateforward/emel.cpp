#pragma once

#include <cstdint>

#include "emel/decoder/compute_executor/actions.hpp"
#include "emel/decoder/compute_executor/events.hpp"
#include "emel/decoder/compute_executor/guards.hpp"
#include "emel/emel.h"
#include "emel/sm.hpp"

namespace emel::decoder::compute_executor {

using Process = boost::sml::back::process<
  event::validate,
  events::validate_done,
  events::validate_error,
  event::prepare_graph,
  events::prepare_graph_done,
  events::prepare_graph_error,
  event::alloc_graph,
  events::alloc_graph_done,
  events::alloc_graph_error,
  event::bind_inputs,
  events::bind_inputs_done,
  events::bind_inputs_error,
  event::run_backend,
  events::run_backend_done,
  events::run_backend_error,
  event::extract_outputs,
  events::extract_outputs_done,
  events::extract_outputs_error,
  events::compute_done,
  events::compute_error>;

struct initialized {};
struct validating {};
struct preparing_graph {};
struct allocating_graph {};
struct binding_inputs {};
struct running_backend {};
struct extracting_outputs {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::execute> / action::begin_execute =
          sml::state<validating>,
      sml::state<validating> + sml::on_entry<event::execute> /
          [](const event::execute & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate validate{
              .request = &ev,
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
          sml::state<preparing_graph>,
      sml::state<validating> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<preparing_graph> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            bool reused = false;
            event::prepare_graph prepare{
              .request = ev.request,
              .reused_out = &reused,
              .error_out = &phase_error,
            };
            process(prepare);
            const event::execute * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::prepare_graph_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::prepare_graph_done{
              .request = request,
              .reused = reused,
            });
          },
      sml::state<preparing_graph> + sml::event<event::prepare_graph> /
          action::run_prepare_graph = sml::state<preparing_graph>,
      sml::state<preparing_graph> + sml::event<events::prepare_graph_done>
          [guard::graph_reused] = sml::state<binding_inputs>,
      sml::state<preparing_graph> + sml::event<events::prepare_graph_done>
          [guard::graph_needs_allocation] = sml::state<allocating_graph>,
      sml::state<preparing_graph> + sml::event<events::prepare_graph_error> =
          sml::state<errored>,

      sml::state<allocating_graph> + sml::on_entry<events::prepare_graph_done> /
          [](const events::prepare_graph_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::alloc_graph alloc{
              .request = ev.request,
              .error_out = &phase_error,
            };
            process(alloc);
            const event::execute * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::alloc_graph_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::alloc_graph_done{
              .request = request,
            });
          },
      sml::state<allocating_graph> + sml::event<event::alloc_graph> /
          action::run_alloc_graph = sml::state<allocating_graph>,
      sml::state<allocating_graph> + sml::event<events::alloc_graph_done> =
          sml::state<binding_inputs>,
      sml::state<allocating_graph> + sml::event<events::alloc_graph_error> =
          sml::state<errored>,

      sml::state<binding_inputs> + sml::on_entry<events::alloc_graph_done> /
          [](const events::alloc_graph_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::bind_inputs bind{
              .request = ev.request,
              .error_out = &phase_error,
            };
            process(bind);
            const event::execute * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::bind_inputs_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::bind_inputs_done{
              .request = request,
            });
          },
      sml::state<binding_inputs> + sml::on_entry<events::prepare_graph_done> /
          [](const events::prepare_graph_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::bind_inputs bind{
              .request = ev.request,
              .error_out = &phase_error,
            };
            process(bind);
            const event::execute * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::bind_inputs_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::bind_inputs_done{
              .request = request,
            });
          },
      sml::state<binding_inputs> + sml::event<event::bind_inputs> / action::run_bind_inputs =
          sml::state<binding_inputs>,
      sml::state<binding_inputs> + sml::event<events::bind_inputs_done> = sml::state<running_backend>,
      sml::state<binding_inputs> + sml::event<events::bind_inputs_error> = sml::state<errored>,

      sml::state<running_backend> + sml::on_entry<events::bind_inputs_done> /
          [](const events::bind_inputs_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::run_backend run{
              .request = ev.request,
              .error_out = &phase_error,
            };
            process(run);
            const event::execute * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::run_backend_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::run_backend_done{
              .request = request,
            });
          },
      sml::state<running_backend> + sml::event<event::run_backend> / action::run_backend =
          sml::state<running_backend>,
      sml::state<running_backend> + sml::event<events::run_backend_done> =
          sml::state<extracting_outputs>,
      sml::state<running_backend> + sml::event<events::run_backend_error> = sml::state<errored>,

      sml::state<extracting_outputs> + sml::on_entry<events::run_backend_done> /
          [](const events::run_backend_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::extract_outputs extract{
              .request = ev.request,
              .error_out = &phase_error,
            };
            process(extract);
            const event::execute * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::extract_outputs_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::extract_outputs_done{
              .request = request,
            });
          },
      sml::state<extracting_outputs> + sml::event<event::extract_outputs> /
          action::run_extract_outputs = sml::state<extracting_outputs>,
      sml::state<extracting_outputs> + sml::event<events::extract_outputs_done> = sml::state<done>,
      sml::state<extracting_outputs> + sml::event<events::extract_outputs_error> =
          sml::state<errored>,

      sml::state<done> + sml::on_entry<events::extract_outputs_done> /
          [](const events::extract_outputs_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::execute * request = ev.request;
            if (request != nullptr && request->outputs_produced_out != nullptr) {
              *request->outputs_produced_out = ctx.outputs_produced;
            }
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = EMEL_OK;
            }
            process(events::compute_done{
              .outputs_produced = ctx.outputs_produced,
              .error_out = request != nullptr ? request->error_out : nullptr,
              .request = request,
            });
          },
      sml::state<done> + sml::event<events::compute_done> / action::on_compute_done =
          sml::state<initialized>,
      sml::state<done> + sml::event<events::compute_error> / action::on_compute_error =
          sml::state<initialized>,

      sml::state<errored> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            int32_t err = EMEL_ERR_BACKEND;
            const event::execute * request = nullptr;
            if constexpr (requires { ev.err; }) {
              err = ev.err;
            }
            if constexpr (requires { ev.request; }) {
              request = ev.request;
            }
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = err;
            }
            process(events::compute_error{
              .err = err,
              .error_out = request != nullptr ? request->error_out : nullptr,
              .request = request,
            });
          },
      sml::state<errored> + sml::event<events::compute_error> / action::on_compute_error =
          sml::state<initialized>

      ,
      sml::state<initialized> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<validating> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<preparing_graph> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<allocating_graph> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<binding_inputs> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<running_backend> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<extracting_outputs> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<done> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<errored> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

  int32_t outputs_produced() const noexcept { return context_.outputs_produced; }

 private:
  action::context context_{};
};

}  // namespace emel::decoder::compute_executor
