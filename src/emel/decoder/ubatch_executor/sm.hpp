#pragma once

#include <cstdint>

#include "emel/decoder/ubatch_executor/actions.hpp"
#include "emel/decoder/ubatch_executor/events.hpp"
#include "emel/decoder/ubatch_executor/guards.hpp"
#include "emel/emel.h"
#include "emel/sm.hpp"

namespace emel::decoder::ubatch_executor {

using Process = boost::sml::back::process<
  event::validate,
  events::validate_done,
  events::validate_error,
  event::prepare_memory,
  events::prepare_memory_done,
  events::prepare_memory_error,
  event::prepare_kv,
  events::prepare_kv_done,
  events::prepare_kv_error,
  event::run_compute,
  events::run_compute_done,
  events::run_compute_error,
  event::extract_outputs,
  events::extract_outputs_done,
  events::extract_outputs_error,
  event::rollback,
  events::rollback_done,
  events::rollback_error,
  events::ubatch_execution_done,
  events::ubatch_execution_error>;

struct initialized {};
struct validating {};
struct preparing_memory {};
struct preparing_kv {};
struct running_compute {};
struct extracting_outputs {};
struct rolling_back {};
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
            if (ev.rollback_attempted_out != nullptr) {
              *ev.rollback_attempted_out = false;
            }
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

      sml::state<validating> + sml::event<event::validate>[guard::valid_execute_request{}] /
          action::run_validate = sml::state<validating>,
      sml::state<validating> + sml::event<event::validate>[guard::invalid_execute_request{}] /
          action::reject_invalid_validate = sml::state<validating>,
      sml::state<validating> + sml::event<events::validate_done> = sml::state<preparing_memory>,
      sml::state<validating> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<preparing_memory> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::execute * request = ev.request;
            event::prepare_memory prepare{
              .memory_coordinator_sm = request != nullptr ? request->memory_coordinator_sm : nullptr,
              .error_out = &phase_error,
            };
            process(prepare);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::prepare_memory_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::prepare_memory_done{
              .request = request,
            });
          },
      sml::state<preparing_memory> +
              sml::event<event::prepare_memory>[guard::valid_prepare_memory_request{}] /
          action::run_prepare_memory = sml::state<preparing_memory>,
      sml::state<preparing_memory> +
              sml::event<event::prepare_memory>[guard::invalid_prepare_memory_request{}] /
          action::reject_invalid_prepare_memory = sml::state<preparing_memory>,
      sml::state<preparing_memory> + sml::event<events::prepare_memory_done> =
          sml::state<preparing_kv>,
      sml::state<preparing_memory> + sml::event<events::prepare_memory_error> =
          sml::state<errored>,

      sml::state<preparing_kv> + sml::on_entry<events::prepare_memory_done> /
          [](const events::prepare_memory_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::execute * request = ev.request;
            event::prepare_kv prepare{
              .kv_cache_sm = request != nullptr ? request->kv_cache_sm : nullptr,
              .error_out = &phase_error,
            };
            process(prepare);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::prepare_kv_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::prepare_kv_done{
              .request = request,
            });
          },
      sml::state<preparing_kv> + sml::event<event::prepare_kv>[guard::valid_prepare_kv_request{}] /
          action::run_prepare_kv = sml::state<preparing_kv>,
      sml::state<preparing_kv> +
              sml::event<event::prepare_kv>[guard::invalid_prepare_kv_request{}] /
          action::reject_invalid_prepare_kv = sml::state<preparing_kv>,
      sml::state<preparing_kv> + sml::event<events::prepare_kv_done> =
          sml::state<running_compute>,
      sml::state<preparing_kv> + sml::event<events::prepare_kv_error> = sml::state<errored>,

      sml::state<running_compute> + sml::on_entry<events::prepare_kv_done> /
          [](const events::prepare_kv_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::execute * request = ev.request;
            event::run_compute compute{
              .kv_cache_sm = request != nullptr ? request->kv_cache_sm : nullptr,
              .request = request,
              .error_out = &phase_error,
            };
            process(compute);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::run_compute_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::run_compute_done{
              .request = request,
            });
          },
      sml::state<running_compute> +
              sml::event<event::run_compute>[guard::valid_run_compute_request{}] /
          action::run_compute = sml::state<running_compute>,
      sml::state<running_compute> +
              sml::event<event::run_compute>[guard::invalid_run_compute_request{}] /
          action::reject_invalid_run_compute = sml::state<running_compute>,
      sml::state<running_compute> + sml::event<events::run_compute_done> =
          sml::state<extracting_outputs>,
      sml::state<running_compute> + sml::event<events::run_compute_error> =
          sml::state<rolling_back>,

      sml::state<extracting_outputs> + sml::on_entry<events::run_compute_done> /
          [](const events::run_compute_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::extract_outputs extract{
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
      sml::state<extracting_outputs> +
              sml::event<event::extract_outputs>[guard::valid_extract_outputs_request{}] /
          action::run_extract_outputs = sml::state<extracting_outputs>,
      sml::state<extracting_outputs> +
              sml::event<event::extract_outputs>[guard::invalid_extract_outputs_request{}] /
          action::reject_invalid_extract_outputs = sml::state<extracting_outputs>,
      sml::state<extracting_outputs> + sml::event<events::extract_outputs_done> =
          sml::state<done>,
      sml::state<extracting_outputs> + sml::event<events::extract_outputs_error> =
          sml::state<rolling_back>,

      sml::state<rolling_back> + sml::on_entry<events::run_compute_error> /
          [](const events::run_compute_error & ev, action::context &, process_t & process) noexcept {
            const event::execute * request = ev.request;
            if (request != nullptr && request->rollback_attempted_out != nullptr) {
              *request->rollback_attempted_out = true;
            }
            int32_t phase_error = EMEL_OK;
            event::rollback rollback{
              .kv_cache_sm = request != nullptr ? request->kv_cache_sm : nullptr,
              .error_out = &phase_error,
            };
            process(rollback);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::rollback_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::rollback_done{
              .request = request,
            });
          },
      sml::state<rolling_back> + sml::on_entry<events::extract_outputs_error> /
          [](const events::extract_outputs_error & ev, action::context &, process_t & process) noexcept {
            const event::execute * request = ev.request;
            if (request != nullptr && request->rollback_attempted_out != nullptr) {
              *request->rollback_attempted_out = true;
            }
            int32_t phase_error = EMEL_OK;
            event::rollback rollback{
              .kv_cache_sm = request != nullptr ? request->kv_cache_sm : nullptr,
              .error_out = &phase_error,
            };
            process(rollback);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::rollback_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::rollback_done{
              .request = request,
            });
          },
      sml::state<rolling_back> + sml::event<event::rollback>[guard::valid_rollback_request{}] /
          action::run_rollback = sml::state<rolling_back>,
      sml::state<rolling_back> + sml::event<event::rollback>[guard::invalid_rollback_request{}] /
          action::reject_invalid_rollback = sml::state<rolling_back>,
      sml::state<rolling_back> + sml::event<events::rollback_done> = sml::state<errored>,
      sml::state<rolling_back> + sml::event<events::rollback_error> = sml::state<errored>,

      sml::state<done> + sml::on_entry<events::extract_outputs_done> /
          [](const events::extract_outputs_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::execute * request = ev.request;
            if (request != nullptr && request->outputs_produced_out != nullptr) {
              *request->outputs_produced_out = ctx.outputs_produced;
            }
            if (request != nullptr && request->kv_tokens_out != nullptr) {
              *request->kv_tokens_out = ctx.kv_tokens;
            }
            if (request != nullptr && request->rollback_attempted_out != nullptr) {
              *request->rollback_attempted_out = false;
            }
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = EMEL_OK;
            }
            process(events::ubatch_execution_done{
              .outputs_produced = ctx.outputs_produced,
              .kv_tokens = ctx.kv_tokens,
              .error_out = request != nullptr ? request->error_out : nullptr,
              .request = request,
            });
          },
      sml::state<done> + sml::event<events::ubatch_execution_done> / action::on_ubatch_execution_done =
          sml::state<initialized>,
      sml::state<done> + sml::event<events::ubatch_execution_error> /
          action::on_ubatch_execution_error = sml::state<initialized>,

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
            if (request != nullptr && request->rollback_attempted_out != nullptr) {
              *request->rollback_attempted_out = true;
            }
            process(events::ubatch_execution_error{
              .err = err,
              .error_out = request != nullptr ? request->error_out : nullptr,
              .request = request,
            });
          },
      sml::state<errored> + sml::event<events::ubatch_execution_error> /
          action::on_ubatch_execution_error = sml::state<initialized>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

  int32_t outputs_produced() const noexcept { return context_.outputs_produced; }
  int32_t kv_tokens() const noexcept { return context_.kv_tokens; }

 private:
  action::context context_{};
};

}  // namespace emel::decoder::ubatch_executor
