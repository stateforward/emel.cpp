#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/sm.hpp"
#include "emel/decoder/actions.hpp"
#include "emel/decoder/events.hpp"
#include "emel/decoder/guards.hpp"

namespace emel::decoder {

using Process = boost::sml::back::process<
  event::validate,
  events::validate_done,
  events::validate_error,
  event::initialize_batch,
  events::initialize_batch_done,
  events::initialize_batch_error,
  event::update_memory,
  events::update_memory_done,
  events::update_memory_error,
  event::prepare_memory_batch,
  events::prepare_memory_batch_done,
  events::prepare_memory_batch_retryable_error,
  events::prepare_memory_batch_permanent_error,
  event::optimize_memory,
  events::optimize_memory_done,
  events::optimize_memory_error,
  event::reserve_output,
  events::reserve_output_done,
  events::reserve_output_error,
  event::process_ubatch,
  events::ubatch_done,
  events::ubatch_error,
  event::rollback_ubatch,
  events::rollback_done,
  events::rollback_error,
  event::finalize_outputs,
  events::finalize_outputs_done,
  events::finalize_outputs_error,
  events::decoding_done,
  events::decoding_error>;

struct initialized {};
struct validating_request {};
struct initializing_batch {};
struct updating_memory_pre {};
struct preparing_memory_batch_initial {};
struct optimizing_memory {};
struct preparing_memory_batch_retry {};
struct reserving_output {};
struct processing_ubatch {};
struct handling_ubatch_failure {};
struct finalizing_outputs {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::decode> / action::begin_decode =
          sml::state<validating_request>,

      sml::state<validating_request> + sml::on_entry<event::decode> /
          [](const event::decode & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate validate{
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
      sml::state<validating_request> + sml::event<event::validate>
          [guard::valid_token_inputs] / action::run_validate = sml::state<validating_request>,
      sml::state<validating_request> + sml::event<event::validate>
          [guard::invalid_token_inputs] / action::reject_invalid_validate =
            sml::state<validating_request>,
      sml::state<validating_request> + sml::event<events::validate_done> =
          sml::state<initializing_batch>,
      sml::state<validating_request> + sml::event<events::validate_error> = sml::state<errored>,

      sml::state<initializing_batch> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::initialize_batch initialize{
              .error_out = &phase_error,
            };
            process(initialize);
            const event::decode * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::initialize_batch_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::initialize_batch_done{
              .request = request,
            });
          },
      sml::state<initializing_batch> + sml::event<event::initialize_batch> /
          action::run_initialize_batch = sml::state<initializing_batch>,
      sml::state<initializing_batch> + sml::event<events::initialize_batch_done> =
          sml::state<updating_memory_pre>,
      sml::state<initializing_batch> + sml::event<events::initialize_batch_error> =
          sml::state<errored>,

      sml::state<updating_memory_pre> + sml::on_entry<events::initialize_batch_done> /
          [](const events::initialize_batch_done & ev, action::context &,
             process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::update_memory update{
              .error_out = &phase_error,
            };
            process(update);
            const event::decode * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::update_memory_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::update_memory_done{
              .request = request,
            });
          },
      sml::state<updating_memory_pre> + sml::event<event::update_memory> / action::run_update_memory =
          sml::state<updating_memory_pre>,
      sml::state<updating_memory_pre> + sml::event<events::update_memory_done> =
          sml::state<preparing_memory_batch_initial>,
      sml::state<updating_memory_pre> + sml::event<events::update_memory_error> =
          sml::state<errored>,

      sml::state<preparing_memory_batch_initial> + sml::on_entry<events::update_memory_done> /
          [](const events::update_memory_done & ev, action::context &,
             process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            bool retryable = false;
            event::prepare_memory_batch prepare{
              .error_out = &phase_error,
              .retryable_out = &retryable,
            };
            process(prepare);
            const event::decode * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              if (retryable) {
                process(events::prepare_memory_batch_retryable_error{
                  .err = phase_error,
                  .request = request,
                });
                return;
              }
              process(events::prepare_memory_batch_permanent_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::prepare_memory_batch_done{
              .request = request,
            });
          },
      sml::state<preparing_memory_batch_initial> + sml::event<event::prepare_memory_batch> /
          action::run_prepare_memory_batch = sml::state<preparing_memory_batch_initial>,
      sml::state<preparing_memory_batch_initial> + sml::event<events::prepare_memory_batch_done> =
          sml::state<reserving_output>,
      sml::state<preparing_memory_batch_initial> +
          sml::event<events::prepare_memory_batch_retryable_error> = sml::state<optimizing_memory>,
      sml::state<preparing_memory_batch_initial> +
          sml::event<events::prepare_memory_batch_permanent_error> = sml::state<errored>,

      sml::state<optimizing_memory> +
          sml::on_entry<events::prepare_memory_batch_retryable_error> /
          [](const events::prepare_memory_batch_retryable_error & ev, action::context &,
             process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::optimize_memory optimize{
              .error_out = &phase_error,
            };
            process(optimize);
            const event::decode * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::optimize_memory_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::optimize_memory_done{
              .request = request,
            });
          },
      sml::state<optimizing_memory> + sml::event<event::optimize_memory> /
          action::run_optimize_memory = sml::state<optimizing_memory>,
      sml::state<optimizing_memory> + sml::event<events::optimize_memory_done> =
          sml::state<preparing_memory_batch_retry>,
      sml::state<optimizing_memory> + sml::event<events::optimize_memory_error> =
          sml::state<errored>,

      sml::state<preparing_memory_batch_retry> + sml::on_entry<events::optimize_memory_done> /
          [](const events::optimize_memory_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            bool retryable = false;
            event::prepare_memory_batch prepare{
              .error_out = &phase_error,
              .retryable_out = &retryable,
            };
            process(prepare);
            const event::decode * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              if (retryable) {
                process(events::prepare_memory_batch_retryable_error{
                  .err = phase_error,
                  .request = request,
                });
                return;
              }
              process(events::prepare_memory_batch_permanent_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::prepare_memory_batch_done{
              .request = request,
            });
          },
      sml::state<preparing_memory_batch_retry> + sml::event<event::prepare_memory_batch> /
          action::run_prepare_memory_batch = sml::state<preparing_memory_batch_retry>,
      sml::state<preparing_memory_batch_retry> + sml::event<events::prepare_memory_batch_done> =
          sml::state<reserving_output>,
      sml::state<preparing_memory_batch_retry> +
          sml::event<events::prepare_memory_batch_retryable_error> = sml::state<errored>,
      sml::state<preparing_memory_batch_retry> +
          sml::event<events::prepare_memory_batch_permanent_error> = sml::state<errored>,

      sml::state<reserving_output> + sml::on_entry<events::prepare_memory_batch_done> /
          [](const events::prepare_memory_batch_done & ev, action::context &,
             process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::reserve_output reserve{
              .error_out = &phase_error,
            };
            process(reserve);
            const event::decode * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::reserve_output_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::reserve_output_done{
              .request = request,
            });
          },
      sml::state<reserving_output> + sml::event<event::reserve_output>
          [guard::valid_outputs_total] / action::run_reserve_output = sml::state<reserving_output>,
      sml::state<reserving_output> + sml::event<event::reserve_output>
          [guard::invalid_outputs_total] / action::reject_invalid_reserve_output =
            sml::state<reserving_output>,
      sml::state<reserving_output> + sml::event<events::reserve_output_done> =
          sml::state<processing_ubatch>,
      sml::state<reserving_output> + sml::event<events::reserve_output_error> =
          sml::state<errored>,

      sml::state<processing_ubatch> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            const event::decode * request = nullptr;
            if constexpr (requires { ev.request; }) {
              request = ev.request;
            }
            int32_t phase_error = EMEL_OK;
            bool rollback_needed = false;
            event::process_ubatch process_event{
              .error_out = &phase_error,
              .rollback_needed_out = &rollback_needed,
            };
            process(process_event);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::ubatch_error{
                .err = phase_error,
                .rollback_needed = rollback_needed,
                .request = request,
              });
              return;
            }
            process(events::ubatch_done{
              .request = request,
            });
          },
      sml::state<processing_ubatch> + sml::event<event::process_ubatch>
          [guard::can_process_ubatch] / action::run_process_ubatch =
          sml::state<processing_ubatch>,
      sml::state<processing_ubatch> + sml::event<event::process_ubatch>
          [guard::cannot_process_ubatch] / action::on_invalid_ubatch_size =
          sml::state<processing_ubatch>,
      sml::state<processing_ubatch> + sml::event<events::ubatch_done>[guard::has_more_ubatches] =
          sml::state<processing_ubatch>,
      sml::state<processing_ubatch> + sml::event<events::ubatch_done>[guard::no_more_ubatches] =
          sml::state<finalizing_outputs>,
      sml::state<processing_ubatch> + sml::event<events::ubatch_error> =
          sml::state<handling_ubatch_failure>,

      sml::state<handling_ubatch_failure> + sml::on_entry<events::ubatch_error> /
          [](const events::ubatch_error & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::rollback_ubatch rollback{
              .error_out = &phase_error,
              .rollback_needed = ev.rollback_needed,
            };
            process(rollback);
            const event::decode * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error != EMEL_OK ? phase_error : ev.err;
            }
            if (phase_error != EMEL_OK) {
              process(events::rollback_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::rollback_done{
              .err = ev.err,
              .request = request,
            });
          },
      sml::state<handling_ubatch_failure> + sml::event<event::rollback_ubatch> /
          action::run_rollback_ubatch = sml::state<handling_ubatch_failure>,
      sml::state<handling_ubatch_failure> + sml::event<events::rollback_done> =
          sml::state<errored>,
      sml::state<handling_ubatch_failure> + sml::event<events::rollback_error> =
          sml::state<errored>,

      sml::state<finalizing_outputs> + sml::on_entry<events::ubatch_done> /
          [](const events::ubatch_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::finalize_outputs finalize{
              .error_out = &phase_error,
            };
            process(finalize);
            const event::decode * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::finalize_outputs_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::finalize_outputs_done{
              .request = request,
            });
          },
      sml::state<finalizing_outputs> + sml::event<event::finalize_outputs> /
          action::run_finalize_outputs = sml::state<finalizing_outputs>,
      sml::state<finalizing_outputs> + sml::event<events::finalize_outputs_done> =
          sml::state<done>,
      sml::state<finalizing_outputs> + sml::event<events::finalize_outputs_error> =
          sml::state<errored>,

      sml::state<done> + sml::on_entry<events::finalize_outputs_done> /
          [](const events::finalize_outputs_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::decode * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = EMEL_OK;
            }
            process(events::decoding_done{
              .outputs = ctx.outputs_total,
              .error_out = request != nullptr ? request->error_out : nullptr,
              .owner_sm = request != nullptr ? request->owner_sm : nullptr,
              .dispatch_event = request != nullptr ? request->dispatch_event : nullptr,
              .request = request,
            });
          },
      sml::state<done> + sml::event<events::decoding_done> / action::dispatch_decoding_done_to_owner =
          sml::state<initialized>,

      sml::state<errored> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            int32_t err = EMEL_ERR_BACKEND;
            const event::decode * request = nullptr;
            if constexpr (requires { ev.err; }) {
              err = ev.err;
            }
            if constexpr (requires { ev.request; }) {
              request = ev.request;
            }
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = err;
            }
            process(events::decoding_error{
              .err = err,
              .error_out = request != nullptr ? request->error_out : nullptr,
              .owner_sm = request != nullptr ? request->owner_sm : nullptr,
              .dispatch_event = request != nullptr ? request->dispatch_event : nullptr,
              .request = request,
            });
          },
      sml::state<errored> + sml::event<events::decoding_error> /
          action::dispatch_decoding_error_to_owner = sml::state<initialized>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

  int32_t outputs_processed() const noexcept { return context_.outputs_processed; }

 private:
  action::context context_{};
};

}  // namespace emel::decoder
