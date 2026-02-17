#pragma once

#include <cstdint>

#include "emel/emel.h"
#include "emel/kv/cache/actions.hpp"
#include "emel/kv/cache/events.hpp"
#include "emel/sm.hpp"

namespace emel::kv::cache {

using Process = boost::sml::back::process<
  event::validate_prepare,
  event::validate_apply,
  event::validate_rollback,
  events::validate_done,
  events::validate_error,
  event::prepare_slots,
  events::prepare_slots_done,
  events::prepare_slots_error,
  event::apply_step,
  events::apply_done,
  events::apply_error,
  event::rollback_step,
  events::rollback_done,
  events::rollback_error,
  event::publish,
  events::publish_done,
  events::publish_error,
  events::kv_done,
  events::kv_error>;

struct initialized {};
struct preparing {};
struct prepared {};
struct applying_ubatch {};
struct rolling_back {};
struct publishing {};
struct done {};
struct errored {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;
    using process_t = Process;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::prepare> / action::begin_prepare =
          sml::state<preparing>,
      sml::state<prepared> + sml::event<event::prepare> / action::begin_prepare =
          sml::state<preparing>,

      sml::state<preparing> + sml::on_entry<event::prepare> /
          [](const event::prepare & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate_prepare validate{
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
                .request = {.prepare = &ev},
              });
              return;
            }
            process(events::validate_done{
              .request = {.prepare = &ev},
            });
          },
      sml::state<preparing> + sml::event<event::validate_prepare> / action::run_validate_prepare =
          sml::state<preparing>,
      sml::state<preparing> + sml::event<events::validate_done> = sml::state<preparing>,
      sml::state<preparing> + sml::event<events::validate_error> = sml::state<errored>,
      sml::state<preparing> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::prepare_slots prepare_slots{
              .error_out = &phase_error,
            };
            process(prepare_slots);
            const event::prepare * request = ev.request.prepare;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::prepare_slots_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::prepare_slots_done{
              .request = ev.request,
            });
          },
      sml::state<preparing> + sml::event<event::prepare_slots> / action::run_prepare_slots =
          sml::state<preparing>,
      sml::state<preparing> + sml::event<events::prepare_slots_done> = sml::state<publishing>,
      sml::state<preparing> + sml::event<events::prepare_slots_error> = sml::state<errored>,

      sml::state<prepared> + sml::event<event::apply_ubatch> / action::begin_apply =
          sml::state<applying_ubatch>,
      sml::state<applying_ubatch> + sml::on_entry<event::apply_ubatch> /
          [](const event::apply_ubatch & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate_apply validate{
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
                .request = {.apply = &ev},
              });
              return;
            }
            process(events::validate_done{
              .request = {.apply = &ev},
            });
          },
      sml::state<applying_ubatch> + sml::event<event::validate_apply> / action::run_validate_apply =
          sml::state<applying_ubatch>,
      sml::state<applying_ubatch> + sml::event<events::validate_done> = sml::state<applying_ubatch>,
      sml::state<applying_ubatch> + sml::event<events::validate_error> = sml::state<errored>,
      sml::state<applying_ubatch> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::apply_step apply_step{
              .request = ev.request.apply,
              .error_out = &phase_error,
            };
            process(apply_step);
            const event::apply_ubatch * request = ev.request.apply;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::apply_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::apply_done{
              .request = ev.request,
            });
          },
      sml::state<applying_ubatch> + sml::event<event::apply_step> / action::run_apply_step =
          sml::state<applying_ubatch>,
      sml::state<applying_ubatch> + sml::event<events::apply_done> = sml::state<publishing>,
      sml::state<applying_ubatch> + sml::event<events::apply_error> = sml::state<errored>,

      sml::state<prepared> + sml::event<event::rollback> / action::begin_rollback =
          sml::state<rolling_back>,
      sml::state<errored> + sml::event<event::rollback> / action::begin_rollback =
          sml::state<rolling_back>,
      sml::state<rolling_back> + sml::on_entry<event::rollback> /
          [](const event::rollback & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate_rollback validate{
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
                .request = {.rollback = &ev},
              });
              return;
            }
            process(events::validate_done{
              .request = {.rollback = &ev},
            });
          },
      sml::state<rolling_back> + sml::event<event::validate_rollback> / action::run_validate_rollback =
          sml::state<rolling_back>,
      sml::state<rolling_back> + sml::event<events::validate_done> = sml::state<rolling_back>,
      sml::state<rolling_back> + sml::event<events::validate_error> = sml::state<errored>,
      sml::state<rolling_back> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::rollback_step rollback_step{
              .request = ev.request.rollback,
              .error_out = &phase_error,
            };
            process(rollback_step);
            const event::rollback * request = ev.request.rollback;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::rollback_error{
                .err = phase_error,
                .request = ev.request,
              });
              return;
            }
            process(events::rollback_done{
              .request = ev.request,
            });
          },
      sml::state<rolling_back> + sml::event<event::rollback_step> / action::run_rollback_step =
          sml::state<rolling_back>,
      sml::state<rolling_back> + sml::event<events::rollback_done> = sml::state<publishing>,
      sml::state<rolling_back> + sml::event<events::rollback_error> = sml::state<errored>,

      sml::state<publishing> + sml::on_entry<events::prepare_slots_done> /
          [](const events::prepare_slots_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::publish publish{
              .error_out = &phase_error,
            };
            process(publish);
            const event::prepare * request = ev.request.prepare;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
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
          },
      sml::state<publishing> + sml::on_entry<events::apply_done> /
          [](const events::apply_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::publish publish{
              .error_out = &phase_error,
            };
            process(publish);
            const event::apply_ubatch * request = ev.request.apply;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
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
          },
      sml::state<publishing> + sml::on_entry<events::rollback_done> /
          [](const events::rollback_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::publish publish{
              .error_out = &phase_error,
            };
            process(publish);
            const event::rollback * request = ev.request.rollback;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
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
          },
      sml::state<publishing> + sml::event<event::publish> / action::run_publish = sml::state<publishing>,
      sml::state<publishing> + sml::event<events::publish_done> = sml::state<done>,
      sml::state<publishing> + sml::event<events::publish_error> = sml::state<errored>,

      sml::state<done> + sml::on_entry<events::publish_done> /
          [](const events::publish_done & ev, action::context & ctx, process_t & process) noexcept {
            const event::prepare * prepare_req = ev.request.prepare;
            const event::apply_ubatch * apply_req = ev.request.apply;
            const event::rollback * rollback_req = ev.request.rollback;
            if (prepare_req != nullptr) {
              int32_t err = EMEL_OK;
              if (prepare_req->slot_offsets_out != nullptr) {
                if (prepare_req->slot_offsets_capacity < ctx.planned_ubatch_count) {
                  err = EMEL_ERR_INVALID_ARGUMENT;
                } else {
                  for (int32_t i = 0; i < ctx.planned_ubatch_count; ++i) {
                    prepare_req->slot_offsets_out[i] = ctx.slot_offsets[i];
                  }
                }
              }
              if (prepare_req->ubatch_count_out != nullptr) {
                *prepare_req->ubatch_count_out = ctx.planned_ubatch_count;
              }
              if (prepare_req->error_out != nullptr) {
                *prepare_req->error_out = err;
              }
              if (err != EMEL_OK) {
                process(events::kv_error{.err = err});
                return;
              }
            }
            if (apply_req != nullptr) {
              if (apply_req->kv_tokens_out != nullptr) {
                *apply_req->kv_tokens_out = ctx.kv_tokens;
              }
              if (apply_req->error_out != nullptr) {
                *apply_req->error_out = EMEL_OK;
              }
            }
            if (rollback_req != nullptr && rollback_req->error_out != nullptr) {
              *rollback_req->error_out = EMEL_OK;
            }
            process(events::kv_done{});
          },
      sml::state<done> + sml::event<events::kv_done> / action::on_kv_done = sml::state<prepared>,
      sml::state<done> + sml::event<events::kv_error> / action::on_kv_error = sml::state<prepared>,

      sml::state<initialized> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<preparing> + sml::event<sml::_> / action::on_unexpected{} = sml::state<errored>,
      sml::state<prepared> + sml::event<sml::_> / action::on_unexpected{} = sml::state<errored>,
      sml::state<applying_ubatch> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<rolling_back> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<publishing> + sml::event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<done> + sml::event<sml::_> / action::on_unexpected{} = sml::state<errored>,

      sml::state<errored> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            int32_t err = EMEL_ERR_BACKEND;
            if constexpr (requires { ev.err; }) {
              err = ev.err;
            }
            process(events::kv_error{.err = err});
          },
      sml::state<errored> + sml::event<events::kv_error> / action::on_kv_error =
          sml::state<prepared>,
      sml::state<errored> + sml::event<sml::_> / action::on_unexpected{} = sml::state<errored>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;
 private:
  action::context context_{};
};

}  // namespace emel::kv::cache
