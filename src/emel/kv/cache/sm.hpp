#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>

#include "emel/emel.h"
#include "emel/kv/cache/actions.hpp"
#include "emel/kv/cache/events.hpp"
#include "emel/kv/cache/guards.hpp"
#include "emel/sm.hpp"

namespace emel::kv::cache {

struct initialized {};
struct prepared {};

struct validating_prepare {};
struct preparing_slots {};
struct preparing_slots_step {};
struct prepare_slots_decision {};

struct applying {};
struct apply_step_validating {};
struct applying_step {};
struct apply_step_decision {};

struct rolling_back {};
struct rollback_step_validating {};
struct rollback_step {};
struct rollback_step_decision {};

struct seq_remove_validating {};
struct seq_remove_step {};
struct seq_remove_step_decision {};

struct seq_copy_validating {};
struct seq_copy_step {};
struct seq_copy_step_decision {};

struct seq_keep_validating {};
struct seq_keep_step {};
struct seq_keep_step_decision {};

struct seq_add_validating {};
struct seq_add_step {};
struct seq_add_step_decision {};

struct seq_div_validating {};
struct seq_div_step {};
struct seq_div_step_decision {};

struct updates_validating {};
struct updates_step {};
struct updates_step_decision {};

struct publishing {};
struct publish_decision {};

struct done {};
struct errored {};

/**
 * KV cache orchestration model.
 *
 * State purposes:
 * - `initialized`/`prepared`: idle states awaiting new requests.
 * - `validating_*`: validate request payloads.
 * - `*_step`: perform bounded cache operations.
 * - `*_decision`: branch on `phase_error`.
 * - `publishing`/`publish_decision`: finalize cache publish phase.
 * - `done`/`errored`: terminal outcomes that return to prepared.
 *
 * Guard semantics:
 * - `valid_*` guards are pure predicates on `(context)`.
 * - `phase_*` guards observe `context.phase_error`.
 *
 * Action side effects:
 * - Actions run bounded steps and set `phase_error`.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::prepare> / action::begin_prepare =
          sml::state<validating_prepare>,
      sml::state<prepared> + sml::event<event::prepare> / action::begin_prepare =
          sml::state<validating_prepare>,

      sml::state<validating_prepare> [guard::valid_prepare_context{}] =
          sml::state<preparing_slots>,
      sml::state<validating_prepare> [guard::invalid_prepare_context{}] /
          action::set_invalid_argument = sml::state<errored>,

      sml::state<preparing_slots> [guard::valid_prepare_slots_context{}] =
          sml::state<preparing_slots_step>,
      sml::state<preparing_slots> [guard::invalid_prepare_slots_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<preparing_slots_step> / action::run_prepare_slots_phase =
          sml::state<prepare_slots_decision>,
      sml::state<prepare_slots_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<prepare_slots_decision> [guard::phase_ok{}] =
          sml::state<publishing>,

      sml::state<prepared> + sml::event<event::apply_ubatch> / action::begin_apply =
          sml::state<applying>,
      sml::state<applying> [guard::valid_apply_context{}] =
          sml::state<apply_step_validating>,
      sml::state<applying> [guard::invalid_apply_context{}] / action::set_invalid_argument =
          sml::state<errored>,
      sml::state<apply_step_validating> [guard::valid_apply_step_context{}] =
          sml::state<applying_step>,
      sml::state<apply_step_validating> [guard::invalid_apply_step_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<applying_step> / action::run_apply_step_phase =
          sml::state<apply_step_decision>,
      sml::state<apply_step_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<apply_step_decision> [guard::phase_ok{}] =
          sml::state<publishing>,

      sml::state<prepared> + sml::event<event::rollback> / action::begin_rollback =
          sml::state<rolling_back>,
      sml::state<errored> + sml::event<event::rollback> / action::begin_rollback =
          sml::state<rolling_back>,
      sml::state<rolling_back> [guard::valid_rollback_context{}] =
          sml::state<rollback_step_validating>,
      sml::state<rolling_back> [guard::invalid_rollback_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<rollback_step_validating> [guard::valid_rollback_step_context{}] =
          sml::state<rollback_step>,
      sml::state<rollback_step_validating> [guard::invalid_rollback_step_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<rollback_step> / action::run_rollback_step_phase =
          sml::state<rollback_step_decision>,
      sml::state<rollback_step_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<rollback_step_decision> [guard::phase_ok{}] =
          sml::state<publishing>,

      sml::state<prepared> + sml::event<event::seq_remove> / action::begin_seq_remove =
          sml::state<seq_remove_validating>,
      sml::state<seq_remove_validating> [guard::valid_seq_remove_context{}] =
          sml::state<seq_remove_step>,
      sml::state<seq_remove_validating> [guard::invalid_seq_remove_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<seq_remove_step> / action::run_seq_remove_phase =
          sml::state<seq_remove_step_decision>,
      sml::state<seq_remove_step_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<seq_remove_step_decision> [guard::phase_ok{}] =
          sml::state<done>,

      sml::state<prepared> + sml::event<event::seq_copy> / action::begin_seq_copy =
          sml::state<seq_copy_validating>,
      sml::state<seq_copy_validating> [guard::valid_seq_copy_context{}] =
          sml::state<seq_copy_step>,
      sml::state<seq_copy_validating> [guard::invalid_seq_copy_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<seq_copy_step> / action::run_seq_copy_phase =
          sml::state<seq_copy_step_decision>,
      sml::state<seq_copy_step_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<seq_copy_step_decision> [guard::phase_ok{}] =
          sml::state<done>,

      sml::state<prepared> + sml::event<event::seq_keep> / action::begin_seq_keep =
          sml::state<seq_keep_validating>,
      sml::state<seq_keep_validating> [guard::valid_seq_keep_context{}] =
          sml::state<seq_keep_step>,
      sml::state<seq_keep_validating> [guard::invalid_seq_keep_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<seq_keep_step> / action::run_seq_keep_phase =
          sml::state<seq_keep_step_decision>,
      sml::state<seq_keep_step_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<seq_keep_step_decision> [guard::phase_ok{}] =
          sml::state<done>,

      sml::state<prepared> + sml::event<event::seq_add> / action::begin_seq_add =
          sml::state<seq_add_validating>,
      sml::state<seq_add_validating> [guard::valid_seq_add_context{}] =
          sml::state<seq_add_step>,
      sml::state<seq_add_validating> [guard::invalid_seq_add_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<seq_add_step> / action::run_seq_add_phase =
          sml::state<seq_add_step_decision>,
      sml::state<seq_add_step_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<seq_add_step_decision> [guard::phase_ok{}] =
          sml::state<done>,

      sml::state<prepared> + sml::event<event::seq_div> / action::begin_seq_div =
          sml::state<seq_div_validating>,
      sml::state<seq_div_validating> [guard::valid_seq_div_context{}] =
          sml::state<seq_div_step>,
      sml::state<seq_div_validating> [guard::invalid_seq_div_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<seq_div_step> / action::run_seq_div_phase =
          sml::state<seq_div_step_decision>,
      sml::state<seq_div_step_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<seq_div_step_decision> [guard::phase_ok{}] =
          sml::state<done>,

      sml::state<prepared> + sml::event<event::apply_updates> / action::begin_apply_updates =
          sml::state<updates_validating>,
      sml::state<updates_validating> [guard::valid_updates_context{}] =
          sml::state<updates_step>,
      sml::state<updates_validating> [guard::invalid_updates_context{}] /
          action::set_invalid_argument = sml::state<errored>,
      sml::state<updates_step> / action::run_updates_phase =
          sml::state<updates_step_decision>,
      sml::state<updates_step_decision> [guard::phase_failed{}] =
          sml::state<errored>,
      sml::state<updates_step_decision> [guard::phase_ok{}] =
          sml::state<done>,

      sml::state<publishing> / action::run_publish_phase = sml::state<publish_decision>,
      sml::state<publish_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<publish_decision> [guard::phase_ok{}] = sml::state<done>,

      sml::state<done> / action::mark_done = sml::state<prepared>,
      sml::state<errored> / action::ensure_last_error = sml::state<prepared>,

      sml::state<initialized> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<prepared> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<validating_prepare> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<preparing_slots> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<preparing_slots_step> + sml::unexpected_event<sml::_> /
          action::on_unexpected{} = sml::state<errored>,
      sml::state<prepare_slots_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected{} = sml::state<errored>,
      sml::state<applying> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<apply_step_validating> + sml::unexpected_event<sml::_> /
          action::on_unexpected{} = sml::state<errored>,
      sml::state<applying_step> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<apply_step_decision> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<rolling_back> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<rollback_step_validating> + sml::unexpected_event<sml::_> /
          action::on_unexpected{} = sml::state<errored>,
      sml::state<rollback_step> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<rollback_step_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected{} = sml::state<errored>,
      sml::state<seq_remove_validating> + sml::unexpected_event<sml::_> /
          action::on_unexpected{} = sml::state<errored>,
      sml::state<seq_remove_step> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<seq_remove_step_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected{} = sml::state<errored>,
      sml::state<seq_copy_validating> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<seq_copy_step> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<seq_copy_step_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected{} = sml::state<errored>,
      sml::state<seq_keep_validating> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<seq_keep_step> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<seq_keep_step_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected{} = sml::state<errored>,
      sml::state<seq_add_validating> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<seq_add_step> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<seq_add_step_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected{} = sml::state<errored>,
      sml::state<seq_div_validating> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<seq_div_step> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<seq_div_step_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected{} = sml::state<errored>,
      sml::state<updates_validating> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<updates_step> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<updates_step_decision> + sml::unexpected_event<sml::_> /
          action::on_unexpected{} = sml::state<errored>,
      sml::state<publishing> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<publish_decision> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<done> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>,
      sml::state<errored> + sml::unexpected_event<sml::_> / action::on_unexpected{} =
          sml::state<errored>
    );
  }
};

struct sm_deps {
  std::unique_ptr<action::context> context_;

  sm_deps() : context_(std::make_unique<action::context>()) {
    // One-time heap allocation keeps kv cache context off the stack.
  }
};

struct sm : private sm_deps, public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : sm_deps(), base_type(*context_) {}

  bool process_event(const event::prepare & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    int32_t err = context_->last_error;
    if (err == EMEL_OK) {
      if (ev.ubatch_count_out != nullptr) {
        *ev.ubatch_count_out = context_->planned_ubatch_count;
      }
      if (ev.slot_offsets_out != nullptr) {
        if (ev.slot_offsets_capacity < context_->planned_ubatch_count) {
          err = EMEL_ERR_INVALID_ARGUMENT;
        } else {
          for (int32_t i = 0; i < context_->planned_ubatch_count; ++i) {
            ev.slot_offsets_out[i] = context_->slot_offsets[i];
          }
        }
      }
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (err != context_->last_error) {
      context_->last_error = err;
    }
    action::clear_request(*context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::apply_ubatch & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_->last_error;
    if (ev.kv_tokens_out != nullptr) {
      *ev.kv_tokens_out = err == EMEL_OK ? context_->kv_tokens : 0;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(*context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::rollback & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_->last_error;
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(*context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::seq_remove & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_->last_error;
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(*context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::seq_copy & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_->last_error;
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(*context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::seq_keep & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_->last_error;
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(*context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::seq_add & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_->last_error;
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(*context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::seq_div & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_->last_error;
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(*context_);
    return accepted && err == EMEL_OK;
  }

  bool process_event(const event::apply_updates & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    const int32_t err = context_->last_error;
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    action::clear_request(*context_);
    return accepted && err == EMEL_OK;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t kv_tokens() const noexcept { return context_->kv_tokens; }
  int32_t last_error() const noexcept { return context_->last_error; }

 private:
  using base_type::raw_sm;
};

}  // namespace emel::kv::cache
