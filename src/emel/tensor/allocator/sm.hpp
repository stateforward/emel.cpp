#pragma once

#include "emel/sm.hpp"
#include "emel/tensor/allocator/actions.hpp"
#include "emel/tensor/allocator/events.hpp"
#include "emel/tensor/allocator/guards.hpp"

namespace emel::tensor::allocator {

struct idle {};
struct validating {};
struct validate_decision {};
struct scanning_tensors {};
struct scan_decision {};
struct partitioning_ranges {};
struct partition_decision {};
struct allocating_ranges {};
struct allocate_decision {};
struct initializing_tensors {};
struct initialize_decision {};
struct assembling_result {};
struct assemble_decision {};
struct done {};
struct errored {};
struct release_decision {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::allocate_tensors> /
        action::begin_allocate_tensors = sml::state<validating>,
      sml::state<validating> / action::run_validate = sml::state<validate_decision>,
      sml::state<validate_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<validate_decision> [guard::phase_ok{}] = sml::state<scanning_tensors>,

      sml::state<scanning_tensors> / action::run_scan_tensors = sml::state<scan_decision>,
      sml::state<scan_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<scan_decision> [guard::phase_ok{}] = sml::state<partitioning_ranges>,

      sml::state<partitioning_ranges> / action::run_partition_ranges =
        sml::state<partition_decision>,
      sml::state<partition_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<partition_decision> [guard::phase_ok{}] = sml::state<allocating_ranges>,

      sml::state<allocating_ranges> / action::run_allocate_ranges =
        sml::state<allocate_decision>,
      sml::state<allocate_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<allocate_decision> [guard::phase_ok{}] = sml::state<initializing_tensors>,

      sml::state<initializing_tensors> / action::run_initialize_tensors =
        sml::state<initialize_decision>,
      sml::state<initialize_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<initialize_decision> [guard::phase_ok{}] = sml::state<assembling_result>,

      sml::state<assembling_result> / action::run_assemble = sml::state<assemble_decision>,
      sml::state<assemble_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<assemble_decision> [guard::phase_ok{}] = sml::state<done>,

      sml::state<done> = sml::state<idle>,
      sml::state<errored> = sml::state<idle>,

      sml::state<idle> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<validating> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<validate_decision> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<scanning_tensors> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<scan_decision> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<partitioning_ranges> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<partition_decision> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<allocating_ranges> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<allocate_decision> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<initializing_tensors> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<initialize_decision> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<assembling_result> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<assemble_decision> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<done> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<errored> + sml::event<event::release> / action::begin_release =
        sml::state<release_decision>,
      sml::state<release_decision> [guard::phase_failed{}] = sml::state<errored>,
      sml::state<release_decision> [guard::phase_ok{}] = sml::state<idle>,

      sml::state<idle> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<errored>,
      sml::state<validating> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<errored>,
      sml::state<validate_decision> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<errored>,
      sml::state<scanning_tensors> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<errored>,
      sml::state<scan_decision> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<errored>,
      sml::state<partitioning_ranges> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<partition_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<allocating_ranges> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<allocate_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<initializing_tensors> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<initialize_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<assembling_result> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<assemble_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>,
      sml::state<done> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<errored>,
      sml::state<errored> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<errored>,
      sml::state<release_decision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<errored>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;
  using base_type::visit_current_states;

  bool process_event(const event::allocate_tensors & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    int32_t err = context_.phase_error;
    if (!accepted && err == EMEL_OK) {
      err = EMEL_ERR_BACKEND;
    }
    if (err == EMEL_OK) {
      if (ev.total_size_out != nullptr) {
        *ev.total_size_out = context_.total_bytes;
      }
      if (ev.chunk_count_out != nullptr) {
        *ev.chunk_count_out = context_.chunk_count;
      }
      if (ev.chunk_sizes_out != nullptr &&
          ev.chunk_sizes_out_count >= context_.chunk_count) {
        for (int32_t i = 0; i < context_.chunk_count; ++i) {
          ev.chunk_sizes_out[i] = context_.chunk_sizes[i];
        }
      }
      if (ev.result_buffer_out != nullptr) {
        *ev.result_buffer_out = context_.result_buffer;
      }
    } else {
      if (ev.total_size_out != nullptr) {
        *ev.total_size_out = 0;
      }
      if (ev.chunk_count_out != nullptr) {
        *ev.chunk_count_out = 0;
      }
      if (ev.result_buffer_out != nullptr) {
        *ev.result_buffer_out = nullptr;
      }
    }

    if (err != EMEL_OK && context_.detail.status == EMEL_OK) {
      action::detail::set_error_detail(
          &context_.detail,
          err,
          event::error_phase::none,
          event::error_reason::unknown,
          -1,
          0);
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ev.detail_out != nullptr) {
      *ev.detail_out = context_.detail;
    }
    action::reset_phase(context_);
    return emel::detail::normalize_event_result(ev, accepted);
  }

  bool process_event(const event::release & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    int32_t err = context_.phase_error;
    if (!accepted && err == EMEL_OK) {
      err = EMEL_ERR_BACKEND;
    }
    if (err != EMEL_OK && context_.detail.status == EMEL_OK) {
      action::detail::set_error_detail(
          &context_.detail,
          err,
          event::error_phase::release,
          event::error_reason::unknown,
          -1,
          0);
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ev.detail_out != nullptr) {
      *ev.detail_out = context_.detail;
    }
    action::reset_phase(context_);
    return emel::detail::normalize_event_result(ev, accepted);
  }

  int32_t total_bytes() const noexcept { return context_.total_bytes; }
  int32_t chunk_count() const noexcept { return context_.chunk_count; }

 private:
  action::context context_{};
};

}  // namespace emel::tensor::allocator
