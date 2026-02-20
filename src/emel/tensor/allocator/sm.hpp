#pragma once

#include "emel/sm.hpp"
#include "emel/tensor/allocator/actions.hpp"
#include "emel/tensor/allocator/events.hpp"
#include "emel/tensor/allocator/guards.hpp"

namespace emel::tensor::allocator {

struct Idle {};
struct Validating {};
struct ValidateDecision {};
struct ScanningTensors {};
struct ScanDecision {};
struct PartitioningRanges {};
struct PartitionDecision {};
struct AllocatingRanges {};
struct AllocateDecision {};
struct InitializingTensors {};
struct InitializeDecision {};
struct AssemblingResult {};
struct AssembleDecision {};
struct Done {};
struct Errored {};
struct ReleaseDecision {};

struct model {
  using context = action::context;

  auto operator()() const {
    namespace sml = boost::sml;
    return sml::make_transition_table(
      *sml::state<Idle> + sml::event<event::allocate_tensors> /
        action::begin_allocate_tensors = sml::state<Validating>,
      sml::state<Validating> / action::run_validate = sml::state<ValidateDecision>,
      sml::state<ValidateDecision> [guard::phase_failed{}] = sml::state<Errored>,
      sml::state<ValidateDecision> [guard::phase_ok{}] = sml::state<ScanningTensors>,

      sml::state<ScanningTensors> / action::run_scan_tensors = sml::state<ScanDecision>,
      sml::state<ScanDecision> [guard::phase_failed{}] = sml::state<Errored>,
      sml::state<ScanDecision> [guard::phase_ok{}] = sml::state<PartitioningRanges>,

      sml::state<PartitioningRanges> / action::run_partition_ranges =
        sml::state<PartitionDecision>,
      sml::state<PartitionDecision> [guard::phase_failed{}] = sml::state<Errored>,
      sml::state<PartitionDecision> [guard::phase_ok{}] = sml::state<AllocatingRanges>,

      sml::state<AllocatingRanges> / action::run_allocate_ranges =
        sml::state<AllocateDecision>,
      sml::state<AllocateDecision> [guard::phase_failed{}] = sml::state<Errored>,
      sml::state<AllocateDecision> [guard::phase_ok{}] = sml::state<InitializingTensors>,

      sml::state<InitializingTensors> / action::run_initialize_tensors =
        sml::state<InitializeDecision>,
      sml::state<InitializeDecision> [guard::phase_failed{}] = sml::state<Errored>,
      sml::state<InitializeDecision> [guard::phase_ok{}] = sml::state<AssemblingResult>,

      sml::state<AssemblingResult> / action::run_assemble = sml::state<AssembleDecision>,
      sml::state<AssembleDecision> [guard::phase_failed{}] = sml::state<Errored>,
      sml::state<AssembleDecision> [guard::phase_ok{}] = sml::state<Done>,

      sml::state<Done> = sml::state<Idle>,
      sml::state<Errored> = sml::state<Idle>,

      sml::state<Idle> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<Validating> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<ValidateDecision> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<ScanningTensors> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<ScanDecision> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<PartitioningRanges> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<PartitionDecision> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<AllocatingRanges> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<AllocateDecision> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<InitializingTensors> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<InitializeDecision> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<AssemblingResult> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<AssembleDecision> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<Done> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<Errored> + sml::event<event::release> / action::begin_release =
        sml::state<ReleaseDecision>,
      sml::state<ReleaseDecision> [guard::phase_failed{}] = sml::state<Errored>,
      sml::state<ReleaseDecision> [guard::phase_ok{}] = sml::state<Idle>,

      sml::state<Idle> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<Errored>,
      sml::state<Validating> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<Errored>,
      sml::state<ValidateDecision> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<Errored>,
      sml::state<ScanningTensors> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<Errored>,
      sml::state<ScanDecision> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<Errored>,
      sml::state<PartitioningRanges> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>,
      sml::state<PartitionDecision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>,
      sml::state<AllocatingRanges> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>,
      sml::state<AllocateDecision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>,
      sml::state<InitializingTensors> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>,
      sml::state<InitializeDecision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>,
      sml::state<AssemblingResult> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>,
      sml::state<AssembleDecision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>,
      sml::state<Done> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<Errored>,
      sml::state<Errored> + sml::unexpected_event<sml::_> / action::on_unexpected =
        sml::state<Errored>,
      sml::state<ReleaseDecision> + sml::unexpected_event<sml::_> /
        action::on_unexpected = sml::state<Errored>
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
