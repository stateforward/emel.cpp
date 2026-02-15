#pragma once

#include "emel/sm.hpp"
#include "emel/tensor/allocator/actions.hpp"
#include "emel/tensor/allocator/events.hpp"
#include "emel/tensor/allocator/guards.hpp"

namespace emel::tensor::allocator {

/**
 * Tensor allocator orchestration model.
 *
 * Runtime invariants:
 * - All orchestration runs through events on this machine boundary.
 * - Phase outcomes route through explicit `_done` / `_error` events only.
 * - Side effects (allocation, backend init, assemble, release) occur in actions only.
 * - Completion is explicit: `events::allocate_done`, `events::allocate_error`,
 *   `events::release_done`, `events::release_error`.
 *
 * State purposes:
 * - `idle`: accepts top-level `event::allocate_tensors` and `event::release`.
 * - `validating`: validates input/event payload and callback contract.
 * - `scanning_tensors`: normalizes tensor metadata and effective sizes.
 * - `partitioning_ranges`: builds chunk assignments and byte offsets.
 * - `allocating_ranges`: allocates per-chunk backing buffers when `no_alloc == false`.
 * - `initializing_tensors`: initializes regular/view tensors via backend callbacks.
 * - `assembling_result`: publishes chunk sizes/total size and assembled result buffer.
 * - `done`: successful terminal for allocate flow (before final completion emission).
 * - `failed`: error terminal for allocate flow (before final error emission).
 * - `releasing`: teardown flow that releases owned allocated buffers.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct idle {};
    struct validating {};
    struct scanning_tensors {};
    struct partitioning_ranges {};
    struct allocating_ranges {};
    struct initializing_tensors {};
    struct assembling_result {};
    struct done {};
    struct failed {};
    struct releasing {};

    return sml::make_transition_table(
      *sml::state<idle> + sml::event<event::allocate_tensors> / action::begin_allocate_tensors =
          sml::state<validating>,

      sml::state<validating> + sml::event<event::validate> / action::run_validate =
          sml::state<validating>,
      sml::state<validating> + sml::event<events::validate_done> =
          sml::state<scanning_tensors>,
      sml::state<validating> + sml::event<events::validate_error> =
          sml::state<failed>,

      sml::state<scanning_tensors> + sml::event<event::scan_tensors> / action::run_scan_tensors =
          sml::state<scanning_tensors>,
      sml::state<scanning_tensors> + sml::event<events::scan_done> =
          sml::state<partitioning_ranges>,
      sml::state<scanning_tensors> + sml::event<events::scan_error> =
          sml::state<failed>,

      sml::state<partitioning_ranges> + sml::event<event::partition_ranges> /
          action::run_partition_ranges = sml::state<partitioning_ranges>,
      sml::state<partitioning_ranges> + sml::event<events::partition_done> =
          sml::state<allocating_ranges>,
      sml::state<partitioning_ranges> + sml::event<events::partition_error> =
          sml::state<failed>,

      sml::state<allocating_ranges> + sml::event<event::allocate_ranges> /
          action::run_allocate_ranges = sml::state<allocating_ranges>,
      sml::state<allocating_ranges> + sml::event<events::allocate_ranges_done> =
          sml::state<initializing_tensors>,
      sml::state<allocating_ranges> + sml::event<events::allocate_ranges_error> =
          sml::state<failed>,

      sml::state<initializing_tensors> + sml::event<event::initialize_tensors> /
          action::run_initialize_tensors = sml::state<initializing_tensors>,
      sml::state<initializing_tensors> + sml::event<events::initialize_tensors_done> =
          sml::state<assembling_result>,
      sml::state<initializing_tensors> + sml::event<events::initialize_tensors_error> =
          sml::state<failed>,

      sml::state<assembling_result> + sml::event<event::assemble> / action::run_assemble =
          sml::state<assembling_result>,
      sml::state<assembling_result> + sml::event<events::assemble_done> =
          sml::state<done>,
      sml::state<assembling_result> + sml::event<events::assemble_error> =
          sml::state<failed>,

      sml::state<done> + sml::event<events::allocate_done> / action::on_allocate_done =
          sml::state<idle>,
      sml::state<failed> + sml::event<events::allocate_error> / action::on_allocate_error =
          sml::state<idle>,

      sml::state<idle> + sml::event<event::release> / action::begin_release = sml::state<releasing>,
      sml::state<validating> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<scanning_tensors> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<partitioning_ranges> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<allocating_ranges> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<initializing_tensors> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<assembling_result> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<done> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<failed> + sml::event<event::release> / action::begin_release =
          sml::state<releasing>,
      sml::state<releasing> + sml::event<events::release_done> / action::on_release_done =
          sml::state<idle>,
      sml::state<releasing> + sml::event<events::release_error> / action::on_release_error =
          sml::state<failed>
    );
  }
};

struct sm : emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::allocate_tensors & ev) {
    if (!base_type::process_event(ev)) return false;
    int32_t phase_error = EMEL_OK;
    if (!run_phase<event::validate, events::validate_done, events::validate_error>(phase_error)) {
      return finalize_allocate_error(phase_error);
    }
    if (!run_phase<event::scan_tensors, events::scan_done, events::scan_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_allocate_error(phase_error);
    }
    if (!run_phase<event::partition_ranges, events::partition_done, events::partition_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_allocate_error(phase_error);
    }
    if (!run_phase<event::allocate_ranges, events::allocate_ranges_done, events::allocate_ranges_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_allocate_error(phase_error);
    }
    if (!run_phase<
            event::initialize_tensors,
            events::initialize_tensors_done,
            events::initialize_tensors_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_allocate_error(phase_error);
    }
    if (!run_phase<event::assemble, events::assemble_done, events::assemble_error>(
            phase_error)) {  // GCOVR_EXCL_BR_LINE
      return finalize_allocate_error(phase_error);
    }
    return base_type::process_event(events::allocate_done{
      .total_bytes = context_.total_bytes,
      .chunk_count = context_.chunk_count,
    });
  }

  bool process_event(const event::release & ev) {
    int32_t phase_error = EMEL_OK;
    event::release release_ev = ev;
    release_ev.error_out = &phase_error;
    if (!base_type::process_event(release_ev)) return false;
    if (phase_error == EMEL_OK) {  // GCOVR_EXCL_BR_LINE
      return base_type::process_event(events::release_done{});
    }
    (void)base_type::process_event(events::release_error{
      .err = phase_error,
    });
    return false;
  }

  int32_t total_bytes() const noexcept { return context_.total_bytes; }
  int32_t chunk_count() const noexcept { return context_.chunk_count; }

  private:
  // Executes one phase in run-to-completion style:
  // 1. dispatch trigger event
  // 2. emit `_done` when action reports no error
  // 3. emit `_error` when action reports an error
  template <class TriggerEvent, class DoneEvent, class ErrorEvent>
  bool run_phase(int32_t & error_out) {
    error_out = EMEL_OK;
    TriggerEvent trigger{};
    trigger.error_out = &error_out;
    if (!base_type::process_event(trigger)) {  // GCOVR_EXCL_BR_LINE
      error_out = EMEL_ERR_BACKEND;
      return false;
    }
    if (error_out == EMEL_OK) {  // GCOVR_EXCL_BR_LINE
      return base_type::process_event(DoneEvent{});
    }
    (void)base_type::process_event(ErrorEvent{
      .err = error_out,
    });
    return false;
  }

  bool finalize_allocate_error(const int32_t error_code) {
    const int32_t err = error_code == EMEL_OK ? EMEL_ERR_BACKEND : error_code;
    (void)base_type::process_event(events::allocate_error{
      .err = err,
    });
    return false;
  }

  action::context context_{};
};

}  // namespace emel::tensor::allocator
