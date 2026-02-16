#pragma once

#include "emel/sm.hpp"
#include "emel/tensor/allocator/actions.hpp"
#include "emel/tensor/allocator/events.hpp"
#include "emel/tensor/allocator/guards.hpp"

namespace emel::tensor::allocator {

using Process = boost::sml::back::process<
  event::validate,
  events::validate_done,
  events::validate_error,
  event::scan_tensors,
  events::scan_done,
  events::scan_error,
  event::partition_ranges,
  events::partition_done,
  events::partition_error,
  event::allocate_ranges,
  events::allocate_ranges_done,
  events::allocate_ranges_error,
  event::initialize_tensors,
  events::initialize_tensors_done,
  events::initialize_tensors_error,
  event::assemble,
  events::assemble_done,
  events::assemble_error,
  events::allocate_done,
  events::allocate_error,
  events::release_done,
  events::release_error>;

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
    using process_t = Process;

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
      sml::state<validating> + sml::on_entry<event::allocate_tensors> /
          [](const event::allocate_tensors & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            event::validate validate{
              .error_out = &phase_error,
              .detail_out = ev.detail_out,
              .chunk_sizes_out = ev.chunk_sizes_out,
              .chunk_sizes_out_count = ev.chunk_sizes_out_count,
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
          sml::state<scanning_tensors>,
      sml::state<validating> + sml::event<events::validate_error> =
          sml::state<failed>,

      sml::state<scanning_tensors> + sml::on_entry<events::validate_done> /
          [](const events::validate_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::allocate_tensors * request = ev.request;
            event::scan_tensors scan{
              .error_out = &phase_error,
              .detail_out = request != nullptr ? request->detail_out : nullptr,
            };
            process(scan);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::scan_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::scan_done{
              .request = request,
            });
          },
      sml::state<scanning_tensors> + sml::event<event::scan_tensors> / action::run_scan_tensors =
          sml::state<scanning_tensors>,
      sml::state<scanning_tensors> + sml::event<events::scan_done> =
          sml::state<partitioning_ranges>,
      sml::state<scanning_tensors> + sml::event<events::scan_error> =
          sml::state<failed>,

      sml::state<partitioning_ranges> + sml::on_entry<events::scan_done> /
          [](const events::scan_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::allocate_tensors * request = ev.request;
            event::partition_ranges partition{
              .error_out = &phase_error,
              .detail_out = request != nullptr ? request->detail_out : nullptr,
            };
            process(partition);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::partition_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::partition_done{
              .request = request,
            });
          },
      sml::state<partitioning_ranges> + sml::event<event::partition_ranges> /
          action::run_partition_ranges = sml::state<partitioning_ranges>,
      sml::state<partitioning_ranges> + sml::event<events::partition_done> =
          sml::state<allocating_ranges>,
      sml::state<partitioning_ranges> + sml::event<events::partition_error> =
          sml::state<failed>,

      sml::state<allocating_ranges> + sml::on_entry<events::partition_done> /
          [](const events::partition_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::allocate_tensors * request = ev.request;
            event::allocate_ranges allocate{
              .error_out = &phase_error,
              .detail_out = request != nullptr ? request->detail_out : nullptr,
            };
            process(allocate);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::allocate_ranges_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::allocate_ranges_done{
              .request = request,
            });
          },
      sml::state<allocating_ranges> + sml::event<event::allocate_ranges> /
          action::run_allocate_ranges = sml::state<allocating_ranges>,
      sml::state<allocating_ranges> + sml::event<events::allocate_ranges_done> =
          sml::state<initializing_tensors>,
      sml::state<allocating_ranges> + sml::event<events::allocate_ranges_error> =
          sml::state<failed>,

      sml::state<initializing_tensors> + sml::on_entry<events::allocate_ranges_done> /
          [](const events::allocate_ranges_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::allocate_tensors * request = ev.request;
            event::initialize_tensors init{
              .error_out = &phase_error,
              .detail_out = request != nullptr ? request->detail_out : nullptr,
            };
            process(init);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::initialize_tensors_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::initialize_tensors_done{
              .request = request,
            });
          },
      sml::state<initializing_tensors> + sml::event<event::initialize_tensors> /
          action::run_initialize_tensors = sml::state<initializing_tensors>,
      sml::state<initializing_tensors> + sml::event<events::initialize_tensors_done> =
          sml::state<assembling_result>,
      sml::state<initializing_tensors> + sml::event<events::initialize_tensors_error> =
          sml::state<failed>,

      sml::state<assembling_result> + sml::on_entry<events::initialize_tensors_done> /
          [](const events::initialize_tensors_done & ev, action::context &, process_t & process) noexcept {
            int32_t phase_error = EMEL_OK;
            const event::allocate_tensors * request = ev.request;
            event::assemble assemble{
              .error_out = &phase_error,
              .detail_out = request != nullptr ? request->detail_out : nullptr,
              .result_buffer_out = request != nullptr ? request->result_buffer_out : nullptr,
              .total_size_out = request != nullptr ? request->total_size_out : nullptr,
              .chunk_sizes_out = request != nullptr ? request->chunk_sizes_out : nullptr,
              .chunk_sizes_out_count = request != nullptr ? request->chunk_sizes_out_count : 0,
              .chunk_count_out = request != nullptr ? request->chunk_count_out : nullptr,
            };
            process(assemble);
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = phase_error;
            }
            if (phase_error != EMEL_OK) {
              process(events::assemble_error{
                .err = phase_error,
                .request = request,
              });
              return;
            }
            process(events::assemble_done{
              .request = request,
            });
          },
      sml::state<assembling_result> + sml::event<event::assemble> / action::run_assemble =
          sml::state<assembling_result>,
      sml::state<assembling_result> + sml::event<events::assemble_done> =
          sml::state<done>,
      sml::state<assembling_result> + sml::event<events::assemble_error> =
          sml::state<failed>,

      sml::state<done> + sml::on_entry<events::assemble_done> /
          [](const events::assemble_done & ev, action::context & ctx,
             process_t & process) noexcept {
            const event::allocate_tensors * request = ev.request;
            if (request != nullptr && request->error_out != nullptr) {
              *request->error_out = EMEL_OK;
            }
            process(events::allocate_done{
              .total_bytes = ctx.total_bytes,
              .chunk_count = ctx.chunk_count,
              .request = request,
            });
          },
      sml::state<done> + sml::event<events::allocate_done> / action::on_allocate_done =
          sml::state<idle>,
      sml::state<done> + sml::event<events::allocate_error> / action::on_allocate_error =
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
      sml::state<releasing> + sml::on_entry<event::release> /
          [](const event::release & ev, action::context &, process_t & process) noexcept {
            const int32_t err = ev.error_out != nullptr ? *ev.error_out : EMEL_OK;
            if (err == EMEL_OK) {
              process(events::release_done{.request = &ev});
            } else {
              process(events::release_error{.err = err, .request = &ev});
            }
          },
      sml::state<releasing> + sml::event<events::release_done> / action::on_release_done =
          sml::state<idle>,
      sml::state<releasing> + sml::event<events::release_error> / action::on_release_error =
          sml::state<failed>,

      sml::state<failed> + sml::on_entry<sml::_> /
          [](const auto & ev, action::context &, process_t & process) noexcept {
            int32_t err = EMEL_ERR_BACKEND;
            if constexpr (requires { ev.err; }) {
              err = ev.err;
            }
            if constexpr (requires { ev.request; }) {
              if constexpr (std::is_same_v<decltype(ev.request), const event::allocate_tensors *>) {
                const event::allocate_tensors * request = ev.request;
                if (request != nullptr && request->error_out != nullptr) {
                  *request->error_out = err;
                }
                if (request != nullptr && request->detail_out != nullptr &&
                    request->detail_out->status == EMEL_OK) {
                  request->detail_out->status = err;
                  request->detail_out->phase = static_cast<uint32_t>(event::error_phase::none);
                  request->detail_out->reason = static_cast<uint32_t>(event::error_reason::unknown);
                }
                process(events::allocate_error{
                  .err = err,
                  .request = request,
                });
                return;
              } else if constexpr (std::is_same_v<decltype(ev.request), const event::release *>) {
                const event::release * request = ev.request;
                if (request != nullptr && request->error_out != nullptr) {
                  *request->error_out = err;
                }
                if (request != nullptr && request->detail_out != nullptr &&
                    request->detail_out->status == EMEL_OK) {
                  request->detail_out->status = err;
                  request->detail_out->phase = static_cast<uint32_t>(event::error_phase::release);
                  request->detail_out->reason = static_cast<uint32_t>(event::error_reason::unknown);
                }
                process(events::release_error{
                  .err = err,
                  .request = request,
                });
                return;
              }
            }
          },
      sml::state<failed> + sml::event<events::allocate_error> / action::on_allocate_error =
          sml::state<idle>,
      sml::state<failed> + sml::event<events::release_error> / action::on_release_error =
          sml::state<idle>
    );
  }
};

struct sm : private emel::detail::process_support<sm, Process>, public emel::sm<model, Process> {
  using base_type = emel::sm<model, Process>;

  sm() : emel::detail::process_support<sm, Process>(this), base_type(context_, this->process_) {}

  using base_type::process_event;

  int32_t total_bytes() const noexcept { return context_.total_bytes; }
  int32_t chunk_count() const noexcept { return context_.chunk_count; }

 private:
  action::context context_{};
};

}  // namespace emel::tensor::allocator
