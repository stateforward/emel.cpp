#pragma once

#include "emel/buffer/chunk_allocator/actions.hpp"
#include "emel/buffer/chunk_allocator/events.hpp"
#include "emel/buffer/chunk_allocator/guards.hpp"
#include "emel/sm.hpp"

namespace emel::buffer::chunk_allocator {

/**
 * Dynamic chunk allocator orchestration model.
 *
 * Parity reference:
 * - `ggml_dyn_tallocr_alloc(...)`
 * - `ggml_dyn_tallocr_free_bytes(...)`
 * - `ggml_dyn_tallocr_reset(...)`
 *
 * State purposes:
 * - `ready`: accepts configure, allocate, release, and reset operations.
 * - `configuring`/`applying_configure`: validates and applies runtime allocator limits.
 * - `validating_allocate`/`selecting_block`/`ensuring_chunk`/`committing_allocate`:
 *   allocation pipeline with best-fit and chunk growth behavior.
 * - `validating_release`/`merging_release`: free-bytes merge pipeline.
 * - `resetting`: clears chunk/free-block runtime state.
 * - `failed`: operation failure terminal before operation-specific `_error`.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct ready {};
    struct configuring {};
    struct applying_configure {};
    struct configure_done {};
    struct validating_allocate {};
    struct selecting_block {};
    struct ensuring_chunk {};
    struct committing_allocate {};
    struct allocate_done {};
    struct validating_release {};
    struct merging_release {};
    struct release_done {};
    struct resetting {};
    struct reset_done {};
    struct failed {};

    return sml::make_transition_table(
      *sml::state<ready> + sml::event<event::configure> / action::begin_configure =
          sml::state<configuring>,
      sml::state<configuring> + sml::event<event::validate_configure> /
          action::run_validate_configure = sml::state<configuring>,
      sml::state<configuring> + sml::event<events::validate_configure_done> =
          sml::state<applying_configure>,
      sml::state<configuring> + sml::event<events::validate_configure_error> = sml::state<failed>,
      sml::state<applying_configure> + sml::event<event::apply_configure> /
          action::run_apply_configure = sml::state<applying_configure>,
      sml::state<applying_configure> + sml::event<events::apply_configure_done> =
          sml::state<configure_done>,
      sml::state<applying_configure> + sml::event<events::apply_configure_error> =
          sml::state<failed>,
      sml::state<configure_done> + sml::event<events::configure_done> / action::on_configure_done =
          sml::state<ready>,

      sml::state<ready> + sml::event<event::allocate> / action::begin_allocate =
          sml::state<validating_allocate>,
      sml::state<validating_allocate> + sml::event<event::validate_allocate> /
          action::run_validate_allocate = sml::state<validating_allocate>,
      sml::state<validating_allocate> + sml::event<events::validate_allocate_done> =
          sml::state<selecting_block>,
      sml::state<validating_allocate> + sml::event<events::validate_allocate_error> =
          sml::state<failed>,
      sml::state<selecting_block> + sml::event<event::select_block> / action::run_select_block =
          sml::state<selecting_block>,
      sml::state<selecting_block> + sml::event<events::select_block_done> =
          sml::state<ensuring_chunk>,
      sml::state<selecting_block> + sml::event<events::select_block_error> = sml::state<failed>,
      sml::state<ensuring_chunk> + sml::event<event::ensure_chunk> / action::run_ensure_chunk =
          sml::state<ensuring_chunk>,
      sml::state<ensuring_chunk> + sml::event<events::ensure_chunk_done> =
          sml::state<committing_allocate>,
      sml::state<ensuring_chunk> + sml::event<events::ensure_chunk_error> = sml::state<failed>,
      sml::state<committing_allocate> + sml::event<event::commit_allocate> /
          action::run_commit_allocate = sml::state<committing_allocate>,
      sml::state<committing_allocate> + sml::event<events::commit_allocate_done> =
          sml::state<allocate_done>,
      sml::state<committing_allocate> + sml::event<events::commit_allocate_error> =
          sml::state<failed>,
      sml::state<allocate_done> + sml::event<events::allocate_done> / action::on_allocate_done =
          sml::state<ready>,

      sml::state<ready> + sml::event<event::release> / action::begin_release =
          sml::state<validating_release>,
      sml::state<validating_release> + sml::event<event::validate_release> /
          action::run_validate_release = sml::state<validating_release>,
      sml::state<validating_release> + sml::event<events::validate_release_done> =
          sml::state<merging_release>,
      sml::state<validating_release> + sml::event<events::validate_release_error> =
          sml::state<failed>,
      sml::state<merging_release> + sml::event<event::merge_release> / action::run_merge_release =
          sml::state<merging_release>,
      sml::state<merging_release> + sml::event<events::merge_release_done> =
          sml::state<release_done>,
      sml::state<merging_release> + sml::event<events::merge_release_error> = sml::state<failed>,
      sml::state<release_done> + sml::event<events::release_done> / action::on_release_done =
          sml::state<ready>,

      sml::state<ready> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<configuring> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<applying_configure> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<configure_done> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<validating_allocate> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<selecting_block> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<ensuring_chunk> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<committing_allocate> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<allocate_done> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<validating_release> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<merging_release> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<release_done> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<failed> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<resetting> + sml::event<event::apply_reset> / action::run_apply_reset =
          sml::state<resetting>,
      sml::state<resetting> + sml::event<events::apply_reset_done> = sml::state<reset_done>,
      sml::state<resetting> + sml::event<events::apply_reset_error> = sml::state<failed>,
      sml::state<reset_done> + sml::event<events::reset_done> / action::on_reset_done =
          sml::state<ready>,

      sml::state<failed> + sml::event<events::configure_error> / action::on_configure_error =
          sml::state<ready>,
      sml::state<failed> + sml::event<events::allocate_error> / action::on_allocate_error =
          sml::state<ready>,
      sml::state<failed> + sml::event<events::release_error> / action::on_release_error =
          sml::state<ready>,
      sml::state<failed> + sml::event<events::reset_error> / action::on_reset_error =
          sml::state<ready>
    );
  }
};

struct sm : emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::configure & ev) {
    if (!base_type::process_event(ev)) return false;

    int32_t phase_error = EMEL_OK;
    if (!run_phase<
            event::validate_configure, events::validate_configure_done,
            events::validate_configure_error>(phase_error)) {
      return finalize_configure_error(phase_error);
    }
    if (!run_phase<event::apply_configure, events::apply_configure_done, events::apply_configure_error>(
            phase_error)) {
      return finalize_configure_error(phase_error);
    }
    return base_type::process_event(events::configure_done{});
  }

  bool process_event(const event::allocate & ev) {
    if (!base_type::process_event(ev)) return false;

    int32_t phase_error = EMEL_OK;
    if (!run_phase<
            event::validate_allocate, events::validate_allocate_done,
            events::validate_allocate_error>(phase_error)) {
      return finalize_allocate_error(phase_error);
    }
    if (!run_phase<event::select_block, events::select_block_done, events::select_block_error>(
            phase_error)) {
      return finalize_allocate_error(phase_error);
    }
    if (!run_phase<event::ensure_chunk, events::ensure_chunk_done, events::ensure_chunk_error>(
            phase_error)) {
      return finalize_allocate_error(phase_error);
    }
    if (!run_phase<
            event::commit_allocate, events::commit_allocate_done,
            events::commit_allocate_error>(phase_error)) {
      return finalize_allocate_error(phase_error);
    }

    return base_type::process_event(events::allocate_done{
      .chunk = context_.result_chunk,
      .offset = context_.result_offset,
      .size = context_.result_size,
    });
  }

  bool process_event(const event::release & ev) {
    if (!base_type::process_event(ev)) return false;

    int32_t phase_error = EMEL_OK;
    if (!run_phase<
            event::validate_release, events::validate_release_done,
            events::validate_release_error>(phase_error)) {
      return finalize_release_error(phase_error);
    }
    if (!run_phase<event::merge_release, events::merge_release_done, events::merge_release_error>(
            phase_error)) {
      return finalize_release_error(phase_error);
    }

    return base_type::process_event(events::release_done{});
  }

  bool process_event(const event::reset & ev) {
    if (!base_type::process_event(ev)) return false;

    int32_t phase_error = EMEL_OK;
    if (!run_phase<event::apply_reset, events::apply_reset_done, events::apply_reset_error>(
            phase_error)) {
      return finalize_reset_error(phase_error);
    }
    return base_type::process_event(events::reset_done{});
  }

  uint64_t alignment() const noexcept { return context_.alignment; }

  uint64_t max_chunk_size() const noexcept { return context_.max_chunk_size; }

  int32_t chunk_count() const noexcept { return context_.chunk_count; }

  uint64_t chunk_max_size(const int32_t chunk) const noexcept {
    if (chunk < 0 || chunk >= context_.chunk_count) return 0;
    return context_.chunks[chunk].max_size;
  }

  int32_t last_error() const noexcept { return context_.last_error; }

 private:
  template <class TriggerEvent, class DoneEvent, class ErrorEvent>
  bool run_phase(int32_t & error_out) {
    error_out = EMEL_OK;
    TriggerEvent trigger{};
    trigger.error_out = &error_out;
    if (!base_type::process_event(trigger)) {
      error_out = EMEL_ERR_BACKEND;
      return false;
    }
    if (error_out == EMEL_OK) {
      return base_type::process_event(DoneEvent{});
    }
    (void)base_type::process_event(ErrorEvent{
      .err = error_out,
    });
    return false;
  }

  bool finalize_configure_error(const int32_t error_code) {
    const int32_t err = error_code == EMEL_OK ? EMEL_ERR_BACKEND : error_code;
    (void)base_type::process_event(events::configure_error{
      .err = err,
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

  bool finalize_release_error(const int32_t error_code) {
    const int32_t err = error_code == EMEL_OK ? EMEL_ERR_BACKEND : error_code;
    (void)base_type::process_event(events::release_error{
      .err = err,
    });
    return false;
  }

  bool finalize_reset_error(const int32_t error_code) {
    const int32_t err = error_code == EMEL_OK ? EMEL_ERR_BACKEND : error_code;
    (void)base_type::process_event(events::reset_error{
      .err = err,
    });
    return false;
  }

  action::context context_{};
};

}  // namespace emel::buffer::chunk_allocator

