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
 * - `configuring`/`applying_configure`/`configure_final`: validates and applies runtime allocator limits.
 * - `validating_allocate`/`selecting_block`/`ensuring_chunk`/`ensure_final`/`allocate_final`:
 *   allocation pipeline with best-fit and chunk growth behavior.
 * - `validating_release`/`merging_release`/`release_final`: free-bytes merge pipeline.
 * - `resetting`/`reset_final`: clears chunk/free-block runtime state.
 * - `failed`: operation failure terminal before operation-specific `_error`.
 */
struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct ready {};
    struct configuring {};
    struct applying_configure {};
    struct configure_final {};
    struct validating_allocate {};
    struct selecting_block {};
    struct ensuring_chunk {};
    struct ensure_final {};
    struct allocate_final {};
    struct validating_release {};
    struct merging_release {};
    struct release_final {};
    struct resetting {};
    struct reset_final {};
    struct failed {};

    return sml::make_transition_table(
      *sml::state<ready> + sml::event<event::configure> [guard::can_configure{}] /
          action::begin_configure = sml::state<configuring>,
      sml::state<ready> + sml::event<event::configure> / action::reject_invalid =
          sml::state<failed>,
      sml::state<configuring> / action::run_validate_configure = sml::state<applying_configure>,
      sml::state<applying_configure> / action::run_apply_configure =
          sml::state<configure_final>,
      sml::state<configure_final> [guard::phase_failed] / action::on_configure_error =
          sml::state<failed>,
      sml::state<configure_final> [guard::phase_ok] / action::on_configure_done =
          sml::state<ready>,

      sml::state<ready> + sml::event<event::allocate> [guard::can_allocate{}] /
          action::begin_allocate = sml::state<validating_allocate>,
      sml::state<ready> + sml::event<event::allocate> / action::reject_invalid =
          sml::state<failed>,
      sml::state<validating_allocate> / action::run_validate_allocate =
          sml::state<selecting_block>,
      sml::state<selecting_block> / action::run_select_block =
          sml::state<ensuring_chunk>,
      sml::state<ensuring_chunk> / action::run_ensure_chunk =
          sml::state<ensure_final>,
      sml::state<ensure_final> [guard::phase_failed] / action::on_allocate_error =
          sml::state<failed>,
      sml::state<ensure_final> [guard::phase_ok] / action::run_commit_allocate =
          sml::state<allocate_final>,
      sml::state<allocate_final> [guard::phase_failed] / action::on_allocate_error =
          sml::state<failed>,
      sml::state<allocate_final> [guard::phase_ok] / action::on_allocate_done =
          sml::state<ready>,

      sml::state<ready> + sml::event<event::release> [guard::can_release{}] /
          action::begin_release = sml::state<validating_release>,
      sml::state<ready> + sml::event<event::release> / action::reject_invalid =
          sml::state<failed>,
      sml::state<validating_release> / action::run_validate_release =
          sml::state<merging_release>,
      sml::state<merging_release> / action::run_merge_release =
          sml::state<release_final>,
      sml::state<release_final> [guard::phase_failed] / action::on_release_error =
          sml::state<failed>,
      sml::state<release_final> [guard::phase_ok] / action::on_release_done =
          sml::state<ready>,

      sml::state<ready> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<configuring> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<applying_configure> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<configure_final> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<validating_allocate> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<selecting_block> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<ensuring_chunk> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<ensure_final> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<allocate_final> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<validating_release> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<merging_release> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<release_final> + sml::event<event::reset> / action::begin_reset =
          sml::state<resetting>,
      sml::state<failed> + sml::event<event::reset> / action::begin_reset = sml::state<resetting>,
      sml::state<resetting> / action::run_apply_reset = sml::state<reset_final>,
      sml::state<reset_final> [guard::phase_failed] / action::on_reset_error =
          sml::state<failed>,
      sml::state<reset_final> [guard::phase_ok] / action::on_reset_done =
          sml::state<ready>,

      sml::state<failed> [guard::always] = sml::state<ready>,

      sml::state<configuring> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<configuring> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<configuring> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,

      sml::state<applying_configure> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<applying_configure> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<applying_configure> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,

      sml::state<configure_final> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<configure_final> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<configure_final> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,

      sml::state<validating_allocate> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<validating_allocate> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<validating_allocate> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,

      sml::state<selecting_block> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<selecting_block> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<selecting_block> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,

      sml::state<ensuring_chunk> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<ensuring_chunk> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<ensuring_chunk> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,

      sml::state<ensure_final> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<ensure_final> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<ensure_final> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,

      sml::state<allocate_final> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<allocate_final> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<allocate_final> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,

      sml::state<validating_release> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<validating_release> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<validating_release> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,

      sml::state<merging_release> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<merging_release> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<merging_release> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,

      sml::state<release_final> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<release_final> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<release_final> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,

      sml::state<resetting> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<resetting> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<resetting> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,
      sml::state<resetting> + sml::event<event::reset> / action::on_unexpected =
          sml::state<failed>,

      sml::state<reset_final> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reset_final> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reset_final> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>,
      sml::state<reset_final> + sml::event<event::reset> / action::on_unexpected =
          sml::state<failed>,

      sml::state<failed> + sml::event<event::configure> / action::on_unexpected =
          sml::state<failed>,
      sml::state<failed> + sml::event<event::allocate> / action::on_unexpected =
          sml::state<failed>,
      sml::state<failed> + sml::event<event::release> / action::on_unexpected =
          sml::state<failed>
    );
  }
};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  using base_type::process_event;

  bool process_event(const event::configure & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    if (context_.phase_error != EMEL_OK && ev.error_out != nullptr) {
      *ev.error_out = action::detail::normalize_error(context_.phase_error, EMEL_ERR_BACKEND);
    }
    return emel::detail::normalize_event_result(ev, accepted);
  }

  bool process_event(const event::allocate & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    if (context_.phase_error != EMEL_OK) {
      if (ev.error_out != nullptr) {
        *ev.error_out = action::detail::normalize_error(context_.phase_error, EMEL_ERR_BACKEND);
      }
    } else {
      if (ev.chunk_out != nullptr) {
        *ev.chunk_out = context_.result_chunk;
      }
      if (ev.offset_out != nullptr) {
        *ev.offset_out = context_.result_offset;
      }
      if (ev.aligned_size_out != nullptr) {
        *ev.aligned_size_out = context_.result_size;
      }
    }
    return emel::detail::normalize_event_result(ev, accepted);
  }

  bool process_event(const event::release & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    if (context_.phase_error != EMEL_OK && ev.error_out != nullptr) {
      *ev.error_out = action::detail::normalize_error(context_.phase_error, EMEL_ERR_BACKEND);
    }
    return emel::detail::normalize_event_result(ev, accepted);
  }

  bool process_event(const event::reset & ev) {
    const bool accepted = this->raw_sm().process_event(ev);
    if (context_.phase_error != EMEL_OK && ev.error_out != nullptr) {
      *ev.error_out = action::detail::normalize_error(context_.phase_error, EMEL_ERR_BACKEND);
    }
    return emel::detail::normalize_event_result(ev, accepted);
  }

  uint64_t alignment() const noexcept { return context_.alignment; }

  uint64_t max_chunk_size() const noexcept { return context_.max_chunk_size; }

  int32_t chunk_count() const noexcept { return context_.chunk_count; }

  uint64_t chunk_max_size(const int32_t chunk) const noexcept {
    if (chunk < 0 || chunk >= context_.chunk_count) return 0;
    return context_.chunks[chunk].max_size;
  }

 private:
  action::context context_{};
};

}  // namespace emel::buffer::chunk_allocator
