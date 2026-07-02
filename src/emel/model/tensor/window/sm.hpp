#pragma once

// benchmark: designed

// Streaming weight window: tensor-owned residency for models larger than RAM.
//
// The machine owns a pinned whole-file mmap source plus a ring of K
// pre-allocated layer slots filled ahead of compute by a small I/O pool. The
// caller (generator initializer / bench fixture) builds per-layer weight
// extents and binds once; the hot path is one acquire per layer which
// publishes a slot view, suspends on the already-submitted slot load when the
// layer is still in flight, and advances the prefetch ring on success.
//
// RTC interpretation (load-bearing): prefetch I/O legally spans top-level
// dispatches because (a) the in-flight state is explicit machine state (slot
// lifecycle `loading` in the residency table — the same category as
// model::tensor's tensor_storage lifecycle), (b) no coroutine continuation
// ever persists across dispatches — between dispatches the only artifacts are
// the running staged copy on the I/O pool and a passive fired flag, and
// (c) completion re-enters the machine as an explicit external event
// (emel::event::completion) delivered by the external-completion co_sm
// backend: required completions drain before the dispatch that required them
// returns, and background prefetch fires are swept ascending at the start of
// the next dispatch (sml.rules §2 explicit-external-event clause).
//
// Acquire is a two-dispatch protocol driven unconditionally by the typed
// process_event wrapper: the begin dispatch decides resident/loading/missing,
// submits and requires the slot load as needed (the backend drain commits it
// before returning), and the finish dispatch routes on the committed
// residency to publish done/error and advance the ring. Both dispatches carry
// the same request payload, so no dispatch-local data touches context.
//
// Invariants: slot_for(layer) = layer % slot_count with prefetch_depth <
// slot_count means sequential acquire never lands on a slot mid-load for a
// different layer; unbind requires every loading slot so the drain joins all
// in-flight copies before teardown; completion source index == slot index.

#include "emel/model/tensor/window/actions.hpp"
#include "emel/model/tensor/window/context.hpp"
#include "emel/model/tensor/window/detail.hpp"
#include "emel/model/tensor/window/events.hpp"
#include "emel/model/tensor/window/guards.hpp"
#include "emel/sm.hpp"

namespace emel::model::tensor::window {

struct state_unbound {};
struct state_bind_request_decision {};
struct state_bind_source_decision {};
struct state_bind_budget_decision {};
struct state_bind_done_callback {};
struct state_bind_error_ready {};
struct state_bind_error_callback {};
struct state_ready {};
struct state_passthrough_ready {};
struct state_acquire_resolve_decision {};
struct state_acquire_resolved {};
struct state_acquire_decision {};
struct state_acquire_pending {};
struct state_passthrough_acquire_resolved {};
struct state_acquire_publish_decision {};
struct state_acquire_advance_decision {};
struct state_acquire_publish_ready {};
struct state_acquire_done_callback {};
struct state_acquire_error_ready {};
struct state_acquire_error_callback {};
struct state_passthrough_acquire_pending {};
struct state_passthrough_acquire_error_ready {};
struct state_passthrough_acquire_error_callback {};
struct state_unbind_pending {};
struct state_unbind_finish_decision {};
struct state_unbind_publish_ready {};
struct state_unbind_done_callback {};
struct state_unbind_error_ready {};
struct state_unbind_error_callback {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // bind: request validation -> whole-file source mapping -> layer plan ->
      // budget decision -> passthrough (fits) or streaming (slots + prime).
        sml::state<state_bind_request_decision> <= *sml::state<state_unbound>
          + sml::event<detail::bind_window_runtime>
          / action::effect_begin_bind
      , sml::state<state_bind_source_decision> <=
          sml::state<state_bind_request_decision>
          + sml::completion<detail::bind_window_runtime>
          [ guard::guard_bind_request_valid{} ]
          / action::effect_map_source
      , sml::state<state_bind_error_ready> <=
          sml::state<state_bind_request_decision>
          + sml::completion<detail::bind_window_runtime>
          [ guard::guard_bind_request_invalid{} ]
          / action::effect_mark_bind_invalid
      , sml::state<state_bind_budget_decision> <=
          sml::state<state_bind_source_decision>
          + sml::completion<detail::bind_window_runtime>
          [ guard::guard_source_map_succeeded{} ]
          / action::effect_scan_layer_plan
      , sml::state<state_bind_error_ready> <=
          sml::state<state_bind_source_decision>
          + sml::completion<detail::bind_window_runtime>
          [ guard::guard_source_map_failed{} ]
          / action::effect_mark_source_map_failed
      , sml::state<state_passthrough_ready> <=
          sml::state<state_bind_budget_decision>
          + sml::completion<detail::bind_window_runtime>
          [ guard::guard_bind_fits_budget{} ]
          / action::effect_activate_passthrough_and_publish
      , sml::state<state_ready> <= sml::state<state_bind_budget_decision>
          + sml::completion<detail::bind_window_runtime>
          [ guard::guard_bind_requires_streaming{} ]
          / action::effect_activate_streaming_and_publish
      , sml::state<state_bind_error_ready> <=
          sml::state<state_bind_budget_decision>
          + sml::completion<detail::bind_window_runtime>
          [ guard::guard_bind_budget_too_small{} ]
          / action::effect_mark_budget_too_small

      //------------------------------------------------------------------------------//
      // bind error publication.
      , sml::state<state_bind_error_callback> <= sml::state<state_bind_error_ready>
          + sml::completion<detail::bind_window_runtime>
          [ guard::guard_bind_error_callback_present{} ]
          / action::effect_publish_bind_error
      , sml::state<state_unbound> <= sml::state<state_bind_error_ready>
          + sml::completion<detail::bind_window_runtime>
          [ guard::guard_bind_error_callback_absent{} ]
          / action::effect_record_bind_error
      , sml::state<state_unbound> <= sml::state<state_bind_error_callback>
          + sml::completion<detail::bind_window_runtime>
          / action::effect_record_bind_error

      //------------------------------------------------------------------------------//
      // bind while already bound: explicit error, state unchanged.
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<detail::bind_window_runtime>
          / action::effect_mark_already_bound_and_publish
      , sml::state<state_passthrough_ready> <= sml::state<state_passthrough_ready>
          + sml::event<detail::bind_window_runtime>
          / action::effect_mark_already_bound_and_publish

      //------------------------------------------------------------------------------//
      // acquire resolve (first dispatch): join a slot still mid-load for a
      // different layer before the settle dispatch may reuse it. The drain
      // delivers the joined completion while in state_acquire_resolved.
      , sml::state<state_acquire_resolve_decision> <= sml::state<state_ready>
          + sml::event<detail::acquire_resolve_runtime>
          / action::effect_begin_acquire_resolve
      , sml::state<state_acquire_resolved> <=
          sml::state<state_acquire_resolve_decision>
          + sml::completion<detail::acquire_resolve_runtime>
          [ guard::guard_resolve_layer_out_of_range{} ]
          / action::effect_mark_resolve_out_of_range
      , sml::state<state_acquire_resolved> <=
          sml::state<state_acquire_resolve_decision>
          + sml::completion<detail::acquire_resolve_runtime>
          [ guard::guard_resolve_slot_busy_other_layer{} ]
          / action::effect_require_busy_slot
      , sml::state<state_acquire_resolved> <=
          sml::state<state_acquire_resolve_decision>
          + sml::completion<detail::acquire_resolve_runtime>
          [ guard::guard_resolve_slot_ready_for_target{} ]
      , sml::state<state_acquire_resolved> <= sml::state<state_acquire_resolved>
          + sml::event<emel::event::completion>
          / action::effect_commit_slot_load

      //------------------------------------------------------------------------------//
      // acquire settle (second dispatch): decide resident / loading /
      // unscheduled for the target. The pending state is where the drain
      // delivers the required slot completion.
      , sml::state<state_acquire_decision> <= sml::state<state_acquire_resolved>
          + sml::event<detail::acquire_runtime>
      , sml::state<state_acquire_pending> <= sml::state<state_acquire_decision>
          + sml::completion<detail::acquire_runtime>
          [ guard::guard_acquire_layer_out_of_range{} ]
          / action::effect_mark_acquire_out_of_range
      , sml::state<state_acquire_pending> <= sml::state<state_acquire_decision>
          + sml::completion<detail::acquire_runtime>
          [ guard::guard_acquire_layer_resident{} ]
      , sml::state<state_acquire_pending> <= sml::state<state_acquire_decision>
          + sml::completion<detail::acquire_runtime>
          [ guard::guard_acquire_layer_loading{} ]
          / action::effect_require_layer_completion
      , sml::state<state_acquire_pending> <= sml::state<state_acquire_decision>
          + sml::completion<detail::acquire_runtime>
          [ guard::guard_acquire_layer_unscheduled{} ]
          / action::effect_submit_and_require_layer

      //------------------------------------------------------------------------------//
      // acquire finish: route on the committed slot state, publish, advance.
      , sml::state<state_acquire_publish_decision> <=
          sml::state<state_acquire_pending>
          + sml::event<detail::acquire_publish_runtime>
      , sml::state<state_acquire_advance_decision> <=
          sml::state<state_acquire_publish_decision>
          + sml::completion<detail::acquire_publish_runtime>
          [ guard::guard_acquire_result_ready{} ]
          / action::effect_stage_acquire_result
      , sml::state<state_acquire_error_ready> <=
          sml::state<state_acquire_publish_decision>
          + sml::completion<detail::acquire_publish_runtime>
          [ guard::guard_acquire_error_pending{} ]
      , sml::state<state_acquire_error_ready> <=
          sml::state<state_acquire_publish_decision>
          + sml::completion<detail::acquire_publish_runtime>
          [ guard::guard_acquire_copy_failed{} ]
          / action::effect_mark_slot_copy_failed
      , sml::state<state_acquire_publish_ready> <=
          sml::state<state_acquire_advance_decision>
          + sml::completion<detail::acquire_publish_runtime>
          [ guard::guard_prefetch_ahead_needed{} ]
          / action::effect_advance_window
      , sml::state<state_acquire_publish_ready> <=
          sml::state<state_acquire_advance_decision>
          + sml::completion<detail::acquire_publish_runtime>
          [ guard::guard_prefetch_ahead_not_needed{} ]
      , sml::state<state_acquire_done_callback> <=
          sml::state<state_acquire_publish_ready>
          + sml::completion<detail::acquire_publish_runtime>
          [ guard::guard_acquire_done_callback_present{} ]
          / action::effect_publish_acquire_done
      , sml::state<state_ready> <= sml::state<state_acquire_publish_ready>
          + sml::completion<detail::acquire_publish_runtime>
          [ guard::guard_acquire_done_callback_absent{} ]
          / action::effect_record_acquire_done
      , sml::state<state_ready> <= sml::state<state_acquire_done_callback>
          + sml::completion<detail::acquire_publish_runtime>
          / action::effect_record_acquire_done
      , sml::state<state_acquire_error_callback> <=
          sml::state<state_acquire_error_ready>
          + sml::completion<detail::acquire_publish_runtime>
          [ guard::guard_acquire_error_callback_present{} ]
          / action::effect_publish_acquire_error
      , sml::state<state_ready> <= sml::state<state_acquire_error_ready>
          + sml::completion<detail::acquire_publish_runtime>
          [ guard::guard_acquire_error_callback_absent{} ]
          / action::effect_record_acquire_error
      , sml::state<state_ready> <= sml::state<state_acquire_error_callback>
          + sml::completion<detail::acquire_publish_runtime>
          / action::effect_record_acquire_error

      //------------------------------------------------------------------------------//
      // acquire outside streaming: explicit not_streaming / not_bound errors
      // published through the same three-dispatch protocol.
      , sml::state<state_passthrough_acquire_resolved> <=
          sml::state<state_passthrough_ready>
          + sml::event<detail::acquire_resolve_runtime>
          / action::effect_begin_resolve_not_streaming
      , sml::state<state_passthrough_acquire_pending> <=
          sml::state<state_passthrough_acquire_resolved>
          + sml::event<detail::acquire_runtime>
      , sml::state<state_passthrough_acquire_error_ready> <=
          sml::state<state_passthrough_acquire_pending>
          + sml::event<detail::acquire_publish_runtime>
      , sml::state<state_passthrough_acquire_error_callback> <=
          sml::state<state_passthrough_acquire_error_ready>
          + sml::completion<detail::acquire_publish_runtime>
          [ guard::guard_acquire_error_callback_present{} ]
          / action::effect_publish_acquire_error
      , sml::state<state_passthrough_ready> <=
          sml::state<state_passthrough_acquire_error_ready>
          + sml::completion<detail::acquire_publish_runtime>
          [ guard::guard_acquire_error_callback_absent{} ]
          / action::effect_record_acquire_error
      , sml::state<state_passthrough_ready> <=
          sml::state<state_passthrough_acquire_error_callback>
          + sml::completion<detail::acquire_publish_runtime>
          / action::effect_record_acquire_error
      , sml::state<state_unbound> <= sml::state<state_unbound>
          + sml::event<detail::acquire_resolve_runtime>
          / action::effect_begin_resolve_not_bound
      , sml::state<state_unbound> <= sml::state<state_unbound>
          + sml::event<detail::acquire_runtime>
      , sml::state<state_unbound> <= sml::state<state_unbound>
          + sml::event<detail::acquire_publish_runtime>
          / action::effect_publish_acquire_error

      //------------------------------------------------------------------------------//
      // completion commit: required slot loads delivered by the drain while an
      // acquire is pending, and background prefetch fires swept at the start
      // of any later dispatch.
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<emel::event::completion>
          / action::effect_commit_slot_load
      , sml::state<state_acquire_pending> <= sml::state<state_acquire_pending>
          + sml::event<emel::event::completion>
          / action::effect_commit_slot_load
      , sml::state<state_unbind_pending> <= sml::state<state_unbind_pending>
          + sml::event<emel::event::completion>
          / action::effect_commit_slot_load
      , sml::state<state_passthrough_ready> <= sml::state<state_passthrough_ready>
          + sml::event<emel::event::completion>
          / action::effect_record_stray_completion
      , sml::state<state_unbound> <= sml::state<state_unbound>
          + sml::event<emel::event::completion>
          / action::effect_record_stray_completion

      //------------------------------------------------------------------------------//
      // unbind: require every in-flight slot (the drain joins them), then the
      // finish dispatch releases the source mapping, frees slots, publishes.
      , sml::state<state_unbind_pending> <= sml::state<state_ready>
          + sml::event<detail::unbind_runtime>
          / action::effect_begin_unbind
      , sml::state<state_unbind_pending> <= sml::state<state_passthrough_ready>
          + sml::event<detail::unbind_runtime>
          / action::effect_begin_unbind
      , sml::state<state_unbind_finish_decision> <= sml::state<state_unbind_pending>
          + sml::event<detail::unbind_finish_runtime>
          / action::effect_release_source_and_reset
      , sml::state<state_unbind_publish_ready> <=
          sml::state<state_unbind_finish_decision>
          + sml::completion<detail::unbind_finish_runtime>
          [ guard::guard_unbind_release_succeeded{} ]
      , sml::state<state_unbind_error_ready> <=
          sml::state<state_unbind_finish_decision>
          + sml::completion<detail::unbind_finish_runtime>
          [ guard::guard_unbind_release_failed{} ]
          / action::effect_mark_unbind_release_failed
      , sml::state<state_unbind_done_callback> <=
          sml::state<state_unbind_publish_ready>
          + sml::completion<detail::unbind_finish_runtime>
          [ guard::guard_unbind_done_callback_present{} ]
          / action::effect_publish_unbind_done
      , sml::state<state_unbound> <= sml::state<state_unbind_publish_ready>
          + sml::completion<detail::unbind_finish_runtime>
          [ guard::guard_unbind_done_callback_absent{} ]
          / action::effect_record_unbind_done
      , sml::state<state_unbound> <= sml::state<state_unbind_done_callback>
          + sml::completion<detail::unbind_finish_runtime>
          / action::effect_record_unbind_done
      , sml::state<state_unbind_error_callback> <=
          sml::state<state_unbind_error_ready>
          + sml::completion<detail::unbind_finish_runtime>
          [ guard::guard_unbind_error_callback_present{} ]
          / action::effect_publish_unbind_error
      , sml::state<state_unbound> <= sml::state<state_unbind_error_ready>
          + sml::completion<detail::unbind_finish_runtime>
          [ guard::guard_unbind_error_callback_absent{} ]
          / action::effect_record_unbind_error
      , sml::state<state_unbound> <= sml::state<state_unbind_error_callback>
          + sml::completion<detail::unbind_finish_runtime>
          / action::effect_record_unbind_error
      , sml::state<state_unbound> <= sml::state<state_unbound>
          + sml::event<detail::unbind_runtime>
          / action::effect_mark_unbind_not_bound
      , sml::state<state_unbound> <= sml::state<state_unbound>
          + sml::event<detail::unbind_finish_runtime>
          / action::effect_publish_unbind_error

      //------------------------------------------------------------------------------//
      // Unexpected event handling: hold position, mark internal error.
      , sml::state<state_unbound> <= sml::state<state_unbound>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_passthrough_ready> <= sml::state<state_passthrough_ready>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_acquire_resolved> <= sml::state<state_acquire_resolved>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_passthrough_acquire_resolved> <=
          sml::state<state_passthrough_acquire_resolved>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_acquire_pending> <= sml::state<state_acquire_pending>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_passthrough_acquire_pending> <=
          sml::state<state_passthrough_acquire_pending>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
      , sml::state<state_unbind_pending> <= sml::state<state_unbind_pending>
          + sml::unexpected_event<sml::_> / action::effect_on_unexpected
    );
    // clang-format on
  }
};

// Typed wrappers drive the two-dispatch protocols unconditionally; runtime
// carriers thread the co_sm scheduler to effects so context stays free of
// dispatch-local pointers.
struct sm : public emel::co_sm<
                model, action::context,
                emel::policy::external_completion_co_policy<detail::k_max_window_slots>> {
  using base_type = emel::co_sm<
      model, action::context,
      emel::policy::external_completion_co_policy<detail::k_max_window_slots>>;
  using base_type::base_type;
  using base_type::is;
  using base_type::process_event;
  using base_type::visit_current_states;

  bool process_event(const event::bind_window &ev) {
    detail::bind_attempt_status status{};
    detail::bind_window_runtime runtime{ev, status, this->scheduler()};
    const bool accepted = base_type::process_event(runtime);
    return accepted && status.ok;
  }

  bool process_event(const event::acquire_layer_window &ev) {
    detail::acquire_attempt_status status{};
    detail::acquire_resolve_runtime resolve{ev, status, this->scheduler()};
    const bool resolved = base_type::process_event(resolve);
    detail::acquire_runtime settle{ev, status, this->scheduler()};
    const bool settled = base_type::process_event(settle);
    detail::acquire_publish_runtime finish{ev, status, this->scheduler()};
    const bool published = base_type::process_event(finish);
    return resolved && settled && published && status.ok;
  }

  bool process_event(const event::unbind_window &ev) {
    detail::unbind_attempt_status status{};
    detail::unbind_runtime begin{ev, status, this->scheduler()};
    const bool began = base_type::process_event(begin);
    detail::unbind_finish_runtime finish{ev, status};
    const bool finished = base_type::process_event(finish);
    return began && finished && status.ok;
  }
};

} // namespace emel::model::tensor::window
