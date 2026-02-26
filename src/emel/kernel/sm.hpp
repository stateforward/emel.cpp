#pragma once


// benchmark: scaffold
// docs: disabled

#include "emel/emel.h"
#include "emel/kernel/errors.hpp"
#include "emel/kernel/actions.hpp"
#include "emel/kernel/context.hpp"
#include "emel/sm.hpp"
#include "emel/kernel/events.hpp"
#include "emel/kernel/guards.hpp"

namespace emel::kernel {

struct ready {};
struct primary_dispatch {};
struct primary_decision {};
struct secondary_dispatch {};
struct secondary_decision {};
struct tertiary_dispatch {};
struct tertiary_decision {};
struct dispatch_decision {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Request validation.
        sml::state<primary_dispatch> <= *sml::state<ready> + sml::event<event::dispatch_scaffold>
                 [ guard::valid_dispatch{} ]
                 / action::begin_dispatch

      , sml::state<primary_dispatch> <= sml::state<ready> + sml::event<event::dispatch_op>
                 [ guard::valid_dispatch{} ]
                 / action::begin_dispatch

      //------------------------------------------------------------------------------//
      // Primary backend phase.
      , sml::state<primary_decision> <= sml::state<primary_dispatch> + sml::completion<event::dispatch_scaffold>
                 / action::request_primary

      , sml::state<primary_decision> <= sml::state<primary_dispatch> + sml::completion<event::dispatch_op>
                 / action::request_primary

      , sml::state<ready> <= sml::state<primary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::primary_done{} ]
                 / action::dispatch_done

      , sml::state<ready> <= sml::state<primary_decision> + sml::completion<event::dispatch_op>
                 [ guard::primary_done{} ]
                 / action::dispatch_done

      , sml::state<secondary_dispatch> <= sml::state<primary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::primary_unsupported{} ]

      , sml::state<secondary_dispatch> <= sml::state<primary_decision> + sml::completion<event::dispatch_op>
                 [ guard::primary_unsupported{} ]

      , sml::state<dispatch_decision> <= sml::state<primary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::primary_failed{} ]

      , sml::state<dispatch_decision> <= sml::state<primary_decision> + sml::completion<event::dispatch_op>
                 [ guard::primary_failed{} ]

      //------------------------------------------------------------------------------//
      // Secondary backend phase.
      , sml::state<secondary_decision> <= sml::state<secondary_dispatch> + sml::completion<event::dispatch_scaffold>
                 / action::request_secondary

      , sml::state<secondary_decision> <= sml::state<secondary_dispatch> + sml::completion<event::dispatch_op>
                 / action::request_secondary

      , sml::state<ready> <= sml::state<secondary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::secondary_done{} ]
                 / action::dispatch_done

      , sml::state<ready> <= sml::state<secondary_decision> + sml::completion<event::dispatch_op>
                 [ guard::secondary_done{} ]
                 / action::dispatch_done

      , sml::state<tertiary_dispatch> <= sml::state<secondary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::secondary_unsupported{} ]

      , sml::state<tertiary_dispatch> <= sml::state<secondary_decision> + sml::completion<event::dispatch_op>
                 [ guard::secondary_unsupported{} ]

      , sml::state<dispatch_decision> <= sml::state<secondary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::secondary_failed{} ]

      , sml::state<dispatch_decision> <= sml::state<secondary_decision> + sml::completion<event::dispatch_op>
                 [ guard::secondary_failed{} ]

      //------------------------------------------------------------------------------//
      // Tertiary backend phase.
      , sml::state<tertiary_decision> <= sml::state<tertiary_dispatch> + sml::completion<event::dispatch_scaffold>
                 / action::request_tertiary

      , sml::state<tertiary_decision> <= sml::state<tertiary_dispatch> + sml::completion<event::dispatch_op>
                 / action::request_tertiary

      , sml::state<ready> <= sml::state<tertiary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::tertiary_done{} ]
                 / action::dispatch_done

      , sml::state<ready> <= sml::state<tertiary_decision> + sml::completion<event::dispatch_op>
                 [ guard::tertiary_done{} ]
                 / action::dispatch_done

      , sml::state<dispatch_decision> <= sml::state<tertiary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::tertiary_unsupported{} ]
                 / action::mark_unsupported

      , sml::state<dispatch_decision> <= sml::state<tertiary_decision> + sml::completion<event::dispatch_op>
                 [ guard::tertiary_unsupported{} ]
                 / action::mark_unsupported

      , sml::state<dispatch_decision> <= sml::state<tertiary_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::tertiary_failed{} ]

      , sml::state<dispatch_decision> <= sml::state<tertiary_decision> + sml::completion<event::dispatch_op>
                 [ guard::tertiary_failed{} ]

      //------------------------------------------------------------------------------//
      // Finalization.
      , sml::state<ready> <= sml::state<dispatch_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::phase_ok{} ]
                 / action::dispatch_done

      , sml::state<ready> <= sml::state<dispatch_decision> + sml::completion<event::dispatch_op>
                 [ guard::phase_ok{} ]
                 / action::dispatch_done

      , sml::state<ready> <= sml::state<dispatch_decision> + sml::completion<event::dispatch_scaffold>
                 [ guard::phase_failed{} ]
                 / action::dispatch_error

      , sml::state<ready> <= sml::state<dispatch_decision> + sml::completion<event::dispatch_op>
                 [ guard::phase_failed{} ]
                 / action::dispatch_error

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<dispatch_decision> <= sml::state<primary_dispatch> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<dispatch_decision> <= sml::state<primary_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<dispatch_decision> <= sml::state<secondary_dispatch> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<dispatch_decision> <= sml::state<secondary_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<dispatch_decision> <= sml::state<tertiary_dispatch> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<dispatch_decision> <= sml::state<tertiary_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected

      , sml::state<ready> <= sml::state<dispatch_decision> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm_with_context<model, action::context> {
  using base_type = emel::sm_with_context<model, action::context>;
  using base_type::base_type;

  bool process_event(const event::scaffold & ev) {
    event::scaffold_ctx ctx{};
    event::dispatch_scaffold evt{ev, ctx};
    const bool accepted = base_type::process_event(evt);
    return accepted && ctx.err == static_cast<int32_t>(emel::error::cast(error::none));
  }

  template <class event_type>
    requires(::emel::kernel::is_op_event_v<event_type>)
  bool process_event(const event_type & ev) {
    event::scaffold_ctx ctx{};
    const event::dispatch_op evt{
      .request = &ev,
      .dispatch_primary = &dispatch_primary<event_type>,
      .dispatch_secondary = &dispatch_secondary<event_type>,
      .dispatch_tertiary = &dispatch_tertiary<event_type>,
      .ctx = ctx,
    };
    const bool accepted = base_type::process_event(evt);
    return accepted && ctx.err == static_cast<int32_t>(emel::error::cast(error::none));
  }

 private:
  template <class event_type>
  static bool dispatch_primary(action::context & ctx, const void * request) {
    return ctx.x86_64_actor.process_event(*static_cast<const event_type *>(request));
  }

  template <class event_type>
  static bool dispatch_secondary(action::context & ctx, const void * request) {
    return ctx.aarch64_actor.process_event(*static_cast<const event_type *>(request));
  }

  template <class event_type>
  static bool dispatch_tertiary(action::context & ctx, const void * request) {
    return ctx.wasm_actor.process_event(*static_cast<const event_type *>(request));
  }
};

using Kernel = sm;

}  // namespace emel::kernel
