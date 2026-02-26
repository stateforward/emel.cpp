#pragma once

// benchmark: scaffold
// docs: disabled

#include "emel/emel.h"
#include "emel/kernel/vulkan/actions.hpp"
#include "emel/kernel/vulkan/events.hpp"
#include "emel/kernel/vulkan/guards.hpp"
#include "emel/kernel/event_traits.hpp"
#include "emel/kernel/op_list.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::vulkan {

struct ready {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Scaffold event.
        sml::state<ready> <= *sml::state<ready> +
               sml::event<::emel::kernel::vulkan::event::dispatch_scaffold>
                 / action::run_scaffold

      //------------------------------------------------------------------------------//
      // Explicit op transitions.
#define EMEL_KERNEL_DEFINE_OP_TRANSITIONS(op_name) \
      , sml::state<ready> <= sml::state<ready> + \
               sml::event<::emel::kernel::vulkan::event::dispatch_##op_name> \
                 [ guard::valid_##op_name{} ] \
                 / action::run_##op_name \
      , sml::state<ready> <= sml::state<ready> + \
               sml::event<::emel::kernel::vulkan::event::dispatch_##op_name> \
                 [ guard::invalid_##op_name{} ] \
                 / action::reject_invalid_##op_name
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DEFINE_OP_TRANSITIONS)
#undef EMEL_KERNEL_DEFINE_OP_TRANSITIONS

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm_with_context<model, action::context> {
  using base_type = emel::sm_with_context<model, action::context>;
  using base_type::base_type;

  bool process_event(const ::emel::kernel::event::scaffold & ev) {
    event::dispatch_ctx ctx{};
    const event::dispatch_scaffold dispatch{ev, ctx};
    return process_dispatch_event(dispatch);
  }

  template <class event_type>
    requires(::emel::kernel::is_op_event_v<event_type>)
  bool process_event(const event_type & ev) {
    event::dispatch_ctx ctx{};
    using dispatch_event_type = event::dispatch_event_for_t<event_type>;
    const dispatch_event_type dispatch{ev, ctx};
    return process_dispatch_event(dispatch);
  }

  int32_t last_error() const noexcept {
    return last_error_;
  }

 private:
  template <class dispatch_event_type>
  bool process_dispatch_event(const dispatch_event_type & ev) {
    const bool accepted = base_type::process_event(ev);
    last_error_ = ev.ctx.err;
    return accepted && ev.ctx.err == EMEL_OK;
  }

  int32_t last_error_ = EMEL_OK;
};

}  // namespace emel::kernel::vulkan
