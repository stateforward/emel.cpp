#pragma once
// benchmark: scaffold

#include "emel/kernel/cq/actions.hpp"
#include "emel/kernel/cq/guards.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::cq {

struct state_ready {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;
    // clang-format off
    return sml::make_transition_table(
        sml::state<state_ready> <= *sml::state<state_ready>
          + sml::event<event::quantize_a8>
          [ guard::guard_quantize_a8{} ] / action::effect_quantize_a8{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::prepare_codebook_q4>
          [ guard::guard_prepare_codebook_q4{} ]
          / action::effect_prepare_codebook_q4{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::prepare_q4>
          [ guard::guard_prepare_q4{} ] / action::effect_prepare_q4{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_prepared_avx2_q4>
          [ guard::guard_execute_prepared_avx2_q4{} ]
          / action::effect_execute_prepared_avx2_q4{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_prepared_avx2_batch4_q4>
          [ guard::guard_execute_prepared_avx2_batch4_q4{} ]
          / action::effect_execute_prepared_avx2_batch4_q4{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_prepared_avx2_rows_q4>
          [ guard::guard_execute_prepared_avx2_rows_q4{} ]
          / action::effect_execute_prepared_avx2_rows_q4{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_prepared_dequant_q4>
          [ guard::guard_execute_prepared_dequant_q4{} ]
          / action::effect_execute_prepared_dequant_q4{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_avx2_q2>
          [ guard::guard_execute_avx2<2u>{} ] / action::effect_execute_avx2<2u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_avx2_q3>
          [ guard::guard_execute_avx2<3u>{} ] / action::effect_execute_avx2<3u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_avx2_q4>
          [ guard::guard_execute_avx2<4u>{} ] / action::effect_execute_avx2<4u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_scalar_q2>
          [ guard::guard_execute_scalar<2u>{} ] / action::effect_execute_scalar<2u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_scalar_q3>
          [ guard::guard_execute_scalar<3u>{} ] / action::effect_execute_scalar<3u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_scalar_q4>
          [ guard::guard_execute_scalar<4u>{} ] / action::effect_execute_scalar<4u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_scalar_ternary>
          [ guard::guard_execute_scalar<5u>{} ] / action::effect_execute_scalar<5u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_scalar_rows_q2>
          [ guard::guard_execute_scalar_rows<2u>{} ] / action::effect_execute_scalar_rows<2u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_scalar_rows_q3>
          [ guard::guard_execute_scalar_rows<3u>{} ] / action::effect_execute_scalar_rows<3u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_scalar_rows_q4>
          [ guard::guard_execute_scalar_rows<4u>{} ] / action::effect_execute_scalar_rows<4u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_scalar_dequant_q2>
          [ guard::guard_execute_scalar_dequant<2u>{} ] / action::effect_execute_scalar_dequant<2u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_scalar_dequant_q3>
          [ guard::guard_execute_scalar_dequant<3u>{} ] / action::effect_execute_scalar_dequant<3u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::execute_scalar_dequant_q4>
          [ guard::guard_execute_scalar_dequant<4u>{} ] / action::effect_execute_scalar_dequant<4u>{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::capture_diagnostics>
          / action::effect_capture_diagnostics{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::capture_prepared_diagnostics>
          / action::effect_capture_prepared_diagnostics{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::capture_a8_diagnostics>
          / action::effect_capture_a8_diagnostics{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::configure_timing>
          / action::effect_configure_timing{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::event<event::capture_timing>
          / action::effect_capture_timing{}
      , sml::state<state_ready> <= sml::state<state_ready>
          + sml::unexpected_event<sml::_>
          / action::effect_on_unexpected{}
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;
  sm() = default;
  sm(const sm &) = delete;
  sm &operator=(const sm &) = delete;

  template <class event_type> bool process_event(const event_type &ev) {
    if constexpr (requires { ev.result.accepted; })
      ev.result.accepted = false;
    const bool handled = base_type::process_event(ev);
    if constexpr (requires { ev.result.accepted; })
      return handled && ev.result.accepted;
    return handled;
  }
};

} // namespace emel::kernel::cq
