#pragma once

#include "emel/emel.h"
#include "emel/kernel/metal/actions.hpp"
#include "emel/kernel/metal/context.hpp"
#include "emel/kernel/metal/errors.hpp"
#include "emel/kernel/metal/events.hpp"
#include "emel/kernel/metal/guards.hpp"
#include "emel/sm.hpp"

// Metal kernel actor: dispatch row per supported op variant, with the
// runtime-behavior choice (variant) fully modeled in the transition table
// guards. Unsupported ops and not-ready Metal hosts land on the explicit
// reject rows or the unexpected-event row - never a silent drop and never a
// fallback to another backend.
namespace emel::kernel::metal {

struct ready {};

struct model {
  auto operator()() const {
    namespace sml = stateforward::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Dispatch event.
        sml::state<ready> <= *sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_request>
                 / action::exec_dispatch

      //------------------------------------------------------------------------------//
      // op_mul_mat: f32, f16, q8_0 variants (most specific first).
      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mul_mat>
                 [ guard::valid_op_mul_mat_q8_0{} ]
                 / action::exec_op_mul_mat_q8_0

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mul_mat>
                 [ guard::valid_op_mul_mat_f16{} ]
                 / action::exec_op_mul_mat_f16

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mul_mat>
                 [ guard::valid_op_mul_mat_f32{} ]
                 / action::exec_op_mul_mat_f32

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mul_mat>
                 [ guard::invalid_op_mul_mat{} ]
                 / action::reject_invalid_op_mul_mat

      //------------------------------------------------------------------------------//
      // op_add: equal-shape and broadcast-row variants.
      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_add>
                 [ guard::valid_op_add_equal{} ]
                 / action::exec_op_add_equal

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_add>
                 [ guard::valid_op_add_broadcast_row{} ]
                 / action::exec_op_add_broadcast_row

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_add>
                 [ guard::invalid_op_add{} ]
                 / action::reject_invalid_op_add

      //------------------------------------------------------------------------------//
      // op_unary: subop variants.
      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_unary>
                 [ guard::valid_op_unary_subop<::emel::kernel::event::unary_subop::abs>{} ]
                 / action::exec_op_unary_subop<::emel::kernel::event::unary_subop::abs>{}

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_unary>
                 [ guard::valid_op_unary_subop<::emel::kernel::event::unary_subop::neg>{} ]
                 / action::exec_op_unary_subop<::emel::kernel::event::unary_subop::neg>{}

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_unary>
                 [ guard::valid_op_unary_subop<::emel::kernel::event::unary_subop::tanh>{} ]
                 / action::exec_op_unary_subop<::emel::kernel::event::unary_subop::tanh>{}

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_unary>
                 [ guard::valid_op_unary_subop<::emel::kernel::event::unary_subop::elu>{} ]
                 / action::exec_op_unary_subop<::emel::kernel::event::unary_subop::elu>{}

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_unary>
                 [ guard::valid_op_unary_subop<::emel::kernel::event::unary_subop::relu>{} ]
                 / action::exec_op_unary_subop<::emel::kernel::event::unary_subop::relu>{}

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_unary>
                 [ guard::valid_op_unary_subop<::emel::kernel::event::unary_subop::gelu>{} ]
                 / action::exec_op_unary_subop<::emel::kernel::event::unary_subop::gelu>{}

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_unary>
                 [ guard::valid_op_unary_subop<::emel::kernel::event::unary_subop::silu>{} ]
                 / action::exec_op_unary_subop<::emel::kernel::event::unary_subop::silu>{}

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_unary>
                 [ guard::valid_op_unary_subop<::emel::kernel::event::unary_subop::exp>{} ]
                 / action::exec_op_unary_subop<::emel::kernel::event::unary_subop::exp>{}

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_unary>
                 [ guard::invalid_op_unary{} ]
                 / action::reject_invalid_op_unary

      //------------------------------------------------------------------------------//
      // op_im2col: f32 and f16 column variants.
      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_im2col>
                 [ guard::valid_op_im2col_f32{} ]
                 / action::exec_op_im2col_f32

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_im2col>
                 [ guard::valid_op_im2col_f16{} ]
                 / action::exec_op_im2col_f16

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_im2col>
                 [ guard::invalid_op_im2col{} ]
                 / action::reject_invalid_op_im2col

      //------------------------------------------------------------------------------//
      // op_conv_transpose_1d: f32 and f16 weight variants.
      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_transpose_1d>
                 [ guard::valid_op_conv_transpose_1d_f32{} ]
                 / action::exec_op_conv_transpose_1d_f32

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_transpose_1d>
                 [ guard::valid_op_conv_transpose_1d_f16{} ]
                 / action::exec_op_conv_transpose_1d_f16

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_transpose_1d>
                 [ guard::invalid_op_conv_transpose_1d{} ]
                 / action::reject_invalid_op_conv_transpose_1d

      //------------------------------------------------------------------------------//
      // op_get_rows: f32 and f16 codebook variants.
      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_get_rows>
                 [ guard::valid_op_get_rows_f32{} ]
                 / action::exec_op_get_rows_f32

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_get_rows>
                 [ guard::valid_op_get_rows_f16{} ]
                 / action::exec_op_get_rows_f16

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_get_rows>
                 [ guard::invalid_op_get_rows{} ]
                 / action::reject_invalid_op_get_rows

      //------------------------------------------------------------------------------//
      // Unsupported ops: explicit reject instead of a silent drop.
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
                 / action::reject_unsupported_op
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::is;
  using base_type::visit_current_states;

  sm() = default;

  bool process_event(const ::emel::kernel::event::dispatch &ev) {
    event::dispatch_ctx ctx{};
    const event::dispatch_request dispatch{ev, ctx};
    return process_dispatch_event(dispatch);
  }

  template <class event_type>
    requires(::emel::kernel::is_op_event_v<event_type>)
  bool process_event(const event_type &ev) {
    event::dispatch_ctx ctx{};
    using dispatch_event_type = event::dispatch_event_for_t<event_type>;
    const dispatch_event_type dispatch{ev, ctx};
    return process_dispatch_event(dispatch);
  }

  uint64_t metal_dispatch_count() const noexcept {
    return this->context_.metal_dispatch_count;
  }

  uint64_t dispatch_generation() const noexcept {
    return this->context_.dispatch_generation;
  }

  bool metal_available() const noexcept {
    return this->context_.metal_available;
  }

private:
  template <class dispatch_event_type>
  bool process_dispatch_event(const dispatch_event_type &ev) {
    const bool accepted = base_type::process_event(ev);
    return accepted &&
           ev.ctx.err == static_cast<int32_t>(emel::error::cast(error::none));
  }
};

} // namespace emel::kernel::metal
