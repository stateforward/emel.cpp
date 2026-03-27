#pragma once

// benchmark: kernel
#include "emel/emel.h"
#include "emel/kernel/errors.hpp"
#include "emel/kernel/metal/actions.hpp"
#include "emel/kernel/metal/events.hpp"
#include "emel/kernel/metal/guards.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/sm.hpp"

namespace emel::kernel::metal {

struct ready {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
      // Dispatch event.
        sml::state<ready> <= *sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_request>
                 / action::exec_dispatch

      //------------------------------------------------------------------------------//
      // Explicit op transitions.
      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_dup>
                 [ guard::valid_op_dup{} ]
                 / action::exec_op_dup

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_dup>
                 [ guard::invalid_op_dup{} ]
                 / action::reject_invalid_op_dup

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_add>
                 [ guard::valid_op_add{} ]
                 / action::exec_op_add

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_add>
                 [ guard::invalid_op_add{} ]
                 / action::reject_invalid_op_add

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_add_id>
                 [ guard::valid_op_add_id{} ]
                 / action::exec_op_add_id

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_add_id>
                 [ guard::invalid_op_add_id{} ]
                 / action::reject_invalid_op_add_id

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_add1>
                 [ guard::valid_op_add1{} ]
                 / action::exec_op_add1

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_add1>
                 [ guard::invalid_op_add1{} ]
                 / action::reject_invalid_op_add1

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_acc>
                 [ guard::valid_op_acc{} ]
                 / action::exec_op_acc

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_acc>
                 [ guard::invalid_op_acc{} ]
                 / action::reject_invalid_op_acc

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_sub>
                 [ guard::valid_op_sub{} ]
                 / action::exec_op_sub

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_sub>
                 [ guard::invalid_op_sub{} ]
                 / action::reject_invalid_op_sub

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mul>
                 [ guard::valid_op_mul{} ]
                 / action::exec_op_mul

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mul>
                 [ guard::invalid_op_mul{} ]
                 / action::reject_invalid_op_mul

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_div>
                 [ guard::valid_op_div{} ]
                 / action::exec_op_div

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_div>
                 [ guard::invalid_op_div{} ]
                 / action::reject_invalid_op_div

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_sqr>
                 [ guard::valid_op_sqr{} ]
                 / action::exec_op_sqr

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_sqr>
                 [ guard::invalid_op_sqr{} ]
                 / action::reject_invalid_op_sqr

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_sqrt>
                 [ guard::valid_op_sqrt{} ]
                 / action::exec_op_sqrt

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_sqrt>
                 [ guard::invalid_op_sqrt{} ]
                 / action::reject_invalid_op_sqrt

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_log>
                 [ guard::valid_op_log{} ]
                 / action::exec_op_log

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_log>
                 [ guard::invalid_op_log{} ]
                 / action::reject_invalid_op_log

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_sin>
                 [ guard::valid_op_sin{} ]
                 / action::exec_op_sin

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_sin>
                 [ guard::invalid_op_sin{} ]
                 / action::reject_invalid_op_sin

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_cos>
                 [ guard::valid_op_cos{} ]
                 / action::exec_op_cos

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_cos>
                 [ guard::invalid_op_cos{} ]
                 / action::reject_invalid_op_cos

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_sum>
                 [ guard::valid_op_sum{} ]
                 / action::exec_op_sum

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_sum>
                 [ guard::invalid_op_sum{} ]
                 / action::reject_invalid_op_sum

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_sum_rows>
                 [ guard::valid_op_sum_rows{} ]
                 / action::exec_op_sum_rows

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_sum_rows>
                 [ guard::invalid_op_sum_rows{} ]
                 / action::reject_invalid_op_sum_rows

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_cumsum>
                 [ guard::valid_op_cumsum{} ]
                 / action::exec_op_cumsum

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_cumsum>
                 [ guard::invalid_op_cumsum{} ]
                 / action::reject_invalid_op_cumsum

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mean>
                 [ guard::valid_op_mean{} ]
                 / action::exec_op_mean

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mean>
                 [ guard::invalid_op_mean{} ]
                 / action::reject_invalid_op_mean

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_argmax>
                 [ guard::valid_op_argmax{} ]
                 / action::exec_op_argmax

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_argmax>
                 [ guard::invalid_op_argmax{} ]
                 / action::reject_invalid_op_argmax

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_count_equal>
                 [ guard::valid_op_count_equal{} ]
                 / action::exec_op_count_equal

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_count_equal>
                 [ guard::invalid_op_count_equal{} ]
                 / action::reject_invalid_op_count_equal

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_repeat>
                 [ guard::valid_op_repeat{} ]
                 / action::exec_op_repeat

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_repeat>
                 [ guard::invalid_op_repeat{} ]
                 / action::reject_invalid_op_repeat

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_repeat_back>
                 [ guard::valid_op_repeat_back{} ]
                 / action::exec_op_repeat_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_repeat_back>
                 [ guard::invalid_op_repeat_back{} ]
                 / action::reject_invalid_op_repeat_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_concat>
                 [ guard::valid_op_concat{} ]
                 / action::exec_op_concat

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_concat>
                 [ guard::invalid_op_concat{} ]
                 / action::reject_invalid_op_concat

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_silu_back>
                 [ guard::valid_op_silu_back{} ]
                 / action::exec_op_silu_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_silu_back>
                 [ guard::invalid_op_silu_back{} ]
                 / action::reject_invalid_op_silu_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_norm>
                 [ guard::valid_op_norm{} ]
                 / action::exec_op_norm

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_norm>
                 [ guard::invalid_op_norm{} ]
                 / action::reject_invalid_op_norm

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_rms_norm>
                 [ guard::valid_op_rms_norm{} ]
                 / action::exec_op_rms_norm

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_rms_norm>
                 [ guard::invalid_op_rms_norm{} ]
                 / action::reject_invalid_op_rms_norm

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_rms_norm_back>
                 [ guard::valid_op_rms_norm_back{} ]
                 / action::exec_op_rms_norm_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_rms_norm_back>
                 [ guard::invalid_op_rms_norm_back{} ]
                 / action::reject_invalid_op_rms_norm_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_group_norm>
                 [ guard::valid_op_group_norm{} ]
                 / action::exec_op_group_norm

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_group_norm>
                 [ guard::invalid_op_group_norm{} ]
                 / action::reject_invalid_op_group_norm

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_l2_norm>
                 [ guard::valid_op_l2_norm{} ]
                 / action::exec_op_l2_norm

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_l2_norm>
                 [ guard::invalid_op_l2_norm{} ]
                 / action::reject_invalid_op_l2_norm

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mul_mat>
                 [ guard::valid_op_mul_mat{} ]
                 / action::exec_op_mul_mat

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mul_mat>
                 [ guard::invalid_op_mul_mat{} ]
                 / action::reject_invalid_op_mul_mat

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mul_mat_argmax>
                 [ guard::valid_op_mul_mat_argmax{} ]
                 / action::exec_op_mul_mat_argmax

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mul_mat_argmax>
                 [ guard::invalid_op_mul_mat_argmax{} ]
                 / action::reject_invalid_op_mul_mat_argmax

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mul_mat_id>
                 [ guard::valid_op_mul_mat_id{} ]
                 / action::exec_op_mul_mat_id

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_mul_mat_id>
                 [ guard::invalid_op_mul_mat_id{} ]
                 / action::reject_invalid_op_mul_mat_id

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_out_prod>
                 [ guard::valid_op_out_prod{} ]
                 / action::exec_op_out_prod

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_out_prod>
                 [ guard::invalid_op_out_prod{} ]
                 / action::reject_invalid_op_out_prod

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_scale>
                 [ guard::valid_op_scale{} ]
                 / action::exec_op_scale

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_scale>
                 [ guard::invalid_op_scale{} ]
                 / action::reject_invalid_op_scale

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_set>
                 [ guard::valid_op_set{} ]
                 / action::exec_op_set

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_set>
                 [ guard::invalid_op_set{} ]
                 / action::reject_invalid_op_set

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_cpy>
                 [ guard::valid_op_cpy{} ]
                 / action::exec_op_cpy

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_cpy>
                 [ guard::invalid_op_cpy{} ]
                 / action::reject_invalid_op_cpy

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_cont>
                 [ guard::valid_op_cont{} ]
                 / action::exec_op_cont

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_cont>
                 [ guard::invalid_op_cont{} ]
                 / action::reject_invalid_op_cont

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_reshape>
                 [ guard::valid_op_reshape{} ]
                 / action::exec_op_reshape

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_reshape>
                 [ guard::invalid_op_reshape{} ]
                 / action::reject_invalid_op_reshape

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_view>
                 [ guard::valid_op_view{} ]
                 / action::exec_op_view

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_view>
                 [ guard::invalid_op_view{} ]
                 / action::reject_invalid_op_view

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_permute>
                 [ guard::valid_op_permute{} ]
                 / action::exec_op_permute

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_permute>
                 [ guard::invalid_op_permute{} ]
                 / action::reject_invalid_op_permute

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_transpose>
                 [ guard::valid_op_transpose{} ]
                 / action::exec_op_transpose

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_transpose>
                 [ guard::invalid_op_transpose{} ]
                 / action::reject_invalid_op_transpose

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_get_rows>
                 [ guard::valid_op_get_rows{} ]
                 / action::exec_op_get_rows

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_get_rows>
                 [ guard::invalid_op_get_rows{} ]
                 / action::reject_invalid_op_get_rows

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_get_rows_back>
                 [ guard::valid_op_get_rows_back{} ]
                 / action::exec_op_get_rows_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_get_rows_back>
                 [ guard::invalid_op_get_rows_back{} ]
                 / action::reject_invalid_op_get_rows_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_set_rows>
                 [ guard::valid_op_set_rows{} ]
                 / action::exec_op_set_rows

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_set_rows>
                 [ guard::invalid_op_set_rows{} ]
                 / action::reject_invalid_op_set_rows

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_diag>
                 [ guard::valid_op_diag{} ]
                 / action::exec_op_diag

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_diag>
                 [ guard::invalid_op_diag{} ]
                 / action::reject_invalid_op_diag

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_diag_mask_inf>
                 [ guard::valid_op_diag_mask_inf{} ]
                 / action::exec_op_diag_mask_inf

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_diag_mask_inf>
                 [ guard::invalid_op_diag_mask_inf{} ]
                 / action::reject_invalid_op_diag_mask_inf

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_diag_mask_zero>
                 [ guard::valid_op_diag_mask_zero{} ]
                 / action::exec_op_diag_mask_zero

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_diag_mask_zero>
                 [ guard::invalid_op_diag_mask_zero{} ]
                 / action::reject_invalid_op_diag_mask_zero

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_soft_max>
                 [ guard::valid_op_soft_max{} ]
                 / action::exec_op_soft_max

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_soft_max>
                 [ guard::invalid_op_soft_max{} ]
                 / action::reject_invalid_op_soft_max

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_soft_max_back>
                 [ guard::valid_op_soft_max_back{} ]
                 / action::exec_op_soft_max_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_soft_max_back>
                 [ guard::invalid_op_soft_max_back{} ]
                 / action::reject_invalid_op_soft_max_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_rope>
                 [ guard::valid_op_rope{} ]
                 / action::exec_op_rope

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_rope>
                 [ guard::invalid_op_rope{} ]
                 / action::reject_invalid_op_rope

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_rope_back>
                 [ guard::valid_op_rope_back{} ]
                 / action::exec_op_rope_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_rope_back>
                 [ guard::invalid_op_rope_back{} ]
                 / action::reject_invalid_op_rope_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_clamp>
                 [ guard::valid_op_clamp{} ]
                 / action::exec_op_clamp

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_clamp>
                 [ guard::invalid_op_clamp{} ]
                 / action::reject_invalid_op_clamp

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_transpose_1d>
                 [ guard::valid_op_conv_transpose_1d{} ]
                 / action::exec_op_conv_transpose_1d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_transpose_1d>
                 [ guard::invalid_op_conv_transpose_1d{} ]
                 / action::reject_invalid_op_conv_transpose_1d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_im2col>
                 [ guard::valid_op_im2col{} ]
                 / action::exec_op_im2col

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_im2col>
                 [ guard::invalid_op_im2col{} ]
                 / action::reject_invalid_op_im2col

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_im2col_back>
                 [ guard::valid_op_im2col_back{} ]
                 / action::exec_op_im2col_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_im2col_back>
                 [ guard::invalid_op_im2col_back{} ]
                 / action::reject_invalid_op_im2col_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_im2col_3d>
                 [ guard::valid_op_im2col_3d{} ]
                 / action::exec_op_im2col_3d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_im2col_3d>
                 [ guard::invalid_op_im2col_3d{} ]
                 / action::reject_invalid_op_im2col_3d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_2d>
                 [ guard::valid_op_conv_2d{} ]
                 / action::exec_op_conv_2d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_2d>
                 [ guard::invalid_op_conv_2d{} ]
                 / action::reject_invalid_op_conv_2d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_3d>
                 [ guard::valid_op_conv_3d{} ]
                 / action::exec_op_conv_3d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_3d>
                 [ guard::invalid_op_conv_3d{} ]
                 / action::reject_invalid_op_conv_3d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_2d_dw>
                 [ guard::valid_op_conv_2d_dw{} ]
                 / action::exec_op_conv_2d_dw

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_2d_dw>
                 [ guard::invalid_op_conv_2d_dw{} ]
                 / action::reject_invalid_op_conv_2d_dw

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_transpose_2d>
                 [ guard::valid_op_conv_transpose_2d{} ]
                 / action::exec_op_conv_transpose_2d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_conv_transpose_2d>
                 [ guard::invalid_op_conv_transpose_2d{} ]
                 / action::reject_invalid_op_conv_transpose_2d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_pool_1d>
                 [ guard::valid_op_pool_1d{} ]
                 / action::exec_op_pool_1d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_pool_1d>
                 [ guard::invalid_op_pool_1d{} ]
                 / action::reject_invalid_op_pool_1d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_pool_2d>
                 [ guard::valid_op_pool_2d{} ]
                 / action::exec_op_pool_2d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_pool_2d>
                 [ guard::invalid_op_pool_2d{} ]
                 / action::reject_invalid_op_pool_2d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_pool_2d_back>
                 [ guard::valid_op_pool_2d_back{} ]
                 / action::exec_op_pool_2d_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_pool_2d_back>
                 [ guard::invalid_op_pool_2d_back{} ]
                 / action::reject_invalid_op_pool_2d_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_upscale>
                 [ guard::valid_op_upscale{} ]
                 / action::exec_op_upscale

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_upscale>
                 [ guard::invalid_op_upscale{} ]
                 / action::reject_invalid_op_upscale

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_pad>
                 [ guard::valid_op_pad{} ]
                 / action::exec_op_pad

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_pad>
                 [ guard::invalid_op_pad{} ]
                 / action::reject_invalid_op_pad

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_pad_reflect_1d>
                 [ guard::valid_op_pad_reflect_1d{} ]
                 / action::exec_op_pad_reflect_1d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_pad_reflect_1d>
                 [ guard::invalid_op_pad_reflect_1d{} ]
                 / action::reject_invalid_op_pad_reflect_1d

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_roll>
                 [ guard::valid_op_roll{} ]
                 / action::exec_op_roll

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_roll>
                 [ guard::invalid_op_roll{} ]
                 / action::reject_invalid_op_roll

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_arange>
                 [ guard::valid_op_arange{} ]
                 / action::exec_op_arange

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_arange>
                 [ guard::invalid_op_arange{} ]
                 / action::reject_invalid_op_arange

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_timestep_embedding>
                 [ guard::valid_op_timestep_embedding{} ]
                 / action::exec_op_timestep_embedding

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_timestep_embedding>
                 [ guard::invalid_op_timestep_embedding{} ]
                 / action::reject_invalid_op_timestep_embedding

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_argsort>
                 [ guard::valid_op_argsort{} ]
                 / action::exec_op_argsort

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_argsort>
                 [ guard::invalid_op_argsort{} ]
                 / action::reject_invalid_op_argsort

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_top_k>
                 [ guard::valid_op_top_k{} ]
                 / action::exec_op_top_k

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_top_k>
                 [ guard::invalid_op_top_k{} ]
                 / action::reject_invalid_op_top_k

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_leaky_relu>
                 [ guard::valid_op_leaky_relu{} ]
                 / action::exec_op_leaky_relu

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_leaky_relu>
                 [ guard::invalid_op_leaky_relu{} ]
                 / action::reject_invalid_op_leaky_relu

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_tri>
                 [ guard::valid_op_tri{} ]
                 / action::exec_op_tri

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_tri>
                 [ guard::invalid_op_tri{} ]
                 / action::reject_invalid_op_tri

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_fill>
                 [ guard::valid_op_fill{} ]
                 / action::exec_op_fill

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_fill>
                 [ guard::invalid_op_fill{} ]
                 / action::reject_invalid_op_fill

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_flash_attn_ext>
                 [ guard::valid_op_flash_attn_ext{} ]
                 / action::exec_op_flash_attn_ext

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_flash_attn_ext>
                 [ guard::invalid_op_flash_attn_ext{} ]
                 / action::reject_invalid_op_flash_attn_ext

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_flash_attn_back>
                 [ guard::valid_op_flash_attn_back{} ]
                 / action::exec_op_flash_attn_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_flash_attn_back>
                 [ guard::invalid_op_flash_attn_back{} ]
                 / action::reject_invalid_op_flash_attn_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_ssm_conv>
                 [ guard::valid_op_ssm_conv{} ]
                 / action::exec_op_ssm_conv

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_ssm_conv>
                 [ guard::invalid_op_ssm_conv{} ]
                 / action::reject_invalid_op_ssm_conv

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_ssm_scan>
                 [ guard::valid_op_ssm_scan{} ]
                 / action::exec_op_ssm_scan

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_ssm_scan>
                 [ guard::invalid_op_ssm_scan{} ]
                 / action::reject_invalid_op_ssm_scan

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_win_part>
                 [ guard::valid_op_win_part{} ]
                 / action::exec_op_win_part

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_win_part>
                 [ guard::invalid_op_win_part{} ]
                 / action::reject_invalid_op_win_part

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_win_unpart>
                 [ guard::valid_op_win_unpart{} ]
                 / action::exec_op_win_unpart

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_win_unpart>
                 [ guard::invalid_op_win_unpart{} ]
                 / action::reject_invalid_op_win_unpart

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_get_rel_pos>
                 [ guard::valid_op_get_rel_pos{} ]
                 / action::exec_op_get_rel_pos

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_get_rel_pos>
                 [ guard::invalid_op_get_rel_pos{} ]
                 / action::reject_invalid_op_get_rel_pos

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_add_rel_pos>
                 [ guard::valid_op_add_rel_pos{} ]
                 / action::exec_op_add_rel_pos

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_add_rel_pos>
                 [ guard::invalid_op_add_rel_pos{} ]
                 / action::reject_invalid_op_add_rel_pos

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_rwkv_wkv6>
                 [ guard::valid_op_rwkv_wkv6{} ]
                 / action::exec_op_rwkv_wkv6

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_rwkv_wkv6>
                 [ guard::invalid_op_rwkv_wkv6{} ]
                 / action::reject_invalid_op_rwkv_wkv6

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_gated_linear_attn>
                 [ guard::valid_op_gated_linear_attn{} ]
                 / action::exec_op_gated_linear_attn

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_gated_linear_attn>
                 [ guard::invalid_op_gated_linear_attn{} ]
                 / action::reject_invalid_op_gated_linear_attn

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_rwkv_wkv7>
                 [ guard::valid_op_rwkv_wkv7{} ]
                 / action::exec_op_rwkv_wkv7

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_rwkv_wkv7>
                 [ guard::invalid_op_rwkv_wkv7{} ]
                 / action::reject_invalid_op_rwkv_wkv7

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_solve_tri>
                 [ guard::valid_op_solve_tri{} ]
                 / action::exec_op_solve_tri

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_solve_tri>
                 [ guard::invalid_op_solve_tri{} ]
                 / action::reject_invalid_op_solve_tri

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_unary>
                 [ guard::valid_op_unary{} ]
                 / action::exec_op_unary

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_unary>
                 [ guard::invalid_op_unary{} ]
                 / action::reject_invalid_op_unary

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_map_custom1>
                 [ guard::valid_op_map_custom1{} ]
                 / action::exec_op_map_custom1

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_map_custom1>
                 [ guard::invalid_op_map_custom1{} ]
                 / action::reject_invalid_op_map_custom1

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_map_custom2>
                 [ guard::valid_op_map_custom2{} ]
                 / action::exec_op_map_custom2

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_map_custom2>
                 [ guard::invalid_op_map_custom2{} ]
                 / action::reject_invalid_op_map_custom2

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_map_custom3>
                 [ guard::valid_op_map_custom3{} ]
                 / action::exec_op_map_custom3

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_map_custom3>
                 [ guard::invalid_op_map_custom3{} ]
                 / action::reject_invalid_op_map_custom3

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_custom>
                 [ guard::valid_op_custom{} ]
                 / action::exec_op_custom

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_custom>
                 [ guard::invalid_op_custom{} ]
                 / action::reject_invalid_op_custom

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_cross_entropy_loss>
                 [ guard::valid_op_cross_entropy_loss{} ]
                 / action::exec_op_cross_entropy_loss

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_cross_entropy_loss>
                 [ guard::invalid_op_cross_entropy_loss{} ]
                 / action::reject_invalid_op_cross_entropy_loss

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_cross_entropy_loss_back>
                 [ guard::valid_op_cross_entropy_loss_back{} ]
                 / action::exec_op_cross_entropy_loss_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_cross_entropy_loss_back>
                 [ guard::invalid_op_cross_entropy_loss_back{} ]
                 / action::reject_invalid_op_cross_entropy_loss_back

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_opt_step_adamw>
                 [ guard::valid_op_opt_step_adamw{} ]
                 / action::exec_op_opt_step_adamw

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_opt_step_adamw>
                 [ guard::invalid_op_opt_step_adamw{} ]
                 / action::reject_invalid_op_opt_step_adamw

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_opt_step_sgd>
                 [ guard::valid_op_opt_step_sgd{} ]
                 / action::exec_op_opt_step_sgd

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_opt_step_sgd>
                 [ guard::invalid_op_opt_step_sgd{} ]
                 / action::reject_invalid_op_opt_step_sgd

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_glu>
                 [ guard::valid_op_glu{} ]
                 / action::exec_op_glu

      , sml::state<ready> <= sml::state<ready> +
               sml::event<::emel::kernel::metal::event::dispatch_op_glu>
                 [ guard::invalid_op_glu{} ]
                 / action::reject_invalid_op_glu

      //------------------------------------------------------------------------------//
      // Unexpected events.
      , sml::state<ready> <= sml::state<ready> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : public emel::sm<model, action::context> {
  using base_type = emel::sm<model, action::context>;
  using base_type::base_type;

  bool process_event(const ::emel::kernel::event::dispatch & ev) {
    event::dispatch_ctx ctx{};
    const event::dispatch_request dispatch{ev, ctx};
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

 private:
  template <class dispatch_event_type>
  bool process_dispatch_event(const dispatch_event_type & ev) {
    const bool accepted = base_type::process_event(ev);
    return accepted && ev.ctx.err == static_cast<int32_t>(emel::error::cast(error::none));
  }
};

}  // namespace emel::kernel::metal
