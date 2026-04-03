#pragma once

#include <array>
#include <cstdint>

#include "emel/emel.h"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/errors.hpp"

namespace emel::kernel::action {
struct context;
}

namespace emel::kernel::event {

struct dispatch {};

//------------------------------------------------------------------------------
// GGML/GGUF opcode events.
//
// Reference source-of-truth:
// - tmp/llama.cpp/ggml/include/ggml.h (enum ggml_op / enum ggml_type)
// - tmp/llama.cpp/ggml/src/ggml-cpu/ops.cpp
//------------------------------------------------------------------------------

enum class dtype : uint8_t {
  f32 = 0,
  f16 = 1,
  q4_0 = 2,
  q4_1 = 3,
  q5_0 = 6,
  q5_1 = 7,
  q8_0 = 8,
  q8_1 = 9,
  q2_k = 10,
  q3_k = 11,
  q4_k = 12,
  q5_k = 13,
  q6_k = 14,
  q8_k = 15,
  iq2_xxs = 16,
  iq2_xs = 17,
  iq3_xxs = 18,
  iq1_s = 19,
  iq4_nl = 20,
  iq3_s = 21,
  iq2_s = 22,
  iq4_xs = 23,
  i8 = 24,
  i16 = 25,
  i32 = 26,
  i64 = 27,
  f64 = 28,
  iq1_m = 29,
  bf16 = 30,
  q4_0_4_4 = 31,
  q4_0_4_8 = 32,
  q4_0_8_8 = 33,
  tq1_0 = 34,
  tq2_0 = 35,
  q1_0_g128 = 41,
  q6_k_x8 = 128,
  q6_k_x8_q8_prepared = 129,
  q6_k_x8_q8_argmax_prepared = 130,
  q8_0_x4_bl4 = 131,
  q8_0_x4_bl8 = 132,
  q4_k_x8_bl4 = 133,
  q4_k_x8_bl8 = 134,
  q8_k_x4 = 135,
  q8_k_x8 = 136,
  unknown = 255,
};

struct tensor_view {
  const void * data = nullptr;
  dtype type = dtype::unknown;
  std::array<uint64_t, 4> ne = {0, 1, 1, 1};
  std::array<uint64_t, 4> nb = {0, 0, 0, 0};
};

struct tensor_view_mut {
  void * data = nullptr;
  dtype type = dtype::unknown;
  std::array<uint64_t, 4> ne = {0, 1, 1, 1};
  std::array<uint64_t, 4> nb = {0, 0, 0, 0};
};

enum class pool_subop : uint8_t {
  max = 0,
  avg = 1,
};

enum class unary_subop : uint8_t {
  abs = 0,
  sgn = 1,
  neg = 2,
  step = 3,
  tanh = 4,
  elu = 5,
  relu = 6,
  sigmoid = 7,
  gelu = 8,
  gelu_quick = 9,
  silu = 10,
  hardswish = 11,
  hardsigmoid = 12,
  exp = 13,
  expm1 = 14,
  softplus = 15,
  gelu_erf = 16,
  xielu = 17,
  floor = 18,
  ceil = 19,
  round = 20,
  trunc = 21,
};

enum class glu_subop : uint8_t {
  reglu = 0,
  geglu = 1,
  swiglu = 2,
  swiglu_oai = 3,
  geglu_erf = 4,
  geglu_quick = 5,
};

#define EMEL_KERNEL_GENERIC_OP_FIELDS        \
  tensor_view src0 = {};                     \
  tensor_view src1 = {};                     \
  tensor_view src2 = {};                     \
  tensor_view_mut dst = {};                  \
  std::array<uint8_t, 64> op_params = {};    \
  uint32_t op_params_size = 0;               \
  uint32_t ith = 0;                          \
  uint32_t nth = 1;

#define EMEL_KERNEL_DECLARE_OP(name) \
  struct name {                      \
    EMEL_KERNEL_GENERIC_OP_FIELDS    \
  };

EMEL_KERNEL_DECLARE_OP(op_dup);
EMEL_KERNEL_DECLARE_OP(op_add);
EMEL_KERNEL_DECLARE_OP(op_add_id);
EMEL_KERNEL_DECLARE_OP(op_add1);
EMEL_KERNEL_DECLARE_OP(op_acc);
EMEL_KERNEL_DECLARE_OP(op_sub);
EMEL_KERNEL_DECLARE_OP(op_mul);
EMEL_KERNEL_DECLARE_OP(op_div);
EMEL_KERNEL_DECLARE_OP(op_sqr);
EMEL_KERNEL_DECLARE_OP(op_sqrt);
EMEL_KERNEL_DECLARE_OP(op_log);
EMEL_KERNEL_DECLARE_OP(op_sin);
EMEL_KERNEL_DECLARE_OP(op_cos);
EMEL_KERNEL_DECLARE_OP(op_sum);
EMEL_KERNEL_DECLARE_OP(op_sum_rows);
EMEL_KERNEL_DECLARE_OP(op_cumsum);
EMEL_KERNEL_DECLARE_OP(op_mean);
EMEL_KERNEL_DECLARE_OP(op_argmax);
EMEL_KERNEL_DECLARE_OP(op_count_equal);
EMEL_KERNEL_DECLARE_OP(op_repeat);
EMEL_KERNEL_DECLARE_OP(op_repeat_back);
EMEL_KERNEL_DECLARE_OP(op_concat);
EMEL_KERNEL_DECLARE_OP(op_silu_back);
EMEL_KERNEL_DECLARE_OP(op_norm);
EMEL_KERNEL_DECLARE_OP(op_rms_norm);
EMEL_KERNEL_DECLARE_OP(op_rms_norm_back);
EMEL_KERNEL_DECLARE_OP(op_group_norm);
EMEL_KERNEL_DECLARE_OP(op_l2_norm);
EMEL_KERNEL_DECLARE_OP(op_mul_mat);
struct op_mul_mat_argmax {
  EMEL_KERNEL_GENERIC_OP_FIELDS
  int32_t * index_out = nullptr;
};
EMEL_KERNEL_DECLARE_OP(op_mul_mat_id);
EMEL_KERNEL_DECLARE_OP(op_out_prod);
EMEL_KERNEL_DECLARE_OP(op_scale);
EMEL_KERNEL_DECLARE_OP(op_set);
EMEL_KERNEL_DECLARE_OP(op_cpy);
EMEL_KERNEL_DECLARE_OP(op_cont);
EMEL_KERNEL_DECLARE_OP(op_reshape);
EMEL_KERNEL_DECLARE_OP(op_view);
EMEL_KERNEL_DECLARE_OP(op_permute);
EMEL_KERNEL_DECLARE_OP(op_transpose);
EMEL_KERNEL_DECLARE_OP(op_get_rows);
EMEL_KERNEL_DECLARE_OP(op_get_rows_back);
EMEL_KERNEL_DECLARE_OP(op_set_rows);
EMEL_KERNEL_DECLARE_OP(op_diag);
EMEL_KERNEL_DECLARE_OP(op_diag_mask_inf);
EMEL_KERNEL_DECLARE_OP(op_diag_mask_zero);
EMEL_KERNEL_DECLARE_OP(op_soft_max);
EMEL_KERNEL_DECLARE_OP(op_soft_max_back);
EMEL_KERNEL_DECLARE_OP(op_rope);
EMEL_KERNEL_DECLARE_OP(op_rope_back);
EMEL_KERNEL_DECLARE_OP(op_clamp);
EMEL_KERNEL_DECLARE_OP(op_conv_transpose_1d);
EMEL_KERNEL_DECLARE_OP(op_im2col);
EMEL_KERNEL_DECLARE_OP(op_im2col_back);
EMEL_KERNEL_DECLARE_OP(op_im2col_3d);
EMEL_KERNEL_DECLARE_OP(op_conv_2d);
EMEL_KERNEL_DECLARE_OP(op_conv_3d);
EMEL_KERNEL_DECLARE_OP(op_conv_2d_dw);
EMEL_KERNEL_DECLARE_OP(op_conv_transpose_2d);

struct op_pool_1d {
  EMEL_KERNEL_GENERIC_OP_FIELDS
  pool_subop pool = pool_subop::avg;
};

struct op_pool_2d {
  EMEL_KERNEL_GENERIC_OP_FIELDS
  pool_subop pool = pool_subop::avg;
};

struct op_pool_2d_back {
  EMEL_KERNEL_GENERIC_OP_FIELDS
  pool_subop pool = pool_subop::avg;
};

EMEL_KERNEL_DECLARE_OP(op_upscale);
EMEL_KERNEL_DECLARE_OP(op_pad);
EMEL_KERNEL_DECLARE_OP(op_pad_reflect_1d);
EMEL_KERNEL_DECLARE_OP(op_roll);
EMEL_KERNEL_DECLARE_OP(op_arange);
EMEL_KERNEL_DECLARE_OP(op_timestep_embedding);
EMEL_KERNEL_DECLARE_OP(op_argsort);
EMEL_KERNEL_DECLARE_OP(op_top_k);
EMEL_KERNEL_DECLARE_OP(op_leaky_relu);
EMEL_KERNEL_DECLARE_OP(op_tri);
EMEL_KERNEL_DECLARE_OP(op_fill);
EMEL_KERNEL_DECLARE_OP(op_flash_attn_ext);
EMEL_KERNEL_DECLARE_OP(op_flash_attn_back);
EMEL_KERNEL_DECLARE_OP(op_ssm_conv);
EMEL_KERNEL_DECLARE_OP(op_ssm_scan);
EMEL_KERNEL_DECLARE_OP(op_win_part);
EMEL_KERNEL_DECLARE_OP(op_win_unpart);
EMEL_KERNEL_DECLARE_OP(op_get_rel_pos);
EMEL_KERNEL_DECLARE_OP(op_add_rel_pos);
EMEL_KERNEL_DECLARE_OP(op_rwkv_wkv6);
EMEL_KERNEL_DECLARE_OP(op_gated_linear_attn);
EMEL_KERNEL_DECLARE_OP(op_rwkv_wkv7);
EMEL_KERNEL_DECLARE_OP(op_solve_tri);

struct op_unary {
  EMEL_KERNEL_GENERIC_OP_FIELDS
  unary_subop subop = unary_subop::abs;
};

EMEL_KERNEL_DECLARE_OP(op_map_custom1);
EMEL_KERNEL_DECLARE_OP(op_map_custom2);
EMEL_KERNEL_DECLARE_OP(op_map_custom3);
EMEL_KERNEL_DECLARE_OP(op_custom);
EMEL_KERNEL_DECLARE_OP(op_cross_entropy_loss);
EMEL_KERNEL_DECLARE_OP(op_cross_entropy_loss_back);
EMEL_KERNEL_DECLARE_OP(op_opt_step_adamw);
EMEL_KERNEL_DECLARE_OP(op_opt_step_sgd);

struct op_glu {
  EMEL_KERNEL_GENERIC_OP_FIELDS
  glu_subop subop = glu_subop::reglu;
};

#undef EMEL_KERNEL_DECLARE_OP
#undef EMEL_KERNEL_GENERIC_OP_FIELDS

enum class phase_outcome : uint8_t {
  unknown = 0,
  done = 1,
  rejected = 2,
};

struct dispatch_ctx {
  bool primary_accepted = false;
  bool secondary_accepted = false;
  bool tertiary_accepted = false;
  bool quaternary_accepted = false;
  bool quinary_accepted = false;
  bool senary_accepted = false;
  phase_outcome primary_outcome = phase_outcome::unknown;
  phase_outcome secondary_outcome = phase_outcome::unknown;
  phase_outcome tertiary_outcome = phase_outcome::unknown;
  phase_outcome quaternary_outcome = phase_outcome::unknown;
  phase_outcome quinary_outcome = phase_outcome::unknown;
  phase_outcome senary_outcome = phase_outcome::unknown;
  int32_t err = static_cast<int32_t>(emel::error::cast(error::none));
};

// Internal event used by kernel::sm wrapper; not part of public API.
struct dispatch_request {
  const dispatch & request;
  dispatch_ctx & ctx;
};

#define EMEL_KERNEL_DECLARE_DISPATCH_EVENT(op_name) \
  struct dispatch_##op_name {                       \
    const op_name & request;                        \
    dispatch_ctx & ctx;                             \
  };
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_DISPATCH_EVENT)
#undef EMEL_KERNEL_DECLARE_DISPATCH_EVENT

template <class event_type>
struct dispatch_event_for;

#define EMEL_KERNEL_DECLARE_DISPATCH_EVENT_TRAIT(op_name) \
  template <>                                              \
  struct dispatch_event_for<op_name> {                    \
    using type = dispatch_##op_name;                      \
  };
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_DECLARE_DISPATCH_EVENT_TRAIT)
#undef EMEL_KERNEL_DECLARE_DISPATCH_EVENT_TRAIT

template <class event_type>
using dispatch_event_for_t = typename dispatch_event_for<event_type>::type;

}  // namespace emel::kernel::event

namespace emel::kernel::events {

struct dispatch_done {};

struct dispatch_error {
  int32_t err = static_cast<int32_t>(emel::error::cast(error::internal_error));
};

}  // namespace emel::kernel::events
