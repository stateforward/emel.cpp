#pragma once

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
// - tmp/llama.cpp/ggml/include/ggml.h (enum ggml_op)
// - tmp/llama.cpp/ggml/src/ggml.c (GGML_OP_NAME / GGML_OP_SYMBOL)
//
// NOTE: These are typed dispatch events. Most op payloads share this
// generic shape until op-specific contracts are finalized.
//------------------------------------------------------------------------------

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

#define EMEL_KERNEL_GENERIC_OP_FIELDS \
  const void * src0 = nullptr;        \
  const void * src1 = nullptr;        \
  const void * src2 = nullptr;        \
  void * dst = nullptr;               \
  uint32_t element_count = 0;         \
  uint32_t dim0 = 0;                  \
  uint32_t dim1 = 0;                  \
  uint32_t dim2 = 0;                  \
  uint32_t dim3 = 0;

#define EMEL_KERNEL_DECLARE_OP(name) \
  struct name {                      \
    EMEL_KERNEL_GENERIC_OP_FIELDS    \
  };

struct op_dup {
  const void * src0 = nullptr;
  void * dst = nullptr;
  uint32_t element_count = 0;
};

struct op_add {
  const void * src0 = nullptr;
  const void * src1 = nullptr;
  void * dst = nullptr;
  uint32_t element_count = 0;
};

EMEL_KERNEL_DECLARE_OP(op_add_id);
EMEL_KERNEL_DECLARE_OP(op_add1);
EMEL_KERNEL_DECLARE_OP(op_acc);
EMEL_KERNEL_DECLARE_OP(op_sub);

struct op_mul {
  const void * src0 = nullptr;
  const void * src1 = nullptr;
  void * dst = nullptr;
  uint32_t element_count = 0;
};

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

struct op_mul_mat {
  const void * src0 = nullptr;
  const void * src1 = nullptr;
  void * dst = nullptr;
  uint32_t row_count = 0;
  uint32_t col_count = 0;
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

struct op_rope {
  const void * src0 = nullptr;
  void * dst = nullptr;
  uint32_t token_count = 0;
};

struct op_soft_max {
  const void * src0 = nullptr;
  void * dst = nullptr;
  uint32_t element_count = 0;
};

EMEL_KERNEL_DECLARE_OP(op_soft_max_back);
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
