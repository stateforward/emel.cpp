#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
#include <immintrin.h>
#endif

// Keep this list aligned with `tmp/llama.cpp/ggml/include/ggml.h` (`enum
// ggml_op`), excluding sentinel entries (`NONE`, `COUNT`).
#define EMEL_KERNEL_OP_EVENT_LIST(X)                                           \
  X(op_dup)                                                                    \
  X(op_add)                                                                    \
  X(op_add_id)                                                                 \
  X(op_add1)                                                                   \
  X(op_acc)                                                                    \
  X(op_sub)                                                                    \
  X(op_mul)                                                                    \
  X(op_div)                                                                    \
  X(op_sqr)                                                                    \
  X(op_sqrt)                                                                   \
  X(op_log)                                                                    \
  X(op_sin)                                                                    \
  X(op_cos)                                                                    \
  X(op_sum)                                                                    \
  X(op_sum_rows)                                                               \
  X(op_cumsum)                                                                 \
  X(op_mean)                                                                   \
  X(op_argmax)                                                                 \
  X(op_count_equal)                                                            \
  X(op_repeat)                                                                 \
  X(op_repeat_back)                                                            \
  X(op_concat)                                                                 \
  X(op_silu_back)                                                              \
  X(op_norm)                                                                   \
  X(op_rms_norm)                                                               \
  X(op_rms_norm_back)                                                          \
  X(op_group_norm)                                                             \
  X(op_l2_norm)                                                                \
  X(op_mul_mat)                                                                \
  X(op_mul_mat_argmax)                                                         \
  X(op_mul_mat_id)                                                             \
  X(op_out_prod)                                                               \
  X(op_scale)                                                                  \
  X(op_set)                                                                    \
  X(op_cpy)                                                                    \
  X(op_cont)                                                                   \
  X(op_reshape)                                                                \
  X(op_view)                                                                   \
  X(op_permute)                                                                \
  X(op_transpose)                                                              \
  X(op_get_rows)                                                               \
  X(op_get_rows_back)                                                          \
  X(op_set_rows)                                                               \
  X(op_diag)                                                                   \
  X(op_diag_mask_inf)                                                          \
  X(op_diag_mask_zero)                                                         \
  X(op_soft_max)                                                               \
  X(op_soft_max_back)                                                          \
  X(op_rope)                                                                   \
  X(op_rope_back)                                                              \
  X(op_clamp)                                                                  \
  X(op_conv_transpose_1d)                                                      \
  X(op_im2col)                                                                 \
  X(op_im2col_back)                                                            \
  X(op_im2col_3d)                                                              \
  X(op_conv_2d)                                                                \
  X(op_conv_3d)                                                                \
  X(op_conv_2d_dw)                                                             \
  X(op_conv_transpose_2d)                                                      \
  X(op_pool_1d)                                                                \
  X(op_pool_2d)                                                                \
  X(op_pool_2d_back)                                                           \
  X(op_upscale)                                                                \
  X(op_pad)                                                                    \
  X(op_pad_reflect_1d)                                                         \
  X(op_roll)                                                                   \
  X(op_arange)                                                                 \
  X(op_timestep_embedding)                                                     \
  X(op_argsort)                                                                \
  X(op_top_k)                                                                  \
  X(op_leaky_relu)                                                             \
  X(op_tri)                                                                    \
  X(op_fill)                                                                   \
  X(op_flash_attn_ext)                                                         \
  X(op_flash_attn_back)                                                        \
  X(op_ssm_conv)                                                               \
  X(op_ssm_scan)                                                               \
  X(op_win_part)                                                               \
  X(op_win_unpart)                                                             \
  X(op_get_rel_pos)                                                            \
  X(op_add_rel_pos)                                                            \
  X(op_rwkv_wkv6)                                                              \
  X(op_gated_linear_attn)                                                      \
  X(op_rwkv_wkv7)                                                              \
  X(op_solve_tri)                                                              \
  X(op_unary)                                                                  \
  X(op_map_custom1)                                                            \
  X(op_map_custom2)                                                            \
  X(op_map_custom3)                                                            \
  X(op_custom)                                                                 \
  X(op_cross_entropy_loss)                                                     \
  X(op_cross_entropy_loss_back)                                                \
  X(op_opt_step_adamw)                                                         \
  X(op_opt_step_sgd)                                                           \
  X(op_glu)

namespace emel::kernel::event {

enum class dtype : uint8_t;
enum class unary_subop : uint8_t;

#define EMEL_KERNEL_FORWARD_DECLARE_EVENT(op_name) struct op_name;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_FORWARD_DECLARE_EVENT)
#undef EMEL_KERNEL_FORWARD_DECLARE_EVENT

} // namespace emel::kernel::event

namespace emel::kernel {

template <class event_type> struct is_op_event : std::false_type {};

#define EMEL_KERNEL_MARK_OP_EVENT(op_name)                                     \
  template <> struct is_op_event<event::op_name> : std::true_type {};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_MARK_OP_EVENT)
#undef EMEL_KERNEL_MARK_OP_EVENT

template <class event_type>
inline constexpr bool is_op_event_v = is_op_event<event_type>::value;

} // namespace emel::kernel

namespace emel::kernel::detail {

inline constexpr uint8_t dtype_f32 = 0;
inline constexpr uint8_t dtype_f16 = 1;
inline constexpr uint8_t dtype_q4_0 = 2;
inline constexpr uint8_t dtype_q4_1 = 3;
inline constexpr uint8_t dtype_q5_0 = 6;
inline constexpr uint8_t dtype_q5_1 = 7;
inline constexpr uint8_t dtype_q8_0 = 8;
inline constexpr uint8_t dtype_q2_k = 10;
inline constexpr uint8_t dtype_q3_k = 11;
inline constexpr uint8_t dtype_q4_k = 12;
inline constexpr uint8_t dtype_q5_k = 13;
inline constexpr uint8_t dtype_q6_k = 14;
inline constexpr uint8_t dtype_q8_k = 15;
inline constexpr uint8_t dtype_i32 = 26;
inline constexpr uint8_t dtype_bf16 = 30;
inline constexpr uint8_t dtype_q6_k_x8 = 36;
inline constexpr uint8_t dtype_q6_k_x8_q8_prepared = 37;
inline constexpr uint8_t dtype_q6_k_x8_q8_argmax_prepared = 38;
inline constexpr uint8_t dtype_q8_0_x4_bl4 = 39;
inline constexpr uint8_t dtype_q8_0_x4_bl8 = 40;
inline constexpr uint8_t dtype_q4_k_x8_bl4 = 41;
inline constexpr uint8_t dtype_q4_k_x8_bl8 = 42;
inline constexpr uint8_t dtype_q8_k_x4 = 43;
inline constexpr uint8_t dtype_q8_k_x8 = 44;
inline constexpr uint64_t flash_attn_workspace_token_capacity = 4096u;

struct flash_attn_workspace {
  alignas(64)
      std::array<float, flash_attn_workspace_token_capacity> score_buffer = {};
  alignas(64)
      std::array<float, flash_attn_workspace_token_capacity> value_buffer = {};
  alignas(64)
      std::array<float, flash_attn_workspace_token_capacity> accum_buffer = {};
  alignas(64) std::array<uint16_t,
                         flash_attn_workspace_token_capacity> q_buffer_f16 = {};
  alignas(64) std::array<
      uint16_t, flash_attn_workspace_token_capacity> accum_buffer_f16 = {};
  uint64_t prepared_tokens = 0;
  uint64_t reuse_count = 0;
};

namespace quant {

constexpr uint64_t QK4_0 = 32u;
constexpr uint64_t QK4_1 = 32u;
constexpr uint64_t QK5_0 = 32u;
constexpr uint64_t QK5_1 = 32u;
constexpr uint64_t QK8_0 = 32u;
constexpr uint64_t QK_K = 256u;
constexpr uint64_t K_SCALE_SIZE = 12u;
constexpr uint64_t MAX_Q8_K_BLOCKS = 128u;
constexpr uint64_t MAX_Q8_0_BLOCKS = MAX_Q8_K_BLOCKS * (QK_K / QK8_0);
constexpr uint64_t Q4_K_X8_ROWS = 8u;
constexpr uint64_t Q6_K_X8_ROWS = 8u;
constexpr uint64_t Q8_0_X4_ROWS = 4u;

struct block_q4_0 {
  uint16_t d = 0;
  std::array<uint8_t, QK4_0 / 2> qs = {};
};

struct block_q4_1 {
  uint16_t d = 0;
  uint16_t m = 0;
  std::array<uint8_t, QK4_1 / 2> qs = {};
};

struct block_q5_0 {
  uint16_t d = 0;
  std::array<uint8_t, 4> qh = {};
  std::array<uint8_t, QK5_0 / 2> qs = {};
};

struct block_q5_1 {
  uint16_t d = 0;
  uint16_t m = 0;
  std::array<uint8_t, 4> qh = {};
  std::array<uint8_t, QK5_1 / 2> qs = {};
};

struct block_q8_0 {
  uint16_t d = 0;
  std::array<int8_t, QK8_0> qs = {};
};

struct block_q8_0x4 {
  std::array<uint16_t, Q8_0_X4_ROWS> d = {};
  std::array<int8_t, QK8_0 * Q8_0_X4_ROWS> qs = {};
};

struct block_q2_k {
  std::array<uint8_t, QK_K / 16> scales = {};
  std::array<uint8_t, QK_K / 4> qs = {};
  uint16_t d = 0;
  uint16_t dmin = 0;
};

struct block_q3_k {
  std::array<uint8_t, QK_K / 8> hmask = {};
  std::array<uint8_t, QK_K / 4> qs = {};
  std::array<uint8_t, 12> scales = {};
  uint16_t d = 0;
};

struct block_q4_k {
  uint16_t d = 0;
  uint16_t dmin = 0;
  std::array<uint8_t, K_SCALE_SIZE> scales = {};
  std::array<uint8_t, QK_K / 2> qs = {};
};

struct block_q5_k {
  uint16_t d = 0;
  uint16_t dmin = 0;
  std::array<uint8_t, K_SCALE_SIZE> scales = {};
  std::array<uint8_t, QK_K / 8> qh = {};
  std::array<uint8_t, QK_K / 2> qs = {};
};

struct block_q6_k {
  std::array<uint8_t, QK_K / 2> ql = {};
  std::array<uint8_t, QK_K / 4> qh = {};
  std::array<int8_t, QK_K / 16> scales = {};
  uint16_t d = 0;
};

struct block_q4_kx8 {
  std::array<uint16_t, Q4_K_X8_ROWS> d = {};
  std::array<uint16_t, Q4_K_X8_ROWS> dmin = {};
  std::array<uint8_t, (QK_K / 32u) * Q4_K_X8_ROWS * 2u> scales = {};
  std::array<uint8_t, (QK_K / 2) * Q4_K_X8_ROWS> qs = {};
};

struct block_q6_kx8 {
  std::array<uint16_t, Q6_K_X8_ROWS> d = {};
  std::array<int8_t, (QK_K / 16) * Q6_K_X8_ROWS> scales = {};
  std::array<uint8_t, (QK_K / 2) * Q6_K_X8_ROWS> ql = {};
  std::array<uint8_t, (QK_K / 4) * Q6_K_X8_ROWS> qh = {};
};

struct block_q6_kx8_q8_prepared {
  std::array<uint16_t, Q6_K_X8_ROWS> d = {};
  std::array<int8_t, (QK_K / 16) * Q6_K_X8_ROWS> scales = {};
  std::array<int8_t, QK_K * Q6_K_X8_ROWS> qs = {};
};

struct block_q6_kx8_q8_argmax_prepared {
  std::array<float, Q6_K_X8_ROWS> d = {};
  std::array<int8_t, (QK_K / 16) * Q6_K_X8_ROWS> scales = {};
  std::array<int8_t, QK_K * Q6_K_X8_ROWS> qs = {};
};

struct block_q8_k {
  float d = 0.0f;
  std::array<int8_t, QK_K> qs = {};
  std::array<int16_t, QK_K / 16> bsums = {};
};

static_assert(sizeof(block_q4_0) == sizeof(uint16_t) + (QK4_0 / 2));
static_assert(sizeof(block_q4_1) == 2 * sizeof(uint16_t) + (QK4_1 / 2));
static_assert(sizeof(block_q5_0) ==
              sizeof(uint16_t) + sizeof(uint32_t) + (QK5_0 / 2));
static_assert(sizeof(block_q5_1) ==
              2 * sizeof(uint16_t) + sizeof(uint32_t) + (QK5_1 / 2));
static_assert(sizeof(block_q8_0) == sizeof(uint16_t) + QK8_0);
static_assert(sizeof(block_q2_k) ==
              2 * sizeof(uint16_t) + (QK_K / 16) + (QK_K / 4));
static_assert(sizeof(block_q3_k) ==
              sizeof(uint16_t) + (QK_K / 4) + (QK_K / 8) + 12);
static_assert(sizeof(block_q4_k) ==
              2 * sizeof(uint16_t) + K_SCALE_SIZE + (QK_K / 2));
static_assert(sizeof(block_q5_k) ==
              2 * sizeof(uint16_t) + K_SCALE_SIZE + (5 * QK_K / 8));
static_assert(sizeof(block_q6_k) ==
              sizeof(uint16_t) + (QK_K / 16) + (3 * QK_K / 4));
static_assert(sizeof(block_q4_kx8) == sizeof(uint16_t) * (Q4_K_X8_ROWS * 2u) +
                                          ((QK_K / 32u) * Q4_K_X8_ROWS * 2u) +
                                          (QK_K / 2) * Q4_K_X8_ROWS);
static_assert(sizeof(block_q6_kx8) == sizeof(uint16_t) * Q6_K_X8_ROWS +
                                          (QK_K / 16) * Q6_K_X8_ROWS +
                                          (3 * QK_K / 4) * Q6_K_X8_ROWS);
static_assert(sizeof(block_q6_kx8_q8_prepared) ==
              sizeof(uint16_t) * Q6_K_X8_ROWS +
                  sizeof(int8_t) * (QK_K / 16) * Q6_K_X8_ROWS +
                  sizeof(int8_t) * QK_K * Q6_K_X8_ROWS);
static_assert(sizeof(block_q6_kx8_q8_argmax_prepared) ==
              sizeof(float) * Q6_K_X8_ROWS +
                  sizeof(int8_t) * (QK_K / 16) * Q6_K_X8_ROWS +
                  sizeof(int8_t) * QK_K * Q6_K_X8_ROWS);
static_assert(sizeof(block_q8_k) ==
              sizeof(float) + QK_K + (QK_K / 16) * sizeof(int16_t));

inline float fp32_from_bits(const uint32_t bits) noexcept {
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline uint32_t fp32_to_bits(const float value) noexcept {
  uint32_t bits = 0u;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

inline float fp16_to_fp32(const uint16_t bits16) noexcept {
  const uint32_t word = static_cast<uint32_t>(bits16) << 16u;
  const uint32_t sign = word & 0x80000000u;
  const uint32_t doubled = word + word;

  const uint32_t exp_offset = 0xE0u << 23u;
  const float exp_scale = 0x1.0p-112f;
  const float normalized =
      fp32_from_bits((doubled >> 4u) + exp_offset) * exp_scale;

  const uint32_t magic_mask = 126u << 23u;
  const float magic_bias = 0.5f;
  const float denormalized =
      fp32_from_bits((doubled >> 17u) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = 1u << 27u;
  const uint32_t result =
      sign | (doubled < denormalized_cutoff ? fp32_to_bits(denormalized)
                                            : fp32_to_bits(normalized));
  return fp32_from_bits(result);
}

inline uint16_t fp32_to_fp16(const float value) noexcept {
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
  float base = (std::fabs(value) * scale_to_inf) * scale_to_zero;

  const uint32_t word = fp32_to_bits(value);
  const uint32_t doubled = word + word;
  const uint32_t sign = word & 0x80000000u;
  uint32_t bias = doubled & 0xFF000000u;
  if (bias < 0x71000000u) {
    bias = 0x71000000u;
  }

  base = fp32_from_bits((bias >> 1u) + 0x07800000u) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13u) & 0x00007C00u;
  const uint32_t mantissa_bits = bits & 0x00000FFFu;
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<uint16_t>((sign >> 16u) |
                               (doubled > 0xFF000000u ? 0x7E00u : nonsign));
}

inline void dequantize_row_q2_k(const block_q2_k *x, float *y,
                                const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK_K);

  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d);
    const float min = fp16_to_fp32(x[i].dmin);
    const uint8_t *q = x[i].qs.data();

    int is = 0;
    for (int n = 0; n < static_cast<int>(QK_K); n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        uint8_t sc = x[i].scales[static_cast<size_t>(is++)];
        const float dl0 = d * static_cast<float>(sc & 0x0fu);
        const float ml0 = min * static_cast<float>(sc >> 4u);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl0 * static_cast<float>((q[l] >> shift) & 0x03u) - ml0;
        }

        sc = x[i].scales[static_cast<size_t>(is++)];
        const float dl1 = d * static_cast<float>(sc & 0x0fu);
        const float ml1 = min * static_cast<float>(sc >> 4u);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl1 * static_cast<float>((q[l + 16] >> shift) & 0x03u) - ml1;
        }
        shift += 2;
      }
      q += 32;
    }
  }
}

inline void dequantize_row_q3_k(const block_q3_k *x, float *y,
                                const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK_K);
  constexpr uint32_t kmask1 = 0x03030303u;
  constexpr uint32_t kmask2 = 0x0f0f0f0fu;

  uint32_t aux[4] = {};
  auto *scales = reinterpret_cast<int8_t *>(aux);

  for (int64_t i = 0; i < nb; ++i) {
    const float d_all = fp16_to_fp32(x[i].d);
    const uint8_t *q = x[i].qs.data();
    const uint8_t *hm = x[i].hmask.data();
    uint8_t m = 1u;

    std::memcpy(aux, x[i].scales.data(), x[i].scales.size());
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
    aux[3] = ((aux[1] >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0u) & kmask1) << 4u);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2u) & kmask1) << 4u);

    int is = 0;
    for (int n = 0; n < static_cast<int>(QK_K); n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        const float dl0 = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          const int8_t q0 = static_cast<int8_t>((q[l + 0] >> shift) & 0x03u) -
                            ((hm[l + 0] & m) ? 0 : 4);
          *y++ = dl0 * static_cast<float>(q0);
        }

        const float dl1 = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          const int8_t q1 = static_cast<int8_t>((q[l + 16] >> shift) & 0x03u) -
                            ((hm[l + 16] & m) ? 0 : 4);
          *y++ = dl1 * static_cast<float>(q1);
        }

        shift += 2;
        m = static_cast<uint8_t>(m << 1u);
      }
      q += 32;
    }
  }
}

inline void get_scale_min_k4(const int j, const uint8_t *q, uint8_t *d,
                             uint8_t *m) noexcept {
  if (j < 4) {
    *d = q[j] & 63u;
    *m = q[j + 4] & 63u;
    return;
  }

  *d = static_cast<uint8_t>((q[j + 4] & 0x0fu) | ((q[j - 4] >> 6u) << 4u));
  *m = static_cast<uint8_t>((q[j + 4] >> 4u) | ((q[j - 0] >> 6u) << 4u));
}

inline void dequantize_row_q4_0(const block_q4_0 *x, float *y,
                                const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK4_0);

  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d);
    for (uint64_t j = 0; j < (QK4_0 / 2u); ++j) {
      const int32_t x0 = static_cast<int32_t>(x[i].qs[j] & 0x0fu) - 8;
      const int32_t x1 = static_cast<int32_t>(x[i].qs[j] >> 4u) - 8;
      y[i * static_cast<int64_t>(QK4_0) + static_cast<int64_t>(j)] =
          static_cast<float>(x0) * d;
      y[i * static_cast<int64_t>(QK4_0) +
        static_cast<int64_t>(j + (QK4_0 / 2u))] = static_cast<float>(x1) * d;
    }
  }
}

inline void dequantize_row_q4_1(const block_q4_1 *x, float *y,
                                const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK4_1);

  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d);
    const float m = fp16_to_fp32(x[i].m);
    for (uint64_t j = 0; j < (QK4_1 / 2u); ++j) {
      const uint8_t x0 = x[i].qs[j] & 0x0fu;
      const uint8_t x1 = x[i].qs[j] >> 4u;
      y[i * static_cast<int64_t>(QK4_1) + static_cast<int64_t>(j)] =
          static_cast<float>(x0) * d + m;
      y[i * static_cast<int64_t>(QK4_1) +
        static_cast<int64_t>(j + (QK4_1 / 2u))] =
          static_cast<float>(x1) * d + m;
    }
  }
}

inline void dequantize_row_q4_k(const block_q4_k *x, float *y,
                                const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK_K);

  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *q = x[i].qs.data();
    const float d = fp16_to_fp32(x[i].d);
    const float min = fp16_to_fp32(x[i].dmin);

    int is = 0;
    for (int j = 0; j < static_cast<int>(QK_K); j += 64) {
      uint8_t sc = 0u;
      uint8_t m = 0u;
      get_scale_min_k4(is + 0, x[i].scales.data(), &sc, &m);
      const float d0 = d * static_cast<float>(sc);
      const float m0 = min * static_cast<float>(m);
      get_scale_min_k4(is + 1, x[i].scales.data(), &sc, &m);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = min * static_cast<float>(m);
      for (int l = 0; l < 32; ++l) {
        *y++ = d0 * static_cast<float>(q[l] & 0x0fu) - m0;
      }
      for (int l = 0; l < 32; ++l) {
        *y++ = d1 * static_cast<float>(q[l] >> 4u) - m1;
      }
      q += 32;
      is += 2;
    }
  }
}

inline void dequantize_row_q5_k(const block_q5_k *x, float *y,
                                const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK_K);

  for (int64_t i = 0; i < nb; ++i) {
    const uint8_t *ql = x[i].qs.data();
    const uint8_t *qh = x[i].qh.data();
    const float d = fp16_to_fp32(x[i].d);
    const float min = fp16_to_fp32(x[i].dmin);
    int is = 0;
    uint8_t u1 = 1u;
    uint8_t u2 = 2u;

    for (int j = 0; j < static_cast<int>(QK_K); j += 64) {
      uint8_t sc = 0u;
      uint8_t m = 0u;
      get_scale_min_k4(is + 0, x[i].scales.data(), &sc, &m);
      const float d0 = d * static_cast<float>(sc);
      const float m0 = min * static_cast<float>(m);
      get_scale_min_k4(is + 1, x[i].scales.data(), &sc, &m);
      const float d1 = d * static_cast<float>(sc);
      const float m1 = min * static_cast<float>(m);
      for (int l = 0; l < 32; ++l) {
        const uint8_t high = (qh[l] & u1) != 0u ? 16u : 0u;
        *y++ = d0 * static_cast<float>((ql[l] & 0x0fu) + high) - m0;
      }
      for (int l = 0; l < 32; ++l) {
        const uint8_t high = (qh[l] & u2) != 0u ? 16u : 0u;
        *y++ = d1 * static_cast<float>((ql[l] >> 4u) + high) - m1;
      }
      ql += 32;
      is += 2;
      u1 = static_cast<uint8_t>(u1 << 2u);
      u2 = static_cast<uint8_t>(u2 << 2u);
    }
  }
}

inline void dequantize_row_q6_k(const block_q6_k *x, float *y,
                                const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK_K);

  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d);
    const uint8_t *ql = x[i].ql.data();
    const uint8_t *qh = x[i].qh.data();
    const int8_t *sc = x[i].scales.data();

    for (int n = 0; n < static_cast<int>(QK_K); n += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int8_t q1 = static_cast<int8_t>((ql[l + 0] & 0x0fu) |
                                              (((qh[l] >> 0u) & 0x03u) << 4u)) -
                          32;
        const int8_t q2 = static_cast<int8_t>((ql[l + 32] & 0x0fu) |
                                              (((qh[l] >> 2u) & 0x03u) << 4u)) -
                          32;
        const int8_t q3 = static_cast<int8_t>(((ql[l + 0] >> 4u) & 0x0fu) |
                                              (((qh[l] >> 4u) & 0x03u) << 4u)) -
                          32;
        const int8_t q4 = static_cast<int8_t>(((ql[l + 32] >> 4u) & 0x0fu) |
                                              (((qh[l] >> 6u) & 0x03u) << 4u)) -
                          32;
        y[l + 0] = d * static_cast<float>(sc[is + 0]) * static_cast<float>(q1);
        y[l + 32] = d * static_cast<float>(sc[is + 2]) * static_cast<float>(q2);
        y[l + 64] = d * static_cast<float>(sc[is + 4]) * static_cast<float>(q3);
        y[l + 96] = d * static_cast<float>(sc[is + 6]) * static_cast<float>(q4);
      }

      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

inline void dequantize_row_q5_0(const block_q5_0 *x, float *y,
                                const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK5_0);

  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d);
    uint32_t qh = 0u;
    std::memcpy(&qh, x[i].qh.data(), sizeof(qh));

    for (uint64_t j = 0; j < (QK5_0 / 2u); ++j) {
      const uint8_t xh_0 = static_cast<uint8_t>(((qh >> j) & 1u) << 4u);
      const uint8_t xh_1 =
          static_cast<uint8_t>(((qh >> (j + (QK5_0 / 2u))) & 1u) << 4u);
      const int32_t x0 = static_cast<int32_t>((x[i].qs[j] & 0x0fu) | xh_0) - 16;
      const int32_t x1 = static_cast<int32_t>((x[i].qs[j] >> 4u) | xh_1) - 16;
      y[i * static_cast<int64_t>(QK5_0) + static_cast<int64_t>(j)] =
          static_cast<float>(x0) * d;
      y[i * static_cast<int64_t>(QK5_0) +
        static_cast<int64_t>(j + (QK5_0 / 2u))] = static_cast<float>(x1) * d;
    }
  }
}

inline void dequantize_row_q5_1(const block_q5_1 *x, float *y,
                                const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK5_1);

  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d);
    const float m = fp16_to_fp32(x[i].m);
    uint32_t qh = 0u;
    std::memcpy(&qh, x[i].qh.data(), sizeof(qh));

    for (uint64_t j = 0; j < (QK5_1 / 2u); ++j) {
      const uint8_t xh_0 = static_cast<uint8_t>(((qh >> j) & 1u) << 4u);
      const uint8_t xh_1 =
          static_cast<uint8_t>(((qh >> (j + (QK5_1 / 2u))) & 1u) << 4u);
      const uint8_t x0 = (x[i].qs[j] & 0x0fu) | xh_0;
      const uint8_t x1 = (x[i].qs[j] >> 4u) | xh_1;
      y[i * static_cast<int64_t>(QK5_1) + static_cast<int64_t>(j)] =
          static_cast<float>(x0) * d + m;
      y[i * static_cast<int64_t>(QK5_1) +
        static_cast<int64_t>(j + (QK5_1 / 2u))] =
          static_cast<float>(x1) * d + m;
    }
  }
}

inline void dequantize_row_q8_0(const block_q8_0 *x, float *y,
                                const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK8_0);

  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d);
    for (uint64_t j = 0; j < QK8_0; ++j) {
      y[i * static_cast<int64_t>(QK8_0) + static_cast<int64_t>(j)] =
          d * static_cast<float>(x[i].qs[j]);
    }
  }
}

inline int nearest_int(const float value) noexcept {
  float biased = value + 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &biased, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

inline void quantize_row_q4_0_ref(const float *x, block_q4_0 *y,
                                  const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK4_0);

  for (int64_t i = 0; i < nb; ++i) {
    float amax = 0.0f;
    float max = 0.0f;
    const float *block = x + i * static_cast<int64_t>(QK4_0);
    for (uint64_t j = 0; j < QK4_0; ++j) {
      const float value = block[j];
      const float abs_value = std::fabs(value);
      if (abs_value > amax) {
        amax = abs_value;
        max = value;
      }
    }

    const float d = max / -8.0f;
    const float inv_d = d != 0.0f ? 1.0f / d : 0.0f;
    y[i].d = fp32_to_fp16(d);

    for (uint64_t j = 0; j < (QK4_0 / 2u); ++j) {
      const float x0 = block[j] * inv_d;
      const float x1 = block[j + (QK4_0 / 2u)] * inv_d;
      const uint8_t xi0 =
          static_cast<uint8_t>(std::clamp(static_cast<int>(x0 + 8.5f), 0, 15));
      const uint8_t xi1 =
          static_cast<uint8_t>(std::clamp(static_cast<int>(x1 + 8.5f), 0, 15));
      y[i].qs[j] = static_cast<uint8_t>((xi0 & 0x0fu) | ((xi1 & 0x0fu) << 4u));
    }
  }
}

inline void quantize_row_q4_1_ref(const float *x, block_q4_1 *y,
                                  const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK4_1);

  for (int64_t i = 0; i < nb; ++i) {
    float min = std::numeric_limits<float>::max();
    float max = -std::numeric_limits<float>::max();
    const float *block = x + i * static_cast<int64_t>(QK4_1);
    for (uint64_t j = 0; j < QK4_1; ++j) {
      const float value = block[j];
      if (value < min) {
        min = value;
      }
      if (value > max) {
        max = value;
      }
    }

    const float d = (max - min) / 15.0f;
    const float inv_d = d != 0.0f ? 1.0f / d : 0.0f;
    y[i].d = fp32_to_fp16(d);
    y[i].m = fp32_to_fp16(min);

    for (uint64_t j = 0; j < (QK4_1 / 2u); ++j) {
      const float x0 = (block[j] - min) * inv_d;
      const float x1 = (block[j + (QK4_1 / 2u)] - min) * inv_d;
      const uint8_t xi0 =
          static_cast<uint8_t>(std::clamp(static_cast<int>(x0 + 0.5f), 0, 15));
      const uint8_t xi1 =
          static_cast<uint8_t>(std::clamp(static_cast<int>(x1 + 0.5f), 0, 15));
      y[i].qs[j] = static_cast<uint8_t>((xi0 & 0x0fu) | ((xi1 & 0x0fu) << 4u));
    }
  }
}

inline void quantize_row_q5_0_ref(const float *x, block_q5_0 *y,
                                  const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK5_0);

  for (int64_t i = 0; i < nb; ++i) {
    float amax = 0.0f;
    float max = 0.0f;
    const float *block = x + i * static_cast<int64_t>(QK5_0);
    for (uint64_t j = 0; j < QK5_0; ++j) {
      const float value = block[j];
      const float abs_value = std::fabs(value);
      if (abs_value > amax) {
        amax = abs_value;
        max = value;
      }
    }

    const float d = max / -16.0f;
    const float inv_d = d != 0.0f ? 1.0f / d : 0.0f;
    y[i].d = fp32_to_fp16(d);

    uint32_t qh = 0u;
    for (uint64_t j = 0; j < (QK5_0 / 2u); ++j) {
      const float x0 = block[j] * inv_d;
      const float x1 = block[j + (QK5_0 / 2u)] * inv_d;
      const uint8_t xi0 =
          static_cast<uint8_t>(std::clamp(static_cast<int>(x0 + 16.5f), 0, 31));
      const uint8_t xi1 =
          static_cast<uint8_t>(std::clamp(static_cast<int>(x1 + 16.5f), 0, 31));
      y[i].qs[j] = static_cast<uint8_t>((xi0 & 0x0fu) | ((xi1 & 0x0fu) << 4u));
      qh |= static_cast<uint32_t>((xi0 & 0x10u) >> 4u) << j;
      qh |= static_cast<uint32_t>((xi1 & 0x10u) >> 4u) << (j + (QK5_0 / 2u));
    }
    std::memcpy(y[i].qh.data(), &qh, sizeof(qh));
  }
}

inline void quantize_row_q8_0_strided(const float *x, const uint64_t stride,
                                      block_q8_0 *y, const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK8_0);

  for (int64_t i = 0; i < nb; ++i) {
    float amax = 0.0f;
    const float *block =
        x + i * static_cast<int64_t>(QK8_0) * static_cast<int64_t>(stride);
    for (uint64_t j = 0; j < QK8_0; ++j) {
      amax = std::max(amax, std::fabs(block[j * stride]));
    }

    const float d = amax / 127.0f;
    const float inv_d = d != 0.0f ? 1.0f / d : 0.0f;
    y[i].d = fp32_to_fp16(d);
    for (uint64_t j = 0; j < QK8_0; ++j) {
      const int quant = static_cast<int>(std::round(block[j * stride] * inv_d));
      y[i].qs[j] = static_cast<int8_t>(std::clamp(quant, -127, 127));
    }
  }
}

inline void quantize_row_q8_k_strided(const float *x, const uint64_t stride,
                                      block_q8_k *y, const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK_K);

  for (int64_t i = 0; i < nb; ++i) {
    float max = 0.0f;
    float amax = 0.0f;
    const float *block =
        x + i * static_cast<int64_t>(QK_K) * static_cast<int64_t>(stride);
    for (uint64_t j = 0; j < QK_K; ++j) {
      const float value = block[j * stride];
      const float abs_value = std::fabs(value);
      if (abs_value > amax) {
        amax = abs_value;
        max = value;
      }
    }

    if (amax == 0.0f) {
      y[i].d = 0.0f;
      y[i].qs.fill(0);
      y[i].bsums.fill(0);
      continue;
    }

    const float inv_scale = -127.0f / max;
    for (uint64_t j = 0; j < QK_K; ++j) {
      const int quant = nearest_int(inv_scale * block[j * stride]);
      y[i].qs[j] = static_cast<int8_t>(std::min(127, quant));
    }

    for (uint64_t j = 0; j < (QK_K / 16); ++j) {
      int sum = 0;
      for (uint64_t l = 0; l < 16; ++l) {
        sum += y[i].qs[j * 16 + l];
      }
      y[i].bsums[j] = static_cast<int16_t>(sum);
    }

    y[i].d = 1.0f / inv_scale;
  }
}

inline constexpr uint64_t
packed_q4_k_x8_group_count(const uint64_t rows) noexcept {
  return (rows + Q4_K_X8_ROWS - 1u) / Q4_K_X8_ROWS;
}

inline constexpr uint64_t
packed_q6_k_x8_group_count(const uint64_t rows) noexcept {
  return (rows + Q6_K_X8_ROWS - 1u) / Q6_K_X8_ROWS;
}

inline constexpr uint64_t
packed_q8_0_x4_group_count(const uint64_t rows) noexcept {
  return (rows + Q8_0_X4_ROWS - 1u) / Q8_0_X4_ROWS;
}

inline size_t packed_q4_k_x8_group_storage_bytes(const uint64_t cols) noexcept {
  if ((cols % QK_K) != 0u) {
    return 0u;
  }
  const uint64_t block_count = cols / QK_K;
  return static_cast<size_t>(block_count) * sizeof(block_q4_kx8);
}

inline size_t packed_q6_k_x8_group_storage_bytes(const uint64_t cols) noexcept {
  if ((cols % QK_K) != 0u) {
    return 0u;
  }
  const uint64_t block_count = cols / QK_K;
  return static_cast<size_t>(block_count) * sizeof(block_q6_kx8);
}

inline size_t packed_q8_0_x4_group_storage_bytes(const uint64_t cols) noexcept {
  if ((cols % QK8_0) != 0u) {
    return 0u;
  }
  const uint64_t block_count = cols / QK8_0;
  return static_cast<size_t>(block_count) * sizeof(block_q8_0x4);
}

inline size_t
prepared_q6_k_x8_q8_group_storage_bytes(const uint64_t cols) noexcept {
  if ((cols % QK_K) != 0u) {
    return 0u;
  }
  const uint64_t block_count = cols / QK_K;
  return static_cast<size_t>(block_count) * sizeof(block_q6_kx8_q8_prepared);
}

inline size_t
argmax_prepared_q6_k_x8_q8_group_storage_bytes(const uint64_t cols) noexcept {
  if ((cols % QK_K) != 0u) {
    return 0u;
  }
  const uint64_t block_count = cols / QK_K;
  return static_cast<size_t>(block_count) *
         sizeof(block_q6_kx8_q8_argmax_prepared);
}

inline block_q4_kx8
make_block_q4_k_x8(const block_q4_k *rows,
                   const uint64_t interleave_block_bytes) noexcept {
  block_q4_kx8 out = {};
  if (interleave_block_bytes == 0u ||
      (interleave_block_bytes != 4u && interleave_block_bytes != 8u) ||
      ((QK_K / 2u) % interleave_block_bytes) != 0u) {
    return out;
  }

  for (uint64_t row = 0; row < Q4_K_X8_ROWS; ++row) {
    out.d[row] = rows[row].d;
    out.dmin[row] = rows[row].dmin;
  }

  const uint64_t end = (QK_K * 4u) / interleave_block_bytes;
  for (uint64_t i = 0; i < end; ++i) {
    const uint64_t src_row = i % Q4_K_X8_ROWS;
    const uint64_t src_offset = (i / Q4_K_X8_ROWS) * interleave_block_bytes;
    const uint64_t dst_offset = i * interleave_block_bytes;
    std::memcpy(out.qs.data() + dst_offset,
                rows[src_row].qs.data() + src_offset,
                static_cast<size_t>(interleave_block_bytes));
  }

  for (uint64_t sb = 0; sb < (QK_K / 64u); ++sb) {
    for (uint64_t half = 0; half < 2u; ++half) {
      const uint64_t scale_index = sb * 2u + half;
      uint8_t *packed = out.scales.data() + (scale_index * Q4_K_X8_ROWS * 2u);
      for (uint64_t row = 0; row < Q4_K_X8_ROWS; ++row) {
        uint8_t scale = 0u;
        uint8_t min = 0u;
        get_scale_min_k4(static_cast<int>(scale_index), rows[row].scales.data(),
                         &scale, &min);
        packed[row] = min;
        packed[row + Q4_K_X8_ROWS] = scale;
      }
    }
  }

  return out;
}

inline block_q4_kx8 make_block_q4_k_x8_bl4(const block_q4_k *rows) noexcept {
  return make_block_q4_k_x8(rows, 4u);
}

inline block_q4_kx8 make_block_q4_k_x8_bl8(const block_q4_k *rows) noexcept {
  return make_block_q4_k_x8(rows, 8u);
}

inline block_q6_kx8 make_block_q6_k_x8(const block_q6_k *rows) noexcept {
  block_q6_kx8 out = {};
  constexpr uint64_t interleave_block_bytes = 8u;
  constexpr uint64_t end_low_bytes = (QK_K * 4u) / interleave_block_bytes;
  constexpr uint64_t end_high_bytes = end_low_bytes / 2u;
  constexpr uint64_t scale_count = QK_K / 16u;

  for (uint64_t row = 0; row < Q6_K_X8_ROWS; ++row) {
    out.d[row] = rows[row].d;
  }

  for (uint64_t i = 0; i < end_low_bytes; ++i) {
    const uint64_t src_row = i % Q6_K_X8_ROWS;
    const uint64_t src_offset = (i / Q6_K_X8_ROWS) * interleave_block_bytes;
    const uint64_t dst_offset = i * interleave_block_bytes;
    std::memcpy(out.ql.data() + dst_offset,
                rows[src_row].ql.data() + src_offset, interleave_block_bytes);
  }

  for (uint64_t i = 0; i < end_high_bytes; ++i) {
    const uint64_t src_row = i % Q6_K_X8_ROWS;
    const uint64_t src_offset = (i / Q6_K_X8_ROWS) * interleave_block_bytes;
    const uint64_t dst_offset = i * interleave_block_bytes;
    std::memcpy(out.qh.data() + dst_offset,
                rows[src_row].qh.data() + src_offset, interleave_block_bytes);
  }

  for (uint64_t row = 0; row < Q6_K_X8_ROWS; ++row) {
    for (uint64_t scale = 0; scale < scale_count; ++scale) {
      out.scales[scale * Q6_K_X8_ROWS + row] = rows[row].scales[scale];
    }
  }

  return out;
}

inline block_q8_0x4
make_block_q8_0_x4(const block_q8_0 *rows,
                   const uint64_t interleave_block_bytes) noexcept {
  block_q8_0x4 out = {};
  if (interleave_block_bytes == 0u || (QK8_0 % interleave_block_bytes) != 0u) {
    return out;
  }

  for (uint64_t row = 0; row < Q8_0_X4_ROWS; ++row) {
    out.d[row] = rows[row].d;
  }

  const uint64_t end = (QK8_0 * Q8_0_X4_ROWS) / interleave_block_bytes;
  for (uint64_t i = 0; i < end; ++i) {
    const uint64_t src_row = i % Q8_0_X4_ROWS;
    const uint64_t src_offset = (i / Q8_0_X4_ROWS) * interleave_block_bytes;
    const uint64_t dst_offset = i * interleave_block_bytes;
    std::memcpy(out.qs.data() + dst_offset,
                rows[src_row].qs.data() + src_offset,
                static_cast<size_t>(interleave_block_bytes));
  }

  return out;
}

inline block_q8_0x4 make_block_q8_0_x4_bl4(const block_q8_0 *rows) noexcept {
  return make_block_q8_0_x4(rows, 4u);
}

inline block_q8_0x4 make_block_q8_0_x4_bl8(const block_q8_0 *rows) noexcept {
  return make_block_q8_0_x4(rows, 8u);
}

inline block_q6_kx8_q8_prepared
make_block_q6_k_x8_q8_prepared(const block_q6_k *rows) noexcept {
  block_q6_kx8_q8_prepared out = {};

  for (uint64_t row = 0; row < Q6_K_X8_ROWS; ++row) {
    out.d[row] = rows[row].d;
    std::array<int8_t, QK_K> decoded = {};
    for (uint64_t half = 0; half < (QK_K / 128u); ++half) {
      const uint64_t half_value_base = half * 128u;
      const uint64_t ql_half_base = half * 64u;
      const uint64_t qh_half_base = half * 32u;
      for (uint64_t lane = 0; lane < 32u; ++lane) {
        const uint8_t qh_byte = rows[row].qh[qh_half_base + lane];
        const uint8_t ql_low = rows[row].ql[ql_half_base + lane];
        const uint8_t ql_high = rows[row].ql[ql_half_base + 32u + lane];
        decoded[half_value_base + lane + 0u] = static_cast<int8_t>(
            static_cast<int32_t>((ql_low & 0x0fu) |
                                 (((qh_byte >> 0u) & 0x03u) << 4u)) -
            32);
        decoded[half_value_base + lane + 32u] = static_cast<int8_t>(
            static_cast<int32_t>((ql_high & 0x0fu) |
                                 (((qh_byte >> 2u) & 0x03u) << 4u)) -
            32);
        decoded[half_value_base + lane + 64u] = static_cast<int8_t>(
            static_cast<int32_t>(((ql_low >> 4u) & 0x0fu) |
                                 (((qh_byte >> 4u) & 0x03u) << 4u)) -
            32);
        decoded[half_value_base + lane + 96u] = static_cast<int8_t>(
            static_cast<int32_t>(((ql_high >> 4u) & 0x0fu) |
                                 (((qh_byte >> 6u) & 0x03u) << 4u)) -
            32);
      }
    }

    for (uint64_t scale = 0; scale < (QK_K / 16u); ++scale) {
      const size_t scale_offset =
          static_cast<size_t>(scale) * Q6_K_X8_ROWS + row;
      out.scales[scale_offset] = rows[row].scales[scale];
#if defined(__ARM_FEATURE_MATMUL_INT8)
      const size_t scale_base =
          static_cast<size_t>(scale) * (Q6_K_X8_ROWS / 2u) * 32u;
      const size_t pair_index = row / 2u;
      const size_t pair_base = scale_base + pair_index * 32u;
      const size_t half_base = pair_base + (row % 2u) * 8u;
      for (uint64_t lane = 0; lane < 8u; ++lane) {
        out.qs[half_base + lane] = decoded[scale * 16u + lane];
        out.qs[half_base + 16u + lane] = decoded[scale * 16u + 8u + lane];
      }
#else
      for (uint64_t lane = 0; lane < 16u; ++lane) {
        const size_t qs_offset =
            (static_cast<size_t>(scale) * Q6_K_X8_ROWS + row) * 16u + lane;
        out.qs[qs_offset] = decoded[scale * 16u + lane];
      }
#endif
    }
  }

  return out;
}

inline block_q6_kx8_q8_argmax_prepared
make_block_q6_k_x8_q8_argmax_prepared(const block_q6_k *rows) noexcept {
  block_q6_kx8_q8_argmax_prepared out = {};

  for (uint64_t row = 0; row < Q6_K_X8_ROWS; ++row) {
    out.d[row] = fp16_to_fp32(rows[row].d);
    std::array<int8_t, QK_K> decoded = {};
    for (uint64_t half = 0; half < (QK_K / 128u); ++half) {
      const uint64_t half_value_base = half * 128u;
      const uint64_t ql_half_base = half * 64u;
      const uint64_t qh_half_base = half * 32u;
      for (uint64_t lane = 0; lane < 32u; ++lane) {
        const uint8_t qh_byte = rows[row].qh[qh_half_base + lane];
        const uint8_t ql_low = rows[row].ql[ql_half_base + lane];
        const uint8_t ql_high = rows[row].ql[ql_half_base + 32u + lane];
        decoded[half_value_base + lane + 0u] = static_cast<int8_t>(
            static_cast<int32_t>((ql_low & 0x0fu) |
                                 (((qh_byte >> 0u) & 0x03u) << 4u)) -
            32);
        decoded[half_value_base + lane + 32u] = static_cast<int8_t>(
            static_cast<int32_t>((ql_high & 0x0fu) |
                                 (((qh_byte >> 2u) & 0x03u) << 4u)) -
            32);
        decoded[half_value_base + lane + 64u] = static_cast<int8_t>(
            static_cast<int32_t>(((ql_low >> 4u) & 0x0fu) |
                                 (((qh_byte >> 4u) & 0x03u) << 4u)) -
            32);
        decoded[half_value_base + lane + 96u] = static_cast<int8_t>(
            static_cast<int32_t>(((ql_high >> 4u) & 0x0fu) |
                                 (((qh_byte >> 6u) & 0x03u) << 4u)) -
            32);
      }
    }

    for (uint64_t scale = 0; scale < (QK_K / 16u); ++scale) {
      out.scales[static_cast<size_t>(scale) * Q6_K_X8_ROWS + row] =
          rows[row].scales[scale];
      const size_t scale_base =
          static_cast<size_t>(scale) * (Q6_K_X8_ROWS / 2u) * 32u;
      const size_t pair_index = row / 2u;
      const size_t pair_base = scale_base + pair_index * 32u;
      const size_t half_base = pair_base + (row % 2u) * 8u;
      for (uint64_t lane = 0; lane < 8u; ++lane) {
        out.qs[half_base + lane] = decoded[scale * 16u + lane];
        out.qs[half_base + 16u + lane] = decoded[scale * 16u + 8u + lane];
      }
    }
  }

  return out;
}

inline bool pack_q6_k_rows_x8(const block_q6_k *src, const uint64_t rows,
                              const uint64_t cols, void *dst) noexcept {
  if (src == nullptr || dst == nullptr) {
    return false;
  }
  if ((cols % QK_K) != 0u) {
    return false;
  }

  const uint64_t block_count = cols / QK_K;
  const uint64_t group_count = packed_q6_k_x8_group_count(rows);
  auto *dst_blocks = static_cast<block_q6_kx8 *>(dst);
  for (uint64_t group = 0; group < group_count; ++group) {
    const uint64_t row_base = group * Q6_K_X8_ROWS;
    for (uint64_t block = 0; block < block_count; ++block) {
      std::array<block_q6_k, Q6_K_X8_ROWS> group_rows = {};
      for (uint64_t row = 0; row < Q6_K_X8_ROWS; ++row) {
        const uint64_t logical_row = row_base + row;
        if (logical_row < rows) {
          group_rows[row] = src[logical_row * block_count + block];
        }
      }
      dst_blocks[group * block_count + block] =
          make_block_q6_k_x8(group_rows.data());
    }
  }
  return true;
}

inline bool pack_q8_0_rows_x4(const block_q8_0 *src, const uint64_t rows,
                              const uint64_t cols, void *dst,
                              const uint64_t interleave_block_bytes) noexcept {
  if (src == nullptr || dst == nullptr) {
    return false;
  }
  if ((cols % QK8_0) != 0u ||
      (interleave_block_bytes != 4u && interleave_block_bytes != 8u)) {
    return false;
  }

  const uint64_t block_count = cols / QK8_0;
  const uint64_t group_count = packed_q8_0_x4_group_count(rows);
  auto *dst_blocks = static_cast<block_q8_0x4 *>(dst);
  for (uint64_t group = 0; group < group_count; ++group) {
    const uint64_t row_base = group * Q8_0_X4_ROWS;
    for (uint64_t block = 0; block < block_count; ++block) {
      std::array<block_q8_0, Q8_0_X4_ROWS> group_rows = {};
      for (uint64_t row = 0; row < Q8_0_X4_ROWS; ++row) {
        const uint64_t logical_row = row_base + row;
        if (logical_row < rows) {
          group_rows[row] = src[logical_row * block_count + block];
        }
      }
      dst_blocks[group * block_count + block] =
          make_block_q8_0_x4(group_rows.data(), interleave_block_bytes);
    }
  }
  return true;
}

inline bool pack_q8_0_rows_x4_bl4(const block_q8_0 *src, const uint64_t rows,
                                  const uint64_t cols, void *dst) noexcept {
  return pack_q8_0_rows_x4(src, rows, cols, dst, 4u);
}

inline bool pack_q8_0_rows_x4_bl8(const block_q8_0 *src, const uint64_t rows,
                                  const uint64_t cols, void *dst) noexcept {
  return pack_q8_0_rows_x4(src, rows, cols, dst, 8u);
}

inline bool pack_q4_k_rows_x8(const block_q4_k *src, const uint64_t rows,
                              const uint64_t cols, void *dst,
                              const uint64_t interleave_block_bytes) noexcept {
  if (src == nullptr || dst == nullptr) {
    return false;
  }
  if ((cols % QK_K) != 0u ||
      (interleave_block_bytes != 4u && interleave_block_bytes != 8u)) {
    return false;
  }

  const uint64_t block_count = cols / QK_K;
  const uint64_t group_count = packed_q4_k_x8_group_count(rows);
  auto *dst_blocks = static_cast<block_q4_kx8 *>(dst);
  for (uint64_t group = 0; group < group_count; ++group) {
    const uint64_t row_base = group * Q4_K_X8_ROWS;
    for (uint64_t block = 0; block < block_count; ++block) {
      std::array<block_q4_k, Q4_K_X8_ROWS> group_rows = {};
      for (uint64_t row = 0; row < Q4_K_X8_ROWS; ++row) {
        const uint64_t logical_row = row_base + row;
        if (logical_row < rows) {
          group_rows[row] = src[logical_row * block_count + block];
        }
      }
      dst_blocks[group * block_count + block] =
          make_block_q4_k_x8(group_rows.data(), interleave_block_bytes);
    }
  }
  return true;
}

inline bool pack_q4_k_rows_x8_bl4(const block_q4_k *src, const uint64_t rows,
                                  const uint64_t cols, void *dst) noexcept {
  return pack_q4_k_rows_x8(src, rows, cols, dst, 4u);
}

inline bool pack_q4_k_rows_x8_bl8(const block_q4_k *src, const uint64_t rows,
                                  const uint64_t cols, void *dst) noexcept {
  return pack_q4_k_rows_x8(src, rows, cols, dst, 8u);
}

inline bool pack_q6_k_rows_x8_q8_prepared(const block_q6_k *src,
                                          const uint64_t rows,
                                          const uint64_t cols,
                                          void *dst) noexcept {
  if (src == nullptr || dst == nullptr) {
    return false;
  }
  if ((cols % QK_K) != 0u) {
    return false;
  }

  const uint64_t block_count = cols / QK_K;
  const uint64_t group_count = packed_q6_k_x8_group_count(rows);
  auto *dst_blocks = static_cast<block_q6_kx8_q8_prepared *>(dst);
  for (uint64_t group = 0; group < group_count; ++group) {
    const uint64_t row_base = group * Q6_K_X8_ROWS;
    for (uint64_t block = 0; block < block_count; ++block) {
      std::array<block_q6_k, Q6_K_X8_ROWS> group_rows = {};
      for (uint64_t row = 0; row < Q6_K_X8_ROWS; ++row) {
        const uint64_t logical_row = row_base + row;
        if (logical_row < rows) {
          group_rows[row] = src[logical_row * block_count + block];
        }
      }
      dst_blocks[group * block_count + block] =
          make_block_q6_k_x8_q8_prepared(group_rows.data());
    }
  }
  return true;
}

inline bool pack_q6_k_rows_x8_q8_argmax_prepared(const block_q6_k *src,
                                                 const uint64_t rows,
                                                 const uint64_t cols,
                                                 void *dst) noexcept {
  if (src == nullptr || dst == nullptr) {
    return false;
  }
  if ((cols % QK_K) != 0u) {
    return false;
  }

  const uint64_t block_count = cols / QK_K;
  const uint64_t group_count = packed_q6_k_x8_group_count(rows);
  auto *dst_blocks = static_cast<block_q6_kx8_q8_argmax_prepared *>(dst);
  for (uint64_t group = 0; group < group_count; ++group) {
    const uint64_t row_base = group * Q6_K_X8_ROWS;
    for (uint64_t block = 0; block < block_count; ++block) {
      std::array<block_q6_k, Q6_K_X8_ROWS> group_rows = {};
      for (uint64_t row = 0; row < Q6_K_X8_ROWS; ++row) {
        const uint64_t logical_row = row_base + row;
        if (logical_row < rows) {
          group_rows[row] = src[logical_row * block_count + block];
        }
      }
      dst_blocks[group * block_count + block] =
          make_block_q6_k_x8_q8_argmax_prepared(group_rows.data());
    }
  }
  return true;
}

} // namespace quant

inline uint64_t select_u64(const bool choose_true, const uint64_t true_value,
                           const uint64_t false_value) noexcept {
  const uint64_t mask =
      static_cast<uint64_t>(0) - static_cast<uint64_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline bool select_bool(const bool choose_true, const bool true_value,
                        const bool false_value) noexcept {
  const std::array<bool, 2> values{false_value, true_value};
  return values[static_cast<size_t>(choose_true)];
}

template <class dtype_type>
inline uint8_t dtype_code(const dtype_type type) noexcept {
  return static_cast<uint8_t>(type);
}

inline bool is_q5_0_dtype(const uint8_t code) noexcept {
  return code == dtype_q5_0;
}

inline bool is_q4_0_dtype(const uint8_t code) noexcept {
  return code == dtype_q4_0;
}

inline bool is_q4_1_dtype(const uint8_t code) noexcept {
  return code == dtype_q4_1;
}

inline bool is_q5_1_dtype(const uint8_t code) noexcept {
  return code == dtype_q5_1;
}

inline bool is_q8_0_dtype(const uint8_t code) noexcept {
  return code == dtype_q8_0;
}

inline bool is_quantized_32_dtype(const uint8_t code) noexcept {
  return is_q4_0_dtype(code) || is_q4_1_dtype(code) || is_q5_0_dtype(code) ||
         is_q5_1_dtype(code) || is_q8_0_dtype(code);
}

inline bool is_q5_k_dtype(const uint8_t code) noexcept {
  return code == dtype_q5_k;
}

inline bool is_quantized_k_dtype(const uint8_t code) noexcept {
  return code == dtype_q2_k || code == dtype_q3_k || code == dtype_q4_k ||
         code == dtype_q6_k;
}

inline bool is_native_quantized_dtype(const uint8_t code) noexcept {
  return is_q4_0_dtype(code) || is_q4_1_dtype(code) || is_q5_0_dtype(code) ||
         is_q8_0_dtype(code) || is_quantized_k_dtype(code);
}

inline uint64_t quantized_block_size(const uint8_t code) noexcept {
  if (is_q4_0_dtype(code)) {
    return quant::QK4_0;
  }
  if (is_q4_1_dtype(code)) {
    return quant::QK4_1;
  }
  if (is_q5_0_dtype(code)) {
    return quant::QK5_0;
  }
  if (is_q5_1_dtype(code)) {
    return quant::QK5_1;
  }
  if (is_q8_0_dtype(code)) {
    return quant::QK8_0;
  }
  if (is_quantized_k_dtype(code)) {
    return quant::QK_K;
  }
  return 0u;
}

inline uint64_t max_quantized_block_count(const uint8_t code) noexcept {
  if (is_quantized_32_dtype(code)) {
    return quant::MAX_Q8_0_BLOCKS;
  }
  if (is_quantized_k_dtype(code)) {
    return quant::MAX_Q8_K_BLOCKS;
  }
  return 0u;
}

inline bool is_rhs_q8_k_dtype(const uint8_t code) noexcept {
  return code == dtype_q8_k;
}

inline bool is_rhs_q8_k_x4_dtype(const uint8_t code) noexcept {
  return code == dtype_q8_k_x4;
}

inline bool is_rhs_q8_k_x8_dtype(const uint8_t code) noexcept {
  return code == dtype_q8_k_x8;
}

inline bool is_rhs_q8_0_dtype(const uint8_t code) noexcept {
  return code == dtype_q8_0;
}

inline bool is_packed_q8_0_vector_bl4_dtype(const uint8_t code) noexcept {
  return code == dtype_q8_0_x4_bl4;
}

inline bool is_packed_q8_0_vector_bl8_dtype(const uint8_t code) noexcept {
  return code == dtype_q8_0_x4_bl8;
}

inline bool is_packed_q8_0_vector_dtype(const uint8_t code) noexcept {
  return is_packed_q8_0_vector_bl4_dtype(code) ||
         is_packed_q8_0_vector_bl8_dtype(code);
}

inline bool is_packed_q4_vector_bl4_dtype(const uint8_t code) noexcept {
  return code == dtype_q4_k_x8_bl4;
}

inline bool is_packed_q4_vector_bl8_dtype(const uint8_t code) noexcept {
  return code == dtype_q4_k_x8_bl8;
}

inline bool is_packed_q4_vector_dtype(const uint8_t code) noexcept {
  return is_packed_q4_vector_bl4_dtype(code) ||
         is_packed_q4_vector_bl8_dtype(code);
}

inline bool is_packed_q6_vector_dtype(const uint8_t code) noexcept {
  return code == dtype_q6_k_x8;
}

inline bool is_prepared_q6_vector_q8_rhs_dtype(const uint8_t code) noexcept {
  return code == dtype_q6_k_x8_q8_prepared;
}

inline bool
is_argmax_prepared_q6_vector_q8_rhs_dtype(const uint8_t code) noexcept {
  return code == dtype_q6_k_x8_q8_argmax_prepared;
}

inline bool is_supported_dtype(const uint8_t code) noexcept {
  return code == dtype_f32 || code == dtype_f16 ||
         is_native_quantized_dtype(code) || is_rhs_q8_k_dtype(code) ||
         is_rhs_q8_k_x4_dtype(code) || is_rhs_q8_k_x8_dtype(code) ||
         is_packed_q8_0_vector_dtype(code) || is_packed_q4_vector_dtype(code) ||
         is_packed_q6_vector_dtype(code) ||
         is_prepared_q6_vector_q8_rhs_dtype(code) ||
         is_argmax_prepared_q6_vector_q8_rhs_dtype(code);
}

inline size_t dtype_size_bytes(const uint8_t code) noexcept {
  if (code == dtype_f32) {
    return sizeof(float);
  }
  if (code == dtype_f16) {
    return sizeof(uint16_t);
  }
  if (code == dtype_bf16) {
    return sizeof(uint16_t);
  }
  if (code == dtype_i32) {
    return sizeof(int32_t);
  }
  if (code == dtype_q4_0) {
    return sizeof(quant::block_q4_0) / quant::QK4_0;
  }
  if (code == dtype_q4_1) {
    return sizeof(quant::block_q4_1) / quant::QK4_1;
  }
  if (code == dtype_q5_0) {
    return sizeof(quant::block_q5_0) / quant::QK5_0;
  }
  if (code == dtype_q5_1) {
    return sizeof(quant::block_q5_1) / quant::QK5_1;
  }
  if (code == dtype_q8_0) {
    return sizeof(quant::block_q8_0) / quant::QK8_0;
  }
  if (code == dtype_q2_k) {
    return sizeof(quant::block_q2_k) / quant::QK_K;
  }
  if (code == dtype_q3_k) {
    return sizeof(quant::block_q3_k) / quant::QK_K;
  }
  if (code == dtype_q4_k) {
    return sizeof(quant::block_q4_k) / quant::QK_K;
  }
  if (code == dtype_q5_k) {
    return sizeof(quant::block_q5_k) / quant::QK_K;
  }
  if (code == dtype_q6_k) {
    return sizeof(quant::block_q6_k) / quant::QK_K;
  }
  if (code == dtype_q8_k) {
    return sizeof(quant::block_q8_k) / quant::QK_K;
  }
  if (code == dtype_q8_k_x4) {
    return sizeof(quant::block_q8_k) / quant::QK_K;
  }
  if (code == dtype_q8_k_x8) {
    return sizeof(quant::block_q8_k) / quant::QK_K;
  }
  return 0u;
}

inline size_t quantized_row_storage_bytes(const uint8_t code,
                                          const uint64_t cols) noexcept {
  if (code == dtype_q4_0) {
    if ((cols % quant::QK4_0) != 0u) {
      return 0u;
    }
    const uint64_t block_count = cols / quant::QK4_0;
    return static_cast<size_t>(block_count) * sizeof(quant::block_q4_0);
  }
  if (code == dtype_q4_1) {
    if ((cols % quant::QK4_1) != 0u) {
      return 0u;
    }
    const uint64_t block_count = cols / quant::QK4_1;
    return static_cast<size_t>(block_count) * sizeof(quant::block_q4_1);
  }
  if (code == dtype_q5_0) {
    if ((cols % quant::QK5_0) != 0u) {
      return 0u;
    }
    const uint64_t block_count = cols / quant::QK5_0;
    return static_cast<size_t>(block_count) * sizeof(quant::block_q5_0);
  }
  if (code == dtype_q5_1) {
    if ((cols % quant::QK5_1) != 0u) {
      return 0u;
    }
    const uint64_t block_count = cols / quant::QK5_1;
    return static_cast<size_t>(block_count) * sizeof(quant::block_q5_1);
  }
  if (code == dtype_q8_0) {
    if ((cols % quant::QK8_0) != 0u) {
      return 0u;
    }
    const uint64_t block_count = cols / quant::QK8_0;
    return static_cast<size_t>(block_count) * sizeof(quant::block_q8_0);
  }
  if ((cols % quant::QK_K) != 0u) {
    return 0u;
  }
  const uint64_t block_count = cols / quant::QK_K;
  if (code == dtype_q2_k) {
    return static_cast<size_t>(block_count) * sizeof(quant::block_q2_k);
  }
  if (code == dtype_q3_k) {
    return static_cast<size_t>(block_count) * sizeof(quant::block_q3_k);
  }
  if (code == dtype_q4_k) {
    return static_cast<size_t>(block_count) * sizeof(quant::block_q4_k);
  }
  if (code == dtype_q5_k) {
    return static_cast<size_t>(block_count) * sizeof(quant::block_q5_k);
  }
  if (code == dtype_q6_k) {
    return static_cast<size_t>(block_count) * sizeof(quant::block_q6_k);
  }
  if (code == dtype_q8_k) {
    return static_cast<size_t>(block_count) * sizeof(quant::block_q8_k);
  }
  if (code == dtype_q8_k_x4) {
    return static_cast<size_t>(block_count) * sizeof(quant::block_q8_k);
  }
  if (code == dtype_q8_k_x8) {
    return static_cast<size_t>(block_count) * sizeof(quant::block_q8_k);
  }
  if (code == dtype_q4_k_x8_bl4 || code == dtype_q4_k_x8_bl8) {
    return quant::packed_q4_k_x8_group_storage_bytes(cols);
  }
  if (code == dtype_q6_k_x8_q8_prepared) {
    return quant::prepared_q6_k_x8_q8_group_storage_bytes(cols);
  }
  if (code == dtype_q6_k_x8_q8_argmax_prepared) {
    return quant::argmax_prepared_q6_k_x8_q8_group_storage_bytes(cols);
  }
  return 0u;
}

inline size_t row_storage_bytes_for_dtype(const uint8_t code,
                                          const uint64_t cols) noexcept {
  if (code == dtype_f32) {
    return static_cast<size_t>(cols) * sizeof(float);
  }
  if (code == dtype_q8_0_x4_bl4 || code == dtype_q8_0_x4_bl8) {
    return quant::packed_q8_0_x4_group_storage_bytes(cols);
  }
  if (code == dtype_q4_k_x8_bl4 || code == dtype_q4_k_x8_bl8) {
    return quant::packed_q4_k_x8_group_storage_bytes(cols);
  }
  if (code == dtype_q6_k_x8) {
    return quant::packed_q6_k_x8_group_storage_bytes(cols);
  }
  if (code == dtype_q6_k_x8_q8_prepared) {
    return quant::prepared_q6_k_x8_q8_group_storage_bytes(cols);
  }
  if (code == dtype_q6_k_x8_q8_argmax_prepared) {
    return quant::argmax_prepared_q6_k_x8_q8_group_storage_bytes(cols);
  }
  return quantized_row_storage_bytes(code, cols);
}

template <class tensor_type>
inline uint64_t tensor_element_count(const tensor_type &tensor) noexcept {
  uint64_t count = 1;
  for (size_t i = 0; i < 4; ++i) {
    count *= tensor.ne[i];
  }
  return count;
}

template <class tensor_type>
inline uint64_t tensor_stride_bytes(const tensor_type &tensor,
                                    const size_t dim) noexcept {
  uint64_t stride = dtype_size_bytes(dtype_code(tensor.type));
  for (size_t i = 0; i < dim; ++i) {
    stride *= tensor.ne[i];
  }
  const std::array<uint64_t, 2> candidates{stride, tensor.nb[dim]};
  return candidates[static_cast<size_t>(tensor.nb[0] != 0)];
}

template <class tensor_type>
inline bool has_valid_tensor_layout(const tensor_type &tensor) noexcept {
  const uint64_t elem_size = dtype_size_bytes(dtype_code(tensor.type));
  const bool elem_valid = elem_size != 0u;
  const bool explicit_stride = tensor.nb[0] != 0u;
  const bool aligned_stride = elem_valid && explicit_stride &&
                              tensor.nb[0] >= elem_size &&
                              (tensor.nb[0] % elem_size) == 0u;

  bool dims_valid = true;
  for (size_t i = 0; i < 4; ++i) {
    const bool invalid_dim = tensor.ne[i] > 1 && tensor.nb[i] == 0;
    dims_valid = dims_valid && !invalid_dim;
  }

  return elem_valid && (!explicit_stride || (aligned_stride && dims_valid));
}

template <class tensor_type>
inline bool is_dense_contiguous(const tensor_type &tensor) noexcept {
  const bool valid_layout = has_valid_tensor_layout(tensor);
  uint64_t expected = dtype_size_bytes(dtype_code(tensor.type));
  bool matches = true;
  for (size_t i = 0; i < 4; ++i) {
    matches = matches && tensor_stride_bytes(tensor, i) == expected;
    expected *= tensor.ne[i];
  }
  return valid_layout && matches;
}

template <class tensor_type>
inline size_t tensor_offset_bytes(const tensor_type &tensor,
                                  const uint64_t idx) noexcept {
  uint64_t remaining = idx;
  size_t offset = 0;
  bool dims_active = true;
  for (size_t d = 0; d < 4; ++d) {
    const bool dim_non_zero = tensor.ne[d] != 0u;
    const bool step_active = dims_active && dim_non_zero;
    const uint64_t dim = select_u64(step_active, tensor.ne[d], 1u);
    const uint64_t coord = remaining % dim;
    const uint64_t stride = tensor_stride_bytes(tensor, d);
    const uint64_t offset_step = coord * stride;
    offset += static_cast<size_t>(select_u64(step_active, offset_step, 0u));
    remaining = select_u64(step_active, remaining / dim, remaining);
    dims_active = dims_active && dim_non_zero;
  }
  return offset;
}

template <class tensor_type>
inline size_t tensor_offset_bytes(const tensor_type &tensor, const uint64_t i0,
                                  const uint64_t i1, const uint64_t i2 = 0,
                                  const uint64_t i3 = 0) noexcept {
  return static_cast<size_t>(i0 * tensor_stride_bytes(tensor, 0) +
                             i1 * tensor_stride_bytes(tensor, 1) +
                             i2 * tensor_stride_bytes(tensor, 2) +
                             i3 * tensor_stride_bytes(tensor, 3));
}

template <class request_type>
inline constexpr bool requires_src1_v =
    std::is_same_v<request_type, event::op_add> ||
    std::is_same_v<request_type, event::op_add_id> ||
    std::is_same_v<request_type, event::op_add1> ||
    std::is_same_v<request_type, event::op_acc> ||
    std::is_same_v<request_type, event::op_sub> ||
    std::is_same_v<request_type, event::op_mul> ||
    std::is_same_v<request_type, event::op_div> ||
    std::is_same_v<request_type, event::op_mul_mat> ||
    std::is_same_v<request_type, event::op_mul_mat_argmax>;

template <class request_type>
inline bool has_required_src0(const request_type &request) noexcept {
  // op_get_rows shares the native-quantized row-layout contract below; the
  // generic fallback's per-element layout check cannot express sub-byte
  // quantized rows (their per-element size truncates to zero).
  if constexpr (std::is_same_v<request_type, event::op_mul_mat> ||
                std::is_same_v<request_type, event::op_mul_mat_argmax> ||
                std::is_same_v<request_type, event::op_get_rows>) {
    const uint8_t src0_type = dtype_code(request.src0.type);
    if (is_packed_q8_0_vector_dtype(src0_type)) {
      const uint64_t cols = request.src0.ne[0];
      const uint64_t rows = request.src0.ne[1];
      const uint64_t group_count = quant::packed_q8_0_x4_group_count(rows);
      const size_t group_bytes =
          quant::packed_q8_0_x4_group_storage_bytes(cols);
      return request.src0.data != nullptr && cols != 0u && rows != 0u &&
             group_bytes != 0u && request.src0.ne[2] == 1u &&
             request.src0.ne[3] == 1u && request.src0.nb[0] == 1u &&
             request.src0.nb[1] == group_bytes &&
             request.src0.nb[2] == group_bytes * group_count &&
             request.src0.nb[3] == request.src0.nb[2];
    }
    if (is_packed_q4_vector_dtype(src0_type)) {
      const uint64_t cols = request.src0.ne[0];
      const uint64_t rows = request.src0.ne[1];
      const uint64_t group_count = quant::packed_q4_k_x8_group_count(rows);
      const size_t group_bytes =
          quant::packed_q4_k_x8_group_storage_bytes(cols);
      return request.src0.data != nullptr && cols != 0u && rows != 0u &&
             group_bytes != 0u && request.src0.ne[2] == 1u &&
             request.src0.ne[3] == 1u && request.src0.nb[0] == 1u &&
             request.src0.nb[1] == group_bytes &&
             request.src0.nb[2] == group_bytes * group_count &&
             request.src0.nb[3] == request.src0.nb[2];
    }
    if (is_prepared_q6_vector_q8_rhs_dtype(src0_type) ||
        is_argmax_prepared_q6_vector_q8_rhs_dtype(src0_type)) {
      const uint64_t cols = request.src0.ne[0];
      const uint64_t rows = request.src0.ne[1];
      const uint64_t group_count = quant::packed_q6_k_x8_group_count(rows);
      const size_t group_bytes =
          is_prepared_q6_vector_q8_rhs_dtype(src0_type)
              ? quant::prepared_q6_k_x8_q8_group_storage_bytes(cols)
              : quant::argmax_prepared_q6_k_x8_q8_group_storage_bytes(cols);
      return request.src0.data != nullptr && cols != 0u && rows != 0u &&
             group_bytes != 0u && request.src0.ne[2] == 1u &&
             request.src0.ne[3] == 1u && request.src0.nb[0] == 1u &&
             request.src0.nb[1] == group_bytes &&
             request.src0.nb[2] == group_bytes * group_count &&
             request.src0.nb[3] == request.src0.nb[2];
    }
    if (is_packed_q6_vector_dtype(src0_type)) {
      const uint64_t cols = request.src0.ne[0];
      const uint64_t rows = request.src0.ne[1];
      const uint64_t group_count = quant::packed_q6_k_x8_group_count(rows);
      const size_t group_bytes =
          quant::packed_q6_k_x8_group_storage_bytes(cols);
      return request.src0.data != nullptr && cols != 0u && rows != 0u &&
             group_bytes != 0u && request.src0.ne[2] == 1u &&
             request.src0.ne[3] == 1u && request.src0.nb[0] == 1u &&
             request.src0.nb[1] == group_bytes &&
             request.src0.nb[2] == group_bytes * group_count &&
             request.src0.nb[3] == request.src0.nb[2];
    }
    if (is_native_quantized_dtype(src0_type)) {
      const uint64_t cols = request.src0.ne[0];
      const uint64_t rows = request.src0.ne[1];
      const size_t row_bytes = quantized_row_storage_bytes(src0_type, cols);
      return request.src0.data != nullptr && row_bytes != 0u && rows != 0u &&
             request.src0.ne[2] == 1u && request.src0.ne[3] == 1u &&
             request.src0.nb[0] == 1u && request.src0.nb[1] == row_bytes &&
             request.src0.nb[2] == row_bytes * rows &&
             request.src0.nb[3] == request.src0.nb[2];
    }
  }
  return request.src0.data != nullptr &&
         is_supported_dtype(dtype_code(request.src0.type)) &&
         has_valid_tensor_layout(request.src0) &&
         tensor_element_count(request.src0) > 0;
}

template <class request_type>
inline bool has_required_src1(const request_type &request) noexcept {
  if constexpr (!requires_src1_v<request_type>) {
    return true;
  }
  if constexpr (std::is_same_v<request_type, event::op_mul_mat>) {
    const uint8_t src1_type = dtype_code(request.src1.type);
    if (is_packed_q8_0_vector_dtype(src1_type)) {
      const uint64_t rows = request.src1.ne[0];
      const uint64_t cols = request.src1.ne[1];
      const uint64_t group_count = quant::packed_q8_0_x4_group_count(rows);
      const size_t group_bytes =
          quant::packed_q8_0_x4_group_storage_bytes(cols);
      return request.src1.data != nullptr && cols != 0u && rows != 0u &&
             group_bytes != 0u && request.src1.ne[2] == 1u &&
             request.src1.ne[3] == 1u && request.src1.nb[0] == 1u &&
             request.src1.nb[1] == group_bytes &&
             request.src1.nb[2] == group_bytes * group_count &&
             request.src1.nb[3] == request.src1.nb[2];
    }
  }
  return request.src1.data != nullptr &&
         is_supported_dtype(dtype_code(request.src1.type)) &&
         has_valid_tensor_layout(request.src1) &&
         tensor_element_count(request.src1) > 0;
}

template <class request_type>
inline bool has_required_dst(const request_type &request) noexcept {
  return request.dst.data != nullptr &&
         is_supported_dtype(dtype_code(request.dst.type)) &&
         has_valid_tensor_layout(request.dst) &&
         tensor_element_count(request.dst) > 0;
}

template <class request_type>
inline bool validate_dispatch_request(const request_type &request) noexcept {
  const bool has_required_buffers = has_required_src0(request) &&
                                    has_required_src1(request) &&
                                    has_required_dst(request);
  const bool has_valid_params =
      request.op_params_size <= request.op_params.size();
  return has_required_buffers && has_valid_params;
}

template <class tensor_type>
inline float read_f32(const tensor_type &tensor, const uint64_t idx) noexcept {
  const bool dense = is_dense_contiguous(tensor);
  const float *data = static_cast<const float *>(tensor.data);
  const char *base = static_cast<const char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, idx);
  const char *dense_src = reinterpret_cast<const char *>(data + idx);
  const char *sparse_src = base + offset;
  const std::array<const char *, 2> srcs{sparse_src, dense_src};
  float out = 0.0f;
  std::memcpy(&out, srcs[static_cast<size_t>(dense)], sizeof(out));
  return out;
}

template <class tensor_type>
inline void write_f32(const tensor_type &tensor, const uint64_t idx,
                      const float value) noexcept {
  const bool dense = is_dense_contiguous(tensor);
  float *data = static_cast<float *>(tensor.data);
  char *base = static_cast<char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, idx);
  char *dense_dst = reinterpret_cast<char *>(data + idx);
  char *sparse_dst = base + offset;
  const std::array<char *, 2> dsts{sparse_dst, dense_dst};
  std::memcpy(dsts[static_cast<size_t>(dense)], &value, sizeof(value));
}

template <class tensor_type>
inline float read_f32_at(const tensor_type &tensor, const uint64_t i0,
                         const uint64_t i1, const uint64_t i2 = 0,
                         const uint64_t i3 = 0) noexcept {
  float out = 0.0f;
  const char *base = static_cast<const char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, i0, i1, i2, i3);
  std::memcpy(&out, base + offset, sizeof(out));
  return out;
}

template <class tensor_type>
inline void write_f32_at(const tensor_type &tensor, const uint64_t i0,
                         const uint64_t i1, const float value,
                         const uint64_t i2 = 0,
                         const uint64_t i3 = 0) noexcept {
  char *base = static_cast<char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, i0, i1, i2, i3);
  std::memcpy(base + offset, &value, sizeof(value));
}

template <class tensor_type>
inline const void *tensor_row_ptr_void(const tensor_type &tensor,
                                       const uint64_t row1,
                                       const uint64_t row2) noexcept {
  const auto *base = static_cast<const char *>(tensor.data);
  return base + row1 * tensor.nb[1] + row2 * tensor.nb[2];
}

template <class elem_type, class tensor_type>
inline const elem_type *tensor_row_ptr_as(const tensor_type &tensor,
                                          const uint64_t row1,
                                          const uint64_t row2) noexcept {
  return reinterpret_cast<const elem_type *>(
      tensor_row_ptr_void(tensor, row1, row2));
}

template <class tensor_type>
inline const float *tensor_row_ptr(const tensor_type &tensor,
                                   const uint64_t row1,
                                   const uint64_t row2) noexcept {
  return tensor_row_ptr_as<float>(tensor, row1, row2);
}

template <class tensor_type>
inline void *tensor_row_ptr_mut_void(const tensor_type &tensor,
                                     const uint64_t row1,
                                     const uint64_t row2) noexcept {
  auto *base = static_cast<char *>(tensor.data);
  return base + row1 * tensor.nb[1] + row2 * tensor.nb[2];
}

template <class elem_type, class tensor_type>
inline elem_type *tensor_row_ptr_mut_as(const tensor_type &tensor,
                                        const uint64_t row1,
                                        const uint64_t row2) noexcept {
  return reinterpret_cast<elem_type *>(
      tensor_row_ptr_mut_void(tensor, row1, row2));
}

template <class tensor_type>
inline float *tensor_row_ptr_mut(const tensor_type &tensor, const uint64_t row1,
                                 const uint64_t row2) noexcept {
  return tensor_row_ptr_mut_as<float>(tensor, row1, row2);
}

#if defined(__ARM_NEON) && defined(__aarch64__)
inline float32x4_t expf4_ggml(float32x4_t x) noexcept {
  const float32x4_t r = vdupq_n_f32(0x1.8p23f);
  const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
  const float32x4_t n = vsubq_f32(z, r);
  const float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n,
                                  vdupq_n_f32(0x1.7f7d1cp-20f));
  const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
  const float32x4_t k = vreinterpretq_f32_u32(
      vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0f))));
  const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126.0f));
  const float32x4_t u = vmulq_f32(b, b);
  const float32x4_t j =
      vfmaq_f32(vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
                vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f),
                                    vdupq_n_f32(0x1.555e66p-3f), b),
                          vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f),
                                    vdupq_n_f32(0x1.0e4020p-7f), b),
                          u),
                u);
  if (!vpaddd_u64(vreinterpretq_u64_u32(c))) {
    return vfmaq_f32(k, j, k);
  }
  const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
  const float32x4_t s1 =
      vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
  const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
  return vbslq_f32(
      vcagtq_f32(n, vdupq_n_f32(192.0f)), vmulq_f32(s1, s1),
      vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}
#endif

inline double exp_and_sum_ggml_f32(const float *src, float *dst,
                                   const uint64_t count,
                                   const float max_value) noexcept {
#if defined(__ARM_NEON) && defined(__aarch64__)
  uint64_t idx = 0u;
  double sum = 0.0;
  const float32x4_t max_vec = vdupq_n_f32(max_value);
  for (; idx + 4u <= count; idx += 4u) {
    const float32x4_t values =
        expf4_ggml(vsubq_f32(vld1q_f32(src + idx), max_vec));
    vst1q_f32(dst + idx, values);
    sum += static_cast<double>(vaddvq_f32(values));
  }
  for (; idx < count; ++idx) {
    const float value = ::expf(src[idx] - max_value);
    dst[idx] = value;
    sum += static_cast<double>(value);
  }
  return sum;
#else
  double sum = 0.0;
  for (uint64_t idx = 0u; idx < count; ++idx) {
    const float value = ::expf(src[idx] - max_value);
    dst[idx] = value;
    sum += static_cast<double>(value);
  }
  return sum;
#endif
}

inline float dot_product_ggml_f16_scores(const float *lhs, const float *rhs,
                                         const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  alignas(16) uint16_t lhs_f16[32] = {};
  alignas(16) uint16_t rhs_f16[32] = {};
  float16x8_t sum[4] = {
      vdupq_n_f16(0.0f),
      vdupq_n_f16(0.0f),
      vdupq_n_f16(0.0f),
      vdupq_n_f16(0.0f),
  };
  uint64_t idx = 0u;
  for (; idx + 32u <= count; idx += 32u) {
    for (uint64_t lane = 0u; lane < 32u; ++lane) {
      lhs_f16[lane] = quant::fp32_to_fp16(lhs[idx + lane]);
      rhs_f16[lane] = quant::fp32_to_fp16(rhs[idx + lane]);
    }

    sum[0] = vfmaq_f16(
        sum[0], vld1q_f16(reinterpret_cast<const __fp16 *>(lhs_f16 + 0u)),
        vld1q_f16(reinterpret_cast<const __fp16 *>(rhs_f16 + 0u)));
    sum[1] = vfmaq_f16(
        sum[1], vld1q_f16(reinterpret_cast<const __fp16 *>(lhs_f16 + 8u)),
        vld1q_f16(reinterpret_cast<const __fp16 *>(rhs_f16 + 8u)));
    sum[2] = vfmaq_f16(
        sum[2], vld1q_f16(reinterpret_cast<const __fp16 *>(lhs_f16 + 16u)),
        vld1q_f16(reinterpret_cast<const __fp16 *>(rhs_f16 + 16u)));
    sum[3] = vfmaq_f16(
        sum[3], vld1q_f16(reinterpret_cast<const __fp16 *>(lhs_f16 + 24u)),
        vld1q_f16(reinterpret_cast<const __fp16 *>(rhs_f16 + 24u)));
  }

  double sumf = 0.0;
  if (idx != 0u) {
    int offset = 2;
    for (int i = 0; i < offset; ++i) {
      sum[i] = vaddq_f16(sum[i], sum[offset + i]);
    }
    offset >>= 1;
    for (int i = 0; i < offset; ++i) {
      sum[i] = vaddq_f16(sum[i], sum[offset + i]);
    }

    const float32x4_t low = vcvt_f32_f16(vget_low_f16(sum[0]));
    const float32x4_t high = vcvt_f32_f16(vget_high_f16(sum[0]));
    sumf = static_cast<double>(vaddvq_f32(vaddq_f32(low, high)));
  }

  for (; idx < count; ++idx) {
    sumf +=
        static_cast<double>(
            quant::fp16_to_fp32(quant::fp32_to_fp16(lhs[idx]))) *
        static_cast<double>(quant::fp16_to_fp32(quant::fp32_to_fp16(rhs[idx])));
  }
  return static_cast<float>(sumf);
#endif
#endif

  double scalar_sum = 0.0;
  for (uint64_t idx = 0u; idx < count; ++idx) {
    scalar_sum +=
        static_cast<double>(
            quant::fp16_to_fp32(quant::fp32_to_fp16(lhs[idx]))) *
        static_cast<double>(quant::fp16_to_fp32(quant::fp32_to_fp16(rhs[idx])));
  }
  return static_cast<float>(scalar_sum);
}

inline float dot_product_f16_f16_scores(const uint16_t *lhs,
                                        const uint16_t *rhs,
                                        const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  float16x8_t sum[4] = {
      vdupq_n_f16(0.0f),
      vdupq_n_f16(0.0f),
      vdupq_n_f16(0.0f),
      vdupq_n_f16(0.0f),
  };
  uint64_t idx = 0u;
  const auto *lhs_f16 = reinterpret_cast<const __fp16 *>(lhs);
  const auto *rhs_f16 = reinterpret_cast<const __fp16 *>(rhs);
  for (; idx + 32u <= count; idx += 32u) {
    sum[0] = vfmaq_f16(sum[0], vld1q_f16(lhs_f16 + idx + 0u),
                       vld1q_f16(rhs_f16 + idx + 0u));
    sum[1] = vfmaq_f16(sum[1], vld1q_f16(lhs_f16 + idx + 8u),
                       vld1q_f16(rhs_f16 + idx + 8u));
    sum[2] = vfmaq_f16(sum[2], vld1q_f16(lhs_f16 + idx + 16u),
                       vld1q_f16(rhs_f16 + idx + 16u));
    sum[3] = vfmaq_f16(sum[3], vld1q_f16(lhs_f16 + idx + 24u),
                       vld1q_f16(rhs_f16 + idx + 24u));
  }

  double sumf = 0.0;
  if (idx != 0u) {
    int offset = 2;
    for (int i = 0; i < offset; ++i) {
      sum[i] = vaddq_f16(sum[i], sum[offset + i]);
    }
    offset >>= 1;
    for (int i = 0; i < offset; ++i) {
      sum[i] = vaddq_f16(sum[i], sum[offset + i]);
    }

    const float32x4_t low = vcvt_f32_f16(vget_low_f16(sum[0]));
    const float32x4_t high = vcvt_f32_f16(vget_high_f16(sum[0]));
    sumf = static_cast<double>(vaddvq_f32(vaddq_f32(low, high)));
  }

  for (; idx < count; ++idx) {
    sumf += static_cast<double>(quant::fp16_to_fp32(lhs[idx])) *
            static_cast<double>(quant::fp16_to_fp32(rhs[idx]));
  }
  return static_cast<float>(sumf);
#endif
#endif

  double scalar_sum = 0.0;
  for (uint64_t idx = 0u; idx < count; ++idx) {
    scalar_sum += static_cast<double>(quant::fp16_to_fp32(lhs[idx])) *
                  static_cast<double>(quant::fp16_to_fp32(rhs[idx]));
  }
  return static_cast<float>(scalar_sum);
}

inline float dot_product_f32_f16_scores(const float *lhs, const uint16_t *rhs,
                                        const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  float32x4_t sum_lo = vdupq_n_f32(0.0f);
  float32x4_t sum_hi = vdupq_n_f32(0.0f);
  uint64_t idx = 0u;
  for (; idx + 8u <= count; idx += 8u) {
    const float32x4_t lhs0 = vld1q_f32(lhs + idx + 0u);
    const float32x4_t lhs1 = vld1q_f32(lhs + idx + 4u);
    const float16x8_t lhs_f16 =
        vcombine_f16(vcvt_f16_f32(lhs0), vcvt_f16_f32(lhs1));
    const float16x8_t rhs_f16 = vreinterpretq_f16_u16(vld1q_u16(rhs + idx));
    sum_lo = vfmlalq_low_f16(sum_lo, lhs_f16, rhs_f16);
    sum_hi = vfmlalq_high_f16(sum_hi, lhs_f16, rhs_f16);
  }

  double sumf = static_cast<double>(vaddvq_f32(vaddq_f32(sum_lo, sum_hi)));
  for (; idx < count; ++idx) {
    sumf += static_cast<double>(
                quant::fp16_to_fp32(quant::fp32_to_fp16(lhs[idx]))) *
            static_cast<double>(quant::fp16_to_fp32(rhs[idx]));
  }
  return static_cast<float>(sumf);
#endif
#endif

  double scalar_sum = 0.0;
  for (uint64_t idx = 0u; idx < count; ++idx) {
    scalar_sum += static_cast<double>(
                      quant::fp16_to_fp32(quant::fp32_to_fp16(lhs[idx]))) *
                  static_cast<double>(quant::fp16_to_fp32(rhs[idx]));
  }
  return static_cast<float>(scalar_sum);
}

inline float dot_product_ggml_f16_scores(const float *lhs, const uint16_t *rhs,
                                         const uint64_t count) noexcept {
  return dot_product_f32_f16_scores(lhs, rhs, count);
}

inline float dot_product_f32(const float *lhs, const float *rhs,
                             const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  float32x4_t sum[4] = {
      vdupq_n_f32(0.0f),
      vdupq_n_f32(0.0f),
      vdupq_n_f32(0.0f),
      vdupq_n_f32(0.0f),
  };
  uint64_t idx = 0u;
  for (; idx + 16u <= count; idx += 16u) {
    sum[0] =
        vfmaq_f32(sum[0], vld1q_f32(lhs + idx + 0u), vld1q_f32(rhs + idx + 0u));
    sum[1] =
        vfmaq_f32(sum[1], vld1q_f32(lhs + idx + 4u), vld1q_f32(rhs + idx + 4u));
    sum[2] =
        vfmaq_f32(sum[2], vld1q_f32(lhs + idx + 8u), vld1q_f32(rhs + idx + 8u));
    sum[3] = vfmaq_f32(sum[3], vld1q_f32(lhs + idx + 12u),
                       vld1q_f32(rhs + idx + 12u));
  }

  float scalar_sum = vaddvq_f32(
      vaddq_f32(vaddq_f32(sum[0], sum[1]), vaddq_f32(sum[2], sum[3])));
  for (; idx < count; ++idx) {
    scalar_sum += lhs[idx] * rhs[idx];
  }
  return scalar_sum;
#else
  float scalar_sum = 0.0f;
  for (uint64_t idx = 0u; idx < count; ++idx) {
    scalar_sum += lhs[idx] * rhs[idx];
  }
  return scalar_sum;
#endif
}

inline void scale_buffer_f32(float *data, const uint64_t count,
                             const float scale) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const float32x4_t scale_vec = vdupq_n_f32(scale);
  uint64_t idx = 0u;
  for (; idx + 16u <= count; idx += 16u) {
    vst1q_f32(data + idx + 0u,
              vmulq_f32(vld1q_f32(data + idx + 0u), scale_vec));
    vst1q_f32(data + idx + 4u,
              vmulq_f32(vld1q_f32(data + idx + 4u), scale_vec));
    vst1q_f32(data + idx + 8u,
              vmulq_f32(vld1q_f32(data + idx + 8u), scale_vec));
    vst1q_f32(data + idx + 12u,
              vmulq_f32(vld1q_f32(data + idx + 12u), scale_vec));
  }
  for (; idx < count; ++idx) {
    data[idx] *= scale;
  }
#else
  for (uint64_t idx = 0u; idx < count; ++idx) {
    data[idx] *= scale;
  }
#endif
}

inline void mad_buffer_f32(float *acc, const float *value, const uint64_t count,
                           const float weight) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const float32x4_t weight_vec = vdupq_n_f32(weight);
  uint64_t idx = 0u;
  for (; idx + 16u <= count; idx += 16u) {
    vst1q_f32(acc + idx + 0u,
              vfmaq_f32(vld1q_f32(acc + idx + 0u), vld1q_f32(value + idx + 0u),
                        weight_vec));
    vst1q_f32(acc + idx + 4u,
              vfmaq_f32(vld1q_f32(acc + idx + 4u), vld1q_f32(value + idx + 4u),
                        weight_vec));
    vst1q_f32(acc + idx + 8u,
              vfmaq_f32(vld1q_f32(acc + idx + 8u), vld1q_f32(value + idx + 8u),
                        weight_vec));
    vst1q_f32(acc + idx + 12u,
              vfmaq_f32(vld1q_f32(acc + idx + 12u),
                        vld1q_f32(value + idx + 12u), weight_vec));
  }
  for (; idx < count; ++idx) {
    acc[idx] += value[idx] * weight;
  }
#else
  for (uint64_t idx = 0u; idx < count; ++idx) {
    acc[idx] += value[idx] * weight;
  }
#endif
}

template <class request_type>
inline bool run_copy(const request_type &request) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  const bool shape_ok = count == tensor_element_count(request.src0);
  const bool dense = shape_ok && is_dense_contiguous(request.src0) &&
                     is_dense_contiguous(request.dst);
  const uint64_t dense_count = count * static_cast<uint64_t>(dense);
  const uint64_t sparse_count =
      count * static_cast<uint64_t>(shape_ok && !dense);

  const float *src_dense = static_cast<const float *>(request.src0.data);
  float *dst_dense = static_cast<float *>(request.dst.data);
  for (uint64_t i = 0; i < dense_count; ++i) {
    dst_dense[i] = src_dense[i];
  }

  for (uint64_t i = 0; i < sparse_count; ++i) {
    write_f32(request.dst, i, read_f32(request.src0, i));
  }
  return shape_ok;
}

template <class request_type, class op_type>
inline bool run_binary(const request_type &request, op_type op) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  const bool incompatible_shape = count != tensor_element_count(request.src0) ||
                                  count != tensor_element_count(request.src1);
  const bool compatible = !incompatible_shape;

  const bool dense = compatible && is_dense_contiguous(request.src0) &&
                     is_dense_contiguous(request.src1) &&
                     is_dense_contiguous(request.dst);
  const uint64_t dense_count = count * static_cast<uint64_t>(dense);
  const uint64_t sparse_count =
      count * static_cast<uint64_t>(compatible && !dense);

  const float *lhs_dense = static_cast<const float *>(request.src0.data);
  const float *rhs_dense = static_cast<const float *>(request.src1.data);
  float *dst_dense = static_cast<float *>(request.dst.data);
  for (uint64_t i = 0; i < dense_count; ++i) {
    dst_dense[i] = op(lhs_dense[i], rhs_dense[i]);
  }

  for (uint64_t i = 0; i < sparse_count; ++i) {
    write_f32(request.dst, i,
              op(read_f32(request.src0, i), read_f32(request.src1, i)));
  }
  return compatible;
}

template <class request_type, class op_type>
inline bool run_unary(const request_type &request, op_type op) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  const bool shape_ok = count == tensor_element_count(request.src0);
  const bool dense = shape_ok && is_dense_contiguous(request.src0) &&
                     is_dense_contiguous(request.dst);
  const uint64_t dense_count = count * static_cast<uint64_t>(dense);
  const uint64_t sparse_count =
      count * static_cast<uint64_t>(shape_ok && !dense);

  const float *src_dense = static_cast<const float *>(request.src0.data);
  float *dst_dense = static_cast<float *>(request.dst.data);
  for (uint64_t i = 0; i < dense_count; ++i) {
    dst_dense[i] = op(src_dense[i]);
  }

  for (uint64_t i = 0; i < sparse_count; ++i) {
    write_f32(request.dst, i, op(read_f32(request.src0, i)));
  }
  return shape_ok;
}

inline float dot_q2_k_q8_k_block_scalar(const quant::block_q2_k &lhs,
                                        const quant::block_q8_k &rhs) noexcept {
  const uint8_t *q2 = lhs.qs.data();
  const int8_t *q8 = rhs.qs.data();
  const uint8_t *scales = lhs.scales.data();

  int sum_mins = 0;
  for (uint64_t j = 0; j < (quant::QK_K / 16); ++j) {
    sum_mins +=
        static_cast<int>(rhs.bsums[j]) * static_cast<int>(scales[j] >> 4u);
  }

  const float d_all = rhs.d * quant::fp16_to_fp32(lhs.d);
  const float d_min = rhs.d * quant::fp16_to_fp32(lhs.dmin);

  int sum = 0;
  int scale_index = 0;
  for (uint64_t block = 0; block < (quant::QK_K / 128); ++block) {
    int shift = 0;
    for (uint64_t group = 0; group < 4; ++group) {
      int scale = static_cast<int>(scales[scale_index++] & 0x0fu);
      int local_sum = 0;
      for (uint64_t l = 0; l < 16; ++l) {
        local_sum += static_cast<int>(q8[l]) *
                     static_cast<int>((q2[l] >> shift) & 0x03u);
      }
      sum += scale * local_sum;

      scale = static_cast<int>(scales[scale_index++] & 0x0fu);
      local_sum = 0;
      for (uint64_t l = 16; l < 32; ++l) {
        local_sum += static_cast<int>(q8[l]) *
                     static_cast<int>((q2[l] >> shift) & 0x03u);
      }
      sum += scale * local_sum;

      shift += 2;
      q8 += 32;
    }
    q2 += 32;
  }

  return d_all * static_cast<float>(sum) - d_min * static_cast<float>(sum_mins);
}

inline float dot_q2_k_q8_k_row_scalar(const quant::block_q2_k *lhs,
                                      const quant::block_q8_k *rhs,
                                      const uint64_t block_count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
#if defined(__ARM_FEATURE_DOTPROD)
  const uint8x16_t m3 = vdupq_n_u8(0x03u);
  const uint8x16_t m4 = vdupq_n_u8(0x0fu);
  const int32x4_t zero = vdupq_n_s32(0);

  int8x16x2_t q2bytes{};
  uint8_t scales_buf[16] = {};
  float sum = 0.0f;

  for (uint64_t block = 0; block < block_count; ++block) {
    const float d = rhs[block].d * quant::fp16_to_fp32(lhs[block].d);
    const float dmin = -rhs[block].d * quant::fp16_to_fp32(lhs[block].dmin);
    const uint8_t *q2 = lhs[block].qs.data();
    const int8_t *q8 = rhs[block].qs.data();
    const uint8_t *scales_ptr = lhs[block].scales.data();

    const uint8x16_t mins_and_scales = vld1q_u8(scales_ptr);
    const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
    vst1q_u8(scales_buf, scales);

    const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
    const int16x8_t q8sums0 = vld1q_s16(rhs[block].bsums.data());
    const int16x8_t q8sums1 = vld1q_s16(rhs[block].bsums.data() + 8);
    const int16x8_t mins16_lo =
        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
    const int16x8_t mins16_hi =
        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));
    const int32x4_t s0 =
        vaddq_s32(vmull_s16(vget_low_s16(mins16_lo), vget_low_s16(q8sums0)),
                  vmull_s16(vget_high_s16(mins16_lo), vget_high_s16(q8sums0)));
    const int32x4_t s1 =
        vaddq_s32(vmull_s16(vget_low_s16(mins16_hi), vget_low_s16(q8sums1)),
                  vmull_s16(vget_high_s16(mins16_hi), vget_high_s16(q8sums1)));
    sum += dmin * static_cast<float>(vaddvq_s32(vaddq_s32(s0, s1)));

    int isum = 0;
    int scale_index = 0;
    for (uint64_t j = 0; j < (quant::QK_K / 128); ++j) {
      const uint8x16_t q2bits0 = vld1q_u8(q2);
      const uint8x16_t q2bits1 = vld1q_u8(q2 + 16);
      q2 += 32;

      {
        const int8x16x2_t q8bytes = {{vld1q_s8(q8), vld1q_s8(q8 + 16)}};
        q8 += 32;
        q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits0, m3));
        q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q2bits1, m3));
        isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[0], q8bytes.val[0])) *
                scales_buf[scale_index + 0];
        isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[1], q8bytes.val[1])) *
                scales_buf[scale_index + 1];
      }
      {
        const int8x16x2_t q8bytes = {{vld1q_s8(q8), vld1q_s8(q8 + 16)}};
        q8 += 32;
        q2bytes.val[0] =
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits0, 2), m3));
        q2bytes.val[1] =
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits1, 2), m3));
        isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[0], q8bytes.val[0])) *
                scales_buf[scale_index + 2];
        isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[1], q8bytes.val[1])) *
                scales_buf[scale_index + 3];
      }
      {
        const int8x16x2_t q8bytes = {{vld1q_s8(q8), vld1q_s8(q8 + 16)}};
        q8 += 32;
        q2bytes.val[0] =
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits0, 4), m3));
        q2bytes.val[1] =
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits1, 4), m3));
        isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[0], q8bytes.val[0])) *
                scales_buf[scale_index + 4];
        isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[1], q8bytes.val[1])) *
                scales_buf[scale_index + 5];
      }
      {
        const int8x16x2_t q8bytes = {{vld1q_s8(q8), vld1q_s8(q8 + 16)}};
        q8 += 32;
        q2bytes.val[0] =
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits0, 6), m3));
        q2bytes.val[1] =
            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits1, 6), m3));
        isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[0], q8bytes.val[0])) *
                scales_buf[scale_index + 6];
        isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[1], q8bytes.val[1])) *
                scales_buf[scale_index + 7];
      }
      scale_index += 8;
    }

    sum += d * static_cast<float>(isum);
  }
  return sum;
#endif
#endif

  float sumf = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    sumf += dot_q2_k_q8_k_block_scalar(lhs[block], rhs[block]);
  }
  return sumf;
}

inline float dot_q3_k_q8_k_block_scalar(const quant::block_q3_k &lhs,
                                        const quant::block_q8_k &rhs) noexcept {
  constexpr uint32_t kmask1 = 0x03030303u;
  constexpr uint32_t kmask2 = 0x0f0f0f0fu;

  alignas(64) int8_t dequant[quant::QK_K] = {};
  int16_t products[8] = {};
  int32_t sums[8] = {};
  uint32_t scale_words[4] = {};
  auto *scales = reinterpret_cast<int8_t *>(scale_words);

  const uint8_t *q3 = lhs.qs.data();
  const uint8_t *hmask = lhs.hmask.data();
  const int8_t *q8 = rhs.qs.data();
  int8_t *out = dequant;
  uint8_t mask = 1u;
  for (uint64_t block = 0; block < quant::QK_K; block += 128) {
    for (uint64_t l = 0; l < 32; ++l) {
      out[l] = static_cast<int8_t>(q3[l] & 0x03u);
    }
    for (uint64_t l = 0; l < 32; ++l) {
      out[l] = static_cast<int8_t>(out[l] - ((hmask[l] & mask) != 0u ? 0 : 4));
    }
    out += 32;
    mask = static_cast<uint8_t>(mask << 1u);
    for (uint64_t l = 0; l < 32; ++l) {
      out[l] = static_cast<int8_t>((q3[l] >> 2u) & 0x03u);
    }
    for (uint64_t l = 0; l < 32; ++l) {
      out[l] = static_cast<int8_t>(out[l] - ((hmask[l] & mask) != 0u ? 0 : 4));
    }
    out += 32;
    mask = static_cast<uint8_t>(mask << 1u);
    for (uint64_t l = 0; l < 32; ++l) {
      out[l] = static_cast<int8_t>((q3[l] >> 4u) & 0x03u);
    }
    for (uint64_t l = 0; l < 32; ++l) {
      out[l] = static_cast<int8_t>(out[l] - ((hmask[l] & mask) != 0u ? 0 : 4));
    }
    out += 32;
    mask = static_cast<uint8_t>(mask << 1u);
    for (uint64_t l = 0; l < 32; ++l) {
      out[l] = static_cast<int8_t>((q3[l] >> 6u) & 0x03u);
    }
    for (uint64_t l = 0; l < 32; ++l) {
      out[l] = static_cast<int8_t>(out[l] - ((hmask[l] & mask) != 0u ? 0 : 4));
    }
    out += 32;
    mask = static_cast<uint8_t>(mask << 1u);
    q3 += 32;
  }

  std::memcpy(scale_words, lhs.scales.data(), lhs.scales.size());
  const uint32_t tmp = scale_words[2];
  scale_words[2] =
      ((scale_words[0] >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
  scale_words[3] =
      ((scale_words[1] >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
  scale_words[0] = (scale_words[0] & kmask2) | (((tmp >> 0u) & kmask1) << 4u);
  scale_words[1] = (scale_words[1] & kmask2) | (((tmp >> 2u) & kmask1) << 4u);

  const int8_t *a = dequant;
  for (uint64_t group = 0; group < (quant::QK_K / 16); ++group) {
    for (uint64_t l = 0; l < 8; ++l) {
      products[l] = static_cast<int16_t>(q8[l] * a[l]);
    }
    for (uint64_t l = 0; l < 8; ++l) {
      sums[l] += static_cast<int32_t>(scales[group] - 32) *
                 static_cast<int32_t>(products[l]);
    }
    q8 += 8;
    a += 8;
    for (uint64_t l = 0; l < 8; ++l) {
      products[l] = static_cast<int16_t>(q8[l] * a[l]);
    }
    for (uint64_t l = 0; l < 8; ++l) {
      sums[l] += static_cast<int32_t>(scales[group] - 32) *
                 static_cast<int32_t>(products[l]);
    }
    q8 += 8;
    a += 8;
  }

  const float d = quant::fp16_to_fp32(lhs.d) * rhs.d;
  float sum = 0.0f;
  for (int32_t lane : sums) {
    sum += d * static_cast<float>(lane);
  }
  return sum;
}

inline float dot_q3_k_q8_k_row_scalar(const quant::block_q3_k *lhs,
                                      const quant::block_q8_k *rhs,
                                      const uint64_t block_count) noexcept {
  constexpr uint32_t kmask1 = 0x03030303u;
  constexpr uint32_t kmask2 = 0x0f0f0f0fu;

  int8_t aux8[quant::QK_K] = {};
  int16_t aux16[8] = {};
  float sums[8] = {};
  int32_t aux32[8] = {};
  uint32_t auxs[4] = {};
  const int8_t *scales = reinterpret_cast<const int8_t *>(auxs);

  float sumf = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    const uint8_t *q3 = lhs[block].qs.data();
    const uint8_t *hm = lhs[block].hmask.data();
    const int8_t *q8 = rhs[block].qs.data();
    std::memset(aux32, 0, sizeof(aux32));
    int8_t *a = aux8;
    uint8_t m = 1u;
    for (uint64_t j = 0; j < quant::QK_K; j += 128u) {
      for (uint64_t l = 0; l < 32; ++l) {
        a[l] = static_cast<int8_t>(q3[l] & 0x03u);
      }
      for (uint64_t l = 0; l < 32; ++l) {
        a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) != 0u ? 0 : 4));
      }
      a += 32;
      m = static_cast<uint8_t>(m << 1u);
      for (uint64_t l = 0; l < 32; ++l) {
        a[l] = static_cast<int8_t>((q3[l] >> 2u) & 0x03u);
      }
      for (uint64_t l = 0; l < 32; ++l) {
        a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) != 0u ? 0 : 4));
      }
      a += 32;
      m = static_cast<uint8_t>(m << 1u);
      for (uint64_t l = 0; l < 32; ++l) {
        a[l] = static_cast<int8_t>((q3[l] >> 4u) & 0x03u);
      }
      for (uint64_t l = 0; l < 32; ++l) {
        a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) != 0u ? 0 : 4));
      }
      a += 32;
      m = static_cast<uint8_t>(m << 1u);
      for (uint64_t l = 0; l < 32; ++l) {
        a[l] = static_cast<int8_t>((q3[l] >> 6u) & 0x03u);
      }
      for (uint64_t l = 0; l < 32; ++l) {
        a[l] = static_cast<int8_t>(a[l] - ((hm[l] & m) != 0u ? 0 : 4));
      }
      a += 32;
      m = static_cast<uint8_t>(m << 1u);
      q3 += 32;
    }

    a = aux8;
    std::memcpy(auxs, lhs[block].scales.data(), lhs[block].scales.size());
    const uint32_t tmp = auxs[2];
    auxs[2] = ((auxs[0] >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
    auxs[3] = ((auxs[1] >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
    auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0u) & kmask1) << 4u);
    auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2u) & kmask1) << 4u);

    for (uint64_t group = 0; group < (quant::QK_K / 16u); ++group) {
      for (uint64_t lane = 0; lane < 8; ++lane) {
        aux16[lane] = static_cast<int16_t>(q8[lane] * a[lane]);
      }
      for (uint64_t lane = 0; lane < 8; ++lane) {
        aux32[lane] += static_cast<int32_t>(scales[group] - 32) *
                       static_cast<int32_t>(aux16[lane]);
      }
      q8 += 8;
      a += 8;
      for (uint64_t lane = 0; lane < 8; ++lane) {
        aux16[lane] = static_cast<int16_t>(q8[lane] * a[lane]);
      }
      for (uint64_t lane = 0; lane < 8; ++lane) {
        aux32[lane] += static_cast<int32_t>(scales[group] - 32) *
                       static_cast<int32_t>(aux16[lane]);
      }
      q8 += 8;
      a += 8;
    }

    const float d = quant::fp16_to_fp32(lhs[block].d) * rhs[block].d;
    for (int lane = 0; lane < 8; ++lane) {
      sums[lane] += d * static_cast<float>(aux32[lane]);
    }
  }

  for (float lane : sums) {
    sumf += lane;
  }
  return sumf;
}

inline float dot_q4_k_q8_k_block_scalar(const quant::block_q4_k &lhs,
                                        const quant::block_q8_k &rhs) noexcept {
  constexpr uint32_t kmask1 = 0x3f3f3f3fu;
  constexpr uint32_t kmask2 = 0x0f0f0f0fu;
  constexpr uint32_t kmask3 = 0x03030303u;

  alignas(64) int8_t aux8[quant::QK_K] = {};
  int16_t aux16[8] = {};
  int32_t aux32[8] = {};
  uint32_t auxs[4] = {};

  const uint8_t *scales = reinterpret_cast<const uint8_t *>(auxs);
  const uint8_t *mins = reinterpret_cast<const uint8_t *>(auxs + 2);

  const uint8_t *q4 = lhs.qs.data();
  const int8_t *q8 = rhs.qs.data();
  int8_t *a = aux8;
  for (uint64_t block = 0; block < (quant::QK_K / 64u); ++block) {
    for (uint64_t lane = 0; lane < 32u; ++lane) {
      a[lane] = static_cast<int8_t>(q4[lane] & 0x0fu);
    }
    a += 32;
    for (uint64_t lane = 0; lane < 32u; ++lane) {
      a[lane] = static_cast<int8_t>(q4[lane] >> 4u);
    }
    a += 32;
    q4 += 32;
  }

  std::memcpy(auxs, lhs.scales.data(), lhs.scales.size());
  auxs[3] = ((auxs[2] >> 4u) & kmask2) | (((auxs[1] >> 6u) & kmask3) << 4u);
  const uint32_t aux = auxs[1] & kmask1;
  auxs[1] = (auxs[2] & kmask2) | (((auxs[0] >> 6u) & kmask3) << 4u);
  auxs[2] = aux;
  auxs[0] &= kmask1;

  int32_t min_sum = 0;
  for (uint64_t group = 0; group < (quant::QK_K / 16u); ++group) {
    min_sum += static_cast<int32_t>(rhs.bsums[group]) *
               static_cast<int32_t>(mins[group / 2u]);
  }

  std::memset(aux32, 0, sizeof(aux32));
  a = aux8;
  int scale_index = 0;
  for (uint64_t group = 0; group < (quant::QK_K / 32u); ++group) {
    const int32_t scale = scales[scale_index++];
    for (uint64_t lane = 0; lane < 8u; ++lane) {
      aux16[lane] = static_cast<int16_t>(q8[lane] * a[lane]);
    }
    for (uint64_t lane = 0; lane < 8u; ++lane) {
      aux32[lane] += scale * static_cast<int32_t>(aux16[lane]);
    }
    q8 += 8;
    a += 8;
    for (uint64_t lane = 0; lane < 8u; ++lane) {
      aux16[lane] = static_cast<int16_t>(q8[lane] * a[lane]);
    }
    for (uint64_t lane = 0; lane < 8u; ++lane) {
      aux32[lane] += scale * static_cast<int32_t>(aux16[lane]);
    }
    q8 += 8;
    a += 8;
    for (uint64_t lane = 0; lane < 8u; ++lane) {
      aux16[lane] = static_cast<int16_t>(q8[lane] * a[lane]);
    }
    for (uint64_t lane = 0; lane < 8u; ++lane) {
      aux32[lane] += scale * static_cast<int32_t>(aux16[lane]);
    }
    q8 += 8;
    a += 8;
    for (uint64_t lane = 0; lane < 8u; ++lane) {
      aux16[lane] = static_cast<int16_t>(q8[lane] * a[lane]);
    }
    for (uint64_t lane = 0; lane < 8u; ++lane) {
      aux32[lane] += scale * static_cast<int32_t>(aux16[lane]);
    }
    q8 += 8;
    a += 8;
  }

  const float d = quant::fp16_to_fp32(lhs.d) * rhs.d;
  const float dmin = quant::fp16_to_fp32(lhs.dmin) * rhs.d;
  float sum = -dmin * static_cast<float>(min_sum);
  for (int32_t lane : aux32) {
    sum += d * static_cast<float>(lane);
  }
  return sum;
}

inline float dot_q4_k_q8_k_row_scalar(const quant::block_q4_k *lhs,
                                      const quant::block_q8_k *rhs,
                                      const uint64_t block_count) noexcept {
  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    sum += dot_q4_k_q8_k_block_scalar(lhs[block], rhs[block]);
  }
  return sum;
}

inline float dot_q5_k_q8_k_block_scalar(const quant::block_q5_k &lhs,
                                        const quant::block_q8_k &rhs) noexcept {
  const uint8_t *ql = lhs.qs.data();
  const uint8_t *qh = lhs.qh.data();
  const int8_t *q8 = rhs.qs.data();
  const float d = quant::fp16_to_fp32(lhs.d);
  const float min = quant::fp16_to_fp32(lhs.dmin);
  float sum = 0.0f;
  int is = 0;
  uint8_t u1 = 1u;
  uint8_t u2 = 2u;

  for (uint64_t j = 0; j < quant::QK_K; j += 64u) {
    uint8_t sc = 0u;
    uint8_t m = 0u;
    quant::get_scale_min_k4(is + 0, lhs.scales.data(), &sc, &m);
    const float d0 = d * static_cast<float>(sc);
    const float m0 = min * static_cast<float>(m);
    quant::get_scale_min_k4(is + 1, lhs.scales.data(), &sc, &m);
    const float d1 = d * static_cast<float>(sc);
    const float m1 = min * static_cast<float>(m);
    int32_t dot0 = 0;
    int32_t dot1 = 0;
    int32_t rhs_sum0 = 0;
    int32_t rhs_sum1 = 0;
    for (uint64_t l = 0; l < 32u; ++l) {
      const int32_t rhs0 = static_cast<int32_t>(q8[j + l]);
      const int32_t rhs1 = static_cast<int32_t>(q8[j + 32u + l]);
      const uint8_t high0 = (qh[l] & u1) != 0u ? 16u : 0u;
      const uint8_t high1 = (qh[l] & u2) != 0u ? 16u : 0u;
      dot0 += static_cast<int32_t>((ql[l] & 0x0fu) + high0) * rhs0;
      dot1 += static_cast<int32_t>((ql[l] >> 4u) + high1) * rhs1;
      rhs_sum0 += rhs0;
      rhs_sum1 += rhs1;
    }
    sum += rhs.d *
           (d0 * static_cast<float>(dot0) - m0 * static_cast<float>(rhs_sum0));
    sum += rhs.d *
           (d1 * static_cast<float>(dot1) - m1 * static_cast<float>(rhs_sum1));
    ql += 32;
    is += 2;
    u1 = static_cast<uint8_t>(u1 << 2u);
    u2 = static_cast<uint8_t>(u2 << 2u);
  }

  return sum;
}

inline float dot_q5_k_q8_k_row_scalar(const quant::block_q5_k *lhs,
                                      const quant::block_q8_k *rhs,
                                      const uint64_t block_count) noexcept {
  float sum = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    sum += dot_q5_k_q8_k_block_scalar(lhs[block], rhs[block]);
  }
  return sum;
}

inline float dot_q6_k_q8_k_block_scalar(const quant::block_q6_k &lhs,
                                        const quant::block_q8_k &rhs) noexcept {
  alignas(64) int8_t dequant[quant::QK_K] = {};
  int16_t products[8] = {};
  int32_t sums[8] = {};

  const uint8_t *ql = lhs.ql.data();
  const uint8_t *qh = lhs.qh.data();
  const int8_t *q8 = rhs.qs.data();
  int8_t *out = dequant;
  for (uint64_t block = 0; block < quant::QK_K; block += 128) {
    for (uint64_t l = 0; l < 32; ++l) {
      out[l + 0] = static_cast<int8_t>((ql[l + 0] & 0x0fu) |
                                       (((qh[l] >> 0u) & 0x03u) << 4u)) -
                   32;
      out[l + 32] = static_cast<int8_t>((ql[l + 32] & 0x0fu) |
                                        (((qh[l] >> 2u) & 0x03u) << 4u)) -
                    32;
      out[l + 64] = static_cast<int8_t>(((ql[l + 0] >> 4u) & 0x0fu) |
                                        (((qh[l] >> 4u) & 0x03u) << 4u)) -
                    32;
      out[l + 96] = static_cast<int8_t>(((ql[l + 32] >> 4u) & 0x0fu) |
                                        (((qh[l] >> 6u) & 0x03u) << 4u)) -
                    32;
    }
    out += 128;
    ql += 64;
    qh += 32;
  }

  const int8_t *a = dequant;
  int scale_index = 0;
  for (uint64_t group = 0; group < (quant::QK_K / 16); ++group) {
    const int scale = lhs.scales[scale_index++];
    for (uint64_t l = 0; l < 8; ++l) {
      products[l] = static_cast<int16_t>(q8[l] * a[l]);
    }
    for (uint64_t l = 0; l < 8; ++l) {
      sums[l] += scale * static_cast<int32_t>(products[l]);
    }
    q8 += 8;
    a += 8;
    for (uint64_t l = 0; l < 8; ++l) {
      products[l] = static_cast<int16_t>(q8[l] * a[l]);
    }
    for (uint64_t l = 0; l < 8; ++l) {
      sums[l] += scale * static_cast<int32_t>(products[l]);
    }
    q8 += 8;
    a += 8;
  }

  const float d = quant::fp16_to_fp32(lhs.d) * rhs.d;
  float sum = 0.0f;
  for (int32_t lane : sums) {
    sum += d * static_cast<float>(lane);
  }
  return sum;
}

inline float dot_q6_k_q8_k_row_scalar(const quant::block_q6_k *lhs,
                                      const quant::block_q8_k *rhs,
                                      const uint64_t block_count) noexcept {
  int8_t aux8[quant::QK_K] = {};
  int16_t aux16[8] = {};
  float sums[8] = {};
  int32_t aux32[8] = {};

  float sumf = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    const uint8_t *q4 = lhs[block].ql.data();
    const uint8_t *qh = lhs[block].qh.data();
    const int8_t *q8 = rhs[block].qs.data();
    std::memset(aux32, 0, sizeof(aux32));
    int8_t *a = aux8;
    for (uint64_t j = 0; j < quant::QK_K; j += 128u) {
      for (uint64_t l = 0; l < 32; ++l) {
        a[l + 0] = static_cast<int8_t>((q4[l + 0] & 0x0fu) |
                                       (((qh[l] >> 0u) & 0x03u) << 4u)) -
                   32;
        a[l + 32] = static_cast<int8_t>((q4[l + 32] & 0x0fu) |
                                        (((qh[l] >> 2u) & 0x03u) << 4u)) -
                    32;
        a[l + 64] = static_cast<int8_t>(((q4[l + 0] >> 4u) & 0x0fu) |
                                        (((qh[l] >> 4u) & 0x03u) << 4u)) -
                    32;
        a[l + 96] = static_cast<int8_t>(((q4[l + 32] >> 4u) & 0x0fu) |
                                        (((qh[l] >> 6u) & 0x03u) << 4u)) -
                    32;
      }
      a += 128;
      q4 += 64;
      qh += 32;
    }

    a = aux8;
    int scale_index = 0;
    for (uint64_t group = 0; group < (quant::QK_K / 16u); ++group) {
      const int scale = lhs[block].scales[scale_index++];
      for (uint64_t lane = 0; lane < 8; ++lane) {
        aux16[lane] = static_cast<int16_t>(q8[lane] * a[lane]);
      }
      for (uint64_t lane = 0; lane < 8; ++lane) {
        aux32[lane] += scale * static_cast<int32_t>(aux16[lane]);
      }
      q8 += 8;
      a += 8;
      for (uint64_t lane = 0; lane < 8; ++lane) {
        aux16[lane] = static_cast<int16_t>(q8[lane] * a[lane]);
      }
      for (uint64_t lane = 0; lane < 8; ++lane) {
        aux32[lane] += scale * static_cast<int32_t>(aux16[lane]);
      }
      q8 += 8;
      a += 8;
    }

    const float d = quant::fp16_to_fp32(lhs[block].d) * rhs[block].d;
    for (int lane = 0; lane < 8; ++lane) {
      sums[lane] += d * static_cast<float>(aux32[lane]);
    }
  }

  for (float lane : sums) {
    sumf += lane;
  }
  return sumf;
}

inline float dot_q4_0_q8_0_row_scalar(const quant::block_q4_0 *lhs,
                                      const quant::block_q8_0 *rhs,
                                      const uint64_t block_count) noexcept {
  float sumf = 0.0f;

  for (uint64_t block = 0; block < block_count; ++block) {
    int32_t sumi0 = 0;
    int32_t sumi1 = 0;
    for (uint64_t j = 0; j < (quant::QK4_0 / 2u); ++j) {
      const int32_t x0 = static_cast<int32_t>(lhs[block].qs[j] & 0x0fu) - 8;
      const int32_t x1 = static_cast<int32_t>(lhs[block].qs[j] >> 4u) - 8;
      sumi0 += x0 * static_cast<int32_t>(rhs[block].qs[j]);
      sumi1 +=
          x1 * static_cast<int32_t>(rhs[block].qs[j + (quant::QK4_0 / 2u)]);
    }

    sumf +=
        static_cast<float>(sumi0 + sumi1) *
        (quant::fp16_to_fp32(lhs[block].d) * quant::fp16_to_fp32(rhs[block].d));
  }

  return sumf;
}

inline float dot_q4_1_q8_0_row_scalar(const quant::block_q4_1 *lhs,
                                      const quant::block_q8_0 *rhs,
                                      const uint64_t block_count) noexcept {
  float sumf = 0.0f;

  for (uint64_t block = 0; block < block_count; ++block) {
    int32_t sumi0 = 0;
    int32_t sumi1 = 0;
    int32_t rhs_sum = 0;
    for (uint64_t j = 0; j < (quant::QK4_1 / 2u); ++j) {
      const int32_t rhs0 = static_cast<int32_t>(rhs[block].qs[j]);
      const int32_t rhs1 =
          static_cast<int32_t>(rhs[block].qs[j + (quant::QK4_1 / 2u)]);
      sumi0 += static_cast<int32_t>(lhs[block].qs[j] & 0x0fu) * rhs0;
      sumi1 += static_cast<int32_t>(lhs[block].qs[j] >> 4u) * rhs1;
      rhs_sum += rhs0 + rhs1;
    }

    const float rhs_d = quant::fp16_to_fp32(rhs[block].d);
    sumf +=
        rhs_d *
        (quant::fp16_to_fp32(lhs[block].d) * static_cast<float>(sumi0 + sumi1) +
         quant::fp16_to_fp32(lhs[block].m) * static_cast<float>(rhs_sum));
  }

  return sumf;
}

inline float dot_q5_0_q8_0_row_scalar(const quant::block_q5_0 *lhs,
                                      const quant::block_q8_0 *rhs,
                                      const uint64_t block_count) noexcept {
  float sumf = 0.0f;

  for (uint64_t block = 0; block < block_count; ++block) {
    uint32_t qh = 0u;
    std::memcpy(&qh, lhs[block].qh.data(), sizeof(qh));

    int32_t sumi0 = 0;
    int32_t sumi1 = 0;
    for (uint64_t j = 0; j < (quant::QK5_0 / 2u); ++j) {
      const uint8_t xh_0 = static_cast<uint8_t>(((qh >> j) & 1u) << 4u);
      const uint8_t xh_1 =
          static_cast<uint8_t>(((qh >> (j + (quant::QK5_0 / 2u))) & 1u) << 4u);
      const int32_t x0 =
          static_cast<int32_t>((lhs[block].qs[j] & 0x0fu) | xh_0) - 16;
      const int32_t x1 =
          static_cast<int32_t>((lhs[block].qs[j] >> 4u) | xh_1) - 16;
      sumi0 += x0 * static_cast<int32_t>(rhs[block].qs[j]);
      sumi1 +=
          x1 * static_cast<int32_t>(rhs[block].qs[j + (quant::QK5_0 / 2u)]);
    }

    sumf +=
        static_cast<float>(sumi0 + sumi1) *
        (quant::fp16_to_fp32(lhs[block].d) * quant::fp16_to_fp32(rhs[block].d));
  }

  return sumf;
}

inline float dot_q5_1_q8_0_row_scalar(const quant::block_q5_1 *lhs,
                                      const quant::block_q8_0 *rhs,
                                      const uint64_t block_count) noexcept {
  float sumf = 0.0f;

  for (uint64_t block = 0; block < block_count; ++block) {
    uint32_t qh = 0u;
    std::memcpy(&qh, lhs[block].qh.data(), sizeof(qh));

    int32_t sumi0 = 0;
    int32_t sumi1 = 0;
    int32_t rhs_sum = 0;
    for (uint64_t j = 0; j < (quant::QK5_1 / 2u); ++j) {
      const uint8_t xh_0 = static_cast<uint8_t>(((qh >> j) & 1u) << 4u);
      const uint8_t xh_1 =
          static_cast<uint8_t>(((qh >> (j + (quant::QK5_1 / 2u))) & 1u) << 4u);
      const int32_t rhs0 = static_cast<int32_t>(rhs[block].qs[j]);
      const int32_t rhs1 =
          static_cast<int32_t>(rhs[block].qs[j + (quant::QK5_1 / 2u)]);
      const int32_t x0 =
          static_cast<int32_t>((lhs[block].qs[j] & 0x0fu) | xh_0);
      const int32_t x1 = static_cast<int32_t>((lhs[block].qs[j] >> 4u) | xh_1);
      sumi0 += x0 * rhs0;
      sumi1 += x1 * rhs1;
      rhs_sum += rhs0 + rhs1;
    }

    const float rhs_d = quant::fp16_to_fp32(rhs[block].d);
    sumf +=
        rhs_d *
        (quant::fp16_to_fp32(lhs[block].d) * static_cast<float>(sumi0 + sumi1) +
         quant::fp16_to_fp32(lhs[block].m) * static_cast<float>(rhs_sum));
  }

  return sumf;
}

inline float dot_q8_0_q8_0_row_scalar(const quant::block_q8_0 *lhs,
                                      const quant::block_q8_0 *rhs,
                                      const uint64_t block_count) noexcept {
  float sumf = 0.0f;

  for (uint64_t block = 0; block < block_count; ++block) {
    int32_t sumi = 0;
    for (uint64_t j = 0; j < quant::QK8_0; ++j) {
      sumi += static_cast<int32_t>(lhs[block].qs[j]) *
              static_cast<int32_t>(rhs[block].qs[j]);
    }

    sumf += static_cast<float>(sumi) * (quant::fp16_to_fp32(lhs[block].d) *
                                        quant::fp16_to_fp32(rhs[block].d));
  }

  return sumf;
}

inline constexpr uint8_t unary_subop_abs = 0u;
inline constexpr uint8_t unary_subop_neg = 2u;
inline constexpr uint8_t unary_subop_tanh = 4u;
inline constexpr uint8_t unary_subop_elu = 5u;
inline constexpr uint8_t unary_subop_relu = 6u;
inline constexpr uint8_t unary_subop_gelu = 8u;
inline constexpr uint8_t unary_subop_silu = 10u;
inline constexpr uint8_t unary_subop_exp = 13u;

// Matches the ggml tanh-approximation GELU constant set so kernel parity
// lanes compare against the reference within fp16-table tolerance.
inline constexpr float k_gelu_coef_a = 0.044715f;
inline constexpr float k_gelu_sqrt_2_over_pi =
    0.79788456080286535587989211986876f;

template <uint8_t subop_code, class request_type>
inline void
execute_scalar_unary_subop_unchecked(const request_type &request) noexcept {
  if constexpr (subop_code == unary_subop_abs) {
    (void)run_unary(request, [](const float v) { return std::fabs(v); });
  } else if constexpr (subop_code == unary_subop_neg) {
    (void)run_unary(request, [](const float v) { return -v; });
  } else if constexpr (subop_code == unary_subop_tanh) {
    (void)run_unary(request, [](const float v) { return std::tanh(v); });
  } else if constexpr (subop_code == unary_subop_elu) {
    (void)run_unary(request,
                    [](const float v) { return v > 0.0f ? v : std::expm1(v); });
  } else if constexpr (subop_code == unary_subop_relu) {
    (void)run_unary(request, [](const float v) { return std::max(0.0f, v); });
  } else if constexpr (subop_code == unary_subop_gelu) {
    // Exact ggml GGML_GELU_FP16 semantics: input and output round through
    // fp16 around the tanh approximation, with the +-10 saturation guards
    // (equivalent to ggml's fp16 lookup table entry for the rounded input).
    (void)run_unary(request, [](const float v) {
      if (v <= -10.0f) {
        return 0.0f;
      }
      if (v >= 10.0f) {
        return v;
      }
      const float quantized = quant::fp16_to_fp32(quant::fp32_to_fp16(v));
      const float approx =
          0.5f * quantized *
          (1.0f + std::tanh(k_gelu_sqrt_2_over_pi *
                            (quantized + k_gelu_coef_a * quantized * quantized *
                                             quantized)));
      return quant::fp16_to_fp32(quant::fp32_to_fp16(approx));
    });
  } else if constexpr (subop_code == unary_subop_silu) {
    (void)run_unary(request,
                    [](const float v) { return v / (1.0f + std::exp(-v)); });
  } else if constexpr (subop_code == unary_subop_exp) {
    (void)run_unary(request, [](const float v) { return std::exp(v); });
  }
}

template <class dispatch_event_type, class context_type, class mark_done_type,
          ::emel::kernel::event::unary_subop subop>
struct exec_scalar_unary_op {
  void operator()(const dispatch_event_type &ev,
                  context_type &ctx) const noexcept {
    execute_scalar_unary_subop_unchecked<static_cast<uint8_t>(subop)>(
        ev.request);
    mark_done_type{}(ev, ctx);
  }
};

template <class dispatch_event_type, class context_type, class simd_guard_type,
          class unary_subop_guard_type>
struct simd_unary_subop_guard {
  bool operator()(const dispatch_event_type &ev,
                  const context_type &ctx) const noexcept {
    return simd_guard_type{}(ev, ctx) && unary_subop_guard_type{}(ev, ctx);
  }
};

template <class dispatch_event_type, class context_type, class valid_guard_type,
          class unary_subop_guard_type>
struct valid_unary_subop_guard {
  bool operator()(const dispatch_event_type &ev,
                  const context_type &ctx) const noexcept {
    return valid_guard_type{}(ev, ctx) && unary_subop_guard_type{}(ev, ctx);
  }
};

template <class request_type>
inline bool run_mul_mat(const request_type &request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool has_empty_dim = k == 0 || m == 0 || n == 0;
  const bool shape_mismatch = request.src1.ne[1] != k ||
                              request.dst.ne[0] != n || request.dst.ne[1] != m;
  const bool invalid_rank =
      request.src0.ne[2] != 1 || request.src0.ne[3] != 1 ||
      request.src1.ne[2] != 1 || request.src1.ne[3] != 1 ||
      request.dst.ne[2] != 1 || request.dst.ne[3] != 1;
  const bool valid = !(has_empty_dim || shape_mismatch || invalid_rank);
  const uint8_t src0_type = dtype_code(request.src0.type);
  const bool q4_0_src0 = is_q4_0_dtype(src0_type);
  const bool q4_1_src0 = is_q4_1_dtype(src0_type);
  const bool q5_0_src0 = is_q5_0_dtype(src0_type);
  const bool q8_0_src0 = is_q8_0_dtype(src0_type);
  const bool quantized_src0 = is_quantized_k_dtype(src0_type);

  if (valid && q4_0_src0) {
    const auto *b_dense = static_cast<const float *>(request.src1.data);
    auto *c_dense = static_cast<float *>(request.dst.data);
    const auto *a_base = static_cast<const uint8_t *>(request.src0.data);
    const size_t row_bytes = request.src0.nb[1];
    const uint64_t block_count = k / quant::QK4_0;
    std::array<quant::block_q8_0, quant::MAX_Q8_0_BLOCKS> q8_blocks = {};
    if (block_count > q8_blocks.size()) {
      return false;
    }

    for (uint64_t j = 0; j < n; ++j) {
      for (uint64_t i = 0; i < m; ++i) {
        c_dense[i * n + j] = 0.0f;
      }
      for (uint64_t block = 0; block < block_count; ++block) {
        quant::quantize_row_q8_0_strided(b_dense + block * quant::QK4_0 * n + j,
                                         n, &q8_blocks[block], quant::QK4_0);
      }
      for (uint64_t i = 0; i < m; ++i) {
        const uint8_t *row_ptr = a_base + i * row_bytes;
        c_dense[i * n + j] = dot_q4_0_q8_0_row_scalar(
            reinterpret_cast<const quant::block_q4_0 *>(row_ptr),
            q8_blocks.data(), block_count);
      }
    }

    return true;
  }

  if (valid && q4_1_src0) {
    const auto *b_dense = static_cast<const float *>(request.src1.data);
    auto *c_dense = static_cast<float *>(request.dst.data);
    const auto *a_base = static_cast<const uint8_t *>(request.src0.data);
    const size_t row_bytes = request.src0.nb[1];
    const uint64_t block_count = k / quant::QK4_1;
    std::array<quant::block_q8_0, quant::MAX_Q8_0_BLOCKS> q8_blocks = {};
    if (block_count > q8_blocks.size()) {
      return false;
    }

    for (uint64_t j = 0; j < n; ++j) {
      for (uint64_t i = 0; i < m; ++i) {
        c_dense[i * n + j] = 0.0f;
      }
      for (uint64_t block = 0; block < block_count; ++block) {
        quant::quantize_row_q8_0_strided(b_dense + block * quant::QK4_1 * n + j,
                                         n, &q8_blocks[block], quant::QK4_1);
      }
      for (uint64_t i = 0; i < m; ++i) {
        const uint8_t *row_ptr = a_base + i * row_bytes;
        c_dense[i * n + j] = dot_q4_1_q8_0_row_scalar(
            reinterpret_cast<const quant::block_q4_1 *>(row_ptr),
            q8_blocks.data(), block_count);
      }
    }

    return true;
  }

  if (valid && q5_0_src0) {
    const auto *b_dense = static_cast<const float *>(request.src1.data);
    auto *c_dense = static_cast<float *>(request.dst.data);
    const auto *a_base = static_cast<const uint8_t *>(request.src0.data);
    const size_t row_bytes = request.src0.nb[1];
    const uint64_t block_count = k / quant::QK5_0;
    std::array<quant::block_q8_0, quant::MAX_Q8_0_BLOCKS> q8_blocks = {};
    if (block_count > q8_blocks.size()) {
      return false;
    }

    for (uint64_t j = 0; j < n; ++j) {
      for (uint64_t i = 0; i < m; ++i) {
        c_dense[i * n + j] = 0.0f;
      }
      for (uint64_t block = 0; block < block_count; ++block) {
        quant::quantize_row_q8_0_strided(b_dense + block * quant::QK5_0 * n + j,
                                         n, &q8_blocks[block], quant::QK5_0);
      }
      for (uint64_t i = 0; i < m; ++i) {
        const uint8_t *row_ptr = a_base + i * row_bytes;
        c_dense[i * n + j] = dot_q5_0_q8_0_row_scalar(
            reinterpret_cast<const quant::block_q5_0 *>(row_ptr),
            q8_blocks.data(), block_count);
      }
    }

    return true;
  }

  if (valid && q8_0_src0) {
    const auto *b_dense = static_cast<const float *>(request.src1.data);
    auto *c_dense = static_cast<float *>(request.dst.data);
    const auto *a_base = static_cast<const uint8_t *>(request.src0.data);
    const size_t row_bytes = request.src0.nb[1];
    const uint64_t block_count = k / quant::QK8_0;
    std::array<quant::block_q8_0, quant::MAX_Q8_0_BLOCKS> q8_blocks = {};
    if (block_count > q8_blocks.size()) {
      return false;
    }

    for (uint64_t j = 0; j < n; ++j) {
      for (uint64_t i = 0; i < m; ++i) {
        c_dense[i * n + j] = 0.0f;
      }
      for (uint64_t block = 0; block < block_count; ++block) {
        quant::quantize_row_q8_0_strided(b_dense + block * quant::QK8_0 * n + j,
                                         n, &q8_blocks[block], quant::QK8_0);
      }
      for (uint64_t i = 0; i < m; ++i) {
        const uint8_t *row_ptr = a_base + i * row_bytes;
        c_dense[i * n + j] = dot_q8_0_q8_0_row_scalar(
            reinterpret_cast<const quant::block_q8_0 *>(row_ptr),
            q8_blocks.data(), block_count);
      }
    }

    return true;
  }

  if (valid && quantized_src0) {
    const auto *b_dense = static_cast<const float *>(request.src1.data);
    auto *c_dense = static_cast<float *>(request.dst.data);
    const auto *a_base = static_cast<const uint8_t *>(request.src0.data);
    const size_t row_bytes = request.src0.nb[1];
    const uint64_t block_count = k / quant::QK_K;
    std::array<quant::block_q8_k, quant::MAX_Q8_K_BLOCKS> q8_blocks = {};
    if (block_count > q8_blocks.size()) {
      return false;
    }

    for (uint64_t j = 0; j < n; ++j) {
      for (uint64_t i = 0; i < m; ++i) {
        c_dense[i * n + j] = 0.0f;
      }
      for (uint64_t block = 0; block < block_count; ++block) {
        quant::quantize_row_q8_k_strided(b_dense + block * quant::QK_K * n + j,
                                         n, &q8_blocks[block], quant::QK_K);
      }
      for (uint64_t i = 0; i < m; ++i) {
        const uint8_t *row_ptr = a_base + i * row_bytes;
        if (src0_type == dtype_q2_k) {
          c_dense[i * n + j] = dot_q2_k_q8_k_row_scalar(
              reinterpret_cast<const quant::block_q2_k *>(row_ptr),
              q8_blocks.data(), block_count);
        } else if (src0_type == dtype_q3_k) {
          c_dense[i * n + j] = dot_q3_k_q8_k_row_scalar(
              reinterpret_cast<const quant::block_q3_k *>(row_ptr),
              q8_blocks.data(), block_count);
        } else if (src0_type == dtype_q4_k) {
          c_dense[i * n + j] = dot_q4_k_q8_k_row_scalar(
              reinterpret_cast<const quant::block_q4_k *>(row_ptr),
              q8_blocks.data(), block_count);
        } else {
          c_dense[i * n + j] = dot_q6_k_q8_k_row_scalar(
              reinterpret_cast<const quant::block_q6_k *>(row_ptr),
              q8_blocks.data(), block_count);
        }
      }
    }

    return true;
  }

  const bool dense = valid && is_dense_contiguous(request.src0) &&
                     is_dense_contiguous(request.src1) &&
                     is_dense_contiguous(request.dst);
  const uint64_t dense_m = m * static_cast<uint64_t>(dense);
  const uint64_t sparse_m = m * static_cast<uint64_t>(valid && !dense);

  const float *a_dense = static_cast<const float *>(request.src0.data);
  const float *b_dense = static_cast<const float *>(request.src1.data);
  float *c_dense = static_cast<float *>(request.dst.data);
  for (uint64_t i = 0; i < dense_m; ++i) {
    for (uint64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (uint64_t p = 0; p < k; ++p) {
        acc += a_dense[i * k + p] * b_dense[p * n + j];
      }
      c_dense[i * n + j] = acc;
    }
  }

  for (uint64_t i = 0; i < sparse_m; ++i) {
    for (uint64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (uint64_t p = 0; p < k; ++p) {
        acc +=
            read_f32_at(request.src0, p, i) * read_f32_at(request.src1, j, p);
      }
      write_f32_at(request.dst, j, i, acc);
    }
  }

  return valid;
}

template <class request_type>
inline bool can_run_mul_mat_argmax(const request_type &request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool has_empty_dim = k == 0 || m == 0 || n != 1u;
  const uint8_t src0_type = dtype_code(request.src0.type);
  const uint8_t src1_type = dtype_code(request.src1.type);
  const uint8_t dst_type = dtype_code(request.dst.type);
  const uint64_t quant_block_size = quantized_block_size(src0_type);
  const uint64_t quant_block_count_limit = max_quantized_block_count(src0_type);
  const bool valid_shape =
      request.src1.ne[1] == k && request.dst.ne[0] == 1u &&
      request.dst.ne[1] == 1u && request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u && request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u && request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u;
  const bool f32_path =
      src0_type == dtype_f32 && src1_type == dtype_f32 &&
      dst_type == dtype_f32 && is_dense_contiguous(request.src0) &&
      is_dense_contiguous(request.src1) && is_dense_contiguous(request.dst);
  const bool quantized_path =
      is_native_quantized_dtype(src0_type) && src1_type == dtype_f32 &&
      dst_type == dtype_f32 && quant_block_size != 0u &&
      (k % quant_block_size) == 0u &&
      (k / quant_block_size) <= quant_block_count_limit &&
      is_dense_contiguous(request.src1) && is_dense_contiguous(request.dst) &&
      request.src0.nb[0] == 1u &&
      request.src0.nb[1] == quantized_row_storage_bytes(src0_type, k) &&
      request.src0.nb[2] == request.src0.nb[1] * m &&
      request.src0.nb[3] == request.src0.nb[2];
  return !has_empty_dim && request.index_out != nullptr && valid_shape &&
         (f32_path || quantized_path);
}

template <class request_type>
inline bool run_mul_mat_argmax(const request_type &request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const bool valid = can_run_mul_mat_argmax(request);
  if (!valid) {
    return false;
  }

  const uint8_t src0_type = dtype_code(request.src0.type);
  const float *b_dense = static_cast<const float *>(request.src1.data);
  float *best_value_out = static_cast<float *>(request.dst.data);
  int32_t best_index = 0;
  float best_value = -std::numeric_limits<float>::infinity();

  if (src0_type == dtype_f32) {
    const float *a_dense = static_cast<const float *>(request.src0.data);
    for (uint64_t row = 0; row < m; ++row) {
      float acc = 0.0f;
      for (uint64_t col = 0; col < k; ++col) {
        acc += a_dense[row * k + col] * b_dense[col];
      }
      if (acc > best_value || row == 0u) {
        best_value = acc;
        best_index = static_cast<int32_t>(row);
      }
    }
    *best_value_out = best_value;
    *request.index_out = best_index;
    return true;
  }

  if (is_q5_0_dtype(src0_type)) {
    const auto *a_base = static_cast<const uint8_t *>(request.src0.data);
    const size_t row_bytes = request.src0.nb[1];
    const uint64_t block_count = k / quant::QK5_0;
    std::array<quant::block_q8_0, quant::MAX_Q8_0_BLOCKS> q8_blocks = {};
    if (block_count > q8_blocks.size()) {
      return false;
    }
    quant::quantize_row_q8_0_strided(b_dense, 1u, q8_blocks.data(),
                                     static_cast<int64_t>(k));
    for (uint64_t row = 0; row < m; ++row) {
      const uint8_t *row_ptr = a_base + row * row_bytes;
      const float value = dot_q5_0_q8_0_row_scalar(
          reinterpret_cast<const quant::block_q5_0 *>(row_ptr),
          q8_blocks.data(), block_count);
      if (value > best_value || row == 0u) {
        best_value = value;
        best_index = static_cast<int32_t>(row);
      }
    }
    *best_value_out = best_value;
    *request.index_out = best_index;
    return true;
  }

  if (is_q8_0_dtype(src0_type)) {
    const auto *a_base = static_cast<const uint8_t *>(request.src0.data);
    const size_t row_bytes = request.src0.nb[1];
    const uint64_t block_count = k / quant::QK8_0;
    std::array<quant::block_q8_0, quant::MAX_Q8_0_BLOCKS> q8_blocks = {};
    if (block_count > q8_blocks.size()) {
      return false;
    }
    quant::quantize_row_q8_0_strided(b_dense, 1u, q8_blocks.data(),
                                     static_cast<int64_t>(k));
    for (uint64_t row = 0; row < m; ++row) {
      const uint8_t *row_ptr = a_base + row * row_bytes;
      const float value = dot_q8_0_q8_0_row_scalar(
          reinterpret_cast<const quant::block_q8_0 *>(row_ptr),
          q8_blocks.data(), block_count);
      if (value > best_value || row == 0u) {
        best_value = value;
        best_index = static_cast<int32_t>(row);
      }
    }
    *best_value_out = best_value;
    *request.index_out = best_index;
    return true;
  }

  if (is_quantized_k_dtype(src0_type)) {
    const auto *a_base = static_cast<const uint8_t *>(request.src0.data);
    const size_t row_bytes = request.src0.nb[1];
    const uint64_t block_count = k / quant::QK_K;
    std::array<quant::block_q8_k, quant::MAX_Q8_K_BLOCKS> q8_blocks = {};
    if (block_count > q8_blocks.size()) {
      return false;
    }
    quant::quantize_row_q8_k_strided(b_dense, 1u, q8_blocks.data(),
                                     static_cast<int64_t>(k));
    for (uint64_t row = 0; row < m; ++row) {
      const uint8_t *row_ptr = a_base + row * row_bytes;
      float value = 0.0f;
      if (src0_type == dtype_q2_k) {
        value = dot_q2_k_q8_k_row_scalar(
            reinterpret_cast<const quant::block_q2_k *>(row_ptr),
            q8_blocks.data(), block_count);
      } else if (src0_type == dtype_q3_k) {
        value = dot_q3_k_q8_k_row_scalar(
            reinterpret_cast<const quant::block_q3_k *>(row_ptr),
            q8_blocks.data(), block_count);
      } else if (src0_type == dtype_q4_k) {
        value = dot_q4_k_q8_k_row_scalar(
            reinterpret_cast<const quant::block_q4_k *>(row_ptr),
            q8_blocks.data(), block_count);
      } else {
        value = dot_q6_k_q8_k_row_scalar(
            reinterpret_cast<const quant::block_q6_k *>(row_ptr),
            q8_blocks.data(), block_count);
      }
      if (value > best_value || row == 0u) {
        best_value = value;
        best_index = static_cast<int32_t>(row);
      }
    }
    *best_value_out = best_value;
    *request.index_out = best_index;
    return true;
  }

  return false;
}

template <class request_type>
inline bool run_soft_max(const request_type &request) noexcept {
  const uint64_t width = request.src0.ne[0];
  const uint64_t count = tensor_element_count(request.src0);
  const bool invalid_shape = width == 0 || count == 0 || count % width != 0 ||
                             count != tensor_element_count(request.dst);
  const bool valid = !invalid_shape;
  const uint64_t safe_width = select_u64(width != 0u, width, 1u);
  const uint64_t rows = (count / safe_width) * static_cast<uint64_t>(valid);

  const bool dense = valid && is_dense_contiguous(request.src0) &&
                     is_dense_contiguous(request.dst);
  const uint64_t dense_rows = rows * static_cast<uint64_t>(dense);
  const uint64_t sparse_rows = rows * static_cast<uint64_t>(!dense);

  const float *src_dense = static_cast<const float *>(request.src0.data);
  float *dst_dense = static_cast<float *>(request.dst.data);
  for (uint64_t row = 0; row < dense_rows; ++row) {
    const uint64_t offset = row * width;
    float max_v = src_dense[offset];
    for (uint64_t i = 1; i < width; ++i) {
      max_v = std::max(max_v, src_dense[offset + i]);
    }

    const double sum = exp_and_sum_ggml_f32(src_dense + offset,
                                            dst_dense + offset, width, max_v);
    for (uint64_t i = 0; i < width; ++i) {
      dst_dense[offset + i] /= static_cast<float>(sum);
    }
  }

  for (uint64_t row = 0; row < sparse_rows; ++row) {
    const uint64_t offset = row * width;
    float max_v = read_f32(request.src0, offset);
    for (uint64_t i = 1; i < width; ++i) {
      max_v = std::max(max_v, read_f32(request.src0, offset + i));
    }

    float sum = 0.0f;
    for (uint64_t i = 0; i < width; ++i) {
      const float e = std::exp(read_f32(request.src0, offset + i) - max_v);
      write_f32(request.dst, offset + i, e);
      sum += e;
    }

    for (uint64_t i = 0; i < width; ++i) {
      write_f32(request.dst, offset + i,
                read_f32(request.dst, offset + i) / sum);
    }
  }

  return valid;
}

template <class request_type>
inline bool can_run_copy(const request_type &request) noexcept {
  return tensor_element_count(request.dst) ==
         tensor_element_count(request.src0);
}

template <class request_type>
inline bool can_run_binary(const request_type &request) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  return count == tensor_element_count(request.src0) &&
         count == tensor_element_count(request.src1);
}

// Row-broadcast binary variant: src1 is one row of dst.ne[0] values applied
// to every dst row (bias adds, norm weights, layer scales). Modeled as its
// own guard-selected transition row per backend.
template <class request_type>
inline bool can_run_binary_broadcast_row(const request_type &request) noexcept {
  const bool same_shape = request.src0.ne[0] == request.dst.ne[0] &&
                          request.src0.ne[1] == request.dst.ne[1] &&
                          request.src0.ne[2] == request.dst.ne[2] &&
                          request.src0.ne[3] == request.dst.ne[3];
  return same_shape && request.dst.ne[0] > 0 &&
         tensor_element_count(request.dst) >
             tensor_element_count(request.src1) &&
         request.src1.ne[0] == request.dst.ne[0] && request.src1.ne[1] == 1 &&
         request.src1.ne[2] == 1 && request.src1.ne[3] == 1 &&
         dtype_code(request.src0.type) == dtype_f32 &&
         dtype_code(request.src1.type) == dtype_f32 &&
         dtype_code(request.dst.type) == dtype_f32 &&
         has_valid_tensor_layout(request.src0) &&
         has_valid_tensor_layout(request.src1) &&
         has_valid_tensor_layout(request.dst);
}

template <class request_type, class op_type>
inline bool run_binary_broadcast_row(const request_type &request,
                                     op_type op) noexcept {
  const uint64_t cols = request.dst.ne[0];
  const uint64_t rows = tensor_element_count(request.dst) / cols;
  for (uint64_t row = 0; row < rows; ++row) {
    for (uint64_t col = 0; col < cols; ++col) {
      const uint64_t index = row * cols + col;
      write_f32(request.dst, index,
                op(read_f32(request.src0, index), read_f32(request.src1, col)));
    }
  }
  return true;
}

template <class request_type>
inline bool can_run_unary(const request_type &request) noexcept {
  return tensor_element_count(request.dst) ==
         tensor_element_count(request.src0);
}

template <class request_type>
inline bool can_run_mul_mat(const request_type &request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool has_empty_dim = k == 0 || m == 0 || n == 0;
  const uint8_t src0_type = dtype_code(request.src0.type);
  const uint8_t src1_type = dtype_code(request.src1.type);
  const uint8_t dst_type = dtype_code(request.dst.type);
  const uint64_t quant_block_size = quantized_block_size(src0_type);
  const uint64_t quant_block_count_limit = max_quantized_block_count(src0_type);
  const bool valid_shape = request.src1.ne[1] == k && request.dst.ne[0] == n &&
                           request.dst.ne[1] == m && request.src0.ne[2] == 1 &&
                           request.src0.ne[3] == 1 && request.src1.ne[2] == 1 &&
                           request.src1.ne[3] == 1 && request.dst.ne[2] == 1 &&
                           request.dst.ne[3] == 1;
  const bool f32_path =
      src0_type == dtype_f32 && src1_type == dtype_f32 && dst_type == dtype_f32;
  const bool quantized_path =
      is_native_quantized_dtype(src0_type) && src1_type == dtype_f32 &&
      dst_type == dtype_f32 && quant_block_size != 0u &&
      (k % quant_block_size) == 0u &&
      (k / quant_block_size) <= quant_block_count_limit &&
      is_dense_contiguous(request.src1) && is_dense_contiguous(request.dst) &&
      request.src0.nb[0] == 1u &&
      request.src0.nb[1] == quantized_row_storage_bytes(src0_type, k) &&
      request.src0.nb[2] == request.src0.nb[1] * m &&
      request.src0.nb[3] == request.src0.nb[2];
  return !has_empty_dim && valid_shape && (f32_path || quantized_path);
}

// ggml-layout f16 matmul (token-parity path for f16 conv/proj weights):
// src0 f16 [k, m] k-fastest, src1 f16 [k, n] k-fastest, dst f32 [m, n]
// m-fastest. Replicates ggml_vec_dot_f16 accumulation order exactly.
template <class request_type>
inline bool can_run_mul_mat_f16(const request_type &request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[1];
  const bool has_empty_dim = k == 0 || m == 0 || n == 0;
  const bool valid_shape = request.src1.ne[0] == k && request.dst.ne[0] == m &&
                           request.dst.ne[1] == n && request.src0.ne[2] == 1 &&
                           request.src0.ne[3] == 1 && request.src1.ne[2] == 1 &&
                           request.src1.ne[3] == 1 && request.dst.ne[2] == 1 &&
                           request.dst.ne[3] == 1;
  return !has_empty_dim && valid_shape &&
         dtype_code(request.src0.type) == dtype_f16 &&
         dtype_code(request.src1.type) == dtype_f16 &&
         dtype_code(request.dst.type) == dtype_f32 &&
         is_dense_contiguous(request.src0) &&
         is_dense_contiguous(request.src1) && is_dense_contiguous(request.dst);
}

// Exact port of the pinned ggml_vec_dot_f16 x86 AVX2+F16C+FMA path (4-way
// __m256 accumulators over 32-element steps, pairwise reduce, double-precision
// scalar tail). The scalar fallback matches ggml's no-SIMD path.
inline float vec_dot_f16_ggml(const int64_t count, const uint16_t *x,
                              const uint16_t *y) noexcept {
  double sumf = 0.0;
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
  const int64_t np = count & ~static_cast<int64_t>(31);
  __m256 sum[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                   _mm256_setzero_ps(), _mm256_setzero_ps()};
  for (int64_t i = 0; i < np; i += 32) {
    for (int j = 0; j < 4; ++j) {
      const __m256 ax = _mm256_cvtph_ps(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(x + i + j * 8)));
      const __m256 ay = _mm256_cvtph_ps(
          _mm_loadu_si128(reinterpret_cast<const __m128i *>(y + i + j * 8)));
      sum[j] = _mm256_fmadd_ps(ax, ay, sum[j]);
    }
  }
  sum[0] = _mm256_add_ps(sum[0], sum[2]);
  sum[1] = _mm256_add_ps(sum[1], sum[3]);
  sum[0] = _mm256_add_ps(sum[0], sum[1]);
  const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(sum[0]),
                               _mm256_extractf128_ps(sum[0], 1));
  const __m128 t1 = _mm_hadd_ps(t0, t0);
  sumf = static_cast<double>(_mm_cvtss_f32(_mm_hadd_ps(t1, t1)));
  for (int64_t i = np; i < count; ++i) {
    sumf += static_cast<double>(quant::fp16_to_fp32(x[i]) *
                                quant::fp16_to_fp32(y[i]));
  }
#else
  for (int64_t i = 0; i < count; ++i) {
    sumf += static_cast<double>(quant::fp16_to_fp32(x[i]) *
                                quant::fp16_to_fp32(y[i]));
  }
#endif
  return static_cast<float>(sumf);
}

inline float bf16_to_fp32(const uint16_t bits16) noexcept {
  const uint32_t bits32 = static_cast<uint32_t>(bits16) << 16u;
  float out = 0.0f;
  std::memcpy(&out, &bits32, sizeof(out));
  return out;
}

// Exact port of ggml_compute_fp32_to_bf16 (round-to-nearest-even with the
// NaN quieting ggml applies).
inline uint16_t fp32_to_bf16(const float value) noexcept {
  uint32_t bits32 = 0;
  std::memcpy(&bits32, &value, sizeof(bits32));
  if ((bits32 & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<uint16_t>((bits32 >> 16u) | 64u);
  }
  return static_cast<uint16_t>((bits32 + (0x7fffu + ((bits32 >> 16u) & 1u))) >>
                               16u);
}

// Exact port of the pinned ggml_vec_dot_bf16 x86 AVX2 path (no AVX512 on
// the supported hosts): four 8-lane accumulators over 32-element steps using
// separate mul+add (not fmadd), (c1+c3)+(c2+c4) combine, movehl/movehdup
// reduce, and a double-precision scalar tail. Scalar fallback matches
// ggml's no-SIMD path.
inline float vec_dot_bf16_ggml(const int64_t count, const uint16_t *x,
                               const uint16_t *y) noexcept {
  double sumf = 0.0;
  int64_t i = 0;
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
  const auto load_bf16 = [](const uint16_t *p) noexcept {
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(
                              reinterpret_cast<const __m128i *>(p))),
                          16));
  };
  __m256 c1 = _mm256_setzero_ps();
  __m256 c2 = _mm256_setzero_ps();
  __m256 c3 = _mm256_setzero_ps();
  __m256 c4 = _mm256_setzero_ps();
  for (; i + 32 <= count; i += 32) {
    c1 = _mm256_add_ps(_mm256_mul_ps(load_bf16(x + i), load_bf16(y + i)), c1);
    c2 = _mm256_add_ps(
        _mm256_mul_ps(load_bf16(x + i + 8), load_bf16(y + i + 8)), c2);
    c3 = _mm256_add_ps(
        _mm256_mul_ps(load_bf16(x + i + 16), load_bf16(y + i + 16)), c3);
    c4 = _mm256_add_ps(
        _mm256_mul_ps(load_bf16(x + i + 24), load_bf16(y + i + 24)), c4);
  }
  c1 = _mm256_add_ps(_mm256_add_ps(c1, c3), _mm256_add_ps(c2, c4));
  __m128 g =
      _mm_add_ps(_mm256_extractf128_ps(c1, 1), _mm256_castps256_ps128(c1));
  g = _mm_add_ps(g, _mm_movehl_ps(g, g));
  g = _mm_add_ss(g, _mm_movehdup_ps(g));
  sumf += static_cast<double>(_mm_cvtss_f32(g));
#endif
  for (; i < count; ++i) {
    sumf += static_cast<double>(bf16_to_fp32(x[i]) * bf16_to_fp32(y[i]));
  }
  return static_cast<float>(sumf);
}

// Exact port of the pinned ggml_vec_dot_f32 x86 AVX2+FMA path: four 8-lane
// fmadd accumulators over 32-element steps, pairwise reduce, FLOAT scalar
// tail (ggml keeps the f32 SIMD path's leftovers in float).
inline float vec_dot_f32_ggml(const int64_t count, const float *x,
                              const float *y) noexcept {
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
  float sumf = 0.0f;
  const int64_t np = count & ~static_cast<int64_t>(31);
  __m256 sum[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                   _mm256_setzero_ps(), _mm256_setzero_ps()};
  for (int64_t i = 0; i < np; i += 32) {
    for (int j = 0; j < 4; ++j) {
      sum[j] = _mm256_fmadd_ps(_mm256_loadu_ps(x + i + j * 8),
                               _mm256_loadu_ps(y + i + j * 8), sum[j]);
    }
  }
  sum[0] = _mm256_add_ps(sum[0], sum[2]);
  sum[1] = _mm256_add_ps(sum[1], sum[3]);
  sum[0] = _mm256_add_ps(sum[0], sum[1]);
  const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(sum[0]),
                               _mm256_extractf128_ps(sum[0], 1));
  const __m128 t1 = _mm_hadd_ps(t0, t0);
  sumf = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
  for (int64_t i = np; i < count; ++i) {
    sumf += x[i] * y[i];
  }
  return sumf;
#else
  double sumf = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    sumf += static_cast<double>(x[i] * y[i]);
  }
  return static_cast<float>(sumf);
#endif
}

#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
// Exact port of ggml_v_expf (ARM optimized-routines polynomial as vendored
// by the pinned ggml).
inline __m256 v_expf_ggml(__m256 x) noexcept {
  const __m256 r = _mm256_set1_ps(0x1.8p23f);
  const __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256 n = _mm256_sub_ps(z, r);
  const __m256 b =
      _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                       _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256 k = _mm256_castsi256_ps(
      _mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(
      _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,
                                      _mm256_set1_ps(0x1.573e2ep-5f)),
                      u,
                      _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,
                                      _mm256_set1_ps(0x1.fffdb6p-2f))),
      u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) {
    return _mm256_fmadd_ps(j, k, k);
  }
  const __m256i g = _mm256_and_si256(
      _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
      _mm256_set1_epi32(static_cast<int32_t>(0x82000000u)));
  const __m256 s1 =
      _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000)));
  const __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n),
                    _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(
          _mm256_castsi256_ps(d),
          _mm256_or_ps(
              _mm256_and_ps(_mm256_castsi256_ps(c),
                            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
              _mm256_andnot_ps(_mm256_castsi256_ps(c),
                               _mm256_fmadd_ps(k, j, k)))));
}
#endif

// Exact port of ggml's soft_max row kernel over a pre-scaled, pre-masked
// row: max (order independent), 8-lane v_expf blocks with per-block
// horizontal reduce into a double sum, libm expf scalar tail, then a
// per-element multiply by (float)(1.0 / sum).
inline void soft_max_row_ggml(const int64_t count, float *data) noexcept {
  float max_value = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < count; ++i) {
    max_value = std::max(max_value, data[i]);
  }
  double sum = 0.0;
  int64_t i = 0;
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
  for (; i + 7 < count; i += 8) {
    const __m256 val = v_expf_ggml(
        _mm256_sub_ps(_mm256_loadu_ps(data + i), _mm256_set1_ps(max_value)));
    _mm256_storeu_ps(data + i, val);
    __m128 val2 =
        _mm_add_ps(_mm256_extractf128_ps(val, 1), _mm256_castps256_ps128(val));
    val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
    val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
    sum += static_cast<double>(_mm_cvtss_f32(val2));
  }
#endif
  for (; i < count; ++i) {
    const float val = std::exp(data[i] - max_value);
    sum += static_cast<double>(val);
    data[i] = val;
  }
  const float inv_sum = static_cast<float>(1.0 / sum);
  for (int64_t j = 0; j < count; ++j) {
    data[j] *= inv_sum;
  }
}

// Exact port of ggml_compute_forward_norm's row math: float mean from the
// double-accumulated sum, AVX2 centered-variance blocks with per-block
// horizontal reduce into a double sum, float scalar tail, then the
// 1/sqrt(var+eps) per-element scale. Writes the normalized row to dst.
inline void norm_row_ggml(const int64_t count, float *dst, const float *x,
                          const float eps) noexcept {
  double sum = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    sum += static_cast<double>(x[i]);
  }
  const float mean = static_cast<float>(sum) / static_cast<float>(count);
  double variance_sum = 0.0;
  int64_t i = 0;
#if defined(__AVX2__) && defined(__F16C__) && defined(__FMA__)
  for (; i + 7 < count; i += 8) {
    const __m256 val =
        _mm256_sub_ps(_mm256_loadu_ps(x + i), _mm256_set1_ps(mean));
    _mm256_storeu_ps(dst + i, val);
    const __m256 sq = _mm256_mul_ps(val, val);
    __m128 val2 =
        _mm_add_ps(_mm256_extractf128_ps(sq, 1), _mm256_castps256_ps128(sq));
    val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
    val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
    variance_sum += static_cast<double>(_mm_cvtss_f32(val2));
  }
#endif
  for (; i < count; ++i) {
    const float val = x[i] - mean;
    dst[i] = val;
    variance_sum += static_cast<double>(val * val);
  }
  const float variance =
      static_cast<float>(variance_sum / static_cast<double>(count));
  const float scale = 1.0f / std::sqrt(variance + eps);
  for (int64_t j = 0; j < count; ++j) {
    dst[j] *= scale;
  }
}

template <class request_type>
inline bool run_mul_mat_f16(const request_type &request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[1];
  const auto *src0 = static_cast<const uint16_t *>(request.src0.data);
  const auto *src1 = static_cast<const uint16_t *>(request.src1.data);
  auto *dst = static_cast<float *>(request.dst.data);
  for (uint64_t col = 0; col < n; ++col) {
    const uint16_t *y = src1 + col * k;
    for (uint64_t row = 0; row < m; ++row) {
      dst[row + col * m] =
          vec_dot_f16_ggml(static_cast<int64_t>(k), src0 + row * k, y);
    }
  }
  return true;
}

template <class request_type>
inline bool can_run_soft_max(const request_type &request) noexcept {
  const uint64_t width = request.src0.ne[0];
  const uint64_t count = tensor_element_count(request.src0);
  return width != 0 && count != 0 && count % width == 0 &&
         count == tensor_element_count(request.dst);
}

template <class request_type>
inline float flash_attn_scale(const request_type &request) noexcept {
  if (request.op_params_size >= sizeof(float)) {
    float scale = 0.0f;
    std::memcpy(&scale, request.op_params.data(), sizeof(scale));
    return scale;
  }

  const float head_dim = static_cast<float>(request.src0.ne[0]);
  return head_dim > 0.0f ? (1.0f / std::sqrt(head_dim)) : 1.0f;
}

template <class request_type>
inline uint64_t flash_attn_active_tokens(const request_type &request) noexcept {
  return request.src1.ne[1];
}

template <class request_type>
inline uint64_t
flash_attn_masked_total_tokens(const request_type &request) noexcept {
  if (request.op_params_size >= sizeof(float) + sizeof(uint32_t)) {
    uint32_t total_tokens = 0u;
    std::memcpy(&total_tokens, request.op_params.data() + sizeof(float),
                sizeof(total_tokens));
    return total_tokens;
  }

  return flash_attn_active_tokens(request);
}

inline float round_fp16_scalar(const float value) noexcept {
  return quant::fp16_to_fp32(quant::fp32_to_fp16(value));
}

inline float round_fp16_weight(const float value) noexcept {
  return round_fp16_scalar(value);
}

inline void scale_f32_scalar(float *data, const float scale,
                             const uint64_t count) noexcept {
  for (uint64_t idx = 0; idx < count; ++idx) {
    data[idx] *= scale;
  }
}

inline void axpy_f32_scalar(float *dst, const float *src, const float alpha,
                            const uint64_t count) noexcept {
  for (uint64_t idx = 0; idx < count; ++idx) {
    dst[idx] += src[idx] * alpha;
  }
}

inline void
scale_f16_effective_accumulator_scalar(float *data, const float scale,
                                       const uint64_t count) noexcept {
  const float rounded_scale = round_fp16_scalar(scale);
  for (uint64_t idx = 0; idx < count; ++idx) {
    data[idx] = round_fp16_scalar(round_fp16_scalar(data[idx]) * rounded_scale);
  }
}

inline void
axpy_f16_effective_accumulator_scalar(float *dst, const float *src,
                                      const float alpha,
                                      const uint64_t count) noexcept {
  const float rounded_alpha = round_fp16_scalar(alpha);
  for (uint64_t idx = 0; idx < count; ++idx) {
    const float rounded_dst = round_fp16_scalar(dst[idx]);
    const float rounded_src = round_fp16_scalar(src[idx]);
    dst[idx] = round_fp16_scalar(rounded_dst + rounded_src * rounded_alpha);
  }
}

inline void convert_f32_to_fp16_buffer_scalar(const float *src, uint16_t *dst,
                                              const uint64_t count) noexcept {
  for (uint64_t idx = 0; idx < count; ++idx) {
    dst[idx] = quant::fp32_to_fp16(src[idx]);
  }
}

inline void zero_f16_buffer_scalar(uint16_t *dst,
                                   const uint64_t count) noexcept {
  std::fill_n(dst, count, static_cast<uint16_t>(0u));
}

inline void scale_f16_buffer_scalar(uint16_t *data, const float scale,
                                    const uint64_t count) noexcept {
  const float rounded_scale = round_fp16_scalar(scale);
  for (uint64_t idx = 0; idx < count; ++idx) {
    const float rounded_value = quant::fp16_to_fp32(data[idx]);
    data[idx] =
        quant::fp32_to_fp16(round_fp16_scalar(rounded_value * rounded_scale));
  }
}

inline void axpy_f16_buffer_scalar(uint16_t *dst, const uint16_t *src,
                                   const float alpha,
                                   const uint64_t count) noexcept {
  const float rounded_alpha = round_fp16_scalar(alpha);
  for (uint64_t idx = 0; idx < count; ++idx) {
    const float rounded_dst = quant::fp16_to_fp32(dst[idx]);
    const float rounded_src = quant::fp16_to_fp32(src[idx]);
    dst[idx] = quant::fp32_to_fp16(
        round_fp16_scalar(rounded_dst + rounded_src * rounded_alpha));
  }
}

inline void convert_f16_buffer_to_f32_scalar(const uint16_t *src, float *dst,
                                             const uint64_t count) noexcept {
  for (uint64_t idx = 0; idx < count; ++idx) {
    dst[idx] = quant::fp16_to_fp32(src[idx]);
  }
}

template <class request_type>
inline bool has_required_src2(const request_type &request) noexcept {
  return request.src2.data != nullptr &&
         is_supported_dtype(dtype_code(request.src2.type)) &&
         has_valid_tensor_layout(request.src2) &&
         tensor_element_count(request.src2) > 0;
}

template <class request_type>
inline bool can_run_flash_attn_ext(const request_type &request) noexcept {
  const uint8_t src0_type = dtype_code(request.src0.type);
  const uint8_t src1_type = dtype_code(request.src1.type);
  const uint8_t src2_type = dtype_code(request.src2.type);
  const uint8_t dst_type = dtype_code(request.dst.type);
  const uint64_t head_dim = request.src0.ne[0];
  const uint64_t query_count = request.src0.ne[1];
  const uint64_t head_count = request.src0.ne[2];
  const uint64_t kv_tokens = flash_attn_active_tokens(request);
  const uint64_t kv_head_count = request.src1.ne[2];
  const uint64_t masked_total_tokens = flash_attn_masked_total_tokens(request);

  const bool explicit_operand_contract =
      src0_type == dtype_f32 && src1_type == dtype_f16 &&
      src2_type == dtype_f16 && dst_type == dtype_f32;
  const bool dims_present = head_dim != 0u && query_count == 1u &&
                            head_count != 0u && kv_tokens != 0u &&
                            kv_head_count != 0u;
  const bool src2_valid = has_required_src2(request);
  const bool shape_match =
      request.src1.ne[0] == head_dim && request.src2.ne[0] == head_dim &&
      request.src2.ne[1] == kv_tokens && request.src2.ne[2] == kv_head_count &&
      request.dst.ne[0] == head_dim && request.dst.ne[1] == query_count &&
      request.dst.ne[2] == head_count && request.src0.ne[3] == 1u &&
      request.src1.ne[3] == 1u && request.src2.ne[3] == 1u &&
      request.dst.ne[3] == 1u && kv_head_count != 0u &&
      (head_count % kv_head_count) == 0u;
  const bool layout_supported = is_dense_contiguous(request.src0) &&
                                has_valid_tensor_layout(request.src1) &&
                                has_valid_tensor_layout(request.src2) &&
                                is_dense_contiguous(request.dst);
  const float scale = flash_attn_scale(request);

  return explicit_operand_contract && dims_present && src2_valid &&
         shape_match && layout_supported && masked_total_tokens >= kv_tokens &&
         std::isfinite(scale) && scale > 0.0f;
}

//------------------------------------------------------------------------------//
// Row/index kernel group (get_rows, norms, rope, im2col, conv_transpose_1d).
// Semantics mirror the ggml reference operand contracts so the paritychecker
// kernel engine can compare against ggml directly.
//------------------------------------------------------------------------------//

template <class tensor_type>
inline int32_t read_i32_at(const tensor_type &tensor, const uint64_t i0,
                           const uint64_t i1 = 0,
                           const uint64_t i2 = 0) noexcept {
  const char *base = static_cast<const char *>(tensor.data);
  int32_t out = 0;
  std::memcpy(&out, base + tensor_offset_bytes(tensor, i0, i1, i2),
              sizeof(out));
  return out;
}

inline bool is_get_rows_float_dtype(const uint8_t code) noexcept {
  return code == dtype_f32 || code == dtype_f16 || code == dtype_bf16;
}

// The supported get_rows source dtype variants are modeled as explicit
// guard-selected transition rows in the backend machines (op_unary pattern);
// this predicate only feeds the shared validity guard.
inline bool is_get_rows_quantized_dtype(const uint8_t code) noexcept {
  return code == dtype_q4_0 || code == dtype_q8_0 || code == dtype_q4_k;
}

// Dequantize/convert one already-selected row into f32. The source dtype is a
// compile-time variant chosen by the transition row, never a runtime branch.
template <uint8_t src_dtype_code>
inline void convert_row_to_f32_as(const void *src, float *dst,
                                  const int64_t cols) noexcept {
  if constexpr (src_dtype_code == dtype_f32) {
    std::memcpy(dst, src, static_cast<size_t>(cols) * sizeof(float));
  } else if constexpr (src_dtype_code == dtype_f16) {
    const uint16_t *bits = static_cast<const uint16_t *>(src);
    for (int64_t i = 0; i < cols; ++i) {
      dst[i] = quant::fp16_to_fp32(bits[i]);
    }
  } else if constexpr (src_dtype_code == dtype_bf16) {
    const uint16_t *bits = static_cast<const uint16_t *>(src);
    for (int64_t i = 0; i < cols; ++i) {
      dst[i] = bf16_to_fp32(bits[i]);
    }
  } else if constexpr (src_dtype_code == dtype_q4_0) {
    quant::dequantize_row_q4_0(static_cast<const quant::block_q4_0 *>(src), dst,
                               cols);
  } else if constexpr (src_dtype_code == dtype_q8_0) {
    quant::dequantize_row_q8_0(static_cast<const quant::block_q8_0 *>(src), dst,
                               cols);
  } else if constexpr (src_dtype_code == dtype_q4_k) {
    quant::dequantize_row_q4_k(static_cast<const quant::block_q4_k *>(src), dst,
                               cols);
  }
}

template <class tensor_type>
inline bool has_quantized_row_layout(const tensor_type &tensor,
                                     const uint8_t code) noexcept {
  const uint64_t cols = tensor.ne[0];
  const size_t row_bytes = quantized_row_storage_bytes(code, cols);
  return row_bytes != 0u && tensor.nb[0] == 1u && tensor.nb[1] == row_bytes &&
         tensor.nb[2] == tensor.nb[1] * tensor.ne[1] &&
         tensor.nb[3] == tensor.nb[2] * tensor.ne[2];
}

template <class request_type>
inline bool can_run_get_rows(const request_type &request) noexcept {
  const uint8_t src0_type = dtype_code(request.src0.type);
  const uint8_t src1_type = dtype_code(request.src1.type);
  const uint8_t dst_type = dtype_code(request.dst.type);
  const uint64_t cols = request.src0.ne[0];
  const uint64_t rows = request.src0.ne[1];

  const bool shapes_ok = cols > 0 && rows > 0 && request.dst.ne[0] == cols &&
                         request.dst.ne[1] == request.src1.ne[0] &&
                         request.dst.ne[2] == request.src1.ne[1] &&
                         request.dst.ne[3] == request.src1.ne[2] &&
                         request.src0.ne[2] == request.src1.ne[1] &&
                         request.src0.ne[3] == request.src1.ne[2] &&
                         request.src1.ne[3] == 1;
  const bool float_src = is_get_rows_float_dtype(src0_type) &&
                         has_valid_tensor_layout(request.src0);
  const bool quant_src = is_get_rows_quantized_dtype(src0_type) &&
                         has_quantized_row_layout(request.src0, src0_type);
  const bool types_ok = src1_type == dtype_i32 && dst_type == dtype_f32;
  const bool layouts_ok =
      has_valid_tensor_layout(request.src1) && is_dense_contiguous(request.dst);
  // The index scan below reads through src1.data, so the guard must require
  // bound index storage before deciding validity.
  if (!(shapes_ok && types_ok && (float_src || quant_src) && layouts_ok &&
        request.src1.data != nullptr)) {
    return false;
  }

  for (uint64_t i2 = 0; i2 < request.src1.ne[2]; ++i2) {
    for (uint64_t i1 = 0; i1 < request.src1.ne[1]; ++i1) {
      for (uint64_t i0 = 0; i0 < request.src1.ne[0]; ++i0) {
        const int32_t row = read_i32_at(request.src1, i0, i1, i2);
        if (row < 0 || static_cast<uint64_t>(row) >= rows) {
          return false;
        }
      }
    }
  }
  return true;
}

template <uint8_t src_dtype_code, class request_type>
inline bool run_get_rows_as(const request_type &request) noexcept {
  const int64_t cols = static_cast<int64_t>(request.src0.ne[0]);
  const char *src_base = static_cast<const char *>(request.src0.data);
  float *dst = static_cast<float *>(request.dst.data);
  const uint64_t nb01 = tensor_stride_bytes(request.src0, 1);
  const uint64_t nb02 = tensor_stride_bytes(request.src0, 2);
  const uint64_t nb03 = tensor_stride_bytes(request.src0, 3);

  uint64_t dst_row = 0;
  for (uint64_t i2 = 0; i2 < request.src1.ne[2]; ++i2) {
    for (uint64_t i1 = 0; i1 < request.src1.ne[1]; ++i1) {
      for (uint64_t i0 = 0; i0 < request.src1.ne[0]; ++i0, ++dst_row) {
        const auto row =
            static_cast<uint64_t>(read_i32_at(request.src1, i0, i1, i2));
        const char *src_row = src_base + row * nb01 + i1 * nb02 + i2 * nb03;
        convert_row_to_f32_as<src_dtype_code>(
            src_row, dst + dst_row * static_cast<uint64_t>(cols), cols);
      }
    }
  }
  return true;
}

inline bool read_op_param_f32(const uint8_t *params, const uint32_t params_size,
                              const uint32_t slot, float &out) noexcept {
  if ((slot + 1u) * sizeof(float) > params_size) {
    return false;
  }
  std::memcpy(&out, params + slot * sizeof(float), sizeof(out));
  return true;
}

inline bool read_op_param_i32(const uint8_t *params, const uint32_t params_size,
                              const uint32_t slot, int32_t &out) noexcept {
  if ((slot + 1u) * sizeof(int32_t) > params_size) {
    return false;
  }
  std::memcpy(&out, params + slot * sizeof(int32_t), sizeof(out));
  return true;
}

template <class request_type>
inline bool norm_row_epsilon(const request_type &request,
                             float &eps_out) noexcept {
  return read_op_param_f32(request.op_params.data(), request.op_params_size, 0u,
                           eps_out) &&
         std::isfinite(eps_out) && eps_out >= 0.0f;
}

template <class request_type>
inline bool can_run_norm_row_op(const request_type &request) noexcept {
  float eps = 0.0f;
  const bool same_shape = request.src0.ne[0] == request.dst.ne[0] &&
                          request.src0.ne[1] == request.dst.ne[1] &&
                          request.src0.ne[2] == request.dst.ne[2] &&
                          request.src0.ne[3] == request.dst.ne[3];
  return request.src0.ne[0] > 0 && same_shape &&
         dtype_code(request.src0.type) == dtype_f32 &&
         dtype_code(request.dst.type) == dtype_f32 &&
         has_valid_tensor_layout(request.src0) &&
         has_valid_tensor_layout(request.dst) && norm_row_epsilon(request, eps);
}

template <class request_type>
inline bool run_rms_norm(const request_type &request) noexcept {
  float eps = 0.0f;
  (void)norm_row_epsilon(request, eps);
  const uint64_t cols = request.src0.ne[0];
  const char *src_base = static_cast<const char *>(request.src0.data);
  char *dst_base = static_cast<char *>(request.dst.data);
  const uint64_t src_nb0 = tensor_stride_bytes(request.src0, 0);
  const uint64_t dst_nb0 = tensor_stride_bytes(request.dst, 0);

  for (uint64_t i3 = 0; i3 < request.src0.ne[3]; ++i3) {
    for (uint64_t i2 = 0; i2 < request.src0.ne[2]; ++i2) {
      for (uint64_t i1 = 0; i1 < request.src0.ne[1]; ++i1) {
        const char *src_row =
            src_base + tensor_offset_bytes(request.src0, 0, i1, i2, i3);
        char *dst_row =
            dst_base + tensor_offset_bytes(request.dst, 0, i1, i2, i3);
        double sum = 0.0;
        for (uint64_t i0 = 0; i0 < cols; ++i0) {
          float value = 0.0f;
          std::memcpy(&value, src_row + i0 * src_nb0, sizeof(value));
          sum += static_cast<double>(value) * static_cast<double>(value);
        }
        const float scale =
            1.0f /
            std::sqrt(static_cast<float>(sum / static_cast<double>(cols)) +
                      eps);
        for (uint64_t i0 = 0; i0 < cols; ++i0) {
          float value = 0.0f;
          std::memcpy(&value, src_row + i0 * src_nb0, sizeof(value));
          const float scaled = value * scale;
          std::memcpy(dst_row + i0 * dst_nb0, &scaled, sizeof(scaled));
        }
      }
    }
  }
  return true;
}

template <class request_type>
inline bool run_norm(const request_type &request) noexcept {
  float eps = 0.0f;
  (void)norm_row_epsilon(request, eps);
  const uint64_t cols = request.src0.ne[0];
  const char *src_base = static_cast<const char *>(request.src0.data);
  char *dst_base = static_cast<char *>(request.dst.data);
  const uint64_t src_nb0 = tensor_stride_bytes(request.src0, 0);
  const uint64_t dst_nb0 = tensor_stride_bytes(request.dst, 0);

  for (uint64_t i3 = 0; i3 < request.src0.ne[3]; ++i3) {
    for (uint64_t i2 = 0; i2 < request.src0.ne[2]; ++i2) {
      for (uint64_t i1 = 0; i1 < request.src0.ne[1]; ++i1) {
        const char *src_row =
            src_base + tensor_offset_bytes(request.src0, 0, i1, i2, i3);
        char *dst_row =
            dst_base + tensor_offset_bytes(request.dst, 0, i1, i2, i3);
        double sum = 0.0;
        for (uint64_t i0 = 0; i0 < cols; ++i0) {
          float value = 0.0f;
          std::memcpy(&value, src_row + i0 * src_nb0, sizeof(value));
          sum += static_cast<double>(value);
        }
        const float mean = static_cast<float>(sum / static_cast<double>(cols));
        double sum2 = 0.0;
        for (uint64_t i0 = 0; i0 < cols; ++i0) {
          float value = 0.0f;
          std::memcpy(&value, src_row + i0 * src_nb0, sizeof(value));
          const float centered = value - mean;
          sum2 += static_cast<double>(centered) * static_cast<double>(centered);
        }
        const float variance =
            static_cast<float>(sum2 / static_cast<double>(cols));
        const float scale = 1.0f / std::sqrt(variance + eps);
        for (uint64_t i0 = 0; i0 < cols; ++i0) {
          float value = 0.0f;
          std::memcpy(&value, src_row + i0 * src_nb0, sizeof(value));
          const float normalized = (value - mean) * scale;
          std::memcpy(dst_row + i0 * dst_nb0, &normalized, sizeof(normalized));
        }
      }
    }
  }
  return true;
}

// ggml rope op_params layout: i32 slots {unused, n_dims, mode, unused,
// n_ctx_orig}, f32 slots {freq_base, freq_scale, ext_factor, attn_factor,
// beta_fast, beta_slow} at indexes 5..10.
struct rope_op_params {
  int32_t n_dims = 0;
  int32_t mode = 0;
  float freq_base = 0.0f;
  float freq_scale = 0.0f;
  float ext_factor = 0.0f;
  float attn_factor = 0.0f;
};

inline constexpr int32_t rope_mode_norm = 0;
inline constexpr int32_t rope_mode_neox = 2;

template <class request_type>
inline bool read_rope_params(const request_type &request,
                             rope_op_params &params_out) noexcept {
  const uint8_t *params = request.op_params.data();
  const uint32_t size = request.op_params_size;
  return read_op_param_i32(params, size, 1u, params_out.n_dims) &&
         read_op_param_i32(params, size, 2u, params_out.mode) &&
         read_op_param_f32(params, size, 5u, params_out.freq_base) &&
         read_op_param_f32(params, size, 6u, params_out.freq_scale) &&
         read_op_param_f32(params, size, 7u, params_out.ext_factor) &&
         read_op_param_f32(params, size, 8u, params_out.attn_factor);
}

template <class request_type>
inline bool can_run_rope(const request_type &request) noexcept {
  rope_op_params params = {};
  const bool same_shape = request.src0.ne[0] == request.dst.ne[0] &&
                          request.src0.ne[1] == request.dst.ne[1] &&
                          request.src0.ne[2] == request.dst.ne[2] &&
                          request.src0.ne[3] == request.dst.ne[3];
  return read_rope_params(request, params) && same_shape &&
         request.src0.data != nullptr && request.src1.data != nullptr &&
         request.src0.ne[0] > 0 && dtype_code(request.src0.type) == dtype_f32 &&
         dtype_code(request.dst.type) == dtype_f32 &&
         dtype_code(request.src1.type) == dtype_i32 &&
         request.src2.data == nullptr &&
         has_valid_tensor_layout(request.src0) &&
         has_valid_tensor_layout(request.src1) &&
         has_valid_tensor_layout(request.dst) &&
         request.src1.ne[0] == request.src0.ne[2] && params.n_dims > 0 &&
         (params.n_dims % 2) == 0 &&
         static_cast<uint64_t>(params.n_dims) <= request.src0.ne[0] &&
         (params.mode == rope_mode_norm || params.mode == rope_mode_neox) &&
         params.ext_factor == 0.0f && params.freq_base > 0.0f &&
         params.freq_scale > 0.0f && std::isfinite(params.attn_factor);
}

// The rotation pairing (norm vs neox) is a compile-time variant chosen by the
// transition row, never a runtime branch.
template <bool neox, class request_type>
inline bool run_rope_as(const request_type &request) noexcept {
  rope_op_params params = {};
  (void)read_rope_params(request, params);
  const uint64_t cols = request.src0.ne[0];
  const auto half_dims = static_cast<uint64_t>(params.n_dims) / 2u;
  const float theta_scale =
      std::pow(params.freq_base, -2.0f / static_cast<float>(params.n_dims));
  const char *src_base = static_cast<const char *>(request.src0.data);
  char *dst_base = static_cast<char *>(request.dst.data);
  const uint64_t src_nb0 = tensor_stride_bytes(request.src0, 0);
  const uint64_t dst_nb0 = tensor_stride_bytes(request.dst, 0);

  for (uint64_t i3 = 0; i3 < request.src0.ne[3]; ++i3) {
    for (uint64_t i2 = 0; i2 < request.src0.ne[2]; ++i2) {
      const auto position = static_cast<float>(read_i32_at(request.src1, i2));
      for (uint64_t i1 = 0; i1 < request.src0.ne[1]; ++i1) {
        const char *src_row =
            src_base + tensor_offset_bytes(request.src0, 0, i1, i2, i3);
        char *dst_row =
            dst_base + tensor_offset_bytes(request.dst, 0, i1, i2, i3);
        const auto load = [&](const uint64_t i0) noexcept {
          float value = 0.0f;
          std::memcpy(&value, src_row + i0 * src_nb0, sizeof(value));
          return value;
        };
        const auto store = [&](const uint64_t i0, const float value) noexcept {
          std::memcpy(dst_row + i0 * dst_nb0, &value, sizeof(value));
        };

        float theta = position;
        for (uint64_t pair = 0; pair < half_dims;
             ++pair, theta *= theta_scale) {
          const float rotated = theta * params.freq_scale;
          const float cos_theta = std::cos(rotated) * params.attn_factor;
          const float sin_theta = std::sin(rotated) * params.attn_factor;
          uint64_t i0a = 2u * pair;
          uint64_t i0b = 2u * pair + 1u;
          if constexpr (neox) {
            i0a = pair;
            i0b = pair + half_dims;
          }
          const float x0 = load(i0a);
          const float x1 = load(i0b);
          store(i0a, x0 * cos_theta - x1 * sin_theta);
          store(i0b, x0 * sin_theta + x1 * cos_theta);
        }
        for (uint64_t i0 = static_cast<uint64_t>(params.n_dims); i0 < cols;
             ++i0) {
          store(i0, load(i0));
        }
      }
    }
  }
  return true;
}

// ggml im2col op_params layout: i32 slots {s0, s1, p0, p1, d0, d1, is_2D}.
struct im2col_op_params {
  int32_t s0 = 0;
  int32_t p0 = 0;
  int32_t d0 = 0;
  int32_t is_2d = 0;
};

template <class request_type>
inline bool read_im2col_params(const request_type &request,
                               im2col_op_params &params_out) noexcept {
  const uint8_t *params = request.op_params.data();
  const uint32_t size = request.op_params_size;
  return read_op_param_i32(params, size, 0u, params_out.s0) &&
         read_op_param_i32(params, size, 2u, params_out.p0) &&
         read_op_param_i32(params, size, 4u, params_out.d0) &&
         read_op_param_i32(params, size, 6u, params_out.is_2d);
}

inline int64_t conv_output_length(const int64_t length, const int64_t kernel,
                                  const int64_t stride, const int64_t padding,
                                  const int64_t dilation) noexcept {
  return (length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
}

template <class request_type>
inline bool can_run_im2col(const request_type &request) noexcept {
  im2col_op_params params = {};
  if (!read_im2col_params(request, params) || params.is_2d != 0 ||
      params.s0 <= 0 || params.d0 <= 0 || params.p0 < 0) {
    return false;
  }

  const uint8_t dst_type = dtype_code(request.dst.type);
  const int64_t kernel = static_cast<int64_t>(request.src0.ne[0]);
  const int64_t channels = static_cast<int64_t>(request.src0.ne[1]);
  const int64_t length = static_cast<int64_t>(request.src1.ne[0]);
  const int64_t out_length =
      conv_output_length(length, kernel, params.s0, params.p0, params.d0);
  const uint8_t src0_type = dtype_code(request.src0.type);
  return request.src1.data != nullptr && kernel > 0 && channels > 0 &&
         length > 0 && out_length > 0 &&
         (src0_type == dtype_f32 || src0_type == dtype_f16) &&
         dtype_code(request.src1.type) == dtype_f32 &&
         (dst_type == dtype_f32 || dst_type == dtype_f16) &&
         request.src1.ne[1] == static_cast<uint64_t>(channels) &&
         request.src1.ne[3] == 1 &&
         request.dst.ne[0] == static_cast<uint64_t>(channels * kernel) &&
         request.dst.ne[1] == static_cast<uint64_t>(out_length) &&
         request.dst.ne[2] == request.src1.ne[2] && request.dst.ne[3] == 1 &&
         has_valid_tensor_layout(request.src1) &&
         is_dense_contiguous(request.dst);
}

// The destination dtype (f32 columns vs f16 columns) is a compile-time
// variant chosen by the transition row, never a runtime branch.
template <bool f16_dst, class request_type>
inline bool run_im2col_as(const request_type &request) noexcept {
  im2col_op_params params = {};
  (void)read_im2col_params(request, params);
  const int64_t kernel = static_cast<int64_t>(request.src0.ne[0]);
  const int64_t channels = static_cast<int64_t>(request.src0.ne[1]);
  const int64_t length = static_cast<int64_t>(request.src1.ne[0]);
  const int64_t out_length = static_cast<int64_t>(request.dst.ne[1]);
  const int64_t batches = static_cast<int64_t>(request.dst.ne[2]);

  float *dst_f32 = static_cast<float *>(request.dst.data);
  uint16_t *dst_f16 = static_cast<uint16_t *>(request.dst.data);
  const uint64_t row_width = static_cast<uint64_t>(channels * kernel);

  for (int64_t batch = 0; batch < batches; ++batch) {
    for (int64_t out = 0; out < out_length; ++out) {
      const uint64_t dst_row =
          (static_cast<uint64_t>(batch) * static_cast<uint64_t>(out_length) +
           static_cast<uint64_t>(out)) *
          row_width;
      for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t tap = 0; tap < kernel; ++tap) {
          const int64_t in = out * params.s0 + tap * params.d0 - params.p0;
          const bool inside = in >= 0 && in < length;
          const float value =
              inside ? read_f32_at(request.src1, static_cast<uint64_t>(in),
                                   static_cast<uint64_t>(channel),
                                   static_cast<uint64_t>(batch))
                     : 0.0f;
          const uint64_t dst_index =
              dst_row +
              static_cast<uint64_t>(channel) * static_cast<uint64_t>(kernel) +
              static_cast<uint64_t>(tap);
          if constexpr (f16_dst) {
            dst_f16[dst_index] = quant::fp32_to_fp16(value);
          } else {
            dst_f32[dst_index] = value;
          }
        }
      }
    }
  }
  return true;
}

// ggml conv_transpose_1d op_params layout: i32 slots {s0, p0, d0}; the
// reference kernel only supports p0 == 0 and d0 == 1.
template <class request_type>
inline bool can_run_conv_transpose_1d(const request_type &request) noexcept {
  int32_t s0 = 0;
  int32_t p0 = 0;
  int32_t d0 = 0;
  const uint8_t src0_type = dtype_code(request.src0.type);
  if (!read_op_param_i32(request.op_params.data(), request.op_params_size, 0u,
                         s0) ||
      !read_op_param_i32(request.op_params.data(), request.op_params_size, 1u,
                         p0) ||
      !read_op_param_i32(request.op_params.data(), request.op_params_size, 2u,
                         d0) ||
      s0 <= 0 || p0 != 0 || d0 != 1) {
    return false;
  }

  const int64_t kernel = static_cast<int64_t>(request.src0.ne[0]);
  const int64_t out_channels = static_cast<int64_t>(request.src0.ne[1]);
  const int64_t in_channels = static_cast<int64_t>(request.src0.ne[2]);
  const int64_t length = static_cast<int64_t>(request.src1.ne[0]);
  const int64_t out_length = (length - 1) * s0 + kernel;
  // The exec reads through both operands; metadata-only tensors must reject
  // here instead of dereferencing null inside the action.
  return request.src0.data != nullptr && request.src1.data != nullptr &&
         kernel > 0 && out_channels > 0 && in_channels > 0 && length > 0 &&
         (src0_type == dtype_f32 || src0_type == dtype_f16) &&
         dtype_code(request.src1.type) == dtype_f32 &&
         dtype_code(request.dst.type) == dtype_f32 && request.src0.ne[3] == 1 &&
         request.src1.ne[1] == static_cast<uint64_t>(in_channels) &&
         request.src1.ne[2] == 1 && request.src1.ne[3] == 1 &&
         request.dst.ne[0] == static_cast<uint64_t>(out_length) &&
         request.dst.ne[1] == static_cast<uint64_t>(out_channels) &&
         request.dst.ne[2] == 1 && request.dst.ne[3] == 1 &&
         has_valid_tensor_layout(request.src0) &&
         has_valid_tensor_layout(request.src1) &&
         is_dense_contiguous(request.dst);
}

// The weight dtype (f32 vs f16 taps) is a compile-time variant chosen by the
// transition row, never a runtime branch.
template <bool f16_weights, class request_type>
inline bool run_conv_transpose_1d_as(const request_type &request) noexcept {
  int32_t s0 = 0;
  (void)read_op_param_i32(request.op_params.data(), request.op_params_size, 0u,
                          s0);
  const int64_t kernel = static_cast<int64_t>(request.src0.ne[0]);
  const int64_t out_channels = static_cast<int64_t>(request.src0.ne[1]);
  const int64_t in_channels = static_cast<int64_t>(request.src0.ne[2]);
  const int64_t length = static_cast<int64_t>(request.src1.ne[0]);
  const int64_t out_length = static_cast<int64_t>(request.dst.ne[0]);

  float *dst = static_cast<float *>(request.dst.data);
  std::fill_n(dst, static_cast<size_t>(out_length * out_channels), 0.0f);

  const char *weight_base = static_cast<const char *>(request.src0.data);
  const uint64_t w_nb0 = tensor_stride_bytes(request.src0, 0);
  const uint64_t w_nb1 = tensor_stride_bytes(request.src0, 1);
  const uint64_t w_nb2 = tensor_stride_bytes(request.src0, 2);

  for (int64_t in_channel = 0; in_channel < in_channels; ++in_channel) {
    for (int64_t in = 0; in < length; ++in) {
      float input = read_f32_at(request.src1, static_cast<uint64_t>(in),
                                static_cast<uint64_t>(in_channel));
      if constexpr (f16_weights) {
        // The reference f16 kernel path rounds the input samples through fp16
        // before the tap multiplies; match that operand pipeline exactly.
        input = quant::fp16_to_fp32(quant::fp32_to_fp16(input));
      }
      for (int64_t out_channel = 0; out_channel < out_channels; ++out_channel) {
        float *dst_row = dst + out_channel * out_length;
        const char *weight_row = weight_base +
                                 static_cast<uint64_t>(out_channel) * w_nb1 +
                                 static_cast<uint64_t>(in_channel) * w_nb2;
        for (int64_t tap = 0; tap < kernel; ++tap) {
          float weight = 0.0f;
          if constexpr (f16_weights) {
            uint16_t bits = 0;
            std::memcpy(&bits, weight_row + static_cast<uint64_t>(tap) * w_nb0,
                        sizeof(bits));
            weight = quant::fp16_to_fp32(bits);
          } else {
            std::memcpy(&weight,
                        weight_row + static_cast<uint64_t>(tap) * w_nb0,
                        sizeof(weight));
          }
          dst_row[in * s0 + tap] += input * weight;
        }
      }
    }
  }
  return true;
}

// Variant guard combiner and exec functors for the ops whose dtype/mode
// variants are modeled as explicit transition rows (op_unary pattern): the
// backend guard supplies the validity predicate, the variant predicate picks
// the row, and the exec functor is instantiated per variant at compile time.
template <class dispatch_event_type, class context_type, class valid_guard_type,
          class variant_guard_type>
struct valid_variant_guard {
  bool operator()(const dispatch_event_type &ev,
                  const context_type &ctx) const noexcept {
    return valid_guard_type{}(ev, ctx) && variant_guard_type{}(ev, ctx);
  }
};

template <class dispatch_event_type, class context_type, class mark_done_type,
          uint8_t src_dtype_code>
struct exec_scalar_get_rows_op {
  void operator()(const dispatch_event_type &ev,
                  context_type &ctx) const noexcept {
    (void)run_get_rows_as<src_dtype_code>(ev.request);
    mark_done_type{}(ev, ctx);
  }
};

template <class dispatch_event_type, class context_type, class mark_done_type>
struct exec_scalar_mul_mat_f16_op {
  void operator()(const dispatch_event_type &ev,
                  context_type &ctx) const noexcept {
    (void)run_mul_mat_f16(ev.request);
    mark_done_type{}(ev, ctx);
  }
};

template <class dispatch_event_type, class context_type, class mark_done_type,
          bool neox_mode>
struct exec_scalar_rope_op {
  void operator()(const dispatch_event_type &ev,
                  context_type &ctx) const noexcept {
    (void)run_rope_as<neox_mode>(ev.request);
    mark_done_type{}(ev, ctx);
  }
};

template <class dispatch_event_type, class context_type, class mark_done_type,
          bool f16_dst>
struct exec_scalar_im2col_op {
  void operator()(const dispatch_event_type &ev,
                  context_type &ctx) const noexcept {
    (void)run_im2col_as<f16_dst>(ev.request);
    mark_done_type{}(ev, ctx);
  }
};

template <class dispatch_event_type, class context_type, class mark_done_type,
          bool f16_weights>
struct exec_scalar_conv_transpose_1d_op {
  void operator()(const dispatch_event_type &ev,
                  context_type &ctx) const noexcept {
    (void)run_conv_transpose_1d_as<f16_weights>(ev.request);
    mark_done_type{}(ev, ctx);
  }
};

template <class dispatch_event_type, class context_type, class mark_done_type,
          bool multiply>
struct exec_scalar_binary_broadcast_row_op {
  void operator()(const dispatch_event_type &ev,
                  context_type &ctx) const noexcept {
    if constexpr (multiply) {
      (void)run_binary_broadcast_row(
          ev.request,
          [](const float lhs, const float rhs) { return lhs * rhs; });
    } else {
      (void)run_binary_broadcast_row(
          ev.request,
          [](const float lhs, const float rhs) { return lhs + rhs; });
    }
    mark_done_type{}(ev, ctx);
  }
};

template <class request_type>
inline bool can_run_unary_subop(const request_type &request) noexcept {
  const auto subop = static_cast<uint8_t>(request.subop);
  const bool supported_subop =
      subop == unary_subop_abs || subop == unary_subop_neg ||
      subop == unary_subop_tanh || subop == unary_subop_elu ||
      subop == unary_subop_relu || subop == unary_subop_gelu ||
      subop == unary_subop_silu || subop == unary_subop_exp;
  return supported_subop && can_run_unary(request);
}

template <class request_type>
inline bool can_execute_scalar(const request_type &request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_dup>) {
    return can_run_copy(request);
  } else if constexpr (std::is_same_v<request_type, event::op_add>) {
    return can_run_binary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_sub>) {
    return can_run_binary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_mul>) {
    return can_run_binary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_div>) {
    return can_run_binary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_sqr>) {
    return can_run_unary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_sqrt>) {
    return can_run_unary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_log>) {
    return can_run_unary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_sin>) {
    return can_run_unary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_cos>) {
    return can_run_unary(request);
  } else if constexpr (std::is_same_v<request_type, event::op_mul_mat>) {
    return can_run_mul_mat(request);
  } else if constexpr (std::is_same_v<request_type, event::op_mul_mat_argmax>) {
    return can_run_mul_mat_argmax(request);
  } else if constexpr (std::is_same_v<request_type, event::op_soft_max>) {
    return can_run_soft_max(request);
  } else if constexpr (std::is_same_v<request_type, event::op_flash_attn_ext>) {
    return can_run_flash_attn_ext(request);
  } else if constexpr (std::is_same_v<request_type, event::op_rms_norm>) {
    return can_run_norm_row_op(request);
  } else if constexpr (std::is_same_v<request_type, event::op_norm>) {
    return can_run_norm_row_op(request);
  } else if constexpr (std::is_same_v<request_type, event::op_unary>) {
    return false;
  }
  // op_get_rows / op_rope / op_im2col / op_conv_transpose_1d execute through
  // explicit per-variant transition rows, not the generic scalar path.
  return false;
}

template <class request_type>
inline bool
run_flash_attn_ext_with_workspace(const request_type &request,
                                  flash_attn_workspace &workspace) noexcept;

template <class request_type>
inline bool run_flash_attn_ext(const request_type &request) noexcept {
  flash_attn_workspace workspace{};
  return run_flash_attn_ext_with_workspace(request, workspace);
}

template <class request_type>
inline bool can_run_flash_attn_ext_with_workspace(
    const request_type &request,
    const flash_attn_workspace &workspace) noexcept {
  const uint64_t head_dim = request.src0.ne[0];
  return can_run_flash_attn_ext(request) &&
         head_dim <= workspace.q_buffer_f16.size() &&
         head_dim <= workspace.accum_buffer_f16.size();
}

template <class request_type>
inline void prepare_flash_attn_workspace_active_kv(
    const request_type &request, flash_attn_workspace &workspace) noexcept {
  const uint64_t kv_tokens = flash_attn_active_tokens(request);
  const bool reusing = workspace.prepared_tokens == kv_tokens;
  workspace.reuse_count += static_cast<uint64_t>(reusing);
  workspace.prepared_tokens = kv_tokens;
}

template <class request_type>
inline void run_flash_attn_ext_active_kv_with_workspace_unchecked(
    const request_type &request, flash_attn_workspace &workspace) noexcept {
  prepare_flash_attn_workspace_active_kv(request, workspace);

  const uint64_t kv_tokens = flash_attn_active_tokens(request);
  const uint64_t head_dim = request.src0.ne[0];
  const uint64_t head_count = request.src0.ne[2];
  const uint64_t kv_head_count = request.src1.ne[2];
  const float scale = flash_attn_scale(request);

  const uint64_t n_rep = head_count / kv_head_count;
  for (uint64_t head = 0; head < head_count; ++head) {
    const uint64_t kv_head = head / n_rep;
    const float *q = tensor_row_ptr(request.src0, 0u, head);
    uint16_t *accum = workspace.accum_buffer_f16.data();
    float *dst = tensor_row_ptr_mut(request.dst, 0u, head);
    convert_f32_to_fp16_buffer_scalar(q, workspace.q_buffer_f16.data(),
                                      head_dim);
    zero_f16_buffer_scalar(accum, head_dim);

    float score_sum = 0.0f;
    float max_score = -std::numeric_limits<float>::infinity();
    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const uint16_t *k =
          tensor_row_ptr_as<uint16_t>(request.src1, token, kv_head);
      const float score = dot_product_f16_f16_scores(
                              workspace.q_buffer_f16.data(), k, head_dim) *
                          scale;
      const float old_max = max_score;
      float max_scale = 1.0f;
      float value_scale = 1.0f;
      if (score > max_score) {
        max_score = score;
        max_scale = std::exp(old_max - max_score);
        scale_f16_buffer_scalar(accum, max_scale, head_dim);
      } else {
        value_scale = std::exp(score - max_score);
      }

      const uint16_t *v =
          tensor_row_ptr_as<uint16_t>(request.src2, token, kv_head);
      axpy_f16_buffer_scalar(accum, v, value_scale, head_dim);
      score_sum = score_sum * max_scale + value_scale;
    }

    convert_f16_buffer_to_f32_scalar(accum, dst, head_dim);
    if (score_sum == 0.0f) {
      std::fill_n(dst, head_dim, 0.0f);
    } else {
      scale_f32_scalar(dst, 1.0f / score_sum, head_dim);
    }
  }
}

template <class request_type>
inline bool run_flash_attn_ext_active_kv_with_workspace(
    const request_type &request, flash_attn_workspace &workspace) noexcept {
  if (!can_run_flash_attn_ext_with_workspace(request, workspace)) {
    return false;
  }
  run_flash_attn_ext_active_kv_with_workspace_unchecked(request, workspace);
  return true;
}

template <class request_type>
inline bool
run_flash_attn_ext_with_workspace(const request_type &request,
                                  flash_attn_workspace &workspace) noexcept {
  if (!can_run_flash_attn_ext_with_workspace(request, workspace)) {
    return false;
  }

  run_flash_attn_ext_active_kv_with_workspace_unchecked(request, workspace);
  return true;
}

template <class request_type>
inline bool can_run_backend_request(const request_type &request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_unary>) {
    return can_run_unary_subop(request);
  } else if constexpr (std::is_same_v<request_type, event::op_get_rows>) {
    return can_run_get_rows(request);
  } else if constexpr (std::is_same_v<request_type, event::op_rope>) {
    return can_run_rope(request);
  } else if constexpr (std::is_same_v<request_type, event::op_im2col>) {
    return can_run_im2col(request);
  } else if constexpr (std::is_same_v<request_type,
                                      event::op_conv_transpose_1d>) {
    return can_run_conv_transpose_1d(request);
  } else if constexpr (std::is_same_v<request_type, event::op_add> ||
                       std::is_same_v<request_type, event::op_mul>) {
    return can_run_binary(request) || can_run_binary_broadcast_row(request);
  } else if constexpr (std::is_same_v<request_type, event::op_mul_mat>) {
    return can_run_mul_mat(request) || can_run_mul_mat_f16(request);
  }
  return can_execute_scalar(request);
}

template <class request_type>
inline void execute_scalar_unchecked(const request_type &request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_dup>) {
    (void)run_copy(request);
  } else if constexpr (std::is_same_v<request_type, event::op_add>) {
    (void)run_binary(
        request, [](const float lhs, const float rhs) { return lhs + rhs; });
  } else if constexpr (std::is_same_v<request_type, event::op_sub>) {
    (void)run_binary(
        request, [](const float lhs, const float rhs) { return lhs - rhs; });
  } else if constexpr (std::is_same_v<request_type, event::op_mul>) {
    (void)run_binary(
        request, [](const float lhs, const float rhs) { return lhs * rhs; });
  } else if constexpr (std::is_same_v<request_type, event::op_div>) {
    (void)run_binary(
        request, [](const float lhs, const float rhs) { return lhs / rhs; });
  } else if constexpr (std::is_same_v<request_type, event::op_sqr>) {
    (void)run_unary(request, [](const float v) { return v * v; });
  } else if constexpr (std::is_same_v<request_type, event::op_sqrt>) {
    (void)run_unary(request, [](const float v) { return std::sqrt(v); });
  } else if constexpr (std::is_same_v<request_type, event::op_log>) {
    (void)run_unary(request, [](const float v) { return std::log(v); });
  } else if constexpr (std::is_same_v<request_type, event::op_sin>) {
    (void)run_unary(request, [](const float v) { return std::sin(v); });
  } else if constexpr (std::is_same_v<request_type, event::op_cos>) {
    (void)run_unary(request, [](const float v) { return std::cos(v); });
  } else if constexpr (std::is_same_v<request_type, event::op_mul_mat>) {
    (void)run_mul_mat(request);
  } else if constexpr (std::is_same_v<request_type, event::op_mul_mat_argmax>) {
    (void)run_mul_mat_argmax(request);
  } else if constexpr (std::is_same_v<request_type, event::op_soft_max>) {
    (void)run_soft_max(request);
  } else if constexpr (std::is_same_v<request_type, event::op_flash_attn_ext>) {
    (void)run_flash_attn_ext(request);
  } else if constexpr (std::is_same_v<request_type, event::op_rms_norm>) {
    (void)run_rms_norm(request);
  } else if constexpr (std::is_same_v<request_type, event::op_norm>) {
    (void)run_norm(request);
  }
}

template <class request_type>
inline bool execute_scalar(const request_type &request) noexcept {
  const bool can_execute = can_execute_scalar(request);
  if (can_execute) {
    execute_scalar_unchecked(request);
  }
  return can_execute;
}

} // namespace emel::kernel::detail
