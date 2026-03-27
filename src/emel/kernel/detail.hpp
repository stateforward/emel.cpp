#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

// Keep this list aligned with `tmp/llama.cpp/ggml/include/ggml.h` (`enum ggml_op`),
// excluding sentinel entries (`NONE`, `COUNT`).
#define EMEL_KERNEL_OP_EVENT_LIST(X) \
  X(op_dup) \
  X(op_add) \
  X(op_add_id) \
  X(op_add1) \
  X(op_acc) \
  X(op_sub) \
  X(op_mul) \
  X(op_div) \
  X(op_sqr) \
  X(op_sqrt) \
  X(op_log) \
  X(op_sin) \
  X(op_cos) \
  X(op_sum) \
  X(op_sum_rows) \
  X(op_cumsum) \
  X(op_mean) \
  X(op_argmax) \
  X(op_count_equal) \
  X(op_repeat) \
  X(op_repeat_back) \
  X(op_concat) \
  X(op_silu_back) \
  X(op_norm) \
  X(op_rms_norm) \
  X(op_rms_norm_back) \
  X(op_group_norm) \
  X(op_l2_norm) \
  X(op_mul_mat) \
  X(op_mul_mat_argmax) \
  X(op_mul_mat_id) \
  X(op_out_prod) \
  X(op_scale) \
  X(op_set) \
  X(op_cpy) \
  X(op_cont) \
  X(op_reshape) \
  X(op_view) \
  X(op_permute) \
  X(op_transpose) \
  X(op_get_rows) \
  X(op_get_rows_back) \
  X(op_set_rows) \
  X(op_diag) \
  X(op_diag_mask_inf) \
  X(op_diag_mask_zero) \
  X(op_soft_max) \
  X(op_soft_max_back) \
  X(op_rope) \
  X(op_rope_back) \
  X(op_clamp) \
  X(op_conv_transpose_1d) \
  X(op_im2col) \
  X(op_im2col_back) \
  X(op_im2col_3d) \
  X(op_conv_2d) \
  X(op_conv_3d) \
  X(op_conv_2d_dw) \
  X(op_conv_transpose_2d) \
  X(op_pool_1d) \
  X(op_pool_2d) \
  X(op_pool_2d_back) \
  X(op_upscale) \
  X(op_pad) \
  X(op_pad_reflect_1d) \
  X(op_roll) \
  X(op_arange) \
  X(op_timestep_embedding) \
  X(op_argsort) \
  X(op_top_k) \
  X(op_leaky_relu) \
  X(op_tri) \
  X(op_fill) \
  X(op_flash_attn_ext) \
  X(op_flash_attn_back) \
  X(op_ssm_conv) \
  X(op_ssm_scan) \
  X(op_win_part) \
  X(op_win_unpart) \
  X(op_get_rel_pos) \
  X(op_add_rel_pos) \
  X(op_rwkv_wkv6) \
  X(op_gated_linear_attn) \
  X(op_rwkv_wkv7) \
  X(op_solve_tri) \
  X(op_unary) \
  X(op_map_custom1) \
  X(op_map_custom2) \
  X(op_map_custom3) \
  X(op_custom) \
  X(op_cross_entropy_loss) \
  X(op_cross_entropy_loss_back) \
  X(op_opt_step_adamw) \
  X(op_opt_step_sgd) \
  X(op_glu)

namespace emel::kernel::event {

enum class dtype : uint8_t;
enum class unary_subop : uint8_t;

#define EMEL_KERNEL_FORWARD_DECLARE_EVENT(op_name) \
  struct op_name;
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_FORWARD_DECLARE_EVENT)
#undef EMEL_KERNEL_FORWARD_DECLARE_EVENT

}  // namespace emel::kernel::event

namespace emel::kernel {

template <class event_type>
struct is_op_event : std::false_type {};

#define EMEL_KERNEL_MARK_OP_EVENT(op_name) \
  template <>                              \
  struct is_op_event<event::op_name> : std::true_type {};
EMEL_KERNEL_OP_EVENT_LIST(EMEL_KERNEL_MARK_OP_EVENT)
#undef EMEL_KERNEL_MARK_OP_EVENT

template <class event_type>
inline constexpr bool is_op_event_v = is_op_event<event_type>::value;

}  // namespace emel::kernel

namespace emel::kernel::detail {

inline constexpr uint8_t dtype_f32 = 0;
inline constexpr uint8_t dtype_f16 = 1;
inline constexpr uint8_t dtype_q4_0 = 2;
inline constexpr uint8_t dtype_q2_k = 10;
inline constexpr uint8_t dtype_q3_k = 11;
inline constexpr uint8_t dtype_q6_k = 14;
inline constexpr uint8_t dtype_q8_k = 15;
inline constexpr uint8_t dtype_q6_k_x8 = 36;
inline constexpr uint8_t dtype_q6_k_x8_q8_prepared = 37;
inline constexpr uint8_t dtype_q6_k_x8_q8_argmax_prepared = 38;
inline constexpr uint64_t flash_attn_workspace_token_capacity = 4096u;

struct flash_attn_workspace {
  alignas(64) std::array<float, flash_attn_workspace_token_capacity> score_buffer = {};
  alignas(64) std::array<float, flash_attn_workspace_token_capacity> value_buffer = {};
  alignas(64) std::array<float, flash_attn_workspace_token_capacity> accum_buffer = {};
  alignas(64) std::array<uint16_t, flash_attn_workspace_token_capacity> q_buffer_f16 = {};
  alignas(64) std::array<uint16_t, flash_attn_workspace_token_capacity> accum_buffer_f16 = {};
  uint64_t prepared_tokens = 0;
  uint64_t reuse_count = 0;
};

namespace quant {

constexpr uint64_t QK_K = 256u;
constexpr uint64_t MAX_Q8_K_BLOCKS = 128u;
constexpr uint64_t Q6_K_X8_ROWS = 8u;

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

struct block_q6_k {
  std::array<uint8_t, QK_K / 2> ql = {};
  std::array<uint8_t, QK_K / 4> qh = {};
  std::array<int8_t, QK_K / 16> scales = {};
  uint16_t d = 0;
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

static_assert(sizeof(block_q2_k) == 2 * sizeof(uint16_t) + (QK_K / 16) + (QK_K / 4));
static_assert(sizeof(block_q3_k) == sizeof(uint16_t) + (QK_K / 4) + (QK_K / 8) + 12);
static_assert(sizeof(block_q6_k) == sizeof(uint16_t) + (QK_K / 16) + (3 * QK_K / 4));
static_assert(
    sizeof(block_q6_kx8) ==
    sizeof(uint16_t) * Q6_K_X8_ROWS + (QK_K / 16) * Q6_K_X8_ROWS + (3 * QK_K / 4) * Q6_K_X8_ROWS);
static_assert(
    sizeof(block_q6_kx8_q8_prepared) ==
    sizeof(uint16_t) * Q6_K_X8_ROWS + sizeof(int8_t) * (QK_K / 16) * Q6_K_X8_ROWS +
        sizeof(int8_t) * QK_K * Q6_K_X8_ROWS);
static_assert(
    sizeof(block_q6_kx8_q8_argmax_prepared) ==
    sizeof(float) * Q6_K_X8_ROWS + sizeof(int8_t) * (QK_K / 16) * Q6_K_X8_ROWS +
        sizeof(int8_t) * QK_K * Q6_K_X8_ROWS);
static_assert(sizeof(block_q8_k) == sizeof(float) + QK_K + (QK_K / 16) * sizeof(int16_t));

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
  const float normalized = fp32_from_bits((doubled >> 4u) + exp_offset) * exp_scale;

  const uint32_t magic_mask = 126u << 23u;
  const float magic_bias = 0.5f;
  const float denormalized = fp32_from_bits((doubled >> 17u) | magic_mask) - magic_bias;

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
  return static_cast<uint16_t>(
      (sign >> 16u) | (doubled > 0xFF000000u ? 0x7E00u : nonsign));
}

inline void dequantize_row_q2_k(const block_q2_k * x, float * y, const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK_K);

  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d);
    const float min = fp16_to_fp32(x[i].dmin);
    const uint8_t * q = x[i].qs.data();

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

inline void dequantize_row_q3_k(const block_q3_k * x, float * y, const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK_K);
  constexpr uint32_t kmask1 = 0x03030303u;
  constexpr uint32_t kmask2 = 0x0f0f0f0fu;

  uint32_t aux[4] = {};
  auto * scales = reinterpret_cast<int8_t *>(aux);

  for (int64_t i = 0; i < nb; ++i) {
    const float d_all = fp16_to_fp32(x[i].d);
    const uint8_t * q = x[i].qs.data();
    const uint8_t * hm = x[i].hmask.data();
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
          const int8_t q0 =
              static_cast<int8_t>((q[l + 0] >> shift) & 0x03u) - ((hm[l + 0] & m) ? 0 : 4);
          *y++ = dl0 * static_cast<float>(q0);
        }

        const float dl1 = d_all * static_cast<float>(scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          const int8_t q1 =
              static_cast<int8_t>((q[l + 16] >> shift) & 0x03u) - ((hm[l + 16] & m) ? 0 : 4);
          *y++ = dl1 * static_cast<float>(q1);
        }

        shift += 2;
        m = static_cast<uint8_t>(m << 1u);
      }
      q += 32;
    }
  }
}

inline void dequantize_row_q6_k(const block_q6_k * x, float * y, const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK_K);

  for (int64_t i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d);
    const uint8_t * ql = x[i].ql.data();
    const uint8_t * qh = x[i].qh.data();
    const int8_t * sc = x[i].scales.data();

    for (int n = 0; n < static_cast<int>(QK_K); n += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int8_t q1 =
            static_cast<int8_t>((ql[l + 0] & 0x0fu) | (((qh[l] >> 0u) & 0x03u) << 4u)) - 32;
        const int8_t q2 =
            static_cast<int8_t>((ql[l + 32] & 0x0fu) | (((qh[l] >> 2u) & 0x03u) << 4u)) - 32;
        const int8_t q3 = static_cast<int8_t>(
                              ((ql[l + 0] >> 4u) & 0x0fu) | (((qh[l] >> 4u) & 0x03u) << 4u)) -
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

inline int nearest_int(const float value) noexcept {
  float biased = value + 12582912.0f;
  int bits = 0;
  std::memcpy(&bits, &biased, sizeof(bits));
  return (bits & 0x007fffff) - 0x00400000;
}

inline void quantize_row_q8_k_strided(const float * x,
                                      const uint64_t stride,
                                      block_q8_k * y,
                                      const int64_t k) noexcept {
  const int64_t nb = k / static_cast<int64_t>(QK_K);

  for (int64_t i = 0; i < nb; ++i) {
    float max = 0.0f;
    float amax = 0.0f;
    const float * block = x + i * static_cast<int64_t>(QK_K) * static_cast<int64_t>(stride);
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

inline constexpr uint64_t packed_q6_k_x8_group_count(const uint64_t rows) noexcept {
  return (rows + Q6_K_X8_ROWS - 1u) / Q6_K_X8_ROWS;
}

inline size_t packed_q6_k_x8_group_storage_bytes(const uint64_t cols) noexcept {
  if ((cols % QK_K) != 0u) {
    return 0u;
  }
  const uint64_t block_count = cols / QK_K;
  return static_cast<size_t>(block_count) * sizeof(block_q6_kx8);
}

inline size_t prepared_q6_k_x8_q8_group_storage_bytes(const uint64_t cols) noexcept {
  if ((cols % QK_K) != 0u) {
    return 0u;
  }
  const uint64_t block_count = cols / QK_K;
  return static_cast<size_t>(block_count) * sizeof(block_q6_kx8_q8_prepared);
}

inline size_t argmax_prepared_q6_k_x8_q8_group_storage_bytes(const uint64_t cols) noexcept {
  if ((cols % QK_K) != 0u) {
    return 0u;
  }
  const uint64_t block_count = cols / QK_K;
  return static_cast<size_t>(block_count) * sizeof(block_q6_kx8_q8_argmax_prepared);
}

inline block_q6_kx8 make_block_q6_k_x8(const block_q6_k * rows) noexcept {
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
                rows[src_row].ql.data() + src_offset,
                interleave_block_bytes);
  }

  for (uint64_t i = 0; i < end_high_bytes; ++i) {
    const uint64_t src_row = i % Q6_K_X8_ROWS;
    const uint64_t src_offset = (i / Q6_K_X8_ROWS) * interleave_block_bytes;
    const uint64_t dst_offset = i * interleave_block_bytes;
    std::memcpy(out.qh.data() + dst_offset,
                rows[src_row].qh.data() + src_offset,
                interleave_block_bytes);
  }

  for (uint64_t row = 0; row < Q6_K_X8_ROWS; ++row) {
    for (uint64_t scale = 0; scale < scale_count; ++scale) {
      out.scales[scale * Q6_K_X8_ROWS + row] = rows[row].scales[scale];
    }
  }

  return out;
}

inline block_q6_kx8_q8_prepared make_block_q6_k_x8_q8_prepared(const block_q6_k * rows) noexcept {
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
        decoded[half_value_base + lane + 0u] =
            static_cast<int8_t>(
                static_cast<int32_t>((ql_low & 0x0fu) | (((qh_byte >> 0u) & 0x03u) << 4u)) -
                32);
        decoded[half_value_base + lane + 32u] =
            static_cast<int8_t>(
                static_cast<int32_t>((ql_high & 0x0fu) | (((qh_byte >> 2u) & 0x03u) << 4u)) -
                32);
        decoded[half_value_base + lane + 64u] =
            static_cast<int8_t>(
                static_cast<int32_t>(((ql_low >> 4u) & 0x0fu) |
                                     (((qh_byte >> 4u) & 0x03u) << 4u)) -
                32);
        decoded[half_value_base + lane + 96u] =
            static_cast<int8_t>(
                static_cast<int32_t>(((ql_high >> 4u) & 0x0fu) |
                                     (((qh_byte >> 6u) & 0x03u) << 4u)) -
                32);
      }
    }

    for (uint64_t scale = 0; scale < (QK_K / 16u); ++scale) {
      const size_t scale_offset = static_cast<size_t>(scale) * Q6_K_X8_ROWS + row;
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
        const size_t qs_offset = (static_cast<size_t>(scale) * Q6_K_X8_ROWS + row) * 16u + lane;
        out.qs[qs_offset] = decoded[scale * 16u + lane];
      }
#endif
    }
  }

  return out;
}

inline block_q6_kx8_q8_argmax_prepared make_block_q6_k_x8_q8_argmax_prepared(
    const block_q6_k * rows) noexcept {
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
        decoded[half_value_base + lane + 0u] =
            static_cast<int8_t>(
                static_cast<int32_t>((ql_low & 0x0fu) | (((qh_byte >> 0u) & 0x03u) << 4u)) -
                32);
        decoded[half_value_base + lane + 32u] =
            static_cast<int8_t>(
                static_cast<int32_t>((ql_high & 0x0fu) | (((qh_byte >> 2u) & 0x03u) << 4u)) -
                32);
        decoded[half_value_base + lane + 64u] =
            static_cast<int8_t>(
                static_cast<int32_t>(((ql_low >> 4u) & 0x0fu) |
                                     (((qh_byte >> 4u) & 0x03u) << 4u)) -
                32);
        decoded[half_value_base + lane + 96u] =
            static_cast<int8_t>(
                static_cast<int32_t>(((ql_high >> 4u) & 0x0fu) |
                                     (((qh_byte >> 6u) & 0x03u) << 4u)) -
                32);
      }
    }

    for (uint64_t scale = 0; scale < (QK_K / 16u); ++scale) {
      out.scales[static_cast<size_t>(scale) * Q6_K_X8_ROWS + row] = rows[row].scales[scale];
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

inline bool pack_q6_k_rows_x8(const block_q6_k * src,
                              const uint64_t rows,
                              const uint64_t cols,
                              void * dst) noexcept {
  if (src == nullptr || dst == nullptr) {
    return false;
  }
  if ((cols % QK_K) != 0u) {
    return false;
  }

  const uint64_t block_count = cols / QK_K;
  const uint64_t group_count = packed_q6_k_x8_group_count(rows);
  auto * dst_blocks = static_cast<block_q6_kx8 *>(dst);
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
      dst_blocks[group * block_count + block] = make_block_q6_k_x8(group_rows.data());
    }
  }
  return true;
}

inline bool pack_q6_k_rows_x8_q8_prepared(const block_q6_k * src,
                                          const uint64_t rows,
                                          const uint64_t cols,
                                          void * dst) noexcept {
  if (src == nullptr || dst == nullptr) {
    return false;
  }
  if ((cols % QK_K) != 0u) {
    return false;
  }

  const uint64_t block_count = cols / QK_K;
  const uint64_t group_count = packed_q6_k_x8_group_count(rows);
  auto * dst_blocks = static_cast<block_q6_kx8_q8_prepared *>(dst);
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
      dst_blocks[group * block_count + block] = make_block_q6_k_x8_q8_prepared(group_rows.data());
    }
  }
  return true;
}

inline bool pack_q6_k_rows_x8_q8_argmax_prepared(const block_q6_k * src,
                                                 const uint64_t rows,
                                                 const uint64_t cols,
                                                 void * dst) noexcept {
  if (src == nullptr || dst == nullptr) {
    return false;
  }
  if ((cols % QK_K) != 0u) {
    return false;
  }

  const uint64_t block_count = cols / QK_K;
  const uint64_t group_count = packed_q6_k_x8_group_count(rows);
  auto * dst_blocks = static_cast<block_q6_kx8_q8_argmax_prepared *>(dst);
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

}  // namespace quant

inline uint64_t select_u64(const bool choose_true,
                           const uint64_t true_value,
                           const uint64_t false_value) noexcept {
  const uint64_t mask = static_cast<uint64_t>(0) - static_cast<uint64_t>(choose_true);
  return (false_value & ~mask) | (true_value & mask);
}

inline bool select_bool(const bool choose_true,
                        const bool true_value,
                        const bool false_value) noexcept {
  const std::array<bool, 2> values{false_value, true_value};
  return values[static_cast<size_t>(choose_true)];
}

template <class dtype_type>
inline uint8_t dtype_code(const dtype_type type) noexcept {
  return static_cast<uint8_t>(type);
}

inline bool is_quantized_k_dtype(const uint8_t code) noexcept {
  return code == dtype_q2_k || code == dtype_q3_k || code == dtype_q6_k;
}

inline bool is_rhs_q8_k_dtype(const uint8_t code) noexcept {
  return code == dtype_q8_k;
}

inline bool is_packed_q6_vector_dtype(const uint8_t code) noexcept {
  return code == dtype_q6_k_x8;
}

inline bool is_prepared_q6_vector_q8_rhs_dtype(const uint8_t code) noexcept {
  return code == dtype_q6_k_x8_q8_prepared;
}

inline bool is_argmax_prepared_q6_vector_q8_rhs_dtype(const uint8_t code) noexcept {
  return code == dtype_q6_k_x8_q8_argmax_prepared;
}

inline bool is_supported_dtype(const uint8_t code) noexcept {
  return code == dtype_f32 || code == dtype_f16 || is_quantized_k_dtype(code) ||
      is_rhs_q8_k_dtype(code) || is_packed_q6_vector_dtype(code) ||
      is_prepared_q6_vector_q8_rhs_dtype(code) ||
      is_argmax_prepared_q6_vector_q8_rhs_dtype(code);
}

inline size_t dtype_size_bytes(const uint8_t code) noexcept {
  const std::array<size_t, 7> size_candidates = {
      0u,
      4u,
      2u,
      sizeof(quant::block_q2_k) / quant::QK_K,
      sizeof(quant::block_q3_k) / quant::QK_K,
      sizeof(quant::block_q6_k) / quant::QK_K,
      sizeof(quant::block_q8_k) / quant::QK_K,
  };
  if (code == dtype_f32) {
    return size_candidates[1];
  }
  if (code == dtype_f16) {
    return size_candidates[2];
  }
  if (code == dtype_q2_k) {
    return size_candidates[3];
  }
  if (code == dtype_q3_k) {
    return size_candidates[4];
  }
  if (code == dtype_q6_k) {
    return size_candidates[5];
  }
  if (code == dtype_q8_k) {
    return size_candidates[6];
  }
  return size_candidates[0];
}

inline size_t quantized_row_storage_bytes(const uint8_t code, const uint64_t cols) noexcept {
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
  if (code == dtype_q6_k) {
    return static_cast<size_t>(block_count) * sizeof(quant::block_q6_k);
  }
  if (code == dtype_q8_k) {
    return static_cast<size_t>(block_count) * sizeof(quant::block_q8_k);
  }
  if (code == dtype_q6_k_x8_q8_prepared) {
    return quant::prepared_q6_k_x8_q8_group_storage_bytes(cols);
  }
  if (code == dtype_q6_k_x8_q8_argmax_prepared) {
    return quant::argmax_prepared_q6_k_x8_q8_group_storage_bytes(cols);
  }
  return 0u;
}

template <class tensor_type>
inline uint64_t tensor_element_count(const tensor_type & tensor) noexcept {
  uint64_t count = 1;
  for (size_t i = 0; i < 4; ++i) {
    count *= tensor.ne[i];
  }
  return count;
}

template <class tensor_type>
inline uint64_t tensor_stride_bytes(const tensor_type & tensor, const size_t dim) noexcept {
  uint64_t stride = dtype_size_bytes(dtype_code(tensor.type));
  for (size_t i = 0; i < dim; ++i) {
    stride *= tensor.ne[i];
  }
  const std::array<uint64_t, 2> candidates{stride, tensor.nb[dim]};
  return candidates[static_cast<size_t>(tensor.nb[0] != 0)];
}

template <class tensor_type>
inline bool has_valid_tensor_layout(const tensor_type & tensor) noexcept {
  const uint64_t elem_size = dtype_size_bytes(dtype_code(tensor.type));
  const bool elem_valid = elem_size != 0u;
  const bool explicit_stride = tensor.nb[0] != 0u;
  const bool aligned_stride =
      explicit_stride && tensor.nb[0] >= elem_size && (tensor.nb[0] % elem_size) == 0u;

  bool dims_valid = true;
  for (size_t i = 0; i < 4; ++i) {
    const bool invalid_dim = tensor.ne[i] > 1 && tensor.nb[i] == 0;
    dims_valid = dims_valid && !invalid_dim;
  }

  return elem_valid && (!explicit_stride || (aligned_stride && dims_valid));
}

template <class tensor_type>
inline bool is_dense_contiguous(const tensor_type & tensor) noexcept {
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
inline size_t tensor_offset_bytes(const tensor_type & tensor, const uint64_t idx) noexcept {
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
inline size_t tensor_offset_bytes(const tensor_type & tensor,
                                  const uint64_t i0,
                                  const uint64_t i1,
                                  const uint64_t i2 = 0,
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
inline bool has_required_src0(const request_type & request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_mul_mat> ||
                std::is_same_v<request_type, event::op_mul_mat_argmax>) {
    const uint8_t src0_type = dtype_code(request.src0.type);
    if (is_prepared_q6_vector_q8_rhs_dtype(src0_type) ||
        is_argmax_prepared_q6_vector_q8_rhs_dtype(src0_type)) {
      const uint64_t cols = request.src0.ne[0];
      const uint64_t rows = request.src0.ne[1];
      const uint64_t group_count = quant::packed_q6_k_x8_group_count(rows);
      const size_t group_bytes = is_prepared_q6_vector_q8_rhs_dtype(src0_type)
          ? quant::prepared_q6_k_x8_q8_group_storage_bytes(cols)
          : quant::argmax_prepared_q6_k_x8_q8_group_storage_bytes(cols);
      return request.src0.data != nullptr &&
             cols != 0u &&
             rows != 0u &&
             group_bytes != 0u &&
             request.src0.ne[2] == 1u &&
             request.src0.ne[3] == 1u &&
             request.src0.nb[0] == 1u &&
             request.src0.nb[1] == group_bytes &&
             request.src0.nb[2] == group_bytes * group_count &&
             request.src0.nb[3] == request.src0.nb[2];
    }
    if (is_packed_q6_vector_dtype(src0_type)) {
      const uint64_t cols = request.src0.ne[0];
      const uint64_t rows = request.src0.ne[1];
      const uint64_t group_count = quant::packed_q6_k_x8_group_count(rows);
      const size_t group_bytes = quant::packed_q6_k_x8_group_storage_bytes(cols);
      return request.src0.data != nullptr &&
             cols != 0u &&
             rows != 0u &&
             group_bytes != 0u &&
             request.src0.ne[2] == 1u &&
             request.src0.ne[3] == 1u &&
             request.src0.nb[0] == 1u &&
             request.src0.nb[1] == group_bytes &&
             request.src0.nb[2] == group_bytes * group_count &&
             request.src0.nb[3] == request.src0.nb[2];
    }
    if (is_quantized_k_dtype(src0_type)) {
      const uint64_t cols = request.src0.ne[0];
      const uint64_t rows = request.src0.ne[1];
      const size_t row_bytes = quantized_row_storage_bytes(src0_type, cols);
      return request.src0.data != nullptr &&
             row_bytes != 0u &&
             rows != 0u &&
             request.src0.ne[2] == 1u &&
             request.src0.ne[3] == 1u &&
             request.src0.nb[0] == 1u &&
             request.src0.nb[1] == row_bytes &&
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
inline bool has_required_src1(const request_type & request) noexcept {
  if constexpr (!requires_src1_v<request_type>) {
    return true;
  }
  return request.src1.data != nullptr &&
         is_supported_dtype(dtype_code(request.src1.type)) &&
         has_valid_tensor_layout(request.src1) &&
         tensor_element_count(request.src1) > 0;
}

template <class request_type>
inline bool has_required_dst(const request_type & request) noexcept {
  return request.dst.data != nullptr &&
         is_supported_dtype(dtype_code(request.dst.type)) &&
         has_valid_tensor_layout(request.dst) &&
         tensor_element_count(request.dst) > 0;
}

template <class request_type>
inline bool validate_dispatch_request(const request_type & request) noexcept {
  const bool has_required_buffers =
      has_required_src0(request) && has_required_src1(request) && has_required_dst(request);
  const bool has_valid_threading = request.ith == 0 && request.nth == 1;
  const bool has_valid_params = request.op_params_size <= request.op_params.size();
  return has_required_buffers && has_valid_threading && has_valid_params;
}

template <class tensor_type>
inline float read_f32(const tensor_type & tensor, const uint64_t idx) noexcept {
  const bool dense = is_dense_contiguous(tensor);
  const float * data = static_cast<const float *>(tensor.data);
  const char * base = static_cast<const char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, idx);
  const char *dense_src = reinterpret_cast<const char *>(data + idx);
  const char *sparse_src = base + offset;
  const std::array<const char *, 2> srcs{sparse_src, dense_src};
  float out = 0.0f;
  std::memcpy(&out, srcs[static_cast<size_t>(dense)], sizeof(out));
  return out;
}

template <class tensor_type>
inline void write_f32(const tensor_type & tensor, const uint64_t idx, const float value) noexcept {
  const bool dense = is_dense_contiguous(tensor);
  float * data = static_cast<float *>(tensor.data);
  char * base = static_cast<char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, idx);
  char *dense_dst = reinterpret_cast<char *>(data + idx);
  char *sparse_dst = base + offset;
  const std::array<char *, 2> dsts{sparse_dst, dense_dst};
  std::memcpy(dsts[static_cast<size_t>(dense)], &value, sizeof(value));
}

template <class tensor_type>
inline float read_f32_at(const tensor_type & tensor, const uint64_t i0, const uint64_t i1,
                         const uint64_t i2 = 0, const uint64_t i3 = 0) noexcept {
  float out = 0.0f;
  const char * base = static_cast<const char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, i0, i1, i2, i3);
  std::memcpy(&out, base + offset, sizeof(out));
  return out;
}

template <class tensor_type>
inline void write_f32_at(const tensor_type & tensor, const uint64_t i0, const uint64_t i1,
                         const float value, const uint64_t i2 = 0,
                         const uint64_t i3 = 0) noexcept {
  char * base = static_cast<char *>(tensor.data);
  const size_t offset = tensor_offset_bytes(tensor, i0, i1, i2, i3);
  std::memcpy(base + offset, &value, sizeof(value));
}

template <class tensor_type>
inline const void * tensor_row_ptr_void(const tensor_type & tensor,
                                        const uint64_t row1,
                                        const uint64_t row2) noexcept {
  const auto * base = static_cast<const char *>(tensor.data);
  return base + row1 * tensor.nb[1] + row2 * tensor.nb[2];
}

template <class elem_type, class tensor_type>
inline const elem_type * tensor_row_ptr_as(const tensor_type & tensor,
                                           const uint64_t row1,
                                           const uint64_t row2) noexcept {
  return reinterpret_cast<const elem_type *>(tensor_row_ptr_void(tensor, row1, row2));
}

template <class tensor_type>
inline const float * tensor_row_ptr(const tensor_type & tensor,
                                    const uint64_t row1,
                                    const uint64_t row2) noexcept {
  return tensor_row_ptr_as<float>(tensor, row1, row2);
}

template <class tensor_type>
inline void * tensor_row_ptr_mut_void(const tensor_type & tensor,
                                      const uint64_t row1,
                                      const uint64_t row2) noexcept {
  auto * base = static_cast<char *>(tensor.data);
  return base + row1 * tensor.nb[1] + row2 * tensor.nb[2];
}

template <class elem_type, class tensor_type>
inline elem_type * tensor_row_ptr_mut_as(const tensor_type & tensor,
                                         const uint64_t row1,
                                         const uint64_t row2) noexcept {
  return reinterpret_cast<elem_type *>(tensor_row_ptr_mut_void(tensor, row1, row2));
}

template <class tensor_type>
inline float * tensor_row_ptr_mut(const tensor_type & tensor,
                                  const uint64_t row1,
                                  const uint64_t row2) noexcept {
  return tensor_row_ptr_mut_as<float>(tensor, row1, row2);
}

#if defined(__ARM_NEON) && defined(__aarch64__)
inline float32x4_t expf4_ggml(float32x4_t x) noexcept {
  const float32x4_t r = vdupq_n_f32(0x1.8p23f);
  const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
  const float32x4_t n = vsubq_f32(z, r);
  const float32x4_t b = vfmsq_f32(
      vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
  const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
  const float32x4_t k =
      vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0f))));
  const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126.0f));
  const float32x4_t u = vmulq_f32(b, b);
  const float32x4_t j = vfmaq_f32(
      vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
      vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b),
                u),
      u);
  if (!vpaddd_u64(vreinterpretq_u64_u32(c))) {
    return vfmaq_f32(k, j, k);
  }
  const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
  const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
  const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
  return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192.0f)),
                   vmulq_f32(s1, s1),
                   vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
}
#endif

inline double exp_and_sum_ggml_f32(const float * src,
                                   float * dst,
                                   const uint64_t count,
                                   const float max_value) noexcept {
#if defined(__ARM_NEON) && defined(__aarch64__)
  uint64_t idx = 0u;
  double sum = 0.0;
  const float32x4_t max_vec = vdupq_n_f32(max_value);
  for (; idx + 4u <= count; idx += 4u) {
    const float32x4_t values = expf4_ggml(vsubq_f32(vld1q_f32(src + idx), max_vec));
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

inline float dot_product_ggml_f16_scores(const float * lhs,
                                         const float * rhs,
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

    sum[0] = vfmaq_f16(sum[0],
                       vld1q_f16(reinterpret_cast<const __fp16 *>(lhs_f16 + 0u)),
                       vld1q_f16(reinterpret_cast<const __fp16 *>(rhs_f16 + 0u)));
    sum[1] = vfmaq_f16(sum[1],
                       vld1q_f16(reinterpret_cast<const __fp16 *>(lhs_f16 + 8u)),
                       vld1q_f16(reinterpret_cast<const __fp16 *>(rhs_f16 + 8u)));
    sum[2] = vfmaq_f16(sum[2],
                       vld1q_f16(reinterpret_cast<const __fp16 *>(lhs_f16 + 16u)),
                       vld1q_f16(reinterpret_cast<const __fp16 *>(rhs_f16 + 16u)));
    sum[3] = vfmaq_f16(sum[3],
                       vld1q_f16(reinterpret_cast<const __fp16 *>(lhs_f16 + 24u)),
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
    sumf += static_cast<double>(quant::fp16_to_fp32(quant::fp32_to_fp16(lhs[idx]))) *
            static_cast<double>(quant::fp16_to_fp32(quant::fp32_to_fp16(rhs[idx])));
  }
  return static_cast<float>(sumf);
#endif
#endif

  double scalar_sum = 0.0;
  for (uint64_t idx = 0u; idx < count; ++idx) {
    scalar_sum += static_cast<double>(quant::fp16_to_fp32(quant::fp32_to_fp16(lhs[idx]))) *
                  static_cast<double>(quant::fp16_to_fp32(quant::fp32_to_fp16(rhs[idx])));
  }
  return static_cast<float>(scalar_sum);
}

inline float dot_product_f16_f16_scores(const uint16_t * lhs,
                                        const uint16_t * rhs,
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
  const auto * lhs_f16 = reinterpret_cast<const __fp16 *>(lhs);
  const auto * rhs_f16 = reinterpret_cast<const __fp16 *>(rhs);
  for (; idx + 32u <= count; idx += 32u) {
    sum[0] = vfmaq_f16(sum[0], vld1q_f16(lhs_f16 + idx + 0u), vld1q_f16(rhs_f16 + idx + 0u));
    sum[1] = vfmaq_f16(sum[1], vld1q_f16(lhs_f16 + idx + 8u), vld1q_f16(rhs_f16 + idx + 8u));
    sum[2] = vfmaq_f16(sum[2], vld1q_f16(lhs_f16 + idx + 16u), vld1q_f16(rhs_f16 + idx + 16u));
    sum[3] = vfmaq_f16(sum[3], vld1q_f16(lhs_f16 + idx + 24u), vld1q_f16(rhs_f16 + idx + 24u));
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

inline float dot_product_f32_f16_scores(const float * lhs,
                                        const uint16_t * rhs,
                                        const uint64_t count) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  float32x4_t sum_lo = vdupq_n_f32(0.0f);
  float32x4_t sum_hi = vdupq_n_f32(0.0f);
  uint64_t idx = 0u;
  for (; idx + 8u <= count; idx += 8u) {
    const float32x4_t lhs0 = vld1q_f32(lhs + idx + 0u);
    const float32x4_t lhs1 = vld1q_f32(lhs + idx + 4u);
    const float16x8_t lhs_f16 = vcombine_f16(vcvt_f16_f32(lhs0), vcvt_f16_f32(lhs1));
    const float16x8_t rhs_f16 = vreinterpretq_f16_u16(vld1q_u16(rhs + idx));
    sum_lo = vfmlalq_low_f16(sum_lo, lhs_f16, rhs_f16);
    sum_hi = vfmlalq_high_f16(sum_hi, lhs_f16, rhs_f16);
  }

  double sumf = static_cast<double>(vaddvq_f32(vaddq_f32(sum_lo, sum_hi)));
  for (; idx < count; ++idx) {
    sumf += static_cast<double>(quant::fp16_to_fp32(quant::fp32_to_fp16(lhs[idx]))) *
            static_cast<double>(quant::fp16_to_fp32(rhs[idx]));
  }
  return static_cast<float>(sumf);
#endif
#endif

  double scalar_sum = 0.0;
  for (uint64_t idx = 0u; idx < count; ++idx) {
    scalar_sum += static_cast<double>(quant::fp16_to_fp32(quant::fp32_to_fp16(lhs[idx]))) *
                  static_cast<double>(quant::fp16_to_fp32(rhs[idx]));
  }
  return static_cast<float>(scalar_sum);
}

inline float dot_product_ggml_f16_scores(const float * lhs,
                                         const uint16_t * rhs,
                                         const uint64_t count) noexcept {
  return dot_product_f32_f16_scores(lhs, rhs, count);
}

inline float dot_product_f32(const float * lhs,
                             const float * rhs,
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
    sum[0] = vfmaq_f32(sum[0], vld1q_f32(lhs + idx + 0u), vld1q_f32(rhs + idx + 0u));
    sum[1] = vfmaq_f32(sum[1], vld1q_f32(lhs + idx + 4u), vld1q_f32(rhs + idx + 4u));
    sum[2] = vfmaq_f32(sum[2], vld1q_f32(lhs + idx + 8u), vld1q_f32(rhs + idx + 8u));
    sum[3] = vfmaq_f32(sum[3], vld1q_f32(lhs + idx + 12u), vld1q_f32(rhs + idx + 12u));
  }

  float scalar_sum =
      vaddvq_f32(vaddq_f32(vaddq_f32(sum[0], sum[1]), vaddq_f32(sum[2], sum[3])));
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

inline void scale_buffer_f32(float * data,
                             const uint64_t count,
                             const float scale) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const float32x4_t scale_vec = vdupq_n_f32(scale);
  uint64_t idx = 0u;
  for (; idx + 16u <= count; idx += 16u) {
    vst1q_f32(data + idx + 0u, vmulq_f32(vld1q_f32(data + idx + 0u), scale_vec));
    vst1q_f32(data + idx + 4u, vmulq_f32(vld1q_f32(data + idx + 4u), scale_vec));
    vst1q_f32(data + idx + 8u, vmulq_f32(vld1q_f32(data + idx + 8u), scale_vec));
    vst1q_f32(data + idx + 12u, vmulq_f32(vld1q_f32(data + idx + 12u), scale_vec));
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

inline void mad_buffer_f32(float * acc,
                           const float * value,
                           const uint64_t count,
                           const float weight) noexcept {
#if defined(__aarch64__) || defined(__ARM_NEON)
  const float32x4_t weight_vec = vdupq_n_f32(weight);
  uint64_t idx = 0u;
  for (; idx + 16u <= count; idx += 16u) {
    vst1q_f32(acc + idx + 0u,
              vfmaq_f32(vld1q_f32(acc + idx + 0u), vld1q_f32(value + idx + 0u), weight_vec));
    vst1q_f32(acc + idx + 4u,
              vfmaq_f32(vld1q_f32(acc + idx + 4u), vld1q_f32(value + idx + 4u), weight_vec));
    vst1q_f32(acc + idx + 8u,
              vfmaq_f32(vld1q_f32(acc + idx + 8u), vld1q_f32(value + idx + 8u), weight_vec));
    vst1q_f32(acc + idx + 12u,
              vfmaq_f32(vld1q_f32(acc + idx + 12u), vld1q_f32(value + idx + 12u), weight_vec));
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
inline bool run_copy(const request_type & request) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  const bool shape_ok = count == tensor_element_count(request.src0);
  const bool dense = shape_ok && is_dense_contiguous(request.src0) &&
                     is_dense_contiguous(request.dst);
  const uint64_t dense_count = count * static_cast<uint64_t>(dense);
  const uint64_t sparse_count = count * static_cast<uint64_t>(shape_ok && !dense);

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
inline bool run_binary(const request_type & request, op_type op) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  const bool incompatible_shape =
      count != tensor_element_count(request.src0) || count != tensor_element_count(request.src1);
  const bool compatible = !incompatible_shape;

  const bool dense = compatible &&
      is_dense_contiguous(request.src0) &&
      is_dense_contiguous(request.src1) &&
      is_dense_contiguous(request.dst);
  const uint64_t dense_count = count * static_cast<uint64_t>(dense);
  const uint64_t sparse_count = count * static_cast<uint64_t>(compatible && !dense);

  const float *lhs_dense = static_cast<const float *>(request.src0.data);
  const float *rhs_dense = static_cast<const float *>(request.src1.data);
  float *dst_dense = static_cast<float *>(request.dst.data);
  for (uint64_t i = 0; i < dense_count; ++i) {
    dst_dense[i] = op(lhs_dense[i], rhs_dense[i]);
  }

  for (uint64_t i = 0; i < sparse_count; ++i) {
    write_f32(request.dst, i, op(read_f32(request.src0, i), read_f32(request.src1, i)));
  }
  return compatible;
}

template <class request_type, class op_type>
inline bool run_unary(const request_type & request, op_type op) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  const bool shape_ok = count == tensor_element_count(request.src0);
  const bool dense = shape_ok && is_dense_contiguous(request.src0) &&
                     is_dense_contiguous(request.dst);
  const uint64_t dense_count = count * static_cast<uint64_t>(dense);
  const uint64_t sparse_count = count * static_cast<uint64_t>(shape_ok && !dense);

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

inline float dot_q2_k_q8_k_block_scalar(const quant::block_q2_k & lhs,
                                        const quant::block_q8_k & rhs) noexcept {
  const uint8_t * q2 = lhs.qs.data();
  const int8_t * q8 = rhs.qs.data();
  const uint8_t * scales = lhs.scales.data();

  int sum_mins = 0;
  for (uint64_t j = 0; j < (quant::QK_K / 16); ++j) {
    sum_mins += static_cast<int>(rhs.bsums[j]) * static_cast<int>(scales[j] >> 4u);
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
        local_sum += static_cast<int>(q8[l]) * static_cast<int>((q2[l] >> shift) & 0x03u);
      }
      sum += scale * local_sum;

      scale = static_cast<int>(scales[scale_index++] & 0x0fu);
      local_sum = 0;
      for (uint64_t l = 16; l < 32; ++l) {
        local_sum += static_cast<int>(q8[l]) * static_cast<int>((q2[l] >> shift) & 0x03u);
      }
      sum += scale * local_sum;

      shift += 2;
      q8 += 32;
    }
    q2 += 32;
  }

  return d_all * static_cast<float>(sum) - d_min * static_cast<float>(sum_mins);
}

inline float dot_q2_k_q8_k_row_scalar(const quant::block_q2_k * lhs,
                                      const quant::block_q8_k * rhs,
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
    const uint8_t * q2 = lhs[block].qs.data();
    const int8_t * q8 = rhs[block].qs.data();
    const uint8_t * scales_ptr = lhs[block].scales.data();

    const uint8x16_t mins_and_scales = vld1q_u8(scales_ptr);
    const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
    vst1q_u8(scales_buf, scales);

    const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
    const int16x8_t q8sums0 = vld1q_s16(rhs[block].bsums.data());
    const int16x8_t q8sums1 = vld1q_s16(rhs[block].bsums.data() + 8);
    const int16x8_t mins16_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
    const int16x8_t mins16_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));
    const int32x4_t s0 = vaddq_s32(
        vmull_s16(vget_low_s16(mins16_lo), vget_low_s16(q8sums0)),
        vmull_s16(vget_high_s16(mins16_lo), vget_high_s16(q8sums0)));
    const int32x4_t s1 = vaddq_s32(
        vmull_s16(vget_low_s16(mins16_hi), vget_low_s16(q8sums1)),
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
        q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits0, 2), m3));
        q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits1, 2), m3));
        isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[0], q8bytes.val[0])) *
            scales_buf[scale_index + 2];
        isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[1], q8bytes.val[1])) *
            scales_buf[scale_index + 3];
      }
      {
        const int8x16x2_t q8bytes = {{vld1q_s8(q8), vld1q_s8(q8 + 16)}};
        q8 += 32;
        q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits0, 4), m3));
        q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits1, 4), m3));
        isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[0], q8bytes.val[0])) *
            scales_buf[scale_index + 4];
        isum += vaddvq_s32(vdotq_s32(zero, q2bytes.val[1], q8bytes.val[1])) *
            scales_buf[scale_index + 5];
      }
      {
        const int8x16x2_t q8bytes = {{vld1q_s8(q8), vld1q_s8(q8 + 16)}};
        q8 += 32;
        q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits0, 6), m3));
        q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits1, 6), m3));
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

inline float dot_q3_k_q8_k_block_scalar(const quant::block_q3_k & lhs,
                                        const quant::block_q8_k & rhs) noexcept {
  constexpr uint32_t kmask1 = 0x03030303u;
  constexpr uint32_t kmask2 = 0x0f0f0f0fu;

  alignas(64) int8_t dequant[quant::QK_K] = {};
  int16_t products[8] = {};
  int32_t sums[8] = {};
  uint32_t scale_words[4] = {};
  auto * scales = reinterpret_cast<int8_t *>(scale_words);

  const uint8_t * q3 = lhs.qs.data();
  const uint8_t * hmask = lhs.hmask.data();
  const int8_t * q8 = rhs.qs.data();
  int8_t * out = dequant;
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
  scale_words[2] = ((scale_words[0] >> 4u) & kmask2) | (((tmp >> 4u) & kmask1) << 4u);
  scale_words[3] = ((scale_words[1] >> 4u) & kmask2) | (((tmp >> 6u) & kmask1) << 4u);
  scale_words[0] = (scale_words[0] & kmask2) | (((tmp >> 0u) & kmask1) << 4u);
  scale_words[1] = (scale_words[1] & kmask2) | (((tmp >> 2u) & kmask1) << 4u);

  const int8_t * a = dequant;
  for (uint64_t group = 0; group < (quant::QK_K / 16); ++group) {
    for (uint64_t l = 0; l < 8; ++l) {
      products[l] = static_cast<int16_t>(q8[l] * a[l]);
    }
    for (uint64_t l = 0; l < 8; ++l) {
      sums[l] += static_cast<int32_t>(scales[group] - 32) * static_cast<int32_t>(products[l]);
    }
    q8 += 8;
    a += 8;
    for (uint64_t l = 0; l < 8; ++l) {
      products[l] = static_cast<int16_t>(q8[l] * a[l]);
    }
    for (uint64_t l = 0; l < 8; ++l) {
      sums[l] += static_cast<int32_t>(scales[group] - 32) * static_cast<int32_t>(products[l]);
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

inline float dot_q3_k_q8_k_row_scalar(const quant::block_q3_k * lhs,
                                      const quant::block_q8_k * rhs,
                                      const uint64_t block_count) noexcept {
  constexpr uint32_t kmask1 = 0x03030303u;
  constexpr uint32_t kmask2 = 0x0f0f0f0fu;

  int8_t aux8[quant::QK_K] = {};
  int16_t aux16[8] = {};
  float sums[8] = {};
  int32_t aux32[8] = {};
  uint32_t auxs[4] = {};
  const int8_t * scales = reinterpret_cast<const int8_t *>(auxs);

  float sumf = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    const uint8_t * q3 = lhs[block].qs.data();
    const uint8_t * hm = lhs[block].hmask.data();
    const int8_t * q8 = rhs[block].qs.data();
    std::memset(aux32, 0, sizeof(aux32));
    int8_t * a = aux8;
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
        aux32[lane] += static_cast<int32_t>(scales[group] - 32) * static_cast<int32_t>(aux16[lane]);
      }
      q8 += 8;
      a += 8;
      for (uint64_t lane = 0; lane < 8; ++lane) {
        aux16[lane] = static_cast<int16_t>(q8[lane] * a[lane]);
      }
      for (uint64_t lane = 0; lane < 8; ++lane) {
        aux32[lane] += static_cast<int32_t>(scales[group] - 32) * static_cast<int32_t>(aux16[lane]);
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

inline float dot_q6_k_q8_k_block_scalar(const quant::block_q6_k & lhs,
                                        const quant::block_q8_k & rhs) noexcept {
  alignas(64) int8_t dequant[quant::QK_K] = {};
  int16_t products[8] = {};
  int32_t sums[8] = {};

  const uint8_t * ql = lhs.ql.data();
  const uint8_t * qh = lhs.qh.data();
  const int8_t * q8 = rhs.qs.data();
  int8_t * out = dequant;
  for (uint64_t block = 0; block < quant::QK_K; block += 128) {
    for (uint64_t l = 0; l < 32; ++l) {
      out[l + 0] = static_cast<int8_t>((ql[l + 0] & 0x0fu) | (((qh[l] >> 0u) & 0x03u) << 4u)) -
          32;
      out[l + 32] = static_cast<int8_t>((ql[l + 32] & 0x0fu) | (((qh[l] >> 2u) & 0x03u) << 4u)) -
          32;
      out[l + 64] =
          static_cast<int8_t>(((ql[l + 0] >> 4u) & 0x0fu) | (((qh[l] >> 4u) & 0x03u) << 4u)) -
          32;
      out[l + 96] =
          static_cast<int8_t>(((ql[l + 32] >> 4u) & 0x0fu) | (((qh[l] >> 6u) & 0x03u) << 4u)) -
          32;
    }
    out += 128;
    ql += 64;
    qh += 32;
  }

  const int8_t * a = dequant;
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

inline float dot_q6_k_q8_k_row_scalar(const quant::block_q6_k * lhs,
                                      const quant::block_q8_k * rhs,
                                      const uint64_t block_count) noexcept {
  int8_t aux8[quant::QK_K] = {};
  int16_t aux16[8] = {};
  float sums[8] = {};
  int32_t aux32[8] = {};

  float sumf = 0.0f;
  for (uint64_t block = 0; block < block_count; ++block) {
    const uint8_t * q4 = lhs[block].ql.data();
    const uint8_t * qh = lhs[block].qh.data();
    const int8_t * q8 = rhs[block].qs.data();
    std::memset(aux32, 0, sizeof(aux32));
    int8_t * a = aux8;
    for (uint64_t j = 0; j < quant::QK_K; j += 128u) {
      for (uint64_t l = 0; l < 32; ++l) {
        a[l + 0] = static_cast<int8_t>((q4[l + 0] & 0x0fu) | (((qh[l] >> 0u) & 0x03u) << 4u)) - 32;
        a[l + 32] = static_cast<int8_t>((q4[l + 32] & 0x0fu) | (((qh[l] >> 2u) & 0x03u) << 4u)) - 32;
        a[l + 64] = static_cast<int8_t>(((q4[l + 0] >> 4u) & 0x0fu) | (((qh[l] >> 4u) & 0x03u) << 4u)) - 32;
        a[l + 96] = static_cast<int8_t>(((q4[l + 32] >> 4u) & 0x0fu) | (((qh[l] >> 6u) & 0x03u) << 4u)) - 32;
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

inline constexpr uint8_t unary_subop_abs = 0u;
inline constexpr uint8_t unary_subop_neg = 2u;
inline constexpr uint8_t unary_subop_relu = 6u;
inline constexpr uint8_t unary_subop_exp = 13u;

template <uint8_t subop_code, class request_type>
inline void execute_scalar_unary_subop_unchecked(const request_type & request) noexcept {
  if constexpr (subop_code == unary_subop_abs) {
    (void) run_unary(request, [](const float v) { return std::fabs(v); });
  } else if constexpr (subop_code == unary_subop_neg) {
    (void) run_unary(request, [](const float v) { return -v; });
  } else if constexpr (subop_code == unary_subop_relu) {
    (void) run_unary(request, [](const float v) { return std::max(0.0f, v); });
  } else if constexpr (subop_code == unary_subop_exp) {
    (void) run_unary(request, [](const float v) { return std::exp(v); });
  }
}

template <class dispatch_event_type, class context_type, class mark_done_type,
          ::emel::kernel::event::unary_subop subop>
struct exec_scalar_unary_op {
  void operator()(const dispatch_event_type & ev, context_type & ctx) const noexcept {
    execute_scalar_unary_subop_unchecked<static_cast<uint8_t>(subop)>(ev.request);
    mark_done_type{}(ev, ctx);
  }
};

template <class dispatch_event_type, class context_type, class simd_guard_type,
          class unary_subop_guard_type>
struct simd_unary_subop_guard {
  bool operator()(const dispatch_event_type & ev, const context_type & ctx) const noexcept {
    return simd_guard_type{}(ev, ctx) && unary_subop_guard_type{}(ev, ctx);
  }
};

template <class dispatch_event_type, class context_type, class valid_guard_type,
          class unary_subop_guard_type>
struct valid_unary_subop_guard {
  bool operator()(const dispatch_event_type & ev, const context_type & ctx) const noexcept {
    return valid_guard_type{}(ev, ctx) && unary_subop_guard_type{}(ev, ctx);
  }
};

template <class request_type>
inline bool run_mul_mat(const request_type & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool has_empty_dim = k == 0 || m == 0 || n == 0;
  const bool shape_mismatch =
      request.src1.ne[1] != k || request.dst.ne[0] != n || request.dst.ne[1] != m;
  const bool invalid_rank =
      request.src0.ne[2] != 1 || request.src0.ne[3] != 1 ||
      request.src1.ne[2] != 1 || request.src1.ne[3] != 1 ||
      request.dst.ne[2] != 1 || request.dst.ne[3] != 1;
  const bool valid = !(has_empty_dim || shape_mismatch || invalid_rank);
  const uint8_t src0_type = dtype_code(request.src0.type);
  const bool quantized_src0 = is_quantized_k_dtype(src0_type);

  if (valid && quantized_src0) {
    const auto * b_dense = static_cast<const float *>(request.src1.data);
    auto * c_dense = static_cast<float *>(request.dst.data);
    const auto * a_base = static_cast<const uint8_t *>(request.src0.data);
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
        quant::quantize_row_q8_k_strided(
            b_dense + block * quant::QK_K * n + j, n, &q8_blocks[block], quant::QK_K);
      }
      for (uint64_t i = 0; i < m; ++i) {
        const uint8_t * row_ptr = a_base + i * row_bytes;
        if (src0_type == dtype_q2_k) {
          c_dense[i * n + j] = dot_q2_k_q8_k_row_scalar(
              reinterpret_cast<const quant::block_q2_k *>(row_ptr), q8_blocks.data(), block_count);
        } else if (src0_type == dtype_q3_k) {
          c_dense[i * n + j] = dot_q3_k_q8_k_row_scalar(
              reinterpret_cast<const quant::block_q3_k *>(row_ptr), q8_blocks.data(), block_count);
        } else {
          c_dense[i * n + j] = dot_q6_k_q8_k_row_scalar(
              reinterpret_cast<const quant::block_q6_k *>(row_ptr), q8_blocks.data(), block_count);
        }
      }
    }

    return true;
  }

  const bool dense = valid &&
      is_dense_contiguous(request.src0) &&
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
        acc += read_f32_at(request.src0, p, i) * read_f32_at(request.src1, j, p);
      }
      write_f32_at(request.dst, j, i, acc);
    }
  }

  return valid;
}

template <class request_type>
inline bool can_run_mul_mat_argmax(const request_type & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool has_empty_dim = k == 0 || m == 0 || n != 1u;
  const uint8_t src0_type = dtype_code(request.src0.type);
  const uint8_t src1_type = dtype_code(request.src1.type);
  const uint8_t dst_type = dtype_code(request.dst.type);
  const bool valid_shape =
      request.src1.ne[1] == k &&
      request.dst.ne[0] == 1u &&
      request.dst.ne[1] == 1u &&
      request.src0.ne[2] == 1u &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[2] == 1u &&
      request.src1.ne[3] == 1u &&
      request.dst.ne[2] == 1u &&
      request.dst.ne[3] == 1u;
  const bool f32_path = src0_type == dtype_f32 &&
      src1_type == dtype_f32 &&
      dst_type == dtype_f32 &&
      is_dense_contiguous(request.src0) &&
      is_dense_contiguous(request.src1) &&
      is_dense_contiguous(request.dst);
  const bool quantized_path = is_quantized_k_dtype(src0_type) &&
      src1_type == dtype_f32 &&
      dst_type == dtype_f32 &&
      (k % quant::QK_K) == 0u &&
      (k / quant::QK_K) <= quant::MAX_Q8_K_BLOCKS &&
      is_dense_contiguous(request.src1) &&
      is_dense_contiguous(request.dst) &&
      request.src0.nb[0] == 1u &&
      request.src0.nb[1] == quantized_row_storage_bytes(src0_type, k) &&
      request.src0.nb[2] == request.src0.nb[1] * m &&
      request.src0.nb[3] == request.src0.nb[2];
  return !has_empty_dim &&
      request.index_out != nullptr &&
      valid_shape &&
      (f32_path || quantized_path);
}

template <class request_type>
inline bool run_mul_mat_argmax(const request_type & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const bool valid = can_run_mul_mat_argmax(request);
  if (!valid) {
    return false;
  }

  const uint8_t src0_type = dtype_code(request.src0.type);
  const float * b_dense = static_cast<const float *>(request.src1.data);
  float * best_value_out = static_cast<float *>(request.dst.data);
  int32_t best_index = 0;
  float best_value = -std::numeric_limits<float>::infinity();

  if (src0_type == dtype_f32) {
    const float * a_dense = static_cast<const float *>(request.src0.data);
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

  if (is_quantized_k_dtype(src0_type)) {
    const auto * a_base = static_cast<const uint8_t *>(request.src0.data);
    const size_t row_bytes = request.src0.nb[1];
    const uint64_t block_count = k / quant::QK_K;
    std::array<quant::block_q8_k, quant::MAX_Q8_K_BLOCKS> q8_blocks = {};
    if (block_count > q8_blocks.size()) {
      return false;
    }
    quant::quantize_row_q8_k_strided(b_dense, 1u, q8_blocks.data(), static_cast<int64_t>(k));
    for (uint64_t row = 0; row < m; ++row) {
      const uint8_t * row_ptr = a_base + row * row_bytes;
      float value = 0.0f;
      if (src0_type == dtype_q2_k) {
        value = dot_q2_k_q8_k_row_scalar(
            reinterpret_cast<const quant::block_q2_k *>(row_ptr), q8_blocks.data(), block_count);
      } else if (src0_type == dtype_q3_k) {
        value = dot_q3_k_q8_k_row_scalar(
            reinterpret_cast<const quant::block_q3_k *>(row_ptr), q8_blocks.data(), block_count);
      } else {
        value = dot_q6_k_q8_k_row_scalar(
            reinterpret_cast<const quant::block_q6_k *>(row_ptr), q8_blocks.data(), block_count);
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
inline bool run_soft_max(const request_type & request) noexcept {
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

    const double sum = exp_and_sum_ggml_f32(src_dense + offset, dst_dense + offset, width, max_v);
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
      write_f32(request.dst, offset + i, read_f32(request.dst, offset + i) / sum);
    }
  }

  return valid;
}

template <class request_type>
inline bool can_run_copy(const request_type & request) noexcept {
  return tensor_element_count(request.dst) == tensor_element_count(request.src0);
}

template <class request_type>
inline bool can_run_binary(const request_type & request) noexcept {
  const uint64_t count = tensor_element_count(request.dst);
  return count == tensor_element_count(request.src0) &&
         count == tensor_element_count(request.src1);
}

template <class request_type>
inline bool can_run_unary(const request_type & request) noexcept {
  return tensor_element_count(request.dst) == tensor_element_count(request.src0);
}

template <class request_type>
inline bool can_run_mul_mat(const request_type & request) noexcept {
  const uint64_t k = request.src0.ne[0];
  const uint64_t m = request.src0.ne[1];
  const uint64_t n = request.src1.ne[0];
  const bool has_empty_dim = k == 0 || m == 0 || n == 0;
  const uint8_t src0_type = dtype_code(request.src0.type);
  const uint8_t src1_type = dtype_code(request.src1.type);
  const uint8_t dst_type = dtype_code(request.dst.type);
  const bool valid_shape = request.src1.ne[1] == k && request.dst.ne[0] == n &&
         request.dst.ne[1] == m && request.src0.ne[2] == 1 &&
         request.src0.ne[3] == 1 && request.src1.ne[2] == 1 &&
         request.src1.ne[3] == 1 && request.dst.ne[2] == 1 &&
         request.dst.ne[3] == 1;
  const bool f32_path = src0_type == dtype_f32 &&
      src1_type == dtype_f32 &&
      dst_type == dtype_f32;
  const bool quantized_path = is_quantized_k_dtype(src0_type) &&
      src1_type == dtype_f32 &&
      dst_type == dtype_f32 &&
      (k % quant::QK_K) == 0u &&
      (k / quant::QK_K) <= quant::MAX_Q8_K_BLOCKS &&
      is_dense_contiguous(request.src1) &&
      is_dense_contiguous(request.dst) &&
      request.src0.nb[0] == 1u &&
      request.src0.nb[1] == quantized_row_storage_bytes(src0_type, k) &&
      request.src0.nb[2] == request.src0.nb[1] * m &&
      request.src0.nb[3] == request.src0.nb[2];
  const bool packed_q6_vector_q8_rhs_path = src0_type == dtype_q6_k_x8 &&
      src1_type == dtype_q8_k &&
      dst_type == dtype_f32 &&
      (k % quant::QK_K) == 0u &&
      (k / quant::QK_K) <= quant::MAX_Q8_K_BLOCKS &&
      request.src0.nb[0] == 1u &&
      request.src0.nb[1] == quant::packed_q6_k_x8_group_storage_bytes(k) &&
      request.src0.nb[2] ==
          request.src0.nb[1] * quant::packed_q6_k_x8_group_count(m) &&
      request.src0.nb[3] == request.src0.nb[2] &&
      request.src1.nb[0] == 1u &&
      request.src1.nb[1] == quantized_row_storage_bytes(src1_type, k) &&
      request.src1.nb[2] == request.src1.nb[1] &&
      request.src1.nb[3] == request.src1.nb[2] &&
      is_dense_contiguous(request.dst);
  const bool prepared_q6_vector_q8_rhs_path = src0_type == dtype_q6_k_x8_q8_prepared &&
      src1_type == dtype_q8_k &&
      dst_type == dtype_f32 &&
      (k % quant::QK_K) == 0u &&
      (k / quant::QK_K) <= quant::MAX_Q8_K_BLOCKS &&
      request.src0.nb[0] == 1u &&
      request.src0.nb[1] == quant::prepared_q6_k_x8_q8_group_storage_bytes(k) &&
      request.src0.nb[2] ==
          request.src0.nb[1] * quant::packed_q6_k_x8_group_count(m) &&
      request.src0.nb[3] == request.src0.nb[2] &&
      request.src1.nb[0] == 1u &&
      request.src1.nb[1] == quantized_row_storage_bytes(src1_type, k) &&
      request.src1.nb[2] == request.src1.nb[1] &&
      request.src1.nb[3] == request.src1.nb[2] &&
      is_dense_contiguous(request.dst);
  const bool argmax_prepared_q6_vector_q8_rhs_path =
      src0_type == dtype_q6_k_x8_q8_argmax_prepared &&
      src1_type == dtype_q8_k &&
      dst_type == dtype_f32 &&
      (k % quant::QK_K) == 0u &&
      (k / quant::QK_K) <= quant::MAX_Q8_K_BLOCKS &&
      request.src0.nb[0] == 1u &&
      request.src0.nb[1] == quant::argmax_prepared_q6_k_x8_q8_group_storage_bytes(k) &&
      request.src0.nb[2] ==
          request.src0.nb[1] * quant::packed_q6_k_x8_group_count(m) &&
      request.src0.nb[3] == request.src0.nb[2] &&
      request.src1.nb[0] == 1u &&
      request.src1.nb[1] == quantized_row_storage_bytes(src1_type, k) &&
      request.src1.nb[2] == request.src1.nb[1] &&
      request.src1.nb[3] == request.src1.nb[2] &&
      is_dense_contiguous(request.dst);
  return !has_empty_dim &&
      valid_shape &&
      (f32_path || quantized_path || packed_q6_vector_q8_rhs_path ||
       prepared_q6_vector_q8_rhs_path || argmax_prepared_q6_vector_q8_rhs_path);
}

template <class request_type>
inline bool can_run_soft_max(const request_type & request) noexcept {
  const uint64_t width = request.src0.ne[0];
  const uint64_t count = tensor_element_count(request.src0);
  return width != 0 && count != 0 && count % width == 0 &&
         count == tensor_element_count(request.dst);
}

template <class request_type>
inline float flash_attn_scale(const request_type & request) noexcept {
  if (request.op_params_size >= sizeof(float)) {
    float scale = 0.0f;
    std::memcpy(&scale, request.op_params.data(), sizeof(scale));
    return scale;
  }

  const float head_dim = static_cast<float>(request.src0.ne[0]);
  return head_dim > 0.0f ? (1.0f / std::sqrt(head_dim)) : 1.0f;
}

template <class request_type>
inline uint64_t flash_attn_active_tokens(const request_type & request) noexcept {
  return request.src1.ne[1];
}

template <class request_type>
inline uint64_t flash_attn_masked_total_tokens(const request_type & request) noexcept {
  if (request.op_params_size >= sizeof(float) + sizeof(uint32_t)) {
    uint32_t total_tokens = 0u;
    std::memcpy(&total_tokens,
                request.op_params.data() + sizeof(float),
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

inline void scale_f32_scalar(float * data, const float scale, const uint64_t count) noexcept {
  for (uint64_t idx = 0; idx < count; ++idx) {
    data[idx] *= scale;
  }
}

inline void axpy_f32_scalar(float * dst, const float * src,
                            const float alpha, const uint64_t count) noexcept {
  for (uint64_t idx = 0; idx < count; ++idx) {
    dst[idx] += src[idx] * alpha;
  }
}

inline void scale_f16_effective_accumulator_scalar(float * data,
                                                   const float scale,
                                                   const uint64_t count) noexcept {
  const float rounded_scale = round_fp16_scalar(scale);
  for (uint64_t idx = 0; idx < count; ++idx) {
    data[idx] = round_fp16_scalar(round_fp16_scalar(data[idx]) * rounded_scale);
  }
}

inline void axpy_f16_effective_accumulator_scalar(float * dst,
                                                  const float * src,
                                                  const float alpha,
                                                  const uint64_t count) noexcept {
  const float rounded_alpha = round_fp16_scalar(alpha);
  for (uint64_t idx = 0; idx < count; ++idx) {
    const float rounded_dst = round_fp16_scalar(dst[idx]);
    const float rounded_src = round_fp16_scalar(src[idx]);
    dst[idx] = round_fp16_scalar(rounded_dst + rounded_src * rounded_alpha);
  }
}

inline void convert_f32_to_fp16_buffer_scalar(const float * src,
                                              uint16_t * dst,
                                              const uint64_t count) noexcept {
  for (uint64_t idx = 0; idx < count; ++idx) {
    dst[idx] = quant::fp32_to_fp16(src[idx]);
  }
}

inline void zero_f16_buffer_scalar(uint16_t * dst, const uint64_t count) noexcept {
  std::fill_n(dst, count, static_cast<uint16_t>(0u));
}

inline void scale_f16_buffer_scalar(uint16_t * data,
                                    const float scale,
                                    const uint64_t count) noexcept {
  const float rounded_scale = round_fp16_scalar(scale);
  for (uint64_t idx = 0; idx < count; ++idx) {
    const float rounded_value = quant::fp16_to_fp32(data[idx]);
    data[idx] = quant::fp32_to_fp16(round_fp16_scalar(rounded_value * rounded_scale));
  }
}

inline void axpy_f16_buffer_scalar(uint16_t * dst,
                                   const uint16_t * src,
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

inline void convert_f16_buffer_to_f32_scalar(const uint16_t * src,
                                             float * dst,
                                             const uint64_t count) noexcept {
  for (uint64_t idx = 0; idx < count; ++idx) {
    dst[idx] = quant::fp16_to_fp32(src[idx]);
  }
}

template <class request_type>
inline bool has_required_src2(const request_type & request) noexcept {
  return request.src2.data != nullptr &&
         is_supported_dtype(dtype_code(request.src2.type)) &&
         has_valid_tensor_layout(request.src2) &&
         tensor_element_count(request.src2) > 0;
}

template <class request_type>
inline bool can_run_flash_attn_ext(const request_type & request) noexcept {
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

  const bool explicit_operand_contract = src0_type == dtype_f32 &&
      src1_type == dtype_f16 &&
      src2_type == dtype_f16 &&
      dst_type == dtype_f32;
  const bool dims_present =
      head_dim != 0u && query_count == 1u && head_count != 0u && kv_tokens != 0u &&
      kv_head_count != 0u;
  const bool src2_valid = has_required_src2(request);
  const bool shape_match =
      request.src1.ne[0] == head_dim &&
      request.src2.ne[0] == head_dim &&
      request.src2.ne[1] == kv_tokens &&
      request.src2.ne[2] == kv_head_count &&
      request.dst.ne[0] == head_dim &&
      request.dst.ne[1] == query_count &&
      request.dst.ne[2] == head_count &&
      request.src0.ne[3] == 1u &&
      request.src1.ne[3] == 1u &&
      request.src2.ne[3] == 1u &&
      request.dst.ne[3] == 1u &&
      (head_count % kv_head_count) == 0u;
  const bool layout_supported =
      is_dense_contiguous(request.src0) &&
      has_valid_tensor_layout(request.src1) &&
      has_valid_tensor_layout(request.src2) &&
      is_dense_contiguous(request.dst);
  const float scale = flash_attn_scale(request);

  return explicit_operand_contract && dims_present && src2_valid && shape_match &&
      layout_supported &&
      masked_total_tokens >= kv_tokens &&
      std::isfinite(scale) &&
      scale > 0.0f;
}

template <class request_type>
inline bool can_run_unary_subop(const request_type & request) noexcept {
  const auto subop = static_cast<uint8_t>(request.subop);
  const bool supported_subop = subop == unary_subop_abs || subop == unary_subop_neg ||
                               subop == unary_subop_relu || subop == unary_subop_exp;
  return supported_subop && can_run_unary(request);
}

template <class request_type>
inline bool can_execute_scalar(const request_type & request) noexcept {
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
  } else if constexpr (std::is_same_v<request_type, event::op_unary>) {
    return false;
  }
  return false;
}

template <class request_type>
inline bool run_flash_attn_ext_with_workspace(const request_type & request,
                                              flash_attn_workspace & workspace) noexcept;

template <class request_type>
inline bool run_flash_attn_ext(const request_type & request) noexcept {
  flash_attn_workspace workspace{};
  return run_flash_attn_ext_with_workspace(request, workspace);
}

template <class request_type>
inline bool run_flash_attn_ext_active_kv_with_workspace(
    const request_type & request,
    flash_attn_workspace & workspace) noexcept {
  const uint64_t kv_tokens = flash_attn_active_tokens(request);
  const uint64_t masked_total_tokens = flash_attn_masked_total_tokens(request);
  if (masked_total_tokens < kv_tokens) {
    return false;
  }

  if (workspace.prepared_tokens == kv_tokens) {
    ++workspace.reuse_count;
  } else {
    workspace.prepared_tokens = kv_tokens;
  }

  const uint64_t head_dim = request.src0.ne[0];
  const uint64_t head_count = request.src0.ne[2];
  const uint64_t kv_head_count = request.src1.ne[2];
  const float scale = flash_attn_scale(request);
  if (head_dim > workspace.q_buffer_f16.size() || head_dim > workspace.accum_buffer_f16.size()) {
    return false;
  }

  (void) masked_total_tokens;
  const uint64_t n_rep = head_count / kv_head_count;
  for (uint64_t head = 0; head < head_count; ++head) {
    const uint64_t kv_head = head / n_rep;
    const float * q = tensor_row_ptr(request.src0, 0u, head);
    uint16_t * accum = workspace.accum_buffer_f16.data();
    float * dst = tensor_row_ptr_mut(request.dst, 0u, head);
    convert_f32_to_fp16_buffer_scalar(q, workspace.q_buffer_f16.data(), head_dim);
    zero_f16_buffer_scalar(accum, head_dim);

    float score_sum = 0.0f;
    float max_score = -std::numeric_limits<float>::infinity();
    for (uint64_t token = 0; token < kv_tokens; ++token) {
      const uint16_t * k = tensor_row_ptr_as<uint16_t>(request.src1, token, kv_head);
      const float score = dot_product_f16_f16_scores(workspace.q_buffer_f16.data(), k, head_dim) *
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

      const uint16_t * v = tensor_row_ptr_as<uint16_t>(request.src2, token, kv_head);
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

  return true;
}

template <class request_type>
inline bool run_flash_attn_ext_with_workspace(const request_type & request,
                                              flash_attn_workspace & workspace) noexcept {
  if (!can_run_flash_attn_ext(request)) {
    return false;
  }

  return run_flash_attn_ext_active_kv_with_workspace(request, workspace);
}

template <class request_type>
inline bool can_run_backend_request(const request_type & request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_unary>) {
    return can_run_unary_subop(request);
  }
  return can_execute_scalar(request);
}

template <class request_type>
inline void execute_scalar_unchecked(const request_type & request) noexcept {
  if constexpr (std::is_same_v<request_type, event::op_dup>) {
    (void) run_copy(request);
  } else if constexpr (std::is_same_v<request_type, event::op_add>) {
    (void) run_binary(request, [](const float lhs, const float rhs) { return lhs + rhs; });
  } else if constexpr (std::is_same_v<request_type, event::op_sub>) {
    (void) run_binary(request, [](const float lhs, const float rhs) { return lhs - rhs; });
  } else if constexpr (std::is_same_v<request_type, event::op_mul>) {
    (void) run_binary(request, [](const float lhs, const float rhs) { return lhs * rhs; });
  } else if constexpr (std::is_same_v<request_type, event::op_div>) {
    (void) run_binary(request, [](const float lhs, const float rhs) { return lhs / rhs; });
  } else if constexpr (std::is_same_v<request_type, event::op_sqr>) {
    (void) run_unary(request, [](const float v) { return v * v; });
  } else if constexpr (std::is_same_v<request_type, event::op_sqrt>) {
    (void) run_unary(request, [](const float v) { return std::sqrt(v); });
  } else if constexpr (std::is_same_v<request_type, event::op_log>) {
    (void) run_unary(request, [](const float v) { return std::log(v); });
  } else if constexpr (std::is_same_v<request_type, event::op_sin>) {
    (void) run_unary(request, [](const float v) { return std::sin(v); });
  } else if constexpr (std::is_same_v<request_type, event::op_cos>) {
    (void) run_unary(request, [](const float v) { return std::cos(v); });
  } else if constexpr (std::is_same_v<request_type, event::op_mul_mat>) {
    (void) run_mul_mat(request);
  } else if constexpr (std::is_same_v<request_type, event::op_mul_mat_argmax>) {
    (void) run_mul_mat_argmax(request);
  } else if constexpr (std::is_same_v<request_type, event::op_soft_max>) {
    (void) run_soft_max(request);
  } else if constexpr (std::is_same_v<request_type, event::op_flash_attn_ext>) {
    (void) run_flash_attn_ext(request);
  }
}

template <class request_type>
inline bool execute_scalar(const request_type & request) noexcept {
  const bool can_execute = can_execute_scalar(request);
  if (can_execute) {
    execute_scalar_unchecked(request);
  }
  return can_execute;
}

}  // namespace emel::kernel::detail
