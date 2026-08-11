#include "emel/kernel/metal/detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "emel/kernel/detail.hpp"

namespace emel::kernel::metal::detail {

namespace {

using ::emel::kernel::detail::dtype_code;
using ::emel::kernel::detail::dtype_size_bytes;
using ::emel::kernel::detail::tensor_element_count;
using ::emel::kernel::detail::tensor_stride_bytes;

inline uint64_t tensor_storage_bytes_impl(const std::array<uint64_t, 4> &ne,
                                          const std::array<uint64_t, 4> &nb,
                                          const uint8_t dtype) noexcept {
  const uint64_t elem = dtype_size_bytes(dtype);
  // Byte-addressable rows (quantized weights and i8 views): each row
  // physically spans nb[1] bytes although ne[0] elements are fewer (a q8_0
  // row is 34 * k / 32 bytes for k elements), and staging copies the full
  // row. Measuring only the last-element offset would under-size the span by
  // up to nb[1] - ne[0] bytes and let a near-capacity request overflow its
  // staging slice.
  if (elem == 1u && nb[0] == 1u && ne[1] != 0u && ne[2] == 1u && ne[3] == 1u) {
    return nb[1] * ne[1];
  }
  uint64_t end_offset = 0u;
  for (size_t d = 0; d < 4; ++d) {
    const uint64_t dim = ne[d] != 0u ? ne[d] : 1u;
    uint64_t stride = nb[d];
    if (nb[0] == 0u) {
      stride = elem;
      for (size_t i = 0; i < d; ++i) {
        stride *= ne[i] != 0u ? ne[i] : 1u;
      }
    }
    end_offset += (dim - 1u) * stride;
  }
  return end_offset + elem;
}

template <class tensor_type>
inline bool stage_tensor(metal_runtime &rt, const uint32_t slice,
                         const tensor_type &t,
                         uint64_t &staged_bytes_out) noexcept {
  if (t.data == nullptr) {
    staged_bytes_out = 0u;
    return true;
  }
  const uint64_t staged = tensor_storage_bytes(t);
  if (staged > k_pool_slice_capacity_bytes) {
    return false;
  }
  staged_bytes_out = staged;
  uint8_t *dst = static_cast<uint8_t *>(rt.slice_contents(slice));
  const uint8_t *src = static_cast<const uint8_t *>(t.data);
  const uint8_t dtype = dtype_code(t.type);
  const uint64_t elem = dtype_size_bytes(dtype);
  const uint64_t count = tensor_element_count(t);

  std::array<uint64_t, 4> st = {};
  for (size_t d = 0; d < 4; ++d) {
    st[d] = tensor_stride_bytes(t, d);
  }

  bool dense = true;
  uint64_t expected = elem;
  for (size_t d = 0; d < 4; ++d) {
    dense = dense && st[d] == expected;
    expected *= t.ne[d] != 0u ? t.ne[d] : 1u;
  }
  if (dense) {
    std::memcpy(dst, src, static_cast<size_t>(count * elem));
    return true;
  }

  const bool row_contiguous =
      st[0] == elem && (t.ne[1] == 0u || st[1] == elem * t.ne[0]);
  if (row_contiguous) {
    const uint64_t row_bytes = t.ne[0] != 0u ? t.ne[0] * elem : 0u;
    const uint64_t rows = t.ne[0] != 0u ? count / t.ne[0] : 0u;
    for (uint64_t row = 0; row < rows; ++row) {
      const uint64_t src_off = st[1] * row;
      std::memcpy(dst + src_off, src + row_bytes * row,
                  static_cast<size_t>(row_bytes));
    }
    return true;
  }

  // Byte-addressable rows (quantized weights and i8 views): each row spans
  // nb[1] bytes over ne[1] rows even though ne[0] * elem is shorter (q8_0
  // rows are 34 * k / 32 bytes for k elements).
  if (elem == 1u && t.ne[1] != 0u && t.ne[2] == 1u && t.ne[3] == 1u) {
    const uint64_t row_bytes = t.nb[1];
    const uint64_t rows = t.ne[1];
    for (uint64_t row = 0; row < rows; ++row) {
      std::memcpy(dst + st[1] * row, src + row_bytes * row,
                  static_cast<size_t>(row_bytes));
    }
    return true;
  }

  uint64_t remaining = count;
  const auto dims = [&](const size_t d) {
    return t.ne[d] != 0u ? t.ne[d] : 1u;
  };
  for (uint64_t idx = 0; idx < count; ++idx) {
    remaining = idx;
    uint64_t offset = 0u;
    bool dims_active = true;
    for (size_t d = 0; d < 4; ++d) {
      const bool step_active = dims_active && t.ne[d] != 0u;
      const uint64_t dim = dims(d);
      const uint64_t coord = step_active ? remaining % dim : 0u;
      offset += step_active ? coord * st[d] : 0u;
      remaining = step_active ? remaining / dim : remaining;
      dims_active = dims_active && t.ne[d] != 0u;
    }
    // Both sides use the same nb-decomposed offset: the staged buffer
    // preserves the source layout for the shaders, and the source element
    // lives at its own layout offset (equal to elem * idx only when dense).
    std::memcpy(dst + offset, src + offset, static_cast<size_t>(elem));
  }
  return true;
}

template <class tensor_type>
inline bool readback_tensor(metal_runtime &rt, const uint32_t slice,
                            const tensor_type &t) noexcept {
  if (t.data == nullptr) {
    return true;
  }
  const uint8_t *src = static_cast<const uint8_t *>(rt.slice_contents(slice));
  uint8_t *dst = static_cast<uint8_t *>(t.data);
  const uint8_t dtype = dtype_code(t.type);
  const uint64_t elem = dtype_size_bytes(dtype);
  const uint64_t count = tensor_element_count(t);

  std::array<uint64_t, 4> st = {};
  for (size_t d = 0; d < 4; ++d) {
    st[d] = tensor_stride_bytes(t, d);
  }

  bool dense = true;
  uint64_t expected = elem;
  for (size_t d = 0; d < 4; ++d) {
    dense = dense && st[d] == expected;
    expected *= t.ne[d] != 0u ? t.ne[d] : 1u;
  }
  if (dense) {
    std::memcpy(dst, src, static_cast<size_t>(count * elem));
    return true;
  }

  const bool row_contiguous =
      st[0] == elem && (t.ne[1] == 0u || st[1] == elem * t.ne[0]);
  if (row_contiguous) {
    const uint64_t row_bytes = t.ne[0] != 0u ? t.ne[0] * elem : 0u;
    const uint64_t rows = t.ne[0] != 0u ? count / t.ne[0] : 0u;
    for (uint64_t row = 0; row < rows; ++row) {
      std::memcpy(dst + row_bytes * row, src + st[1] * row,
                  static_cast<size_t>(row_bytes));
    }
    return true;
  }

  // Byte-addressable rows (mirror of the stage path above).
  if (elem == 1u && t.ne[1] != 0u && t.ne[2] == 1u && t.ne[3] == 1u) {
    const uint64_t row_bytes = t.nb[1];
    const uint64_t rows = t.ne[1];
    for (uint64_t row = 0; row < rows; ++row) {
      std::memcpy(dst + row_bytes * row, src + st[1] * row,
                  static_cast<size_t>(row_bytes));
    }
    return true;
  }

  for (uint64_t idx = 0; idx < count; ++idx) {
    uint64_t remaining = idx;
    uint64_t offset = 0u;
    bool dims_active = true;
    for (size_t d = 0; d < 4; ++d) {
      const bool step_active = dims_active && t.ne[d] != 0u;
      const uint64_t dim = t.ne[d] != 0u ? t.ne[d] : 1u;
      const uint64_t coord = step_active ? remaining % dim : 0u;
      offset += step_active ? coord * st[d] : 0u;
      remaining = step_active ? remaining / dim : remaining;
      dims_active = dims_active && t.ne[d] != 0u;
    }
    // Mirror of the stage path: dst receives its element at its own layout
    // offset, which equals elem * idx only when dense.
    std::memcpy(dst + offset, src + offset, static_cast<size_t>(elem));
  }
  return true;
}

inline shader_tensor
make_shader_tensor(const ::emel::kernel::event::tensor_view &t) noexcept {
  shader_tensor out = {};
  out.ne = t.ne;
  for (size_t d = 0; d < 4; ++d) {
    out.nb[d] = tensor_stride_bytes(t, d);
  }
  out.dtype = dtype_code(t.type);
  return out;
}

inline shader_tensor
make_shader_tensor(const ::emel::kernel::event::tensor_view_mut &t) noexcept {
  shader_tensor out = {};
  out.ne = t.ne;
  for (size_t d = 0; d < 4; ++d) {
    out.nb[d] = tensor_stride_bytes(t, d);
  }
  out.dtype = dtype_code(t.type);
  return out;
}

inline uint32_t grid_threads(const uint64_t count) noexcept {
  const uint64_t clamped = std::min<uint64_t>(count, 0xFFFFFFFu);
  return static_cast<uint32_t>(std::max<uint64_t>(clamped, 1u));
}

} // namespace

uint64_t
tensor_storage_bytes(const ::emel::kernel::event::tensor_view &t) noexcept {
  return tensor_storage_bytes_impl(t.ne, t.nb, dtype_code(t.type));
}

uint64_t
tensor_storage_bytes(const ::emel::kernel::event::tensor_view_mut &t) noexcept {
  return tensor_storage_bytes_impl(t.ne, t.nb, dtype_code(t.type));
}

} // namespace emel::kernel::metal::detail

//------------------------------------------------------------------------------//
// Apple implementation: MSL library, pipeline states, staging pool, and the
// synchronous dispatch path. Compiled as Objective-C++ (CMake sets the OBJCXX
// language for this TU on Apple); other hosts get the compile-out stub below.
//------------------------------------------------------------------------------//

#if defined(__APPLE__)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace emel::kernel::metal::detail {

namespace {

constexpr const char *k_msl_source = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct tensor_info {
    uint64_t ne[4];
    uint64_t nb[4];
    uint32_t dtype;
    uint32_t reserved;
};

struct emel_params {
    tensor_info src0;
    tensor_info src1;
    tensor_info dst;
    int32_t i32[8];
    float f32[8];
};

constant constexpr uint32_t EMEL_DTYPE_F32 = 0u;
constant constexpr uint32_t EMEL_DTYPE_F16 = 1u;
constant constexpr uint32_t EMEL_DTYPE_Q8_0 = 8u;
constant constexpr uint32_t EMEL_DTYPE_I32 = 26u;

static inline uint64_t emel_offset(const constant tensor_info &t, uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3) {
    return i0 * t.nb[0] + i1 * t.nb[1] + i2 * t.nb[2] + i3 * t.nb[3];
}

static inline float emel_load_f32(const device uint8_t *base, const constant tensor_info &t, uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3) {
    const device float *p = (const device float *)(base + emel_offset(t, i0, i1, i2, i3));
    return *p;
}

static inline float emel_load_f16(const device uint8_t *base, const constant tensor_info &t, uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3) {
    const device half *p = (const device half *)(base + emel_offset(t, i0, i1, i2, i3));
    return float(*p);
}

static inline int32_t emel_load_i32(const device uint8_t *base, const constant tensor_info &t, uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3) {
    const device int32_t *p = (const device int32_t *)(base + emel_offset(t, i0, i1, i2, i3));
    return *p;
}

static inline void emel_store_f32(device uint8_t *base, const constant tensor_info &t, uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3, float v) {
    device float *p = (device float *)(base + emel_offset(t, i0, i1, i2, i3));
    *p = v;
}

// Matches ::emel::kernel::detail::execute_scalar_unary_subop_unchecked for the
// supported subops (abs, neg, tanh, elu, relu, gelu, silu, exp), including the
// fp16 rounding steps of the reference gelu. The branch below is GPU-side
// numeric work for an already-chosen variant: the transition rows pin
// ev.request.subop == subop (guard::valid_op_unary_subop), so the identity
// default is unreachable and must stay in sync with
// guard::detail::can_run_unary_subop_set.
static inline float emel_unary(float v, int32_t subop) {
    if (subop == 0) { return fabs(v); }        // abs
    if (subop == 2) { return -v; }             // neg
    if (subop == 4) { return tanh(v); }        // tanh
    if (subop == 5) { return v > 0.0f ? v : (exp(v) - 1.0f); }  // elu
    if (subop == 6) { return max(0.0f, v); }   // relu
    if (subop == 8) {                          // gelu (ggml fp16 path)
        if (v <= -10.0f) { return 0.0f; }
        if (v >= 10.0f) { return v; }
        const float q = float(half(v));
        const float approx = 0.5f * q * (1.0f + tanh(0.79788456080286535588f * (q + 0.044715f * q * q * q)));
        return float(half(approx));
    }
    if (subop == 10) { return v / (1.0f + exp(-v)); }  // silu
    if (subop == 13) { return exp(v); }        // exp
    return v;
}

// dst[i,j] = sum_q src0[q,i] * src1[q,j] (ggml mul_mat semantics).
kernel void emel_mul_mat_f32(device const uint8_t *src0 [[buffer(1)]],
                             device const uint8_t *src1 [[buffer(2)]],
                             device uint8_t *dst [[buffer(3)]],
                             constant const emel_params &p [[buffer(0)]],
                             uint tid [[thread_position_in_grid]]) {
    const uint64_t k = p.src0.ne[0];
    const uint64_t m = p.src0.ne[1];
    const uint64_t n = p.src1.ne[0];
    const uint64_t total = m * n;
    if (tid >= total) { return; }
    const uint64_t i = tid / n;
    const uint64_t j = tid % n;
    float acc = 0.0f;
    for (uint64_t q = 0; q < k; ++q) {
        // src1 is [n, k]: the dot index lives in dim 1, the output column in
        // dim 0 (ggml mul_mat transposes src1); matches run_mul_mat's dense
        // b_dense[p * n + j] read.
        acc += emel_load_f32(src0, p.src0, q, i, 0, 0) * emel_load_f32(src1, p.src1, j, q, 0, 0);
    }
    emel_store_f32(dst, p.dst, j, i, 0, 0, acc);
}

// f16 src0 weights against f16 src1, f32 accumulation. The f16 row contract
// (run_mul_mat_f16) names the output [m, n] with n = src1.ne[1], the
// transpose of the f32 orientation: dst[i, j] = sum_q src0[q, i] * src1[q, j].
kernel void emel_mul_mat_f16(device const uint8_t *src0 [[buffer(1)]],
                             device const uint8_t *src1 [[buffer(2)]],
                             device uint8_t *dst [[buffer(3)]],
                             constant const emel_params &p [[buffer(0)]],
                             uint tid [[thread_position_in_grid]]) {
    const uint64_t k = p.src0.ne[0];
    const uint64_t m = p.src0.ne[1];
    const uint64_t n = p.src1.ne[1];
    const uint64_t total = m * n;
    if (tid >= total) { return; }
    const uint64_t i = tid / n;
    const uint64_t j = tid % n;
    float acc = 0.0f;
    for (uint64_t q = 0; q < k; ++q) {
        acc += emel_load_f16(src0, p.src0, q, i, 0, 0) * emel_load_f16(src1, p.src1, q, j, 0, 0);
    }
    emel_store_f32(dst, p.dst, i, j, 0, 0, acc);
}

// q8_0 src0 rows against f32 src1: per block, quantize the RHS column with the
// reference round/clamp, then an int32 dot scaled by the fp16-rounded scale
// product (mirrors quantize_row_q8_0_strided + dot_q8_0_q8_0_row_scalar).
kernel void emel_mul_mat_q8_0(device const uint8_t *src0 [[buffer(1)]],
                              device const uint8_t *src1 [[buffer(2)]],
                              device uint8_t *dst [[buffer(3)]],
                              constant const emel_params &p [[buffer(0)]],
                              uint tid [[thread_position_in_grid]]) {
    const uint64_t k = p.src0.ne[0];
    const uint64_t m = p.src0.ne[1];
    const uint64_t n = p.src1.ne[0];
    const uint64_t total = m * n;
    if (tid >= total) { return; }
    const uint64_t i = tid / n;
    const uint64_t j = tid % n;
    const uint64_t blocks = k / 32u;
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; ++b) {
        // q8_0 row layout: 34 bytes per block (fp16 scale + 32 int8), the
        // row stride comes from the view; block b sits at b * 34 bytes.
        const device uint8_t *wbase = src0 + i * p.src0.nb[1] + b * 34u;
        const float wd = float(*(const device half *)wbase);
        const device int8_t *wqs = (const device int8_t *)(wbase + 2);
        float amax = 0.0f;
        for (uint64_t t2 = 0; t2 < 32u; ++t2) {
            amax = max(amax, fabs(emel_load_f32(src1, p.src1, j, b * 32u + t2, 0, 0)));
        }
        const float d = amax / 127.0f;
        const float inv_d = d != 0.0f ? 1.0f / d : 0.0f;
        const half dh = half(d);
        int32_t sumi = 0;
        for (uint64_t t2 = 0; t2 < 32u; ++t2) {
            const float x = emel_load_f32(src1, p.src1, j, b * 32u + t2, 0, 0);
            const int32_t q = clamp((int32_t)round(x * inv_d), -127, 127);
            sumi += (int32_t)wqs[t2] * q;
        }
        acc += (float)sumi * (wd * float(dh));
    }
    emel_store_f32(dst, p.dst, j, i, 0, 0, acc);
}

static inline void emel_decompose(uint64_t idx, const constant tensor_info &t, thread uint64_t (&out)[4]) {
    out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
    bool active = true;
    for (uint32_t d = 0; d < 4; ++d) {
        const uint64_t dim = t.ne[d] != 0u ? t.ne[d] : 1u;
        if (active) {
            out[d] = idx % dim;
            idx /= dim;
            active = t.ne[d] != 0u;
        }
    }
}

// Elementwise add over the dst count (strided reads, mirrors run_binary).
kernel void emel_add(device const uint8_t *src0 [[buffer(1)]],
                     device const uint8_t *src1 [[buffer(2)]],
                     device uint8_t *dst [[buffer(3)]],
                     constant const emel_params &p [[buffer(0)]],
                     uint tid [[thread_position_in_grid]]) {
    const uint64_t count = (uint64_t)p.i32[0];
    if (tid >= count) { return; }
    // Each operand decomposes the linear index over its own shape (the
    // scalar run_binary does the same); equal counts do not imply equal ne.
    uint64_t c0[4] = {0, 0, 0, 0};
    uint64_t c1[4] = {0, 0, 0, 0};
    uint64_t cd[4] = {0, 0, 0, 0};
    emel_decompose(tid, p.src0, c0);
    emel_decompose(tid, p.src1, c1);
    emel_decompose(tid, p.dst, cd);
    emel_store_f32(dst, p.dst, cd[0], cd[1], cd[2], cd[3],
                   emel_load_f32(src0, p.src0, c0[0], c0[1], c0[2], c0[3]) +
                   emel_load_f32(src1, p.src1, c1[0], c1[1], c1[2], c1[3]));
}

// Bias/residual broadcast: dst[row, col] = src0[row, col] + src1[col].
kernel void emel_add_broadcast_row(device const uint8_t *src0 [[buffer(1)]],
                                   device const uint8_t *src1 [[buffer(2)]],
                                   device uint8_t *dst [[buffer(3)]],
                                   constant const emel_params &p [[buffer(0)]],
                                   uint tid [[thread_position_in_grid]]) {
    const uint64_t cols = (uint64_t)p.i32[0];
    const uint64_t count = (uint64_t)p.i32[1];
    if (tid >= count) { return; }
    const uint64_t row = tid / cols;
    const uint64_t col = tid % cols;
    const uint64_t i0 = col;
    const uint64_t i1 = row;
    const float lhs = emel_load_f32(src0, p.src0, i0, i1, 0, 0);
    const float rhs = emel_load_f32(src1, p.src1, col, 0, 0, 0);
    emel_store_f32(dst, p.dst, i0, i1, 0, 0, lhs + rhs);
}

// Elementwise unary over the dst count; subop in i32[0].
kernel void emel_unary(device const uint8_t *src0 [[buffer(1)]],
                       device uint8_t *dst [[buffer(3)]],
                       constant const emel_params &p [[buffer(0)]],
                       uint tid [[thread_position_in_grid]]) {
    const uint64_t count = (uint64_t)p.i32[0];
    if (tid >= count) { return; }
    uint64_t c0[4] = {0, 0, 0, 0};
    uint64_t cd[4] = {0, 0, 0, 0};
    emel_decompose(tid, p.src0, c0);
    emel_decompose(tid, p.dst, cd);
    emel_store_f32(dst, p.dst, cd[0], cd[1], cd[2], cd[3],
                   emel_unary(emel_load_f32(src0, p.src0, c0[0], c0[1], c0[2], c0[3]), p.i32[1]));
}

// conv1d im2col: dst[out, channel*taps+tap] = src1[in], in = out*s0 + tap*d0 - p0.
kernel void emel_im2col_f32(device const uint8_t *src0 [[buffer(1)]],
                            device const uint8_t *src1 [[buffer(2)]],
                            device uint8_t *dst [[buffer(3)]],
                            constant const emel_params &p [[buffer(0)]],
                            uint tid [[thread_position_in_grid]]) {
    const int64_t taps = (int64_t)p.src0.ne[0];
    const int64_t channels = (int64_t)p.src0.ne[1];
    const int64_t length = (int64_t)p.src1.ne[0];
    const int64_t out_length = (int64_t)p.dst.ne[1];
    const int64_t batches = (int64_t)p.dst.ne[2];
    const int64_t s0 = (int64_t)p.i32[0];
    const int64_t p0 = (int64_t)p.i32[2];
    const int64_t d0 = (int64_t)p.i32[4];
    const uint64_t total = (uint64_t)(batches * out_length);
    if (tid >= total) { return; }
    const int64_t batch = (int64_t)(tid / (uint64_t)out_length);
    const int64_t out = (int64_t)(tid % (uint64_t)out_length);
    const uint64_t row_width = (uint64_t)(channels * taps);
    const uint64_t dst_row = ((uint64_t)batch * (uint64_t)out_length + (uint64_t)out) * row_width;
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t tap = 0; tap < taps; ++tap) {
            const int64_t in = out * s0 + tap * d0 - p0;
            const bool inside = in >= 0 && in < length;
            const float value = inside ? emel_load_f32(src1, p.src1, (uint64_t)in, (uint64_t)channel, (uint64_t)batch, 0) : 0.0f;
            emel_store_f32(dst, p.dst, dst_row + (uint64_t)channel * (uint64_t)taps + (uint64_t)tap, 0, 0, 0, value);
        }
    }
}

// f16 im2col output: values round through half (ggml f16 operand class).
kernel void emel_im2col_f16(device const uint8_t *src0 [[buffer(1)]],
                            device const uint8_t *src1 [[buffer(2)]],
                            device uint8_t *dst [[buffer(3)]],
                            constant const emel_params &p [[buffer(0)]],
                            uint tid [[thread_position_in_grid]]) {
    const int64_t taps = (int64_t)p.src0.ne[0];
    const int64_t channels = (int64_t)p.src0.ne[1];
    const int64_t length = (int64_t)p.src1.ne[0];
    const int64_t out_length = (int64_t)p.dst.ne[1];
    const int64_t batches = (int64_t)p.dst.ne[2];
    const int64_t s0 = (int64_t)p.i32[0];
    const int64_t p0 = (int64_t)p.i32[2];
    const int64_t d0 = (int64_t)p.i32[4];
    const uint64_t total = (uint64_t)(batches * out_length);
    if (tid >= total) { return; }
    const int64_t batch = (int64_t)(tid / (uint64_t)out_length);
    const int64_t out = (int64_t)(tid % (uint64_t)out_length);
    const uint64_t row_width = (uint64_t)(channels * taps);
    const uint64_t dst_row = ((uint64_t)batch * (uint64_t)out_length + (uint64_t)out) * row_width;
    for (int64_t channel = 0; channel < channels; ++channel) {
        for (int64_t tap = 0; tap < taps; ++tap) {
            const int64_t in = out * s0 + tap * d0 - p0;
            const bool inside = in >= 0 && in < length;
            const float value = inside ? emel_load_f32(src1, p.src1, (uint64_t)in, (uint64_t)channel, (uint64_t)batch, 0) : 0.0f;
            device half *p_half = (device half *)(dst + emel_offset(p.dst, dst_row + (uint64_t)channel * (uint64_t)taps + (uint64_t)tap, 0, 0, 0));
            *p_half = half(value);
        }
    }
}

// Transposed conv1d: dst[o, oc] = sum over ic, tap with o == in*s0 + tap.
kernel void emel_conv_transpose_1d_f32(device const uint8_t *src0 [[buffer(1)]],
                                       device const uint8_t *src1 [[buffer(2)]],
                                       device uint8_t *dst [[buffer(3)]],
                                       constant const emel_params &p [[buffer(0)]],
                                       uint tid [[thread_position_in_grid]]) {
    const int64_t taps = (int64_t)p.src0.ne[0];
    const int64_t out_channels = (int64_t)p.src0.ne[1];
    const int64_t in_channels = (int64_t)p.src0.ne[2];
    const int64_t length = (int64_t)p.src1.ne[0];
    const int64_t out_length = (int64_t)p.dst.ne[0];
    const int64_t s0 = (int64_t)p.i32[0];
    const uint64_t total = (uint64_t)(out_channels * out_length);
    if (tid >= total) { return; }
    const int64_t oc = (int64_t)(tid / (uint64_t)out_length);
    const int64_t o = (int64_t)(tid % (uint64_t)out_length);
    float acc = 0.0f;
    for (int64_t ic = 0; ic < in_channels; ++ic) {
        for (int64_t tap = 0; tap < taps; ++tap) {
            const int64_t rem = o - tap;
            if (rem >= 0 && (rem % s0) == 0) {
                const int64_t in = rem / s0;
                if (in < length) {
                    const float input = emel_load_f32(src1, p.src1, (uint64_t)in, (uint64_t)ic, 0, 0);
                    const float w = emel_load_f32(src0, p.src0, (uint64_t)tap, (uint64_t)oc, (uint64_t)ic, 0);
                    acc += input * w;
                }
            }
        }
    }
    emel_store_f32(dst, p.dst, (uint64_t)o, (uint64_t)oc, 0, 0, acc);
}

// f16 weights variant: inputs round through half before the tap multiplies
// (matches run_conv_transpose_1d_as<true>).
kernel void emel_conv_transpose_1d_f16(device const uint8_t *src0 [[buffer(1)]],
                                       device const uint8_t *src1 [[buffer(2)]],
                                       device uint8_t *dst [[buffer(3)]],
                                       constant const emel_params &p [[buffer(0)]],
                                       uint tid [[thread_position_in_grid]]) {
    const int64_t taps = (int64_t)p.src0.ne[0];
    const int64_t out_channels = (int64_t)p.src0.ne[1];
    const int64_t in_channels = (int64_t)p.src0.ne[2];
    const int64_t length = (int64_t)p.src1.ne[0];
    const int64_t out_length = (int64_t)p.dst.ne[0];
    const int64_t s0 = (int64_t)p.i32[0];
    const uint64_t total = (uint64_t)(out_channels * out_length);
    if (tid >= total) { return; }
    const int64_t oc = (int64_t)(tid / (uint64_t)out_length);
    const int64_t o = (int64_t)(tid % (uint64_t)out_length);
    float acc = 0.0f;
    for (int64_t ic = 0; ic < in_channels; ++ic) {
        for (int64_t tap = 0; tap < taps; ++tap) {
            const int64_t rem = o - tap;
            if (rem >= 0 && (rem % s0) == 0) {
                const int64_t in = rem / s0;
                if (in < length) {
                    const float input = float(half(emel_load_f32(src1, p.src1, (uint64_t)in, (uint64_t)ic, 0, 0)));
                    const float w = emel_load_f16(src0, p.src0, (uint64_t)tap, (uint64_t)oc, (uint64_t)ic, 0);
                    acc += input * w;
                }
            }
        }
    }
    emel_store_f32(dst, p.dst, (uint64_t)o, (uint64_t)oc, 0, 0, acc);
}

// Codebook gather: dst[row, col] = src0[src1[row], col]; src1 elements are i32.
kernel void emel_get_rows_f32(device const uint8_t *src0 [[buffer(1)]],
                              device const uint8_t *src1 [[buffer(2)]],
                              device uint8_t *dst [[buffer(3)]],
                              constant const emel_params &p [[buffer(0)]],
                              uint tid [[thread_position_in_grid]]) {
    const uint64_t cols = p.src0.ne[0];
    const uint64_t rows = (uint64_t)p.i32[0];
    const uint64_t total = rows * cols;
    if (tid >= total) { return; }
    const uint64_t dst_row = tid / cols;
    const uint64_t col = tid % cols;
    uint64_t remaining = dst_row;
    uint64_t i0 = 0, i1 = 0, i2 = 0;
    bool active = true;
    for (uint32_t d = 0; d < 3; ++d) {
        const uint64_t dim = p.src1.ne[d] != 0u ? p.src1.ne[d] : 1u;
        if (active) {
            const uint64_t coord = remaining % dim;
            if (d == 0) { i0 = coord; } else if (d == 1) { i1 = coord; } else { i2 = coord; }
            remaining /= dim;
            active = p.src1.ne[d] != 0u;
        }
    }
    const int32_t row = emel_load_i32(src1, p.src1, i0, i1, i2, 0);
    emel_store_f32(dst, p.dst, col, dst_row, 0, 0,
                   emel_load_f32(src0, p.src0, col, (uint64_t)row, 0, 0));
}

// f16 codebook rows, f32 dst.
kernel void emel_get_rows_f16(device const uint8_t *src0 [[buffer(1)]],
                              device const uint8_t *src1 [[buffer(2)]],
                              device uint8_t *dst [[buffer(3)]],
                              constant const emel_params &p [[buffer(0)]],
                              uint tid [[thread_position_in_grid]]) {
    const uint64_t cols = p.src0.ne[0];
    const uint64_t rows = (uint64_t)p.i32[0];
    const uint64_t total = rows * cols;
    if (tid >= total) { return; }
    const uint64_t dst_row = tid / cols;
    const uint64_t col = tid % cols;
    uint64_t remaining = dst_row;
    uint64_t i0 = 0, i1 = 0, i2 = 0;
    bool active = true;
    for (uint32_t d = 0; d < 3; ++d) {
        const uint64_t dim = p.src1.ne[d] != 0u ? p.src1.ne[d] : 1u;
        if (active) {
            const uint64_t coord = remaining % dim;
            if (d == 0) { i0 = coord; } else if (d == 1) { i1 = coord; } else { i2 = coord; }
            remaining /= dim;
            active = p.src1.ne[d] != 0u;
        }
    }
    const int32_t row = emel_load_i32(src1, p.src1, i0, i1, i2, 0);
    emel_store_f32(dst, p.dst, col, dst_row, 0, 0,
                   emel_load_f16(src0, p.src0, col, (uint64_t)row, 0, 0));
}
)MSL";

constexpr const char *k_kernel_names[] = {
    "emel_mul_mat_f32",
    "emel_mul_mat_f16",
    "emel_mul_mat_q8_0",
    "emel_add",
    "emel_add_broadcast_row",
    "emel_unary",
    "emel_im2col_f32",
    "emel_im2col_f16",
    "emel_conv_transpose_1d_f32",
    "emel_conv_transpose_1d_f16",
    "emel_get_rows_f32",
    "emel_get_rows_f16",
};

constexpr uint32_t k_kernel_count =
    sizeof(k_kernel_names) / sizeof(k_kernel_names[0]);

// Threadgroup width for every kernel.
constexpr uint32_t k_threads_per_threadgroup = 256u;

} // namespace

struct metal_runtime::impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  std::array<id<MTLComputePipelineState>, k_kernel_count> pipelines = {};
  std::array<id<MTLBuffer>, k_pool_slice_count> slices = {};
  uint32_t free_mask = (1u << k_pool_slice_count) - 1u;
  bool available = false;

  ~impl() {
    @autoreleasepool {
      for (auto &slice : slices) {
        [slice release];
      }
      for (auto &pipeline : pipelines) {
        [pipeline release];
      }
      [library release];
      [queue release];
      [device release];
    }
  }
};

metal_runtime::metal_runtime() noexcept {
  try {
    auto owned = std::make_unique<impl>();
    @autoreleasepool {
      owned->device = MTLCreateSystemDefaultDevice();
      if (owned->device == nil) {
        impl_ = std::move(owned);
        return;
      }
      owned->queue = [owned->device newCommandQueue];
      if (owned->queue == nil) {
        impl_ = std::move(owned);
        return;
      }
      NSError *error = nil;
      owned->library = [owned->device
          newLibraryWithSource:[NSString stringWithUTF8String:k_msl_source]
                       options:nil
                         error:&error];
      if (owned->library == nil) {
        impl_ = std::move(owned);
        return;
      }
      for (uint32_t i = 0; i < k_kernel_count; ++i) {
        id<MTLFunction> function = [owned->library
            newFunctionWithName:[NSString
                                    stringWithUTF8String:k_kernel_names[i]]];
        if (function == nil) {
          impl_ = std::move(owned);
          return;
        }
        owned->pipelines[i] =
            [owned->device newComputePipelineStateWithFunction:function
                                                         error:&error];
        [function release];
        if (owned->pipelines[i] == nil) {
          impl_ = std::move(owned);
          return;
        }
      }
      for (uint32_t i = 0; i < k_pool_slice_count; ++i) {
        owned->slices[i] =
            [owned->device newBufferWithLength:k_pool_slice_capacity_bytes
                                       options:MTLResourceStorageModeShared];
        if (owned->slices[i] == nil) {
          impl_ = std::move(owned);
          return;
        }
      }
      owned->available = true;
    }
    impl_ = std::move(owned);
  } catch (...) {
    impl_ = nullptr;
  }
}

metal_runtime::~metal_runtime() = default;

bool metal_runtime::available() const noexcept {
  return impl_ != nullptr && impl_->available;
}

bool metal_runtime::acquire_slices(uint32_t (&out)[3]) noexcept {
  if (!available()) {
    return false;
  }
  uint32_t acquired = 0u;
  for (uint32_t i = 0; i < k_pool_slice_count && acquired < 3u; ++i) {
    if ((impl_->free_mask & (1u << i)) != 0u) {
      impl_->free_mask &= ~(1u << i);
      out[acquired++] = i;
    }
  }
  if (acquired < 3u) {
    for (uint32_t a = 0; a < acquired; ++a) {
      impl_->free_mask |= (1u << out[a]);
    }
    return false;
  }
  return true;
}

void metal_runtime::release_slices(const uint32_t (&slices)[3]) noexcept {
  if (impl_ == nullptr) {
    return;
  }
  for (uint32_t a = 0; a < 3u; ++a) {
    impl_->free_mask |= (1u << slices[a]);
  }
}

void *metal_runtime::slice_contents(uint32_t index) noexcept {
  if (impl_ == nullptr || index >= k_pool_slice_count) {
    return nullptr;
  }
  return [impl_->slices[index] contents];
}

bool metal_runtime::launch(kernel_id kernel, const shader_params &params,
                           const uint32_t (&slices)[3],
                           uint32_t threads) noexcept {
  if (!available()) {
    return false;
  }
  const uint32_t kernel_index = static_cast<uint32_t>(kernel);
  if (kernel_index >= k_kernel_count || impl_->pipelines[kernel_index] == nil) {
    return false;
  }
  @autoreleasepool {
    // Metal's API requires one MTLCommandBuffer + MTLComputeCommandEncoder
    // per dispatch. These are the ONLY per-dispatch allocations and they are
    // created inside this same-scope autorelease pool, so they are drained
    // before the action returns: nothing outlives the RTC boundary and net
    // growth is zero. This is the explicitly approved, scope-drained
    // exception to the no-allocation-during-dispatch rule; all data staging
    // uses the preallocated slice pool (no C++ heap allocation here).
    id<MTLCommandBuffer> command_buffer = [impl_->queue commandBuffer];
    if (command_buffer == nil) {
      return false;
    }
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoder];
    if (encoder == nil) {
      return false;
    }
    [encoder setComputePipelineState:impl_->pipelines[kernel_index]];
    [encoder setBytes:&params length:sizeof(params) atIndex:0];
    [encoder setBuffer:impl_->slices[slices[0]] offset:0 atIndex:1];
    [encoder setBuffer:impl_->slices[slices[1]] offset:0 atIndex:2];
    [encoder setBuffer:impl_->slices[slices[2]] offset:0 atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(threads, 1u, 1u)
        threadsPerThreadgroup:MTLSizeMake(k_threads_per_threadgroup, 1u, 1u)];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
  }
  return true;
}

//------------------------------------------------------------------------------//
// Op compute helpers.
//------------------------------------------------------------------------------//

bool run_mul_mat_f32(
    metal_runtime &rt,
    const ::emel::kernel::event::op_mul_mat &request) noexcept {
  if (!rt.available()) {
    return false;
  }
  uint32_t slices[3] = {};
  if (!rt.acquire_slices(slices)) {
    return false;
  }
  bool ok = false;
  do {
    uint64_t bytes = 0u;
    if (!stage_tensor(rt, slices[0], request.src0, bytes) ||
        !stage_tensor(rt, slices[1], request.src1, bytes)) {
      break;
    }
    shader_params params = {};
    params.src0 = make_shader_tensor(request.src0);
    params.src1 = make_shader_tensor(request.src1);
    params.dst = make_shader_tensor(request.dst);
    const uint64_t m = request.src0.ne[1];
    const uint64_t n = request.src1.ne[0];
    if (!rt.launch(kernel_id::mul_mat_f32, params, slices,
                   grid_threads(m * n))) {
      break;
    }
    ok = readback_tensor(rt, slices[2], request.dst);
  } while (false);
  rt.release_slices(slices);
  return ok;
}

bool run_mul_mat_f16(
    metal_runtime &rt,
    const ::emel::kernel::event::op_mul_mat &request) noexcept {
  if (!rt.available()) {
    return false;
  }
  uint32_t slices[3] = {};
  if (!rt.acquire_slices(slices)) {
    return false;
  }
  bool ok = false;
  do {
    uint64_t bytes = 0u;
    if (!stage_tensor(rt, slices[0], request.src0, bytes) ||
        !stage_tensor(rt, slices[1], request.src1, bytes)) {
      break;
    }
    shader_params params = {};
    params.src0 = make_shader_tensor(request.src0);
    params.src1 = make_shader_tensor(request.src1);
    params.dst = make_shader_tensor(request.dst);
    const uint64_t m = request.src0.ne[1];
    const uint64_t n = request.src1.ne[1];
    if (!rt.launch(kernel_id::mul_mat_f16, params, slices,
                   grid_threads(m * n))) {
      break;
    }
    ok = readback_tensor(rt, slices[2], request.dst);
  } while (false);
  rt.release_slices(slices);
  return ok;
}

bool run_mul_mat_q8_0(
    metal_runtime &rt,
    const ::emel::kernel::event::op_mul_mat &request) noexcept {
  if (!rt.available()) {
    return false;
  }
  uint32_t slices[3] = {};
  if (!rt.acquire_slices(slices)) {
    return false;
  }
  bool ok = false;
  do {
    uint64_t bytes = 0u;
    if (!stage_tensor(rt, slices[0], request.src0, bytes) ||
        !stage_tensor(rt, slices[1], request.src1, bytes)) {
      break;
    }
    shader_params params = {};
    params.src0 = make_shader_tensor(request.src0);
    params.src1 = make_shader_tensor(request.src1);
    params.dst = make_shader_tensor(request.dst);
    const uint64_t m = request.src0.ne[1];
    const uint64_t n = request.src1.ne[0];
    if (!rt.launch(kernel_id::mul_mat_q8_0, params, slices,
                   grid_threads(m * n))) {
      break;
    }
    ok = readback_tensor(rt, slices[2], request.dst);
  } while (false);
  rt.release_slices(slices);
  return ok;
}

bool run_add(metal_runtime &rt,
             const ::emel::kernel::event::op_add &request) noexcept {
  if (!rt.available()) {
    return false;
  }
  uint32_t slices[3] = {};
  if (!rt.acquire_slices(slices)) {
    return false;
  }
  bool ok = false;
  do {
    uint64_t bytes = 0u;
    if (!stage_tensor(rt, slices[0], request.src0, bytes) ||
        !stage_tensor(rt, slices[1], request.src1, bytes)) {
      break;
    }
    shader_params params = {};
    params.src0 = make_shader_tensor(request.src0);
    params.src1 = make_shader_tensor(request.src1);
    params.dst = make_shader_tensor(request.dst);
    params.i32[0] = static_cast<int32_t>(tensor_element_count(request.dst));
    if (!rt.launch(kernel_id::add, params, slices,
                   grid_threads(tensor_element_count(request.dst)))) {
      break;
    }
    ok = readback_tensor(rt, slices[2], request.dst);
  } while (false);
  rt.release_slices(slices);
  return ok;
}

bool run_add_broadcast_row(
    metal_runtime &rt, const ::emel::kernel::event::op_add &request) noexcept {
  if (!rt.available()) {
    return false;
  }
  uint32_t slices[3] = {};
  if (!rt.acquire_slices(slices)) {
    return false;
  }
  bool ok = false;
  do {
    uint64_t bytes = 0u;
    if (!stage_tensor(rt, slices[0], request.src0, bytes) ||
        !stage_tensor(rt, slices[1], request.src1, bytes)) {
      break;
    }
    shader_params params = {};
    params.src0 = make_shader_tensor(request.src0);
    params.src1 = make_shader_tensor(request.src1);
    params.dst = make_shader_tensor(request.dst);
    const uint64_t cols = request.dst.ne[0];
    params.i32[0] = static_cast<int32_t>(cols);
    params.i32[1] = static_cast<int32_t>(tensor_element_count(request.dst));
    if (!rt.launch(kernel_id::add_broadcast_row, params, slices,
                   grid_threads(tensor_element_count(request.dst)))) {
      break;
    }
    ok = readback_tensor(rt, slices[2], request.dst);
  } while (false);
  rt.release_slices(slices);
  return ok;
}

bool run_unary(metal_runtime &rt,
               const ::emel::kernel::event::op_unary &request) noexcept {
  if (!rt.available()) {
    return false;
  }
  uint32_t slices[3] = {};
  if (!rt.acquire_slices(slices)) {
    return false;
  }
  bool ok = false;
  do {
    uint64_t bytes = 0u;
    if (!stage_tensor(rt, slices[0], request.src0, bytes)) {
      break;
    }
    shader_params params = {};
    params.src0 = make_shader_tensor(request.src0);
    params.dst = make_shader_tensor(request.dst);
    params.i32[0] = static_cast<int32_t>(tensor_element_count(request.dst));
    params.i32[1] = static_cast<int32_t>(request.subop);
    if (!rt.launch(kernel_id::unary, params, slices,
                   grid_threads(tensor_element_count(request.dst)))) {
      break;
    }
    ok = readback_tensor(rt, slices[2], request.dst);
  } while (false);
  rt.release_slices(slices);
  return ok;
}

bool run_im2col_f32(metal_runtime &rt,
                    const ::emel::kernel::event::op_im2col &request) noexcept {
  if (!rt.available()) {
    return false;
  }
  uint32_t slices[3] = {};
  if (!rt.acquire_slices(slices)) {
    return false;
  }
  bool ok = false;
  do {
    uint64_t bytes = 0u;
    if (!stage_tensor(rt, slices[0], request.src0, bytes) ||
        !stage_tensor(rt, slices[1], request.src1, bytes)) {
      break;
    }
    shader_params params = {};
    params.src0 = make_shader_tensor(request.src0);
    params.src1 = make_shader_tensor(request.src1);
    params.dst = make_shader_tensor(request.dst);
    ::emel::kernel::detail::im2col_op_params op_params = {};
    (void)::emel::kernel::detail::read_im2col_params(request, op_params);
    params.i32[0] = op_params.s0;
    params.i32[2] = op_params.p0;
    params.i32[4] = op_params.d0;
    const uint64_t threads = tensor_element_count(request.dst) /
                             (request.dst.ne[0] != 0u ? request.dst.ne[0] : 1u);
    if (!rt.launch(kernel_id::im2col_f32, params, slices,
                   grid_threads(threads))) {
      break;
    }
    ok = readback_tensor(rt, slices[2], request.dst);
  } while (false);
  rt.release_slices(slices);
  return ok;
}

bool run_im2col_f16(metal_runtime &rt,
                    const ::emel::kernel::event::op_im2col &request) noexcept {
  if (!rt.available()) {
    return false;
  }
  uint32_t slices[3] = {};
  if (!rt.acquire_slices(slices)) {
    return false;
  }
  bool ok = false;
  do {
    uint64_t bytes = 0u;
    if (!stage_tensor(rt, slices[0], request.src0, bytes) ||
        !stage_tensor(rt, slices[1], request.src1, bytes)) {
      break;
    }
    shader_params params = {};
    params.src0 = make_shader_tensor(request.src0);
    params.src1 = make_shader_tensor(request.src1);
    params.dst = make_shader_tensor(request.dst);
    ::emel::kernel::detail::im2col_op_params op_params = {};
    (void)::emel::kernel::detail::read_im2col_params(request, op_params);
    params.i32[0] = op_params.s0;
    params.i32[2] = op_params.p0;
    params.i32[4] = op_params.d0;
    const uint64_t threads = tensor_element_count(request.dst) /
                             (request.dst.ne[0] != 0u ? request.dst.ne[0] : 1u);
    if (!rt.launch(kernel_id::im2col_f16, params, slices,
                   grid_threads(threads))) {
      break;
    }
    ok = readback_tensor(rt, slices[2], request.dst);
  } while (false);
  rt.release_slices(slices);
  return ok;
}

bool run_conv_transpose_1d_f32(
    metal_runtime &rt,
    const ::emel::kernel::event::op_conv_transpose_1d &request) noexcept {
  if (!rt.available()) {
    return false;
  }
  uint32_t slices[3] = {};
  if (!rt.acquire_slices(slices)) {
    return false;
  }
  bool ok = false;
  do {
    uint64_t bytes = 0u;
    if (!stage_tensor(rt, slices[0], request.src0, bytes) ||
        !stage_tensor(rt, slices[1], request.src1, bytes)) {
      break;
    }
    shader_params params = {};
    params.src0 = make_shader_tensor(request.src0);
    params.src1 = make_shader_tensor(request.src1);
    params.dst = make_shader_tensor(request.dst);
    int32_t s0 = 0;
    (void)::emel::kernel::detail::read_op_param_i32(
        request.op_params.data(), request.op_params_size, 0u, s0);
    params.i32[0] = s0;
    const uint64_t threads =
        request.dst.ne[0] * (request.dst.ne[1] != 0u ? request.dst.ne[1] : 1u);
    if (!rt.launch(kernel_id::conv_transpose_1d_f32, params, slices,
                   grid_threads(threads))) {
      break;
    }
    ok = readback_tensor(rt, slices[2], request.dst);
  } while (false);
  rt.release_slices(slices);
  return ok;
}

bool run_conv_transpose_1d_f16(
    metal_runtime &rt,
    const ::emel::kernel::event::op_conv_transpose_1d &request) noexcept {
  if (!rt.available()) {
    return false;
  }
  uint32_t slices[3] = {};
  if (!rt.acquire_slices(slices)) {
    return false;
  }
  bool ok = false;
  do {
    uint64_t bytes = 0u;
    if (!stage_tensor(rt, slices[0], request.src0, bytes) ||
        !stage_tensor(rt, slices[1], request.src1, bytes)) {
      break;
    }
    shader_params params = {};
    params.src0 = make_shader_tensor(request.src0);
    params.src1 = make_shader_tensor(request.src1);
    params.dst = make_shader_tensor(request.dst);
    int32_t s0 = 0;
    (void)::emel::kernel::detail::read_op_param_i32(
        request.op_params.data(), request.op_params_size, 0u, s0);
    params.i32[0] = s0;
    const uint64_t threads =
        request.dst.ne[0] * (request.dst.ne[1] != 0u ? request.dst.ne[1] : 1u);
    if (!rt.launch(kernel_id::conv_transpose_1d_f16, params, slices,
                   grid_threads(threads))) {
      break;
    }
    ok = readback_tensor(rt, slices[2], request.dst);
  } while (false);
  rt.release_slices(slices);
  return ok;
}

bool run_get_rows_f32(
    metal_runtime &rt,
    const ::emel::kernel::event::op_get_rows &request) noexcept {
  if (!rt.available()) {
    return false;
  }
  uint32_t slices[3] = {};
  if (!rt.acquire_slices(slices)) {
    return false;
  }
  bool ok = false;
  do {
    uint64_t bytes = 0u;
    if (!stage_tensor(rt, slices[0], request.src0, bytes) ||
        !stage_tensor(rt, slices[1], request.src1, bytes)) {
      break;
    }
    shader_params params = {};
    params.src0 = make_shader_tensor(request.src0);
    params.src1 = make_shader_tensor(request.src1);
    params.dst = make_shader_tensor(request.dst);
    params.i32[0] = static_cast<int32_t>(tensor_element_count(request.src1));
    if (!rt.launch(kernel_id::get_rows_f32, params, slices,
                   grid_threads(tensor_element_count(request.src1) *
                                request.src0.ne[0]))) {
      break;
    }
    ok = readback_tensor(rt, slices[2], request.dst);
  } while (false);
  rt.release_slices(slices);
  return ok;
}

bool run_get_rows_f16(
    metal_runtime &rt,
    const ::emel::kernel::event::op_get_rows &request) noexcept {
  if (!rt.available()) {
    return false;
  }
  uint32_t slices[3] = {};
  if (!rt.acquire_slices(slices)) {
    return false;
  }
  bool ok = false;
  do {
    uint64_t bytes = 0u;
    if (!stage_tensor(rt, slices[0], request.src0, bytes) ||
        !stage_tensor(rt, slices[1], request.src1, bytes)) {
      break;
    }
    shader_params params = {};
    params.src0 = make_shader_tensor(request.src0);
    params.src1 = make_shader_tensor(request.src1);
    params.dst = make_shader_tensor(request.dst);
    params.i32[0] = static_cast<int32_t>(tensor_element_count(request.src1));
    if (!rt.launch(kernel_id::get_rows_f16, params, slices,
                   grid_threads(tensor_element_count(request.src1) *
                                request.src0.ne[0]))) {
      break;
    }
    ok = readback_tensor(rt, slices[2], request.dst);
  } while (false);
  rt.release_slices(slices);
  return ok;
}

} // namespace emel::kernel::metal::detail

#else // !defined(__APPLE__)

namespace emel::kernel::metal::detail {

struct metal_runtime::impl {};

metal_runtime::metal_runtime() noexcept = default;
metal_runtime::~metal_runtime() = default;

bool metal_runtime::available() const noexcept { return false; }

bool metal_runtime::acquire_slices(uint32_t (&)[3]) noexcept { return false; }

void metal_runtime::release_slices(const uint32_t (&)[3]) noexcept {}

void *metal_runtime::slice_contents(uint32_t) noexcept { return nullptr; }

bool metal_runtime::launch(kernel_id, const shader_params &,
                           const uint32_t (&)[3], uint32_t) noexcept {
  return false;
}

bool run_mul_mat_f32(metal_runtime &,
                     const ::emel::kernel::event::op_mul_mat &) noexcept {
  return false;
}

bool run_mul_mat_f16(metal_runtime &,
                     const ::emel::kernel::event::op_mul_mat &) noexcept {
  return false;
}

bool run_mul_mat_q8_0(metal_runtime &,
                      const ::emel::kernel::event::op_mul_mat &) noexcept {
  return false;
}

bool run_add(metal_runtime &, const ::emel::kernel::event::op_add &) noexcept {
  return false;
}

bool run_add_broadcast_row(metal_runtime &,
                           const ::emel::kernel::event::op_add &) noexcept {
  return false;
}

bool run_unary(metal_runtime &,
               const ::emel::kernel::event::op_unary &) noexcept {
  return false;
}

bool run_im2col_f32(metal_runtime &,
                    const ::emel::kernel::event::op_im2col &) noexcept {
  return false;
}

bool run_im2col_f16(metal_runtime &,
                    const ::emel::kernel::event::op_im2col &) noexcept {
  return false;
}

bool run_conv_transpose_1d_f32(
    metal_runtime &,
    const ::emel::kernel::event::op_conv_transpose_1d &) noexcept {
  return false;
}

bool run_conv_transpose_1d_f16(
    metal_runtime &,
    const ::emel::kernel::event::op_conv_transpose_1d &) noexcept {
  return false;
}

bool run_get_rows_f32(metal_runtime &,
                      const ::emel::kernel::event::op_get_rows &) noexcept {
  return false;
}

bool run_get_rows_f16(metal_runtime &,
                      const ::emel::kernel::event::op_get_rows &) noexcept {
  return false;
}

} // namespace emel::kernel::metal::detail

#endif // defined(__APPLE__)
