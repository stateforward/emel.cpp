#include "emel/speech/codec/mimi/detail.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

#include "emel/kernel/detail.hpp"
#include "emel/model/moshi/detail.hpp"

namespace emel::speech::codec::mimi::detail {

namespace {

namespace kev = emel::kernel::event;

constexpr float k_layer_norm_eps = 1e-5f;

const emel::model::data::tensor_record *
find_tensor(const emel::model::data &model_data,
            const std::string_view name) noexcept {
  for (uint32_t index = 0u; index < model_data.n_tensors; ++index) {
    const auto &tensor = model_data.tensors[index];
    if (emel::model::tensor_name_view(model_data, tensor) == name) {
      return &tensor;
    }
  }
  return nullptr;
}

const emel::model::data::tensor_record *
find_layer_tensor(const emel::model::data &model_data, const char *format,
                  const int32_t index, const char *suffix) noexcept {
  char buffer[96] = {};
  const int written =
      std::snprintf(buffer, sizeof(buffer), format, index, suffix);
  if (written <= 0 || static_cast<size_t>(written) >= sizeof(buffer)) {
    return nullptr;
  }
  return find_tensor(model_data,
                     std::string_view{buffer, static_cast<size_t>(written)});
}

bool tensor_is_float(const emel::model::data::tensor_record &tensor) noexcept {
  const auto code = static_cast<uint8_t>(tensor.type);
  return code == emel::kernel::detail::dtype_f32 ||
         code == emel::kernel::detail::dtype_f16;
}

uint64_t
tensor_elements(const emel::model::data::tensor_record &tensor) noexcept {
  uint64_t count = 1;
  for (int32_t dim = 0; dim < tensor.n_dims; ++dim) {
    count *= static_cast<uint64_t>(tensor.dims[static_cast<size_t>(dim)]);
  }
  return count;
}

// Arena cursor for one-time prepare copies.
struct arena_cursor {
  std::span<float> arena = {};
  uint64_t used = 0;

  float *take(const uint64_t floats) noexcept {
    if (used + floats > arena.size()) {
      return nullptr;
    }
    float *out = arena.data() + used;
    used += floats;
    return out;
  }
};

// Reads one element of an f32/f16 source tensor as f32.
float source_value(const emel::model::data::tensor_record &tensor,
                   const uint64_t index) noexcept {
  const auto code = static_cast<uint8_t>(tensor.type);
  if (code == emel::kernel::detail::dtype_f16) {
    uint16_t bits = 0;
    std::memcpy(
        &bits, static_cast<const uint8_t *>(tensor.data) + index * sizeof(bits),
        sizeof(bits));
    return emel::kernel::detail::quant::fp16_to_fp32(bits);
  }
  float out = 0.0f;
  std::memcpy(&out,
              static_cast<const uint8_t *>(tensor.data) + index * sizeof(out),
              sizeof(out));
  return out;
}

// Copies a [dim] vector (norm weights, biases, scales) into the arena.
const float *prepare_vector(arena_cursor &cursor,
                            const emel::model::data::tensor_record *tensor,
                            const int64_t expected) noexcept {
  if (tensor == nullptr || !tensor_is_float(*tensor) ||
      tensor_elements(*tensor) != static_cast<uint64_t>(expected)) {
    return nullptr;
  }
  float *out = cursor.take(static_cast<uint64_t>(expected));
  if (out == nullptr) {
    return nullptr;
  }
  for (int64_t index = 0; index < expected; ++index) {
    out[index] = source_value(*tensor, static_cast<uint64_t>(index));
  }
  return out;
}

// Prepares a linear weight stored [in, out] (in fastest) as its transpose
// [out, in] (out fastest) so mul_mat consumes it as src1 without runtime
// transposes: prepared[out + in * out_count].
const float *prepare_linear(arena_cursor &cursor,
                            const emel::model::data::tensor_record *tensor,
                            const int64_t in_count,
                            const int64_t out_count) noexcept {
  if (tensor == nullptr || !tensor_is_float(*tensor) ||
      tensor_elements(*tensor) !=
          static_cast<uint64_t>(in_count) * static_cast<uint64_t>(out_count)) {
    return nullptr;
  }
  float *out = cursor.take(static_cast<uint64_t>(in_count) *
                           static_cast<uint64_t>(out_count));
  if (out == nullptr) {
    return nullptr;
  }
  for (int64_t o = 0; o < out_count; ++o) {
    for (int64_t i = 0; i < in_count; ++i) {
      out[o + i * out_count] =
          source_value(*tensor, static_cast<uint64_t>(i + o * in_count));
    }
  }
  return out;
}

// Prepares a conv kernel stored [taps, in, out] as the column-major GEMM
// operand [out, in*taps]: prepared[out + (in*taps + tap) * out_count].
// Copies raw f16 halfs verbatim into the arena (ggml reshape layout is the
// source layout, so no repacking). Returns nullptr when the source is not
// f16 or capacity is exceeded.
const uint16_t *prepare_raw_f16(arena_cursor &cursor,
                                const emel::model::data::tensor_record *tensor,
                                const uint64_t expected_halfs) noexcept {
  if (tensor == nullptr ||
      static_cast<uint8_t>(tensor->type) != emel::kernel::detail::dtype_f16 ||
      tensor_elements(*tensor) != expected_halfs || tensor->data == nullptr) {
    return nullptr;
  }
  float *slot = cursor.take((expected_halfs + 1u) / 2u);
  if (slot == nullptr) {
    return nullptr;
  }
  std::memcpy(slot, tensor->data, expected_halfs * sizeof(uint16_t));
  return reinterpret_cast<const uint16_t *>(slot);
}

const float *prepare_conv_gemm(arena_cursor &cursor,
                               const emel::model::data::tensor_record *tensor,
                               const int64_t taps, const int64_t in_count,
                               const int64_t out_count) noexcept {
  if (tensor == nullptr || !tensor_is_float(*tensor) ||
      tensor_elements(*tensor) != static_cast<uint64_t>(taps) *
                                      static_cast<uint64_t>(in_count) *
                                      static_cast<uint64_t>(out_count)) {
    return nullptr;
  }
  float *out = cursor.take(static_cast<uint64_t>(taps) *
                           static_cast<uint64_t>(in_count) *
                           static_cast<uint64_t>(out_count));
  if (out == nullptr) {
    return nullptr;
  }
  for (int64_t o = 0; o < out_count; ++o) {
    for (int64_t i = 0; i < in_count; ++i) {
      for (int64_t t = 0; t < taps; ++t) {
        out[o + (i * taps + t) * out_count] = source_value(
            *tensor, static_cast<uint64_t>(t + i * taps + o * taps * in_count));
      }
    }
  }
  return out;
}

// Prepares a transposed-conv kernel stored [taps, out, in] verbatim as f32
// (the op_conv_transpose_1d kernel consumes the reference layout directly).
const float *
prepare_conv_transpose(arena_cursor &cursor,
                       const emel::model::data::tensor_record *tensor,
                       const uint64_t elements) noexcept {
  if (tensor == nullptr || !tensor_is_float(*tensor) ||
      tensor_elements(*tensor) != elements) {
    return nullptr;
  }
  float *out = cursor.take(elements);
  if (out == nullptr) {
    return nullptr;
  }
  for (uint64_t index = 0; index < elements; ++index) {
    out[index] = source_value(*tensor, index);
  }
  return out;
}

kev::tensor_view make_view(const float *data, const uint64_t ne0,
                           const uint64_t ne1 = 1, const uint64_t ne2 = 1) {
  kev::tensor_view view{};
  view.data = data;
  view.type = kev::dtype::f32;
  view.ne = {ne0, ne1, ne2, 1};
  view.nb[0] = sizeof(float);
  view.nb[1] = view.nb[0] * ne0;
  view.nb[2] = view.nb[1] * ne1;
  view.nb[3] = view.nb[2] * ne2;
  return view;
}

kev::tensor_view make_f16_view(const uint16_t *data, const uint64_t ne0,
                               const uint64_t ne1 = 1) {
  kev::tensor_view view{};
  view.data = data;
  view.type = kev::dtype::f16;
  view.ne = {ne0, ne1, 1, 1};
  view.nb[0] = sizeof(uint16_t);
  view.nb[1] = view.nb[0] * ne0;
  view.nb[2] = view.nb[1] * ne1;
  view.nb[3] = view.nb[2];
  return view;
}

kev::tensor_view_mut make_f16_view_mut(uint16_t *data, const uint64_t ne0,
                                       const uint64_t ne1 = 1) {
  kev::tensor_view_mut view{};
  view.data = data;
  view.type = kev::dtype::f16;
  view.ne = {ne0, ne1, 1, 1};
  view.nb[0] = sizeof(uint16_t);
  view.nb[1] = view.nb[0] * ne0;
  view.nb[2] = view.nb[1] * ne1;
  view.nb[3] = view.nb[2];
  return view;
}

kev::tensor_view make_strided_view(const float *data, const uint64_t ne0,
                                   const uint64_t ne1, const uint64_t nb0_bytes,
                                   const uint64_t nb1_bytes) {
  kev::tensor_view view{};
  view.data = data;
  view.type = kev::dtype::f32;
  view.ne = {ne0, ne1, 1, 1};
  view.nb = {nb0_bytes, nb1_bytes, nb1_bytes * ne1, nb1_bytes * ne1};
  return view;
}

kev::tensor_view_mut make_view_mut(float *data, const uint64_t ne0,
                                   const uint64_t ne1 = 1,
                                   const uint64_t ne2 = 1) {
  kev::tensor_view_mut view{};
  view.data = data;
  view.type = kev::dtype::f32;
  view.ne = {ne0, ne1, ne2, 1};
  view.nb[0] = sizeof(float);
  view.nb[1] = view.nb[0] * ne0;
  view.nb[2] = view.nb[1] * ne1;
  view.nb[3] = view.nb[2] * ne2;
  return view;
}

kev::tensor_view_mut make_strided_view_mut(float *data, const uint64_t ne0,
                                           const uint64_t ne1,
                                           const uint64_t nb0_bytes,
                                           const uint64_t nb1_bytes) {
  kev::tensor_view_mut view{};
  view.data = data;
  view.type = kev::dtype::f32;
  view.ne = {ne0, ne1, 1, 1};
  view.nb = {nb0_bytes, nb1_bytes, nb1_bytes * ne1, nb1_bytes * ne1};
  return view;
}

bool dispatch_copy(codec_runtime &runtime, const kev::tensor_view &src,
                   const kev::tensor_view_mut &dst) noexcept {
  kev::op_dup ev{};
  ev.src0 = src;
  ev.dst = dst;
  return runtime.kernel.process_event(ev);
}

bool dispatch_add(codec_runtime &runtime, const kev::tensor_view &lhs,
                  const kev::tensor_view &rhs,
                  const kev::tensor_view_mut &dst) noexcept {
  kev::op_add ev{};
  ev.src0 = lhs;
  ev.src1 = rhs;
  ev.dst = dst;
  return runtime.kernel.process_event(ev);
}

bool dispatch_mul(codec_runtime &runtime, const kev::tensor_view &lhs,
                  const kev::tensor_view &rhs,
                  const kev::tensor_view_mut &dst) noexcept {
  kev::op_mul ev{};
  ev.src0 = lhs;
  ev.src1 = rhs;
  ev.dst = dst;
  return runtime.kernel.process_event(ev);
}

bool dispatch_sub(codec_runtime &runtime, const kev::tensor_view &lhs,
                  const kev::tensor_view &rhs,
                  const kev::tensor_view_mut &dst) noexcept {
  kev::op_sub ev{};
  ev.src0 = lhs;
  ev.src1 = rhs;
  ev.dst = dst;
  return runtime.kernel.process_event(ev);
}

bool dispatch_mul_mat(codec_runtime &runtime, const kev::tensor_view &src0,
                      const kev::tensor_view &src1,
                      const kev::tensor_view_mut &dst) noexcept {
  kev::op_mul_mat ev{};
  ev.src0 = src0;
  ev.src1 = src1;
  ev.dst = dst;
  return runtime.kernel.process_event(ev);
}

bool dispatch_unary(codec_runtime &runtime, const kev::unary_subop subop,
                    const kev::tensor_view &src,
                    const kev::tensor_view_mut &dst) noexcept {
  kev::op_unary ev{};
  ev.src0 = src;
  ev.dst = dst;
  ev.subop = subop;
  return runtime.kernel.process_event(ev);
}

template <class event_type>
void set_param_i32(event_type &ev, const uint32_t slot,
                   const int32_t value) noexcept {
  std::memcpy(ev.op_params.data() + slot * sizeof(int32_t), &value,
              sizeof(value));
  const uint32_t needed = (slot + 1u) * sizeof(int32_t);
  ev.op_params_size = ev.op_params_size < needed ? needed : ev.op_params_size;
}

template <class event_type>
void set_param_f32(event_type &ev, const uint32_t slot,
                   const float value) noexcept {
  std::memcpy(ev.op_params.data() + slot * sizeof(float), &value,
              sizeof(value));
  const uint32_t needed = (slot + 1u) * sizeof(float);
  ev.op_params_size = ev.op_params_size < needed ? needed : ev.op_params_size;
}

bool dispatch_layer_norm(codec_runtime &runtime, const float *weight,
                         const float *bias, float *data, const uint64_t dim,
                         const uint64_t rows) noexcept {
  kev::op_norm norm_ev{};
  norm_ev.src0 = make_view(data, dim, rows);
  norm_ev.dst = make_view_mut(data, dim, rows);
  set_param_f32(norm_ev, 0u, k_layer_norm_eps);
  return runtime.kernel.process_event(norm_ev) &&
         dispatch_mul(runtime, make_view(data, dim, rows),
                      make_view(weight, dim), make_view_mut(data, dim, rows)) &&
         dispatch_add(runtime, make_view(data, dim, rows), make_view(bias, dim),
                      make_view_mut(data, dim, rows));
}

// Workspace partition for one frame. All slices point into the caller-owned
// workspace span; sizes were fixed by required_workspace_floats.
struct workspace_plan {
  std::span<float> channel_major = {};
  std::span<float> columns = {};
  std::span<float> scratch_a = {};
  std::span<float> scratch_b = {};
  std::span<float> scratch_c = {};
};

bool carve_workspace(const std::span<float> workspace,
                     const uint64_t channel_major, const uint64_t columns,
                     const uint64_t scratch,
                     workspace_plan &plan_out) noexcept {
  const uint64_t total = channel_major + columns + 3u * scratch;
  if (workspace.size() < total) {
    return false;
  }
  uint64_t offset = 0;
  plan_out.channel_major = workspace.subspan(offset, channel_major);
  offset += channel_major;
  plan_out.columns = workspace.subspan(offset, columns);
  offset += columns;
  plan_out.scratch_a = workspace.subspan(offset, scratch);
  offset += scratch;
  plan_out.scratch_b = workspace.subspan(offset, scratch);
  offset += scratch;
  plan_out.scratch_c = workspace.subspan(offset, scratch);
  return true;
}

// Per-model buffer bounds computed once at bind (also reused by the sizing
// functions): the widest channel-major stage buffer, the widest im2col
// column buffer, and the widest time-major frame across all stages.
struct codec_extents {
  uint64_t channel_major = 0;
  uint64_t columns = 0;
  uint64_t frame = 0;
  uint64_t state = 0;
};

void account_conv(codec_extents &extents, const int64_t in_channels,
                  const int64_t out_channels, const int64_t taps,
                  const int64_t stride, const int64_t in_length,
                  const bool transposed) noexcept {
  const int64_t state_len = taps - stride;
  if (transposed) {
    const int64_t lout_full = (in_length - 1) * stride + taps;
    extents.channel_major = std::max<uint64_t>(
        extents.channel_major,
        static_cast<uint64_t>(in_channels) * static_cast<uint64_t>(in_length));
    extents.columns = std::max<uint64_t>(extents.columns,
                                         static_cast<uint64_t>(out_channels) *
                                             static_cast<uint64_t>(lout_full));
    extents.state +=
        static_cast<uint64_t>(out_channels) * static_cast<uint64_t>(lout_full);
    extents.frame = std::max<uint64_t>(
        extents.frame, static_cast<uint64_t>(out_channels) *
                           static_cast<uint64_t>(in_length * stride));
  } else {
    const int64_t padded = state_len + in_length;
    const int64_t out_length = in_length / stride;
    extents.channel_major = std::max<uint64_t>(
        extents.channel_major,
        static_cast<uint64_t>(in_channels) * static_cast<uint64_t>(padded));
    extents.columns = std::max<uint64_t>(extents.columns,
                                         static_cast<uint64_t>(in_channels) *
                                             static_cast<uint64_t>(taps) *
                                             static_cast<uint64_t>(out_length));
    extents.state +=
        static_cast<uint64_t>(in_channels) * static_cast<uint64_t>(state_len);
    extents.frame = std::max<uint64_t>(extents.frame,
                                       static_cast<uint64_t>(out_channels) *
                                           static_cast<uint64_t>(out_length));
  }
  extents.frame =
      std::max<uint64_t>(extents.frame, static_cast<uint64_t>(in_channels) *
                                            static_cast<uint64_t>(in_length));
}

} // namespace

//------------------------------------------------------------------------------//
// Compute helpers.
//------------------------------------------------------------------------------//

template <bool conv_f16>
bool compute_streaming_conv(codec_runtime &runtime, const conv_weights &conv,
                            codec_streaming_state &state, frame_buffer &io,
                            std::span<float> workspace) noexcept {
  runtime.kernel.set_kind(runtime.kernel_kind);
  const int64_t in_channels = conv.in_channels;
  const int64_t out_channels = conv.out_channels;
  const int64_t taps = conv.taps;
  const int64_t stride = conv.stride;
  const int64_t length = io.length;
  const int64_t state_len = taps - stride;
  const int64_t padded = state_len + length;
  const int64_t out_length = length / stride;
  if (io.channels != in_channels || length <= 0 || out_length <= 0 ||
      conv.frame_length != length) {
    return false;
  }

  workspace_plan plan{};
  const uint64_t channel_major_floats =
      static_cast<uint64_t>(in_channels) * static_cast<uint64_t>(padded);
  const uint64_t column_floats = static_cast<uint64_t>(in_channels) *
                                 static_cast<uint64_t>(taps) *
                                 static_cast<uint64_t>(out_length);
  if (!carve_workspace(workspace, channel_major_floats, column_floats, 0u,
                       plan)) {
    return false;
  }

  float *channel_major = plan.channel_major.data();
  float *state_slice = state.arena.data() + conv.state_offset;
  const uint64_t row_bytes = static_cast<uint64_t>(padded) * sizeof(float);

  // state -> [c][0..state_len), input -> [c][state_len..padded)
  const bool staged =
      (state_len == 0 ||
       dispatch_copy(runtime,
                     make_view(state_slice, static_cast<uint64_t>(state_len),
                               static_cast<uint64_t>(in_channels)),
                     make_strided_view_mut(channel_major,
                                           static_cast<uint64_t>(state_len),
                                           static_cast<uint64_t>(in_channels),
                                           sizeof(float), row_bytes))) &&
      dispatch_copy(
          runtime,
          make_strided_view(io.data, static_cast<uint64_t>(length),
                            static_cast<uint64_t>(in_channels),
                            static_cast<uint64_t>(in_channels) * sizeof(float),
                            sizeof(float)),
          make_strided_view_mut(
              channel_major + state_len, static_cast<uint64_t>(length),
              static_cast<uint64_t>(in_channels), sizeof(float), row_bytes)) &&
      (state_len == 0 ||
       dispatch_copy(
           runtime,
           make_strided_view(
               channel_major + length, static_cast<uint64_t>(state_len),
               static_cast<uint64_t>(in_channels), sizeof(float), row_bytes),
           make_view_mut(state_slice, static_cast<uint64_t>(state_len),
                         static_cast<uint64_t>(in_channels))));
  if (!staged) {
    return false;
  }

  kev::op_im2col im2col_ev{};
  im2col_ev.src0 = make_view(conv.weight, static_cast<uint64_t>(taps),
                             static_cast<uint64_t>(in_channels));
  im2col_ev.src1 = make_view(channel_major, static_cast<uint64_t>(padded),
                             static_cast<uint64_t>(in_channels));
  if constexpr (conv_f16) {
    // reference operand class: the im2col output is rounded to f16, exactly
    // as ggml_conv_1d does for f16 weights
    im2col_ev.src0 = make_f16_view(conv.weight_f16, static_cast<uint64_t>(taps),
                                   static_cast<uint64_t>(in_channels));
    im2col_ev.dst = make_f16_view_mut(
        reinterpret_cast<uint16_t *>(plan.columns.data()),
        static_cast<uint64_t>(in_channels) * static_cast<uint64_t>(taps),
        static_cast<uint64_t>(out_length));
  } else {
    im2col_ev.dst = make_view_mut(plan.columns.data(),
                                  static_cast<uint64_t>(in_channels) *
                                      static_cast<uint64_t>(taps),
                                  static_cast<uint64_t>(out_length));
  }
  set_param_i32(im2col_ev, 0u, static_cast<int32_t>(stride));
  set_param_i32(im2col_ev, 1u, 0);
  set_param_i32(im2col_ev, 2u, 0);
  set_param_i32(im2col_ev, 3u, 0);
  set_param_i32(im2col_ev, 4u, 1);
  set_param_i32(im2col_ev, 5u, 0);
  set_param_i32(im2col_ev, 6u, 0);
  if (!runtime.kernel.process_event(im2col_ev)) {
    return false;
  }

  bool multiplied = false;
  if constexpr (conv_f16) {
    // ggml layout: src0 = raw f16 weight rows [k, out], src1 = f16 columns
    // [k, positions], dst f32 [out, positions] (bit-exact ggml mul_mat)
    multiplied = dispatch_mul_mat(
        runtime,
        make_f16_view(conv.weight_f16,
                      static_cast<uint64_t>(in_channels) *
                          static_cast<uint64_t>(taps),
                      static_cast<uint64_t>(out_channels)),
        make_f16_view(reinterpret_cast<const uint16_t *>(plan.columns.data()),
                      static_cast<uint64_t>(in_channels) *
                          static_cast<uint64_t>(taps),
                      static_cast<uint64_t>(out_length)),
        make_view_mut(io.data, static_cast<uint64_t>(out_channels),
                      static_cast<uint64_t>(out_length)));
  } else {
    multiplied = dispatch_mul_mat(
        runtime,
        make_view(plan.columns.data(),
                  static_cast<uint64_t>(in_channels) *
                      static_cast<uint64_t>(taps),
                  static_cast<uint64_t>(out_length)),
        make_view(conv.weight, static_cast<uint64_t>(out_channels),
                  static_cast<uint64_t>(in_channels) *
                      static_cast<uint64_t>(taps)),
        make_view_mut(io.data, static_cast<uint64_t>(out_channels),
                      static_cast<uint64_t>(out_length)));
  }
  if (!multiplied) {
    return false;
  }

  if (conv.bias != nullptr &&
      !dispatch_add(runtime,
                    make_view(io.data, static_cast<uint64_t>(out_channels),
                              static_cast<uint64_t>(out_length)),
                    make_view(conv.bias, static_cast<uint64_t>(out_channels)),
                    make_view_mut(io.data, static_cast<uint64_t>(out_channels),
                                  static_cast<uint64_t>(out_length)))) {
    return false;
  }

  io.channels = static_cast<int32_t>(out_channels);
  io.length = static_cast<int32_t>(out_length);
  return true;
}

template <bool depthwise>
bool compute_streaming_conv_transpose_impl(
    codec_runtime &runtime, const conv_weights &conv,
    codec_streaming_state &state, frame_buffer &io,
    std::span<float> workspace) noexcept {
  runtime.kernel.set_kind(runtime.kernel_kind);
  const int64_t in_channels = conv.in_channels;
  const int64_t out_channels = conv.out_channels;
  const int64_t taps = conv.taps;
  const int64_t stride = conv.stride;
  const int64_t length = io.length;
  const int64_t tail = taps - stride;
  const int64_t lout_full = (length - 1) * stride + taps;
  const int64_t emitted = lout_full - tail;
  if (io.channels != in_channels || length <= 0 || emitted <= 0 ||
      (depthwise && in_channels != out_channels)) {
    return false;
  }

  workspace_plan plan{};
  const uint64_t channel_major_floats =
      static_cast<uint64_t>(in_channels) * static_cast<uint64_t>(length);
  const uint64_t full_floats =
      static_cast<uint64_t>(out_channels) * static_cast<uint64_t>(lout_full);
  if (!carve_workspace(workspace, channel_major_floats, full_floats, 0u,
                       plan)) {
    return false;
  }

  float *channel_major = plan.channel_major.data();
  float *full = plan.columns.data();
  float *state_slice = state.arena.data() + conv.state_offset;

  if (!dispatch_copy(
          runtime,
          make_strided_view(io.data, static_cast<uint64_t>(length),
                            static_cast<uint64_t>(in_channels),
                            static_cast<uint64_t>(in_channels) * sizeof(float),
                            sizeof(float)),
          make_view_mut(channel_major, static_cast<uint64_t>(length),
                        static_cast<uint64_t>(in_channels)))) {
    return false;
  }

  if constexpr (depthwise) {
    // Grouped (groups == channels) transposed conv: one bounded kernel
    // dispatch per channel over the already-chosen algorithm.
    for (int64_t channel = 0; channel < in_channels; ++channel) {
      kev::op_conv_transpose_1d ev{};
      ev.src0 = make_view(conv.weight + channel * taps,
                          static_cast<uint64_t>(taps), 1, 1);
      ev.src1 = make_view(channel_major + channel * length,
                          static_cast<uint64_t>(length), 1);
      ev.dst = make_view_mut(full + channel * lout_full,
                             static_cast<uint64_t>(lout_full), 1);
      set_param_i32(ev, 0u, static_cast<int32_t>(stride));
      set_param_i32(ev, 1u, 0);
      set_param_i32(ev, 2u, 1);
      if (!runtime.kernel.process_event(ev)) {
        return false;
      }
    }
  } else {
    kev::op_conv_transpose_1d ev{};
    ev.src0 = make_view(conv.weight, static_cast<uint64_t>(taps),
                        static_cast<uint64_t>(out_channels),
                        static_cast<uint64_t>(in_channels));
    ev.src1 = make_view(channel_major, static_cast<uint64_t>(length),
                        static_cast<uint64_t>(in_channels));
    ev.dst = make_view_mut(full, static_cast<uint64_t>(lout_full),
                           static_cast<uint64_t>(out_channels));
    set_param_i32(ev, 0u, static_cast<int32_t>(stride));
    set_param_i32(ev, 1u, 0);
    set_param_i32(ev, 2u, 1);
    if (!runtime.kernel.process_event(ev)) {
      return false;
    }
  }

  const uint64_t full_row_bytes =
      static_cast<uint64_t>(lout_full) * sizeof(float);
  const uint64_t state_row_bytes = full_row_bytes;
  const bool overlapped =
      tail == 0 ||
      dispatch_add(runtime,
                   make_strided_view(full, static_cast<uint64_t>(tail),
                                     static_cast<uint64_t>(out_channels),
                                     sizeof(float), full_row_bytes),
                   make_strided_view(state_slice + (lout_full - tail),
                                     static_cast<uint64_t>(tail),
                                     static_cast<uint64_t>(out_channels),
                                     sizeof(float), state_row_bytes),
                   make_strided_view_mut(full, static_cast<uint64_t>(tail),
                                         static_cast<uint64_t>(out_channels),
                                         sizeof(float), full_row_bytes));
  if (!overlapped) {
    return false;
  }

  // Save the full pre-bias output as next frame's overlap state.
  if (!dispatch_copy(runtime,
                     make_view(full, static_cast<uint64_t>(lout_full),
                               static_cast<uint64_t>(out_channels)),
                     make_view_mut(state_slice,
                                   static_cast<uint64_t>(lout_full),
                                   static_cast<uint64_t>(out_channels)))) {
    return false;
  }

  // Trim the withheld tail and transpose back to time-major.
  if (!dispatch_copy(runtime,
                     make_strided_view(full,
                                       static_cast<uint64_t>(out_channels),
                                       static_cast<uint64_t>(emitted),
                                       full_row_bytes, sizeof(float)),
                     make_view_mut(io.data, static_cast<uint64_t>(out_channels),
                                   static_cast<uint64_t>(emitted)))) {
    return false;
  }

  if (conv.bias != nullptr &&
      !dispatch_add(runtime,
                    make_view(io.data, static_cast<uint64_t>(out_channels),
                              static_cast<uint64_t>(emitted)),
                    make_view(conv.bias, static_cast<uint64_t>(out_channels)),
                    make_view_mut(io.data, static_cast<uint64_t>(out_channels),
                                  static_cast<uint64_t>(emitted)))) {
    return false;
  }

  io.channels = static_cast<int32_t>(out_channels);
  io.length = static_cast<int32_t>(emitted);
  return true;
}

bool compute_streaming_conv_transpose(codec_runtime &runtime,
                                      const conv_weights &conv,
                                      codec_streaming_state &state,
                                      frame_buffer &io,
                                      std::span<float> workspace) noexcept {
  return compute_streaming_conv_transpose_impl<false>(runtime, conv, state, io,
                                                      workspace);
}

bool compute_streaming_conv_transpose_depthwise(
    codec_runtime &runtime, const conv_weights &conv,
    codec_streaming_state &state, frame_buffer &io,
    std::span<float> workspace) noexcept {
  return compute_streaming_conv_transpose_impl<true>(runtime, conv, state, io,
                                                     workspace);
}

template <bool conv_f16>
bool compute_seanet_stack(codec_runtime &runtime,
                          std::span<const seanet_layer_weights> layers,
                          codec_streaming_state &state, frame_buffer &io,
                          std::span<float> workspace) noexcept {
  const uint64_t half = workspace.size() / 2u;
  const std::span<float> residual_span = workspace.subspan(half);
  for (const auto &layer : layers) {
    if (layer.kind == seanet_layer_kind::none) {
      continue;
    }
    if (layer.kind == seanet_layer_kind::elu) {
      const uint64_t count =
          static_cast<uint64_t>(io.channels) * static_cast<uint64_t>(io.length);
      if (!dispatch_unary(runtime, kev::unary_subop::elu,
                          make_view(io.data, count),
                          make_view_mut(io.data, count))) {
        return false;
      }
      continue;
    }
    if (layer.kind == seanet_layer_kind::conv) {
      if (!compute_streaming_conv<conv_f16>(runtime, layer.conv, state, io,
                                            workspace.subspan(0, half))) {
        return false;
      }
      continue;
    }
    if (layer.kind == seanet_layer_kind::conv_transpose) {
      if (!compute_streaming_conv_transpose(runtime, layer.conv, state, io,
                                            workspace.subspan(0, half))) {
        return false;
      }
      continue;
    }
    // resnet: residual = x; x = conv_k1(elu(conv_k3(elu(x)))); x += residual
    const uint64_t count =
        static_cast<uint64_t>(io.channels) * static_cast<uint64_t>(io.length);
    if (residual_span.size() < count) {
      return false;
    }
    float *residual = residual_span.data();
    const int32_t in_channels = io.channels;
    const int32_t in_length = io.length;
    const bool block_ok =
        dispatch_copy(runtime, make_view(io.data, count),
                      make_view_mut(residual, count)) &&
        dispatch_unary(runtime, kev::unary_subop::elu,
                       make_view(io.data, count),
                       make_view_mut(io.data, count)) &&
        compute_streaming_conv<conv_f16>(runtime, layer.resnet.block_1, state,
                                         io, workspace.subspan(0, half)) &&
        dispatch_unary(
            runtime, kev::unary_subop::elu,
            make_view(io.data, static_cast<uint64_t>(io.channels) *
                                   static_cast<uint64_t>(io.length)),
            make_view_mut(io.data, static_cast<uint64_t>(io.channels) *
                                       static_cast<uint64_t>(io.length))) &&
        compute_streaming_conv<conv_f16>(runtime, layer.resnet.block_3, state,
                                         io, workspace.subspan(0, half));
    if (!block_ok || io.channels != in_channels || io.length != in_length) {
      return false;
    }
    if (!dispatch_add(runtime, make_view(io.data, count),
                      make_view(residual, count),
                      make_view_mut(io.data, count))) {
      return false;
    }
  }
  return true;
}

bool compute_transformer(codec_runtime &runtime,
                         const transformer_weights &weights,
                         codec_streaming_state &state, int64_t &positions,
                         frame_buffer &io,
                         std::span<float> workspace) noexcept {
  runtime.kernel.set_kind(runtime.kernel_kind);
  const int64_t dim = weights.dim;
  const int64_t heads = weights.head_count;
  const int64_t head_dim = dim / heads;
  const int64_t context = weights.context;
  const int64_t tokens = io.length;
  if (io.channels != dim || tokens <= 0 || heads <= 0 ||
      head_dim * heads != dim) {
    return false;
  }

  // workspace: normed | qkv | scores | attn | mlp | position staging
  const uint64_t need =
      static_cast<uint64_t>(dim) + 3u * static_cast<uint64_t>(dim) +
      static_cast<uint64_t>(context) + 2u * static_cast<uint64_t>(dim) +
      static_cast<uint64_t>(weights.layers[0].mlp_dim) + 4u;
  if (workspace.size() < need) {
    return false;
  }
  float *normed = workspace.data();
  float *qkv = normed + dim;
  float *scores = qkv + 3 * dim;
  float *attn = scores + context;
  float *proj = attn + dim;
  float *mlp = proj + dim;
  float *position_staging = mlp + weights.layers[0].mlp_dim;

  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  for (int64_t token = 0; token < tokens; ++token) {
    float *x = io.data + token * dim;
    const int64_t position = positions + token;
    const int64_t slot = position % context;
    const int64_t window = (position + 1) < context ? (position + 1) : context;

    for (int32_t layer_index = 0; layer_index < weights.layer_count;
         ++layer_index) {
      const auto &layer = weights.layers[static_cast<size_t>(layer_index)];
      float *k_ring = state.arena.data() + weights.state_offset +
                      static_cast<uint64_t>(layer_index) * 2u *
                          static_cast<uint64_t>(context) *
                          static_cast<uint64_t>(dim);
      float *v_ring =
          k_ring + static_cast<uint64_t>(context) * static_cast<uint64_t>(dim);

      // norm1 -> fused qkv
      const bool normed_ok =
          dispatch_copy(runtime, make_view(x, static_cast<uint64_t>(dim)),
                        make_view_mut(normed, static_cast<uint64_t>(dim))) &&
          dispatch_layer_norm(runtime, layer.norm1_weight, layer.norm1_bias,
                              normed, static_cast<uint64_t>(dim), 1u) &&
          dispatch_mul_mat(
              runtime, make_view(normed, static_cast<uint64_t>(dim), 1u),
              make_view(layer.in_proj, 3u * static_cast<uint64_t>(dim),
                        static_cast<uint64_t>(dim)),
              make_view_mut(qkv, 3u * static_cast<uint64_t>(dim), 1u));
      if (!normed_ok) {
        return false;
      }

      // rope on q and k (neox pairing is not used by the codec transformers:
      // the reference applies interleaved-pair rope)
      int32_t position_i32 = static_cast<int32_t>(position);
      std::memcpy(position_staging, &position_i32, sizeof(position_i32));
      for (int32_t half = 0; half < 2; ++half) {
        kev::op_rope rope_ev{};
        rope_ev.src0 =
            make_view(qkv + half * dim, static_cast<uint64_t>(head_dim),
                      static_cast<uint64_t>(heads), 1u);
        rope_ev.src1 = make_view(position_staging, 1u);
        rope_ev.src1.type = kev::dtype::i32;
        rope_ev.dst =
            make_view_mut(qkv + half * dim, static_cast<uint64_t>(head_dim),
                          static_cast<uint64_t>(heads), 1u);
        set_param_i32(rope_ev, 1u, static_cast<int32_t>(head_dim));
        set_param_i32(rope_ev, 2u, 0);
        set_param_f32(rope_ev, 5u, static_cast<float>(weights.max_period));
        set_param_f32(rope_ev, 6u, 1.0f);
        set_param_f32(rope_ev, 7u, 0.0f);
        set_param_f32(rope_ev, 8u, 1.0f);
        set_param_f32(rope_ev, 9u, 32.0f);
        set_param_f32(rope_ev, 10u, 1.0f);
        if (!runtime.kernel.process_event(rope_ev)) {
          return false;
        }
      }

      // append k, v at the ring slot, rounded through bf16 exactly as the
      // reference caches them (moshi casts K and V to BF16 in the cache)
      for (int64_t d = 0; d < dim; ++d) {
        k_ring[slot * dim + d] = emel::kernel::detail::bf16_to_fp32(
            emel::kernel::detail::fp32_to_bf16(qkv[dim + d]));
        v_ring[slot * dim + d] = emel::kernel::detail::bf16_to_fp32(
            emel::kernel::detail::fp32_to_bf16(qkv[2 * dim + d]));
      }

      // windowed attention per head; the ring window is at most two
      // contiguous segments
      const int64_t seg1_start = (position + 1 - window) % context;
      const int64_t seg1_len =
          seg1_start + window <= context ? window : context - seg1_start;
      const int64_t seg2_len = window - seg1_len;
      for (int64_t head = 0; head < heads; ++head) {
        const float *q_head = qkv + head * head_dim;
        const uint64_t head_offset =
            static_cast<uint64_t>(head) * static_cast<uint64_t>(head_dim);
        const bool scored =
            dispatch_mul_mat(
                runtime,
                make_strided_view(k_ring + seg1_start * dim + head_offset,
                                  static_cast<uint64_t>(head_dim),
                                  static_cast<uint64_t>(seg1_len),
                                  sizeof(float),
                                  static_cast<uint64_t>(dim) * sizeof(float)),
                make_view(q_head, 1u, static_cast<uint64_t>(head_dim)),
                make_view_mut(scores, 1u, static_cast<uint64_t>(seg1_len))) &&
            (seg2_len == 0 ||
             dispatch_mul_mat(
                 runtime,
                 make_strided_view(
                     k_ring + head_offset, static_cast<uint64_t>(head_dim),
                     static_cast<uint64_t>(seg2_len), sizeof(float),
                     static_cast<uint64_t>(dim) * sizeof(float)),
                 make_view(q_head, 1u, static_cast<uint64_t>(head_dim)),
                 make_view_mut(scores + seg1_len, 1u,
                               static_cast<uint64_t>(seg2_len))));
        if (!scored) {
          return false;
        }

        kev::op_soft_max softmax_ev{};
        softmax_ev.src0 = make_view(scores, static_cast<uint64_t>(window), 1u);
        softmax_ev.dst =
            make_view_mut(scores, static_cast<uint64_t>(window), 1u);
        set_param_f32(softmax_ev, 0u, attn_scale);
        if (!runtime.kernel.process_event(softmax_ev)) {
          return false;
        }

        float *attn_head = attn + head * head_dim;
        const bool attended =
            dispatch_mul_mat(
                runtime, make_view(scores, static_cast<uint64_t>(seg1_len), 1u),
                make_strided_view(v_ring + seg1_start * dim + head_offset,
                                  static_cast<uint64_t>(head_dim),
                                  static_cast<uint64_t>(seg1_len),
                                  sizeof(float),
                                  static_cast<uint64_t>(dim) * sizeof(float)),
                make_view_mut(attn_head, static_cast<uint64_t>(head_dim),
                              1u)) &&
            (seg2_len == 0 ||
             (dispatch_mul_mat(
                  runtime,
                  make_view(scores + seg1_len, static_cast<uint64_t>(seg2_len),
                            1u),
                  make_strided_view(
                      v_ring + head_offset, static_cast<uint64_t>(head_dim),
                      static_cast<uint64_t>(seg2_len), sizeof(float),
                      static_cast<uint64_t>(dim) * sizeof(float)),
                  make_view_mut(proj, static_cast<uint64_t>(head_dim), 1u)) &&
              dispatch_add(
                  runtime,
                  make_view(attn_head, static_cast<uint64_t>(head_dim)),
                  make_view(proj, static_cast<uint64_t>(head_dim)),
                  make_view_mut(attn_head, static_cast<uint64_t>(head_dim)))));
        if (!attended) {
          return false;
        }
      }

      // out proj, layer scale, residual
      const bool projected =
          dispatch_mul_mat(
              runtime, make_view(attn, static_cast<uint64_t>(dim), 1u),
              make_view(layer.out_proj, static_cast<uint64_t>(dim),
                        static_cast<uint64_t>(dim)),
              make_view_mut(proj, static_cast<uint64_t>(dim), 1u)) &&
          dispatch_mul(
              runtime, make_view(proj, static_cast<uint64_t>(dim), 1u),
              make_view(layer.layer_scale_1, static_cast<uint64_t>(dim)),
              make_view_mut(proj, static_cast<uint64_t>(dim), 1u)) &&
          dispatch_add(runtime, make_view(x, static_cast<uint64_t>(dim)),
                       make_view(proj, static_cast<uint64_t>(dim)),
                       make_view_mut(x, static_cast<uint64_t>(dim)));
      if (!projected) {
        return false;
      }

      // norm2 -> linear1 -> gelu -> linear2 -> layer scale -> residual
      const int64_t mlp_dim = layer.mlp_dim;
      const bool updated =
          dispatch_copy(runtime, make_view(x, static_cast<uint64_t>(dim)),
                        make_view_mut(normed, static_cast<uint64_t>(dim))) &&
          dispatch_layer_norm(runtime, layer.norm2_weight, layer.norm2_bias,
                              normed, static_cast<uint64_t>(dim), 1u) &&
          dispatch_mul_mat(
              runtime, make_view(normed, static_cast<uint64_t>(dim), 1u),
              make_view(layer.linear1, static_cast<uint64_t>(mlp_dim),
                        static_cast<uint64_t>(dim)),
              make_view_mut(mlp, static_cast<uint64_t>(mlp_dim), 1u)) &&
          dispatch_unary(runtime, kev::unary_subop::gelu,
                         make_view(mlp, static_cast<uint64_t>(mlp_dim)),
                         make_view_mut(mlp, static_cast<uint64_t>(mlp_dim))) &&
          dispatch_mul_mat(
              runtime, make_view(mlp, static_cast<uint64_t>(mlp_dim), 1u),
              make_view(layer.linear2, static_cast<uint64_t>(dim),
                        static_cast<uint64_t>(mlp_dim)),
              make_view_mut(proj, static_cast<uint64_t>(dim), 1u)) &&
          dispatch_mul(
              runtime, make_view(proj, static_cast<uint64_t>(dim), 1u),
              make_view(layer.layer_scale_2, static_cast<uint64_t>(dim)),
              make_view_mut(proj, static_cast<uint64_t>(dim), 1u)) &&
          dispatch_add(runtime, make_view(x, static_cast<uint64_t>(dim)),
                       make_view(proj, static_cast<uint64_t>(dim)),
                       make_view_mut(x, static_cast<uint64_t>(dim)));
      if (!updated) {
        return false;
      }
    }
  }

  positions += tokens;
  return true;
}

template <bool conv_f16>
bool compute_rvq_encode(codec_runtime &runtime, const frame_buffer &latent,
                        std::span<int32_t> codes_out,
                        std::span<float> workspace) noexcept {
  runtime.kernel.set_kind(runtime.kernel_kind);
  const auto &quantizer = runtime.quantizer;
  const int64_t dim = runtime.dim;
  const int64_t codebook_dim = quantizer.codebook_dim;
  const int64_t entries = quantizer.codebook_entries;
  const int64_t frames = latent.length;
  const int64_t n_q = runtime.n_q;
  if (latent.channels != dim || frames <= 0 ||
      codes_out.size() <
          static_cast<size_t>(n_q) * static_cast<size_t>(frames)) {
    return false;
  }
  const uint64_t need = 2u * static_cast<uint64_t>(codebook_dim) + 2u +
                        (static_cast<uint64_t>(dim) + 1u) / 2u + 1u;
  if (workspace.size() < need) {
    return false;
  }
  float *residual = workspace.data(); // [codebook_dim + 1] with trailing 1
  float *chosen = residual + codebook_dim + 1;
  auto *staged_f16 =
      reinterpret_cast<uint16_t *>(chosen + codebook_dim + 1); // [dim] halfs

  for (int64_t frame = 0; frame < frames; ++frame) {
    const float *x = latent.data + frame * dim;
    const rvq_split_weights *splits[2] = {&quantizer.semantic,
                                          &quantizer.acoustic};
    int64_t code_row = 0;
    for (const rvq_split_weights *split : splits) {
      if (split->level_count <= 0) {
        continue;
      }
      // project into codebook space; append the bias-fold constant
      bool projected_in = false;
      if constexpr (conv_f16) {
        // reference conv1x1 path: x rounds to f16 (ggml im2col k=1), then
        // the ggml-exact f16 matmul against the raw f16 projection
        for (int64_t d = 0; d < dim; ++d) {
          staged_f16[d] = ::emel::kernel::detail::quant::fp32_to_fp16(x[d]);
        }
        projected_in = dispatch_mul_mat(
            runtime,
            make_f16_view(split->input_proj_f16, static_cast<uint64_t>(dim),
                          static_cast<uint64_t>(codebook_dim)),
            make_f16_view(staged_f16, static_cast<uint64_t>(dim), 1u),
            make_view_mut(residual, static_cast<uint64_t>(codebook_dim), 1u));
      } else {
        projected_in = dispatch_mul_mat(
            runtime, make_view(x, static_cast<uint64_t>(dim), 1u),
            make_view(split->input_proj, static_cast<uint64_t>(codebook_dim),
                      static_cast<uint64_t>(dim)),
            make_view_mut(residual, static_cast<uint64_t>(codebook_dim), 1u));
      }
      if (!projected_in) {
        return false;
      }
      residual[codebook_dim] = 1.0f;

      for (int32_t level = 0; level < split->level_count; ++level) {
        int32_t index = -1;
        float score = 0.0f;
        kev::op_mul_mat_argmax argmax_ev{};
        argmax_ev.src0 =
            make_view(split->search_tables[static_cast<size_t>(level)],
                      static_cast<uint64_t>(codebook_dim) + 1u,
                      static_cast<uint64_t>(entries));
        argmax_ev.src1 =
            make_view(residual, 1u, static_cast<uint64_t>(codebook_dim) + 1u);
        argmax_ev.dst = make_view_mut(&score, 1u, 1u);
        argmax_ev.index_out = &index;
        if (!runtime.kernel.process_event(argmax_ev) || index < 0) {
          return false;
        }
        codes_out[static_cast<size_t>(code_row * frames + frame)] = index;
        ++code_row;

        if (level + 1 < split->level_count) {
          kev::op_get_rows rows_ev{};
          rows_ev.src0 = make_view(split->codebooks[static_cast<size_t>(level)],
                                   static_cast<uint64_t>(codebook_dim),
                                   static_cast<uint64_t>(entries));
          rows_ev.src1 = make_view(reinterpret_cast<const float *>(&index), 1u);
          rows_ev.src1.type = kev::dtype::i32;
          rows_ev.dst =
              make_view_mut(chosen, static_cast<uint64_t>(codebook_dim), 1u);
          if (!runtime.kernel.process_event(rows_ev) ||
              !dispatch_sub(
                  runtime,
                  make_view(residual, static_cast<uint64_t>(codebook_dim)),
                  make_view(chosen, static_cast<uint64_t>(codebook_dim)),
                  make_view_mut(residual,
                                static_cast<uint64_t>(codebook_dim)))) {
            return false;
          }
        }
      }
    }
    if (code_row != n_q) {
      return false;
    }
  }
  return true;
}

template <bool conv_f16>
bool compute_rvq_decode(codec_runtime &runtime, std::span<const int32_t> codes,
                        int32_t frames, frame_buffer &latent_out,
                        std::span<float> workspace) noexcept {
  runtime.kernel.set_kind(runtime.kernel_kind);
  const auto &quantizer = runtime.quantizer;
  const int64_t dim = runtime.dim;
  const int64_t codebook_dim = quantizer.codebook_dim;
  const int64_t entries = quantizer.codebook_entries;
  const int64_t n_q = runtime.n_q;
  if (frames <= 0 ||
      codes.size() < static_cast<size_t>(n_q) * static_cast<size_t>(frames)) {
    return false;
  }
  const uint64_t need = 2u * static_cast<uint64_t>(codebook_dim) +
                        static_cast<uint64_t>(dim) + 2u +
                        (static_cast<uint64_t>(codebook_dim) + 1u) / 2u + 1u;
  if (workspace.size() < need) {
    return false;
  }
  float *summed = workspace.data();
  float *chosen = summed + codebook_dim;
  float *projected = chosen + codebook_dim;
  auto *staged_f16 =
      reinterpret_cast<uint16_t *>(projected + dim); // [codebook_dim] halfs

  for (int64_t frame = 0; frame < frames; ++frame) {
    float *out = latent_out.data + frame * dim;
    std::memset(out, 0, static_cast<size_t>(dim) * sizeof(float));
    const rvq_split_weights *splits[2] = {&quantizer.semantic,
                                          &quantizer.acoustic};
    int64_t code_row = 0;
    for (const rvq_split_weights *split : splits) {
      if (split->level_count <= 0) {
        continue;
      }
      std::memset(summed, 0, static_cast<size_t>(codebook_dim) * sizeof(float));
      for (int32_t level = 0; level < split->level_count; ++level) {
        const int32_t index =
            codes[static_cast<size_t>(code_row * frames + frame)];
        ++code_row;
        if (index < 0 || index >= entries) {
          return false;
        }
        kev::op_get_rows rows_ev{};
        rows_ev.src0 = make_view(split->codebooks[static_cast<size_t>(level)],
                                 static_cast<uint64_t>(codebook_dim),
                                 static_cast<uint64_t>(entries));
        rows_ev.src1 = make_view(reinterpret_cast<const float *>(&index), 1u);
        rows_ev.src1.type = kev::dtype::i32;
        rows_ev.dst =
            make_view_mut(chosen, static_cast<uint64_t>(codebook_dim), 1u);
        if (!runtime.kernel.process_event(rows_ev) ||
            !dispatch_add(
                runtime, make_view(summed, static_cast<uint64_t>(codebook_dim)),
                make_view(chosen, static_cast<uint64_t>(codebook_dim)),
                make_view_mut(summed, static_cast<uint64_t>(codebook_dim)))) {
          return false;
        }
      }
      bool projected_out = false;
      if constexpr (conv_f16) {
        // reference conv1x1 path: summed residual rounds to f16, then the
        // ggml-exact f16 matmul against the raw f16 output projection
        for (int64_t c = 0; c < codebook_dim; ++c) {
          staged_f16[c] =
              ::emel::kernel::detail::quant::fp32_to_fp16(summed[c]);
        }
        projected_out = dispatch_mul_mat(
            runtime,
            make_f16_view(split->output_proj_f16,
                          static_cast<uint64_t>(codebook_dim),
                          static_cast<uint64_t>(dim)),
            make_f16_view(staged_f16, static_cast<uint64_t>(codebook_dim), 1u),
            make_view_mut(projected, static_cast<uint64_t>(dim), 1u));
      } else {
        projected_out = dispatch_mul_mat(
            runtime, make_view(summed, static_cast<uint64_t>(codebook_dim), 1u),
            make_view(split->output_proj, static_cast<uint64_t>(dim),
                      static_cast<uint64_t>(codebook_dim)),
            make_view_mut(projected, static_cast<uint64_t>(dim), 1u));
      }
      const bool projected_ok =
          projected_out &&
          dispatch_add(runtime, make_view(out, static_cast<uint64_t>(dim)),
                       make_view(projected, static_cast<uint64_t>(dim)),
                       make_view_mut(out, static_cast<uint64_t>(dim)));
      if (!projected_ok) {
        return false;
      }
    }
    if (code_row != n_q) {
      return false;
    }
  }

  latent_out.channels = static_cast<int32_t>(dim);
  latent_out.length = frames;
  return true;
}

//------------------------------------------------------------------------------//
// Planning, sizing, and binding.
//------------------------------------------------------------------------------//

namespace {

struct seanet_plan_layer {
  seanet_layer_kind kind = seanet_layer_kind::none;
  int32_t stride = 1;
  const emel::model::data::tensor_record *weight = nullptr;
  const emel::model::data::tensor_record *bias = nullptr;
  const emel::model::data::tensor_record *res1_weight = nullptr;
  const emel::model::data::tensor_record *res1_bias = nullptr;
  const emel::model::data::tensor_record *res3_weight = nullptr;
  const emel::model::data::tensor_record *res3_bias = nullptr;
  int64_t in_length = 0;
  // Resolved geometry: GGUF collapses trailing size-1 dims, so channel
  // counts derive from the chain walk plus element counts, never from the
  // dim list alone.
  int64_t taps = 0;
  int64_t in_channels = 0;
  int64_t out_channels = 0;
  int64_t res_taps = 0;
  int64_t res_half = 0;
};

struct transformer_plan {
  int32_t layer_count = 0;
  int32_t mlp_dim = 0;
  uint64_t prepared = 0;
};

struct codec_plan {
  std::array<seanet_plan_layer, k_max_seanet_layers> encoder = {};
  std::array<seanet_plan_layer, k_max_seanet_layers> decoder = {};
  seanet_plan_layer downsample = {};
  seanet_plan_layer upsample = {};
  transformer_plan encoder_transformer = {};
  transformer_plan decoder_transformer = {};
  codec_extents extents = {};
  uint64_t prepared_floats = 0;
  int32_t frame_samples = 0;
  int64_t encoder_tokens = 0;
  int64_t decoder_tokens = 0;
  bool conv_f16 = false;
  bool valid = false;
};

// Resolves conv geometry given the input channel count known from the chain
// walk: taps from dim 0, out channels from the element count.
bool resolve_conv_geometry(const emel::model::data::tensor_record *weight,
                           const int64_t in_channels, int64_t &taps_out,
                           int64_t &out_channels_out) noexcept {
  if (weight == nullptr || weight->n_dims < 1 || weight->n_dims > 3 ||
      in_channels <= 0) {
    return false;
  }
  taps_out = weight->dims[0];
  if (taps_out <= 0) {
    return false;
  }
  const uint64_t total = tensor_elements(*weight);
  const uint64_t divisor =
      static_cast<uint64_t>(taps_out) * static_cast<uint64_t>(in_channels);
  if (divisor == 0u || total % divisor != 0u) {
    return false;
  }
  out_channels_out = static_cast<int64_t>(total / divisor);
  return out_channels_out > 0;
}

uint64_t conv_prepared_floats(const int64_t taps, const int64_t in_channels,
                              const int64_t out_channels) noexcept {
  return static_cast<uint64_t>(taps) * static_cast<uint64_t>(in_channels) *
             static_cast<uint64_t>(out_channels) +
         static_cast<uint64_t>(out_channels);
}

// f16 raw-weight arena floats for one conv when the model carries f16 convs
uint64_t conv_f16_prepared_floats(const int64_t taps, const int64_t in_channels,
                                  const int64_t out_channels) noexcept {
  const uint64_t halfs = static_cast<uint64_t>(taps) *
                         static_cast<uint64_t>(in_channels) *
                         static_cast<uint64_t>(out_channels);
  return (halfs + 1u) / 2u;
}

bool plan_seanet(
    const emel::model::data &model_data, const char *family,
    const std::array<seanet_layer_spec, k_max_seanet_layers> &topology,
    std::array<seanet_plan_layer, k_max_seanet_layers> &layers_out,
    int64_t &length, int64_t &channels, const bool conv_f16,
    codec_extents &extents, uint64_t &prepared) noexcept {
  char conv_format[64] = {};
  char convtr_format[64] = {};
  char res_format[72] = {};
  std::snprintf(conv_format, sizeof(conv_format),
                "mimi.%s.model.%%d.conv.conv.%%s", family);
  std::snprintf(convtr_format, sizeof(convtr_format),
                "mimi.%s.model.%%d.convtr.convtr.%%s", family);
  std::snprintf(res_format, sizeof(res_format), "mimi.%s.model.%%d.block.%%s",
                family);

  for (int32_t index = 0; index < k_max_seanet_layers; ++index) {
    const auto &spec = topology[static_cast<size_t>(index)];
    auto &layer = layers_out[static_cast<size_t>(index)];
    layer.kind = spec.kind;
    layer.stride = spec.stride;
    layer.in_length = length;
    layer.in_channels = channels;
    if (spec.kind == seanet_layer_kind::elu) {
      continue;
    }
    if (spec.kind == seanet_layer_kind::resnet) {
      layer.res1_weight = find_layer_tensor(model_data, res_format, index,
                                            "1.conv.conv.weight");
      layer.res1_bias =
          find_layer_tensor(model_data, res_format, index, "1.conv.conv.bias");
      layer.res3_weight = find_layer_tensor(model_data, res_format, index,
                                            "3.conv.conv.weight");
      layer.res3_bias =
          find_layer_tensor(model_data, res_format, index, "3.conv.conv.bias");
      int64_t res_half = 0;
      int64_t res_out = 0;
      if (!resolve_conv_geometry(layer.res1_weight, channels, layer.res_taps,
                                 res_half) ||
          !resolve_conv_geometry(layer.res3_weight, res_half,
                                 layer.out_channels, res_out) ||
          layer.out_channels != 1 || res_out != channels) {
        return false;
      }
      // res3 is a k1 conv (out_channels slot above held its taps=1)
      layer.out_channels = channels;
      layer.res_half = res_half;
      account_conv(extents, channels, res_half, layer.res_taps, 1, length,
                   false);
      account_conv(extents, res_half, channels, 1, 1, length, false);
      prepared += conv_prepared_floats(layer.res_taps, channels, res_half) +
                  conv_prepared_floats(1, res_half, channels);
      prepared += conv_f16 ? conv_f16_prepared_floats(layer.res_taps, channels,
                                                      res_half) +
                                 conv_f16_prepared_floats(1, res_half, channels)
                           : 0u;
      continue;
    }

    const bool transposed = spec.kind == seanet_layer_kind::conv_transpose;
    layer.weight = find_layer_tensor(
        model_data, transposed ? convtr_format : conv_format, index, "weight");
    layer.bias = find_layer_tensor(
        model_data, transposed ? convtr_format : conv_format, index, "bias");
    if (!resolve_conv_geometry(layer.weight, channels, layer.taps,
                               layer.out_channels)) {
      return false;
    }
    if (transposed) {
      account_conv(extents, channels, layer.out_channels, layer.taps,
                   spec.stride, length, true);
      prepared +=
          conv_prepared_floats(layer.taps, channels, layer.out_channels);
      length = length * spec.stride;
    } else {
      if (length % spec.stride != 0) {
        return false;
      }
      account_conv(extents, channels, layer.out_channels, layer.taps,
                   spec.stride, length, false);
      prepared +=
          conv_prepared_floats(layer.taps, channels, layer.out_channels);
      prepared += conv_f16 ? conv_f16_prepared_floats(layer.taps, channels,
                                                      layer.out_channels)
                           : 0u;
      length = length / spec.stride;
    }
    channels = layer.out_channels;
  }
  return true;
}

bool plan_transformer(const emel::model::data &model_data, const char *family,
                      transformer_plan &plan_out,
                      codec_extents &extents) noexcept {
  const auto &mimi = model_data.mimi;
  const int64_t dim = mimi.dim;
  char format[96] = {};
  std::snprintf(format, sizeof(format), "mimi.%s.transformer.layers.%%d.%%s",
                family);
  int32_t layer_count = 0;
  int32_t mlp_dim = 0;
  for (int32_t index = 0; index < mimi.transformer_num_layers; ++index) {
    const auto *norm1 =
        find_layer_tensor(model_data, format, index, "norm1.weight");
    const auto *in_proj = find_layer_tensor(model_data, format, index,
                                            "self_attn.in_projs.0.weight");
    const auto *linear1 =
        find_layer_tensor(model_data, format, index, "linear1.weight");
    if (norm1 == nullptr || in_proj == nullptr || linear1 == nullptr ||
        in_proj->dims[0] != dim || in_proj->dims[1] != 3 * dim ||
        linear1->dims[0] != dim) {
      return false;
    }
    mlp_dim = static_cast<int32_t>(linear1->dims[1]);
    ++layer_count;
    plan_out.prepared +=
        static_cast<uint64_t>(dim) * 3u * static_cast<uint64_t>(dim) +
        static_cast<uint64_t>(dim) * static_cast<uint64_t>(dim) +
        2u * static_cast<uint64_t>(dim) * static_cast<uint64_t>(mlp_dim) +
        6u * static_cast<uint64_t>(dim);
  }
  if (layer_count != mimi.transformer_num_layers || mlp_dim <= 0) {
    return false;
  }
  plan_out.layer_count = layer_count;
  plan_out.mlp_dim = mlp_dim;
  extents.state += static_cast<uint64_t>(layer_count) * 2u *
                   static_cast<uint64_t>(mimi.transformer_context) *
                   static_cast<uint64_t>(dim);
  extents.frame =
      std::max<uint64_t>(extents.frame, 2u * static_cast<uint64_t>(dim));
  return true;
}

bool plan_codec(const emel::model::data &model_data,
                codec_plan &plan_out) noexcept {
  const auto &mimi = model_data.mimi;
  if (model_data.moshi_component_id !=
          emel::model::data::moshi_component::mimi ||
      mimi.dim <= 0 || mimi.n_q <= 0 || mimi.frame_rate <= 0.0f) {
    return false;
  }
  plan_out.frame_samples = static_cast<int32_t>(
      static_cast<float>(mimi.sample_rate) / mimi.frame_rate);
  if (plan_out.frame_samples <= 0) {
    return false;
  }

  // Bound operand class: models storing f16 conv weights run the reference
  // f16 pipeline (detected on the first encoder conv; bind cross-checks the
  // rest via prepare_raw_f16 failures).
  const auto *first_conv =
      find_tensor(model_data, "mimi.encoder.model.0.conv.conv.weight");
  plan_out.conv_f16 =
      first_conv != nullptr &&
      static_cast<uint8_t>(first_conv->type) == emel::kernel::detail::dtype_f16;

  int64_t length = plan_out.frame_samples;
  int64_t channels = 1;
  if (!plan_seanet(model_data, "encoder", k_encoder_topology, plan_out.encoder,
                   length, channels, plan_out.conv_f16, plan_out.extents,
                   plan_out.prepared_floats) ||
      channels != mimi.dim) {
    return false;
  }
  plan_out.encoder_tokens = length;
  if (!plan_transformer(model_data, "encoder_transformer",
                        plan_out.encoder_transformer, plan_out.extents)) {
    return false;
  }
  plan_out.prepared_floats += plan_out.encoder_transformer.prepared;

  // downsample conv (25 Hz -> 12.5 Hz)
  plan_out.downsample.weight =
      find_tensor(model_data, "mimi.downsample.conv.conv.conv.weight");
  plan_out.downsample.stride = 2;
  plan_out.downsample.in_length = length;
  plan_out.downsample.in_channels = channels;
  if (!resolve_conv_geometry(plan_out.downsample.weight, channels,
                             plan_out.downsample.taps,
                             plan_out.downsample.out_channels) ||
      plan_out.downsample.out_channels != mimi.dim || length % 2 != 0) {
    return false;
  }
  account_conv(plan_out.extents, channels, plan_out.downsample.out_channels,
               plan_out.downsample.taps, 2, length, false);
  plan_out.prepared_floats += conv_prepared_floats(
      plan_out.downsample.taps, channels, plan_out.downsample.out_channels);
  plan_out.prepared_floats +=
      plan_out.conv_f16
          ? conv_f16_prepared_floats(plan_out.downsample.taps, channels,
                                     plan_out.downsample.out_channels)
          : 0u;
  length = length / 2;
  if (length != 1) {
    return false;
  }

  // quantizer tables (per split: input/output projections + per-level
  // codebook + search table)
  const uint64_t cb_dim = static_cast<uint64_t>(mimi.codebook_dim);
  const uint64_t entries = static_cast<uint64_t>(mimi.card);
  const uint64_t per_split_proj = 2u * cb_dim * static_cast<uint64_t>(mimi.dim);
  const uint64_t per_level = cb_dim * entries + (cb_dim + 1u) * entries;
  plan_out.prepared_floats +=
      2u * per_split_proj + static_cast<uint64_t>(mimi.n_q) * per_level;
  // raw f16 projections (input + output per split)
  plan_out.prepared_floats +=
      plan_out.conv_f16
          ? 4u * ((cb_dim * static_cast<uint64_t>(mimi.dim) + 1u) / 2u)
          : 0u;

  // decode chain
  int64_t decode_length = 1;
  int64_t decode_channels = mimi.dim;
  plan_out.upsample.weight =
      find_tensor(model_data, "mimi.upsample.convtr.convtr.convtr.weight");
  plan_out.upsample.stride = 2;
  plan_out.upsample.in_length = decode_length;
  plan_out.upsample.in_channels = decode_channels;
  plan_out.upsample.out_channels = decode_channels;
  // depthwise: stored [taps, 1, channels] with elements == taps * channels
  if (plan_out.upsample.weight == nullptr ||
      plan_out.upsample.weight->dims[0] <= 0 ||
      tensor_elements(*plan_out.upsample.weight) !=
          static_cast<uint64_t>(plan_out.upsample.weight->dims[0]) *
              static_cast<uint64_t>(decode_channels)) {
    return false;
  }
  plan_out.upsample.taps = plan_out.upsample.weight->dims[0];
  account_conv(plan_out.extents, decode_channels, decode_channels,
               plan_out.upsample.taps, 2, decode_length, true);
  plan_out.prepared_floats += static_cast<uint64_t>(plan_out.upsample.taps) *
                              static_cast<uint64_t>(decode_channels);
  decode_length = decode_length * 2;
  plan_out.decoder_tokens = decode_length;

  if (!plan_transformer(model_data, "decoder_transformer",
                        plan_out.decoder_transformer, plan_out.extents)) {
    return false;
  }
  plan_out.prepared_floats += plan_out.decoder_transformer.prepared;

  if (!plan_seanet(model_data, "decoder", k_decoder_topology, plan_out.decoder,
                   decode_length, decode_channels, plan_out.conv_f16,
                   plan_out.extents, plan_out.prepared_floats)) {
    return false;
  }
  if (decode_length != plan_out.frame_samples || decode_channels != 1) {
    return false;
  }

  plan_out.valid = true;
  return true;
}

uint64_t transformer_workspace_floats(const emel::model::data &model_data,
                                      const int32_t mlp_dim) noexcept {
  const auto &mimi = model_data.mimi;
  return static_cast<uint64_t>(mimi.dim) * 6u +
         static_cast<uint64_t>(mimi.transformer_context) +
         static_cast<uint64_t>(mlp_dim) + 8u;
}

uint64_t workspace_floats_from_plan(const emel::model::data &model_data,
                                    const codec_plan &plan) noexcept {
  const uint64_t conv_ws = plan.extents.channel_major + plan.extents.columns;
  const uint64_t transformer_ws =
      std::max(transformer_workspace_floats(model_data,
                                            plan.encoder_transformer.mlp_dim),
               transformer_workspace_floats(model_data,
                                            plan.decoder_transformer.mlp_dim));
  const uint64_t rvq_ws =
      3u * static_cast<uint64_t>(model_data.mimi.codebook_dim) +
      static_cast<uint64_t>(model_data.mimi.dim) + 8u;
  // seanet stack splits the workspace in half (stage scratch | residual)
  const uint64_t stack_ws = 2u * std::max(conv_ws, plan.extents.frame);
  return std::max(std::max(stack_ws, transformer_ws), rvq_ws) + 64u;
}

// Binds one conv from its plan-resolved geometry (taps / in / out already
// derived from the chain walk, so collapsed GGUF dims are irrelevant here).
bool bind_conv(arena_cursor &cursor, uint64_t &state_cursor,
               const seanet_plan_layer &plan_layer, const bool transposed,
               const bool with_f16, conv_weights &conv_out) noexcept {
  const int64_t taps = plan_layer.taps;
  const int64_t in_channels = plan_layer.in_channels;
  const int64_t out_channels = plan_layer.out_channels;
  if (taps <= 0 || in_channels <= 0 || out_channels <= 0) {
    return false;
  }
  if (transposed) {
    // stored [taps, out, in], consumed verbatim by op_conv_transpose_1d
    conv_out.weight = prepare_conv_transpose(
        cursor, plan_layer.weight,
        static_cast<uint64_t>(taps) * static_cast<uint64_t>(in_channels) *
            static_cast<uint64_t>(out_channels));
  } else {
    conv_out.weight = prepare_conv_gemm(cursor, plan_layer.weight, taps,
                                        in_channels, out_channels);
    if (with_f16) {
      // reference operand class: keep the raw f16 taps alongside the f32
      // canonical copy; every conv in an f16 model must itself be f16
      conv_out.weight_f16 = prepare_raw_f16(
          cursor, plan_layer.weight,
          static_cast<uint64_t>(taps) * static_cast<uint64_t>(in_channels) *
              static_cast<uint64_t>(out_channels));
      if (conv_out.weight_f16 == nullptr) {
        return false;
      }
    }
  }
  conv_out.in_channels = static_cast<int32_t>(in_channels);
  conv_out.out_channels = static_cast<int32_t>(out_channels);
  conv_out.bias =
      plan_layer.bias != nullptr
          ? prepare_vector(cursor, plan_layer.bias, conv_out.out_channels)
          : nullptr;
  if (conv_out.weight == nullptr ||
      (plan_layer.bias != nullptr && conv_out.bias == nullptr)) {
    return false;
  }
  conv_out.taps = static_cast<int32_t>(taps);
  conv_out.stride = plan_layer.stride;
  conv_out.frame_length = static_cast<int32_t>(plan_layer.in_length);
  conv_out.state_offset = state_cursor;
  if (transposed) {
    const int64_t lout_full =
        (plan_layer.in_length - 1) * plan_layer.stride + taps;
    conv_out.state_floats = static_cast<uint64_t>(lout_full) *
                            static_cast<uint64_t>(conv_out.out_channels);
  } else {
    conv_out.state_floats = static_cast<uint64_t>(taps - plan_layer.stride) *
                            static_cast<uint64_t>(conv_out.in_channels);
  }
  state_cursor += conv_out.state_floats;
  return true;
}

bool bind_transformer(arena_cursor &cursor, uint64_t &state_cursor,
                      const emel::model::data &model_data, const char *family,
                      const transformer_plan &plan, const int32_t frame_tokens,
                      transformer_weights &weights_out) noexcept {
  const auto &mimi = model_data.mimi;
  const int64_t dim = mimi.dim;
  char format[96] = {};
  std::snprintf(format, sizeof(format), "mimi.%s.transformer.layers.%%d.%%s",
                family);
  weights_out.layer_count = plan.layer_count;
  weights_out.dim = mimi.dim;
  weights_out.head_count = mimi.transformer_num_heads;
  weights_out.context = mimi.transformer_context;
  weights_out.max_period = mimi.transformer_max_period;
  weights_out.frame_tokens = frame_tokens;
  weights_out.state_offset = state_cursor;
  weights_out.state_floats = static_cast<uint64_t>(plan.layer_count) * 2u *
                             static_cast<uint64_t>(mimi.transformer_context) *
                             static_cast<uint64_t>(dim);
  state_cursor += weights_out.state_floats;

  for (int32_t index = 0; index < plan.layer_count; ++index) {
    auto &layer = weights_out.layers[static_cast<size_t>(index)];
    layer.mlp_dim = plan.mlp_dim;
    layer.norm1_weight = prepare_vector(
        cursor, find_layer_tensor(model_data, format, index, "norm1.weight"),
        dim);
    layer.norm1_bias = prepare_vector(
        cursor, find_layer_tensor(model_data, format, index, "norm1.bias"),
        dim);
    layer.in_proj =
        prepare_linear(cursor,
                       find_layer_tensor(model_data, format, index,
                                         "self_attn.in_projs.0.weight"),
                       dim, 3 * dim);
    layer.out_proj =
        prepare_linear(cursor,
                       find_layer_tensor(model_data, format, index,
                                         "self_attn.out_projs.0.weight"),
                       dim, dim);
    layer.layer_scale_1 = prepare_vector(
        cursor,
        find_layer_tensor(model_data, format, index, "layer_scale_1.scale"),
        dim);
    layer.norm2_weight = prepare_vector(
        cursor, find_layer_tensor(model_data, format, index, "norm2.weight"),
        dim);
    layer.norm2_bias = prepare_vector(
        cursor, find_layer_tensor(model_data, format, index, "norm2.bias"),
        dim);
    layer.linear1 = prepare_linear(
        cursor, find_layer_tensor(model_data, format, index, "linear1.weight"),
        dim, plan.mlp_dim);
    layer.linear2 = prepare_linear(
        cursor, find_layer_tensor(model_data, format, index, "linear2.weight"),
        plan.mlp_dim, dim);
    layer.layer_scale_2 = prepare_vector(
        cursor,
        find_layer_tensor(model_data, format, index, "layer_scale_2.scale"),
        dim);
    if (layer.norm1_weight == nullptr || layer.norm1_bias == nullptr ||
        layer.in_proj == nullptr || layer.out_proj == nullptr ||
        layer.layer_scale_1 == nullptr || layer.norm2_weight == nullptr ||
        layer.norm2_bias == nullptr || layer.linear1 == nullptr ||
        layer.linear2 == nullptr || layer.layer_scale_2 == nullptr) {
      return false;
    }
  }
  return true;
}

bool bind_rvq_split(arena_cursor &cursor, const emel::model::data &model_data,
                    const char *split, const int32_t level_count,
                    const bool with_f16,
                    rvq_split_weights &split_out) noexcept {
  const auto &mimi = model_data.mimi;
  const int64_t dim = mimi.dim;
  const int64_t cb_dim = mimi.codebook_dim;
  const int64_t entries = mimi.card;
  char name[96] = {};

  std::snprintf(name, sizeof(name), "mimi.quantizer.%s.input_proj.weight",
                split);
  split_out.input_proj =
      prepare_linear(cursor, find_tensor(model_data, name), dim, cb_dim);
  std::snprintf(name, sizeof(name), "mimi.quantizer.%s.output_proj.weight",
                split);
  split_out.output_proj =
      prepare_linear(cursor, find_tensor(model_data, name), cb_dim, dim);
  if (split_out.input_proj == nullptr || split_out.output_proj == nullptr) {
    return false;
  }
  if (with_f16) {
    // reference operand class keeps the raw f16 conv1x1 projections
    std::snprintf(name, sizeof(name), "mimi.quantizer.%s.input_proj.weight",
                  split);
    split_out.input_proj_f16 = prepare_raw_f16(
        cursor, find_tensor(model_data, name),
        static_cast<uint64_t>(dim) * static_cast<uint64_t>(cb_dim));
    std::snprintf(name, sizeof(name), "mimi.quantizer.%s.output_proj.weight",
                  split);
    split_out.output_proj_f16 = prepare_raw_f16(
        cursor, find_tensor(model_data, name),
        static_cast<uint64_t>(dim) * static_cast<uint64_t>(cb_dim));
    if (split_out.input_proj_f16 == nullptr ||
        split_out.output_proj_f16 == nullptr) {
      return false;
    }
  }

  split_out.level_count = level_count;
  for (int32_t level = 0; level < level_count; ++level) {
    std::snprintf(name, sizeof(name),
                  "mimi.quantizer.%s.vq.layers.%d._codebook.embedding", split,
                  level);
    const auto *embedding = find_tensor(model_data, name);
    if (embedding == nullptr || !tensor_is_float(*embedding) ||
        embedding->n_dims != 2 || embedding->dims[0] != cb_dim ||
        embedding->dims[1] != entries) {
      return false;
    }
    float *codebook = cursor.take(static_cast<uint64_t>(cb_dim) *
                                  static_cast<uint64_t>(entries));
    float *table = cursor.take((static_cast<uint64_t>(cb_dim) + 1u) *
                               static_cast<uint64_t>(entries));
    if (codebook == nullptr || table == nullptr) {
      return false;
    }
    for (int64_t entry = 0; entry < entries; ++entry) {
      float norm2 = 0.0f;
      for (int64_t component = 0; component < cb_dim; ++component) {
        const float value = source_value(
            *embedding, static_cast<uint64_t>(component + entry * cb_dim));
        codebook[component + entry * cb_dim] = value;
        table[component + entry * (cb_dim + 1)] = value;
        norm2 += value * value;
      }
      table[cb_dim + entry * (cb_dim + 1)] = -0.5f * norm2;
    }
    split_out.codebooks[static_cast<size_t>(level)] = codebook;
    split_out.search_tables[static_cast<size_t>(level)] = table;
  }
  return true;
}

bool bind_seanet(
    arena_cursor &cursor, uint64_t &state_cursor,
    const std::array<seanet_plan_layer, k_max_seanet_layers> &plan_layers,
    const bool with_f16,
    std::array<seanet_layer_weights, k_max_seanet_layers>
        &layers_out) noexcept {
  for (int32_t index = 0; index < k_max_seanet_layers; ++index) {
    const auto &plan_layer = plan_layers[static_cast<size_t>(index)];
    auto &layer = layers_out[static_cast<size_t>(index)];
    layer.kind = plan_layer.kind;
    if (plan_layer.kind == seanet_layer_kind::conv ||
        plan_layer.kind == seanet_layer_kind::conv_transpose) {
      if (!bind_conv(cursor, state_cursor, plan_layer,
                     plan_layer.kind == seanet_layer_kind::conv_transpose,
                     with_f16, layer.conv)) {
        return false;
      }
    } else if (plan_layer.kind == seanet_layer_kind::resnet) {
      seanet_plan_layer shim{};
      shim.stride = 1;
      shim.in_length = plan_layer.in_length;
      shim.weight = plan_layer.res1_weight;
      shim.bias = plan_layer.res1_bias;
      shim.taps = plan_layer.res_taps;
      shim.in_channels = plan_layer.in_channels;
      shim.out_channels = plan_layer.res_half;
      if (!bind_conv(cursor, state_cursor, shim, false, with_f16,
                     layer.resnet.block_1)) {
        return false;
      }
      shim.weight = plan_layer.res3_weight;
      shim.bias = plan_layer.res3_bias;
      shim.taps = 1;
      shim.in_channels = plan_layer.res_half;
      shim.out_channels = plan_layer.in_channels;
      if (!bind_conv(cursor, state_cursor, shim, false, with_f16,
                     layer.resnet.block_3)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

uint64_t
required_prepared_floats(const emel::model::data &model_data) noexcept {
  codec_plan plan{};
  return plan_codec(model_data, plan) ? plan.prepared_floats + 64u : 0u;
}

uint64_t required_state_floats(const emel::model::data &model_data) noexcept {
  codec_plan plan{};
  return plan_codec(model_data, plan) ? plan.extents.state + 64u : 0u;
}

uint64_t
required_workspace_floats(const emel::model::data &model_data) noexcept {
  codec_plan plan{};
  return plan_codec(model_data, plan)
             ? workspace_floats_from_plan(model_data, plan)
             : 0u;
}

uint64_t required_frame_floats(const emel::model::data &model_data) noexcept {
  codec_plan plan{};
  return plan_codec(model_data, plan) ? plan.extents.frame + 64u : 0u;
}

bool bind_codec_runtime(const emel::model::data &model_data,
                        std::span<float> prepared, std::span<float> state,
                        codec_runtime &runtime_out,
                        codec_streaming_state &state_out) noexcept {
  codec_plan plan{};
  if (!plan_codec(model_data, plan) || prepared.size() < plan.prepared_floats ||
      state.size() < plan.extents.state) {
    return false;
  }

  const auto &mimi = model_data.mimi;
  runtime_out.model = &model_data;
  runtime_out.sample_rate = mimi.sample_rate;
  runtime_out.frame_samples = plan.frame_samples;
  runtime_out.n_q = mimi.n_q;
  runtime_out.dim = mimi.dim;
  runtime_out.prepared_arena = prepared;
  runtime_out.quantizer.codebook_entries = mimi.card;
  runtime_out.quantizer.codebook_dim = mimi.codebook_dim;

  arena_cursor cursor{prepared, 0};
  uint64_t state_cursor = 0;

  if (!bind_seanet(cursor, state_cursor, plan.encoder, plan.conv_f16,
                   runtime_out.encoder_layers) ||
      !bind_transformer(cursor, state_cursor, model_data, "encoder_transformer",
                        plan.encoder_transformer,
                        static_cast<int32_t>(plan.encoder_tokens),
                        runtime_out.encoder_transformer)) {
    return false;
  }

  if (!bind_conv(cursor, state_cursor, plan.downsample, false, plan.conv_f16,
                 runtime_out.downsample)) {
    return false;
  }

  runtime_out.conv_f16 = plan.conv_f16;
  if (!bind_rvq_split(cursor, model_data, "rvq_first", mimi.semantic_n_q,
                      plan.conv_f16, runtime_out.quantizer.semantic) ||
      !bind_rvq_split(cursor, model_data, "rvq_rest",
                      mimi.n_q - mimi.semantic_n_q, plan.conv_f16,
                      runtime_out.quantizer.acoustic)) {
    return false;
  }

  {
    // depthwise upsample: stored [taps, 1, channels]; keep the per-channel
    // tap layout verbatim
    const int64_t taps = plan.upsample.taps;
    const int64_t channels = plan.upsample.out_channels;
    auto &upsample = runtime_out.upsample;
    upsample.weight = prepare_conv_transpose(
        cursor, plan.upsample.weight,
        static_cast<uint64_t>(taps) * static_cast<uint64_t>(channels));
    if (upsample.weight == nullptr) {
      return false;
    }
    upsample.bias = nullptr;
    upsample.taps = static_cast<int32_t>(taps);
    upsample.in_channels = static_cast<int32_t>(channels);
    upsample.out_channels = static_cast<int32_t>(channels);
    upsample.stride = 2;
    upsample.frame_length = 1;
    upsample.state_offset = state_cursor;
    const int64_t lout_full = (1 - 1) * 2 + taps;
    upsample.state_floats =
        static_cast<uint64_t>(lout_full) * static_cast<uint64_t>(channels);
    state_cursor += upsample.state_floats;
  }

  if (!bind_transformer(cursor, state_cursor, model_data, "decoder_transformer",
                        plan.decoder_transformer,
                        static_cast<int32_t>(plan.decoder_tokens),
                        runtime_out.decoder_transformer) ||
      !bind_seanet(cursor, state_cursor, plan.decoder, plan.conv_f16,
                   runtime_out.decoder_layers)) {
    return false;
  }

  runtime_out.prepared_floats_used = cursor.used;
  state_out.arena = state;
  reset_streaming_state(runtime_out, state_out);
  return state_cursor <= state.size();
}

template bool compute_seanet_stack<false>(codec_runtime &,
                                          std::span<const seanet_layer_weights>,
                                          codec_streaming_state &,
                                          frame_buffer &,
                                          std::span<float>) noexcept;
template bool compute_seanet_stack<true>(codec_runtime &,
                                         std::span<const seanet_layer_weights>,
                                         codec_streaming_state &,
                                         frame_buffer &,
                                         std::span<float>) noexcept;
template bool compute_streaming_conv<false>(codec_runtime &,
                                            const conv_weights &,
                                            codec_streaming_state &,
                                            frame_buffer &,
                                            std::span<float>) noexcept;
template bool compute_streaming_conv<true>(codec_runtime &,
                                           const conv_weights &,
                                           codec_streaming_state &,
                                           frame_buffer &,
                                           std::span<float>) noexcept;
template bool compute_rvq_encode<false>(codec_runtime &, const frame_buffer &,
                                        std::span<int32_t>,
                                        std::span<float>) noexcept;
template bool compute_rvq_encode<true>(codec_runtime &, const frame_buffer &,
                                       std::span<int32_t>,
                                       std::span<float>) noexcept;
template bool compute_rvq_decode<false>(codec_runtime &,
                                        std::span<const int32_t>, int32_t,
                                        frame_buffer &,
                                        std::span<float>) noexcept;
template bool compute_rvq_decode<true>(codec_runtime &,
                                       std::span<const int32_t>, int32_t,
                                       frame_buffer &,
                                       std::span<float>) noexcept;

void reset_streaming_state(const codec_runtime &runtime,
                           codec_streaming_state &state) noexcept {
  (void)runtime;
  std::memset(state.arena.data(), 0, state.arena.size() * sizeof(float));
  state.encoder_positions = 0;
  state.decoder_positions = 0;
}

} // namespace emel::speech::codec::mimi::detail
