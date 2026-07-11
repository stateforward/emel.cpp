#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string_view>

#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/memory/view.hpp"
#include "emel/model/data.hpp"

namespace emel::speech::predictor::moshi::executor::detail {

using dtype = emel::kernel::event::dtype;
using tensor_record = emel::model::data::tensor_record;
using tensor_view = emel::kernel::event::tensor_view;
using tensor_view_mut = emel::kernel::event::tensor_view_mut;

inline constexpr uint64_t k_max_hidden_dim = 8192u;
inline constexpr uint64_t k_max_qkv_dim = k_max_hidden_dim * 3u;
inline constexpr uint64_t k_max_gating_projection_dim = k_max_hidden_dim * 8u;
inline constexpr uint64_t k_max_gating_dim = k_max_gating_projection_dim / 2u;
inline constexpr uint64_t k_max_temporal_context = 8192u;
inline constexpr uint64_t k_max_depformer_context = 512u;
inline constexpr uint64_t k_max_sampling_card = 65536u;
inline constexpr uint64_t k_max_sampling_top_k = 1024u;
inline constexpr int32_t k_token_zero = -1;

struct streaming_kv_view {
  std::span<uint16_t> key_cache = {};
  std::span<uint16_t> value_cache = {};
  std::span<const size_t> layer_cache_offsets = {};
  int32_t layer_count = 0;
  int32_t position_capacity = 0;
  int32_t kv_dim = 0;
};

using temporal_kv_view = streaming_kv_view;
using depformer_kv_view = streaming_kv_view;

template <class tensor_type>
void fill_default_nb(tensor_type &tensor) noexcept {
  const uint64_t elem_size = emel::kernel::detail::dtype_size_bytes(
      emel::kernel::detail::dtype_code(tensor.type));
  tensor.nb[0] = elem_size;
  tensor.nb[1] = tensor.nb[0] * tensor.ne[0];
  tensor.nb[2] = tensor.nb[1] * tensor.ne[1];
  tensor.nb[3] = tensor.nb[2] * tensor.ne[2];
}

inline const tensor_record *find_tensor(const emel::model::data &model,
                                        const std::string_view name) noexcept {
  for (uint32_t index = 0u; index < model.n_tensors; ++index) {
    const auto &tensor = model.tensors[index];
    if (emel::model::tensor_name_view(model, tensor) == name) {
      return &tensor;
    }
  }
  return nullptr;
}

inline const tensor_record *find_indexed_tensor(const emel::model::data &model,
                                                const char *format,
                                                const int32_t index) noexcept {
  char buffer[96] = {};
  const int written = std::snprintf(buffer, sizeof(buffer), format, index);
  if (written <= 0 || static_cast<size_t>(written) >= sizeof(buffer)) {
    return nullptr;
  }
  return find_tensor(model,
                     std::string_view{buffer, static_cast<size_t>(written)});
}

inline const tensor_record *
find_lm_transformer_tensor(const emel::model::data &model, const int32_t layer,
                           const char *suffix) noexcept {
  char buffer[128] = {};
  const int written = std::snprintf(
      buffer, sizeof(buffer), "lm.transformer.layers.%d.%s", layer, suffix);
  if (written <= 0 || static_cast<size_t>(written) >= sizeof(buffer)) {
    return nullptr;
  }
  return find_tensor(model,
                     std::string_view{buffer, static_cast<size_t>(written)});
}

inline const tensor_record *
find_lm_transformer_projection(const emel::model::data &model,
                               const int32_t layer) noexcept {
  const auto *split =
      find_lm_transformer_tensor(model, layer, "self_attn.in_projs.0.weight");
  if (split != nullptr) {
    return split;
  }
  return find_lm_transformer_tensor(model, layer, "self_attn.in_proj_weight");
}

inline const tensor_record *
find_depformer_tensor(const emel::model::data &model, const int32_t layer,
                      const char *suffix) noexcept {
  char buffer[160] = {};
  const int written = std::snprintf(buffer, sizeof(buffer),
                                    "lm.depformer.layers.%d.%s", layer, suffix);
  if (written <= 0 || static_cast<size_t>(written) >= sizeof(buffer)) {
    return nullptr;
  }
  return find_tensor(model,
                     std::string_view{buffer, static_cast<size_t>(written)});
}

inline const tensor_record *
find_depformer_codebook_tensor(const emel::model::data &model,
                               const int32_t layer, const char *format,
                               const int32_t codebook) noexcept {
  char suffix[128] = {};
  const int suffix_written =
      std::snprintf(suffix, sizeof(suffix), format, codebook);
  if (suffix_written <= 0 ||
      static_cast<size_t>(suffix_written) >= sizeof(suffix)) {
    return nullptr;
  }
  return find_depformer_tensor(model, layer, suffix);
}

inline const tensor_record *
find_depformer_projection(const emel::model::data &model, const int32_t layer,
                          const int32_t codebook) noexcept {
  const auto *split = find_depformer_codebook_tensor(
      model, layer, "self_attn.in_projs.%d.weight", codebook);
  if (split != nullptr) {
    return split;
  }
  return find_depformer_tensor(model, layer, "self_attn.in_proj_weight");
}

inline bool supported_get_rows_dtype(const dtype type) noexcept {
  const auto code = emel::kernel::detail::dtype_code(type);
  return code == emel::kernel::detail::dtype_f32 ||
         code == emel::kernel::detail::dtype_f16 ||
         code == emel::kernel::detail::dtype_bf16 ||
         code == emel::kernel::detail::dtype_q4_0 ||
         code == emel::kernel::detail::dtype_q8_0 ||
         code == emel::kernel::detail::dtype_q4_k;
}

inline bool supported_argmax_dtype(const dtype type) noexcept {
  const auto code = emel::kernel::detail::dtype_code(type);
  return code == emel::kernel::detail::dtype_f32 ||
         emel::kernel::detail::is_native_quantized_dtype(code) ||
         emel::kernel::detail::is_packed_q4_vector_dtype(code);
}

inline bool supported_mul_mat_dtype(const dtype type) noexcept {
  const auto code = emel::kernel::detail::dtype_code(type);
  return code == emel::kernel::detail::dtype_f32 ||
         code == emel::kernel::detail::dtype_f16 ||
         emel::kernel::detail::is_native_quantized_dtype(code) ||
         emel::kernel::detail::is_packed_q4_vector_dtype(code);
}

inline bool is_quantized_row_tensor(const dtype type) noexcept {
  const auto code = emel::kernel::detail::dtype_code(type);
  return emel::kernel::detail::is_native_quantized_dtype(code) ||
         emel::kernel::detail::is_packed_q4_vector_dtype(code);
}

inline bool bind_tensor_view(const tensor_record &tensor,
                             tensor_view &view) noexcept {
  if (tensor.data == nullptr || tensor.n_dims <= 0 || tensor.dims[0] <= 0) {
    return false;
  }

  view = {};
  view.data = tensor.data;
  view.type = static_cast<dtype>(tensor.type);
  view.ne = {
      static_cast<uint64_t>(tensor.dims[0]),
      tensor.n_dims > 1 ? static_cast<uint64_t>(tensor.dims[1]) : 1u,
      tensor.n_dims > 2 ? static_cast<uint64_t>(tensor.dims[2]) : 1u,
      tensor.n_dims > 3 ? static_cast<uint64_t>(tensor.dims[3]) : 1u,
  };

  if (is_quantized_row_tensor(view.type)) {
    const uint8_t code = emel::kernel::detail::dtype_code(view.type);
    if (emel::kernel::detail::is_packed_q4_vector_dtype(code)) {
      const uint64_t group_count =
          emel::kernel::detail::quant::packed_q4_k_x8_group_count(view.ne[1]);
      const size_t group_bytes =
          emel::kernel::detail::quant::packed_q4_k_x8_group_storage_bytes(
              view.ne[0]);
      if (group_count == 0u || group_bytes == 0u) {
        return false;
      }
      view.nb[0] = 1u;
      view.nb[1] = group_bytes;
      view.nb[2] = group_bytes * group_count;
      view.nb[3] = view.nb[2];
      return true;
    }
    const size_t row_bytes =
        emel::kernel::detail::quantized_row_storage_bytes(code, view.ne[0]);
    if (row_bytes == 0u) {
      return false;
    }
    view.nb[0] = 1u;
    view.nb[1] = row_bytes;
    view.nb[2] = row_bytes * view.ne[1];
    view.nb[3] = view.nb[2] * view.ne[2];
    return true;
  }

  fill_default_nb(view);
  return view.nb[0] != 0u;
}

inline tensor_view make_f32_src(const float *data, const uint64_t ne0,
                                const uint64_t ne1 = 1u) noexcept {
  tensor_view view{};
  view.data = data;
  view.type = dtype::f32;
  view.ne = {ne0, ne1, 1u, 1u};
  fill_default_nb(view);
  return view;
}

inline tensor_view make_i32_src(const int32_t *data,
                                const uint64_t ne0) noexcept {
  tensor_view view{};
  view.data = data;
  view.type = dtype::i32;
  view.ne = {ne0, 1u, 1u, 1u};
  fill_default_nb(view);
  return view;
}

inline tensor_view_mut make_f32_dst(float *data, const uint64_t ne0,
                                    const uint64_t ne1 = 1u) noexcept {
  tensor_view_mut view{};
  view.data = data;
  view.type = dtype::f32;
  view.ne = {ne0, ne1, 1u, 1u};
  fill_default_nb(view);
  return view;
}

template <class event_type>
inline void set_op_param_f32(event_type &ev, const uint32_t slot,
                             const float value) noexcept {
  const auto offset = slot * static_cast<uint32_t>(sizeof(float));
  if (offset + sizeof(float) <= ev.op_params.size()) {
    auto *bytes = ev.op_params.data() + offset;
    std::memcpy(bytes, &value, sizeof(float));
    ev.op_params_size = offset + static_cast<uint32_t>(sizeof(float));
  }
}

template <class event_type>
inline void set_op_param_i32(event_type &ev, const uint32_t slot,
                             const int32_t value) noexcept {
  const auto offset = slot * static_cast<uint32_t>(sizeof(int32_t));
  if (offset + sizeof(int32_t) <= ev.op_params.size()) {
    auto *bytes = ev.op_params.data() + offset;
    std::memcpy(bytes, &value, sizeof(int32_t));
    ev.op_params_size = offset + static_cast<uint32_t>(sizeof(int32_t));
  }
}

inline bool tensor_shape(const tensor_record *tensor, const int64_t ne0,
                         const int64_t ne1 = 1) noexcept {
  return tensor != nullptr && tensor->data != nullptr && tensor->n_dims >= 1 &&
         tensor->dims[0] == ne0 &&
         (tensor->n_dims == 1 || tensor->dims[1] == ne1);
}

inline bool token_in_embedding_range(const int32_t token,
                                     const int32_t card) noexcept {
  return token >= k_token_zero && token <= card;
}

} // namespace emel::speech::predictor::moshi::executor::detail
