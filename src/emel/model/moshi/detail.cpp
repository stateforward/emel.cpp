#include "emel/model/moshi/detail.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <limits>

#include "emel/gguf/loader/detail.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"

namespace emel::model::moshi::detail {

namespace {

constexpr std::string_view k_architecture = "moshi";
constexpr std::string_view k_component_key = "moshi.component";
constexpr std::string_view k_component_lm = "lm";
constexpr std::string_view k_component_mimi = "mimi";
constexpr std::string_view k_component_voice = "voice";
constexpr std::string_view k_gating_silu = "silu";
constexpr std::string_view k_norm_rms_f32 = "rms_norm_f32";
constexpr std::string_view k_pos_emb_rope = "rope";
constexpr std::string_view k_pos_emb_none = "none";
constexpr std::string_view k_voice_format_personaplex = "personaplex_prompt_v1";

constexpr std::string_view k_lm_prefix = "lm.";
constexpr std::string_view k_lm_text_emb_prefix = "lm.text_emb.";
constexpr std::string_view k_lm_audio_emb_prefix = "lm.emb.";
constexpr std::string_view k_lm_transformer_prefix = "lm.transformer.";
constexpr std::string_view k_lm_out_norm_prefix = "lm.out_norm.";
constexpr std::string_view k_lm_text_linear_prefix = "lm.text_linear.";
constexpr std::string_view k_lm_linears_prefix = "lm.linears.";
constexpr std::string_view k_lm_depformer_in_prefix = "lm.depformer_in.";
constexpr std::string_view k_lm_depformer_prefix = "lm.depformer.";
constexpr std::string_view k_lm_depformer_text_emb_prefix =
    "lm.depformer_text_emb.";
constexpr std::string_view k_lm_depformer_emb_prefix = "lm.depformer_emb.";

constexpr std::string_view k_mimi_prefix = "mimi.";
constexpr std::string_view k_mimi_encoder_prefix = "mimi.encoder.";
constexpr std::string_view k_mimi_encoder_transformer_prefix =
    "mimi.encoder_transformer.";
constexpr std::string_view k_mimi_downsample_prefix = "mimi.downsample.";
constexpr std::string_view k_mimi_quantizer_prefix = "mimi.quantizer.";
constexpr std::string_view k_mimi_upsample_prefix = "mimi.upsample.";
constexpr std::string_view k_mimi_decoder_transformer_prefix =
    "mimi.decoder_transformer.";
constexpr std::string_view k_mimi_decoder_prefix = "mimi.decoder.";

constexpr std::string_view k_voice_prefix = "voice.";
constexpr std::string_view k_voice_embeddings_name = "voice.embeddings";
constexpr std::string_view k_voice_cache_name = "voice.cache";

bool require_u32_as_i32(const emel::model::detail::kv_binding &binding,
                        const std::string_view key, int32_t &field) noexcept {
  uint64_t value = 0u;
  const auto *entry = emel::model::detail::find_kv_entry(binding, key);
  if (entry == nullptr ||
      !emel::model::detail::decode_integer_value(binding, *entry, value) ||
      value > static_cast<uint64_t>(INT32_MAX)) {
    return false;
  }

  field = static_cast<int32_t>(value);
  return true;
}

bool require_positive_i32(const emel::model::detail::kv_binding &binding,
                          const std::string_view key, int32_t &field) noexcept {
  return require_u32_as_i32(binding, key, field) && field > 0;
}

bool require_string(const emel::model::detail::kv_binding &binding,
                    const std::string_view key,
                    const std::string_view expected) noexcept {
  const auto *entry = emel::model::detail::find_kv_entry(binding, key);
  if (entry == nullptr) {
    return false;
  }

  std::string_view value = {};
  return emel::model::detail::decode_string_value(binding, *entry, value) &&
         value == expected;
}

bool require_bool(const emel::model::detail::kv_binding &binding,
                  const std::string_view key, bool &field) noexcept {
  const auto *entry = emel::model::detail::find_kv_entry(binding, key);
  if (entry == nullptr) {
    return false;
  }

  bool value = false;
  if (!emel::model::detail::decode_bool_value(binding, *entry, value)) {
    return false;
  }

  field = value;
  return true;
}

bool assign_optional_i32(const emel::model::detail::kv_binding &binding,
                         const std::string_view key, int32_t &field) noexcept {
  const auto *entry = emel::model::detail::find_kv_entry(binding, key);
  if (entry == nullptr) {
    return true;
  }

  return require_u32_as_i32(binding, key, field);
}

bool require_nonnegative_i32(const emel::model::detail::kv_binding &binding,
                             const std::string_view key,
                             int32_t &field) noexcept {
  return require_u32_as_i32(binding, key, field) && field >= 0;
}

bool require_f32(const emel::model::detail::hparam_loader &loader,
                 const std::string_view key, float &field) noexcept {
  const auto *entry = emel::model::detail::find_kv_entry(loader.binding, key);
  return entry != nullptr && loader.assign_f32(key, field);
}

bool require_i32_array(const emel::model::detail::kv_binding &binding,
                       const std::string_view key, const std::span<int32_t> dst,
                       uint32_t &count_out) noexcept {
  const auto *entry = emel::model::detail::find_kv_entry(binding, key);
  if (entry == nullptr) {
    return false;
  }

  emel::model::detail::array_header header = {};
  if (!emel::model::detail::decode_array_header(binding, *entry, header) ||
      header.count == 0u || header.count > dst.size()) {
    return false;
  }

  for (uint32_t index = 0u; index < header.count; ++index) {
    uint64_t value = 0u;
    if (!emel::model::detail::decode_uint_array_element(binding, *entry, index,
                                                        value) ||
        value > static_cast<uint64_t>(INT32_MAX)) {
      return false;
    }
    dst[index] = static_cast<int32_t>(value);
  }

  count_out = static_cast<uint32_t>(header.count);
  return true;
}

bool assign_optional_i32_array(
    const emel::model::detail::kv_binding &binding, const std::string_view key,
    const std::span<int32_t> dst, uint32_t &count_out) noexcept {
  const auto *entry = emel::model::detail::find_kv_entry(binding, key);
  if (entry == nullptr) {
    count_out = 0u;
    return true;
  }
  return require_i32_array(binding, key, dst, count_out);
}

bool validate_lm_inference_hparams(
    const emel::model::data::moshi_lm_hparams &lm) noexcept {
  if (!lm.depformer_weights_per_step) {
    return true;
  }

  const uint32_t codebook_count = static_cast<uint32_t>(lm.n_q) + 1u;
  if (lm.inference_dep_q <= 0 || lm.inference_dep_q > lm.dep_q ||
      lm.inference_dep_q > lm.n_q ||
      lm.inference_prompt_token_count != codebook_count ||
      lm.inference_pre_text_silence_frames < 0 ||
      lm.inference_post_text_silence_frames < 0) {
    return false;
  }

  if (lm.inference_prompt_tokens[0] < 0 ||
      lm.inference_prompt_tokens[0] >= lm.text_card) {
    return false;
  }
  for (uint32_t index = 1u; index < codebook_count; ++index) {
    const int32_t token = lm.inference_prompt_tokens[index];
    if (token < 0 || token >= lm.card) {
      return false;
    }
  }

  if (lm.depformer_weight_schedule_count > 0u) {
    if (lm.depformer_weight_schedule_count != static_cast<uint32_t>(lm.dep_q)) {
      return false;
    }
    for (uint32_t index = 0u; index < lm.depformer_weight_schedule_count;
         ++index) {
      const int32_t weight_index = lm.depformer_weight_schedule[index];
      if (weight_index < 0 || weight_index >= lm.dep_q) {
        return false;
      }
    }
  }

  return true;
}

bool load_lm_hparams(const emel::model::detail::hparam_loader &loader,
                     emel::model::data &model_out) noexcept {
  auto &lm = model_out.moshi_lm;
  const auto &binding = loader.binding;

  if (!require_positive_i32(binding, "moshi.lm.card", lm.card) ||
      !require_positive_i32(binding, "moshi.lm.n_q", lm.n_q) ||
      !require_u32_as_i32(binding, "moshi.lm.dep_q", lm.dep_q) ||
      !require_positive_i32(binding, "moshi.lm.text_card", lm.text_card) ||
      !require_u32_as_i32(binding, "moshi.lm.existing_text_padding_id",
                          lm.text_padding_id) ||
      !require_positive_i32(binding, "moshi.lm.dim", lm.dim) ||
      !require_positive_i32(binding, "moshi.lm.num_layers", lm.num_layers) ||
      !require_positive_i32(binding, "moshi.lm.num_heads", lm.num_heads) ||
      !require_positive_i32(binding, "moshi.lm.context", lm.context) ||
      !require_positive_i32(binding, "moshi.lm.max_period", lm.max_period) ||
      !require_positive_i32(binding, "moshi.lm.dim_feedforward",
                            lm.dim_feedforward) ||
      !require_string(binding, "moshi.lm.gating", k_gating_silu) ||
      !require_string(binding, "moshi.lm.norm", k_norm_rms_f32) ||
      !require_string(binding, "moshi.lm.positional_embedding",
                      k_pos_emb_rope) ||
      !require_bool(binding, "moshi.lm.causal", lm.causal) ||
      !require_bool(binding, "moshi.lm.cross_attention", lm.cross_attention) ||
      !require_bool(binding, "moshi.lm.demux_second_stream",
                    lm.demux_second_stream) ||
      !require_i32_array(binding, "moshi.lm.delays",
                         std::span<int32_t>{lm.delays}, lm.delay_count) ||
      !assign_optional_i32(binding, "moshi.lm.extra_heads.num_heads",
                           lm.extra_heads_num_heads)) {
    return false;
  }

  const bool consistent =
      lm.dep_q >= 0 && lm.dep_q <= lm.n_q &&
      // n_q < k_max_delays bounds the n_q + 1 delay slot without evaluating
      // a signed addition that can overflow on a malformed n_q.
      lm.n_q < emel::model::data::moshi_lm_hparams::k_max_delays &&
      lm.delay_count == static_cast<uint32_t>(lm.n_q) + 1u &&
      lm.dim % lm.num_heads == 0 && lm.text_padding_id >= 0 &&
      lm.text_padding_id < lm.text_card;
  if (!consistent) {
    return false;
  }

  if (lm.dep_q > 0 &&
      (!require_positive_i32(binding, "moshi.lm.depformer.dim",
                             lm.depformer_dim) ||
       !require_positive_i32(binding, "moshi.lm.depformer.num_heads",
                             lm.depformer_num_heads) ||
       !require_positive_i32(binding, "moshi.lm.depformer.num_layers",
                             lm.depformer_num_layers) ||
       !require_positive_i32(binding, "moshi.lm.depformer.dim_feedforward",
                             lm.depformer_dim_feedforward) ||
       !require_positive_i32(binding, "moshi.lm.depformer.context",
                             lm.depformer_context) ||
       !require_positive_i32(binding, "moshi.lm.depformer.max_period",
                             lm.depformer_max_period) ||
       !require_string(binding, "moshi.lm.depformer.gating", k_gating_silu) ||
       !require_string(binding, "moshi.lm.depformer.pos_emb", k_pos_emb_none) ||
       !require_bool(binding, "moshi.lm.depformer.multi_linear",
                     lm.depformer_multi_linear) ||
       !require_bool(binding, "moshi.lm.depformer.weights_per_step",
                     lm.depformer_weights_per_step) ||
       !assign_optional_i32_array(
           binding, "moshi.lm.depformer.weights_per_step_schedule",
           std::span<int32_t>{lm.depformer_weight_schedule},
           lm.depformer_weight_schedule_count) ||
       !assign_optional_i32(binding, "moshi.lm.depformer.low_rank_embeddings",
                            lm.depformer_low_rank_embeddings) ||
       !lm.depformer_multi_linear ||
       lm.depformer_dim % lm.depformer_num_heads != 0)) {
    return false;
  }

  if (lm.depformer_weights_per_step &&
      (!require_positive_i32(binding, "moshi.lm.inference.dep_q",
                             lm.inference_dep_q) ||
       !require_nonnegative_i32(binding,
                                "moshi.lm.inference.pre_text_silence_frames",
                                lm.inference_pre_text_silence_frames) ||
       !require_nonnegative_i32(binding,
                                "moshi.lm.inference.post_text_silence_frames",
                                lm.inference_post_text_silence_frames) ||
       !require_i32_array(binding, "moshi.lm.inference.prompt_tokens",
                          std::span<int32_t>{lm.inference_prompt_tokens},
                          lm.inference_prompt_token_count))) {
    return false;
  }

  if (!validate_lm_inference_hparams(lm)) {
    return false;
  }

  model_out.moshi_component_id = emel::model::data::moshi_component::lm;
  model_out.params.n_embd = lm.dim;
  model_out.params.n_embd_out = lm.dim;
  model_out.params.n_layer = lm.num_layers;
  model_out.params.n_head = lm.num_heads;
  model_out.params.n_head_kv = lm.num_heads;
  model_out.params.n_ff = lm.dim_feedforward;
  model_out.params.n_ctx = lm.context;
  model_out.params.n_vocab = lm.text_card;
  model_out.params.n_rot = lm.dim / lm.num_heads;
  model_out.params.rope_freq_base = static_cast<float>(lm.max_period);
  model_out.params.attention_causal = lm.causal;
  model_out.params.decoder_block_count = lm.depformer_num_layers;
  return true;
}

bool load_mimi_hparams(const emel::model::detail::hparam_loader &loader,
                       emel::model::data &model_out) noexcept {
  auto &mimi = model_out.mimi;
  const auto &binding = loader.binding;

  if (!require_positive_i32(binding, "moshi.mimi.sample_rate",
                            mimi.sample_rate) ||
      !require_f32(loader, "moshi.mimi.frame_rate", mimi.frame_rate) ||
      !require_positive_i32(binding, "moshi.mimi.n_q", mimi.n_q) ||
      !require_positive_i32(binding, "moshi.mimi.card", mimi.card) ||
      !require_positive_i32(binding, "moshi.mimi.dim", mimi.dim) ||
      !require_positive_i32(binding, "moshi.mimi.semantic_n_q",
                            mimi.semantic_n_q) ||
      !require_positive_i32(binding, "moshi.mimi.codebook_dim",
                            mimi.codebook_dim) ||
      !require_positive_i32(binding, "moshi.mimi.transformer.num_layers",
                            mimi.transformer_num_layers) ||
      !require_positive_i32(binding, "moshi.mimi.transformer.num_heads",
                            mimi.transformer_num_heads) ||
      !require_positive_i32(binding, "moshi.mimi.transformer.context",
                            mimi.transformer_context) ||
      !require_positive_i32(binding, "moshi.mimi.transformer.max_period",
                            mimi.transformer_max_period) ||
      !std::isfinite(mimi.frame_rate) || mimi.frame_rate <= 0.0f ||
      mimi.semantic_n_q >= mimi.n_q ||
      mimi.dim % mimi.transformer_num_heads != 0 ||
      // The codec planner's rotary kernel halves head_dim, so an odd
      // per-head size must reject at the hparam gate too.
      ((mimi.dim / mimi.transformer_num_heads) % 2) != 0) {
    return false;
  }

  model_out.moshi_component_id = emel::model::data::moshi_component::mimi;
  model_out.params.n_embd = mimi.dim;
  model_out.params.n_embd_out = mimi.dim;
  model_out.params.n_layer = mimi.transformer_num_layers;
  model_out.params.n_head = mimi.transformer_num_heads;
  model_out.params.n_head_kv = mimi.transformer_num_heads;
  model_out.params.n_ctx = mimi.transformer_context;
  model_out.params.n_features = mimi.n_q;
  model_out.params.rope_freq_base =
      static_cast<float>(mimi.transformer_max_period);
  return true;
}

bool load_voice_hparams(const emel::model::detail::hparam_loader &loader,
                        emel::model::data &model_out) noexcept {
  if (!require_string(loader.binding, "moshi.voice.format",
                      k_voice_format_personaplex)) {
    return false;
  }

  model_out.moshi_component_id = emel::model::data::moshi_component::voice;
  return true;
}

bool tensor_has_storage(
    const emel::model::data::tensor_record &tensor) noexcept {
  if (tensor.data == nullptr || tensor.data_size == 0u || tensor.n_dims <= 0 ||
      tensor.n_dims > static_cast<int32_t>(tensor.dims.size())) {
    return false;
  }

  std::array<uint64_t, 4> dims = {1u, 1u, 1u, 1u};
  for (int32_t dim = 0; dim < tensor.n_dims; ++dim) {
    if (tensor.dims[static_cast<size_t>(dim)] <= 0) {
      return false;
    }
    dims[static_cast<size_t>(dim)] =
        static_cast<uint64_t>(tensor.dims[static_cast<size_t>(dim)]);
  }

  uint64_t required_bytes = 0u;
  const emel::error::type size_err =
      emel::gguf::loader::detail::compute_tensor_data_size(
          dims, static_cast<uint32_t>(tensor.n_dims),
          static_cast<uint32_t>(tensor.type), required_bytes);
  if (size_err != emel::error::cast(emel::gguf::loader::error::none)) {
    return false;
  }

  return tensor.data_size >= required_bytes;
}

bool tensor_has_lm_execution_storage(
    const emel::model::data::tensor_record &tensor) noexcept {
  if (!tensor_has_storage(tensor)) {
    return false;
  }

  // Keep this list aligned with the maintained generator row dequantization
  // path in text/generator/detail.hpp. GGUF may define more layouts than the
  // LM runtime can consume; storage-known is not execution-supported.
  switch (tensor.type) {
  case 0:  // F32
  case 2:  // Q4_0
  case 8:  // Q8_0
  case 10: // Q2_K
  case 11: // Q3_K
  case 12: // Q4_K
  case 14: // Q6_K
  case 42: // EMEL Q4_K x8 BL8
    return true;
  default:
    return false;
  }
}

bool assign_family_view(
    const emel::model::data &model_data, const std::string_view prefix,
    family_view &family_out,
    const bool require_lm_execution_storage = false) noexcept {
  family_out = {};
  family_out.prefix = prefix;

  for (uint32_t index = 0u; index < model_data.n_tensors; ++index) {
    const auto &tensor = model_data.tensors[index];
    const auto name = emel::model::tensor_name_view(model_data, tensor);
    const bool has_storage = require_lm_execution_storage
                                 ? tensor_has_lm_execution_storage(tensor)
                                 : tensor_has_storage(tensor);
    if (!name.starts_with(prefix) || !has_storage) {
      continue;
    }

    if (family_out.tensor_count == 0u) {
      family_out.first.tensor = &tensor;
      family_out.first.name = name;
    }
    ++family_out.tensor_count;
  }

  return family_out.tensor_count > 0u;
}

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

bool bind_exact_tensor(const emel::model::data &model_data,
                       const std::string_view name,
                       tensor_view &view_out) noexcept {
  const auto *tensor = find_tensor(model_data, name);
  if (tensor == nullptr) {
    return false;
  }
  view_out = tensor_view{
      .tensor = tensor,
      .name = emel::model::tensor_name_view(model_data, *tensor),
  };
  return true;
}

bool has_tensor(const emel::model::data &model_data,
                const std::string_view name) noexcept {
  const auto *tensor = find_tensor(model_data, name);
  return tensor != nullptr && tensor_has_storage(*tensor);
}

// The codec bind's prepare_vector/prepare_matrix_raw/prepare_linear helpers
// accept any dims layout with the expected total element count, so contract
// probes for those tensors must compare element counts, not exact shapes.
bool has_tensor_with_elements(const emel::model::data &model_data,
                              const std::string_view name,
                              const uint64_t expected_elements) noexcept {
  const auto *tensor = find_tensor(model_data, name);
  if (tensor == nullptr || !tensor_has_storage(*tensor)) {
    return false;
  }
  uint64_t elements = 1u;
  for (int32_t dim = 0; dim < tensor->n_dims; ++dim) {
    elements *= static_cast<uint64_t>(tensor->dims[static_cast<size_t>(dim)]);
  }
  return elements == expected_elements;
}

// The codec bind consumes vectors, codebooks, and float-class matrices
// through prepare helpers that accept only f32/f16 (tensor_is_float), so
// contract probes for those tensors must require the float class, not merely
// enough stored bytes.
bool has_float_tensor_with_elements(const emel::model::data &model_data,
                                    const std::string_view name,
                                    const uint64_t expected_elements) noexcept {
  constexpr int32_t k_dtype_f32 = 0;
  constexpr int32_t k_dtype_f16 = 1;
  const auto *tensor = find_tensor(model_data, name);
  return tensor != nullptr &&
         (tensor->type == k_dtype_f32 || tensor->type == k_dtype_f16) &&
         has_tensor_with_elements(model_data, name, expected_elements);
}

// The codec plan selects the q8 bind path from the first projection of a
// family and the bind then requires every projection in that family to share
// the class, so projection probes compare element count and dtype class
// together. The float class must be one of the dtypes the binder actually
// consumes (prepare_matrix_raw/prepare_linear accept f32/f16 only), not
// merely any non-q8 payload of sufficient size.
bool has_projection_tensor(const emel::model::data &model_data,
                           const std::string_view name,
                           const uint64_t expected_elements,
                           const bool q8_class) noexcept {
  constexpr int32_t k_dtype_f32 = 0;
  constexpr int32_t k_dtype_f16 = 1;
  constexpr int32_t k_dtype_q8_0 = 8;
  const auto *tensor = find_tensor(model_data, name);
  if (tensor == nullptr ||
      !has_tensor_with_elements(model_data, name, expected_elements)) {
    return false;
  }
  return q8_class ? tensor->type == k_dtype_q8_0
                  : tensor->type == k_dtype_f32 || tensor->type == k_dtype_f16;
}

bool require_tensor_shape(const emel::model::data &model_data,
                          const std::string_view name,
                          const std::initializer_list<int64_t> dims) noexcept {
  const auto *tensor = find_tensor(model_data, name);
  if (tensor == nullptr || !tensor_has_storage(*tensor) ||
      tensor->n_dims != static_cast<int32_t>(dims.size())) {
    return false;
  }

  size_t index = 0u;
  for (const int64_t dim : dims) {
    if (tensor->dims[index] != dim) {
      return false;
    }
    ++index;
  }

  return true;
}

bool has_indexed_tensor(const emel::model::data &model_data, const char *format,
                        const int32_t index) noexcept {
  char buffer[96] = {};
  const int written = std::snprintf(buffer, sizeof(buffer), format, index);
  return written > 0 && static_cast<size_t>(written) < sizeof(buffer) &&
         has_tensor(model_data,
                    std::string_view{buffer, static_cast<size_t>(written)});
}

// A transformer self-attention block stores its input projection either fused
// (`in_proj_weight`, the safetensors layout moshi.cpp caches verbatim) or
// pre-split (`in_projs.0.weight`).
bool has_lm_transformer_block(const emel::model::data &model_data,
                              const int32_t block_index) noexcept {
  return has_indexed_tensor(model_data, "lm.transformer.layers.%d.norm1.alpha",
                            block_index) &&
         has_indexed_tensor(model_data, "lm.transformer.layers.%d.norm2.alpha",
                            block_index) &&
         (has_indexed_tensor(
              model_data, "lm.transformer.layers.%d.self_attn.in_proj_weight",
              block_index) ||
          has_indexed_tensor(
              model_data,
              "lm.transformer.layers.%d.self_attn.in_projs.0.weight",
              block_index)) &&
         (has_indexed_tensor(model_data,
                             "lm.transformer.layers.%d.gating.linear_in.weight",
                             block_index) ||
          has_indexed_tensor(
              model_data, "lm.transformer.layers.%d.gating.0.linear_in.weight",
              block_index));
}

bool has_depformer_block(const emel::model::data &model_data,
                         const int32_t block_index) noexcept {
  return has_indexed_tensor(model_data, "lm.depformer.layers.%d.norm1.alpha",
                            block_index) &&
         has_indexed_tensor(model_data, "lm.depformer.layers.%d.norm2.alpha",
                            block_index) &&
         (has_indexed_tensor(model_data,
                             "lm.depformer.layers.%d.self_attn.in_proj_weight",
                             block_index) ||
          has_indexed_tensor(
              model_data, "lm.depformer.layers.%d.self_attn.in_projs.0.weight",
              block_index));
}

uint32_t count_tensors_with_storage(
    const emel::model::data &model_data,
    const bool require_lm_execution_storage = false) noexcept {
  uint32_t count = 0u;
  for (uint32_t index = 0u; index < model_data.n_tensors; ++index) {
    const bool has_storage =
        require_lm_execution_storage
            ? tensor_has_lm_execution_storage(model_data.tensors[index])
            : tensor_has_storage(model_data.tensors[index]);
    if (has_storage) {
      ++count;
    }
  }

  return count;
}

emel::error::type validate_lm_contract(const emel::model::data &model_data,
                                       lm_contract &contract_out) noexcept {
  const auto &lm = model_data.moshi_lm;
  family_view component = {};
  const bool families_ok =
      assign_family_view(model_data, k_lm_prefix, component, true) &&
      component.tensor_count == count_tensors_with_storage(model_data) &&
      assign_family_view(model_data, k_lm_text_emb_prefix,
                         contract_out.text_emb, true) &&
      assign_family_view(model_data, k_lm_audio_emb_prefix,
                         contract_out.audio_emb, true) &&
      contract_out.audio_emb.tensor_count >= static_cast<uint32_t>(lm.n_q) &&
      assign_family_view(model_data, k_lm_transformer_prefix,
                         contract_out.transformer, true) &&
      assign_family_view(model_data, k_lm_out_norm_prefix,
                         contract_out.out_norm, true) &&
      assign_family_view(model_data, k_lm_text_linear_prefix,
                         contract_out.text_linear, true);
  if (!families_ok) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const int64_t dim = lm.dim;
  const int64_t text_card = lm.text_card;
  const int64_t card = lm.card;
  const bool shapes_ok =
      require_tensor_shape(model_data, "lm.text_emb.weight",
                           {dim, text_card + 1}) &&
      require_tensor_shape(model_data, "lm.text_linear.weight",
                           {dim, text_card}) &&
      require_tensor_shape(model_data, "lm.emb.0.weight", {dim, card + 1}) &&
      has_tensor(model_data, "lm.out_norm.alpha");
  if (!shapes_ok) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  if (lm.n_q > emel::model::data::moshi_lm_hparams::k_max_delays) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  const auto *text_embedding = find_tensor(model_data, "lm.text_emb.weight");
  contract_out.text_embedding = tensor_view{
      .tensor = text_embedding,
      .name = emel::model::tensor_name_view(model_data, *text_embedding),
  };
  const auto *output_norm = find_tensor(model_data, "lm.out_norm.alpha");
  contract_out.output_norm = tensor_view{
      .tensor = output_norm,
      .name = emel::model::tensor_name_view(model_data, *output_norm),
  };
  const auto *text_output_projection =
      find_tensor(model_data, "lm.text_linear.weight");
  contract_out.text_output_projection = tensor_view{
      .tensor = text_output_projection,
      .name =
          emel::model::tensor_name_view(model_data, *text_output_projection),
  };
  for (int32_t codebook = 0; codebook < lm.n_q; ++codebook) {
    char name[96] = {};
    const int written =
        std::snprintf(name, sizeof(name), "lm.emb.%d.weight", codebook);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(name)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    const auto *audio_embedding = find_tensor(
        model_data, std::string_view{name, static_cast<size_t>(written)});
    if (audio_embedding == nullptr) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    contract_out.audio_embeddings[static_cast<size_t>(codebook)] = tensor_view{
        .tensor = audio_embedding,
        .name = emel::model::tensor_name_view(model_data, *audio_embedding),
    };
  }

  if (lm.num_layers > emel::model::data::moshi_lm_hparams::k_max_delays) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  for (int32_t block = 0; block < lm.num_layers; ++block) {
    if (!has_lm_transformer_block(model_data, block)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    auto &layer = contract_out.temporal_layers[static_cast<size_t>(block)];
    char name[128] = {};
    int written = std::snprintf(name, sizeof(name),
                                "lm.transformer.layers.%d.norm1.alpha", block);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
        !bind_exact_tensor(model_data,
                           std::string_view{name, static_cast<size_t>(written)},
                           layer.norm1)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    written = std::snprintf(
        name, sizeof(name),
        "lm.transformer.layers.%d.self_attn.in_projs.0.weight", block);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(name)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    (void)bind_exact_tensor(
        model_data, std::string_view{name, static_cast<size_t>(written)},
        layer.split_input_projection);
    written = std::snprintf(name, sizeof(name),
                            "lm.transformer.layers.%d.self_attn.in_proj_weight",
                            block);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(name)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    (void)bind_exact_tensor(
        model_data, std::string_view{name, static_cast<size_t>(written)},
        layer.fused_input_projection);
    written = std::snprintf(
        name, sizeof(name),
        "lm.transformer.layers.%d.self_attn.out_projs.0.weight", block);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
        !bind_exact_tensor(model_data,
                           std::string_view{name, static_cast<size_t>(written)},
                           layer.output_projection)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    written = std::snprintf(name, sizeof(name),
                            "lm.transformer.layers.%d.norm2.alpha", block);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
        !bind_exact_tensor(model_data,
                           std::string_view{name, static_cast<size_t>(written)},
                           layer.norm2)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    written = std::snprintf(name, sizeof(name),
                            "lm.transformer.layers.%d.gating.linear_in.weight",
                            block);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
        !bind_exact_tensor(model_data,
                           std::string_view{name, static_cast<size_t>(written)},
                           layer.gating_input)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    written = std::snprintf(name, sizeof(name),
                            "lm.transformer.layers.%d.gating.linear_out.weight",
                            block);
    if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
        !bind_exact_tensor(model_data,
                           std::string_view{name, static_cast<size_t>(written)},
                           layer.gating_output)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
  }

  if (lm.dep_q > 0) {
    const bool depformer_ok =
        assign_family_view(model_data, k_lm_linears_prefix,
                           contract_out.linears, true) &&
        contract_out.linears.tensor_count >= static_cast<uint32_t>(lm.dep_q) &&
        assign_family_view(model_data, k_lm_depformer_in_prefix,
                           contract_out.depformer_in, true) &&
        assign_family_view(model_data, k_lm_depformer_prefix,
                           contract_out.depformer, true) &&
        assign_family_view(model_data, k_lm_depformer_text_emb_prefix,
                           contract_out.depformer_text_emb, true) &&
        (lm.dep_q < 2 ||
         (assign_family_view(model_data, k_lm_depformer_emb_prefix,
                             contract_out.depformer_emb, true) &&
          contract_out.depformer_emb.tensor_count >=
              static_cast<uint32_t>(lm.dep_q) - 1u));
    if (!depformer_ok) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    std::array<bool, emel::model::data::moshi_lm_hparams::k_max_delays>
        required_weight_indices = {};
    if (lm.depformer_weights_per_step &&
        lm.depformer_weight_schedule_count > 0u) {
      for (uint32_t index = 0u; index < lm.depformer_weight_schedule_count;
           ++index) {
        const int32_t weight_index = lm.depformer_weight_schedule[index];
        if (weight_index < 0 || weight_index >= lm.dep_q) {
          return emel::error::cast(emel::model::loader::error::model_invalid);
        }
        required_weight_indices[static_cast<size_t>(weight_index)] = true;
      }
    } else {
      std::fill_n(required_weight_indices.begin(), lm.dep_q, true);
    }

    const auto *depformer_text_embedding =
        find_tensor(model_data, "lm.depformer_text_emb.weight");
    if (depformer_text_embedding == nullptr) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    contract_out.depformer_text_embedding = tensor_view{
        .tensor = depformer_text_embedding,
        .name = emel::model::tensor_name_view(model_data,
                                              *depformer_text_embedding),
    };
    for (int32_t codebook = 0; codebook < lm.dep_q; ++codebook) {
      if (!required_weight_indices[static_cast<size_t>(codebook)]) {
        continue;
      }
      char name[96] = {};
      int written = std::snprintf(name, sizeof(name),
                                  "lm.depformer_in.%d.weight", codebook);
      if (written <= 0 || static_cast<size_t>(written) >= sizeof(name)) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
      }
      const auto *input_projection = find_tensor(
          model_data, std::string_view{name, static_cast<size_t>(written)});
      if (input_projection == nullptr) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
      }
      contract_out.depformer_input_projections[static_cast<size_t>(codebook)] =
          tensor_view{
              .tensor = input_projection,
              .name =
                  emel::model::tensor_name_view(model_data, *input_projection),
          };

      written =
          std::snprintf(name, sizeof(name), "lm.linears.%d.weight", codebook);
      if (written <= 0 || static_cast<size_t>(written) >= sizeof(name)) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
      }
      const auto *output_projection = find_tensor(
          model_data, std::string_view{name, static_cast<size_t>(written)});
      if (output_projection == nullptr) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
      }
      contract_out.depformer_output_projections[static_cast<size_t>(codebook)] =
          tensor_view{
              .tensor = output_projection,
              .name =
                  emel::model::tensor_name_view(model_data, *output_projection),
          };

      if (codebook > 0) {
        written = std::snprintf(name, sizeof(name),
                                "lm.depformer_emb.%d.weight", codebook - 1);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(name)) {
          return emel::error::cast(emel::model::loader::error::model_invalid);
        }
        const auto *audio_embedding = find_tensor(
            model_data, std::string_view{name, static_cast<size_t>(written)});
        if (audio_embedding == nullptr) {
          return emel::error::cast(emel::model::loader::error::model_invalid);
        }
        contract_out.depformer_audio_embeddings[static_cast<size_t>(
            codebook - 1)] = tensor_view{
            .tensor = audio_embedding,
            .name = emel::model::tensor_name_view(model_data, *audio_embedding),
        };
      }
    }

    if (lm.depformer_num_layers >
        emel::model::data::moshi_lm_hparams::k_max_delays) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    for (int32_t block = 0; block < lm.depformer_num_layers; ++block) {
      if (!has_depformer_block(model_data, block)) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
      }
      auto &layer = contract_out.depformer_layers[static_cast<size_t>(block)];
      char name[160] = {};
      int written = std::snprintf(name, sizeof(name),
                                  "lm.depformer.layers.%d.norm1.alpha", block);
      if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
          !bind_exact_tensor(
              model_data, std::string_view{name, static_cast<size_t>(written)},
              layer.norm1)) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
      }
      written = std::snprintf(name, sizeof(name),
                              "lm.depformer.layers.%d.norm2.alpha", block);
      if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
          !bind_exact_tensor(
              model_data, std::string_view{name, static_cast<size_t>(written)},
              layer.norm2)) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
      }
      for (int32_t codebook = 0; codebook < lm.dep_q; ++codebook) {
        if (!required_weight_indices[static_cast<size_t>(codebook)]) {
          continue;
        }
        auto &codebook_layer = layer.codebooks[static_cast<size_t>(codebook)];
        written =
            std::snprintf(name, sizeof(name),
                          "lm.depformer.layers.%d.self_attn.in_projs.%d.weight",
                          block, codebook);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(name)) {
          return emel::error::cast(emel::model::loader::error::model_invalid);
        }
        const bool split_input_bound = bind_exact_tensor(
            model_data, std::string_view{name, static_cast<size_t>(written)},
            codebook_layer.split_input_projection);
        written = std::snprintf(
            name, sizeof(name),
            "lm.depformer.layers.%d.self_attn.in_proj_weight", block);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(name)) {
          return emel::error::cast(emel::model::loader::error::model_invalid);
        }
        const bool fused_input_bound = bind_exact_tensor(
            model_data, std::string_view{name, static_cast<size_t>(written)},
            codebook_layer.fused_input_projection);
        if (!split_input_bound && !fused_input_bound) {
          return emel::error::cast(emel::model::loader::error::model_invalid);
        }
        written = std::snprintf(
            name, sizeof(name),
            "lm.depformer.layers.%d.self_attn.out_projs.%d.weight", block,
            codebook);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
            !bind_exact_tensor(
                model_data,
                std::string_view{name, static_cast<size_t>(written)},
                codebook_layer.output_projection)) {
          return emel::error::cast(emel::model::loader::error::model_invalid);
        }
        written =
            std::snprintf(name, sizeof(name),
                          "lm.depformer.layers.%d.gating.%d.linear_in.weight",
                          block, codebook);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
            !bind_exact_tensor(
                model_data,
                std::string_view{name, static_cast<size_t>(written)},
                codebook_layer.gating_input)) {
          return emel::error::cast(emel::model::loader::error::model_invalid);
        }
        written =
            std::snprintf(name, sizeof(name),
                          "lm.depformer.layers.%d.gating.%d.linear_out.weight",
                          block, codebook);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
            !bind_exact_tensor(
                model_data,
                std::string_view{name, static_cast<size_t>(written)},
                codebook_layer.gating_output)) {
          return emel::error::cast(emel::model::loader::error::model_invalid);
        }
      }
    }
  }

  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type validate_mimi_contract(const emel::model::data &model_data,
                                         mimi_contract &contract_out) noexcept {
  const auto &mimi = model_data.mimi;
  family_view component = {};
  const bool families_ok =
      assign_family_view(model_data, k_mimi_prefix, component) &&
      component.tensor_count == count_tensors_with_storage(model_data) &&
      assign_family_view(model_data, k_mimi_encoder_prefix,
                         contract_out.encoder) &&
      assign_family_view(model_data, k_mimi_encoder_transformer_prefix,
                         contract_out.encoder_transformer) &&
      assign_family_view(model_data, k_mimi_downsample_prefix,
                         contract_out.downsample) &&
      assign_family_view(model_data, k_mimi_quantizer_prefix,
                         contract_out.quantizer) &&
      assign_family_view(model_data, k_mimi_upsample_prefix,
                         contract_out.upsample) &&
      assign_family_view(model_data, k_mimi_decoder_transformer_prefix,
                         contract_out.decoder_transformer) &&
      assign_family_view(model_data, k_mimi_decoder_prefix,
                         contract_out.decoder);
  if (!families_ok) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const int64_t codebook_dim = mimi.codebook_dim;
  const int64_t card = mimi.card;
  // bind_rvq_split consumes every acoustic level 0..level_count-1 with the
  // full {codebook_dim, card} embedding shape, so the contract must probe
  // each rvq_rest codebook and its shape, not just existence of the last
  // index: a model missing an intermediate level or carrying a wrong-shaped
  // codebook would otherwise validate here and fail initialization later.
  // bind_rvq_split stores each split's levels in fixed arrays of 32
  // (k_max_quantizer_levels in the codec), so per-split counts past the cap
  // must reject here even when every declared codebook tensor exists. The
  // codec also caps codebook_dim/card at 2^16 (k_max_conv_geometry_extent)
  // to keep prepared/search-table sizing representable, so oversized
  // metadata must reject here too.
  constexpr int32_t k_max_rvq_split_levels = 32;
  constexpr int64_t k_max_mimi_extent = int64_t{1} << 16;
  bool quantizer_ok = mimi.n_q > mimi.semantic_n_q &&
                      mimi.semantic_n_q <= k_max_rvq_split_levels &&
                      mimi.n_q - mimi.semantic_n_q <= k_max_rvq_split_levels &&
                      mimi.codebook_dim <= k_max_mimi_extent &&
                      mimi.card <= k_max_mimi_extent;
  // bind_rvq_split consumes rvq_first levels 0..semantic_n_q-1 the same way
  // it consumes the acoustic levels, so every semantic codebook needs the
  // full shape probe too (a split past the codec's fixed level arrays also
  // fails here: its codebooks cannot all be present).
  for (int32_t level = 0; quantizer_ok && level < mimi.semantic_n_q; ++level) {
    char name[96] = {};
    const int written = std::snprintf(name, sizeof(name),
                                      "mimi.quantizer.rvq_first.vq.layers.%d."
                                      "_codebook.embedding",
                                      level);
    quantizer_ok =
        written > 0 && static_cast<size_t>(written) < sizeof(name) &&
        require_tensor_shape(
            model_data, std::string_view{name, static_cast<size_t>(written)},
            {codebook_dim, card}) &&
        // bind_rvq_split consumes codebooks via tensor_is_float
        has_float_tensor_with_elements(
            model_data, std::string_view{name, static_cast<size_t>(written)},
            static_cast<uint64_t>(codebook_dim) * static_cast<uint64_t>(card));
  }
  for (int32_t level = 0; quantizer_ok && level < mimi.n_q - mimi.semantic_n_q;
       ++level) {
    char name[96] = {};
    const int written = std::snprintf(name, sizeof(name),
                                      "mimi.quantizer.rvq_rest.vq.layers.%d."
                                      "_codebook.embedding",
                                      level);
    quantizer_ok =
        written > 0 && static_cast<size_t>(written) < sizeof(name) &&
        require_tensor_shape(
            model_data, std::string_view{name, static_cast<size_t>(written)},
            {codebook_dim, card}) &&
        has_float_tensor_with_elements(
            model_data, std::string_view{name, static_cast<size_t>(written)},
            static_cast<uint64_t>(codebook_dim) * static_cast<uint64_t>(card));
  }
  if (!quantizer_ok) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  // bind_transformer consumes every configured layer's norms, layer scales,
  // attention projections, and MLP linears (plan_transformer additionally
  // pins the in_proj/linear1 shapes), so the contract must probe the full
  // per-layer tensor set for both transformer families: a truncated family
  // or a missing layer tensor otherwise validates here and fails the codec
  // bind walk later. The codec plan selects the q8-vs-float projection class
  // from the first encoder in_proj and the bind requires every projection to
  // match it, so mixed-class projections must reject here too.
  const int64_t dim = mimi.dim;
  const uint64_t dim_elements = static_cast<uint64_t>(dim);
  constexpr int32_t k_dtype_q8_0 = 8;
  const auto *first_proj = find_tensor(
      model_data,
      "mimi.encoder_transformer.transformer.layers.0.self_attn.in_projs.0."
      "weight");
  const bool proj_q8 =
      first_proj != nullptr && first_proj->type == k_dtype_q8_0;
  // The q8 matvec kernels require block-aligned contraction widths
  // (k % 32 == 0); a q8 class with a misaligned dim would bind and then
  // fail silently inside the codec compute loops.
  constexpr int64_t k_q8_row_block = 32;
  if (proj_q8 && mimi.dim % k_q8_row_block != 0) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  static constexpr const char *k_transformer_families[2] = {
      "encoder_transformer", "decoder_transformer"};
  // plan_transformer stores layers in fixed arrays of 16
  // (k_max_transformer_layers in the codec), so a larger configured count
  // must reject here even when every declared layer's tensors exist.
  constexpr int32_t k_max_mimi_transformer_layers = 16;
  bool transformers_ok =
      mimi.transformer_num_layers <= k_max_mimi_transformer_layers;
  for (size_t family = 0; transformers_ok && family < 2u; ++family) {
    int64_t family_mlp = 0;
    for (int32_t layer = 0;
         transformers_ok && layer < mimi.transformer_num_layers; ++layer) {
      char base[96] = {};
      const int base_len =
          std::snprintf(base, sizeof(base), "mimi.%s.transformer.layers.%d.",
                        k_transformer_families[family], layer);
      if (base_len <= 0 || static_cast<size_t>(base_len) >= sizeof(base)) {
        transformers_ok = false;
        break;
      }
      const auto layer_name = [&base, base_len](const char *suffix,
                                                char (&name_out)[144]) {
        const int written = std::snprintf(name_out, sizeof(name_out), "%.*s%s",
                                          base_len, base, suffix);
        return written > 0 && static_cast<size_t>(written) < sizeof(name_out)
                   ? std::string_view{name_out, static_cast<size_t>(written)}
                   : std::string_view{};
      };
      char name[144] = {};
      const auto *linear1_tensor =
          find_tensor(model_data, layer_name("linear1.weight", name));
      if (linear1_tensor == nullptr || !tensor_has_storage(*linear1_tensor) ||
          linear1_tensor->n_dims != 2 || linear1_tensor->dims[0] != dim ||
          linear1_tensor->dims[1] <= 0 ||
          (linear1_tensor->type == k_dtype_q8_0) != proj_q8) {
        transformers_ok = false;
        break;
      }
      // The planner stores one MLP width per family and the bind uses it for
      // every layer's linear1/linear2, so mixed widths must reject here.
      if (layer == 0) {
        family_mlp = linear1_tensor->dims[1];
      } else if (linear1_tensor->dims[1] != family_mlp) {
        transformers_ok = false;
        break;
      }
      if (proj_q8 && family_mlp % k_q8_row_block != 0) {
        transformers_ok = false;
        break;
      }
      const uint64_t mlp_elements =
          static_cast<uint64_t>(linear1_tensor->dims[1]);
      transformers_ok =
          has_projection_tensor(model_data, layer_name("linear1.weight", name),
                                dim_elements * mlp_elements, proj_q8) &&
          require_tensor_shape(model_data,
                               layer_name("self_attn.in_projs.0.weight", name),
                               {dim, 3 * dim}) &&
          has_projection_tensor(model_data,
                                layer_name("self_attn.in_projs.0.weight", name),
                                dim_elements * 3u * dim_elements, proj_q8) &&
          has_projection_tensor(
              model_data, layer_name("self_attn.out_projs.0.weight", name),
              dim_elements * dim_elements, proj_q8) &&
          has_projection_tensor(model_data, layer_name("linear2.weight", name),
                                dim_elements * mlp_elements, proj_q8) &&
          // norms, biases, and layer scales bind through prepare_vector,
          // which accepts only f32/f16.
          has_float_tensor_with_elements(
              model_data, layer_name("norm1.weight", name), dim_elements) &&
          has_float_tensor_with_elements(
              model_data, layer_name("norm1.bias", name), dim_elements) &&
          has_float_tensor_with_elements(
              model_data, layer_name("norm2.weight", name), dim_elements) &&
          has_float_tensor_with_elements(
              model_data, layer_name("norm2.bias", name), dim_elements) &&
          has_float_tensor_with_elements(
              model_data, layer_name("layer_scale_1.scale", name),
              dim_elements) &&
          has_float_tensor_with_elements(
              model_data, layer_name("layer_scale_2.scale", name),
              dim_elements);
    }
  }
  if (!transformers_ok) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  // bind_rvq_split consumes the input/output 1x1 projections for both splits
  // before any encode or decode can run; the contract must require them (by
  // element count, matching prepare_linear/prepare_raw_q8_0) so a component
  // missing a projection rejects at load instead of failing quantizer bind.
  // The RVQ q8-vs-float class is probed from the first split's input
  // projection, mirroring plan_codec, and in the f16 conv operand class the
  // non-q8 projections must themselves be f16: bind_rvq_split prepares raw
  // f16 copies (prepare_raw_f16) whenever conv_f16 is selected and the
  // split is not q8.
  constexpr int32_t k_dtype_f16 = 1;
  const auto *first_conv =
      find_tensor(model_data, "mimi.encoder.model.0.conv.conv.weight");
  const bool conv_f16 =
      first_conv != nullptr && first_conv->type == k_dtype_f16;
  const uint64_t proj_elements = static_cast<uint64_t>(mimi.dim) *
                                 static_cast<uint64_t>(mimi.codebook_dim);
  const auto *first_rvq_proj =
      find_tensor(model_data, "mimi.quantizer.rvq_first.input_proj.weight");
  const bool rvq_q8 =
      first_rvq_proj != nullptr && first_rvq_proj->type == k_dtype_q8_0;
  if (rvq_q8 && (mimi.dim % k_q8_row_block != 0 ||
                 mimi.codebook_dim % k_q8_row_block != 0)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  static constexpr const char *k_rvq_splits[2] = {"rvq_first", "rvq_rest"};
  const auto rvq_projection_ok =
      [&model_data, proj_elements, rvq_q8,
       conv_f16](const std::string_view name_view) noexcept {
        if (!has_projection_tensor(model_data, name_view, proj_elements,
                                   rvq_q8)) {
          return false;
        }
        if (rvq_q8 || !conv_f16) {
          return true;
        }
        const auto *tensor = find_tensor(model_data, name_view);
        return tensor != nullptr && tensor->type == k_dtype_f16;
      };
  bool projections_ok = true;
  for (size_t split = 0; projections_ok && split < 2u; ++split) {
    char name[96] = {};
    const int in_len =
        std::snprintf(name, sizeof(name), "mimi.quantizer.%s.input_proj.weight",
                      k_rvq_splits[split]);
    projections_ok =
        in_len > 0 && static_cast<size_t>(in_len) < sizeof(name) &&
        rvq_projection_ok(std::string_view{name, static_cast<size_t>(in_len)});
    if (!projections_ok) {
      break;
    }
    const int out_len = std::snprintf(name, sizeof(name),
                                      "mimi.quantizer.%s.output_proj.weight",
                                      k_rvq_splits[split]);
    projections_ok =
        out_len > 0 && static_cast<size_t>(out_len) < sizeof(name) &&
        rvq_projection_ok(std::string_view{name, static_cast<size_t>(out_len)});
  }
  if (!projections_ok) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  // The codec planner walks the fixed mimi_v0_1 SEANet module topology and
  // resolves every weight against the running channel/stride chain
  // (plan_seanet / resolve_conv_geometry / bind_conv), so the contract must
  // mirror the geometry, not merely the names: a named conv whose element
  // count does not divide the chain, a resnet that does not return to its
  // input width through a k1 conv, or a strided kernel shorter than its
  // stride otherwise validates and fails codec initialization. plan_codec
  // also selects the f16 conv operand class from the first encoder conv and
  // bind_conv requires raw f16 taps on every non-transposed conv.
  const auto resolve_conv = [&model_data,
                             conv_f16](const std::string_view name,
                                       const int64_t in_channels,
                                       const bool transposed, int64_t &taps_out,
                                       int64_t &out_channels_out) noexcept {
    const auto *weight = find_tensor(model_data, name);
    constexpr int32_t k_dtype_f32 = 0;
    if (weight == nullptr || !tensor_has_storage(*weight) ||
        weight->n_dims < 1 || weight->n_dims > 3 || in_channels <= 0 ||
        // prepare_conv_gemm/prepare_conv_transpose consume only f32/f16, and
        // the f16 conv class additionally requires raw f16 taps on every
        // non-transposed conv.
        (weight->type != k_dtype_f32 && weight->type != k_dtype_f16) ||
        (!transposed && conv_f16 && weight->type != k_dtype_f16)) {
      return false;
    }
    taps_out = weight->dims[0];
    // The codec caps every conv geometry extent at 2^16
    // (k_max_conv_geometry_extent) to keep its sizing representable.
    constexpr int64_t k_max_conv_extent = int64_t{1} << 16;
    if (taps_out <= 0 || taps_out > k_max_conv_extent) {
      return false;
    }
    uint64_t total = 1u;
    for (int32_t dim = 0; dim < weight->n_dims; ++dim) {
      total *= static_cast<uint64_t>(weight->dims[static_cast<size_t>(dim)]);
    }
    const uint64_t divisor =
        static_cast<uint64_t>(taps_out) * static_cast<uint64_t>(in_channels);
    if (divisor == 0u || total % divisor != 0u) {
      return false;
    }
    out_channels_out = static_cast<int64_t>(total / divisor);
    return out_channels_out > 0 && out_channels_out <= k_max_conv_extent;
  };
  // Fixed mimi_v0_1 module topology (index, kind, stride) mirroring the
  // codec's k_encoder_topology/k_decoder_topology; kind: 0 conv, 1 convtr,
  // 2 resnet.
  struct seanet_module {
    int32_t index;
    uint8_t kind;
    int32_t stride;
  };
  static constexpr seanet_module k_encoder_modules[] = {
      {0, 0, 1}, {1, 2, 1}, {3, 0, 4},  {4, 2, 1},  {6, 0, 5},
      {7, 2, 1}, {9, 0, 6}, {10, 2, 1}, {12, 0, 8}, {14, 0, 1},
  };
  static constexpr seanet_module k_decoder_modules[] = {
      {0, 0, 1}, {2, 1, 8}, {3, 2, 1},  {5, 1, 6},  {6, 2, 1},
      {8, 1, 5}, {9, 2, 1}, {11, 1, 4}, {12, 2, 1}, {14, 0, 1},
  };
  const auto walk_seanet = [&resolve_conv](
                               const char *family,
                               std::span<const seanet_module> modules,
                               int64_t channels,
                               int64_t &channels_out) noexcept {
    for (const seanet_module &module : modules) {
      char name[112] = {};
      if (module.kind == 2u) {
        int64_t taps1 = 0;
        int64_t half = 0;
        int64_t taps3 = 0;
        int64_t res_out = 0;
        int written = std::snprintf(name, sizeof(name),
                                    "mimi.%s.model.%d.block.1.conv.conv.weight",
                                    family, module.index);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
            !resolve_conv(std::string_view{name, static_cast<size_t>(written)},
                          channels, false, taps1, half)) {
          return false;
        }
        written = std::snprintf(name, sizeof(name),
                                "mimi.%s.model.%d.block.3.conv.conv.weight",
                                family, module.index);
        if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
            !resolve_conv(std::string_view{name, static_cast<size_t>(written)},
                          half, false, taps3, res_out) ||
            taps3 != 1 || res_out != channels) {
          return false;
        }
        continue;
      }
      const bool transposed = module.kind == 1u;
      const int written = std::snprintf(
          name, sizeof(name), "mimi.%s.model.%d.%s", family, module.index,
          transposed ? "convtr.convtr.weight" : "conv.conv.weight");
      int64_t taps = 0;
      int64_t out_channels = 0;
      if (written <= 0 || static_cast<size_t>(written) >= sizeof(name) ||
          !resolve_conv(std::string_view{name, static_cast<size_t>(written)},
                        channels, transposed, taps, out_channels) ||
          taps < module.stride) {
        return false;
      }
      channels = out_channels;
    }
    channels_out = channels;
    return true;
  };
  int64_t encoder_channels = 0;
  int64_t decoder_channels = 0;
  int64_t down_taps = 0;
  int64_t down_out = 0;
  const auto *upsample =
      find_tensor(model_data, "mimi.upsample.convtr.convtr.convtr.weight");
  uint64_t upsample_elements = 0u;
  if (upsample != nullptr && tensor_has_storage(*upsample)) {
    upsample_elements = 1u;
    for (int32_t dim = 0; dim < upsample->n_dims; ++dim) {
      upsample_elements *=
          static_cast<uint64_t>(upsample->dims[static_cast<size_t>(dim)]);
    }
  }
  const bool seanet_ok =
      walk_seanet("encoder", std::span<const seanet_module>{k_encoder_modules},
                  1, encoder_channels) &&
      encoder_channels == dim &&
      walk_seanet("decoder", std::span<const seanet_module>{k_decoder_modules},
                  dim, decoder_channels) &&
      decoder_channels == 1 &&
      resolve_conv("mimi.downsample.conv.conv.conv.weight", dim, false,
                   down_taps, down_out) &&
      down_out == dim && down_taps >= 2 &&
      // depthwise stride-2 upsample: elements == taps * dim, taps >= stride;
      // prepare_conv_transpose consumes only f32/f16 and geometry extents
      // stay under the codec's 2^16 cap.
      upsample != nullptr && upsample->n_dims >= 1 && upsample->dims[0] >= 2 &&
      upsample->dims[0] <= (int64_t{1} << 16) &&
      (upsample->type == 0 || upsample->type == k_dtype_f16) &&
      upsample_elements ==
          static_cast<uint64_t>(upsample->dims[0]) * static_cast<uint64_t>(dim);
  if (!seanet_ok) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  // The fixed stride chain (encoder 4*5*6*8, downsample 2) must reduce one
  // frame to exactly one token, so the codec accepts only a frame of 1920
  // samples: plan_codec truncates sample_rate / frame_rate and plan_seanet
  // rejects any length the strides cannot divide down to one. The hparam
  // gate already guarantees a finite positive frame_rate.
  constexpr float k_max_frame_samples = 1.0e8f;
  constexpr int32_t k_mimi_frame_samples = 1920;
  const float frame_ratio =
      static_cast<float>(mimi.sample_rate) / mimi.frame_rate;
  if (!(frame_ratio >= 1.0f) || frame_ratio > k_max_frame_samples ||
      static_cast<int32_t>(frame_ratio) != k_mimi_frame_samples) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
validate_voice_contract(const emel::model::data &model_data,
                        voice_contract &contract_out) noexcept {
  family_view component = {};
  const auto *embeddings = find_tensor(model_data, k_voice_embeddings_name);
  const auto *cache = find_tensor(model_data, k_voice_cache_name);
  const bool voice_ok =
      assign_family_view(model_data, k_voice_prefix, component) &&
      component.tensor_count == count_tensors_with_storage(model_data) &&
      embeddings != nullptr && tensor_has_storage(*embeddings) &&
      cache != nullptr && tensor_has_storage(*cache);
  if (!voice_ok) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  contract_out.embeddings = tensor_view{embeddings, k_voice_embeddings_name};
  contract_out.cache = tensor_view{cache, k_voice_cache_name};
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type validate_contract(const emel::model::data &model_data,
                                    execution_contract *contract_out) noexcept {
  if (!is_execution_architecture(
          emel::model::architecture_name_view(model_data))) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  execution_contract contract = {};
  contract.model = &model_data;
  contract.component = model_data.moshi_component_id;

  emel::error::type result =
      emel::error::cast(emel::model::loader::error::model_invalid);
  if (contract.component == emel::model::data::moshi_component::lm) {
    result = validate_lm_contract(model_data, contract.lm);
  } else if (contract.component == emel::model::data::moshi_component::mimi) {
    result = validate_mimi_contract(model_data, contract.mimi);
  } else if (contract.component == emel::model::data::moshi_component::voice) {
    result = validate_voice_contract(model_data, contract.voice);
  }

  if (result != emel::error::cast(emel::model::loader::error::none)) {
    return result;
  }

  if (contract_out != nullptr) {
    *contract_out = contract;
  }

  return emel::error::cast(emel::model::loader::error::none);
}

} // namespace

bool is_execution_architecture(const std::string_view architecture) noexcept {
  return architecture == k_architecture;
}

bool load_hparams(const emel::model::detail::hparam_loader &loader,
                  emel::model::data &model_out) noexcept {
  const auto *component_entry =
      emel::model::detail::find_kv_entry(loader.binding, k_component_key);
  if (component_entry == nullptr) {
    return false;
  }

  std::string_view component = {};
  if (!emel::model::detail::decode_string_value(loader.binding,
                                                *component_entry, component)) {
    return false;
  }

  if (component == k_component_lm) {
    return load_lm_hparams(loader, model_out);
  }
  if (component == k_component_mimi) {
    return load_mimi_hparams(loader, model_out);
  }
  if (component == k_component_voice) {
    return load_voice_hparams(loader, model_out);
  }
  return false;
}

emel::error::type
build_execution_contract(const emel::model::data &model_data,
                         execution_contract &contract_out) noexcept {
  contract_out = {};
  return validate_contract(model_data, &contract_out);
}

emel::error::type validate_data(const emel::model::data &model_data) noexcept {
  return validate_contract(model_data, nullptr);
}

emel::error::type
validate_execution_contract(const emel::model::data &model_data) noexcept {
  return validate_contract(model_data, nullptr);
}

} // namespace emel::model::moshi::detail
