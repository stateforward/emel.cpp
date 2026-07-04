#include "emel/model/moshi/detail.hpp"

#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <limits>

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
       !assign_optional_i32(binding, "moshi.lm.depformer.low_rank_embeddings",
                            lm.depformer_low_rank_embeddings) ||
       !lm.depformer_multi_linear ||
       lm.depformer_dim % lm.depformer_num_heads != 0)) {
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

  uint64_t elements = 1u;
  for (int32_t dim = 0; dim < tensor.n_dims; ++dim) {
    if (tensor.dims[static_cast<size_t>(dim)] <= 0) {
      return false;
    }
    const uint64_t extent =
        static_cast<uint64_t>(tensor.dims[static_cast<size_t>(dim)]);
    // Malformed dimensions can wrap the element product to a small count and
    // mark a tiny payload as fully backed; reject unrepresentable products.
    if (extent > UINT64_MAX / elements) {
      return false;
    }
    elements *= extent;
  }

  // Runtime binding and kernel reads consume the full dtype payload, so the
  // contract must require the dtype-sized byte count, not merely a non-empty
  // buffer. Codes are the GGUF/ggml tensor type ids the loader stores.
  constexpr int32_t k_dtype_f32 = 0;
  constexpr int32_t k_dtype_f16 = 1;
  constexpr int32_t k_dtype_q8_0 = 8;
  constexpr int32_t k_dtype_bf16 = 30;
  constexpr uint64_t k_q8_0_block_elements = 32u;
  constexpr uint64_t k_q8_0_block_bytes = 34u;
  uint64_t required_bytes = 0u;
  if (tensor.type == k_dtype_f32) {
    if (elements > UINT64_MAX / sizeof(float)) {
      return false;
    }
    required_bytes = elements * sizeof(float);
  } else if (tensor.type == k_dtype_f16 || tensor.type == k_dtype_bf16) {
    if (elements > UINT64_MAX / sizeof(uint16_t)) {
      return false;
    }
    required_bytes = elements * sizeof(uint16_t);
  } else if (tensor.type == k_dtype_q8_0) {
    if (elements % k_q8_0_block_elements != 0u ||
        elements / k_q8_0_block_elements > UINT64_MAX / k_q8_0_block_bytes) {
      return false;
    }
    required_bytes = elements / k_q8_0_block_elements * k_q8_0_block_bytes;
  } else {
    // Conservative floor for dtypes the Moshi families never carry: at least
    // one byte per element.
    required_bytes = elements;
  }

  return tensor.data_size >= required_bytes;
}

bool assign_family_view(const emel::model::data &model_data,
                        const std::string_view prefix,
                        family_view &family_out) noexcept {
  family_out = {};
  family_out.prefix = prefix;

  for (uint32_t index = 0u; index < model_data.n_tensors; ++index) {
    const auto &tensor = model_data.tensors[index];
    const auto name = emel::model::tensor_name_view(model_data, tensor);
    if (!name.starts_with(prefix) || !tensor_has_storage(tensor)) {
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

// The codec plan selects the q8 bind path from the first projection of a
// family and the bind then requires every projection in that family to share
// the class, so projection probes compare element count and q8-vs-float
// class together.
bool has_projection_tensor(const emel::model::data &model_data,
                           const std::string_view name,
                           const uint64_t expected_elements,
                           const bool q8_class) noexcept {
  constexpr int32_t k_dtype_q8_0 = 8;
  const auto *tensor = find_tensor(model_data, name);
  return tensor != nullptr &&
         has_tensor_with_elements(model_data, name, expected_elements) &&
         ((tensor->type == k_dtype_q8_0) == q8_class);
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

uint32_t
count_tensors_with_storage(const emel::model::data &model_data) noexcept {
  uint32_t count = 0u;
  for (uint32_t index = 0u; index < model_data.n_tensors; ++index) {
    if (tensor_has_storage(model_data.tensors[index])) {
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
      assign_family_view(model_data, k_lm_prefix, component) &&
      component.tensor_count == count_tensors_with_storage(model_data) &&
      assign_family_view(model_data, k_lm_text_emb_prefix,
                         contract_out.text_emb) &&
      assign_family_view(model_data, k_lm_audio_emb_prefix,
                         contract_out.audio_emb) &&
      contract_out.audio_emb.tensor_count >= static_cast<uint32_t>(lm.n_q) &&
      assign_family_view(model_data, k_lm_transformer_prefix,
                         contract_out.transformer) &&
      assign_family_view(model_data, k_lm_out_norm_prefix,
                         contract_out.out_norm) &&
      assign_family_view(model_data, k_lm_text_linear_prefix,
                         contract_out.text_linear);
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

  for (int32_t block = 0; block < lm.num_layers; ++block) {
    if (!has_lm_transformer_block(model_data, block)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
  }

  if (lm.dep_q > 0) {
    const bool depformer_ok =
        assign_family_view(model_data, k_lm_linears_prefix,
                           contract_out.linears) &&
        contract_out.linears.tensor_count >= static_cast<uint32_t>(lm.dep_q) &&
        assign_family_view(model_data, k_lm_depformer_in_prefix,
                           contract_out.depformer_in) &&
        assign_family_view(model_data, k_lm_depformer_prefix,
                           contract_out.depformer) &&
        assign_family_view(model_data, k_lm_depformer_text_emb_prefix,
                           contract_out.depformer_text_emb) &&
        (lm.dep_q < 2 ||
         (assign_family_view(model_data, k_lm_depformer_emb_prefix,
                             contract_out.depformer_emb) &&
          contract_out.depformer_emb.tensor_count >=
              static_cast<uint32_t>(lm.dep_q) - 1u));
    if (!depformer_ok) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }

    for (int32_t block = 0; block < lm.depformer_num_layers; ++block) {
      if (!has_depformer_block(model_data, block)) {
        return emel::error::cast(emel::model::loader::error::model_invalid);
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
  bool quantizer_ok = mimi.n_q > mimi.semantic_n_q;
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
    quantizer_ok = written > 0 && static_cast<size_t>(written) < sizeof(name) &&
                   require_tensor_shape(
                       model_data,
                       std::string_view{name, static_cast<size_t>(written)},
                       {codebook_dim, card});
  }
  for (int32_t level = 0;
       quantizer_ok && level < mimi.n_q - mimi.semantic_n_q; ++level) {
    char name[96] = {};
    const int written = std::snprintf(name, sizeof(name),
                                      "mimi.quantizer.rvq_rest.vq.layers.%d."
                                      "_codebook.embedding",
                                      level);
    quantizer_ok = written > 0 && static_cast<size_t>(written) < sizeof(name) &&
                   require_tensor_shape(
                       model_data,
                       std::string_view{name, static_cast<size_t>(written)},
                       {codebook_dim, card});
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
  static constexpr const char *k_transformer_families[2] = {
      "encoder_transformer", "decoder_transformer"};
  bool transformers_ok = true;
  for (size_t family = 0; transformers_ok && family < 2u; ++family) {
    for (int32_t layer = 0;
         transformers_ok && layer < mimi.transformer_num_layers; ++layer) {
      char base[96] = {};
      const int base_len = std::snprintf(
          base, sizeof(base), "mimi.%s.transformer.layers.%d.",
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
      const uint64_t mlp_elements =
          static_cast<uint64_t>(linear1_tensor->dims[1]);
      transformers_ok =
          require_tensor_shape(model_data,
                               layer_name("self_attn.in_projs.0.weight", name),
                               {dim, 3 * dim}) &&
          has_projection_tensor(model_data,
                                layer_name("self_attn.in_projs.0.weight", name),
                                dim_elements * 3u * dim_elements, proj_q8) &&
          has_projection_tensor(model_data,
                                layer_name("self_attn.out_projs.0.weight",
                                           name),
                                dim_elements * dim_elements, proj_q8) &&
          has_projection_tensor(model_data, layer_name("linear2.weight", name),
                                dim_elements * mlp_elements, proj_q8) &&
          has_tensor_with_elements(model_data, layer_name("norm1.weight", name),
                                   dim_elements) &&
          has_tensor_with_elements(model_data, layer_name("norm1.bias", name),
                                   dim_elements) &&
          has_tensor_with_elements(model_data, layer_name("norm2.weight", name),
                                   dim_elements) &&
          has_tensor_with_elements(model_data, layer_name("norm2.bias", name),
                                   dim_elements) &&
          has_tensor_with_elements(model_data,
                                   layer_name("layer_scale_1.scale", name),
                                   dim_elements) &&
          has_tensor_with_elements(model_data,
                                   layer_name("layer_scale_2.scale", name),
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
  // projection, mirroring plan_codec.
  const uint64_t proj_elements =
      static_cast<uint64_t>(mimi.dim) * static_cast<uint64_t>(mimi.codebook_dim);
  const auto *first_rvq_proj =
      find_tensor(model_data, "mimi.quantizer.rvq_first.input_proj.weight");
  const bool rvq_q8 =
      first_rvq_proj != nullptr && first_rvq_proj->type == k_dtype_q8_0;
  static constexpr const char *k_rvq_splits[2] = {"rvq_first", "rvq_rest"};
  bool projections_ok = true;
  for (size_t split = 0; projections_ok && split < 2u; ++split) {
    char name[96] = {};
    const int in_len =
        std::snprintf(name, sizeof(name),
                      "mimi.quantizer.%s.input_proj.weight", k_rvq_splits[split]);
    projections_ok =
        in_len > 0 && static_cast<size_t>(in_len) < sizeof(name) &&
        has_projection_tensor(
            model_data, std::string_view{name, static_cast<size_t>(in_len)},
            proj_elements, rvq_q8);
    if (!projections_ok) {
      break;
    }
    const int out_len = std::snprintf(name, sizeof(name),
                                      "mimi.quantizer.%s.output_proj.weight",
                                      k_rvq_splits[split]);
    projections_ok =
        out_len > 0 && static_cast<size_t>(out_len) < sizeof(name) &&
        has_projection_tensor(
            model_data, std::string_view{name, static_cast<size_t>(out_len)},
            proj_elements, rvq_q8);
  }
  if (!projections_ok) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  // The codec planner walks the fixed mimi_v0_1 SEANet module topology by
  // exact tensor name (plan_seanet's conv/convtr/resnet indexes plus the
  // downsample/upsample bridge convs), so the contract must require every
  // fixed-topology weight: a component with each family merely non-empty
  // otherwise validates and fails codec initialization later.
  static constexpr const char *k_encoder_seanet_tensors[] = {
      "model.0.conv.conv.weight",
      "model.1.block.1.conv.conv.weight",
      "model.1.block.3.conv.conv.weight",
      "model.3.conv.conv.weight",
      "model.4.block.1.conv.conv.weight",
      "model.4.block.3.conv.conv.weight",
      "model.6.conv.conv.weight",
      "model.7.block.1.conv.conv.weight",
      "model.7.block.3.conv.conv.weight",
      "model.9.conv.conv.weight",
      "model.10.block.1.conv.conv.weight",
      "model.10.block.3.conv.conv.weight",
      "model.12.conv.conv.weight",
      "model.14.conv.conv.weight",
  };
  static constexpr const char *k_decoder_seanet_tensors[] = {
      "model.0.conv.conv.weight",
      "model.2.convtr.convtr.weight",
      "model.3.block.1.conv.conv.weight",
      "model.3.block.3.conv.conv.weight",
      "model.5.convtr.convtr.weight",
      "model.6.block.1.conv.conv.weight",
      "model.6.block.3.conv.conv.weight",
      "model.8.convtr.convtr.weight",
      "model.9.block.1.conv.conv.weight",
      "model.9.block.3.conv.conv.weight",
      "model.11.convtr.convtr.weight",
      "model.12.block.1.conv.conv.weight",
      "model.12.block.3.conv.conv.weight",
      "model.14.conv.conv.weight",
  };
  bool seanet_ok = true;
  for (const char *suffix : k_encoder_seanet_tensors) {
    char name[96] = {};
    const int written =
        std::snprintf(name, sizeof(name), "mimi.encoder.%s", suffix);
    seanet_ok = seanet_ok && written > 0 &&
                static_cast<size_t>(written) < sizeof(name) &&
                has_tensor(model_data,
                           std::string_view{name, static_cast<size_t>(written)});
  }
  for (const char *suffix : k_decoder_seanet_tensors) {
    char name[96] = {};
    const int written =
        std::snprintf(name, sizeof(name), "mimi.decoder.%s", suffix);
    seanet_ok = seanet_ok && written > 0 &&
                static_cast<size_t>(written) < sizeof(name) &&
                has_tensor(model_data,
                           std::string_view{name, static_cast<size_t>(written)});
  }
  seanet_ok = seanet_ok &&
              has_tensor(model_data, "mimi.downsample.conv.conv.conv.weight") &&
              has_tensor(model_data,
                         "mimi.upsample.convtr.convtr.convtr.weight");
  if (!seanet_ok) {
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
