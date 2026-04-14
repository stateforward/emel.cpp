#include "emel/model/omniembed/detail.hpp"

#include <array>
#include <cstring>
#include <limits>

#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"

namespace emel::model::omniembed::detail {

namespace {

constexpr std::string_view k_architecture = "omniembed";
constexpr std::string_view k_text_encoder_prefix = "text_encoder.";
constexpr std::string_view k_text_projection_prefix = "text_projection.";
constexpr std::string_view k_image_encoder_prefix = "image_encoder.";
constexpr std::string_view k_image_projection_prefix = "image_projection.";
constexpr std::string_view k_audio_encoder_prefix = "audio_encoder.";
constexpr std::string_view k_audio_projection_prefix = "audio_projection.";
constexpr std::string_view k_image_encoder_name_key = "omniembed.image_encoder_name";
constexpr std::string_view k_audio_encoder_name_key = "omniembed.audio_encoder_name";
constexpr std::string_view k_mobilenetv4_medium_encoder =
    "mobilenetv4_conv_medium.e180_r384_in12k";
constexpr std::string_view k_efficientat_mn20_as_encoder = "efficientat_mn20_as";
constexpr int32_t k_mobilenetv4_medium_image_size = 384;
constexpr std::array<float, 3> k_imagenet_mean = {0.485f, 0.456f, 0.406f};
constexpr std::array<float, 3> k_imagenet_std = {0.229f, 0.224f, 0.225f};
constexpr int32_t k_efficientat_mn20_as_sample_rate = 32000;
constexpr int32_t k_efficientat_mn20_as_n_fft = 1024;
constexpr int32_t k_efficientat_mn20_as_win_length = 800;
constexpr int32_t k_efficientat_mn20_as_hop_size = 320;
constexpr int32_t k_efficientat_mn20_as_num_mel_bins = 128;
constexpr float k_efficientat_mn20_as_low_frequency = 0.0f;
constexpr float k_efficientat_mn20_as_high_frequency = 15000.0f;
constexpr float k_efficientat_mn20_as_preemphasis = 0.97f;
constexpr float k_efficientat_mn20_as_log_offset = 1.0e-5f;
constexpr float k_efficientat_mn20_as_normalize_bias = 4.5f;
constexpr float k_efficientat_mn20_as_normalize_scale = 5.0f;

bool tensor_has_storage(const emel::model::data::tensor_record & tensor) noexcept {
  if (tensor.data == nullptr || tensor.data_size == 0u || tensor.n_dims <= 0) {
    return false;
  }

  for (int32_t dim = 0; dim < tensor.n_dims && dim < static_cast<int32_t>(tensor.dims.size());
       ++dim) {
    if (tensor.dims[static_cast<size_t>(dim)] <= 0) {
      return false;
    }
  }

  return true;
}

bool copy_matryoshka_dimensions(const emel::model::detail::kv_binding & binding,
                                const std::string_view key,
                                std::array<int32_t, emel::model::data::k_max_matryoshka_dims> & dst,
                                uint32_t & count_out) noexcept {
  const auto * entry = emel::model::detail::find_kv_entry(binding, key);
  if (entry == nullptr) {
    count_out = 0u;
    return true;
  }

  emel::model::detail::array_header header = {};
  if (!emel::model::detail::decode_array_header(binding, *entry, header) ||
      header.count > static_cast<uint64_t>(dst.size())) {
    return false;
  }

  const size_t element_size = emel::model::detail::scalar_array_element_size(header.element_type);
  if (element_size == 0u ||
      header.payload.size() != static_cast<size_t>(header.count) * element_size) {
    return false;
  }

  dst.fill(0);
  namespace constants = emel::gguf::loader::detail::constants;
  for (uint64_t index = 0u; index < header.count; ++index) {
    const auto element = header.payload.subspan(static_cast<size_t>(index) * element_size,
                                                element_size);
    int64_t value = 0;
    switch (header.element_type) {
      case constants::gguf_type_uint8:
        value = element[0];
        break;
      case constants::gguf_type_int8:
        value = static_cast<int8_t>(element[0]);
        break;
      case constants::gguf_type_uint16:
        value = static_cast<int64_t>(static_cast<uint16_t>(element[0]) |
                                     (static_cast<uint16_t>(element[1]) << 8u));
        break;
      case constants::gguf_type_int16:
        value = static_cast<int16_t>(static_cast<uint16_t>(element[0]) |
                                     (static_cast<uint16_t>(element[1]) << 8u));
        break;
      case constants::gguf_type_uint32:
        value = static_cast<int64_t>(emel::model::detail::read_u32_le(element));
        break;
      case constants::gguf_type_int32:
        value = static_cast<int32_t>(emel::model::detail::read_u32_le(element));
        break;
      case constants::gguf_type_uint64: {
        const uint64_t raw = emel::model::detail::read_u64_le(element);
        if (raw > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
          return false;
        }
        value = static_cast<int64_t>(raw);
        break;
      }
      case constants::gguf_type_int64: {
        const int64_t raw = static_cast<int64_t>(emel::model::detail::read_u64_le(element));
        value = raw;
        break;
      }
      default:
        return false;
    }

    if (value <= 0 || value > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }
    dst[static_cast<size_t>(index)] = static_cast<int32_t>(value);
  }

  count_out = static_cast<uint32_t>(header.count);
  return true;
}

bool validate_matryoshka_dimensions(
    const std::array<int32_t, emel::model::data::k_max_matryoshka_dims> & dimensions,
    const uint32_t dimension_count,
    const int32_t embedding_length) noexcept {
  if (dimension_count == 0u || embedding_length <= 0) {
    return false;
  }

  int32_t previous = embedding_length + 1;
  for (uint32_t index = 0u; index < dimension_count; ++index) {
    const int32_t current = dimensions[index];
    if (current <= 0 || current > embedding_length || current >= previous) {
      return false;
    }
    previous = current;
  }

  return true;
}

bool assign_family_view(const emel::model::data & model_data,
                        const std::string_view prefix,
                        family_view & family_out) noexcept {
  family_out = {};
  family_out.prefix = prefix;

  for (uint32_t index = 0u; index < model_data.n_tensors; ++index) {
    const auto & tensor = model_data.tensors[index];
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

bool copy_metadata_string(emel::model::data::metadata & metadata,
                          emel::model::data::metadata::string_view & field_out,
                          const std::string_view value) noexcept {
  field_out = {};
  if (value.empty()) {
    return true;
  }

  const size_t begin = static_cast<size_t>(metadata.blob_bytes_used);
  const size_t length = value.size();
  if (begin + length > metadata.blob.size()) {
    return false;
  }

  std::memcpy(metadata.blob.data() + begin, value.data(), length);
  field_out.offset = metadata.blob_bytes_used;
  field_out.length = static_cast<uint32_t>(length);
  metadata.blob_bytes_used += static_cast<uint32_t>(length);
  return true;
}

bool assign_omniembed_string(const emel::model::detail::kv_binding & binding,
                             const std::string_view key,
                             emel::model::data::metadata & metadata,
                             emel::model::data::metadata::string_view & field_out) noexcept {
  const auto * entry = emel::model::detail::find_kv_entry(binding, key);
  if (entry == nullptr) {
    field_out = {};
    return true;
  }

  std::string_view value = {};
  return emel::model::detail::decode_string_value(binding, *entry, value) &&
      copy_metadata_string(metadata, field_out, value);
}

void assign_vision_preprocessing_contract(
    const std::string_view encoder_name,
    emel::model::data::metadata::clip_vision & vision_out) noexcept {
  vision_out.image_size = 0;
  vision_out.preproc_image_size = 0;
  vision_out.image_mean_count = 0u;
  vision_out.image_std_count = 0u;
  vision_out.image_mean.fill(0.0f);
  vision_out.image_std.fill(0.0f);

  if (encoder_name != k_mobilenetv4_medium_encoder) {
    return;
  }

  vision_out.image_size = k_mobilenetv4_medium_image_size;
  vision_out.preproc_image_size = k_mobilenetv4_medium_image_size;
  vision_out.image_mean_count = static_cast<uint32_t>(k_imagenet_mean.size());
  vision_out.image_std_count = static_cast<uint32_t>(k_imagenet_std.size());
  for (size_t index = 0; index < k_imagenet_mean.size(); ++index) {
    vision_out.image_mean[index] = k_imagenet_mean[index];
    vision_out.image_std[index] = k_imagenet_std[index];
  }
}

void assign_audio_preprocessing_contract(
    const std::string_view encoder_name,
    emel::model::data::metadata::clip_audio & audio_out) noexcept {
  audio_out.sample_rate = 0;
  audio_out.n_fft = 0;
  audio_out.win_length = 0;
  audio_out.hop_size = 0;
  audio_out.num_mel_bins = 0;
  audio_out.low_frequency = 0.0f;
  audio_out.high_frequency = 0.0f;
  audio_out.preemphasis_coefficient = 0.0f;
  audio_out.log_offset = 0.0f;
  audio_out.normalize_bias = 0.0f;
  audio_out.normalize_scale = 0.0f;

  if (encoder_name != k_efficientat_mn20_as_encoder) {
    return;
  }

  audio_out.sample_rate = k_efficientat_mn20_as_sample_rate;
  audio_out.n_fft = k_efficientat_mn20_as_n_fft;
  audio_out.win_length = k_efficientat_mn20_as_win_length;
  audio_out.hop_size = k_efficientat_mn20_as_hop_size;
  audio_out.num_mel_bins = k_efficientat_mn20_as_num_mel_bins;
  audio_out.low_frequency = k_efficientat_mn20_as_low_frequency;
  audio_out.high_frequency = k_efficientat_mn20_as_high_frequency;
  audio_out.preemphasis_coefficient = k_efficientat_mn20_as_preemphasis;
  audio_out.log_offset = k_efficientat_mn20_as_log_offset;
  audio_out.normalize_bias = k_efficientat_mn20_as_normalize_bias;
  audio_out.normalize_scale = k_efficientat_mn20_as_normalize_scale;
}

emel::error::type validate_contract(const emel::model::data & model_data,
                                    execution_contract * contract_out) noexcept {
  if (!is_execution_architecture(emel::model::architecture_name_view(model_data)) ||
      model_data.params.n_embd_out <= 0 ||
      model_data.meta.clip_vision_data.embedding_length <= 0 ||
      model_data.meta.clip_audio_data.embedding_length <= 0 ||
      model_data.meta.clip_vision_data.projection_dim != model_data.params.n_embd_out ||
      model_data.meta.clip_audio_data.projection_dim != model_data.params.n_embd_out ||
      model_data.meta.clip_vision_data.preproc_image_size <= 0 ||
      model_data.meta.clip_audio_data.sample_rate <= 0 ||
      model_data.meta.clip_audio_data.n_fft <= 0 ||
      model_data.meta.clip_audio_data.win_length <= 0 ||
      model_data.meta.clip_audio_data.hop_size <= 0 ||
      model_data.meta.clip_audio_data.num_mel_bins <= 0 ||
      model_data.meta.clip_audio_data.preemphasis_coefficient <= 0.0f ||
      model_data.meta.clip_audio_data.log_offset <= 0.0f ||
      model_data.meta.clip_audio_data.normalize_scale <= 0.0f ||
      !model_data.meta.clip_data.has_vision_encoder ||
      !model_data.meta.clip_data.has_audio_encoder ||
      !validate_matryoshka_dimensions(model_data.params.matryoshka_dimensions,
                                      model_data.params.matryoshka_dimension_count,
                                      model_data.params.n_embd_out)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  execution_contract contract = {};
  contract.model = &model_data;
  contract.embedding_length = model_data.params.n_embd_out;
  contract.image_encoder_length = model_data.meta.clip_vision_data.embedding_length;
  contract.audio_encoder_length = model_data.meta.clip_audio_data.embedding_length;
  contract.matryoshka_dimension_count = model_data.params.matryoshka_dimension_count;
  contract.matryoshka_dimensions = model_data.params.matryoshka_dimensions;

  const bool families_ok =
      assign_family_view(model_data, k_text_encoder_prefix, contract.text_encoder) &&
      assign_family_view(model_data, k_text_projection_prefix, contract.text_projection) &&
      assign_family_view(model_data, k_image_encoder_prefix, contract.image_encoder) &&
      assign_family_view(model_data, k_image_projection_prefix, contract.image_projection) &&
      assign_family_view(model_data, k_audio_encoder_prefix, contract.audio_encoder) &&
      assign_family_view(model_data, k_audio_projection_prefix, contract.audio_projection);
  if (!families_ok) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  if (contract_out != nullptr) {
    *contract_out = contract;
  }
  return emel::error::cast(emel::model::loader::error::none);
}

}  // namespace

bool is_execution_architecture(const std::string_view architecture) noexcept {
  return architecture == k_architecture;
}

bool load_hparams(const emel::model::detail::hparam_loader & loader,
                  emel::model::data & model_out) noexcept {
  if (!loader.assign_i32("omniembed.embed_dim", model_out.params.n_embd_out) ||
      !loader.assign_i32(
          "omniembed.image_encoder_dim", model_out.meta.clip_vision_data.embedding_length) ||
      !loader.assign_i32(
          "omniembed.audio_encoder_dim", model_out.meta.clip_audio_data.embedding_length) ||
      !copy_matryoshka_dimensions(
          loader.binding,
          "omniembed.matryoshka_dims",
          model_out.params.matryoshka_dimensions,
          model_out.params.matryoshka_dimension_count)) {
    return false;
  }

  if (!assign_omniembed_string(loader.binding,
                               k_image_encoder_name_key,
                               model_out.meta,
                               model_out.meta.clip_vision_data.encoder_name) ||
      !assign_omniembed_string(loader.binding,
                               k_audio_encoder_name_key,
                               model_out.meta,
                               model_out.meta.clip_audio_data.encoder_name)) {
    return false;
  }

  model_out.params.n_embd = model_out.params.n_embd_out;
  model_out.meta.clip_data.has_vision_encoder = model_out.meta.clip_vision_data.embedding_length > 0;
  model_out.meta.clip_data.has_audio_encoder = model_out.meta.clip_audio_data.embedding_length > 0;
  model_out.meta.clip_vision_data.projection_dim = model_out.params.n_embd_out;
  model_out.meta.clip_audio_data.projection_dim = model_out.params.n_embd_out;
  assign_vision_preprocessing_contract(
      emel::model::metadata_string_view(model_out.meta, model_out.meta.clip_vision_data.encoder_name),
      model_out.meta.clip_vision_data);
  assign_audio_preprocessing_contract(
      emel::model::metadata_string_view(model_out.meta, model_out.meta.clip_audio_data.encoder_name),
      model_out.meta.clip_audio_data);
  return true;
}

emel::error::type build_execution_contract(const emel::model::data & model_data,
                                           execution_contract & contract_out) noexcept {
  contract_out = {};
  return validate_contract(model_data, &contract_out);
}

emel::error::type validate_data(const emel::model::data & model_data) noexcept {
  return validate_contract(model_data, nullptr);
}

emel::error::type validate_execution_contract(const emel::model::data & model_data) noexcept {
  return validate_contract(model_data, nullptr);
}

}  // namespace emel::model::omniembed::detail
