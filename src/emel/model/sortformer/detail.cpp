#include "emel/model/sortformer/detail.hpp"

#include <climits>
#include <cstddef>
#include <initializer_list>

#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"

namespace emel::model::sortformer::detail {

namespace {

constexpr std::string_view k_architecture = "sortformer";
constexpr std::string_view k_source_format = "nemo";
constexpr std::string_view k_tensor_name_scheme = "compact_v1";
constexpr std::string_view k_outtype = "f32";
constexpr std::string_view k_feature_extractor_prefix = "prep.";
constexpr std::string_view k_encoder_prefix = "enc.";
constexpr std::string_view k_modules_prefix = "mods.";
constexpr std::string_view k_transformer_encoder_prefix = "te.";
constexpr int32_t k_sample_rate = 16000;
constexpr int32_t k_speaker_count = 4;
constexpr int32_t k_frame_shift_ms = 80;
constexpr int32_t k_chunk_len = 188;
constexpr int32_t k_chunk_right_context = 1;
constexpr int32_t k_fifo_len = 0;
constexpr int32_t k_spkcache_update_period = 188;
constexpr int32_t k_spkcache_len = 188;

bool require_string(const emel::model::detail::kv_binding & binding,
                    const std::string_view key,
                    const std::string_view expected) noexcept {
  const auto * entry = emel::model::detail::find_kv_entry(binding, key);
  if (entry == nullptr) {
    return false;
  }

  std::string_view value = {};
  return emel::model::detail::decode_string_value(binding, *entry, value) && value == expected;
}

bool assign_i32_any(const emel::model::detail::kv_binding & binding,
                    const std::initializer_list<std::string_view> keys,
                    int32_t & field) noexcept {
  const auto * entry = emel::model::detail::find_kv_entry_any(binding, keys);
  if (entry == nullptr) {
    return true;
  }

  uint64_t value = 0u;
  if (!emel::model::detail::decode_integer_value(binding, *entry, value) ||
      value > static_cast<uint64_t>(INT32_MAX)) {
    return false;
  }

  field = static_cast<int32_t>(value);
  return true;
}

bool require_positive_i32(const emel::model::detail::hparam_loader & loader,
                          const std::string_view key,
                          int32_t & field) noexcept {
  field = 0;
  return loader.assign_i32(key, field) && field > 0;
}

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

emel::error::type validate_contract(const emel::model::data & model_data,
                                    execution_contract * contract_out) noexcept {
  if (!is_execution_architecture(emel::model::architecture_name_view(model_data)) ||
      model_data.params.n_features != k_speaker_count) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  execution_contract contract = {};
  contract.model = &model_data;
  contract.sample_rate = k_sample_rate;
  contract.speaker_count = k_speaker_count;
  contract.frame_shift_ms = k_frame_shift_ms;
  contract.chunk_len = k_chunk_len;
  contract.chunk_right_context = k_chunk_right_context;
  contract.fifo_len = k_fifo_len;
  contract.spkcache_update_period = k_spkcache_update_period;
  contract.spkcache_len = k_spkcache_len;

  const bool families_ok =
      assign_family_view(model_data, k_feature_extractor_prefix, contract.feature_extractor) &&
      assign_family_view(model_data, k_encoder_prefix, contract.encoder) &&
      assign_family_view(model_data, k_modules_prefix, contract.modules) &&
      assign_family_view(model_data, k_transformer_encoder_prefix,
                         contract.transformer_encoder);
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
  int32_t original_tensor_count = 0;
  int32_t tensor_count = 0;
  int32_t skipped_tensor_count = 0;
  int32_t sample_rate = k_sample_rate;
  int32_t speaker_count = k_speaker_count;
  int32_t chunk_len = k_chunk_len;
  int32_t chunk_right_context = k_chunk_right_context;
  int32_t fifo_len = k_fifo_len;
  int32_t spkcache_update_period = k_spkcache_update_period;
  int32_t spkcache_len = k_spkcache_len;

  if (!require_string(loader.binding, "sortformer.source.format", k_source_format) ||
      !require_string(loader.binding, "sortformer.tensor_name_scheme", k_tensor_name_scheme) ||
      !require_string(loader.binding, "sortformer.outtype", k_outtype) ||
      !require_positive_i32(loader, "sortformer.original_tensor_count", original_tensor_count) ||
      !require_positive_i32(loader, "sortformer.tensor_count", tensor_count) ||
      !loader.assign_i32("sortformer.skipped_tensor_count", skipped_tensor_count) ||
      !assign_i32_any(loader.binding,
                      {"sortformer.config.preprocessor.sample_rate",
                       "sortformer.config.sample_rate"},
                      sample_rate) ||
      !assign_i32_any(loader.binding,
                      {"sortformer.config.sortformer_modules.num_spks",
                       "sortformer.config.sortformer_modules.num_speakers",
                       "sortformer.config.num_spks"},
                      speaker_count) ||
      !assign_i32_any(loader.binding,
                      {"sortformer.config.sortformer_modules.chunk_len"},
                      chunk_len) ||
      !assign_i32_any(loader.binding,
                      {"sortformer.config.sortformer_modules.chunk_right_context"},
                      chunk_right_context) ||
      !assign_i32_any(loader.binding,
                      {"sortformer.config.sortformer_modules.fifo_len"},
                      fifo_len) ||
      !assign_i32_any(loader.binding,
                      {"sortformer.config.sortformer_modules.spkcache_update_period"},
                      spkcache_update_period) ||
      !assign_i32_any(loader.binding,
                      {"sortformer.config.sortformer_modules.spkcache_len"},
                      spkcache_len) ||
      skipped_tensor_count < 0 ||
      sample_rate != k_sample_rate ||
      speaker_count != k_speaker_count ||
      chunk_len != k_chunk_len ||
      chunk_right_context != k_chunk_right_context ||
      fifo_len != k_fifo_len ||
      spkcache_update_period != k_spkcache_update_period ||
      spkcache_len != k_spkcache_len) {
    return false;
  }

  model_out.params.n_features = speaker_count;
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

}  // namespace emel::model::sortformer::detail
