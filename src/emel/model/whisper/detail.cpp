#include "emel/model/whisper/detail.hpp"

#include <array>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"

namespace emel::model::whisper::detail {

namespace {

constexpr std::string_view k_architecture = "whisper";
constexpr std::string_view k_mel_filters_prefix = "mel_filters";
constexpr std::string_view k_encoder_prefix = "model.encoder.";
constexpr std::string_view k_decoder_prefix = "model.decoder.";
constexpr int32_t k_sample_rate = 16000;
constexpr int32_t k_mel_bin_count = 80;
constexpr int32_t k_vocab_size = 51865;
constexpr int32_t k_embedding_length = 384;
constexpr int32_t k_feed_forward_length = 1536;
constexpr int32_t k_attention_head_count = 6;
constexpr int32_t k_encoder_context_length = 1500;
constexpr int32_t k_decoder_context_length = 448;
constexpr int32_t k_encoder_block_count = 4;
constexpr int32_t k_decoder_block_count = 4;
constexpr uint32_t k_gguf_version = 3u;
constexpr uint32_t k_gguf_alignment = 32u;
constexpr uint32_t k_gguf_type_uint32 = 4u;
constexpr uint32_t k_gguf_type_string = 8u;
constexpr uint32_t k_dtype_f32 = 0u;
constexpr uint32_t k_dtype_f16 = 1u;
constexpr uint32_t k_dtype_q8_0 = 8u;

struct legacy_reader {
  std::span<const uint8_t> bytes = {};
  size_t offset = 0u;

  bool read(std::span<const uint8_t> &out, const size_t size) noexcept {
    if (offset + size > bytes.size()) {
      return false;
    }
    out = bytes.subspan(offset, size);
    offset += size;
    return true;
  }

  bool read_u32(uint32_t &out) noexcept {
    std::span<const uint8_t> raw = {};
    if (!read(raw, sizeof(out))) {
      return false;
    }
    std::memcpy(&out, raw.data(), sizeof(out));
    return true;
  }

  bool read_i32(int32_t &out) noexcept {
    std::span<const uint8_t> raw = {};
    if (!read(raw, sizeof(out))) {
      return false;
    }
    std::memcpy(&out, raw.data(), sizeof(out));
    return true;
  }
};

struct normalized_tensor_record {
  std::string name = {};
  std::array<uint64_t, 4> dims = {};
  uint32_t n_dims = 0u;
  uint32_t dtype = 0u;
  std::span<const uint8_t> data = {};
};

struct legacy_hparams {
  int32_t n_vocab = 0;
  int32_t n_audio_ctx = 0;
  int32_t n_audio_state = 0;
  int32_t n_audio_head = 0;
  int32_t n_audio_layer = 0;
  int32_t n_text_ctx = 0;
  int32_t n_text_state = 0;
  int32_t n_text_head = 0;
  int32_t n_text_layer = 0;
  int32_t n_mels = 0;
  int32_t ftype = 0;
};

bool legacy_read_hparams(legacy_reader &reader,
                         legacy_hparams &hparams) noexcept {
  return reader.read_i32(hparams.n_vocab) &&
         reader.read_i32(hparams.n_audio_ctx) &&
         reader.read_i32(hparams.n_audio_state) &&
         reader.read_i32(hparams.n_audio_head) &&
         reader.read_i32(hparams.n_audio_layer) &&
         reader.read_i32(hparams.n_text_ctx) &&
         reader.read_i32(hparams.n_text_state) &&
         reader.read_i32(hparams.n_text_head) &&
         reader.read_i32(hparams.n_text_layer) &&
         reader.read_i32(hparams.n_mels) && reader.read_i32(hparams.ftype);
}

bool tensor_data_size(const uint32_t dtype, const std::array<uint64_t, 4> &dims,
                      const uint32_t n_dims, size_t &size_out) noexcept {
  uint64_t elements = 1u;
  for (uint32_t index = 0u; index < n_dims; ++index) {
    elements *= dims[index];
  }
  if (dtype == k_dtype_f32) {
    size_out = static_cast<size_t>(elements * 4u);
    return true;
  }
  if (dtype == k_dtype_f16) {
    size_out = static_cast<size_t>(elements * 2u);
    return true;
  }
  if (dtype == k_dtype_q8_0 && n_dims > 0u && dims[0] % 32u == 0u) {
    size_out = static_cast<size_t>((elements / 32u) * 34u);
    return true;
  }
  return false;
}

bool canonical_attention_suffix(const std::string_view suffix,
                                std::string &out) {
  if (suffix == "query.weight") {
    out = "q_proj.weight";
  } else if (suffix == "query.bias") {
    out = "q_proj.bias";
  } else if (suffix == "key.weight") {
    out = "k_proj.weight";
  } else if (suffix == "value.weight") {
    out = "v_proj.weight";
  } else if (suffix == "value.bias") {
    out = "v_proj.bias";
  } else if (suffix == "out.weight") {
    out = "out_proj.weight";
  } else if (suffix == "out.bias") {
    out = "out_proj.bias";
  } else {
    return false;
  }
  return true;
}

bool canonical_encoder_suffix(const std::string_view suffix, std::string &out) {
  if (suffix == "attn_ln.weight") {
    out = "self_attn_layer_norm.weight";
  } else if (suffix == "attn_ln.bias") {
    out = "self_attn_layer_norm.bias";
  } else if (suffix == "attn.query.weight") {
    out = "self_attn.q_proj.weight";
  } else if (suffix == "attn.query.bias") {
    out = "self_attn.q_proj.bias";
  } else if (suffix == "attn.key.weight") {
    out = "self_attn.k_proj.weight";
  } else if (suffix == "attn.value.weight") {
    out = "self_attn.v_proj.weight";
  } else if (suffix == "attn.value.bias") {
    out = "self_attn.v_proj.bias";
  } else if (suffix == "attn.out.weight") {
    out = "self_attn.out_proj.weight";
  } else if (suffix == "attn.out.bias") {
    out = "self_attn.out_proj.bias";
  } else if (suffix == "mlp_ln.weight") {
    out = "final_layer_norm.weight";
  } else if (suffix == "mlp_ln.bias") {
    out = "final_layer_norm.bias";
  } else if (suffix == "mlp.0.weight") {
    out = "fc1.weight";
  } else if (suffix == "mlp.0.bias") {
    out = "fc1.bias";
  } else if (suffix == "mlp.2.weight") {
    out = "fc2.weight";
  } else if (suffix == "mlp.2.bias") {
    out = "fc2.bias";
  } else {
    return false;
  }
  return true;
}

bool canonical_decoder_suffix(const std::string_view suffix, std::string &out) {
  if (suffix.starts_with("cross_attn_ln.")) {
    out = "encoder_attn_layer_norm.";
    out += suffix.substr(std::string_view{"cross_attn_ln."}.size());
    return true;
  }
  return canonical_encoder_suffix(suffix, out);
}

bool canonical_name(const std::string_view name, std::string &out) {
  if (name == "encoder.positional_embedding") {
    out = "model.encoder.embed_positions.weight";
    return true;
  }
  if (name == "encoder.conv1.weight" || name == "encoder.conv1.bias" ||
      name == "encoder.conv2.weight" || name == "encoder.conv2.bias") {
    out = "model.";
    out += name;
    return true;
  }
  if (name == "encoder.ln_post.weight") {
    out = "model.encoder.layer_norm.weight";
    return true;
  }
  if (name == "encoder.ln_post.bias") {
    out = "model.encoder.layer_norm.bias";
    return true;
  }
  if (name == "decoder.positional_embedding") {
    out = "model.decoder.embed_positions.weight";
    return true;
  }
  if (name == "decoder.token_embedding.weight") {
    out = "model.decoder.embed_tokens.weight";
    return true;
  }
  if (name == "decoder.ln.weight") {
    out = "model.decoder.layer_norm.weight";
    return true;
  }
  if (name == "decoder.ln.bias") {
    out = "model.decoder.layer_norm.bias";
    return true;
  }

  for (int32_t block = 0; block < k_encoder_block_count; ++block) {
    char encoder_prefix[32] = {};
    char decoder_prefix[32] = {};
    char cross_prefix[48] = {};
    std::snprintf(encoder_prefix, sizeof(encoder_prefix), "encoder.blocks.%d.",
                  block);
    std::snprintf(decoder_prefix, sizeof(decoder_prefix), "decoder.blocks.%d.",
                  block);
    std::snprintf(cross_prefix, sizeof(cross_prefix),
                  "decoder.blocks.%d.cross_attn.", block);
    const std::string_view encoder_view{encoder_prefix};
    const std::string_view decoder_view{decoder_prefix};
    const std::string_view cross_view{cross_prefix};
    std::string suffix = {};
    if (name.starts_with(encoder_view) &&
        canonical_encoder_suffix(name.substr(encoder_view.size()), suffix)) {
      out = "model.encoder.layers." + std::to_string(block) + "." + suffix;
      return true;
    }
    if (name.starts_with(cross_view) &&
        canonical_attention_suffix(name.substr(cross_view.size()), suffix)) {
      out = "model.decoder.layers." + std::to_string(block) + ".encoder_attn." +
            suffix;
      return true;
    }
    if (name.starts_with(decoder_view) &&
        canonical_decoder_suffix(name.substr(decoder_view.size()), suffix)) {
      out = "model.decoder.layers." + std::to_string(block) + "." + suffix;
      return true;
    }
  }
  return false;
}

void canonical_dims(const std::string &name,
                    normalized_tensor_record &tensor) noexcept {
  const bool vector_as_matrix =
      name.ends_with(".bias") || name.ends_with(".weight");
  if (vector_as_matrix && tensor.n_dims == 2u && tensor.dims[0] == 1u) {
    tensor.n_dims = 1u;
    tensor.dims[0] = tensor.dims[1];
    tensor.dims[1] = 0u;
  }
}

void append_u32(std::vector<uint8_t> &out, const uint32_t value) {
  const uint8_t bytes[] = {
      static_cast<uint8_t>(value & 0xffu),
      static_cast<uint8_t>((value >> 8u) & 0xffu),
      static_cast<uint8_t>((value >> 16u) & 0xffu),
      static_cast<uint8_t>((value >> 24u) & 0xffu),
  };
  out.insert(out.end(), bytes, bytes + sizeof(bytes));
}

void append_u64(std::vector<uint8_t> &out, const uint64_t value) {
  for (uint32_t shift = 0u; shift < 64u; shift += 8u) {
    out.push_back(static_cast<uint8_t>((value >> shift) & 0xffu));
  }
}

void append_string(std::vector<uint8_t> &out, const std::string_view value) {
  append_u64(out, static_cast<uint64_t>(value.size()));
  out.insert(out.end(), value.begin(), value.end());
}

void append_kv_string(std::vector<uint8_t> &out, const std::string_view key,
                      const std::string_view value) {
  append_string(out, key);
  append_u32(out, k_gguf_type_string);
  append_string(out, value);
}

void append_kv_u32(std::vector<uint8_t> &out, const std::string_view key,
                   const uint32_t value) {
  append_string(out, key);
  append_u32(out, k_gguf_type_uint32);
  append_u32(out, value);
}

void append_padding(std::vector<uint8_t> &out) {
  const size_t padding =
      (k_gguf_alignment - (out.size() % k_gguf_alignment)) % k_gguf_alignment;
  out.insert(out.end(), padding, 0u);
}

bool parse_legacy_lmgg(std::span<const uint8_t> source, legacy_hparams &hparams,
                       std::vector<normalized_tensor_record> &tensors) {
  legacy_reader reader{source, 0u};
  std::span<const uint8_t> magic = {};
  if (!reader.read(magic, 4u) || std::memcmp(magic.data(), "lmgg", 4u) != 0 ||
      !legacy_read_hparams(reader, hparams) ||
      hparams.n_audio_layer != k_encoder_block_count ||
      hparams.n_text_layer != k_decoder_block_count) {
    return false;
  }

  int32_t n_mel = 0;
  int32_t n_fft = 0;
  std::span<const uint8_t> mel_data = {};
  if (!reader.read_i32(n_mel) || !reader.read_i32(n_fft) ||
      n_mel != k_mel_bin_count || n_fft <= 0 ||
      !reader.read(mel_data, static_cast<size_t>(n_mel) *
                                 static_cast<size_t>(n_fft) * sizeof(float))) {
    return false;
  }
  tensors.push_back(normalized_tensor_record{
      .name = "mel_filters",
      .dims = {static_cast<uint64_t>(n_fft), static_cast<uint64_t>(n_mel), 0u,
               0u},
      .n_dims = 2u,
      .dtype = k_dtype_f32,
      .data = mel_data,
  });

  int32_t vocab_size = 0;
  if (!reader.read_i32(vocab_size) || vocab_size <= 0) {
    return false;
  }
  for (int32_t index = 0; index < vocab_size; ++index) {
    uint32_t token_size = 0u;
    std::span<const uint8_t> token = {};
    if (!reader.read_u32(token_size) || !reader.read(token, token_size)) {
      return false;
    }
  }

  while (reader.offset < source.size()) {
    int32_t n_dims_raw = 0;
    int32_t name_size_raw = 0;
    int32_t dtype_raw = 0;
    if (source.size() - reader.offset < 12u || !reader.read_i32(n_dims_raw) ||
        !reader.read_i32(name_size_raw) || !reader.read_i32(dtype_raw) ||
        n_dims_raw <= 0 || n_dims_raw > 4 || name_size_raw <= 0) {
      return false;
    }
    normalized_tensor_record tensor{};
    tensor.n_dims = static_cast<uint32_t>(n_dims_raw);
    tensor.dtype = static_cast<uint32_t>(dtype_raw);
    for (uint32_t dim = 0u; dim < tensor.n_dims; ++dim) {
      int32_t value = 0;
      if (!reader.read_i32(value) || value <= 0) {
        return false;
      }
      tensor.dims[dim] = static_cast<uint64_t>(value);
    }
    std::span<const uint8_t> name_bytes = {};
    if (!reader.read(name_bytes, static_cast<size_t>(name_size_raw))) {
      return false;
    }
    const std::string_view source_name{
        reinterpret_cast<const char *>(name_bytes.data()), name_bytes.size()};
    if (!canonical_name(source_name, tensor.name)) {
      return false;
    }
    canonical_dims(tensor.name, tensor);
    size_t tensor_size = 0u;
    if (!tensor_data_size(tensor.dtype, tensor.dims, tensor.n_dims,
                          tensor_size) ||
        !reader.read(tensor.data, tensor_size)) {
      return false;
    }
    tensors.push_back(std::move(tensor));
  }

  return true;
}

void write_gguf(const legacy_hparams &hparams,
                const std::vector<normalized_tensor_record> &tensors,
                std::vector<uint8_t> &gguf_out) {
  gguf_out.clear();
  gguf_out.insert(gguf_out.end(), {'G', 'G', 'U', 'F'});
  append_u32(gguf_out, k_gguf_version);
  append_u64(gguf_out, tensors.size());
  append_u64(gguf_out, 4u);
  append_kv_string(gguf_out, "general.architecture", "whisper");
  append_kv_u32(gguf_out, "general.alignment", k_gguf_alignment);
  append_kv_u32(gguf_out, "whisper.n_mels",
                static_cast<uint32_t>(hparams.n_mels));
  append_kv_u32(gguf_out, "whisper.n_vocab",
                static_cast<uint32_t>(hparams.n_vocab));

  uint64_t tensor_offset = 0u;
  for (const auto &tensor : tensors) {
    append_string(gguf_out, tensor.name);
    append_u32(gguf_out, tensor.n_dims);
    for (uint32_t dim = 0u; dim < tensor.n_dims; ++dim) {
      append_u64(gguf_out, tensor.dims[dim]);
    }
    append_u32(gguf_out, tensor.dtype);
    append_u64(gguf_out, tensor_offset);
    tensor_offset += tensor.data.size();
    tensor_offset += (k_gguf_alignment - (tensor_offset % k_gguf_alignment)) %
                     k_gguf_alignment;
  }

  append_padding(gguf_out);
  for (const auto &tensor : tensors) {
    gguf_out.insert(gguf_out.end(), tensor.data.begin(), tensor.data.end());
    append_padding(gguf_out);
  }
}

bool require_u32_as_i32(const emel::model::detail::hparam_loader &loader,
                        const std::string_view key, int32_t &field) noexcept {
  uint64_t value = 0u;
  const auto *entry = emel::model::detail::find_kv_entry(loader.binding, key);
  if (entry == nullptr ||
      !emel::model::detail::decode_integer_value(loader.binding, *entry,
                                                 value) ||
      value > static_cast<uint64_t>(INT32_MAX)) {
    return false;
  }

  field = static_cast<int32_t>(value);
  return true;
}

bool tensor_has_storage(
    const emel::model::data::tensor_record &tensor) noexcept {
  if (tensor.data == nullptr || tensor.data_size == 0u || tensor.n_dims <= 0) {
    return false;
  }

  for (int32_t dim = 0;
       dim < tensor.n_dims && dim < static_cast<int32_t>(tensor.dims.size());
       ++dim) {
    if (tensor.dims[static_cast<size_t>(dim)] <= 0) {
      return false;
    }
  }

  return true;
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

bool has_encoder_block(const emel::model::data &model_data,
                       const int32_t block_index) noexcept {
  char buffer[64] = {};
  const int written = std::snprintf(
      buffer, sizeof(buffer), "model.encoder.layers.%d.self_attn.q_proj.weight",
      block_index);
  return written > 0 && static_cast<size_t>(written) < sizeof(buffer) &&
         find_tensor(model_data,
                     std::string_view{buffer, static_cast<size_t>(written)}) !=
             nullptr;
}

bool has_decoder_block(const emel::model::data &model_data,
                       const int32_t block_index) noexcept {
  char buffer[64] = {};
  const int written = std::snprintf(
      buffer, sizeof(buffer),
      "model.decoder.layers.%d.encoder_attn.q_proj.weight", block_index);
  return written > 0 && static_cast<size_t>(written) < sizeof(buffer) &&
         find_tensor(model_data,
                     std::string_view{buffer, static_cast<size_t>(written)}) !=
             nullptr;
}

emel::error::type validate_contract(const emel::model::data &model_data,
                                    execution_contract *contract_out) noexcept {
  if (!is_execution_architecture(
          emel::model::architecture_name_view(model_data)) ||
      model_data.params.n_features != k_mel_bin_count ||
      model_data.params.n_vocab != k_vocab_size ||
      model_data.params.n_embd != k_embedding_length ||
      model_data.params.n_ff != k_feed_forward_length ||
      model_data.params.n_head != k_attention_head_count ||
      model_data.params.n_ctx != k_decoder_context_length ||
      model_data.params.n_layer != k_encoder_block_count ||
      model_data.params.decoder_block_count != k_decoder_block_count ||
      !require_tensor_shape(model_data, "mel_filters",
                            {201, k_mel_bin_count}) ||
      !require_tensor_shape(model_data, "model.encoder.conv1.weight",
                            {3, k_mel_bin_count, k_embedding_length}) ||
      !require_tensor_shape(model_data, "model.encoder.embed_positions.weight",
                            {k_embedding_length, k_encoder_context_length}) ||
      !require_tensor_shape(model_data, "model.decoder.embed_tokens.weight",
                            {k_embedding_length, k_vocab_size}) ||
      !require_tensor_shape(model_data, "model.decoder.embed_positions.weight",
                            {k_embedding_length, k_decoder_context_length})) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  for (int32_t block = 0; block < k_encoder_block_count; ++block) {
    if (!has_encoder_block(model_data, block)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
  }

  for (int32_t block = 0; block < k_decoder_block_count; ++block) {
    if (!has_decoder_block(model_data, block)) {
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
  }

  execution_contract contract = {};
  contract.model = &model_data;
  contract.sample_rate = k_sample_rate;
  contract.mel_bin_count = k_mel_bin_count;
  contract.vocab_size = k_vocab_size;
  contract.embedding_length = k_embedding_length;
  contract.feed_forward_length = k_feed_forward_length;
  contract.attention_head_count = k_attention_head_count;
  contract.encoder_context_length = k_encoder_context_length;
  contract.decoder_context_length = k_decoder_context_length;
  contract.encoder_block_count = k_encoder_block_count;
  contract.decoder_block_count = k_decoder_block_count;

  const bool families_ok =
      assign_family_view(model_data, k_mel_filters_prefix,
                         contract.mel_filters) &&
      assign_family_view(model_data, k_encoder_prefix, contract.encoder) &&
      assign_family_view(model_data, k_decoder_prefix, contract.decoder);
  if (!families_ok) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
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

bool is_legacy_lmgg_whisper(std::span<const uint8_t> source) noexcept {
  return source.size() >= 4u && std::memcmp(source.data(), "lmgg", 4u) == 0;
}

bool normalize_legacy_lmgg_to_gguf(std::span<const uint8_t> source,
                                   std::vector<uint8_t> &gguf_out) {
  legacy_hparams hparams = {};
  std::vector<normalized_tensor_record> tensors = {};
  if (!parse_legacy_lmgg(source, hparams, tensors)) {
    return false;
  }
  write_gguf(hparams, tensors, gguf_out);
  return !gguf_out.empty();
}

bool load_hparams(const emel::model::detail::hparam_loader &loader,
                  emel::model::data &model_out) noexcept {
  int32_t n_mels = 0;
  int32_t n_vocab = 0;
  if (!require_u32_as_i32(loader, "whisper.n_mels", n_mels) ||
      !require_u32_as_i32(loader, "whisper.n_vocab", n_vocab) ||
      n_mels != k_mel_bin_count || n_vocab != k_vocab_size) {
    return false;
  }

  model_out.params.n_features = n_mels;
  model_out.params.n_vocab = n_vocab;
  model_out.params.n_embd = k_embedding_length;
  model_out.params.n_embd_out = k_embedding_length;
  model_out.params.n_ff = k_feed_forward_length;
  model_out.params.n_head = k_attention_head_count;
  model_out.params.n_head_kv = k_attention_head_count;
  model_out.params.n_ctx = k_decoder_context_length;
  model_out.params.n_layer = k_encoder_block_count;
  model_out.params.decoder_block_count = k_decoder_block_count;
  return true;
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

} // namespace emel::model::whisper::detail
