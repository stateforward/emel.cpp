#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/io/loader/sm.hpp"
#include "emel/io/read/sm.hpp"
#include "emel/kernel/detail.hpp"
#include "emel/kernel/events.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/llama/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/omniembed/detail.hpp"
#include "emel/model/sortformer/detail.hpp"
#include "emel/model/tensor/sm.hpp"
#include "emel/model/whisper/detail.hpp"

namespace {

struct owner_state {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::model::loader::error::none);
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
  emel::io::loader::event::strategy_kind requested_io_strategy =
      emel::io::loader::event::strategy_kind::none;
  emel::io::loader::event::strategy_kind used_io_strategy =
      emel::io::loader::event::strategy_kind::none;
};

void on_done(void *object,
             const emel::model::loader::events::load_done &ev) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->done = true;
  owner->error = false;
  owner->bytes_total = ev.bytes_total;
  owner->bytes_done = ev.bytes_done;
  owner->used_mmap = ev.used_mmap;
  owner->used_io_strategy = ev.used_io_strategy;
}

void on_error(void *object,
              const emel::model::loader::events::load_error &ev) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->done = false;
  owner->error = true;
  owner->err = ev.err;
  owner->requested_io_strategy = ev.requested_io_strategy;
  owner->used_io_strategy = ev.used_io_strategy;
}

emel::error::type
parse_ok(void *, const emel::model::loader::event::load &req) noexcept {
  req.model_data.n_tensors = 1;
  req.model_data.n_layers = 1;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type parse_model_path_weights_ok(
    void *, const emel::model::loader::event::load &req) noexcept {
  req.model_data.n_tensors = 1;
  req.model_data.n_layers = 1;
  req.model_data.weights_data = req.model_data.tensors.data();
  req.model_data.weights_size = 777u;
  req.model_data.weights_split_count = 2u;
  req.model_data.weights_split_offsets[0] = 11u;
  req.model_data.weights_split_offsets[1] = 22u;
  req.model_data.weights_split_sizes[0] = 333u;
  req.model_data.weights_split_sizes[1] = 444u;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
parse_tensor_span_ok(void *,
                     const emel::model::loader::event::load &req) noexcept {
  req.model_data.n_tensors = 1;
  req.model_data.n_layers = 1;
  auto &tensor = req.model_data.tensors[0];
  tensor.file_offset = 2048u;
  tensor.data_size = 128u;
  tensor.file_index = 3u;
  tensor.data = &tensor;
  return emel::error::cast(emel::model::loader::error::none);
}

struct read_copy_parse_state {
  std::array<uint8_t, 4> *target = nullptr;
};

emel::error::type
parse_read_copy_tensor(void *object,
                       const emel::model::loader::event::load &req) noexcept {
  auto *state = static_cast<read_copy_parse_state *>(object);
  req.model_data.n_tensors = 1;
  req.model_data.n_layers = 1;
  auto &tensor = req.model_data.tensors[0];
  tensor.file_offset = 2u;
  tensor.data_size = state->target->size();
  tensor.file_index = 0u;
  tensor.data = state->target->data();
  return emel::error::cast(emel::model::loader::error::none);
}

struct read_copy_batch_parse_state {
  std::array<uint8_t, 4> *first_target = nullptr;
  std::array<uint8_t, 3> *second_target = nullptr;
};

emel::error::type parse_read_copy_two_tensors(
    void *object, const emel::model::loader::event::load &req) noexcept {
  auto *state = static_cast<read_copy_batch_parse_state *>(object);
  req.model_data.n_tensors = 2;
  req.model_data.n_layers = 1;

  auto &first = req.model_data.tensors[0];
  first.file_offset = 2u;
  first.data_size = state->first_target->size();
  first.file_index = 0u;
  first.data = state->first_target->data();

  auto &second = req.model_data.tensors[1];
  second.file_offset = 5u;
  second.data_size = state->second_target->size();
  second.file_index = 0u;
  second.data = state->second_target->data();

  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
parse_no_tensors(void *, const emel::model::loader::event::load &req) noexcept {
  req.model_data.n_tensors = 0;
  req.model_data.n_layers = 1;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
parse_fail(void *, const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::parse_failed);
}

emel::error::type
parse_backend_fail(void *, const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::backend_error);
}

emel::error::type
parse_model_invalid_fail(void *,
                         const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::model_invalid);
}

emel::error::type
parse_internal_fail(void *, const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::internal_error);
}

emel::error::type
parse_untracked_fail(void *,
                     const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::untracked);
}

emel::error::type parse_io_strategy_unavailable_fail(
    void *, const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::io_strategy_unavailable);
}

emel::error::type
parse_unknown_fail(void *, const emel::model::loader::event::load &) noexcept {
  return emel::error::type{0x5e17u};
}

struct tensor_loader_fixture {
  emel::model::tensor::sm machine{};
  std::array<emel::model::tensor::effect_request, 4> effect_requests{};
  std::array<emel::model::tensor::effect_result, 4> effect_results{};

  void bind(emel::model::loader::event::load &request) noexcept {
    request.tensor_loader = &machine;
    request.effect_requests = std::span{effect_requests};
    request.effect_results = std::span{effect_results};
  }
};

emel::error::type
map_layers_ok(void *, const emel::model::loader::event::load &req) noexcept {
  req.model_data.n_layers = 2;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
validate_structure_ok(void *,
                      const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
validate_architecture_ok(void *,
                         const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::none);
}

void copy_name(
    std::array<char, emel::model::data::k_max_architecture_name> &dest,
    const std::string_view value) {
  dest.fill('\0');
  const size_t count = std::min(dest.size() - 1u, value.size());
  for (size_t i = 0; i < count; ++i) {
    dest[i] = value[i];
  }
}

void copy_metadata_string(emel::model::data::metadata &metadata,
                          emel::model::data::metadata::string_view &field,
                          const std::string_view value) {
  field = {};
  REQUIRE(metadata.blob_bytes_used + value.size() <= metadata.blob.size());
  const auto offset = static_cast<size_t>(metadata.blob_bytes_used);
  std::memcpy(metadata.blob.data() + offset, value.data(), value.size());
  field.offset = metadata.blob_bytes_used;
  field.length = static_cast<uint32_t>(value.size());
  metadata.blob_bytes_used += static_cast<uint32_t>(value.size());
}

void append_tensor_name(emel::model::data &model,
                        emel::model::data::tensor_record &tensor,
                        const std::string_view name) {
  tensor.name_offset = model.name_bytes_used;
  tensor.name_length = static_cast<uint32_t>(name.size());
  for (size_t i = 0; i < name.size(); ++i) {
    model.name_storage[model.name_bytes_used + static_cast<uint32_t>(i)] =
        name[i];
  }
  model.name_bytes_used += static_cast<uint32_t>(name.size());
  tensor.n_dims = 2;
  tensor.dims[0] = 8;
  tensor.dims[1] = 8;
  tensor.data = &tensor;
  tensor.data_size = 64u;
}

void append_tensor_with_shape(emel::model::data &model,
                              emel::model::data::tensor_record &tensor,
                              const std::string_view name,
                              const std::initializer_list<int64_t> dims) {
  append_tensor_name(model, tensor, name);
  tensor.n_dims = static_cast<int32_t>(dims.size());
  tensor.data_size = 1u;
  size_t index = 0u;
  for (const int64_t dim : dims) {
    tensor.dims[index] = dim;
    ++index;
    tensor.data_size *= static_cast<uint64_t>(dim > 0 ? dim : 0);
  }
}

void build_canonical_model(emel::model::data &model,
                           const int32_t block_count) {
  std::memset(&model, 0, sizeof(model));
  copy_name(model.architecture_name, "llama");
  model.n_layers = block_count;
  model.params.n_embd = 64;
  model.params.n_ctx = 128;
  model.weights_data = model.tensors.data();
  model.weights_size = 4096u;

  uint32_t tensor_index = 0u;
  const auto add = [&](const std::string_view name) {
    append_tensor_name(model, model.tensors[tensor_index], name);
    ++tensor_index;
  };
  const auto add_block = [&](const int32_t block,
                             const std::string_view suffix) {
    add(std::string{"blk."} + std::to_string(block) + "." +
        std::string{suffix});
  };

  add("token_embd.weight");
  add("output_norm.weight");
  add("output.weight");
  for (int32_t block = 0; block < block_count; ++block) {
    add_block(block, "attn_norm.weight");
    add_block(block, "attn_q.weight");
    add_block(block, "attn_k.weight");
    add_block(block, "attn_v.weight");
    add_block(block, "attn_output.weight");
    add_block(block, "ffn_norm.weight");
    add_block(block, "ffn_gate.weight");
    add_block(block, "ffn_down.weight");
    add_block(block, "ffn_up.weight");
  }
  model.n_tensors = tensor_index;
}

void build_qwen3_model(emel::model::data &model, const int32_t block_count,
                       const bool include_q_norm, const bool include_k_norm) {
  std::memset(&model, 0, sizeof(model));
  copy_name(model.architecture_name, "qwen3");
  model.n_layers = block_count;
  model.params.n_embd = 64;
  model.params.n_ctx = 128;
  model.weights_data = model.tensors.data();
  model.weights_size = 4096u;

  uint32_t tensor_index = 0u;
  const auto add = [&](const std::string_view name) {
    append_tensor_name(model, model.tensors[tensor_index], name);
    ++tensor_index;
  };
  const auto add_block = [&](const int32_t block,
                             const std::string_view suffix) {
    add(std::string{"blk."} + std::to_string(block) + "." +
        std::string{suffix});
  };

  add("token_embd.weight");
  add("output_norm.weight");
  add("output.weight");
  for (int32_t block = 0; block < block_count; ++block) {
    add_block(block, "attn_norm.weight");
    add_block(block, "attn_q.weight");
    add_block(block, "attn_k.weight");
    add_block(block, "attn_v.weight");
    if (include_q_norm) {
      add_block(block, "attn_q_norm.weight");
    }
    if (include_k_norm) {
      add_block(block, "attn_k_norm.weight");
    }
    add_block(block, "attn_output.weight");
    add_block(block, "ffn_norm.weight");
    add_block(block, "ffn_gate.weight");
    add_block(block, "ffn_down.weight");
    add_block(block, "ffn_up.weight");
  }
  model.n_tensors = tensor_index;
}

bool is_lfm2_attention_layer(const int32_t block_index) {
  switch (block_index) {
  case 2:
  case 5:
  case 8:
  case 10:
  case 12:
  case 14:
    return true;
  default:
    return false;
  }
}

void build_lfm2_model(emel::model::data &model, const bool include_output_norm,
                      const bool corrupt_conv_block_contract) {
  std::memset(&model, 0, sizeof(model));
  copy_name(model.architecture_name, "lfm2");
  model.n_layers = 16;
  model.params.n_layer = 16;
  model.params.n_ctx = 128000;
  model.params.n_embd = 2048;
  model.params.n_embd_out = 2048;
  model.params.n_head = 32;
  model.params.n_head_kv = 8;
  model.params.n_vocab = 65536;
  model.params.shortconv_l_cache = 3;
  model.params.rope_freq_base = 1000000.0f;
  model.weights_data = model.tensors.data();
  model.weights_size = 4096u;

  uint32_t tensor_index = 0u;
  const auto add = [&](const std::string_view name) {
    append_tensor_name(model, model.tensors[tensor_index], name);
    ++tensor_index;
  };
  const auto add_block = [&](const int32_t block,
                             const std::string_view suffix) {
    add(std::string{"blk."} + std::to_string(block) + "." +
        std::string{suffix});
  };

  add("token_embd.weight");
  if (include_output_norm) {
    add("token_embd_norm.weight");
  }

  for (int32_t block = 0; block < model.n_layers; ++block) {
    add_block(block, "attn_norm.weight");
    add_block(block, "ffn_norm.weight");
    add_block(block, "ffn_gate.weight");
    add_block(block, "ffn_down.weight");
    add_block(block, "ffn_up.weight");

    if (is_lfm2_attention_layer(block)) {
      add_block(block, "attn_q.weight");
      add_block(block, "attn_k.weight");
      add_block(block, "attn_v.weight");
      add_block(block, "attn_q_norm.weight");
      add_block(block, "attn_k_norm.weight");
      add_block(block, "attn_output.weight");
      continue;
    }

    add_block(block, "shortconv.conv.weight");
    add_block(block, "shortconv.in_proj.weight");
    add_block(block, "shortconv.out_proj.weight");
  }

  if (corrupt_conv_block_contract) {
    add("blk.0.attn_q.weight");
  }

  model.n_tensors = tensor_index;
}

void build_gemma4_model(emel::model::data &model,
                        const bool include_output_weight) {
  std::memset(&model, 0, sizeof(model));
  copy_name(model.architecture_name, "gemma4");
  model.n_layers = 35;
  model.params.n_layer = 35;
  model.params.n_ctx = 131072;
  model.params.n_embd = 1536;
  model.params.n_embd_out = 1536;
  model.params.n_ff = 6144;
  model.params.n_head = 8;
  model.params.n_head_kv = 1;
  model.params.attention_key_length = 512;
  model.params.attention_key_length_swa = 256;
  model.params.attention_value_length = 512;
  model.params.attention_value_length_swa = 256;
  model.params.n_vocab = 262144;
  model.params.n_rot = 512;
  model.params.n_rot_swa = 256;
  model.params.full_attention_interval = 5;
  model.params.embd_length_per_layer_input = 256;
  model.params.attention_sliding_window = 512;
  model.params.attention_shared_kv_layers = 20;
  model.params.attention_layer_norm_rms_epsilon = 1e-6f;
  model.params.final_logit_softcapping = 30.0f;
  model.params.rope_freq_base = 1000000.0f;
  model.params.rope_freq_base_swa = 10000.0f;
  model.params.tie_word_embeddings = true;
  model.params.attention_sliding_window_pattern_count = 35u;
  for (int32_t block = 0; block < model.n_layers; ++block) {
    model.params
        .attention_sliding_window_pattern_flags[static_cast<size_t>(block)] =
        ((block + 1) % 5 == 0) ? 0u : 1u;
  }
  model.weights_data = model.tensors.data();
  model.weights_size = 4096u;

  uint32_t tensor_index = 0u;
  const auto add = [&](const std::string_view name) {
    append_tensor_name(model, model.tensors[tensor_index], name);
    ++tensor_index;
  };
  const auto add_block = [&](const int32_t block,
                             const std::string_view suffix) {
    add(std::string{"blk."} + std::to_string(block) + "." +
        std::string{suffix});
  };

  add("token_embd.weight");
  add("output_norm.weight");
  if (include_output_weight) {
    add("output.weight");
  }

  for (int32_t block = 0; block < model.n_layers; ++block) {
    add_block(block, "attn_norm.weight");
    add_block(block, "attn_q.weight");
    add_block(block, "attn_k.weight");
    add_block(block, "attn_v.weight");
    add_block(block, "attn_q_norm.weight");
    add_block(block, "attn_k_norm.weight");
    add_block(block, "attn_output.weight");
    add_block(block, "ffn_norm.weight");
    add_block(block, "ffn_gate.weight");
    add_block(block, "ffn_down.weight");
    add_block(block, "ffn_up.weight");
  }

  model.n_tensors = tensor_index;
}

void build_omniembed_model(emel::model::data &model,
                           const bool include_audio_projection) {
  std::memset(&model, 0, sizeof(model));
  copy_name(model.architecture_name, "omniembed");
  model.params.n_embd = 1280;
  model.params.n_embd_out = 1280;
  model.params.matryoshka_dimension_count = 4u;
  model.params.matryoshka_dimensions[0] = 768;
  model.params.matryoshka_dimensions[1] = 512;
  model.params.matryoshka_dimensions[2] = 256;
  model.params.matryoshka_dimensions[3] = 128;
  model.meta.clip_data.has_vision_encoder = true;
  model.meta.clip_data.has_audio_encoder = true;
  model.meta.clip_vision_data.embedding_length = 640;
  model.meta.clip_vision_data.projection_dim = 1280;
  model.meta.clip_vision_data.image_size = 384;
  model.meta.clip_vision_data.preproc_image_size = 384;
  model.meta.clip_audio_data.embedding_length = 768;
  model.meta.clip_audio_data.projection_dim = 1280;
  model.meta.clip_audio_data.sample_rate = 32000;
  model.meta.clip_audio_data.n_fft = 1024;
  model.meta.clip_audio_data.win_length = 800;
  model.meta.clip_audio_data.hop_size = 320;
  model.meta.clip_audio_data.num_mel_bins = 128;
  model.meta.clip_audio_data.low_frequency = 0.0f;
  model.meta.clip_audio_data.high_frequency = 15000.0f;
  model.meta.clip_audio_data.preemphasis_coefficient = 0.97f;
  model.meta.clip_audio_data.log_offset = 1.0e-5f;
  model.meta.clip_audio_data.normalize_bias = 4.5f;
  model.meta.clip_audio_data.normalize_scale = 5.0f;
  copy_metadata_string(model.meta, model.meta.clip_vision_data.encoder_name,
                       "mobilenetv4_conv_medium.e180_r384_in12k");
  copy_metadata_string(model.meta, model.meta.clip_audio_data.encoder_name,
                       "efficientat_mn20_as");
  model.weights_data = model.tensors.data();
  model.weights_size = 4096u;

  uint32_t tensor_index = 0u;
  const auto add = [&](const std::string_view name) {
    append_tensor_name(model, model.tensors[tensor_index], name);
    ++tensor_index;
  };

  add("text_encoder.backbone.weight");
  add("text_projection.project.weight");
  add("image_encoder.stem.weight");
  add("image_projection.project.weight");
  add("audio_encoder.stem.weight");
  if (include_audio_projection) {
    add("audio_projection.project.weight");
  }

  model.n_tensors = tensor_index;
}

void build_sortformer_model(emel::model::data &model,
                            const bool include_modules_family) {
  std::memset(&model, 0, sizeof(model));
  copy_name(model.architecture_name, "sortformer");
  model.params.n_features = 4;
  model.weights_data = model.tensors.data();
  model.weights_size = 4096u;

  uint32_t tensor_index = 0u;
  const auto add = [&](const std::string_view name) {
    append_tensor_name(model, model.tensors[tensor_index], name);
    ++tensor_index;
  };

  add("prep.feat.fb");
  add("enc.l0.conv.dw.w");
  if (include_modules_family) {
    add("mods.ep.w");
  }
  add("te.l0.sa.q.w");

  model.n_tensors = tensor_index;
}

void build_whisper_model(emel::model::data &model,
                         const bool include_decoder_cross_attn) {
  std::memset(&model, 0, sizeof(model));
  copy_name(model.architecture_name, "whisper");
  model.params.n_features = 80;
  model.params.n_vocab = 51865;
  model.params.n_embd = 384;
  model.params.n_embd_out = 384;
  model.params.n_ff = 1536;
  model.params.n_head = 6;
  model.params.n_head_kv = 6;
  model.params.n_ctx = 448;
  model.params.n_layer = 4;
  model.params.decoder_block_count = 4;
  model.weights_data = model.tensors.data();
  model.weights_size = 4096u;

  uint32_t tensor_index = 0u;
  const auto add = [&](const std::string_view name,
                       const std::initializer_list<int64_t> dims) {
    append_tensor_with_shape(model, model.tensors[tensor_index], name, dims);
    ++tensor_index;
  };
  const auto add_encoder_block = [&](const int32_t block) {
    add(std::string{"model.encoder.layers."} + std::to_string(block) +
            ".self_attn.q_proj.weight",
        {384, 384});
  };
  const auto add_decoder_block = [&](const int32_t block) {
    if (include_decoder_cross_attn) {
      add(std::string{"model.decoder.layers."} + std::to_string(block) +
              ".encoder_attn.q_proj.weight",
          {384, 384});
    }
  };

  add("mel_filters", {201, 80});
  add("model.encoder.conv1.weight", {3, 80, 384});
  add("model.encoder.embed_positions.weight", {384, 1500});
  add("model.decoder.embed_tokens.weight", {384, 51865});
  add("model.decoder.embed_positions.weight", {384, 448});
  add("model.encoder.layer_norm.weight", {384});
  add("model.decoder.layer_norm.weight", {384});
  for (int32_t block = 0; block < 4; ++block) {
    add_encoder_block(block);
    add_decoder_block(block);
  }

  model.n_tensors = tensor_index;
}

template <class value_type>
void append_scalar(std::vector<uint8_t> &bytes, const value_type value) {
  using unsigned_type = std::make_unsigned_t<value_type>;
  const unsigned_type raw = static_cast<unsigned_type>(value);
  for (size_t i = 0u; i < sizeof(value_type); ++i) {
    bytes.push_back(static_cast<uint8_t>((raw >> (i * 8u)) & 0xffu));
  }
}

void append_string_bytes(std::vector<uint8_t> &bytes,
                         const std::string_view value) {
  append_scalar<uint64_t>(bytes, static_cast<uint64_t>(value.size()));
  bytes.insert(bytes.end(), value.begin(), value.end());
}

void append_legacy_whisper_tensor(std::vector<uint8_t> &bytes,
                                  const std::string_view name) {
  append_scalar<int32_t>(bytes, 2);
  append_scalar<int32_t>(bytes, static_cast<int32_t>(name.size()));
  append_scalar<int32_t>(bytes, 0);
  append_scalar<int32_t>(bytes, 1);
  append_scalar<int32_t>(bytes, 1);
  bytes.insert(bytes.end(), name.begin(), name.end());
  append_scalar<uint32_t>(bytes, 0u);
}

void append_kv_entry(std::vector<uint8_t> &arena,
                     std::vector<emel::gguf::loader::kv_entry> &entries,
                     const std::string_view key, const uint32_t value_type,
                     const std::span<const uint8_t> value_bytes) {
  emel::gguf::loader::kv_entry entry = {};
  entry.key_offset = static_cast<uint32_t>(arena.size());
  entry.key_length = static_cast<uint32_t>(key.size());
  arena.insert(arena.end(), key.begin(), key.end());
  entry.value_offset = static_cast<uint32_t>(arena.size());
  entry.value_length = static_cast<uint32_t>(value_bytes.size());
  entry.value_type = value_type;
  arena.insert(arena.end(), value_bytes.begin(), value_bytes.end());
  entries.push_back(entry);
}

void append_kv_string(std::vector<uint8_t> &arena,
                      std::vector<emel::gguf::loader::kv_entry> &entries,
                      const std::string_view key,
                      const std::string_view value) {
  std::vector<uint8_t> encoded = {};
  append_string_bytes(encoded, value);
  append_kv_entry(arena, entries, key,
                  emel::gguf::loader::detail::constants::gguf_type_string,
                  std::span<const uint8_t>{encoded});
}

void append_kv_bool(std::vector<uint8_t> &arena,
                    std::vector<emel::gguf::loader::kv_entry> &entries,
                    const std::string_view key, const bool value) {
  const std::array<uint8_t, 1> encoded = {
      static_cast<uint8_t>(value ? 1u : 0u)};
  append_kv_entry(arena, entries, key,
                  emel::gguf::loader::detail::constants::gguf_type_bool,
                  std::span<const uint8_t>{encoded});
}

void append_kv_u32(std::vector<uint8_t> &arena,
                   std::vector<emel::gguf::loader::kv_entry> &entries,
                   const std::string_view key, const uint32_t value) {
  std::vector<uint8_t> encoded = {};
  append_scalar<uint32_t>(encoded, value);
  append_kv_entry(arena, entries, key,
                  emel::gguf::loader::detail::constants::gguf_type_uint32,
                  std::span<const uint8_t>{encoded});
}

void append_kv_i32(std::vector<uint8_t> &arena,
                   std::vector<emel::gguf::loader::kv_entry> &entries,
                   const std::string_view key, const int32_t value) {
  std::vector<uint8_t> encoded = {};
  append_scalar<int32_t>(encoded, value);
  append_kv_entry(arena, entries, key,
                  emel::gguf::loader::detail::constants::gguf_type_int32,
                  std::span<const uint8_t>{encoded});
}

void append_kv_f32(std::vector<uint8_t> &arena,
                   std::vector<emel::gguf::loader::kv_entry> &entries,
                   const std::string_view key, const float value) {
  std::array<uint8_t, sizeof(float)> encoded = {};
  std::memcpy(encoded.data(), &value, sizeof(float));
  append_kv_entry(arena, entries, key,
                  emel::gguf::loader::detail::constants::gguf_type_float32,
                  std::span<const uint8_t>{encoded});
}

void append_kv_f64(std::vector<uint8_t> &arena,
                   std::vector<emel::gguf::loader::kv_entry> &entries,
                   const std::string_view key, const double value) {
  std::array<uint8_t, sizeof(double)> encoded = {};
  std::memcpy(encoded.data(), &value, sizeof(double));
  append_kv_entry(arena, entries, key,
                  emel::gguf::loader::detail::constants::gguf_type_float64,
                  std::span<const uint8_t>{encoded});
}

void append_kv_string_array(std::vector<uint8_t> &arena,
                            std::vector<emel::gguf::loader::kv_entry> &entries,
                            const std::string_view key,
                            const std::span<const std::string_view> values) {
  std::vector<uint8_t> encoded = {};
  append_scalar<uint32_t>(
      encoded, emel::gguf::loader::detail::constants::gguf_type_string);
  append_scalar<uint64_t>(encoded, static_cast<uint64_t>(values.size()));
  for (const std::string_view value : values) {
    append_string_bytes(encoded, value);
  }
  append_kv_entry(arena, entries, key,
                  emel::gguf::loader::detail::constants::gguf_type_array,
                  std::span<const uint8_t>{encoded});
}

void append_kv_u32_array(std::vector<uint8_t> &arena,
                         std::vector<emel::gguf::loader::kv_entry> &entries,
                         const std::string_view key,
                         const std::span<const uint32_t> values) {
  std::vector<uint8_t> encoded = {};
  append_scalar<uint32_t>(
      encoded, emel::gguf::loader::detail::constants::gguf_type_uint32);
  append_scalar<uint64_t>(encoded, static_cast<uint64_t>(values.size()));
  for (const uint32_t value : values) {
    append_scalar<uint32_t>(encoded, value);
  }
  append_kv_entry(arena, entries, key,
                  emel::gguf::loader::detail::constants::gguf_type_array,
                  std::span<const uint8_t>{encoded});
}

template <class value_type>
void append_kv_scalar_array(std::vector<uint8_t> &arena,
                            std::vector<emel::gguf::loader::kv_entry> &entries,
                            const std::string_view key,
                            const uint32_t element_type,
                            const std::span<const value_type> values) {
  std::vector<uint8_t> encoded = {};
  append_scalar<uint32_t>(encoded, element_type);
  append_scalar<uint64_t>(encoded, static_cast<uint64_t>(values.size()));
  for (const value_type value : values) {
    append_scalar<value_type>(encoded, value);
  }
  append_kv_entry(arena, entries, key,
                  emel::gguf::loader::detail::constants::gguf_type_array,
                  std::span<const uint8_t>{encoded});
}

void append_kv_f32_array(std::vector<uint8_t> &arena,
                         std::vector<emel::gguf::loader::kv_entry> &entries,
                         const std::string_view key,
                         const std::span<const float> values) {
  std::vector<uint8_t> encoded = {};
  append_scalar<uint32_t>(
      encoded, emel::gguf::loader::detail::constants::gguf_type_float32);
  append_scalar<uint64_t>(encoded, static_cast<uint64_t>(values.size()));
  for (const float value : values) {
    std::array<uint8_t, sizeof(float)> bytes = {};
    std::memcpy(bytes.data(), &value, sizeof(float));
    encoded.insert(encoded.end(), bytes.begin(), bytes.end());
  }
  append_kv_entry(arena, entries, key,
                  emel::gguf::loader::detail::constants::gguf_type_array,
                  std::span<const uint8_t>{encoded});
}

std::string_view vocab_piece(const emel::model::data::vocab &vocab,
                             const uint32_t token_id) {
  const auto &entry = vocab.entries[token_id];
  return std::string_view{
      vocab.token_storage.data() + entry.text_offset,
      entry.text_length,
  };
}

std::string_view merge_piece(const emel::model::data::vocab &vocab,
                             const uint32_t merge_id) {
  return std::string_view{
      vocab.merge_storage.data() + vocab.merge_offsets[merge_id],
      vocab.merge_lengths[merge_id],
  };
}

std::filesystem::path repo_root() {
  return std::filesystem::path{__FILE__}
      .parent_path()
      .parent_path()
      .parent_path()
      .parent_path();
}

std::filesystem::path whisper_fixture_path() {
  return repo_root() / "tests" / "models" / "model-tiny-q80.gguf";
}

std::filesystem::path whisper_q4_0_fixture_path() {
  return repo_root() / "tests" / "models" / "whisper-tiny-q4_0.gguf";
}

std::filesystem::path whisper_q4_1_fixture_path() {
  return repo_root() / "tests" / "models" / "whisper-tiny-q4_1.gguf";
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  REQUIRE(stream.good());

  stream.seekg(0, std::ios::end);
  const std::streamsize size = stream.tellg();
  REQUIRE(size > 0);
  stream.seekg(0, std::ios::beg);

  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char *>(bytes.data()), size);
  REQUIRE(stream.good());
  return bytes;
}

std::string read_text_file(const std::filesystem::path &path) {
  std::ifstream stream(path);
  REQUIRE(stream.good());
  return std::string{std::istreambuf_iterator<char>{stream},
                     std::istreambuf_iterator<char>{}};
}

std::string_view function_source(const std::string &source,
                                 const std::string_view function_name) {
  const size_t name_pos = source.find(function_name);
  REQUIRE(name_pos != std::string::npos);

  const size_t body_begin = source.find('{', name_pos);
  REQUIRE(body_begin != std::string::npos);

  size_t depth = 0u;
  for (size_t cursor = body_begin; cursor < source.size(); ++cursor) {
    if (source[cursor] == '{') {
      depth += 1u;
    } else if (source[cursor] == '}') {
      REQUIRE(depth > 0u);
      depth -= 1u;
      if (depth == 0u) {
        return std::string_view{source.data() + name_pos,
                                cursor + 1u - name_pos};
      }
    }
  }

  FAIL("function body not closed");
  return {};
}

size_t count_occurrences(const std::string_view source,
                         const std::string_view needle) {
  size_t count = 0u;
  size_t cursor = 0u;
  while (cursor < source.size()) {
    const size_t next = source.find(needle, cursor);
    if (next == std::string_view::npos) {
      return count;
    }
    ++count;
    cursor = next + needle.size();
  }
  return count;
}

void noop_probe_done(const emel::gguf::loader::events::probe_done &) {}
void noop_probe_error(const emel::gguf::loader::events::probe_error &) {}
void noop_bind_done(const emel::gguf::loader::events::bind_done &) {}
void noop_bind_error(const emel::gguf::loader::events::bind_error &) {}
void noop_parse_done(const emel::gguf::loader::events::parse_done &) {}
void noop_parse_error(const emel::gguf::loader::events::parse_error &) {}

using parse_callback_fn = emel::error::type (*)(
    void *, const emel::model::loader::event::load &) noexcept;

void check_loader_parse_error(const parse_callback_fn parse_fn,
                              const emel::error::type expected) {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_fn};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err == expected);
}

void materialize_tensor_names_from_file(
    emel::model::data &model_data, const std::vector<uint8_t> &file_bytes) {
  model_data.name_bytes_used = 0u;
  for (uint32_t index = 0u; index < model_data.n_tensors; ++index) {
    auto &tensor = model_data.tensors[index];
    const size_t source_offset = static_cast<size_t>(tensor.name_offset);
    const size_t length = static_cast<size_t>(tensor.name_length);
    REQUIRE(source_offset + length <= file_bytes.size());
    REQUIRE(static_cast<size_t>(model_data.name_bytes_used) + length <=
            model_data.name_storage.size());

    std::memcpy(model_data.name_storage.data() + model_data.name_bytes_used,
                file_bytes.data() + source_offset, length);
    tensor.name_offset = model_data.name_bytes_used;
    model_data.name_bytes_used += static_cast<uint32_t>(length);
  }
}

bool load_whisper_fixture_binding(
    const std::vector<uint8_t> &file_bytes, std::vector<uint8_t> &kv_arena,
    std::vector<emel::gguf::loader::kv_entry> &kv_entries,
    emel::model::data &model_out) {
  emel::gguf::loader::sm loader{};
  emel::gguf::loader::requirements requirements = {};
  const auto on_probe_done =
      emel::gguf::loader::event::probe_done_fn::from<&noop_probe_done>();
  const auto on_probe_error =
      emel::gguf::loader::event::probe_error_fn::from<&noop_probe_error>();
  const auto on_bind_done =
      emel::gguf::loader::event::bind_done_fn::from<&noop_bind_done>();
  const auto on_bind_error =
      emel::gguf::loader::event::bind_error_fn::from<&noop_bind_error>();
  const auto on_parse_done =
      emel::gguf::loader::event::parse_done_fn::from<&noop_parse_done>();
  const auto on_parse_error =
      emel::gguf::loader::event::parse_error_fn::from<&noop_parse_error>();

  const emel::gguf::loader::event::probe probe{
      std::span<const uint8_t>{file_bytes}, requirements, on_probe_done,
      on_probe_error};
  if (!loader.process_event(probe) || requirements.tensor_count == 0u ||
      requirements.tensor_count > model_out.tensors.size()) {
    return false;
  }

  kv_arena.resize(static_cast<size_t>(
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements)));
  kv_entries.resize(requirements.kv_count);
  model_out.n_tensors = requirements.tensor_count;

  const emel::gguf::loader::event::bind_storage bind{
      std::span<uint8_t>{kv_arena},
      std::span<emel::gguf::loader::kv_entry>{kv_entries},
      std::span<emel::model::data::tensor_record>{model_out.tensors.data(),
                                                  model_out.n_tensors},
      on_bind_done, on_bind_error};
  if (!loader.process_event(bind)) {
    return false;
  }

  const emel::gguf::loader::event::parse parse{
      std::span<const uint8_t>{file_bytes}, on_parse_done, on_parse_error};
  if (!loader.process_event(parse)) {
    return false;
  }

  return true;
}

} // namespace

TEST_CASE("model loader lifecycle succeeds on full load path") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};
  tensor_loader_fixture tensor_loader{};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  tensor_loader.bind(request);
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK(machine.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.bytes_total == sizeof(file_bytes));
  CHECK(owner.bytes_done == sizeof(file_bytes));
  CHECK_FALSE(owner.used_mmap);
}

TEST_CASE("model loader rejects io strategy when no io actor is bound") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr,
                                                         parse_tensor_span_ok};
  tensor_loader_fixture tensor_loader{};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  tensor_loader.bind(request);
  request.io_strategy = emel::io::loader::event::strategy_kind::read_copy;
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::io_strategy_unavailable));
  CHECK(owner.requested_io_strategy ==
        emel::io::loader::event::strategy_kind::read_copy);
  CHECK(owner.used_io_strategy == emel::io::loader::event::strategy_kind::none);
  CHECK(tensor_loader.effect_requests[0].kind ==
        emel::model::tensor::effect_kind::k_io_load);
  CHECK(tensor_loader.effect_requests[0].strategy ==
        emel::io::loader::event::strategy_kind::read_copy);
  CHECK(tensor_loader.effect_requests[0].tensor_id == 0);
  CHECK(tensor_loader.effect_requests[0].file_index == 3u);
  CHECK(tensor_loader.effect_requests[0].offset == 2048u);
  CHECK(tensor_loader.effect_requests[0].size == 128u);
  CHECK(tensor_loader.effect_requests[0].target == &model->tensors[0]);
}

TEST_CASE(
    "model loader dispatches io actor and fails closed before strategies") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  emel::io::loader::sm io_loader{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr,
                                                         parse_tensor_span_ok};
  tensor_loader_fixture tensor_loader{};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  tensor_loader.bind(request);
  request.io_loader = &io_loader;
  request.io_strategy = emel::io::loader::event::strategy_kind::mapped_file;
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::io_strategy_unavailable));
  CHECK(owner.requested_io_strategy ==
        emel::io::loader::event::strategy_kind::mapped_file);
  CHECK(owner.used_io_strategy == emel::io::loader::event::strategy_kind::none);
  CHECK(tensor_loader.effect_requests[0].kind ==
        emel::model::tensor::effect_kind::k_io_load);
  CHECK(tensor_loader.effect_requests[0].strategy ==
        emel::io::loader::event::strategy_kind::mapped_file);
  CHECK(tensor_loader.effect_requests[0].tensor_id == 0);
  CHECK(tensor_loader.effect_requests[0].file_index == 3u);
  CHECK(tensor_loader.effect_requests[0].offset == 2048u);
  CHECK(tensor_loader.effect_requests[0].size == 128u);
  CHECK(tensor_loader.effect_results[0].handle == nullptr);

  owner = {};
  request.io_loader = nullptr;
  request.io_strategy = emel::io::loader::event::strategy_kind::none;
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};

  CHECK(machine.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
}

TEST_CASE("model loader loads read/copy tensors through maintained actors") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  emel::io::read::sm read_actor{};
  emel::io::loader::sm io_loader{{.io_read = &read_actor}};
  owner_state owner{};
  tensor_loader_fixture tensor_loader{};
  std::array<uint8_t, 4> target{};
  read_copy_parse_state parse_state{.target = &target};
  emel::model::loader::event::parse_model_fn parse_model{
      &parse_state, parse_read_copy_tensor};

  std::array<uint8_t, 8> file_bytes{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
  emel::model::loader::event::load request{*model, parse_model};
  request.model_path = "fixture.gguf";
  request.file_image = file_bytes.data();
  request.file_size = file_bytes.size();
  tensor_loader.bind(request);
  std::array<emel::io::event::tensor_load_span, 1> io_load_spans{};
  request.io_load_spans = std::span{io_load_spans};
  request.io_loader = &io_loader;
  request.io_strategy = emel::io::loader::event::strategy_kind::read_copy;
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  const bool accepted = machine.process_event(request);
  CAPTURE(owner.err);
  CHECK(accepted);
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.bytes_total == file_bytes.size());
  CHECK(owner.bytes_done == file_bytes.size());
  CHECK_FALSE(owner.used_mmap);
  CHECK(owner.used_io_strategy ==
        emel::io::loader::event::strategy_kind::read_copy);
  CHECK(tensor_loader.effect_requests[0].kind ==
        emel::model::tensor::effect_kind::k_io_load);
  CHECK(tensor_loader.effect_requests[0].strategy ==
        emel::io::loader::event::strategy_kind::read_copy);
  CHECK(tensor_loader.effect_results[0].handle == target.data());
  CHECK(model->tensors[0].data == target.data());
  CHECK(target[0] == 'c');
  CHECK(target[1] == 'd');
  CHECK(target[2] == 'e');
  CHECK(target[3] == 'f');
}

TEST_CASE("model loader read copy uses one io loader batch dispatch") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  emel::io::read::sm read_actor{};
  emel::io::loader::sm io_loader{{.io_read = &read_actor}};
  owner_state owner{};
  tensor_loader_fixture tensor_loader{};
  std::array<uint8_t, 4> first_target{};
  std::array<uint8_t, 3> second_target{};
  read_copy_batch_parse_state parse_state{.first_target = &first_target,
                                          .second_target = &second_target};
  emel::model::loader::event::parse_model_fn parse_model{
      &parse_state, parse_read_copy_two_tensors};

  std::array<uint8_t, 9> file_bytes{'a', 'b', 'c', 'd', 'e',
                                    'f', 'g', 'h', 'i'};
  emel::model::loader::event::load request{*model, parse_model};
  request.model_path = "fixture.gguf";
  request.file_image = file_bytes.data();
  request.file_size = file_bytes.size();
  tensor_loader.bind(request);
  std::array<emel::io::event::tensor_load_span, 2> io_load_spans{};
  request.io_load_spans = std::span{io_load_spans};
  request.io_loader = &io_loader;
  request.io_strategy = emel::io::loader::event::strategy_kind::read_copy;
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  const bool accepted = machine.process_event(request);
  CAPTURE(owner.err);
  CHECK(accepted);
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK_FALSE(owner.used_mmap);
  CHECK(owner.used_io_strategy ==
        emel::io::loader::event::strategy_kind::read_copy);
  CHECK(tensor_loader.effect_requests[0].kind ==
        emel::model::tensor::effect_kind::k_io_load);
  CHECK(tensor_loader.effect_requests[1].kind ==
        emel::model::tensor::effect_kind::k_io_load);
  CHECK(tensor_loader.effect_results[0].handle == first_target.data());
  CHECK(tensor_loader.effect_results[1].handle == second_target.data());
  CHECK(model->tensors[0].data == first_target.data());
  CHECK(model->tensors[1].data == second_target.data());
  CHECK(first_target[0] == 'c');
  CHECK(first_target[1] == 'd');
  CHECK(first_target[2] == 'e');
  CHECK(first_target[3] == 'f');
  CHECK(second_target[0] == 'f');
  CHECK(second_target[1] == 'g');
  CHECK(second_target[2] == 'h');
}

TEST_CASE("model loader read copy requires request owned io batch span") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  emel::io::read::sm read_actor{};
  emel::io::loader::sm io_loader{{.io_read = &read_actor}};
  owner_state owner{};
  tensor_loader_fixture tensor_loader{};
  std::array<uint8_t, 4> target{};
  read_copy_parse_state parse_state{.target = &target};
  emel::model::loader::event::parse_model_fn parse_model{
      &parse_state, parse_read_copy_tensor};

  std::array<uint8_t, 8> file_bytes{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
  emel::model::loader::event::load request{*model, parse_model};
  request.model_path = "fixture.gguf";
  request.file_image = file_bytes.data();
  request.file_size = file_bytes.size();
  tensor_loader.bind(request);
  request.io_loader = &io_loader;
  request.io_strategy = emel::io::loader::event::strategy_kind::read_copy;
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  const bool accepted = machine.process_event(request);
  CHECK_FALSE(accepted);
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::io_strategy_unavailable));
  CHECK(owner.requested_io_strategy ==
        emel::io::loader::event::strategy_kind::read_copy);
  CHECK(owner.used_io_strategy == emel::io::loader::event::strategy_kind::none);
  CHECK(tensor_loader.effect_results[0].handle == nullptr);
  CHECK(target[0] == 0u);
  CHECK(target[1] == 0u);
  CHECK(target[2] == 0u);
  CHECK(target[3] == 0u);
}

TEST_CASE("model loader read copy batch error keeps used strategy unset") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  emel::io::loader::sm io_loader{};
  owner_state owner{};
  tensor_loader_fixture tensor_loader{};
  std::array<uint8_t, 4> target{};
  read_copy_parse_state parse_state{.target = &target};
  emel::model::loader::event::parse_model_fn parse_model{
      &parse_state, parse_read_copy_tensor};

  std::array<uint8_t, 8> file_bytes{'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
  emel::model::loader::event::load request{*model, parse_model};
  request.model_path = "fixture.gguf";
  request.file_image = file_bytes.data();
  request.file_size = file_bytes.size();
  tensor_loader.bind(request);
  std::array<emel::io::event::tensor_load_span, 1> io_load_spans{};
  request.io_load_spans = std::span{io_load_spans};
  request.io_loader = &io_loader;
  request.io_strategy = emel::io::loader::event::strategy_kind::read_copy;
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  const bool accepted = machine.process_event(request);
  CHECK_FALSE(accepted);
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::io_strategy_unavailable));
  CHECK(owner.requested_io_strategy ==
        emel::io::loader::event::strategy_kind::read_copy);
  CHECK(owner.used_io_strategy == emel::io::loader::event::strategy_kind::none);
  CHECK(tensor_loader.effect_results[0].handle == nullptr);
  CHECK(target[0] == 0u);
  CHECK(target[1] == 0u);
  CHECK(target[2] == 0u);
  CHECK(target[3] == 0u);
}

TEST_CASE("model loader reports tensor bind errors from the tensor actor") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr,
                                                         parse_tensor_span_ok};
  tensor_loader_fixture tensor_loader{};

  std::array<emel::model::data::tensor_record, 1> busy_tensors{};
  emel::model::tensor::event::bind_storage bind{std::span{busy_tensors}};
  REQUIRE(tensor_loader.machine.process_event(bind));
  std::array<emel::model::tensor::effect_request, 1> busy_effects{};
  emel::model::tensor::event::plan_load plan{std::span{busy_effects}};
  REQUIRE(tensor_loader.machine.process_event(plan));

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  tensor_loader.bind(request);
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::invalid_request));
}

TEST_CASE("model loader supports absent completion callbacks") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};
  tensor_loader_fixture tensor_loader{};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  tensor_loader.bind(request);
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};

  CHECK(machine.process_event(request));
}

TEST_CASE("model loader supports absent error callbacks") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  emel::model::loader::event::load request{*model, parse_model};

  CHECK_FALSE(machine.process_event(request));
}

TEST_CASE("model loader rejects non-vocab loads with no tensors") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr,
                                                         parse_no_tensors};
  tensor_loader_fixture tensor_loader{};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  tensor_loader.bind(request);
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("model loader unexpected events mark runtime context internal") {
  struct unexpected_runtime_event {
    emel::model::loader::event::load_ctx &ctx;
  };

  emel::model::loader::sm machine{};
  emel::model::loader::event::load_ctx ctx{};

  CHECK(machine.process_event(unexpected_runtime_event{ctx}));
  CHECK(ctx.err ==
        emel::error::cast(emel::model::loader::error::internal_error));
}

TEST_CASE("model loader preserves parser weight metadata on model-path load") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{
      nullptr, parse_model_path_weights_ok};
  tensor_loader_fixture tensor_loader{};

  emel::model::loader::event::load request{*model, parse_model};
  request.model_path = "model.gguf";
  tensor_loader.bind(request);
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK(machine.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.bytes_total == 777u);
  CHECK(owner.bytes_done == 777u);
  CHECK_FALSE(owner.used_mmap);
  CHECK(model->weights_data == model->tensors.data());
  CHECK(model->weights_size == 777u);
  CHECK(model->weights_split_count == 2u);
  CHECK(model->weights_split_offsets[0] == 11u);
  CHECK(model->weights_split_offsets[1] == 22u);
  CHECK(model->weights_split_sizes[0] == 333u);
  CHECK(model->weights_split_sizes[1] == 444u);
}

TEST_CASE("model loader rejects missing source payload") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  emel::model::loader::event::load request{*model, parse_model};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::invalid_request));
}

TEST_CASE(
    "model loader allows vocab-only parse without weight and map callbacks") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.vocab_only = true;
  request.check_tensors = false;
  request.validate_architecture = false;
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK(machine.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.bytes_total == 0);
  CHECK(owner.bytes_done == 0);
  CHECK_FALSE(owner.used_mmap);
}

TEST_CASE("model loader propagates parse failure") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_fail};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::parse_failed));
}

TEST_CASE("model loader classifies parse error variants") {
  check_loader_parse_error(
      parse_backend_fail,
      emel::error::cast(emel::model::loader::error::backend_error));
  check_loader_parse_error(
      parse_model_invalid_fail,
      emel::error::cast(emel::model::loader::error::model_invalid));
  check_loader_parse_error(
      parse_internal_fail,
      emel::error::cast(emel::model::loader::error::internal_error));
  check_loader_parse_error(
      parse_untracked_fail,
      emel::error::cast(emel::model::loader::error::untracked));
  check_loader_parse_error(
      parse_io_strategy_unavailable_fail,
      emel::error::cast(emel::model::loader::error::io_strategy_unavailable));
  check_loader_parse_error(parse_unknown_fail, emel::error::type{0x5e17u});
}

TEST_CASE("model loader rejects full load without tensor loader") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::invalid_request));
}

TEST_CASE("model loader rejects full load without map_layers callback") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};
  tensor_loader_fixture tensor_loader{};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  tensor_loader.bind(request);
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::invalid_request));
}

TEST_CASE("model loader rejects full load without structure validator") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};
  tensor_loader_fixture tensor_loader{};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  tensor_loader.bind(request);
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::invalid_request));
}

TEST_CASE("model loader rejects full load without architecture validator") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};
  tensor_loader_fixture tensor_loader{};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  tensor_loader.bind(request);
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::invalid_request));
}

TEST_CASE("model loader can skip architecture validation after full load") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};
  tensor_loader_fixture tensor_loader{};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  tensor_loader.bind(request);
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture = false;
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK(machine.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
}

TEST_CASE("model loader rejects full load without tensor effect storage") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};
  emel::model::tensor::sm tensor_loader{};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.tensor_loader = &tensor_loader;
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::invalid_request));
}

TEST_CASE("model loader tensor bulk phase does not use local capture routing") {
  const std::string actions_source = read_text_file(
      repo_root() / "src" / "emel" / "model" / "loader" / "actions.hpp");
  const std::string sm_source = read_text_file(repo_root() / "src" / "emel" /
                                               "model" / "loader" / "sm.hpp");

  CHECK(actions_source.find("tensor_load_capture") == std::string::npos);
  CHECK(actions_source.find("capture.bind_done") == std::string::npos);
  CHECK(actions_source.find("capture.plan_done") == std::string::npos);
  CHECK(actions_source.find("capture.apply_done") == std::string::npos);
  CHECK(actions_source.find("detail::map_tensor_error(capture.err)") ==
        std::string::npos);
  CHECK(sm_source.find("state_tensor_bind_decision") != std::string::npos);
  CHECK(sm_source.find("state_tensor_plan_decision") != std::string::npos);
  CHECK(sm_source.find("state_tensor_apply_decision") != std::string::npos);
}

TEST_CASE("model loader tensor outcomes use typed phase events not result enum "
          "routing") {
  const std::string actions_source = read_text_file(
      repo_root() / "src" / "emel" / "model" / "loader" / "actions.hpp");
  const std::string events_source = read_text_file(
      repo_root() / "src" / "emel" / "model" / "loader" / "events.hpp");
  const std::string guards_source = read_text_file(
      repo_root() / "src" / "emel" / "model" / "loader" / "guards.hpp");

  CHECK(events_source.find("tensor_load_result") == std::string::npos);
  CHECK(events_source.find("tensor_load_result_kind") == std::string::npos);
  CHECK(actions_source.find("reset_tensor_result") == std::string::npos);
  CHECK(actions_source.find("on_tensor_bind_done") == std::string::npos);
  CHECK(actions_source.find("on_tensor_plan_done") == std::string::npos);
  CHECK(actions_source.find("on_tensor_apply_done") == std::string::npos);
  CHECK(guards_source.find("tensor_result_is") == std::string::npos);
  CHECK(guards_source.find("guard_tensor_bind_done") == std::string::npos);
  CHECK(guards_source.find("guard_tensor_plan_done") == std::string::npos);
  CHECK(guards_source.find("guard_tensor_apply_done") == std::string::npos);
}

TEST_CASE(
    "model loader io boundary uses actor events without helper exposure") {
  const std::string actions_source = read_text_file(
      repo_root() / "src" / "emel" / "model" / "loader" / "actions.hpp");
  const std::string events_source = read_text_file(
      repo_root() / "src" / "emel" / "model" / "loader" / "events.hpp");
  const std::string sm_source = read_text_file(repo_root() / "src" / "emel" /
                                               "model" / "loader" / "sm.hpp");

  CHECK(actions_source.find("emel/io/loader/actions.hpp") == std::string::npos);
  CHECK(actions_source.find("emel/io/loader/detail.hpp") == std::string::npos);
  CHECK(actions_source.find("emel/io/read/events.hpp") == std::string::npos);
  CHECK(actions_source.find("read_tensor_request") == std::string::npos);
  CHECK(actions_source.find("effect_dispatch_io_loads") == std::string::npos);
  CHECK(actions_source.find("effect_dispatch_io_load_batch") !=
        std::string::npos);
  CHECK(actions_source.find("std::ifstream") == std::string::npos);
  CHECK(actions_source.find("::read(") == std::string::npos);
  CHECK(actions_source.find("pread(") == std::string::npos);
  CHECK(actions_source.find("ReadFile") == std::string::npos);
  CHECK(actions_source.find("CreateFile") == std::string::npos);
  CHECK(actions_source.find("map_io_error") == std::string::npos);
  const std::string_view batch_dispatch =
      function_source(actions_source, "effect_dispatch_io_load_batch");
  CHECK(count_occurrences(batch_dispatch, "io_loader->process_event(load)") ==
        1u);
  CHECK(batch_dispatch.find("for (") == std::string_view::npos);
  CHECK(events_source.find("emel/io/loader/sm.hpp") == std::string::npos);
  CHECK(sm_source.find("state_io_load_dispatch") != std::string::npos);
  CHECK(sm_source.find("io_load_error_strategy_unavailable") !=
        std::string::npos);
}

TEST_CASE("maintained tool read copy surfaces avoid direct io read events") {
  const std::array tool_sources{
      "tools/bench/generation_bench.cpp",
      "tools/bench/diarization/sortformer_fixture.hpp",
      "tools/embedded_size/emel_probe/main.cpp",
      "tools/paritychecker/parity_engines.cpp",
  };

  for (const auto *source_path : tool_sources) {
    CAPTURE(source_path);
    const std::string source = read_text_file(repo_root() / source_path);
    CHECK(source.find("bind_model_load_io_strategy") != std::string::npos);
    CHECK(source.find("emel/io/read/events.hpp") == std::string::npos);
    CHECK(source.find("emel/io/read/detail.hpp") == std::string::npos);
    CHECK(source.find("read_tensor_request") == std::string::npos);
    CHECK(source.find("read_file_bytes") == std::string::npos);
    CHECK(source.find("emel/io/source/any.hpp") != std::string::npos);
    CHECK(source.find("emel::io::source::load_file_bytes") !=
          std::string::npos);
    CHECK(source.find("emel::io::read::event::read_tensor") ==
          std::string::npos);
    CHECK(source.find("process_event(capture)") == std::string::npos);
  }
}

TEST_CASE("model loader done evidence reports the public io strategy used") {
  const std::string events_source = read_text_file(
      repo_root() / "src" / "emel" / "model" / "loader" / "events.hpp");
  const std::string actions_source = read_text_file(
      repo_root() / "src" / "emel" / "model" / "loader" / "actions.hpp");
  const std::array tool_sources{
      "tools/bench/generation_bench.cpp",
      "tools/bench/diarization/sortformer_fixture.hpp",
      "tools/embedded_size/emel_probe/main.cpp",
      "tools/paritychecker/parity_engines.cpp",
  };

  CHECK(events_source.find("used_io_strategy") != std::string::npos);
  CHECK(events_source.find("requested_io_strategy") != std::string::npos);
  CHECK(actions_source.find("effect_mark_io_strategy_used") !=
        std::string::npos);
  for (const auto *source_path : tool_sources) {
    CAPTURE(source_path);
    const std::string source = read_text_file(repo_root() / source_path);
    CHECK(source.find("bind_model_load_io_strategy") != std::string::npos);
    CHECK(source.find(".used_io_strategy = ev.used_io_strategy") !=
          std::string::npos);
    CHECK(source.find("process_event(capture)") == std::string::npos);
  }
}

TEST_CASE(
    "model tensor state-machine wrappers keep context and optional output "
    "routing in transitions") {
  const std::string detail_source = read_text_file(
      repo_root() / "src" / "emel" / "model" / "tensor" / "detail.hpp");
  const std::string context_source = read_text_file(
      repo_root() / "src" / "emel" / "model" / "tensor" / "context.hpp");
  const std::string sm_source = read_text_file(repo_root() / "src" / "emel" /
                                               "model" / "tensor" / "sm.hpp");

  CHECK(detail_source.find("bind_or_sink") == std::string::npos);
  CHECK(detail_source.find("choices[") == std::string::npos);
  CHECK(context_source.find(std::string{"bound_"} + "count") ==
        std::string::npos);
  CHECK(sm_source.find("this->context_") == std::string::npos);
  CHECK(sm_source.find("bind_or_sink") == std::string::npos);
}

TEST_CASE(
    "maintained tool parse callbacks do not resize gguf kv storage during "
    "loader dispatch") {
  struct callback_source {
    const char *path;
    const char *function_name;
    const char *prebind_name;
  };

  const callback_source sources[] = {
      {"tools/bench/generation_bench.cpp", "run_emel_parse_model",
       "prebind_emel_gguf_storage"},
      {"tools/bench/diarization/sortformer_fixture.hpp", "run_emel_parse_model",
       "prebind_emel_gguf_storage"},
      {"tools/embedded_size/emel_probe/main.cpp", "run_emel_parse_model",
       "prebind_emel_gguf_storage"},
      {"tools/paritychecker/parity_engines.cpp", "parse_gguf_kv_storage",
       "prebind_gguf_kv_storage"},
  };

  for (const auto &source_info : sources) {
    CAPTURE(source_info.path);
    const std::string source = read_text_file(repo_root() / source_info.path);
    const std::string_view callback =
        function_source(source, source_info.function_name);

    CHECK(source.find(source_info.prebind_name) != std::string::npos);
    CHECK(callback.find(".kv_arena.resize") == std::string_view::npos);
    CHECK(callback.find(".kv_entries.resize") == std::string_view::npos);
  }
}

TEST_CASE("io boundary closeout tests avoid actor internal reach-through") {
  const std::array test_sources{
      "tests/io/loader/lifecycle_tests.cpp",
      "tests/model/tensor/lifecycle_tests.cpp",
      "tests/model/loader/lifecycle_tests.cpp",
  };
  const std::array forbidden{
      std::string{"#include \"emel/io/loader/"} + "actions.hpp\"",
      std::string{"#include \"emel/io/loader/"} + "detail.hpp\"",
      std::string{"#include \"emel/io/loader/"} + "guards.hpp\"",
      std::string{"#include \"emel/model/tensor/"} + "actions.hpp\"",
      std::string{"#include \"emel/model/tensor/"} + "detail.hpp\"",
      std::string{"#include \"emel/model/tensor/"} + "guards.hpp\"",
      std::string{"#include \"emel/model/loader/"} + "actions.hpp\"",
      std::string{"#include \"emel/model/loader/"} + "detail.hpp\"",
      std::string{"#include \"emel/model/loader/"} + "guards.hpp\"",
      std::string{"emel::io::loader::"} + "action::",
      std::string{"emel::io::loader::"} + "detail::",
      std::string{"emel::io::loader::"} + "guard::",
      std::string{"emel::model::tensor::"} + "action::",
      std::string{"emel::model::tensor::"} + "detail::",
      std::string{"emel::model::tensor::"} + "guard::",
      std::string{"emel::model::loader::"} + "action::",
      std::string{"emel::model::loader::"} + "detail::",
      std::string{"emel::model::loader::"} + "guard::",
  };

  for (const auto *source_path : test_sources) {
    CAPTURE(source_path);
    const std::string source = read_text_file(repo_root() / source_path);
    for (const auto &needle : forbidden) {
      CAPTURE(needle);
      CHECK(source.find(needle) == std::string::npos);
    }
  }
}

TEST_CASE("model_llama_detail_builds_execution_view_for_canonical_tensor_set") {
  auto model = std::make_unique<emel::model::data>();
  build_canonical_model(*model, 2);

  emel::model::llama::detail::execution_view view = {};
  const auto err =
      emel::model::llama::detail::build_execution_view(*model, view);

  CHECK(err == emel::error::cast(emel::model::loader::error::none));
  CHECK(view.model == model.get());
  CHECK(view.block_count == 2);
  CHECK(view.token_embedding.name == "token_embd.weight");
  CHECK(view.output_norm.name == "output_norm.weight");
  CHECK(view.output.name == "output.weight");

  emel::model::llama::detail::block_view block = {};
  CHECK(emel::model::llama::detail::lookup_block_view(view, 1, block) ==
        emel::error::cast(emel::model::loader::error::none));
  CHECK(block.index == 1);
  CHECK(block.attention_norm.name == "blk.1.attn_norm.weight");
  CHECK(block.feed_forward_up.name == "blk.1.ffn_up.weight");
}

TEST_CASE("model_llama_detail_builds_execution_view_without_contiguous_weights_"
          "blob") {
  auto model = std::make_unique<emel::model::data>();
  build_canonical_model(*model, 2);
  model->weights_data = nullptr;
  model->weights_size = 0u;

  emel::model::llama::detail::execution_view view = {};
  const auto err =
      emel::model::llama::detail::build_execution_view(*model, view);

  CHECK(err == emel::error::cast(emel::model::loader::error::none));
  CHECK(view.model == model.get());
  CHECK(view.block_count == 2);
  CHECK(view.token_embedding.tensor != nullptr);
  CHECK(view.output.tensor != nullptr);
}

TEST_CASE("model_data_tensor_name_view_rejects_out_of_bounds_storage_range") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::data::tensor_record tensor = {};
  tensor.name_offset = static_cast<uint32_t>(model->name_storage.size() - 1u);
  tensor.name_length = 4u;

  CHECK(emel::model::tensor_name_view(*model, tensor).empty());
}

TEST_CASE("model_data_try_parse_block_index_accepts_canonical_block_name") {
  int32_t block_index = -1;

  CHECK(
      emel::model::try_parse_block_index("blk.12.attn_q.weight", block_index));
  CHECK(block_index == 12);
}

TEST_CASE(
    "model_data_try_parse_block_index_rejects_names_without_block_prefix") {
  int32_t block_index = -1;

  CHECK_FALSE(emel::model::try_parse_block_index("layer.12.attn_q.weight",
                                                 block_index));
}

TEST_CASE("model_data_try_parse_block_index_rejects_prefix_without_digits") {
  int32_t block_index = -1;

  CHECK_FALSE(emel::model::try_parse_block_index("blk.", block_index));
  CHECK_FALSE(
      emel::model::try_parse_block_index("blk.attn_q.weight", block_index));
  CHECK_FALSE(
      emel::model::try_parse_block_index("blk.12attn_q.weight", block_index));
}

TEST_CASE("model_llama_detail_rejects_missing_required_tensor") {
  auto model = std::make_unique<emel::model::data>();
  build_canonical_model(*model, 1);
  model->tensors[3].data = nullptr;
  model->tensors[3].data_size = 0u;

  emel::model::llama::detail::execution_view view = {};
  const auto err =
      emel::model::llama::detail::build_execution_view(*model, view);

  CHECK(err == emel::error::cast(emel::model::loader::error::model_invalid));
  CHECK(view.model == nullptr);
}

TEST_CASE(
    "model_llama_detail_builds_qwen3_execution_view_for_canonical_tensor_set") {
  auto model = std::make_unique<emel::model::data>();
  build_qwen3_model(*model, 1, true, true);

  emel::model::llama::detail::execution_view view = {};
  const auto err =
      emel::model::llama::detail::build_execution_view(*model, view);

  CHECK(err == emel::error::cast(emel::model::loader::error::none));
  CHECK(view.model == model.get());
  CHECK(view.block_count == 1);
  CHECK(view.output.name == "output.weight");

  emel::model::llama::detail::block_view block = {};
  CHECK(emel::model::llama::detail::lookup_block_view(view, 0, block) ==
        emel::error::cast(emel::model::loader::error::none));
  CHECK(block.attention_q_norm.name == "blk.0.attn_q_norm.weight");
  CHECK(block.attention_k_norm.name == "blk.0.attn_k_norm.weight");
}

TEST_CASE("model_llama_detail_rejects_qwen3_execution_view_without_attention_q_"
          "norm") {
  auto model = std::make_unique<emel::model::data>();
  build_qwen3_model(*model, 1, false, true);

  emel::model::llama::detail::execution_view view = {};
  const auto err =
      emel::model::llama::detail::build_execution_view(*model, view);

  CHECK(err == emel::error::cast(emel::model::loader::error::model_invalid));
  CHECK(view.model == nullptr);
}

TEST_CASE("model_llama_detail_rejects_qwen3_execution_view_without_attention_k_"
          "norm") {
  auto model = std::make_unique<emel::model::data>();
  build_qwen3_model(*model, 1, true, false);

  emel::model::llama::detail::execution_view view = {};
  const auto err =
      emel::model::llama::detail::build_execution_view(*model, view);

  CHECK(err == emel::error::cast(emel::model::loader::error::model_invalid));
  CHECK(view.model == nullptr);
}

TEST_CASE("model_llama_detail_builds_qwen3_execution_view_with_tied_output_"
          "fallback") {
  auto model = std::make_unique<emel::model::data>();
  build_qwen3_model(*model, 1, true, true);

  for (uint32_t idx = 0; idx < model->n_tensors; ++idx) {
    auto &tensor = model->tensors[idx];
    if (emel::model::tensor_name_view(*model, tensor) == "output.weight") {
      tensor.data = nullptr;
      tensor.data_size = 0u;
      break;
    }
  }

  emel::model::llama::detail::execution_view view = {};
  const auto err =
      emel::model::llama::detail::build_execution_view(*model, view);

  CHECK(err == emel::error::cast(emel::model::loader::error::none));
  CHECK(view.model == model.get());
  CHECK(view.output.name == "token_embd.weight");
}

TEST_CASE("model_execution_contract_accepts_canonical_llama_contract") {
  auto model = std::make_unique<emel::model::data>();
  build_canonical_model(*model, 1);

  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::none));
}

TEST_CASE("model_execution_contract_rejects_unsupported_architecture") {
  auto model = std::make_unique<emel::model::data>();
  build_canonical_model(*model, 1);
  copy_name(model->architecture_name, "unsupported");

  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("model_execution_contract_accepts_canonical_lfm2_hybrid_contract") {
  auto model = std::make_unique<emel::model::data>();
  build_lfm2_model(*model, true, false);

  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::none));
}

TEST_CASE("model_llama_detail_builds_lfm2_topology_with_hybrid_tensor_count") {
  auto model = std::make_unique<emel::model::data>();
  build_lfm2_model(*model, true, false);

  emel::model::llama::detail::execution_view view = {};
  REQUIRE(emel::model::llama::detail::build_execution_view(*model, view) ==
          emel::error::cast(emel::model::loader::error::none));

  emel::model::llama::detail::topology topology = {};
  REQUIRE(emel::model::llama::detail::build_topology(view, topology) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(topology.tensor_count == 149u);
  CHECK(topology.node_count == 149u);
}

TEST_CASE(
    "model_execution_contract_rejects_lfm2_without_token_embedding_norm") {
  auto model = std::make_unique<emel::model::data>();
  build_lfm2_model(*model, false, false);

  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("model_execution_contract_rejects_lfm2_with_noncanonical_hybrid_"
          "block_tensors") {
  auto model = std::make_unique<emel::model::data>();
  build_lfm2_model(*model, true, true);

  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("model_execution_contract_rejects_lfm2_attention_block_with_"
          "shortconv_weights") {
  auto model = std::make_unique<emel::model::data>();
  build_lfm2_model(*model, true, false);
  append_tensor_name(*model, model->tensors[model->n_tensors],
                     "blk.2.shortconv.conv.weight");
  ++model->n_tensors;

  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("model_detail_loads_gemma4_hparams_from_gguf_binding") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  const std::array<std::string_view, 2> tokens = {"<bos>", "hello"};
  const std::array<uint32_t, 35> feed_forward_lengths = {
      6144u, 6144u, 6144u, 6144u, 6144u, 6144u, 6144u, 6144u, 6144u,
      6144u, 6144u, 6144u, 6144u, 6144u, 6144u, 6144u, 6144u, 6144u,
      6144u, 6144u, 6144u, 6144u, 6144u, 6144u, 6144u, 6144u, 6144u,
      6144u, 6144u, 6144u, 6144u, 6144u, 6144u, 6144u, 6144u,
  };
  const std::array<uint32_t, 35> sliding_pattern = {
      1u, 1u, 1u, 1u, 0u, 1u, 1u, 1u, 1u, 0u, 1u, 1u, 1u, 1u, 0u, 1u, 1u, 1u,
      1u, 0u, 1u, 1u, 1u, 1u, 0u, 1u, 1u, 1u, 1u, 0u, 1u, 1u, 1u, 1u, 0u,
  };

  append_kv_string(arena, entries, "general.architecture", "gemma4");
  append_kv_u32(arena, entries, "gemma4.context_length", 131072u);
  append_kv_u32(arena, entries, "gemma4.embedding_length", 1536u);
  append_kv_u32(arena, entries, "gemma4.embedding_length_per_layer_input",
                256u);
  append_kv_u32_array(arena, entries, "gemma4.feed_forward_length",
                      std::span<const uint32_t>{feed_forward_lengths});
  append_kv_u32(arena, entries, "gemma4.attention.head_count", 8u);
  append_kv_u32(arena, entries, "gemma4.attention.head_count_kv", 1u);
  append_kv_u32(arena, entries, "gemma4.attention.key_length", 512u);
  append_kv_u32(arena, entries, "gemma4.attention.key_length_swa", 256u);
  append_kv_u32(arena, entries, "gemma4.attention.value_length", 512u);
  append_kv_u32(arena, entries, "gemma4.attention.value_length_swa", 256u);
  append_kv_u32(arena, entries, "gemma4.block_count", 35u);
  append_kv_u32(arena, entries, "gemma4.vocab_size", 262144u);
  append_kv_u32(arena, entries, "gemma4.attention.sliding_window", 512u);
  append_kv_u32(arena, entries, "gemma4.attention.shared_kv_layers", 20u);
  append_kv_u32(arena, entries, "gemma4.rope.dimension_count", 512u);
  append_kv_u32(arena, entries, "gemma4.rope.dimension_count_swa", 256u);
  append_kv_f32(arena, entries, "gemma4.attention.layer_norm_rms_epsilon",
                1e-6f);
  append_kv_f32(arena, entries, "gemma4.final_logit_softcapping", 30.0f);
  append_kv_f32(arena, entries, "gemma4.rope.freq_base", 1000000.0f);
  append_kv_f32(arena, entries, "gemma4.rope.freq_base_swa", 10000.0f);
  append_kv_u32_array(arena, entries, "gemma4.attention.sliding_window_pattern",
                      std::span<const uint32_t>{sliding_pattern});
  append_kv_string_array(arena, entries, "tokenizer.tokens",
                         std::span<const std::string_view>{tokens});

  auto model = std::make_unique<emel::model::data>();
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  CHECK(emel::model::architecture_name_view(*model) == "gemma4");
  CHECK(emel::model::is_gemma4_execution_architecture(
      emel::model::architecture_name_view(*model)));
  CHECK(model->params.n_ctx == 131072);
  CHECK(model->params.n_layer == 35);
  CHECK(model->params.n_ff == 6144);
  CHECK(model->params.attention_key_length == 512);
  CHECK(model->params.attention_key_length_swa == 256);
  CHECK(model->params.attention_value_length == 512);
  CHECK(model->params.attention_value_length_swa == 256);
  CHECK(model->params.attention_shared_kv_layers == 20);
  CHECK(model->params.n_rot == 512);
  CHECK(model->params.n_rot_swa == 256);
  CHECK(model->params.full_attention_interval == 5);
  CHECK(model->params.final_logit_softcapping == doctest::Approx(30.0f));
  CHECK(model->params.rope_freq_base == doctest::Approx(1000000.0f));
  CHECK(model->params.rope_freq_base_swa == doctest::Approx(10000.0f));
  CHECK(model->params.tie_word_embeddings);
  CHECK(model->params.attention_sliding_window_pattern_count == 35u);
  CHECK(model->params.attention_sliding_window_pattern_flags[4] == 0u);
  CHECK(model->params.attention_sliding_window_pattern_flags[5] == 1u);
  CHECK(model->vocab_data.n_tokens == 2u);
}

TEST_CASE(
    "model_detail_rejects_unknown_hparams_architecture_from_gguf_binding") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  auto model = std::make_unique<emel::model::data>();

  append_kv_string(arena, entries, "general.architecture", "unknown-arch");

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  CHECK_FALSE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  CHECK_FALSE(emel::model::is_supported_execution_architecture("unknown-arch"));
  CHECK_FALSE(emel::model::is_lfm2_execution_architecture("unknown-arch"));
  CHECK_FALSE(emel::model::is_gemma4_execution_architecture("unknown-arch"));
}

TEST_CASE("model_detail_loads_llama_hparams_from_gguf_binding") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  auto model = std::make_unique<emel::model::data>();
  const std::array<std::string_view, 3> tokens = {"<s>", "hello", "</s>"};

  append_kv_string(arena, entries, "general.architecture", "llama");
  append_kv_u32(arena, entries, "llama.context_length", 4096u);
  append_kv_u32(arena, entries, "llama.embedding_length", 256u);
  append_kv_u32(arena, entries, "llama.embedding_length_out", 320u);
  append_kv_u32(arena, entries, "llama.feed_forward_length", 768u);
  append_kv_u32(arena, entries, "llama.attention.head_count", 8u);
  append_kv_u32(arena, entries, "llama.attention.head_count_kv", 2u);
  append_kv_u32(arena, entries, "llama.rope.dimension_count", 32u);
  append_kv_u32(arena, entries, "llama.block_count", 12u);
  append_kv_u32(arena, entries, "llama.vocab_size", 32000u);
  append_kv_f32(arena, entries, "llama.attention.layer_norm_epsilon", 1e-5f);
  append_kv_f64(arena, entries, "llama.attention.layer_norm_rms_epsilon", 1e-6);
  append_kv_f32(arena, entries, "llama.attention.clamp_kqv", 4.0f);
  append_kv_f32(arena, entries, "llama.attn_logit_softcapping", 12.0f);
  append_kv_f32(arena, entries, "llama.final_logit_softcapping", 8.0f);
  append_kv_f32(arena, entries, "llama.residual_scale", 0.5f);
  append_kv_f32(arena, entries, "llama.embedding_scale", 1.5f);
  append_kv_f32(arena, entries, "llama.rope.freq_base", 10000.0f);
  append_kv_f64(arena, entries, "llama.rope.freq_base_swa", 500000.0);
  append_kv_string_array(arena, entries, "tokenizer.ggml.tokens",
                         std::span<const std::string_view>{tokens});

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  CHECK(emel::model::architecture_name_view(*model) == "llama");
  CHECK(emel::model::is_supported_execution_architecture("llama"));
  CHECK_FALSE(emel::model::is_lfm2_execution_architecture("llama"));
  CHECK_FALSE(emel::model::is_gemma4_execution_architecture("llama"));
  CHECK(model->params.n_ctx == 4096);
  CHECK(model->params.n_embd == 256);
  CHECK(model->params.n_embd_out == 320);
  CHECK(model->params.n_ff == 768);
  CHECK(model->params.n_head == 8);
  CHECK(model->params.n_head_kv == 2);
  CHECK(model->params.n_rot == 32);
  CHECK(model->params.n_layer == 12);
  CHECK(model->params.n_vocab == 32000);
  CHECK(model->params.attention_layer_norm_epsilon == doctest::Approx(1e-5f));
  CHECK(model->params.attention_layer_norm_rms_epsilon ==
        doctest::Approx(1e-6f));
  CHECK(model->params.attention_clamp_kqv == doctest::Approx(4.0f));
  CHECK(model->params.attn_logit_softcapping == doctest::Approx(12.0f));
  CHECK(model->params.final_logit_softcapping == doctest::Approx(8.0f));
  CHECK(model->params.residual_scale == doctest::Approx(0.5f));
  CHECK(model->params.embedding_scale == doctest::Approx(1.5f));
  CHECK(model->params.rope_freq_base == doctest::Approx(10000.0f));
  CHECK(model->params.rope_freq_base_swa == doctest::Approx(500000.0f));
  CHECK(model->vocab_data.n_tokens == 3u);
}

TEST_CASE(
    "model_detail_loads_qwen3_hparams_from_gguf_binding_with_vocab_fallback") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  auto model = std::make_unique<emel::model::data>();
  const std::array<std::string_view, 4> tokens = {"<|bos|>", "A", "B",
                                                  "<|eos|>"};

  append_kv_string(arena, entries, "general.architecture", "qwen3");
  append_kv_u32(arena, entries, "qwen3.context_length", 32768u);
  append_kv_u32(arena, entries, "qwen3.embedding_length", 1024u);
  append_kv_u32(arena, entries, "qwen3.feed_forward_length", 4096u);
  append_kv_u32(arena, entries, "qwen3.attention.head_count", 16u);
  append_kv_u32(arena, entries, "qwen3.attention.head_count_kv", 2u);
  append_kv_u32(arena, entries, "qwen3.attention.key_length", 64u);
  append_kv_u32(arena, entries, "qwen3.attention.value_length", 80u);
  append_kv_u32(arena, entries, "qwen3.block_count", 24u);
  append_kv_f32(arena, entries, "qwen3.attention.layer_norm_rms_epsilon",
                1e-6f);
  append_kv_f32(arena, entries, "qwen3.rope.freq_base", 1000000.0f);
  append_kv_string_array(arena, entries, "tokenizer.ggml.tokens",
                         std::span<const std::string_view>{tokens});

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  CHECK(emel::model::architecture_name_view(*model) == "qwen3");
  CHECK(emel::model::is_supported_execution_architecture("qwen3"));
  CHECK(model->params.n_ctx == 32768);
  CHECK(model->params.n_embd == 1024);
  CHECK(model->params.n_embd_out == 1024);
  CHECK(model->params.n_ff == 4096);
  CHECK(model->params.n_head == 16);
  CHECK(model->params.n_head_kv == 2);
  CHECK(model->params.attention_key_length == 64);
  CHECK(model->params.attention_value_length == 80);
  CHECK(model->params.n_rot == 64);
  CHECK(model->params.n_layer == 24);
  CHECK(model->params.n_vocab == 4);
  CHECK(model->params.tie_word_embeddings);
  CHECK(model->params.attention_layer_norm_rms_epsilon ==
        doctest::Approx(1e-6f));
  CHECK(model->params.rope_freq_base == doctest::Approx(1000000.0f));
  CHECK(model->vocab_data.n_tokens == 4u);
}

TEST_CASE("model_detail_loads_lfm2_hparams_from_gguf_binding") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  auto model = std::make_unique<emel::model::data>();
  const std::array<uint32_t, 2> head_count_kv = {0u, 8u};

  append_kv_string(arena, entries, "general.architecture", "lfm2");
  append_kv_u32(arena, entries, "lfm2.context_length", 128000u);
  append_kv_u32(arena, entries, "lfm2.embedding_length", 2048u);
  append_kv_u32(arena, entries, "lfm2.feed_forward_length", 8192u);
  append_kv_u32(arena, entries, "lfm2.attention.head_count", 32u);
  append_kv_u32(arena, entries, "lfm2.block_count", 16u);
  append_kv_u32(arena, entries, "lfm2.vocab_size", 65536u);
  append_kv_u32(arena, entries, "lfm2.shortconv.l_cache", 3u);
  append_kv_f32(arena, entries, "lfm2.attention.layer_norm_rms_epsilon", 1e-6f);
  append_kv_f32(arena, entries, "lfm2.rope.freq_base", 1000000.0f);
  append_kv_u32_array(arena, entries, "lfm2.attention.head_count_kv",
                      std::span<const uint32_t>{head_count_kv});

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  CHECK(emel::model::architecture_name_view(*model) == "lfm2");
  CHECK(emel::model::is_supported_execution_architecture("lfm2"));
  CHECK(emel::model::is_lfm2_execution_architecture("lfm2"));
  CHECK_FALSE(emel::model::is_gemma4_execution_architecture("lfm2"));
  CHECK(model->params.n_ctx == 128000);
  CHECK(model->params.n_embd == 2048);
  CHECK(model->params.n_embd_out == 2048);
  CHECK(model->params.n_ff == 8192);
  CHECK(model->params.n_head == 32);
  CHECK(model->params.n_head_kv == 8);
  CHECK(model->params.n_layer == 16);
  CHECK(model->params.n_vocab == 65536);
  CHECK(model->params.shortconv_l_cache == 3);
  CHECK(model->params.tie_word_embeddings);
  CHECK(model->params.attention_layer_norm_rms_epsilon ==
        doctest::Approx(1e-6f));
  CHECK(model->params.rope_freq_base == doctest::Approx(1000000.0f));
}

TEST_CASE("model_detail_loads_omniembed_hparams_from_gguf_binding") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  auto model = std::make_unique<emel::model::data>();
  const std::array<std::string_view, 3> tokens = {"[PAD]", "hello", "[SEP]"};
  const std::array<uint32_t, 4> matryoshka_dims = {768u, 512u, 256u, 128u};

  append_kv_string(arena, entries, "general.architecture", "omniembed");
  append_kv_u32(arena, entries, "omniembed.embed_dim", 1280u);
  append_kv_string(arena, entries, "omniembed.image_encoder_name",
                   "mobilenetv4_conv_medium.e180_r384_in12k");
  append_kv_u32(arena, entries, "omniembed.image_encoder_dim", 1280u);
  append_kv_string(arena, entries, "omniembed.audio_encoder_name",
                   "efficientat_mn20_as");
  append_kv_u32(arena, entries, "omniembed.audio_encoder_dim", 1920u);
  append_kv_u32_array(arena, entries, "omniembed.matryoshka_dims",
                      std::span<const uint32_t>{matryoshka_dims});
  append_kv_string_array(arena, entries, "tokenizer.tokens",
                         std::span<const std::string_view>{tokens});

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  CHECK(emel::model::architecture_name_view(*model) == "omniembed");
  CHECK(emel::model::is_supported_execution_architecture("omniembed"));
  CHECK(emel::model::is_omniembed_execution_architecture("omniembed"));
  CHECK(model->params.n_embd == 1280);
  CHECK(model->params.n_embd_out == 1280);
  CHECK(model->params.matryoshka_dimension_count == 4u);
  CHECK(model->params.matryoshka_dimensions[0] == 768);
  CHECK(model->params.matryoshka_dimensions[3] == 128);
  CHECK(model->meta.clip_data.has_vision_encoder);
  CHECK(model->meta.clip_data.has_audio_encoder);
  CHECK(emel::model::metadata_string_view(
            model->meta, model->meta.clip_vision_data.encoder_name) ==
        "mobilenetv4_conv_medium.e180_r384_in12k");
  CHECK(model->meta.clip_vision_data.embedding_length == 1280);
  CHECK(model->meta.clip_vision_data.projection_dim == 1280);
  CHECK(model->meta.clip_vision_data.image_size == 384);
  CHECK(model->meta.clip_vision_data.preproc_image_size == 384);
  CHECK(model->meta.clip_vision_data.image_mean_count == 3u);
  CHECK(model->meta.clip_vision_data.image_std_count == 3u);
  CHECK(model->meta.clip_vision_data.image_mean[0] == doctest::Approx(0.485f));
  CHECK(model->meta.clip_vision_data.image_std[2] == doctest::Approx(0.225f));
  CHECK(emel::model::metadata_string_view(
            model->meta, model->meta.clip_audio_data.encoder_name) ==
        "efficientat_mn20_as");
  CHECK(model->meta.clip_audio_data.embedding_length == 1920);
  CHECK(model->meta.clip_audio_data.projection_dim == 1280);
  CHECK(model->meta.clip_audio_data.sample_rate == 32000);
  CHECK(model->meta.clip_audio_data.n_fft == 1024);
  CHECK(model->meta.clip_audio_data.win_length == 800);
  CHECK(model->meta.clip_audio_data.hop_size == 320);
  CHECK(model->meta.clip_audio_data.num_mel_bins == 128);
  CHECK(model->meta.clip_audio_data.high_frequency ==
        doctest::Approx(15000.0f));
  CHECK(model->meta.clip_audio_data.preemphasis_coefficient ==
        doctest::Approx(0.97f));
  CHECK(model->meta.clip_audio_data.log_offset == doctest::Approx(1.0e-5f));
  CHECK(model->meta.clip_audio_data.normalize_bias == doctest::Approx(4.5f));
  CHECK(model->meta.clip_audio_data.normalize_scale == doctest::Approx(5.0f));
  CHECK(model->vocab_data.n_tokens == 3u);
}

TEST_CASE("model_detail_loads_omniembed_hparams_from_i16_matryoshka_binding") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  auto model = std::make_unique<emel::model::data>();
  const std::array<int16_t, 4> matryoshka_dims = {768, 512, 256, 128};

  append_kv_string(arena, entries, "general.architecture", "omniembed");
  append_kv_u32(arena, entries, "omniembed.embed_dim", 1280u);
  append_kv_u32(arena, entries, "omniembed.image_encoder_dim", 640u);
  append_kv_u32(arena, entries, "omniembed.audio_encoder_dim", 768u);
  append_kv_scalar_array(arena, entries, "omniembed.matryoshka_dims",
                         emel::gguf::loader::detail::constants::gguf_type_int16,
                         std::span<const int16_t>{matryoshka_dims});

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  CHECK(model->params.matryoshka_dimension_count == 4u);
  CHECK(model->params.matryoshka_dimensions[0] == 768);
  CHECK(model->params.matryoshka_dimensions[3] == 128);
}

TEST_CASE("model_detail_loads_omniembed_hparams_from_i64_matryoshka_binding") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  auto model = std::make_unique<emel::model::data>();
  const std::array<int64_t, 3> matryoshka_dims = {1024, 512, 256};

  append_kv_string(arena, entries, "general.architecture", "omniembed");
  append_kv_u32(arena, entries, "omniembed.embed_dim", 1280u);
  append_kv_u32(arena, entries, "omniembed.image_encoder_dim", 640u);
  append_kv_u32(arena, entries, "omniembed.audio_encoder_dim", 768u);
  append_kv_scalar_array(arena, entries, "omniembed.matryoshka_dims",
                         emel::gguf::loader::detail::constants::gguf_type_int64,
                         std::span<const int64_t>{matryoshka_dims});

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  CHECK(model->params.matryoshka_dimension_count == 3u);
  CHECK(model->params.matryoshka_dimensions[0] == 1024);
  CHECK(model->params.matryoshka_dimensions[2] == 256);
}

TEST_CASE("model_detail_loads_omniembed_hparams_without_matryoshka_array") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  auto model = std::make_unique<emel::model::data>();

  append_kv_string(arena, entries, "general.architecture", "omniembed");
  append_kv_u32(arena, entries, "omniembed.embed_dim", 1280u);
  append_kv_u32(arena, entries, "omniembed.image_encoder_dim", 640u);
  append_kv_u32(arena, entries, "omniembed.audio_encoder_dim", 768u);

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  CHECK(model->params.matryoshka_dimension_count == 0u);
}

TEST_CASE("model_omniembed_detail_builds_multimodal_execution_contract") {
  auto model = std::make_unique<emel::model::data>();
  build_omniembed_model(*model, true);

  emel::model::omniembed::detail::execution_contract contract = {};
  REQUIRE(emel::model::omniembed::detail::build_execution_contract(*model,
                                                                   contract) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(contract.embedding_length == 1280);
  CHECK(contract.image_encoder_length == 640);
  CHECK(contract.audio_encoder_length == 768);
  CHECK(contract.matryoshka_dimension_count == 4u);
  CHECK(contract.matryoshka_dimensions[1] == 512);
  CHECK(contract.text_encoder.tensor_count == 1u);
  CHECK(contract.text_projection.tensor_count == 1u);
  CHECK(contract.image_encoder.tensor_count == 1u);
  CHECK(contract.image_projection.tensor_count == 1u);
  CHECK(contract.audio_encoder.tensor_count == 1u);
  CHECK(contract.audio_projection.tensor_count == 1u);
  CHECK(contract.text_encoder.first.name == "text_encoder.backbone.weight");
  CHECK(contract.audio_projection.first.name ==
        "audio_projection.project.weight");
  CHECK(emel::model::omniembed::detail::validate_data(*model) ==
        emel::error::cast(emel::model::loader::error::none));
  CHECK(emel::model::omniembed::detail::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::none));
  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::none));
}

TEST_CASE("model_execution_contract_rejects_omniembed_without_audio_projection_"
          "family") {
  auto model = std::make_unique<emel::model::data>();
  build_omniembed_model(*model, false);

  emel::model::omniembed::detail::execution_contract contract = {};
  CHECK(emel::model::omniembed::detail::build_execution_contract(*model,
                                                                 contract) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("model_execution_contract_rejects_omniembed_with_invalid_matryoshka_"
          "shape") {
  auto model = std::make_unique<emel::model::data>();
  build_omniembed_model(*model, true);
  model->params.matryoshka_dimensions[1] = 1536;

  CHECK(emel::model::omniembed::detail::validate_data(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
  CHECK(emel::model::omniembed::detail::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("model_omniembed_detail_ignores_unusable_family_tensor_storage") {
  auto model = std::make_unique<emel::model::data>();
  build_omniembed_model(*model, true);

  auto &unusable = model->tensors[model->n_tensors];
  append_tensor_name(*model, unusable, "text_encoder.unusable.weight");
  unusable.n_dims = 1;
  unusable.dims[0] = 0;
  unusable.data = &unusable;
  unusable.data_size = 8u;
  ++model->n_tensors;

  emel::model::omniembed::detail::execution_contract contract = {};
  REQUIRE(emel::model::omniembed::detail::build_execution_contract(*model,
                                                                   contract) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(contract.text_encoder.tensor_count == 1u);
}

TEST_CASE("model_detail_loads_sortformer_hparams_from_gguf_binding") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  auto model = std::make_unique<emel::model::data>();

  append_kv_string(arena, entries, "general.architecture", "sortformer");
  append_kv_string(arena, entries, "sortformer.source.format", "nemo");
  append_kv_string(arena, entries, "sortformer.tensor_name_scheme",
                   "compact_v1");
  append_kv_string(arena, entries, "sortformer.outtype", "f32");
  append_kv_u32(arena, entries, "sortformer.original_tensor_count", 128u);
  append_kv_u32(arena, entries, "sortformer.tensor_count", 132u);
  append_kv_u32(arena, entries, "sortformer.skipped_tensor_count", 3u);
  append_kv_u32(arena, entries, "sortformer.config.preprocessor.sample_rate",
                16000u);
  append_kv_u32(arena, entries, "sortformer.config.sortformer_modules.num_spks",
                4u);
  append_kv_u32(arena, entries,
                "sortformer.config.sortformer_modules.chunk_len", 188u);
  append_kv_u32(arena, entries,
                "sortformer.config.sortformer_modules.chunk_right_context", 1u);
  append_kv_u32(arena, entries, "sortformer.config.sortformer_modules.fifo_len",
                0u);
  append_kv_u32(arena, entries,
                "sortformer.config.sortformer_modules.spkcache_update_period",
                188u);
  append_kv_u32(arena, entries,
                "sortformer.config.sortformer_modules.spkcache_len", 188u);

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  CHECK(emel::model::architecture_name_view(*model) == "sortformer");
  CHECK(emel::model::is_supported_execution_architecture("sortformer"));
  CHECK(
      emel::model::sortformer::detail::is_execution_architecture("sortformer"));
  CHECK_FALSE(emel::model::is_omniembed_execution_architecture("sortformer"));
  CHECK(model->params.n_features == 4);
}

TEST_CASE(
    "model_detail_rejects_sortformer_hparams_with_wrong_source_contract") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  auto model = std::make_unique<emel::model::data>();

  append_kv_string(arena, entries, "general.architecture", "sortformer");
  append_kv_string(arena, entries, "sortformer.source.format", "onnx");
  append_kv_string(arena, entries, "sortformer.tensor_name_scheme",
                   "compact_v1");
  append_kv_string(arena, entries, "sortformer.outtype", "f32");
  append_kv_u32(arena, entries, "sortformer.original_tensor_count", 128u);
  append_kv_u32(arena, entries, "sortformer.tensor_count", 132u);

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  CHECK_FALSE(emel::model::detail::load_hparams_from_gguf(binding, *model));
}

TEST_CASE("model_detail_loads_whisper_tiny_hparams_from_gguf_binding") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  auto model = std::make_unique<emel::model::data>();

  append_kv_string(arena, entries, "general.architecture", "whisper");
  append_kv_u32(arena, entries, "whisper.n_mels", 80u);
  append_kv_u32(arena, entries, "whisper.n_vocab", 51865u);

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  CHECK(emel::model::architecture_name_view(*model) == "whisper");
  CHECK(emel::model::is_supported_execution_architecture("whisper"));
  CHECK(emel::model::is_whisper_execution_architecture("whisper"));
  CHECK_FALSE(emel::model::is_omniembed_execution_architecture("whisper"));
  CHECK(model->params.n_features == 80);
  CHECK(model->params.n_vocab == 51865);
  CHECK(model->params.n_embd == 384);
  CHECK(model->params.n_embd_out == 384);
  CHECK(model->params.n_ff == 1536);
  CHECK(model->params.n_head == 6);
  CHECK(model->params.n_head_kv == 6);
  CHECK(model->params.n_ctx == 448);
  CHECK(model->params.n_layer == 4);
  CHECK(model->params.decoder_block_count == 4);
}

TEST_CASE(
    "model_detail_rejects_whisper_hparams_with_noncanonical_mel_contract") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};
  auto model = std::make_unique<emel::model::data>();

  append_kv_string(arena, entries, "general.architecture", "whisper");
  append_kv_u32(arena, entries, "whisper.n_mels", 128u);
  append_kv_u32(arena, entries, "whisper.n_vocab", 51865u);

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  CHECK_FALSE(emel::model::detail::load_hparams_from_gguf(binding, *model));
}

TEST_CASE("model_sortformer_detail_builds_execution_contract") {
  auto model = std::make_unique<emel::model::data>();
  build_sortformer_model(*model, true);

  emel::model::sortformer::detail::execution_contract contract = {};
  REQUIRE(emel::model::sortformer::detail::build_execution_contract(*model,
                                                                    contract) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(contract.sample_rate == 16000);
  CHECK(contract.speaker_count == 4);
  CHECK(contract.frame_shift_ms == 80);
  CHECK(contract.chunk_len == 188);
  CHECK(contract.chunk_right_context == 1);
  CHECK(contract.fifo_len == 0);
  CHECK(contract.spkcache_update_period == 188);
  CHECK(contract.spkcache_len == 188);
  CHECK(contract.feature_extractor.first.name == "prep.feat.fb");
  CHECK(contract.encoder.first.name == "enc.l0.conv.dw.w");
  CHECK(contract.modules.first.name == "mods.ep.w");
  CHECK(contract.transformer_encoder.first.name == "te.l0.sa.q.w");
  CHECK(emel::model::sortformer::detail::validate_data(*model) ==
        emel::error::cast(emel::model::loader::error::none));
  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::none));
}

TEST_CASE("model_sortformer_detail_rejects_missing_modules_family") {
  auto model = std::make_unique<emel::model::data>();
  build_sortformer_model(*model, false);

  emel::model::sortformer::detail::execution_contract contract = {};
  CHECK(emel::model::sortformer::detail::build_execution_contract(*model,
                                                                  contract) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("model_sortformer_detail_rejects_noncanonical_stream_contract") {
  auto model = std::make_unique<emel::model::data>();
  build_sortformer_model(*model, true);
  model->params.n_features = 3;

  CHECK(emel::model::sortformer::detail::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("model_whisper_detail_builds_tiny_gguf_execution_contract") {
  auto model = std::make_unique<emel::model::data>();
  build_whisper_model(*model, true);

  emel::model::whisper::detail::execution_contract contract = {};
  REQUIRE(emel::model::whisper::detail::build_execution_contract(*model,
                                                                 contract) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(contract.sample_rate == 16000);
  CHECK(contract.mel_bin_count == 80);
  CHECK(contract.vocab_size == 51865);
  CHECK(contract.embedding_length == 384);
  CHECK(contract.feed_forward_length == 1536);
  CHECK(contract.attention_head_count == 6);
  CHECK(contract.encoder_context_length == 1500);
  CHECK(contract.decoder_context_length == 448);
  CHECK(contract.encoder_block_count == 4);
  CHECK(contract.decoder_block_count == 4);
  CHECK(contract.mel_filters.first.name == "mel_filters");
  CHECK(contract.encoder.first.name == "model.encoder.conv1.weight");
  CHECK(contract.decoder.first.name == "model.decoder.embed_tokens.weight");
  CHECK(emel::model::whisper::detail::validate_data(*model) ==
        emel::error::cast(emel::model::loader::error::none));
  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::none));
}

TEST_CASE("model_whisper_detail_rejects_missing_decoder_cross_attention") {
  auto model = std::make_unique<emel::model::data>();
  build_whisper_model(*model, false);

  emel::model::whisper::detail::execution_contract contract = {};
  CHECK(emel::model::whisper::detail::build_execution_contract(*model,
                                                               contract) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("model_whisper_detail_rejects_noncanonical_vocab_contract") {
  auto model = std::make_unique<emel::model::data>();
  build_whisper_model(*model, true);
  model->params.n_vocab = 50257;

  CHECK(emel::model::whisper::detail::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE(
    "model_whisper_detail_builds_execution_contract_from_pinned_real_fixture") {
  const auto fixture_path = whisper_fixture_path();
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE(
        "skipping real Whisper fixture parse test because fixture is missing: "
        << fixture_path.string());
    return;
  }

  auto model = std::make_unique<emel::model::data>();
  const std::vector<uint8_t> file_bytes = read_binary_file(fixture_path);
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  REQUIRE(
      load_whisper_fixture_binding(file_bytes, kv_arena, kv_entries, *model));

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{kv_arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{kv_entries},
  };
  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  model->weights_data = file_bytes.data();
  model->weights_size = file_bytes.size();
  materialize_tensor_names_from_file(*model, file_bytes);

  emel::model::whisper::detail::execution_contract contract = {};
  REQUIRE(emel::model::whisper::detail::build_execution_contract(*model,
                                                                 contract) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(emel::model::architecture_name_view(*model) == "whisper");
  CHECK(model->params.n_features == 80);
  CHECK(model->params.n_vocab == 51865);
  CHECK(contract.mel_filters.tensor_count > 0u);
  CHECK(contract.encoder.tensor_count > 0u);
  CHECK(contract.decoder.tensor_count > 0u);
  CHECK(contract.mel_filters.first.name == "mel_filters");
  CHECK(contract.encoder.first.name.starts_with("model.encoder."));
  CHECK(contract.decoder.first.name.starts_with("model.decoder."));
  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::none));
}

TEST_CASE(
    "model_whisper_fixture_carries_no_embedded_tokenizer_metadata_in_gguf") {
  // The Candle-style oxide-lab/whisper-tiny-GGUF fixtures intentionally do
  // not embed tokenizer metadata; the maintained Whisper tokenizer is shipped
  // as an external `tokenizer-tiny.json` sibling in the same upstream repo
  // (mirroring how the maintained TE slice ships `mdbr-leaf-ir-vocab.txt`
  // alongside its GGUF). This test source-backs that contract truth so a
  // future ASR runtime phase wires the external tokenizer asset explicitly
  // instead of silently falling back to absent GGUF tokenizer keys.
  const auto fixture_path = whisper_fixture_path();
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping real Whisper tokenizer-absence test because fixture is "
            "missing: "
            << fixture_path.string());
    return;
  }

  auto model = std::make_unique<emel::model::data>();
  const std::vector<uint8_t> file_bytes = read_binary_file(fixture_path);
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  REQUIRE(
      load_whisper_fixture_binding(file_bytes, kv_arena, kv_entries, *model));

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{kv_arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{kv_entries},
  };

  // No tokenizer model identity should be present in the GGUF.
  CHECK(emel::model::detail::find_kv_entry_any(
            binding, {"tokenizer.model", "tokenizer.ggml.model"}) == nullptr);

  // No tokenizer token array should be present in the GGUF.
  CHECK(emel::model::detail::find_kv_entry_any(
            binding, {"tokenizer.tokens", "tokenizer.ggml.tokens"}) == nullptr);

  // load_vocab_from_gguf must therefore not synthesize a vocab from the
  // Whisper GGUF: it should leave the model in the no-tokenizer state.
  const bool loaded =
      emel::model::detail::load_vocab_from_gguf(binding, model->vocab_data);
  CHECK_FALSE(loaded);
  CHECK(model->vocab_data.tokenizer_model_id ==
        emel::model::data::tokenizer_model::UNKNOWN);
}

TEST_CASE("model_whisper_detail_builds_execution_contract_from_q4_0_fixture") {
  const auto fixture_path = whisper_q4_0_fixture_path();
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping real Whisper q4_0 fixture parse test because fixture is "
            "missing: "
            << fixture_path.string());
    return;
  }

  auto model = std::make_unique<emel::model::data>();
  const std::vector<uint8_t> file_bytes = read_binary_file(fixture_path);
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  REQUIRE(
      load_whisper_fixture_binding(file_bytes, kv_arena, kv_entries, *model));

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{kv_arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{kv_entries},
  };
  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  model->weights_data = file_bytes.data();
  model->weights_size = file_bytes.size();
  materialize_tensor_names_from_file(*model, file_bytes);

  emel::model::whisper::detail::execution_contract contract = {};
  REQUIRE(emel::model::whisper::detail::build_execution_contract(*model,
                                                                 contract) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(emel::model::architecture_name_view(*model) == "whisper");
  CHECK(model->params.n_features == 80);
  CHECK(model->params.n_vocab == 51865);
  CHECK(contract.mel_filters.tensor_count > 0u);
  CHECK(contract.encoder.tensor_count > 0u);
  CHECK(contract.decoder.tensor_count > 0u);
  CHECK(contract.mel_filters.first.name == "mel_filters");
  CHECK(contract.encoder.first.name.starts_with("model.encoder."));
  CHECK(contract.decoder.first.name.starts_with("model.decoder."));
  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::none));
}

TEST_CASE("model_whisper_detail_builds_execution_contract_from_q4_1_fixture") {
  const auto fixture_path = whisper_q4_1_fixture_path();
  if (!std::filesystem::exists(fixture_path)) {
    MESSAGE("skipping real Whisper q4_1 fixture parse test because fixture is "
            "missing: "
            << fixture_path.string());
    return;
  }

  auto model = std::make_unique<emel::model::data>();
  const std::vector<uint8_t> file_bytes = read_binary_file(fixture_path);
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  REQUIRE(
      load_whisper_fixture_binding(file_bytes, kv_arena, kv_entries, *model));

  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{kv_arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{kv_entries},
  };
  REQUIRE(emel::model::detail::load_hparams_from_gguf(binding, *model));
  model->weights_data = file_bytes.data();
  model->weights_size = file_bytes.size();
  materialize_tensor_names_from_file(*model, file_bytes);

  emel::model::whisper::detail::execution_contract contract = {};
  REQUIRE(emel::model::whisper::detail::build_execution_contract(*model,
                                                                 contract) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(emel::model::architecture_name_view(*model) == "whisper");
  CHECK(model->params.n_features == 80);
  CHECK(model->params.n_vocab == 51865);
  CHECK(contract.mel_filters.tensor_count > 0u);
  CHECK(contract.encoder.tensor_count > 0u);
  CHECK(contract.decoder.tensor_count > 0u);
  CHECK(contract.mel_filters.first.name == "mel_filters");
  CHECK(contract.encoder.first.name.starts_with("model.encoder."));
  CHECK(contract.decoder.first.name.starts_with("model.decoder."));
  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::none));
}

TEST_CASE(
    "model_whisper_loader_rejects_whisper_cpp_lmgg_sibling_magic_synthetic") {
  // Build a minimal in-memory file image whose first four bytes are the
  // whisper.cpp `lmgg` binary marker rather than the GGUF `GGUF` magic.
  // The EMEL-owned GGUF loader probe must reject this without producing a
  // valid requirements descriptor, so it cannot bootstrap EMEL Whisper state
  // from a whisper.cpp/ sibling artifact.
  std::vector<uint8_t> file_bytes(64u, 0u);
  file_bytes[0] = 'l';
  file_bytes[1] = 'm';
  file_bytes[2] = 'g';
  file_bytes[3] = 'g';

  emel::gguf::loader::requirements requirements = {};
  const emel::error::type err = emel::gguf::loader::detail::probe_requirements(
      std::span<const uint8_t>{file_bytes}, requirements);
  CHECK(err == emel::gguf::loader::detail::cast_loader_error(
                   emel::gguf::loader::error::model_invalid));
  CHECK(requirements.tensor_count == 0u);
  CHECK(requirements.kv_count == 0u);
}

TEST_CASE(
    "model_execution_contract_accepts_canonical_gemma4_shared_kv_contract") {
  auto model = std::make_unique<emel::model::data>();
  build_gemma4_model(*model, false);

  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::none));
}

TEST_CASE("model_llama_detail_builds_gemma4_execution_view_with_shared_kv_and_"
          "tied_output_fallback") {
  auto model = std::make_unique<emel::model::data>();
  build_gemma4_model(*model, false);

  for (uint32_t idx = 0; idx < model->n_tensors; ++idx) {
    auto &tensor = model->tensors[idx];
    if (emel::model::tensor_name_view(*model, tensor) ==
        "blk.15.attn_v.weight") {
      tensor.data = nullptr;
      tensor.data_size = 0u;
      break;
    }
  }

  emel::model::llama::detail::execution_view view = {};
  REQUIRE(emel::model::llama::detail::build_execution_view(*model, view) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(view.output.name == "token_embd.weight");

  emel::model::llama::detail::block_view dedicated = {};
  REQUIRE(emel::model::llama::detail::lookup_block_view(view, 0, dedicated) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(dedicated.attention_v.name == "blk.0.attn_v.weight");

  emel::model::llama::detail::block_view shared = {};
  REQUIRE(emel::model::llama::detail::lookup_block_view(view, 15, shared) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(shared.attention_v.name == "blk.15.attn_k.weight");
}

TEST_CASE("model_llama_detail_builds_gemma4_topology_with_shared_kv_tail_"
          "tensor_count") {
  auto model = std::make_unique<emel::model::data>();
  build_gemma4_model(*model, false);

  emel::model::llama::detail::execution_view view = {};
  REQUIRE(emel::model::llama::detail::build_execution_view(*model, view) ==
          emel::error::cast(emel::model::loader::error::none));

  emel::model::llama::detail::topology topology = {};
  REQUIRE(emel::model::llama::detail::build_topology(view, topology) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(topology.tensor_count == 333u);
  CHECK(topology.node_count == 333u);
}

TEST_CASE("model_llama_detail_builds_gemma4_quantized_path_audit_with_shared_"
          "kv_fallback") {
  auto model = std::make_unique<emel::model::data>();
  build_gemma4_model(*model, false);

  for (uint32_t idx = 0; idx < model->n_tensors; ++idx) {
    auto &tensor = model->tensors[idx];
    const std::string_view name = emel::model::tensor_name_view(*model, tensor);
    if (name == "token_embd.weight") {
      tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::q2_k);
      continue;
    }
    if (name.ends_with("attn_k.weight") || name.ends_with("attn_v.weight")) {
      tensor.type = static_cast<int32_t>(emel::kernel::event::dtype::q6_k);
    }
  }

  emel::model::llama::detail::execution_view view = {};
  REQUIRE(emel::model::llama::detail::build_execution_view(*model, view) ==
          emel::error::cast(emel::model::loader::error::none));

  const auto audit =
      emel::model::llama::detail::build_quantized_path_audit(view);
  const auto find_stage =
      [&](const emel::model::llama::detail::quantized_stage_family family)
      -> const emel::model::llama::detail::quantized_stage_audit & {
    for (const auto &stage : audit.stages) {
      if (stage.family == family) {
        return stage;
      }
    }
    FAIL("missing quantized stage audit");
    return audit.stages[0];
  };

  const auto &token_embedding = find_stage(
      emel::model::llama::detail::quantized_stage_family::token_embedding);
  const auto &output =
      find_stage(emel::model::llama::detail::quantized_stage_family::output);
  const auto &attention_v = find_stage(
      emel::model::llama::detail::quantized_stage_family::attention_v);
  const auto &attention_q_norm = find_stage(
      emel::model::llama::detail::quantized_stage_family::attention_q_norm);
  const auto &attention_k_norm = find_stage(
      emel::model::llama::detail::quantized_stage_family::attention_k_norm);

  CHECK(token_embedding.contract ==
        emel::model::llama::detail::quantized_contract_kind::
            approved_dense_f32_by_contract);
  CHECK(output.tensor_type ==
        static_cast<int32_t>(emel::kernel::event::dtype::q2_k));
  CHECK(output.contract ==
        emel::model::llama::detail::quantized_contract_kind::native_quantized);
  CHECK(attention_v.tensor_type ==
        static_cast<int32_t>(emel::kernel::event::dtype::q6_k));
  CHECK(attention_v.contract ==
        emel::model::llama::detail::quantized_contract_kind::native_quantized);
  CHECK(attention_v.consistent_across_layers);
  CHECK(attention_q_norm.contract ==
        emel::model::llama::detail::quantized_contract_kind::
            approved_dense_f32_by_contract);
  CHECK(attention_k_norm.contract ==
        emel::model::llama::detail::quantized_contract_kind::
            approved_dense_f32_by_contract);
}

TEST_CASE("model_llama_detail_quantized_audit_name_helpers_publish_supported_"
          "labels") {
  using quantized_contract_kind =
      emel::model::llama::detail::quantized_contract_kind;
  using quantized_stage_family =
      emel::model::llama::detail::quantized_stage_family;

  CHECK(emel::model::llama::detail::quantized_stage_family_name(
            quantized_stage_family::attention_v) == "attention_v");
  CHECK(emel::model::llama::detail::quantized_contract_kind_name(
            quantized_contract_kind::not_applicable) == "not_applicable");
  CHECK(emel::model::llama::detail::tensor_type_name(
            static_cast<int32_t>(emel::kernel::event::dtype::q4_k)) == "q4_k");
  CHECK(emel::model::llama::detail::tensor_type_name(
            emel::kernel::detail::dtype_q4_0) == "q4_0");
  CHECK(emel::model::llama::detail::tensor_type_name(-7) == "unknown");
}

TEST_CASE("model_execution_contract_rejects_gemma4_without_canonical_sliding_"
          "window_pattern") {
  auto model = std::make_unique<emel::model::data>();
  build_gemma4_model(*model, false);
  model->params.attention_sliding_window_pattern_flags[4] = 1u;

  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("model_execution_contract_rejects_gemma4_without_required_dedicated_"
          "v_projection") {
  auto model = std::make_unique<emel::model::data>();
  build_gemma4_model(*model, false);

  for (uint32_t idx = 0; idx < model->n_tensors; ++idx) {
    auto &tensor = model->tensors[idx];
    if (emel::model::tensor_name_view(*model, tensor) ==
        "blk.0.attn_v.weight") {
      tensor.data = nullptr;
      tensor.data_size = 0u;
      break;
    }
  }

  CHECK(emel::model::validate_execution_contract(*model) ==
        emel::error::cast(emel::model::loader::error::model_invalid));
}

TEST_CASE("model_detail_loads_gguf_vocab_metadata_without_llama_bootstrap") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};

  const std::array<std::string_view, 2> tokens = {"<|pad|>", "hello"};
  const std::array<std::string_view, 1> merges = {"h ello"};
  const std::array<uint32_t, 2> token_types = {3u, 1u};
  const std::array<float, 2> scores = {0.0f, 1.5f};

  append_kv_string(arena, entries, "tokenizer.model", "gpt2");
  append_kv_string(arena, entries, "tokenizer.pre", "lfm2");
  append_kv_string_array(arena, entries, "tokenizer.tokens",
                         std::span<const std::string_view>{tokens});
  append_kv_u32_array(arena, entries, "tokenizer.token_type",
                      std::span<const uint32_t>{token_types});
  append_kv_u32(arena, entries, "tokenizer.token_type_count", 4u);
  append_kv_f32_array(arena, entries, "tokenizer.scores",
                      std::span<const float>{scores});
  append_kv_string_array(arena, entries, "tokenizer.merges",
                         std::span<const std::string_view>{merges});
  append_kv_u32(arena, entries, "tokenizer.bos_token_id", 0u);
  append_kv_u32(arena, entries, "tokenizer.eos_token_id", 0u);
  append_kv_u32(arena, entries, "tokenizer.padding_token_id", 0u);
  append_kv_bool(arena, entries, "tokenizer.add_eos_token", false);

  auto vocab = std::make_unique<emel::model::data::vocab>();
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  CHECK(emel::model::detail::load_vocab_from_gguf(binding, *vocab));
  CHECK(vocab->tokenizer_model_id == emel::model::data::tokenizer_model::BPE);
  CHECK(vocab->tokenizer_pre_id == emel::model::data::tokenizer_pre::LLAMA3);
  CHECK(std::string_view{vocab->tokenizer_model_name.data()} == "gpt2");
  CHECK(std::string_view{vocab->tokenizer_pre_name.data()} == "lfm2");
  CHECK(vocab->n_tokens == 2u);
  CHECK(vocab->n_token_types == 4u);
  CHECK(vocab_piece(*vocab, 0u) == "<|pad|>");
  CHECK(vocab_piece(*vocab, 1u) == "hello");
  CHECK(vocab->entries[0].type == 3);
  CHECK(vocab->entries[1].type == 1);
  CHECK(vocab->entries[1].score == doctest::Approx(1.5f));
  CHECK(vocab->n_merges == 1u);
  CHECK(merge_piece(*vocab, 0u) == "h ello");
  CHECK(vocab->bos_id == 0);
  CHECK(vocab->eos_id == 0);
  CHECK(vocab->pad_id == 0);
  CHECK(vocab->add_bos);
  CHECK_FALSE(vocab->add_eos);
  CHECK(vocab->ignore_merges);
}

TEST_CASE("model_detail_preserves_vocab_default_flags_when_gguf_omits_optional_"
          "t5_fields") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};

  const std::array<std::string_view, 4> tokens = {"<pad>", "</s>", "<unk>",
                                                  "\xE2\x96\x81"};
  const std::array<uint32_t, 4> token_types = {3u, 3u, 2u, 1u};

  append_kv_string(arena, entries, "tokenizer.ggml.model", "t5");
  append_kv_string(arena, entries, "tokenizer.ggml.pre", "default");
  append_kv_string_array(arena, entries, "tokenizer.ggml.tokens",
                         std::span<const std::string_view>{tokens});
  append_kv_u32_array(arena, entries, "tokenizer.ggml.token_type",
                      std::span<const uint32_t>{token_types});
  append_kv_bool(arena, entries, "tokenizer.ggml.add_space_prefix", true);
  append_kv_bool(arena, entries, "tokenizer.ggml.remove_extra_whitespaces",
                 true);

  auto vocab = std::make_unique<emel::model::data::vocab>();
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_vocab_from_gguf(binding, *vocab));
  CHECK(vocab->tokenizer_model_id == emel::model::data::tokenizer_model::UGM);
  CHECK(vocab->pad_id == 0);
  CHECK(vocab->eos_id == 1);
  CHECK(vocab->unk_id == 2);
  CHECK(vocab->add_space_prefix);
  CHECK(vocab->remove_extra_whitespaces);
  CHECK(vocab->escape_whitespaces);
}

TEST_CASE("model_detail_preserves_negative_vocab_sentinels_when_gguf_omits_"
          "optional_ids") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};

  const std::array<std::string_view, 2> tokens = {"<s>", "hello"};
  const std::array<uint32_t, 2> token_types = {3u, 1u};

  append_kv_string(arena, entries, "tokenizer.ggml.model", "rwkv");
  append_kv_string_array(arena, entries, "tokenizer.ggml.tokens",
                         std::span<const std::string_view>{tokens});
  append_kv_u32_array(arena, entries, "tokenizer.ggml.token_type",
                      std::span<const uint32_t>{token_types});
  append_kv_u32(arena, entries, "tokenizer.ggml.bos_token_id", 0u);
  append_kv_u32(arena, entries, "tokenizer.ggml.eos_token_id", 0u);

  auto vocab = std::make_unique<emel::model::data::vocab>();
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_vocab_from_gguf(binding, *vocab));
  CHECK(vocab->tokenizer_model_id == emel::model::data::tokenizer_model::RWKV);
  CHECK(vocab->bos_id == 0);
  CHECK(vocab->eos_id == 0);
  CHECK(vocab->unk_id == -1);
  CHECK(vocab->sep_id == -1);
  CHECK(vocab->pad_id == -1);
}

TEST_CASE("model_detail_preserves_negative_vocab_sentinels_when_gguf_uses_"
          "signed_optional_ids") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};

  const std::array<std::string_view, 2> tokens = {"<s>", "hello"};
  const std::array<uint32_t, 2> token_types = {3u, 1u};

  append_kv_string(arena, entries, "tokenizer.ggml.model", "rwkv");
  append_kv_string_array(arena, entries, "tokenizer.ggml.tokens",
                         std::span<const std::string_view>{tokens});
  append_kv_u32_array(arena, entries, "tokenizer.ggml.token_type",
                      std::span<const uint32_t>{token_types});
  append_kv_u32(arena, entries, "tokenizer.ggml.bos_token_id", 0u);
  append_kv_u32(arena, entries, "tokenizer.ggml.eos_token_id", 0u);
  append_kv_i32(arena, entries, "tokenizer.ggml.padding_token_id", -1);
  append_kv_i32(arena, entries, "tokenizer.ggml.prefix_token_id", -1);
  append_kv_i32(arena, entries, "tokenizer.ggml.fim_pre_token_id", -1);

  auto vocab = std::make_unique<emel::model::data::vocab>();
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::load_vocab_from_gguf(binding, *vocab));
  CHECK(vocab->pad_id == -1);
  CHECK(vocab->prefix_id == -1);
  CHECK(vocab->fim_pre_id == -1);
}

TEST_CASE("model_detail_loads_gemma4_vocab_metadata_with_large_merge_table") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};

  const std::array<std::string_view, 2> tokens = {"<bos>", "<eos>"};
  const std::array<uint32_t, 2> token_types = {3u, 3u};
  const std::array<float, 2> scores = {0.0f, 0.0f};
  const std::vector<std::string_view> merges(400001u, std::string_view{});

  append_kv_string(arena, entries, "tokenizer.ggml.model", "gemma4");
  append_kv_string_array(arena, entries, "tokenizer.ggml.tokens",
                         std::span<const std::string_view>{tokens});
  append_kv_u32_array(arena, entries, "tokenizer.ggml.token_type",
                      std::span<const uint32_t>{token_types});
  append_kv_u32(arena, entries, "tokenizer.ggml.token_type_count", 4u);
  append_kv_f32_array(arena, entries, "tokenizer.ggml.scores",
                      std::span<const float>{scores});
  append_kv_string_array(arena, entries, "tokenizer.ggml.merges",
                         std::span<const std::string_view>{merges});
  append_kv_u32(arena, entries, "tokenizer.ggml.bos_token_id", 0u);
  append_kv_u32(arena, entries, "tokenizer.ggml.eos_token_id", 1u);
  append_kv_bool(arena, entries, "tokenizer.ggml.add_bos_token", true);
  append_kv_bool(arena, entries, "tokenizer.ggml.add_space_prefix", true);

  auto vocab = std::make_unique<emel::model::data::vocab>();
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  CHECK(emel::model::detail::load_vocab_from_gguf(binding, *vocab));
  CHECK(vocab->tokenizer_model_id == emel::model::data::tokenizer_model::SPM);
  CHECK(std::string_view{vocab->tokenizer_model_name.data()} == "gemma4");
  CHECK(vocab->n_merges == 400001u);
  CHECK(vocab->bos_id == 0);
  CHECK(vocab->eos_id == 1);
  CHECK(vocab->add_bos);
  CHECK(vocab->add_space_prefix);
}

TEST_CASE(
    "whisper_detail_normalizes_legacy_lmgg_artifact_to_source_owned_gguf") {
  std::vector<uint8_t> legacy = {'l', 'm', 'g', 'g'};
  append_scalar<int32_t>(legacy, 51865);
  append_scalar<int32_t>(legacy, 1500);
  append_scalar<int32_t>(legacy, 384);
  append_scalar<int32_t>(legacy, 6);
  append_scalar<int32_t>(legacy, 4);
  append_scalar<int32_t>(legacy, 448);
  append_scalar<int32_t>(legacy, 384);
  append_scalar<int32_t>(legacy, 6);
  append_scalar<int32_t>(legacy, 4);
  append_scalar<int32_t>(legacy, 80);
  append_scalar<int32_t>(legacy, 7);
  append_scalar<int32_t>(legacy, 80);
  append_scalar<int32_t>(legacy, 1);
  for (int32_t index = 0; index < 80; ++index) {
    append_scalar<uint32_t>(legacy, 0u);
  }
  append_scalar<int32_t>(legacy, 1);
  append_scalar<uint32_t>(legacy, 0u);

  const std::array<std::string_view, 11> root_tensors = {
      "encoder.positional_embedding",
      "encoder.conv1.weight",
      "encoder.conv1.bias",
      "encoder.conv2.weight",
      "encoder.conv2.bias",
      "encoder.ln_post.weight",
      "encoder.ln_post.bias",
      "decoder.positional_embedding",
      "decoder.token_embedding.weight",
      "decoder.ln.weight",
      "decoder.ln.bias",
  };
  for (const auto name : root_tensors) {
    append_legacy_whisper_tensor(legacy, name);
  }

  const std::array<std::string_view, 15> encoder_suffixes = {
      "attn_ln.weight",  "attn_ln.bias",    "attn.query.weight",
      "attn.query.bias", "attn.key.weight", "attn.value.weight",
      "attn.value.bias", "attn.out.weight", "attn.out.bias",
      "mlp_ln.weight",   "mlp_ln.bias",     "mlp.0.weight",
      "mlp.0.bias",      "mlp.2.weight",    "mlp.2.bias",
  };
  const std::array<std::string_view, 24> decoder_suffixes = {
      "attn_ln.weight",
      "attn_ln.bias",
      "attn.query.weight",
      "attn.query.bias",
      "attn.key.weight",
      "attn.value.weight",
      "attn.value.bias",
      "attn.out.weight",
      "attn.out.bias",
      "cross_attn_ln.weight",
      "cross_attn_ln.bias",
      "cross_attn.query.weight",
      "cross_attn.query.bias",
      "cross_attn.key.weight",
      "cross_attn.value.weight",
      "cross_attn.value.bias",
      "cross_attn.out.weight",
      "cross_attn.out.bias",
      "mlp_ln.weight",
      "mlp_ln.bias",
      "mlp.0.weight",
      "mlp.0.bias",
      "mlp.2.weight",
      "mlp.2.bias",
  };
  for (int32_t block = 0; block < 4; ++block) {
    for (const auto suffix : encoder_suffixes) {
      append_legacy_whisper_tensor(legacy, "encoder.blocks." +
                                               std::to_string(block) + "." +
                                               std::string{suffix});
    }
    for (const auto suffix : decoder_suffixes) {
      append_legacy_whisper_tensor(legacy, "decoder.blocks." +
                                               std::to_string(block) + "." +
                                               std::string{suffix});
    }
  }

  REQUIRE(emel::model::whisper::detail::is_legacy_lmgg_whisper(
      std::span<const uint8_t>{legacy}));

  std::vector<uint8_t> gguf = {};
  REQUIRE(emel::model::whisper::detail::normalize_legacy_lmgg_to_gguf(
      std::span<const uint8_t>{legacy}, gguf));
  REQUIRE(gguf.size() > 4u);
  CHECK(std::string_view{reinterpret_cast<const char *>(gguf.data()), 4u} ==
        "GGUF");
  const std::string_view gguf_text{reinterpret_cast<const char *>(gguf.data()),
                                   gguf.size()};
  CHECK(gguf_text.find("model.encoder.layers.3.self_attn.q_proj.weight") !=
        std::string_view::npos);
  CHECK(gguf_text.find("model.decoder.layers.3.encoder_attn.q_proj.weight") !=
        std::string_view::npos);

  emel::gguf::loader::sm loader{};
  emel::gguf::loader::requirements requirements = {};
  const auto on_probe_done =
      emel::gguf::loader::event::probe_done_fn::from<&noop_probe_done>();
  const auto on_probe_error =
      emel::gguf::loader::event::probe_error_fn::from<&noop_probe_error>();
  const emel::gguf::loader::event::probe probe{std::span<const uint8_t>{gguf},
                                               requirements, on_probe_done,
                                               on_probe_error};
  REQUIRE(loader.process_event(probe));
  CHECK(requirements.kv_count == 4u);
  CHECK(requirements.tensor_count == 168u);
}

TEST_CASE("whisper_detail_rejects_legacy_tensor_size_overflow") {
  std::vector<uint8_t> legacy = {'l', 'm', 'g', 'g'};
  append_scalar<int32_t>(legacy, 51865);
  append_scalar<int32_t>(legacy, 1500);
  append_scalar<int32_t>(legacy, 384);
  append_scalar<int32_t>(legacy, 6);
  append_scalar<int32_t>(legacy, 4);
  append_scalar<int32_t>(legacy, 448);
  append_scalar<int32_t>(legacy, 384);
  append_scalar<int32_t>(legacy, 6);
  append_scalar<int32_t>(legacy, 4);
  append_scalar<int32_t>(legacy, 80);
  append_scalar<int32_t>(legacy, 7);
  append_scalar<int32_t>(legacy, 80);
  append_scalar<int32_t>(legacy, 1);
  for (int32_t index = 0; index < 80; ++index) {
    append_scalar<uint32_t>(legacy, 0u);
  }
  append_scalar<int32_t>(legacy, 1);
  append_scalar<uint32_t>(legacy, 0u);

  const std::string_view tensor_name = "encoder.positional_embedding";
  append_scalar<int32_t>(legacy, 4);
  append_scalar<int32_t>(legacy, static_cast<int32_t>(tensor_name.size()));
  append_scalar<int32_t>(legacy, 0);
  append_scalar<int32_t>(legacy, 65536);
  append_scalar<int32_t>(legacy, 65536);
  append_scalar<int32_t>(legacy, 65536);
  append_scalar<int32_t>(legacy, 65536);
  legacy.insert(legacy.end(), tensor_name.begin(), tensor_name.end());

  std::vector<uint8_t> gguf = {};
  CHECK_FALSE(emel::model::whisper::detail::normalize_legacy_lmgg_to_gguf(
      std::span<const uint8_t>{legacy}, gguf));
}
