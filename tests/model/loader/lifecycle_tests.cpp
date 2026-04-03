#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/llama/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/sm.hpp"

namespace {

struct owner_state {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::model::loader::error::none);
  uint64_t bytes_total = 0;
  uint64_t bytes_done = 0;
  bool used_mmap = false;
};

void on_done(void *object,
             const emel::model::loader::events::load_done &ev) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->done = true;
  owner->error = false;
  owner->bytes_total = ev.bytes_total;
  owner->bytes_done = ev.bytes_done;
  owner->used_mmap = ev.used_mmap;
}

void on_error(void *object,
              const emel::model::loader::events::load_error &ev) noexcept {
  auto *owner = static_cast<owner_state *>(object);
  owner->done = false;
  owner->error = true;
  owner->err = ev.err;
}

emel::error::type
parse_ok(void *, const emel::model::loader::event::load &req) noexcept {
  req.model_data.n_tensors = 1;
  req.model_data.n_layers = 1;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
parse_fail(void *, const emel::model::loader::event::load &) noexcept {
  return emel::error::cast(emel::model::loader::error::parse_failed);
}

emel::error::type load_weights_ok(void *,
                                  const emel::model::loader::event::load &req,
                                  uint64_t &bytes_total, uint64_t &bytes_done,
                                  bool &used_mmap) noexcept {
  static_cast<void>(req);
  bytes_total = 4096;
  bytes_done = 4096;
  used_mmap = true;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type
load_weights_backend_error(void *, const emel::model::loader::event::load &,
                           uint64_t &bytes_total, uint64_t &bytes_done,
                           bool &used_mmap) noexcept {
  bytes_total = 0;
  bytes_done = 0;
  used_mmap = false;
  return emel::error::cast(emel::model::loader::error::backend_error);
}

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

void append_kv_f32(std::vector<uint8_t> &arena,
                   std::vector<emel::gguf::loader::kv_entry> &entries,
                   const std::string_view key, const float value) {
  std::array<uint8_t, sizeof(float)> encoded = {};
  std::memcpy(encoded.data(), &value, sizeof(float));
  append_kv_entry(arena, entries, key,
                  emel::gguf::loader::detail::constants::gguf_type_float32,
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

std::string_view
metadata_piece(const emel::model::data::metadata &metadata,
               const emel::model::data::metadata::string_view view) {
  if (static_cast<size_t>(view.offset) + static_cast<size_t>(view.length) >
      metadata.blob.size()) {
    return {};
  }
  return std::string_view{
      metadata.blob.data() + view.offset,
      view.length,
  };
}

} // namespace

TEST_CASE("model loader lifecycle succeeds on full load path") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.load_weights = {nullptr, load_weights_ok};
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK(machine.process_event(request));
  CHECK(owner.done);
  CHECK_FALSE(owner.error);
  CHECK(owner.bytes_total == 4096);
  CHECK(owner.bytes_done == 4096);
  CHECK(owner.used_mmap);
}

TEST_CASE(
    "model_detail_populate_model_metadata_from_gguf_reads_qwen3_rope_domain") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};

  append_kv_string(arena, entries, "general.architecture", "qwen3");
  append_kv_u32(arena, entries, "qwen3.context_length", 32768u);
  append_kv_u32(arena, entries, "qwen3.embedding_length", 2048u);
  append_kv_u32(arena, entries, "qwen3.feed_forward_length", 6144u);
  append_kv_u32(arena, entries, "qwen3.attention.head_count", 16u);
  append_kv_u32(arena, entries, "qwen3.attention.head_count_kv", 8u);
  append_kv_u32(arena, entries, "qwen3.attention.key_length", 128u);
  append_kv_u32(arena, entries, "qwen3.attention.value_length", 128u);
  append_kv_u32(arena, entries, "qwen3.block_count", 28u);
  append_kv_f32(arena, entries, "qwen3.attention.layer_norm_rms_epsilon",
                1.0e-6f);
  append_kv_f32(arena, entries, "qwen3.rope.freq_base", 1000000.0f);
  append_kv_string(arena, entries, "qwen3.rope.scaling.type", "yarn");
  append_kv_f32(arena, entries, "qwen3.rope.scaling.factor", 4.0f);
  append_kv_u32(arena, entries, "qwen3.rope.scaling.original_context_length",
                8192u);
  append_kv_string_array(arena, entries, "tokenizer.tokens",
                         std::array<std::string_view, 2>{"a", "b"});
  append_kv_string(arena, entries, "tokenizer.chat_template", "{{ messages }}");

  auto model_data = std::make_unique<emel::model::data>();
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::populate_model_metadata_from_gguf(binding,
                                                                 *model_data) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(emel::model::architecture_name_view(*model_data) == "qwen3");
  CHECK(model_data->params.rope_type_value ==
        emel::model::data::rope_type::neox);
  CHECK(model_data->params.rope_scaling_type_value ==
        emel::model::data::rope_scaling_type::yarn);
  CHECK(model_data->params.rope_scaling_factor == doctest::Approx(4.0f));
  CHECK(model_data->params.rope_scaling_orig_ctx_len == 8192);
  CHECK(model_data->params.n_rot == 128);
  CHECK(model_data->vocab_data.n_tokens == 2u);
  CHECK(metadata_piece(model_data->meta,
                       model_data->meta.rope_data.scaling_type) == "yarn");
  CHECK(metadata_piece(model_data->meta,
                       model_data->meta.tokenizer_data.chat_template) ==
        "{{ messages }}");
}

TEST_CASE(
    "model_detail_populate_model_metadata_from_gguf_reads_llama_rope_domain") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};

  append_kv_string(arena, entries, "general.architecture", "llama");
  append_kv_u32(arena, entries, "llama.context_length", 4096u);
  append_kv_u32(arena, entries, "llama.embedding_length", 4096u);
  append_kv_u32(arena, entries, "llama.embedding_length_out", 4096u);
  append_kv_u32(arena, entries, "llama.feed_forward_length", 11008u);
  append_kv_u32(arena, entries, "llama.attention.head_count", 32u);
  append_kv_u32(arena, entries, "llama.attention.head_count_kv", 8u);
  append_kv_u32(arena, entries, "llama.rope.dimension_count", 128u);
  append_kv_u32(arena, entries, "llama.block_count", 32u);
  append_kv_u32(arena, entries, "llama.vocab_size", 32000u);
  append_kv_f32(arena, entries, "llama.attention.layer_norm_epsilon", 1.0e-5f);
  append_kv_f32(arena, entries, "llama.attention.layer_norm_rms_epsilon",
                1.0e-5f);
  append_kv_f32(arena, entries, "llama.attention.clamp_kqv", 0.0f);
  append_kv_f32(arena, entries, "llama.attn_logit_softcapping", 2.0f);
  append_kv_f32(arena, entries, "llama.final_logit_softcapping", 3.0f);
  append_kv_f32(arena, entries, "llama.residual_scale", 1.0f);
  append_kv_f32(arena, entries, "llama.embedding_scale", 1.0f);
  append_kv_f32(arena, entries, "llama.rope.freq_base", 10000.0f);
  append_kv_f32(arena, entries, "llama.rope.freq_base_swa", 5000.0f);
  append_kv_f32(arena, entries, "llama.rope.scale_linear", 8.0f);
  append_kv_f32(arena, entries, "llama.rope.scaling.factor", 8.0f);
  append_kv_f32(arena, entries, "llama.rope.scaling.attn_factor", 1.25f);
  append_kv_u32(arena, entries, "llama.rope.scaling.original_context_length",
                4096u);
  append_kv_bool(arena, entries, "llama.rope.scaling.finetuned", true);
  append_kv_f32(arena, entries, "llama.rope.scaling.yarn_log_multiplier", 2.0f);
  append_kv_f32(arena, entries, "llama.rope.scaling.yarn_ext_factor", 1.0f);
  append_kv_f32(arena, entries, "llama.rope.scaling.yarn_attn_factor", 0.75f);
  append_kv_f32(arena, entries, "llama.rope.scaling.yarn_beta_fast", 48.0f);
  append_kv_f32(arena, entries, "llama.rope.scaling.yarn_beta_slow", 2.0f);
  append_kv_string(arena, entries, "llama.rope.scaling.type", "linear");
  append_kv_bool(arena, entries, "llama.use_parallel_residual", true);
  append_kv_string_array(arena, entries, "tokenizer.tokens",
                         std::array<std::string_view, 2>{"a", "b"});
  append_kv_string(arena, entries, "tokenizer.chat_template", "{{ llama }}");

  auto model_data = std::make_unique<emel::model::data>();
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::populate_model_metadata_from_gguf(binding,
                                                                 *model_data) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(emel::model::architecture_name_view(*model_data) == "llama");
  CHECK(model_data->params.rope_type_value ==
        emel::model::data::rope_type::norm);
  CHECK(model_data->params.rope_scaling_type_value ==
        emel::model::data::rope_scaling_type::linear);
  CHECK(model_data->params.rope_scaling_factor == doctest::Approx(8.0f));
  CHECK(model_data->params.use_parallel_residual);
  CHECK(metadata_piece(model_data->meta,
                       model_data->meta.rope_data.scaling_type) == "linear");
}

TEST_CASE(
    "model_detail_populate_model_metadata_from_gguf_reads_lfm2_rope_domain") {
  std::vector<uint8_t> arena = {};
  std::vector<emel::gguf::loader::kv_entry> entries = {};

  append_kv_string(arena, entries, "general.architecture", "lfm2");
  append_kv_u32(arena, entries, "lfm2.context_length", 128000u);
  append_kv_u32(arena, entries, "lfm2.embedding_length", 2048u);
  append_kv_u32(arena, entries, "lfm2.feed_forward_length", 8192u);
  append_kv_u32(arena, entries, "lfm2.attention.head_count", 32u);
  append_kv_u32(arena, entries, "lfm2.block_count", 16u);
  append_kv_u32(arena, entries, "lfm2.vocab_size", 65536u);
  append_kv_u32(arena, entries, "lfm2.shortconv.l_cache", 3u);
  append_kv_f32(arena, entries, "lfm2.attention.layer_norm_rms_epsilon",
                1.0e-6f);
  append_kv_f32(arena, entries, "lfm2.rope.freq_base", 1000000.0f);
  append_kv_f32(arena, entries, "lfm2.rope.scale_linear", 4.0f);
  append_kv_f32(arena, entries, "lfm2.rope.scaling.factor", 4.0f);
  append_kv_f32(arena, entries, "lfm2.rope.scaling.attn_factor", 1.5f);
  append_kv_u32(arena, entries, "lfm2.rope.scaling.original_context_length",
                8192u);
  append_kv_bool(arena, entries, "lfm2.rope.scaling.finetuned", false);
  append_kv_f32(arena, entries, "lfm2.rope.scaling.yarn_log_multiplier", 2.0f);
  append_kv_f32(arena, entries, "lfm2.rope.scaling.yarn_ext_factor", 1.0f);
  append_kv_f32(arena, entries, "lfm2.rope.scaling.yarn_attn_factor", 0.75f);
  append_kv_f32(arena, entries, "lfm2.rope.scaling.yarn_beta_fast", 48.0f);
  append_kv_f32(arena, entries, "lfm2.rope.scaling.yarn_beta_slow", 2.0f);
  append_kv_string(arena, entries, "lfm2.rope.scaling.type", "yarn");
  append_kv_u32_array(arena, entries, "lfm2.attention.head_count_kv",
                      std::array<uint32_t, 3>{0u, 8u, 0u});
  append_kv_string_array(arena, entries, "tokenizer.tokens",
                         std::array<std::string_view, 2>{"x", "y"});
  append_kv_string(arena, entries, "tokenizer.chat_template", "{{ lfm2 }}");

  auto model_data = std::make_unique<emel::model::data>();
  const emel::model::detail::kv_binding binding{
      .arena = std::span<const uint8_t>{arena},
      .entries = std::span<const emel::gguf::loader::kv_entry>{entries},
  };

  REQUIRE(emel::model::detail::populate_model_metadata_from_gguf(binding,
                                                                 *model_data) ==
          emel::error::cast(emel::model::loader::error::none));
  CHECK(emel::model::architecture_name_view(*model_data) == "lfm2");
  CHECK(model_data->params.rope_type_value ==
        emel::model::data::rope_type::neox);
  CHECK(model_data->params.rope_scaling_type_value ==
        emel::model::data::rope_scaling_type::yarn);
  CHECK(model_data->params.n_head_kv == 8);
  CHECK(model_data->params.n_embd_out == model_data->params.n_embd);
  CHECK(metadata_piece(model_data->meta,
                       model_data->meta.rope_data.scaling_type) == "yarn");
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

TEST_CASE("model loader rejects full load without load_weights callback") {
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

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.load_weights = {nullptr, load_weights_ok};
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

TEST_CASE("model loader propagates load_weights backend error") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::sm machine{};
  owner_state owner{};
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};

  uint8_t file_bytes[8] = {};
  emel::model::loader::event::load request{*model, parse_model};
  request.file_image = file_bytes;
  request.file_size = sizeof(file_bytes);
  request.load_weights = {nullptr, load_weights_backend_error};
  request.map_layers = {nullptr, map_layers_ok};
  request.validate_structure = {nullptr, validate_structure_ok};
  request.validate_architecture_impl = {nullptr, validate_architecture_ok};
  request.on_done = {&owner, on_done};
  request.on_error = {&owner, on_error};

  CHECK_FALSE(machine.process_event(request));
  CHECK_FALSE(owner.done);
  CHECK(owner.error);
  CHECK(owner.err ==
        emel::error::cast(emel::model::loader::error::backend_error));
}

TEST_CASE(
    "model loader unclassified error guard matches only unclassified codes") {
  auto model = std::make_unique<emel::model::data>();
  emel::model::loader::event::parse_model_fn parse_model{nullptr, parse_ok};
  emel::model::loader::event::load request{*model, parse_model};
  emel::model::loader::event::load_ctx load_ctx{};
  emel::model::loader::event::load_runtime runtime{request, load_ctx};
  const auto guard = emel::model::loader::guard::error_unclassified_code{};

  load_ctx.err = emel::error::cast(emel::model::loader::error::none);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = emel::error::cast(emel::model::loader::error::invalid_request);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = emel::error::cast(emel::model::loader::error::parse_failed);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = emel::error::cast(emel::model::loader::error::backend_error);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = emel::error::cast(emel::model::loader::error::model_invalid);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = emel::error::cast(emel::model::loader::error::internal_error);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = emel::error::cast(emel::model::loader::error::untracked);
  CHECK_FALSE(guard(runtime));
  load_ctx.err = static_cast<emel::error::type>(0xFFFFu);
  CHECK(guard(runtime));
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
