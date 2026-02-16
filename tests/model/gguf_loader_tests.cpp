#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string_view>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "doctest/doctest.h"
#include "emel/model/gguf/loader.hpp"

namespace {

bool make_temp_path(char * out, const size_t capacity) {
  if (out == nullptr || capacity == 0) {
    return false;
  }
  const char * tmp = nullptr;
#if defined(_WIN32)
  tmp = std::getenv("TEMP");
  if (tmp == nullptr || tmp[0] == '\0') {
    tmp = ".";
  }
  const int pid = _getpid();
#else
  tmp = std::getenv("TMPDIR");
  if (tmp == nullptr || tmp[0] == '\0') {
    tmp = "/tmp";
  }
  const int pid = getpid();
#endif
  static uint64_t counter = 0;
  const uint64_t stamp = ++counter;
#if defined(_WIN32)
  const int written =
    std::snprintf(out, capacity, "%s\\emel_gguf_loader_%d_%llu.gguf", tmp, pid,
                  static_cast<unsigned long long>(stamp));
#else
  const int written =
    std::snprintf(out, capacity, "%s/emel_gguf_loader_%d_%llu.gguf", tmp, pid,
                  static_cast<unsigned long long>(stamp));
#endif
  return written > 0 && static_cast<size_t>(written) < capacity;
}

bool write_u64(std::FILE * file, const uint64_t value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_u32(std::FILE * file, const uint32_t value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_i32(std::FILE * file, const int32_t value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_f32(std::FILE * file, const float value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_string(std::FILE * file, const char * value) {
  const uint64_t len = std::strlen(value);
  if (!write_u64(file, len)) {
    return false;
  }
  return std::fwrite(value, 1, len, file) == len;
}

bool write_header(std::FILE * file, const uint32_t version, const int64_t n_tensors,
                  const int64_t n_kv) {
  const char magic[4] = {'G', 'G', 'U', 'F'};
  if (std::fwrite(magic, 1, sizeof(magic), file) != sizeof(magic)) {
    return false;
  }
  if (!write_u32(file, version)) {
    return false;
  }
  if (std::fwrite(&n_tensors, 1, sizeof(n_tensors), file) != sizeof(n_tensors)) {
    return false;
  }
  if (std::fwrite(&n_kv, 1, sizeof(n_kv), file) != sizeof(n_kv)) {
    return false;
  }
  return true;
}

bool write_kv_string(std::FILE * file, const char * key, const char * value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::model::gguf::value_type::k_string);
  if (!write_i32(file, type)) {
    return false;
  }
  return write_string(file, value);
}

bool write_kv_u32(std::FILE * file, const char * key, const uint32_t value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::model::gguf::value_type::k_u32);
  if (!write_i32(file, type)) {
    return false;
  }
  return write_u32(file, value);
}

bool write_kv_f32(std::FILE * file, const char * key, const float value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::model::gguf::value_type::k_f32);
  if (!write_i32(file, type)) {
    return false;
  }
  return write_f32(file, value);
}

bool write_kv_bool(std::FILE * file, const char * key, const bool value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::model::gguf::value_type::k_bool);
  if (!write_i32(file, type)) {
    return false;
  }
  const uint8_t raw = value ? 1 : 0;
  return std::fwrite(&raw, 1, sizeof(raw), file) == sizeof(raw);
}

bool write_kv_array_header(std::FILE * file, const char * key,
                           const emel::model::gguf::value_type elem_type,
                           const uint64_t count) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::model::gguf::value_type::k_array);
  if (!write_i32(file, type)) {
    return false;
  }
  const int32_t elem = static_cast<int32_t>(elem_type);
  if (!write_i32(file, elem)) {
    return false;
  }
  return write_u64(file, count);
}

bool write_kv_string_array(std::FILE * file, const char * key,
                           const std::array<const char *, 3> & values) {
  if (!write_kv_array_header(file, key,
                             emel::model::gguf::value_type::k_string,
                             values.size())) {
    return false;
  }
  for (const char * value : values) {
    if (!write_string(file, value)) {
      return false;
    }
  }
  return true;
}

bool write_kv_f32_array(std::FILE * file, const char * key,
                        const std::array<float, 3> & values) {
  if (!write_kv_array_header(file, key,
                             emel::model::gguf::value_type::k_f32,
                             values.size())) {
    return false;
  }
  for (float value : values) {
    if (!write_f32(file, value)) {
      return false;
    }
  }
  return true;
}

bool write_kv_i32_array(std::FILE * file, const char * key,
                        const std::array<int32_t, 3> & values) {
  if (!write_kv_array_header(file, key,
                             emel::model::gguf::value_type::k_i32,
                             values.size())) {
    return false;
  }
  for (int32_t value : values) {
    if (!write_i32(file, value)) {
      return false;
    }
  }
  return true;
}

bool write_kv_u8_array(std::FILE * file, const char * key,
                       const std::array<uint8_t, 4> & values) {
  if (!write_kv_array_header(file, key,
                             emel::model::gguf::value_type::k_u8,
                             values.size())) {
    return false;
  }
  return std::fwrite(values.data(), 1, values.size(), file) == values.size();
}

bool write_kv_bool_array(std::FILE * file, const char * key,
                         const std::array<uint8_t, 3> & values) {
  if (!write_kv_array_header(file, key,
                             emel::model::gguf::value_type::k_bool,
                             values.size())) {
    return false;
  }
  return std::fwrite(values.data(), 1, values.size(), file) == values.size();
}

bool write_bad_kv(std::FILE * file,
                  const char * key,
                  const emel::model::gguf::value_type type,
                  const bool is_array,
                  const emel::model::gguf::value_type elem_type) {
  if (!write_string(file, key)) {
    return false;
  }
  if (is_array) {
    const int32_t arr_type = static_cast<int32_t>(emel::model::gguf::value_type::k_array);
    if (!write_i32(file, arr_type)) {
      return false;
    }
    const int32_t elem = static_cast<int32_t>(elem_type);
    if (!write_i32(file, elem)) {
      return false;
    }
    if (!write_u64(file, 1)) {
      return false;
    }
    if (elem_type == emel::model::gguf::value_type::k_string) {
      return write_string(file, "x");
    }
    if (elem_type == emel::model::gguf::value_type::k_u32) {
      return write_u32(file, 1);
    }
    if (elem_type == emel::model::gguf::value_type::k_f32) {
      return write_f32(file, 1.0f);
    }
    if (elem_type == emel::model::gguf::value_type::k_bool) {
      const uint8_t raw = 1;
      return std::fwrite(&raw, 1, sizeof(raw), file) == sizeof(raw);
    }
    return true;
  }
  const int32_t raw_type = static_cast<int32_t>(type);
  if (!write_i32(file, raw_type)) {
    return false;
  }
  if (type == emel::model::gguf::value_type::k_string) {
    return write_string(file, "x");
  }
  if (type == emel::model::gguf::value_type::k_u32) {
    return write_u32(file, 1);
  }
  if (type == emel::model::gguf::value_type::k_i32) {
    return write_i32(file, 1);
  }
  if (type == emel::model::gguf::value_type::k_f32) {
    return write_f32(file, 1.0f);
  }
  if (type == emel::model::gguf::value_type::k_bool) {
    const uint8_t raw = 1;
    return std::fwrite(&raw, 1, sizeof(raw), file) == sizeof(raw);
  }
  return true;
}

constexpr auto k_suffix_u32_keys = std::array{
  ".context_length",
  ".vocab_size",
  ".embedding_length",
  ".embedding_length_out",
  ".features_length",
  ".feed_forward_length",
  ".leading_dense_block_count",
  ".expert_feed_forward_length",
  ".expert_shared_feed_forward_length",
  ".expert_chunk_feed_forward_length",
  ".expert_count",
  ".expert_used_count",
  ".expert_shared_count",
  ".expert_group_count",
  ".expert_group_used_count",
  ".expert_gating_func",
  ".experts_per_group",
  ".moe_every_n_layers",
  ".nextn_predict_layers",
  ".n_deepstack_layers",
  ".pooling_type",
  ".decoder_start_token_id",
  ".decoder_block_count",
  ".rescale_every_n_layers",
  ".time_mix_extra_dim",
  ".time_decay_extra_dim",
  ".token_shift_count",
  ".interleave_moe_layer_step",
  ".full_attention_interval",
  ".altup.active_idx",
  ".altup.num_inputs",
  ".embedding_length_per_layer_input",
  ".dense_2_feat_in",
  ".dense_2_feat_out",
  ".dense_3_feat_in",
  ".dense_3_feat_out",
  ".attention.head_count",
  ".attention.head_count_kv",
  ".attention.key_length",
  ".attention.value_length",
  ".attention.group_norm_groups",
  ".attention.q_lora_rank",
  ".attention.kv_lora_rank",
  ".attention.decay_lora_rank",
  ".attention.iclr_lora_rank",
  ".attention.value_residual_mix_lora_rank",
  ".attention.gate_lora_rank",
  ".attention.relative_buckets_count",
  ".attention.sliding_window",
  ".attention.sliding_window_pattern",
  ".attention.temperature_length",
  ".attention.key_length_mla",
  ".attention.value_length_mla",
  ".attention.indexer.head_count",
  ".attention.indexer.key_length",
  ".attention.indexer.top_k",
  ".attention.shared_kv_layers",
  ".rope.dimension_count",
  ".rope.scaling.original_context_length",
  ".ssm.conv_kernel",
  ".ssm.inner_size",
  ".ssm.state_size",
  ".ssm.time_step_rank",
  ".ssm.group_count",
  ".kda.head_dim",
  ".wkv.head_size",
  ".posnet.embedding_length",
  ".posnet.block_count",
  ".convnext.embedding_length",
  ".convnext.block_count",
  ".shortconv.l_cache"
};

constexpr auto k_suffix_f32_keys = std::array{
  ".expert_weights_scale",
  ".expert_group_scale",
  ".logit_scale",
  ".attn_logit_softcapping",
  ".router_logit_softcapping",
  ".final_logit_softcapping",
  ".residual_scale",
  ".embedding_scale",
  ".activation_sparsity_scale",
  ".attention.max_alibi_bias",
  ".attention.clamp_kqv",
  ".attention.layer_norm_epsilon",
  ".attention.layer_norm_rms_epsilon",
  ".attention.group_norm_epsilon",
  ".attention.scale",
  ".attention.output_scale",
  ".attention.temperature_scale",
  ".rope.freq_base",
  ".rope.freq_base_swa",
  ".rope.scale_linear",
  ".rope.scaling.factor",
  ".rope.scaling.attn_factor",
  ".rope.scaling.yarn_log_multiplier",
  ".rope.scaling.yarn_ext_factor",
  ".rope.scaling.yarn_attn_factor",
  ".rope.scaling.yarn_beta_fast",
  ".rope.scaling.yarn_beta_slow"
};

constexpr auto k_suffix_bool_keys = std::array{
  ".use_parallel_residual",
  ".expert_weights_norm",
  ".swin_norm",
  ".attention.causal",
  ".rope.scaling.finetuned",
  ".ssm.dt_b_c_rms"
};

constexpr auto k_suffix_string_keys = std::array{
  ".tensor_data_layout",
  ".rope.scaling.type"
};

bool write_tensor_info(std::FILE * file, const char * name, const int32_t type,
                       const std::array<int64_t, 4> & dims, const uint64_t offset) {
  if (!write_string(file, name)) {
    return false;
  }
  const uint32_t n_dims = 2;
  if (!write_u32(file, n_dims)) {
    return false;
  }
  if (std::fwrite(&dims[0], 1, sizeof(int64_t), file) != sizeof(int64_t)) {
    return false;
  }
  if (std::fwrite(&dims[1], 1, sizeof(int64_t), file) != sizeof(int64_t)) {
    return false;
  }
  if (!write_i32(file, type)) {
    return false;
  }
  return write_u64(file, offset);
}

bool pad_to_alignment(std::FILE * file, const uint64_t alignment) {
  const long pos = std::ftell(file);
  if (pos < 0) {
    return false;
  }
  const uint64_t aligned =
    emel::model::gguf::align_up_u64(static_cast<uint64_t>(pos), alignment);
  if (aligned == static_cast<uint64_t>(pos)) {
    return true;
  }
  const uint64_t padding = aligned - static_cast<uint64_t>(pos);
  std::array<uint8_t, 64> zeros = {};
  uint64_t remaining = padding;
  while (remaining > 0) {
    const uint64_t chunk = std::min<uint64_t>(remaining, zeros.size());
    if (std::fwrite(zeros.data(), 1, static_cast<size_t>(chunk), file) != chunk) {
      return false;
    }
    remaining -= chunk;
  }
  return true;
}

bool write_minimal_gguf(const char * path, const float weight_value) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  constexpr int64_t n_tensors = 1;
  constexpr int64_t n_kv = 3;
  if (!write_header(file, emel::model::gguf::k_gguf_version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_string(file, "general.architecture", "test") ||
      !write_kv_u32(file, "test.block_count", 1) ||
      !write_kv_bool(file, "test.attention.causal", true)) {
    std::fclose(file);
    return false;
  }
  std::array<int64_t, 4> dims = {1, 1, 1, 1};
  const int32_t type_raw = static_cast<int32_t>(emel::model::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "weight", type_raw, dims, 0)) {
    std::fclose(file);
    return false;
  }
  if (!pad_to_alignment(file, emel::model::gguf::k_default_alignment)) {
    std::fclose(file);
    return false;
  }
  std::array<uint8_t, emel::model::gguf::k_default_alignment> payload = {};
  std::memcpy(payload.data(), &weight_value, sizeof(weight_value));
  if (std::fwrite(payload.data(), 1, payload.size(), file) != payload.size()) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

bool write_vocab_arrays_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  constexpr int64_t n_tensors = 1;
  constexpr int64_t n_kv = 7;
  if (!write_header(file, emel::model::gguf::k_gguf_version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_string(file, "general.architecture", "test") ||
      !write_kv_u32(file, "test.block_count", 1) ||
      !write_kv_string_array(file, "tokenizer.ggml.tokens",
                             std::array<const char *, 3>{"a", "b", "c"}) ||
      !write_kv_f32_array(file, "tokenizer.ggml.scores",
                          std::array<float, 3>{0.1f, 0.2f, 0.3f}) ||
      !write_kv_i32_array(file, "tokenizer.ggml.token_type",
                          std::array<int32_t, 3>{1, 2, 3}) ||
      !write_kv_string_array(file, "tokenizer.ggml.merges",
                             std::array<const char *, 3>{"ab", "bc", "cd"}) ||
      !write_kv_u8_array(file, "tokenizer.ggml.precompiled_charsmap",
                         std::array<uint8_t, 4>{0, 1, 2, 3})) {
    std::fclose(file);
    return false;
  }
  std::array<int64_t, 4> dims = {1, 1, 1, 1};
  const int32_t type_raw = static_cast<int32_t>(emel::model::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "weight", type_raw, dims, 0)) {
    std::fclose(file);
    return false;
  }
  if (!pad_to_alignment(file, emel::model::gguf::k_default_alignment)) {
    std::fclose(file);
    return false;
  }
  std::array<uint8_t, emel::model::gguf::k_default_alignment> payload = {};
  if (std::fwrite(payload.data(), 1, payload.size(), file) != payload.size()) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

bool write_suffix_params_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_kv =
    2 +
    static_cast<int64_t>(k_suffix_u32_keys.size()) +
    static_cast<int64_t>(k_suffix_f32_keys.size()) +
    static_cast<int64_t>(k_suffix_bool_keys.size()) +
    static_cast<int64_t>(k_suffix_string_keys.size()) +
    5;
  if (!write_header(file, emel::model::gguf::k_gguf_version, 1, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_string(file, "general.architecture", "test") ||
      !write_kv_u32(file, "test.block_count", 1)) {
    std::fclose(file);
    return false;
  }
  char key[128] = {};
  for (const char * suffix : k_suffix_u32_keys) {
    std::snprintf(key, sizeof(key), "test%s", suffix);
    if (!write_kv_u32(file, key, 2)) {
      std::fclose(file);
      return false;
    }
  }
  for (const char * suffix : k_suffix_f32_keys) {
    std::snprintf(key, sizeof(key), "test%s", suffix);
    if (!write_kv_f32(file, key, 1.5f)) {
      std::fclose(file);
      return false;
    }
  }
  for (const char * suffix : k_suffix_bool_keys) {
    std::snprintf(key, sizeof(key), "test%s", suffix);
    if (!write_kv_bool(file, key, true)) {
      std::fclose(file);
      return false;
    }
  }
  for (const char * suffix : k_suffix_string_keys) {
    std::snprintf(key, sizeof(key), "test%s", suffix);
    if (!write_kv_string(file, key, "value")) {
      std::fclose(file);
      return false;
    }
  }
  if (!write_kv_i32_array(file, "test.rope.dimension_sections",
                          std::array<int32_t, 3>{16, 16, 16}) ||
      !write_kv_bool_array(file, "test.attention.sliding_window_pattern",
                           std::array<uint8_t, 3>{1, 0, 1}) ||
      !write_kv_f32_array(file, "test.swiglu_clamp_exp",
                          std::array<float, 3>{0.1f, 0.2f, 0.3f}) ||
      !write_kv_f32_array(file, "test.swiglu_clamp_shexp",
                          std::array<float, 3>{0.4f, 0.5f, 0.6f}) ||
      !write_kv_string_array(file, "test.classifier.output_labels",
                             std::array<const char *, 3>{"a", "b", "c"})) {
    std::fclose(file);
    return false;
  }

  std::array<int64_t, 4> dims = {1, 1, 1, 1};
  const int32_t type_raw = static_cast<int32_t>(emel::model::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "weight", type_raw, dims, 0)) {
    std::fclose(file);
    return false;
  }
  if (!pad_to_alignment(file, emel::model::gguf::k_default_alignment)) {
    std::fclose(file);
    return false;
  }
  std::array<uint8_t, emel::model::gguf::k_default_alignment> payload = {};
  if (std::fwrite(payload.data(), 1, payload.size(), file) != payload.size()) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

}  // namespace

TEST_CASE("gguf helpers cover key parsing and arithmetic") {
  using emel::model::gguf::add_overflow_u64;
  using emel::model::gguf::align_up_u64;
  using emel::model::gguf::key_equals;
  using emel::model::gguf::key_has_suffix;
  using emel::model::gguf::key_has_suffix_primary;
  using emel::model::gguf::metadata_string_equals;
  using emel::model::gguf::mul_overflow_u64;
  using emel::model::gguf::parse_indexed_key;
  using emel::model::gguf::prefix_is_primary_arch;

  CHECK(align_up_u64(0, 32) == 0);
  CHECK(align_up_u64(33, 32) == 64);

  uint64_t out = 0;
  CHECK(!mul_overflow_u64(0, 123, out));
  CHECK(out == 0);
  CHECK(mul_overflow_u64(std::numeric_limits<uint64_t>::max(), 2, out));

  CHECK(!add_overflow_u64(1, 2, out));
  CHECK(out == 3);
  CHECK(add_overflow_u64(std::numeric_limits<uint64_t>::max(), 1, out));

  CHECK(key_equals("abc", 3, "abc"));
  CHECK(!key_equals("abcd", 4, "abc"));

  const char * key = "test.block_count";
  uint64_t prefix_len = 0;
  CHECK(key_has_suffix(key, std::strlen(key), ".block_count", prefix_len));
  CHECK(prefix_len == 4);

  emel::model::gguf::context ctx = {};
  std::memcpy(ctx.architecture.data(), "test", 4);
  ctx.architecture_len = 4;
  CHECK(prefix_is_primary_arch(ctx, key, prefix_len));
  CHECK(!prefix_is_primary_arch(ctx, "clip.test.block_count",
                                std::strlen("clip.test.block_count") -
                                  std::strlen(".block_count")));
  prefix_len = 0;
  CHECK(key_has_suffix_primary(ctx, key, std::strlen(key), ".block_count", prefix_len));

  uint32_t index = 0;
  CHECK(parse_indexed_key("general.base_model.12.name",
                          std::strlen("general.base_model.12.name"),
                          "general.base_model.", ".name", index));
  CHECK(index == 12);
  CHECK(!parse_indexed_key("general.base_model.xx.name",
                           std::strlen("general.base_model.xx.name"),
                           "general.base_model.", ".name", index));

  emel::model::data::metadata meta = {};
  emel::model::data::metadata::string_view view = {};
  CHECK(emel::model::gguf::store_metadata_string(meta, "hello", 5, view));
  CHECK(metadata_string_equals(meta, view, "hello", 5));
  CHECK(!metadata_string_equals(meta, view, "world", 5));
}

TEST_CASE("gguf reader handles strings and keys") {
  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);

  CHECK(write_u64(file, 5));
  CHECK(std::fwrite("hello", 1, 5, file) == 5);
  std::rewind(file);

  emel::model::gguf::reader r{file};
  char buffer[8] = {};
  uint64_t len = 0;
  CHECK(r.read_string(buffer, sizeof(buffer), len));
  CHECK(len == 5);
  CHECK(std::strcmp(buffer, "hello") == 0);
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  CHECK(write_u64(file, 5));
  CHECK(std::fwrite("hello", 1, 5, file) == 5);
  std::rewind(file);
  emel::model::gguf::reader r2{file};
  char small[4] = {};
  len = 123;
  CHECK(emel::model::gguf::read_key(r2, small, sizeof(small), len));
  CHECK(len == 0);
  CHECK(small[0] == '\0');
  std::fclose(file);
}

TEST_CASE("gguf value parsing handles numeric conversions") {
  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  const int32_t neg = -1;
  CHECK(write_i32(file, neg));
  std::rewind(file);
  emel::model::gguf::reader r{file};
  uint32_t u32 = 0;
  CHECK(!emel::model::gguf::parse_u32_value(r, emel::model::gguf::value_type::k_i32, u32));
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  const uint32_t big = std::numeric_limits<uint32_t>::max();
  CHECK(write_u32(file, big));
  std::rewind(file);
  emel::model::gguf::reader r2{file};
  int32_t i32 = 0;
  CHECK(!emel::model::gguf::parse_i32_value(r2, emel::model::gguf::value_type::k_u32, i32));
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  const int8_t flag = -1;
  CHECK(std::fwrite(&flag, 1, sizeof(flag), file) == sizeof(flag));
  std::rewind(file);
  emel::model::gguf::reader r3{file};
  bool value = false;
  CHECK(emel::model::gguf::parse_bool_value(r3, emel::model::gguf::value_type::k_i8, value));
  CHECK(value);
  std::fclose(file);
}

TEST_CASE("gguf parser reads vocab arrays") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_vocab_arrays_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  CHECK(model.vocab_data.n_tokens == 3);
  CHECK(model.vocab_data.entries[0].score == doctest::Approx(0.1f));
  CHECK(model.vocab_data.entries[2].type == 3);
  CHECK(model.vocab_data.n_merges == 3);
  CHECK(model.vocab_data.precompiled_charsmap_size == 4);
  CHECK(model.vocab_data.precompiled_charsmap[2] == 2);

  std::remove(path);
}

TEST_CASE("gguf parser covers suffix parameters") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_suffix_params_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  CHECK(model.params.n_ctx == 2);
  CHECK(model.params.attention_causal);
  CHECK(model.params.rope_scaling_finetuned);
  CHECK(model.params.rope_dimension_sections_count == 3);

  std::remove(path);
}

TEST_CASE("gguf parser rejects bad kv types") {
  const auto numeric_keys = std::array{
    "general.quantization_version",
    "general.file_type",
    "general.sampling.top_k",
    "general.sampling.top_p",
    "general.sampling.min_p",
    "general.sampling.xtc_probability",
    "general.sampling.xtc_threshold",
    "general.sampling.temp",
    "general.sampling.penalty_last_n",
    "general.sampling.penalty_repeat",
    "general.sampling.mirostat",
    "general.sampling.mirostat_tau",
    "general.sampling.mirostat_eta",
    "general.base_model.count",
    "general.dataset.count",
    "general.alignment",
    "split.count",
    "split.no",
    "split.tensors.count",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.add_sep_token",
    "tokenizer.ggml.add_space_prefix",
    "tokenizer.ggml.remove_extra_whitespaces",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.cls_token_id",
    "tokenizer.ggml.mask_token_id",
    "tokenizer.ggml.prefix_token_id",
    "tokenizer.ggml.suffix_token_id",
    "tokenizer.ggml.middle_token_id",
    "tokenizer.ggml.fim_pre_token_id",
    "tokenizer.ggml.fim_suf_token_id",
    "tokenizer.ggml.fim_mid_token_id",
    "tokenizer.ggml.fim_pad_token_id",
    "tokenizer.ggml.fim_rep_token_id",
    "tokenizer.ggml.fim_sep_token_id",
    "adapter.lora.alpha",
    "imatrix.chunk_count",
    "imatrix.chunk_size",
    "clip.has_vision_encoder",
    "clip.has_audio_encoder",
    "clip.has_llava_projector",
    "clip.use_gelu",
    "clip.use_silu",
    "clip.vision.image_size",
    "clip.vision.image_min_pixels",
    "clip.vision.image_max_pixels",
    "clip.vision.preproc_image_size",
    "clip.vision.patch_size",
    "clip.vision.embedding_length",
    "clip.vision.feed_forward_length",
    "clip.vision.projection_dim",
    "clip.vision.block_count",
    "clip.vision.spatial_merge_size",
    "clip.vision.n_wa_pattern",
    "clip.vision.window_size",
    "clip.vision.attention.head_count",
    "clip.vision.attention.layer_norm_epsilon",
    "clip.vision.projector.scale_factor",
    "clip.audio.num_mel_bins",
    "clip.audio.embedding_length",
    "clip.audio.feed_forward_length",
    "clip.audio.projection_dim",
    "clip.audio.block_count",
    "clip.audio.attention.head_count",
    "clip.audio.attention.layer_norm_epsilon",
    "clip.audio.projector.stack_factor",
    "diffusion.shift_logits"
  };

  const auto string_keys = std::array{
    "general.architecture",
    "general.type",
    "general.sampling.sequence",
    "general.name",
    "general.author",
    "general.version",
    "general.organization",
    "general.finetune",
    "general.basename",
    "general.description",
    "general.quantized_by",
    "general.size_label",
    "general.license",
    "general.license.name",
    "general.license.link",
    "general.url",
    "general.doi",
    "general.uuid",
    "general.repo_url",
    "general.source.url",
    "general.source.doi",
    "general.source.uuid",
    "general.source.repo_url",
    "general.source.huggingface.repository",
    "general.base_model.0.name",
    "general.base_model.0.author",
    "general.base_model.0.version",
    "general.base_model.0.organization",
    "general.base_model.0.description",
    "general.base_model.0.url",
    "general.base_model.0.doi",
    "general.base_model.0.uuid",
    "general.base_model.0.repo_url",
    "general.dataset.0.name",
    "general.dataset.0.author",
    "general.dataset.0.version",
    "general.dataset.0.organization",
    "general.dataset.0.description",
    "general.dataset.0.url",
    "general.dataset.0.doi",
    "general.dataset.0.uuid",
    "general.dataset.0.repo_url",
    "tokenizer.huggingface.json",
    "tokenizer.rwkv.world",
    "tokenizer.chat_template",
    "tokenizer.chat_template.default",
    "adapter.type",
    "adapter.lora.task_name",
    "adapter.lora.prompt_prefix",
    "clip.projector_type",
    "clip.vision.projector_type",
    "clip.audio.projector_type"
  };

  const auto array_string_keys = std::array{
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.merges",
    "general.tags",
    "general.languages",
    "tokenizer.chat_templates",
    "imatrix.datasets",
    "test.classifier.output_labels"
  };

  const auto array_numeric_keys = std::array{
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.precompiled_charsmap",
    "adapter.alora.invocation_tokens",
    "test.rope.dimension_sections",
    "clip.vision.image_mean",
    "clip.vision.image_std",
    "clip.vision.wa_layer_indexes",
    "clip.vision.is_deepstack_layers",
    "test.attention.sliding_window_pattern",
    "test.swiglu_clamp_exp",
    "test.swiglu_clamp_shexp",
    "xielu.alpha_p",
    "xielu.alpha_q",
    "xielu.eps",
    "xielu.lambda",
    "xielu.max",
    "xielu.min"
  };

  auto run_bad_case = [&](const char * key,
                          const emel::model::gguf::value_type type,
                          const bool is_array,
                          const emel::model::gguf::value_type elem_type) {
    char path[1024] = {};
    CHECK(make_temp_path(path, sizeof(path)));
    std::FILE * file = std::fopen(path, "wb");
    REQUIRE(file != nullptr);
    const bool needs_arch = std::strncmp(key, "test.", 5) == 0;
    const int64_t n_kv = needs_arch ? 2 : 1;
    REQUIRE(write_header(file, emel::model::gguf::k_gguf_version, 0, n_kv));
    if (needs_arch) {
      REQUIRE(write_kv_string(file, "general.architecture", "test"));
    }
    REQUIRE(write_bad_kv(file, key, type, is_array, elem_type));
    std::fclose(file);

    emel::model::data model = {};
    emel::model::gguf::context ctx = {};
    emel::model::loader::event::load request{model};
    request.model_path = path;
    request.format_ctx = &ctx;
    int32_t err = EMEL_OK;
    const bool ok = emel::model::gguf::map_parser(request, &err);
    if (!ok) {
      CHECK(err != EMEL_OK);
    }
    std::remove(path);
  };

  for (const char * key : numeric_keys) {
    run_bad_case(key, emel::model::gguf::value_type::k_string, false,
                 emel::model::gguf::value_type::k_string);
  }
  for (const char * key : string_keys) {
    run_bad_case(key, emel::model::gguf::value_type::k_u32, false,
                 emel::model::gguf::value_type::k_u32);
  }
  for (const char * key : array_string_keys) {
    run_bad_case(key, emel::model::gguf::value_type::k_array, true,
                 emel::model::gguf::value_type::k_u32);
  }
  for (const char * key : array_numeric_keys) {
    run_bad_case(key, emel::model::gguf::value_type::k_array, true,
                 emel::model::gguf::value_type::k_string);
  }
  for (const char * suffix : k_suffix_u32_keys) {
    char key[128] = {};
    std::snprintf(key, sizeof(key), "test%s", suffix);
    run_bad_case(key, emel::model::gguf::value_type::k_string, false,
                 emel::model::gguf::value_type::k_string);
  }
  for (const char * suffix : k_suffix_f32_keys) {
    char key[128] = {};
    std::snprintf(key, sizeof(key), "test%s", suffix);
    run_bad_case(key, emel::model::gguf::value_type::k_string, false,
                 emel::model::gguf::value_type::k_string);
  }
  for (const char * suffix : k_suffix_bool_keys) {
    char key[128] = {};
    std::snprintf(key, sizeof(key), "test%s", suffix);
    run_bad_case(key, emel::model::gguf::value_type::k_string, false,
                 emel::model::gguf::value_type::k_string);
  }
  for (const char * suffix : k_suffix_string_keys) {
    char key[128] = {};
    std::snprintf(key, sizeof(key), "test%s", suffix);
    run_bad_case(key, emel::model::gguf::value_type::k_u32, false,
                 emel::model::gguf::value_type::k_u32);
  }
}

TEST_CASE("gguf metadata string and arrays parse") {
  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);

  CHECK(write_u64(file, 4));
  CHECK(std::fwrite("meta", 1, 4, file) == 4);
  std::rewind(file);

  emel::model::gguf::reader r{file};
  emel::model::data::metadata meta = {};
  emel::model::data::metadata::string_view out = {};
  CHECK(emel::model::gguf::read_metadata_string(
    r, emel::model::gguf::value_type::k_string, meta, out));
  CHECK(out.length == 4);
  CHECK(std::string_view(meta.blob.data() + out.offset, out.length) == "meta");
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  const int32_t elem_type =
    static_cast<int32_t>(emel::model::gguf::value_type::k_string);
  CHECK(write_i32(file, elem_type));
  CHECK(write_u64(file, 2));
  CHECK(write_string(file, "a"));
  CHECK(write_string(file, "b"));
  std::rewind(file);
  emel::model::gguf::reader r2{file};
  std::array<emel::model::data::metadata::string_view,
             emel::model::data::k_max_metadata_list> list = {};
  uint32_t count = 0;
  CHECK(emel::model::gguf::read_metadata_string_array(
    r2, emel::model::gguf::value_type::k_array, meta, list, count));
  CHECK(count == 2);
  CHECK(std::string_view(meta.blob.data() + list[1].offset, list[1].length) == "b");
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  const int32_t u32_type =
    static_cast<int32_t>(emel::model::gguf::value_type::k_u32);
  CHECK(write_i32(file, u32_type));
  CHECK(write_u64(file, 3));
  CHECK(write_u32(file, 1));
  CHECK(write_u32(file, 2));
  CHECK(write_u32(file, 3));
  std::rewind(file);
  emel::model::gguf::reader r3{file};
  std::array<uint32_t, emel::model::data::k_max_metadata_arrays> u32_out = {};
  uint32_t u32_count = 0;
  CHECK(emel::model::gguf::read_u32_array(
    r3, emel::model::gguf::value_type::k_array, u32_out, u32_count));
  CHECK(u32_count == 3);
  CHECK(u32_out[2] == 3);
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  const int32_t i32_type =
    static_cast<int32_t>(emel::model::gguf::value_type::k_i32);
  CHECK(write_i32(file, i32_type));
  CHECK(write_u64(file, 2));
  CHECK(write_i32(file, -1));
  CHECK(write_i32(file, 7));
  std::rewind(file);
  emel::model::gguf::reader r4{file};
  std::array<int32_t, 4> i32_out = {};
  int32_t i32_count = 0;
  CHECK(emel::model::gguf::read_i32_array(
    r4, emel::model::gguf::value_type::k_array, i32_out, i32_count));
  CHECK(i32_count == 2);
  CHECK(i32_out[0] == -1);
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  const int32_t f32_type =
    static_cast<int32_t>(emel::model::gguf::value_type::k_f32);
  CHECK(write_i32(file, f32_type));
  CHECK(write_u64(file, 2));
  CHECK(write_f32(file, 0.25f));
  CHECK(write_f32(file, 0.5f));
  std::rewind(file);
  emel::model::gguf::reader r5{file};
  std::array<float, emel::model::data::k_max_metadata_arrays> f32_out = {};
  uint32_t f32_count = 0;
  CHECK(emel::model::gguf::read_f32_array(
    r5, emel::model::gguf::value_type::k_array, f32_out, f32_count));
  CHECK(f32_count == 2);
  CHECK(f32_out[1] == doctest::Approx(0.5f));
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  const int32_t bool_type =
    static_cast<int32_t>(emel::model::gguf::value_type::k_bool);
  CHECK(write_i32(file, bool_type));
  CHECK(write_u64(file, 3));
  const uint8_t raw[3] = {1, 0, 1};
  CHECK(std::fwrite(raw, 1, sizeof(raw), file) == sizeof(raw));
  std::rewind(file);
  emel::model::gguf::reader r6{file};
  std::array<uint8_t, emel::model::data::k_max_metadata_arrays> b_out = {};
  uint32_t b_count = 0;
  CHECK(emel::model::gguf::read_bool_array(
    r6, emel::model::gguf::value_type::k_array, b_out, b_count));
  CHECK(b_count == 3);
  CHECK(b_out[1] == 0);
  std::fclose(file);
}

TEST_CASE("gguf parser reports invalid headers") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  std::FILE * file = std::fopen(path, "wb");
  REQUIRE(file != nullptr);
  const char magic[4] = {'B', 'A', 'D', '!'};
  CHECK(std::fwrite(magic, 1, sizeof(magic), file) == sizeof(magic));
  std::fclose(file);

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;
  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::map_parser(request, &err));
  CHECK(err != EMEL_OK);
  std::remove(path);
}

TEST_CASE("gguf helper routines validate tensor metadata") {
  emel::model::data model = {};
  std::array<uint8_t, 32> buffer = {};
  model.n_tensors = 1;
  model.weights_split_count = 1;
  model.weights_split_sizes[0] = buffer.size();
  auto & record = model.tensors[0];
  record.data_size = 4;
  record.file_index = 0;
  record.file_offset = 0;
  record.type = static_cast<int32_t>(emel::model::gguf::tensor_type::k_f32);
  record.data = buffer.data();

  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
  CHECK(emel::model::gguf::validate_tensor_data(model, &err));
  CHECK(err == EMEL_OK);

  record.data = nullptr;
  CHECK(!emel::model::gguf::validate_tensor_data(model, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  record.data = buffer.data();
  record.file_index = 2;
  CHECK(!emel::model::gguf::validate_tensor_data(model, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  record.file_index = 0;
  record.file_offset = buffer.size() + 1;
  CHECK(!emel::model::gguf::validate_tensor_data(model, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  record.file_offset = 0;
  record.type = -1;
  CHECK(!emel::model::gguf::validate_tensor_data(model, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
}

TEST_CASE("gguf tensor size and reset helpers") {
  uint64_t size = 0;
  std::array<int64_t, 4> dims = {1, 1, 1, 1};
  CHECK(emel::model::gguf::compute_tensor_size(
    dims, emel::model::gguf::tensor_type::k_f32, size));
  CHECK(size == 4);

  const int32_t block =
    emel::model::gguf::blck_size_for(emel::model::gguf::tensor_type::k_q4_0);
  std::array<int64_t, 4> bad_dims = {static_cast<int64_t>(block + 1), 1, 1, 1};
  CHECK(!emel::model::gguf::compute_tensor_size(
    bad_dims, emel::model::gguf::tensor_type::k_q4_0, size));

  emel::model::data model = {};
  model.n_tensors = 1;
  model.weights_split_count = 2;
  model.weights_split_sizes[0] = 16;
  model.weights_split_offsets[0] = 8;
  model.weights_size = 16;
  model.name_bytes_used = 5;
  emel::model::gguf::reset_model_data(model);
  CHECK(model.n_tensors == 0);
  CHECK(model.weights_split_count == 1);
  CHECK(model.weights_size == 0);

  emel::model::gguf::context ctx = {};
  ctx.owns_file = true;
  ctx.file = std::tmpfile();
  ctx.alignment = 64;
  ctx.split_count = 3;
  emel::model::gguf::reset_context(ctx);
  CHECK(ctx.file == nullptr);
  CHECK(ctx.alignment == emel::model::gguf::k_default_alignment);
  CHECK(ctx.split_count == 1);
}

TEST_CASE("gguf header and split metadata validation") {
  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  CHECK(write_header(file, emel::model::gguf::k_gguf_version, 1, 0));
  std::rewind(file);
  emel::model::gguf::reader r{file};
  emel::model::gguf::context ctx = {};
  int64_t n_tensors = 0;
  int64_t n_kv = 0;
  CHECK(emel::model::gguf::parse_header(r, ctx, n_tensors, n_kv));
  CHECK(n_tensors == 1);
  CHECK(n_kv == 0);
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  const char bad_magic[4] = {'B', 'A', 'D', '!'};
  CHECK(std::fwrite(bad_magic, 1, sizeof(bad_magic), file) == sizeof(bad_magic));
  std::rewind(file);
  emel::model::gguf::reader r2{file};
  CHECK(!emel::model::gguf::parse_header(r2, ctx, n_tensors, n_kv));
  std::fclose(file);

  ctx.split_count = 1;
  ctx.split_no = 0;
  ctx.split_tensors_count = 0;
  int32_t err = EMEL_OK;
  CHECK(emel::model::gguf::validate_split_metadata(ctx, 1, err));
  CHECK(err == EMEL_OK);

  err = EMEL_OK;
  CHECK(!emel::model::gguf::validate_split_metadata(ctx, -1, err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
}

TEST_CASE("gguf array value helpers consume streams") {
  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  CHECK(write_u64(file, 1));
  CHECK(std::fwrite("x", 1, 1, file) == 1);
  CHECK(write_u64(file, 1));
  CHECK(std::fwrite("y", 1, 1, file) == 1);
  std::rewind(file);

  emel::model::gguf::reader r{file};
  emel::model::data::metadata meta = {};
  std::array<emel::model::data::metadata::string_view,
             emel::model::data::k_max_metadata_list> out = {};
  uint32_t count = 0;
  CHECK(emel::model::gguf::read_metadata_string_array_values(
    r, emel::model::gguf::value_type::k_string, 2, meta, out, count));
  CHECK(count == 2);
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  CHECK(write_u32(file, 3));
  CHECK(write_u32(file, 4));
  std::rewind(file);
  emel::model::gguf::reader r2{file};
  std::array<uint32_t, emel::model::data::k_max_metadata_arrays> u32_out = {};
  uint32_t u32_count = 0;
  CHECK(emel::model::gguf::read_u32_array_values(
    r2, emel::model::gguf::value_type::k_u32, 2, u32_out, u32_count));
  CHECK(u32_count == 2);
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  CHECK(write_i32(file, -2));
  CHECK(write_i32(file, 6));
  std::rewind(file);
  emel::model::gguf::reader r3{file};
  std::array<int32_t, 4> i32_out = {};
  int32_t i32_count = 0;
  CHECK(emel::model::gguf::read_i32_array_values(
    r3, emel::model::gguf::value_type::k_i32, 2, i32_out, i32_count));
  CHECK(i32_count == 2);
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  CHECK(write_f32(file, 0.25f));
  CHECK(write_f32(file, 0.75f));
  std::rewind(file);
  emel::model::gguf::reader r4{file};
  std::array<float, emel::model::data::k_max_metadata_arrays> f32_out = {};
  uint32_t f32_count = 0;
  CHECK(emel::model::gguf::read_f32_array_values(
    r4, emel::model::gguf::value_type::k_f32, 2, f32_out, f32_count));
  CHECK(f32_out[1] == doctest::Approx(0.75f));
  std::fclose(file);

  file = std::tmpfile();
  REQUIRE(file != nullptr);
  const uint8_t flags[3] = {1, 0, 1};
  CHECK(std::fwrite(flags, 1, sizeof(flags), file) == sizeof(flags));
  std::rewind(file);
  emel::model::gguf::reader r5{file};
  std::array<uint8_t, emel::model::data::k_max_metadata_arrays> bool_out = {};
  uint32_t bool_count = 0;
  CHECK(emel::model::gguf::read_bool_array_values(
    r5, emel::model::gguf::value_type::k_u8, 3, bool_out, bool_count));
  CHECK(bool_count == 3);
  std::fclose(file);
}

TEST_CASE("gguf validate_row_data covers tensor types") {
  std::array<uint8_t, 512> storage = {};
  for (int32_t t = static_cast<int32_t>(emel::model::gguf::tensor_type::k_f32);
       t < static_cast<int32_t>(emel::model::gguf::tensor_type::k_count); ++t) {
    const auto type = static_cast<emel::model::gguf::tensor_type>(t);
    const uint32_t type_size = emel::model::gguf::type_size_for(type);
    if (type_size == 0) {
      CHECK(!emel::model::gguf::validate_row_data(type, nullptr, 1));
      continue;
    }
    REQUIRE(type_size <= storage.size());
    CHECK(emel::model::gguf::validate_row_data(type, storage.data(), type_size));
  }
  CHECK(!emel::model::gguf::validate_row_data(
    static_cast<emel::model::gguf::tensor_type>(
      static_cast<int32_t>(emel::model::gguf::tensor_type::k_count)),
    storage.data(), 4));
}

TEST_CASE("gguf weight loader handles streamed and mapped loads") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path, 3.5f));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::array<uint8_t, 64> weights_buffer = {};
  request.weights_buffer = weights_buffer.data();
  request.weights_buffer_size = weights_buffer.size();
  request.model_data.weights_split_count = 1;
  request.model_data.weights_split_sizes[0] = 0;
  request.model_data.weights_split_offsets[0] = 0;

  uint64_t done = 0;
  uint64_t total = 0;
  bool progress_seen = false;
  auto progress = [](float, void * user) -> bool {
    auto * flag = static_cast<bool *>(user);
    *flag = true;
    return true;
  };

  emel::model::weight_loader::event::load_weights ev = {};
  ev.loader_request = &request;
  ev.no_alloc = false;
  ev.check_tensors = false;
  ev.request_direct_io = false;
  ev.direct_io_supported = false;
  ev.progress_callback = progress;
  ev.progress_user_data = &progress_seen;

  CHECK(emel::model::gguf::load_streamed(ev, &done, &total, &err));
  CHECK(err == EMEL_OK);
  CHECK(done == total);
  CHECK(progress_seen);
  CHECK(request.model_data.weights_data == weights_buffer.data());

  float loaded = 0.0f;
  std::memcpy(&loaded, weights_buffer.data(), sizeof(float));
  CHECK(loaded == doctest::Approx(3.5f));

  emel::model::weight_loader::event::load_weights mmap_ev = ev;
  mmap_ev.no_alloc = true;
  done = 0;
  total = 0;
#if defined(_WIN32)
  CHECK(!emel::model::gguf::map_mmap(mmap_ev, &done, &total, &err));
  CHECK(err == EMEL_ERR_FORMAT_UNSUPPORTED);
#else
  CHECK(emel::model::gguf::map_mmap(mmap_ev, &done, &total, &err));
  CHECK(err == EMEL_OK);
  CHECK(done == 0);
  CHECK(total == request.model_data.weights_size);
#endif

  std::remove(path);
}

TEST_CASE("gguf weight loader validates inputs") {
  emel::model::weight_loader::event::load_weights ev = {};
  int32_t err = EMEL_OK;
  CHECK(!emel::model::gguf::init_mappings(ev, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  CHECK(!emel::model::gguf::validate_weights(ev, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_OK;
  CHECK(!emel::model::gguf::clean_up_weights(ev, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}
