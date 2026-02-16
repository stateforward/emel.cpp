#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
    std::snprintf(out, capacity, "%s\\emel_gguf_%d_%llu.gguf", tmp, pid,
                  static_cast<unsigned long long>(stamp));
#else
  const int written =
    std::snprintf(out, capacity, "%s/emel_gguf_%d_%llu.gguf", tmp, pid,
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

bool write_string(std::FILE * file, const char * value) {
  const uint64_t len = std::strlen(value);
  if (!write_u64(file, len)) {
    return false;
  }
  return std::fwrite(value, 1, len, file) == len;
}

bool write_header(std::FILE * file, const uint32_t version, const int64_t n_tensors, const int64_t n_kv) {
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

bool write_kv_i32(std::FILE * file, const char * key, const int32_t value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::model::gguf::value_type::k_i32);
  if (!write_i32(file, type)) {
    return false;
  }
  return write_i32(file, value);
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
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

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

bool write_vocab_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 1;
  const int64_t n_kv = 29;
  if (!write_header(file, emel::model::gguf::k_gguf_version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_bool(file, "tokenizer.ggml.add_bos_token", true) ||
      !write_kv_bool(file, "tokenizer.ggml.add_eos_token", false) ||
      !write_kv_bool(file, "tokenizer.ggml.add_sep_token", true) ||
      !write_kv_bool(file, "tokenizer.ggml.add_space_prefix", true) ||
      !write_kv_bool(file, "tokenizer.ggml.remove_extra_whitespaces", true) ||
      !write_kv_i32(file, "tokenizer.ggml.padding_token_id", 11) ||
      !write_kv_i32(file, "tokenizer.ggml.cls_token_id", 12) ||
      !write_kv_i32(file, "tokenizer.ggml.mask_token_id", 13) ||
      !write_kv_i32(file, "tokenizer.ggml.prefix_token_id", 14) ||
      !write_kv_i32(file, "tokenizer.ggml.suffix_token_id", 15) ||
      !write_kv_i32(file, "tokenizer.ggml.middle_token_id", 16) ||
      !write_kv_i32(file, "tokenizer.ggml.fim_pre_token_id", 17) ||
      !write_kv_i32(file, "tokenizer.ggml.fim_suf_token_id", 18) ||
      !write_kv_i32(file, "tokenizer.ggml.fim_mid_token_id", 19) ||
      !write_kv_i32(file, "tokenizer.ggml.fim_pad_token_id", 20) ||
      !write_kv_i32(file, "tokenizer.ggml.fim_rep_token_id", 21) ||
      !write_kv_i32(file, "tokenizer.ggml.fim_sep_token_id", 22) ||
      !write_kv_u32(file, "test.context_length", 1024) ||
      !write_kv_u32(file, "test.embedding_length", 512) ||
      !write_kv_u32(file, "test.embedding_length_out", 256) ||
      !write_kv_u32(file, "test.feed_forward_length", 2048) ||
      !write_kv_u32(file, "test.attention.head_count", 8) ||
      !write_kv_u32(file, "test.attention.head_count_kv", 4) ||
      !write_kv_u32(file, "test.rope.dimension_count", 64) ||
      !write_kv_f32(file, "test.rope.freq_base", 20000.0f) ||
      !write_kv_f32(file, "test.rope.scale_linear", 1.5f) ||
      !write_kv_f32(file, "test.rope.scaling.factor", 2.0f) ||
      !write_kv_f32(file, "test.rope.scaling.attn_factor", 1.25f) ||
      !write_kv_u32(file, "test.rope.scaling.original_context_length", 2048)) {
    std::fclose(file);
    return false;
  }

  std::array<int64_t, 4> dims = {1, 1, 1, 1};
  const int32_t type_raw = static_cast<int32_t>(emel::model::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "weight", type_raw, dims, 0)) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

}  // namespace

TEST_CASE("gguf parser reads tokenizer flags and ids") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_vocab_gguf(path));

  emel::model::data model = {};
  emel::model::gguf::context ctx = {};
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
  CHECK(emel::model::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  CHECK(model.vocab_data.add_bos);
  CHECK(!model.vocab_data.add_eos);
  CHECK(model.vocab_data.add_sep);
  CHECK(model.vocab_data.add_space_prefix);
  CHECK(model.vocab_data.remove_extra_whitespaces);
  CHECK(model.vocab_data.pad_id == 11);
  CHECK(model.vocab_data.cls_id == 12);
  CHECK(model.vocab_data.mask_id == 13);
  CHECK(model.vocab_data.prefix_id == 14);
  CHECK(model.vocab_data.suffix_id == 15);
  CHECK(model.vocab_data.middle_id == 16);
  CHECK(model.vocab_data.fim_pre_id == 17);
  CHECK(model.vocab_data.fim_suf_id == 18);
  CHECK(model.vocab_data.fim_mid_id == 19);
  CHECK(model.vocab_data.fim_pad_id == 20);
  CHECK(model.vocab_data.fim_rep_id == 21);
  CHECK(model.vocab_data.fim_sep_id == 22);
  CHECK(model.params.n_ctx == 1024);
  CHECK(model.params.n_embd == 512);
  CHECK(model.params.n_embd_out == 256);
  CHECK(model.params.n_ff == 2048);
  CHECK(model.params.n_head == 8);
  CHECK(model.params.n_head_kv == 4);
  CHECK(model.params.n_rot == 64);
  CHECK(model.params.rope_freq_base == doctest::Approx(20000.0f));
  CHECK(model.params.rope_scale_linear == doctest::Approx(1.5f));
  CHECK(model.params.rope_scaling_factor == doctest::Approx(2.0f));
  CHECK(model.params.rope_scaling_attn_factor == doctest::Approx(1.25f));
  CHECK(model.params.rope_scaling_orig_ctx_len == 2048);

  std::remove(path);
}
