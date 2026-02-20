#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#ifndef MAP_ANON
#define MAP_ANON MAP_ANONYMOUS
#endif
#endif

#include "doctest/doctest.h"
#include "emel/emel.h"
#include "emel/parser/gguf/actions.hpp"

namespace {

emel::model::data model{};

struct upload_probe {
  uint64_t begin_total = 0;
  uint64_t bytes = 0;
  uint64_t last_offset = 0;
  uint32_t begin_calls = 0;
  uint32_t end_calls = 0;
};

struct upload_fail_probe {
  uint32_t chunk_calls = 0;
};

bool upload_begin(void * ctx, const uint64_t total_bytes, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (ctx == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  auto * probe = static_cast<upload_probe *>(ctx);
  probe->begin_calls += 1;
  probe->begin_total = total_bytes;
  return true;
}

bool upload_chunk(void * ctx, const void * data, const uint64_t size, const uint64_t offset,
                  int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (ctx == nullptr || (data == nullptr && size > 0)) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  auto * probe = static_cast<upload_probe *>(ctx);
  probe->bytes += size;
  probe->last_offset = offset;
  return true;
}

bool upload_end(void * ctx, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (ctx == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  auto * probe = static_cast<upload_probe *>(ctx);
  probe->end_calls += 1;
  return true;
}

bool upload_begin_fail(void *, const uint64_t, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_ERR_BACKEND;
  }
  return false;
}

bool upload_begin_noop(void * ctx, const uint64_t, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return ctx != nullptr;
}

bool upload_chunk_fail(void * ctx, const void *, const uint64_t, const uint64_t,
                       int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (ctx == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  auto * probe = static_cast<upload_fail_probe *>(ctx);
  probe->chunk_calls += 1;
  if (probe->chunk_calls == 1) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_BACKEND;
    }
    return false;
  }
  return true;
}

bool upload_end_fail(void *, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_ERR_BACKEND;
  }
  return false;
}

bool upload_end_noop(void * ctx, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return ctx != nullptr;
}

bool write_u32(std::FILE * file, const uint32_t value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_i64(std::FILE * file, const int64_t value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_u64(std::FILE * file, const uint64_t value) {
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_string(std::FILE * file, const char * value) {
  const uint64_t len = std::strlen(value);
  if (!write_u64(file, len)) {
    return false;
  }
  return std::fwrite(value, 1, len, file) == len;
}

bool write_kv_string(std::FILE * file, const char * key, const char * value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_string);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  return write_string(file, value);
}

bool write_kv_u32(std::FILE * file, const char * key, const uint32_t value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_u32);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  return write_u32(file, value);
}

bool write_kv_i32(std::FILE * file, const char * key, const int32_t value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_i32);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  return std::fwrite(&value, 1, sizeof(value), file) == sizeof(value);
}

bool write_kv_bool(std::FILE * file, const char * key, const bool value) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_bool);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  const uint8_t raw = value ? 1 : 0;
  return std::fwrite(&raw, 1, sizeof(raw), file) == sizeof(raw);
}

bool write_kv_array_u32(std::FILE * file, const char * key, const uint32_t * values, const uint64_t count) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_array);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::parser::gguf::value_type::k_u32);
  if (std::fwrite(&elem_type, 1, sizeof(elem_type), file) != sizeof(elem_type)) {
    return false;
  }
  if (!write_u64(file, count)) {
    return false;
  }
  return std::fwrite(values, sizeof(uint32_t), count, file) == count;
}

bool write_kv_array_f32(std::FILE * file, const char * key, const float * values, const uint64_t count) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_array);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::parser::gguf::value_type::k_f32);
  if (std::fwrite(&elem_type, 1, sizeof(elem_type), file) != sizeof(elem_type)) {
    return false;
  }
  if (!write_u64(file, count)) {
    return false;
  }
  return std::fwrite(values, sizeof(float), count, file) == count;
}

bool write_kv_array_i32(std::FILE * file, const char * key, const int32_t * values, const uint64_t count) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_array);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::parser::gguf::value_type::k_i32);
  if (std::fwrite(&elem_type, 1, sizeof(elem_type), file) != sizeof(elem_type)) {
    return false;
  }
  if (!write_u64(file, count)) {
    return false;
  }
  return std::fwrite(values, sizeof(int32_t), count, file) == count;
}

bool write_kv_array_u8(std::FILE * file, const char * key, const uint8_t * values, const uint64_t count) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_array);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::parser::gguf::value_type::k_u8);
  if (std::fwrite(&elem_type, 1, sizeof(elem_type), file) != sizeof(elem_type)) {
    return false;
  }
  if (!write_u64(file, count)) {
    return false;
  }
  return std::fwrite(values, sizeof(uint8_t), count, file) == count;
}

bool write_kv_array_string(std::FILE * file, const char * key, const char * const * values,
                           const uint64_t count) {
  if (!write_string(file, key)) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_array);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  const int32_t elem_type = static_cast<int32_t>(emel::parser::gguf::value_type::k_string);
  if (std::fwrite(&elem_type, 1, sizeof(elem_type), file) != sizeof(elem_type)) {
    return false;
  }
  if (!write_u64(file, count)) {
    return false;
  }
  for (uint64_t i = 0; i < count; ++i) {
    if (!write_string(file, values[i])) {
      return false;
    }
  }
  return true;
}

bool write_long_key_kv(std::FILE * file, const uint64_t key_len, const uint32_t value) {
  if (!write_u64(file, key_len)) {
    return false;
  }
  std::array<char, 300> key = {};
  std::memset(key.data(), 'a', key.size());
  if (std::fwrite(key.data(), 1, key_len, file) != key_len) {
    return false;
  }
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::value_type::k_u32);
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  return write_u32(file, value);
}

bool write_header(std::FILE * file, const uint32_t version, const int64_t n_tensors, const int64_t n_kv) {
  const char magic[4] = {'G', 'G', 'U', 'F'};
  if (std::fwrite(magic, 1, sizeof(magic), file) != sizeof(magic)) {
    return false;
  }
  if (!write_u32(file, version)) {
    return false;
  }
  if (!write_i64(file, n_tensors) || !write_i64(file, n_kv)) {
    return false;
  }
  return true;
}

bool write_tensor_info(std::FILE * file, const char * name, const int32_t type,
                       const std::array<int64_t, 4> & dims, const uint64_t offset);

bool write_split_gguf(const char * path, const uint32_t split_count, const uint32_t split_no,
                      const uint32_t split_tensors_count) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 1;
  const int64_t n_kv = 3;
  if (!write_header(file, emel::parser::gguf::k_gguf_version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, emel::parser::gguf::k_key_split_count, split_count)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, emel::parser::gguf::k_key_split_no, split_no)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, emel::parser::gguf::k_key_split_tensors, split_tensors_count)) {
    std::fclose(file);
    return false;
  }
  std::array<int64_t, 4> dims = {1, 1, 1, 1};
  const int32_t type_raw = static_cast<int32_t>(emel::parser::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "weight", type_raw, dims, 0)) {
    std::fclose(file);
    return false;
  }
  const long pos = std::ftell(file);
  if (pos < 0) {
    std::fclose(file);
    return false;
  }
  const uint64_t aligned =
    emel::parser::gguf::align_up_u64(static_cast<uint64_t>(pos),
                                    emel::parser::gguf::k_default_alignment);
  const uint64_t padding = aligned - static_cast<uint64_t>(pos);
  if (padding > 0) {
    std::array<uint8_t, 64> zeros = {};
    uint64_t remaining = padding;
    while (remaining > 0) {
      const size_t chunk = static_cast<size_t>(std::min<uint64_t>(remaining, zeros.size()));
      if (std::fwrite(zeros.data(), 1, chunk, file) != chunk) {
        std::fclose(file);
        return false;
      }
      remaining -= chunk;
    }
  }
  uint64_t tensor_bytes = 0;
  if (!emel::parser::gguf::compute_tensor_size(dims,
                                              emel::parser::gguf::tensor_type::k_f32,
                                              tensor_bytes)) {
    std::fclose(file);
    return false;
  }
  uint64_t data_bytes =
    emel::parser::gguf::align_up_u64(tensor_bytes, emel::parser::gguf::k_default_alignment);
  std::array<uint8_t, 64> zeros = {};
  while (data_bytes > 0) {
    const size_t chunk = static_cast<size_t>(std::min<uint64_t>(data_bytes, zeros.size()));
    if (std::fwrite(zeros.data(), 1, chunk, file) != chunk) {
      std::fclose(file);
      return false;
    }
    data_bytes -= chunk;
  }
  std::fclose(file);
  return true;
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
  if (!write_i64(file, dims[0])) {
    return false;
  }
  if (!write_i64(file, dims[1])) {
    return false;
  }
  if (std::fwrite(&type, 1, sizeof(type), file) != sizeof(type)) {
    return false;
  }
  return write_u64(file, offset);
}

bool write_minimal_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 1;
  const int64_t n_kv = 4;
  const uint32_t version = emel::parser::gguf::k_gguf_version;
  if (!write_header(file, version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_string(file, emel::parser::gguf::k_key_architecture, "llama")) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, "llama.block_count", 2)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, "llama.context_length", 2048)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, "llama.embedding_length", 4096)) {
    std::fclose(file);
    return false;
  }
  const std::array<int64_t, 4> dims = {32, 1, 1, 1};
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "tensor.weight", type, dims, 0)) {
    std::fclose(file);
    return false;
  }
  const long meta_end = std::ftell(file);
  if (meta_end < 0) {
    std::fclose(file);
    return false;
  }
  const uint64_t alignment = emel::parser::gguf::k_default_alignment;
  const uint64_t aligned =
    emel::parser::gguf::align_up_u64(static_cast<uint64_t>(meta_end), alignment);
  const uint64_t padding = aligned - static_cast<uint64_t>(meta_end);
  if (padding > 0) {
    std::array<uint8_t, emel::parser::gguf::k_default_alignment> zeros = {};
    if (std::fwrite(zeros.data(), 1, static_cast<size_t>(padding), file) != padding) {
      std::fclose(file);
      return false;
    }
  }
  std::array<uint8_t, 128> weights = {};
  if (std::fwrite(weights.data(), 1, weights.size(), file) != weights.size()) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

bool write_minimal_split_gguf(const char * path, const uint16_t split_count,
                              const uint16_t split_no, const float value) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 1;
  const int64_t n_kv = 5;
  const uint32_t version = emel::parser::gguf::k_gguf_version;
  if (!write_header(file, version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_string(file, emel::parser::gguf::k_key_architecture, "llama")) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, "llama.block_count", 2)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, emel::parser::gguf::k_key_split_count, split_count)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, emel::parser::gguf::k_key_split_no, split_no)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, emel::parser::gguf::k_key_split_tensors, 1)) {
    std::fclose(file);
    return false;
  }
  const std::array<int64_t, 4> dims = {32, 1, 1, 1};
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "tensor.weight", type, dims, 0)) {
    std::fclose(file);
    return false;
  }
  const long meta_end = std::ftell(file);
  if (meta_end < 0) {
    std::fclose(file);
    return false;
  }
  const uint64_t alignment = emel::parser::gguf::k_default_alignment;
  const uint64_t aligned =
    emel::parser::gguf::align_up_u64(static_cast<uint64_t>(meta_end), alignment);
  const uint64_t padding = aligned - static_cast<uint64_t>(meta_end);
  std::array<uint8_t, 64> pad = {};
  if (padding > pad.size() || std::fwrite(pad.data(), 1, padding, file) != padding) {
    std::fclose(file);
    return false;
  }
  const uint32_t n_values = 32;
  for (uint32_t i = 0; i < n_values; ++i) {
    if (std::fwrite(&value, 1, sizeof(value), file) != sizeof(value)) {
      std::fclose(file);
      return false;
    }
  }
  std::fclose(file);
  return true;
}

bool write_gguf_without_arch(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 1;
  const int64_t n_kv = 1;
  const uint32_t version = emel::parser::gguf::k_gguf_version;
  if (!write_header(file, version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, "llama.block_count", 2)) {
    std::fclose(file);
    return false;
  }
  const std::array<int64_t, 4> dims = {32, 1, 1, 1};
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "tensor.weight", type, dims, 0)) {
    std::fclose(file);
    return false;
  }
  const long meta_end = std::ftell(file);
  if (meta_end < 0) {
    std::fclose(file);
    return false;
  }
  const uint64_t alignment = emel::parser::gguf::k_default_alignment;
  const uint64_t aligned =
    emel::parser::gguf::align_up_u64(static_cast<uint64_t>(meta_end), alignment);
  const uint64_t padding = aligned - static_cast<uint64_t>(meta_end);
  if (padding > 0) {
    std::array<uint8_t, emel::parser::gguf::k_default_alignment> zeros = {};
    if (std::fwrite(zeros.data(), 1, static_cast<size_t>(padding), file) != padding) {
      std::fclose(file);
      return false;
    }
  }
  std::array<uint8_t, 128> weights = {};
  if (std::fwrite(weights.data(), 1, weights.size(), file) != weights.size()) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

bool write_bad_alignment_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 0;
  const int64_t n_kv = 1;
  const uint32_t version = emel::parser::gguf::k_gguf_version;
  if (!write_header(file, version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, emel::parser::gguf::k_key_alignment, 3)) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

bool write_bad_tensor_type_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 1;
  const int64_t n_kv = 2;
  const uint32_t version = emel::parser::gguf::k_gguf_version;
  if (!write_header(file, version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_string(file, emel::parser::gguf::k_key_architecture, "llama")) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, "llama.block_count", 2)) {
    std::fclose(file);
    return false;
  }
  const std::array<int64_t, 4> dims = {32, 1, 1, 1};
  const int32_t type = 99;
  if (!write_tensor_info(file, "tensor.weight", type, dims, 0)) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

bool write_bad_tensor_offset_gguf(const char * path) {
  std::FILE * file = std::fopen(path, "wb");
  if (file == nullptr) {
    return false;
  }
  const int64_t n_tensors = 1;
  const int64_t n_kv = 2;
  const uint32_t version = emel::parser::gguf::k_gguf_version;
  if (!write_header(file, version, n_tensors, n_kv)) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_string(file, emel::parser::gguf::k_key_architecture, "llama")) {
    std::fclose(file);
    return false;
  }
  if (!write_kv_u32(file, "llama.block_count", 2)) {
    std::fclose(file);
    return false;
  }
  const std::array<int64_t, 4> dims = {32, 1, 1, 1};
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::tensor_type::k_f32);
  if (!write_tensor_info(file, "tensor.weight", type, dims, 128)) {
    std::fclose(file);
    return false;
  }
  std::fclose(file);
  return true;
}

bool make_temp_path(char * out, const size_t capacity) {
#if defined(_WIN32)
  return tmpnam_s(out, capacity) == 0;
#else
  if (capacity < 20) {
    return false;
  }
  std::snprintf(out, capacity, "/tmp/emel_ggufXXXXXX");
  const int fd = mkstemp(out);
  if (fd < 0) {
    return false;
  }
  close(fd);
  return true;
#endif
}

}  // namespace

TEST_CASE("gguf loader parses metadata and tensors") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  emel::parser::event::parse_model parse_request{
    .model = &model,
    .model_path = path,
    .format_ctx = &ctx
  };

  CHECK(emel::parser::gguf::parse_architecture(parse_request, &err));
  CHECK(err == EMEL_OK);
  CHECK(std::strcmp(model.architecture_name.data(), "llama") == 0);

  CHECK(emel::parser::gguf::parse_hparams(parse_request, &err));
  CHECK(err == EMEL_OK);
  CHECK(model.n_layers == 2);

  CHECK(emel::parser::gguf::map_tensors(parse_request, &err));
  CHECK(err == EMEL_OK);
  CHECK(model.n_tensors == 1);
  CHECK(model.tensors[0].data_size == 128);
  CHECK(model.weights_size == 128);

  std::remove(path);
}

TEST_CASE("gguf loader rejects missing architecture") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_gguf_without_arch(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  emel::parser::event::parse_model parse_request{
    .model = &model,
    .model_path = path,
    .format_ctx = &ctx
  };

  CHECK(!emel::parser::gguf::parse_architecture(parse_request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}

TEST_CASE("gguf loader rejects invalid alignment") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_bad_alignment_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(!emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}

TEST_CASE("gguf loader rejects invalid tensor type") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_bad_tensor_type_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(!emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}

TEST_CASE("gguf loader rejects invalid tensor offsets") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_bad_tensor_offset_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(!emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}

TEST_CASE("gguf loader accepts long key and array kv") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));

  std::FILE * file = std::fopen(path, "wb");
  REQUIRE(file != nullptr);
  const int64_t n_tensors = 1;
  const int64_t n_kv = 4;
  const uint32_t version = emel::parser::gguf::k_gguf_version;
  REQUIRE(write_header(file, version, n_tensors, n_kv));
  REQUIRE(write_kv_string(file, emel::parser::gguf::k_key_architecture, "llama"));
  REQUIRE(write_kv_u32(file, "llama.block_count", 2));
  const uint32_t arr[2] = {1, 2};
  REQUIRE(write_kv_array_u32(file, "dummy.array", arr, 2));
  REQUIRE(write_long_key_kv(file, 260, 7));
  const std::array<int64_t, 4> dims = {32, 1, 1, 1};
  const int32_t type = static_cast<int32_t>(emel::parser::gguf::tensor_type::k_f32);
  REQUIRE(write_tensor_info(file, "tensor.weight", type, dims, 0));
  const long meta_end = std::ftell(file);
  REQUIRE(meta_end >= 0);
  const uint64_t aligned =
    emel::parser::gguf::align_up_u64(static_cast<uint64_t>(meta_end),
                                    emel::parser::gguf::k_default_alignment);
  const uint64_t padding = aligned - static_cast<uint64_t>(meta_end);
  if (padding > 0) {
    std::array<uint8_t, emel::parser::gguf::k_default_alignment> zeros = {};
    REQUIRE(std::fwrite(zeros.data(), 1, static_cast<size_t>(padding), file) == padding);
  }
  std::array<uint8_t, 128> weights = {};
  REQUIRE(std::fwrite(weights.data(), 1, weights.size(), file) == weights.size());
  std::fclose(file);

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::remove(path);
}

TEST_CASE("gguf loader checks architecture list") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;
  const char * archs[1] = {"llama"};
  request.architectures = archs;
  request.n_architectures = 1;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  emel::parser::event::parse_model parse_request{
    .model = &model,
    .model_path = path,
    .architectures = archs,
    .n_architectures = 1,
    .format_ctx = &ctx
  };

  CHECK(emel::parser::gguf::parse_architecture(parse_request, &err));
  CHECK(emel::parser::gguf::map_architecture(parse_request, &err));
  CHECK(err == EMEL_OK);

  std::remove(path);
}

TEST_CASE("gguf loader streams weights into provided buffer") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::array<uint8_t, 128> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_OK);
  CHECK(done == 128);
  CHECK(total == 128);
  CHECK(model.tensors[0].data == buffer.data());

  std::remove(path);
}

TEST_CASE("gguf loader streams weights with upload callbacks") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::array<uint8_t, 128> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  upload_probe probe{};
  emel::model::weight_loader::event::load_weights load_req{
    .upload_ctx = &probe,
    .upload_begin = upload_begin,
    .upload_chunk = upload_chunk,
    .upload_end = upload_end,
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_OK);
  CHECK(done == total);
  CHECK(probe.begin_calls == 1);
  CHECK(probe.end_calls == 1);
  CHECK(probe.begin_total == total);
  CHECK(probe.bytes == total);
  CHECK(probe.last_offset < total);

  std::remove(path);
}

TEST_CASE("gguf loader rejects partial upload callbacks") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::array<uint8_t, 128> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  upload_probe probe{};
  emel::model::weight_loader::event::load_weights load_req{
    .upload_ctx = &probe,
    .upload_begin = upload_begin,
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  std::remove(path);
}

TEST_CASE("gguf loader reports upload begin failure") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::array<uint8_t, 128> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  emel::model::weight_loader::event::load_weights load_req{
    .upload_ctx = nullptr,
    .upload_begin = upload_begin_fail,
    .upload_chunk = upload_chunk,
    .upload_end = upload_end,
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_BACKEND);

  std::remove(path);
}

TEST_CASE("gguf loader reports upload chunk failure") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::array<uint8_t, 128> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  upload_fail_probe probe{};
  emel::model::weight_loader::event::load_weights load_req{
    .upload_ctx = &probe,
    .upload_begin = upload_begin_noop,
    .upload_chunk = upload_chunk_fail,
    .upload_end = upload_end_noop,
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_BACKEND);

  std::remove(path);
}

TEST_CASE("gguf loader reports upload end failure") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::array<uint8_t, 128> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  upload_probe probe{};
  emel::model::weight_loader::event::load_weights load_req{
    .upload_ctx = &probe,
    .upload_begin = upload_begin,
    .upload_chunk = upload_chunk,
    .upload_end = upload_end_fail,
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_BACKEND);

  std::remove(path);
}

TEST_CASE("gguf validate_split_metadata rejects invalid values") {
  using emel::parser::gguf::validate_split_metadata;

  emel::parser::gguf::context ctx{};
  int32_t err = EMEL_OK;

  ctx.split_count = 0;
  ctx.split_no = 0;
  ctx.split_tensors_count = 0;
  CHECK(!validate_split_metadata(ctx, 1, err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  ctx.split_count = 2;
  ctx.split_no = 2;
  ctx.split_tensors_count = 1;
  err = EMEL_OK;
  CHECK(!validate_split_metadata(ctx, 1, err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  ctx.split_count = 1;
  ctx.split_no = 1;
  ctx.split_tensors_count = 1;
  err = EMEL_OK;
  CHECK(!validate_split_metadata(ctx, 1, err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  ctx.split_count = 1;
  ctx.split_no = 0;
  ctx.split_tensors_count = 2;
  err = EMEL_OK;
  CHECK(!validate_split_metadata(ctx, 1, err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  ctx.split_count = 1;
  ctx.split_no = 0;
  ctx.split_tensors_count = 1;
  err = EMEL_OK;
  CHECK(validate_split_metadata(ctx, 1, err));
  CHECK(err == EMEL_OK);

  CHECK(emel::parser::gguf::is_little_endian());
}

TEST_CASE("gguf map_parser rejects negative kv count") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));

  std::FILE * file = std::fopen(path, "wb");
  REQUIRE(file != nullptr);
  CHECK(write_header(file, emel::parser::gguf::k_gguf_version, 0, -1));
  std::fclose(file);

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(!emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}

TEST_CASE("gguf map_parser rejects negative tensor count") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));

  std::FILE * file = std::fopen(path, "wb");
  REQUIRE(file != nullptr);
  CHECK(write_header(file, emel::parser::gguf::k_gguf_version, -1, 0));
  std::fclose(file);

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(!emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}

TEST_CASE("gguf map_parser rejects invalid split metadata") {
  char base_path[1024] = {};
  CHECK(make_temp_path(base_path, sizeof(base_path)));
  char path[1024] = {};
  std::snprintf(path, sizeof(path), "%s.gguf", base_path);
  CHECK(write_split_gguf(path, 1, 1, 1));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(!emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}

TEST_CASE("gguf map_parser rejects mismatched split metadata") {
  char base_path[1024] = {};
  CHECK(make_temp_path(base_path, sizeof(base_path)));
  char path[1024] = {};
  std::snprintf(path, sizeof(path), "%s.gguf", base_path);
  CHECK(write_split_gguf(path, 2, 0, 1));

  char split_path[1024] = {};
  CHECK(emel::parser::gguf::format_split_path(path, 1, 2, split_path, sizeof(split_path)));
  CHECK(write_split_gguf(split_path, 2, 0, 1));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(!emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
  std::remove(split_path);
}

#if !defined(__linux__)
TEST_CASE("gguf loader rejects direct io on unsupported platforms") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::array<uint8_t, 128> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  emel::model::weight_loader::event::load_weights load_req{
    .request_direct_io = true,
    .direct_io_supported = true,
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_FORMAT_UNSUPPORTED);

  std::remove(path);
}
#endif

#if defined(__linux__)
TEST_CASE("gguf loader streams weights with direct io") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  const uint64_t padded_size =
    emel::parser::gguf::align_up_u64(ctx.data_offset + ctx.data_size,
                                    emel::parser::gguf::k_direct_io_alignment);
  std::FILE * pad = std::fopen(path, "ab");
  REQUIRE(pad != nullptr);
  if (padded_size > 0) {
    CHECK(std::fseek(pad, static_cast<long>(padded_size - 1), SEEK_SET) == 0);
    const uint8_t zero = 0;
    CHECK(std::fwrite(&zero, 1, 1, pad) == 1);
  }
  std::fclose(pad);

  std::array<uint8_t, 128> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  emel::model::weight_loader::event::load_weights load_req{
    .request_direct_io = true,
    .direct_io_supported = true,
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_OK);
  CHECK(done == total);
  CHECK(model.tensors[0].data == buffer.data());

  std::remove(path);
}
#endif

TEST_CASE("gguf loader rejects too-small weight buffer") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  std::array<uint8_t, 64> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  std::remove(path);
}

TEST_CASE("gguf loader validates structure") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  CHECK(emel::parser::gguf::validate_structure(request, &err));
  CHECK(err == EMEL_OK);

  ctx.data_offset = 0;
  CHECK(!emel::parser::gguf::validate_structure(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::remove(path);
}


#if !defined(_WIN32)
TEST_CASE("gguf loader maps weights via mmap") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(emel::parser::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_OK);
  CHECK(done == 128);
  CHECK(total == 128);
  CHECK(model.tensors[0].data != nullptr);

  emel::parser::gguf::reset_context(ctx);
  std::remove(path);
}
#endif

TEST_CASE("gguf tensor type helpers cover switch cases") {
  using emel::parser::gguf::blck_size_for;
  using emel::parser::gguf::tensor_type;
  using emel::parser::gguf::type_size_for;

  for (int32_t i = 0; i < static_cast<int32_t>(tensor_type::k_count); ++i) {
    const auto type = static_cast<tensor_type>(i);
    (void)blck_size_for(type);
    (void)type_size_for(type);
  }

  CHECK(blck_size_for(tensor_type::k_f32) > 0);
  CHECK(type_size_for(tensor_type::k_f32) > 0);
  CHECK(blck_size_for(tensor_type::k_q4_0) > 0);
  CHECK(type_size_for(tensor_type::k_q4_0) > 0);
  CHECK(blck_size_for(tensor_type::k_q8_0) > 0);
  CHECK(type_size_for(tensor_type::k_q8_0) > 0);

  const auto invalid = static_cast<tensor_type>(static_cast<int32_t>(tensor_type::k_count));
  CHECK(blck_size_for(invalid) == 0);
  CHECK(type_size_for(invalid) == 0);
}

TEST_CASE("gguf parses u32 values across types") {
  using emel::parser::gguf::parse_u32_value;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint32_t value = 123u;
    REQUIRE(std::fwrite(&value, 1, sizeof(value), file) == sizeof(value));
    std::rewind(file);
    reader r{file};
    uint32_t out = 0;
    CHECK(parse_u32_value(r, value_type::k_u32, out));
    CHECK(out == value);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const int32_t value = 7;
    REQUIRE(std::fwrite(&value, 1, sizeof(value), file) == sizeof(value));
    std::rewind(file);
    reader r{file};
    uint32_t out = 0;
    CHECK(parse_u32_value(r, value_type::k_i32, out));
    CHECK(out == 7u);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const int32_t value = -1;
    REQUIRE(std::fwrite(&value, 1, sizeof(value), file) == sizeof(value));
    std::rewind(file);
    reader r{file};
    uint32_t out = 0;
    CHECK(!parse_u32_value(r, value_type::k_i32, out));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint16_t value = 5u;
    REQUIRE(std::fwrite(&value, 1, sizeof(value), file) == sizeof(value));
    std::rewind(file);
    reader r{file};
    uint32_t out = 0;
    CHECK(parse_u32_value(r, value_type::k_u16, out));
    CHECK(out == 5u);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint32_t value = 9u;
    REQUIRE(std::fwrite(&value, 1, sizeof(value), file) == sizeof(value));
    std::rewind(file);
    reader r{file};
    uint32_t out = 0;
    CHECK(!parse_u32_value(r, value_type::k_f32, out));
    std::fclose(file);
  }
}

TEST_CASE("gguf compute_tensor_size handles invalid dimensions") {
  using emel::parser::gguf::compute_tensor_size;
  using emel::parser::gguf::tensor_type;

  uint64_t out = 0;
  std::array<int64_t, emel::parser::gguf::k_max_dims> dims = {32, 1, 1, 1};
  CHECK(compute_tensor_size(dims, tensor_type::k_f32, out));
  CHECK(out > 0);

  dims = {-1, 1, 1, 1};
  CHECK(!compute_tensor_size(dims, tensor_type::k_f32, out));

  dims = {31, 1, 1, 1};
  CHECK(!compute_tensor_size(dims, tensor_type::k_q4_0, out));

  dims = {std::numeric_limits<int64_t>::max(), 2, 1, 1};
  CHECK(!compute_tensor_size(dims, tensor_type::k_f32, out));
}

TEST_CASE("gguf read_key and skip_value cover branches") {
  using emel::parser::gguf::read_key;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::skip_value;
  using emel::parser::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 300;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    std::array<char, 300> payload = {};
    std::memset(payload.data(), 'a', payload.size());
    REQUIRE(std::fwrite(payload.data(), 1, payload.size(), file) == payload.size());
    std::rewind(file);
    reader r{file};
    std::array<char, 8> out = {};
    uint64_t out_len = 0;
    CHECK(read_key(r, out.data(), out.size(), out_len));
    CHECK(out_len == 0);
    CHECK(out[0] == '\0');
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 4;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char key[] = {'t', 'e', 's', 't'};
    REQUIRE(std::fwrite(key, 1, sizeof(key), file) == sizeof(key));
    std::rewind(file);
    reader r{file};
    uint64_t out_len = 0;
    CHECK(read_key(r, nullptr, 0, out_len));
    CHECK(out_len == len);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint32_t val = 0x12345678u;
    REQUIRE(std::fwrite(&val, 1, sizeof(val), file) == sizeof(val));
    std::rewind(file);
    reader r{file};
    CHECK(skip_value(r, value_type::k_u32, 1));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 3;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char str[] = {'f', 'o', 'o'};
    REQUIRE(std::fwrite(str, 1, sizeof(str), file) == sizeof(str));
    std::rewind(file);
    reader r{file};
    CHECK(skip_value(r, value_type::k_string, 1));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    reader r{file};
    CHECK(!skip_value(r, value_type::k_count, 1));
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_header rejects invalid versions") {
  using emel::parser::gguf::parse_header;
  using emel::parser::gguf::reader;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const char bad_magic[4] = {'B', 'A', 'D', '!'};
    REQUIRE(std::fwrite(bad_magic, 1, sizeof(bad_magic), file) == sizeof(bad_magic));
    std::rewind(file);
    reader r{file};
    emel::parser::gguf::context ctx{};
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    CHECK(!parse_header(r, ctx, n_tensors, n_kv));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_header(file, 0, 0, 0));
    std::rewind(file);
    reader r{file};
    emel::parser::gguf::context ctx{};
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    CHECK(!parse_header(r, ctx, n_tensors, n_kv));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint32_t bad_version = emel::parser::gguf::k_gguf_version + 1;
    REQUIRE(write_header(file, bad_version, 0, 0));
    std::rewind(file);
    reader r{file};
    emel::parser::gguf::context ctx{};
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    CHECK(!parse_header(r, ctx, n_tensors, n_kv));
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_kv binds pending block count") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_kv_u32(file, "llama.block_count", 7));
  REQUIRE(write_kv_string(file, emel::parser::gguf::k_key_architecture, "llama"));
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = 0;
  CHECK(parse_kv(r, ctx, model, 2, err));
  CHECK(ctx.block_count == 7);
  std::fclose(file);
}

TEST_CASE("gguf parser reports missing model data") {
  using emel::parser::gguf::map_architecture;
  using emel::parser::gguf::map_tensors;
  using emel::parser::gguf::parse_hparams;

  emel::parser::event::parse_model parse_request{};
  int32_t err = EMEL_ERR_INVALID_ARGUMENT;

  CHECK(!map_tensors(parse_request, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_ERR_INVALID_ARGUMENT;
  CHECK(!parse_hparams(parse_request, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  err = EMEL_ERR_INVALID_ARGUMENT;
  CHECK(!map_architecture(parse_request, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gguf map_architecture rejects mismatched list") {
  emel::model::data model{};
  std::memcpy(model.architecture_name.data(), "llama", 6);

  const char * list[] = {"gpt", "other"};
  emel::parser::event::parse_model parse_request{};
  parse_request.model = &model;
  parse_request.architectures = list;
  parse_request.n_architectures = 2;

  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::map_architecture(parse_request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
}

TEST_CASE("gguf parse_kv handles arrays and alignment") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint32_t values[] = {1};
    REQUIRE(write_kv_array_u32(file, "numbers", values, 1));
    std::rewind(file);

    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(parse_kv(r, ctx, model, 1, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, emel::parser::gguf::k_key_alignment));
    const int32_t type = static_cast<int32_t>(value_type::k_u32);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    const uint32_t alignment = 3;
    REQUIRE(write_u32(file, alignment));
    std::rewind(file);

    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, model, 1, err));
    CHECK(err == EMEL_ERR_MODEL_INVALID);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "bad.array"));
    const int32_t type = static_cast<int32_t>(value_type::k_array);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    const int32_t elem_type = static_cast<int32_t>(value_type::k_count);
    REQUIRE(std::fwrite(&elem_type, 1, sizeof(elem_type), file) == sizeof(elem_type));
    const uint64_t count = 1;
    REQUIRE(write_u64(file, count));
    std::rewind(file);

    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, model, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_kv handles split metadata") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_kv_u32(file, emel::parser::gguf::k_key_split_count, 2));
  REQUIRE(write_kv_u32(file, emel::parser::gguf::k_key_split_no, 1));
  REQUIRE(write_kv_u32(file, emel::parser::gguf::k_key_split_tensors, 4));
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(parse_kv(r, ctx, model, 3, err));
  CHECK(ctx.split_count == 2);
  CHECK(ctx.split_no == 1);
  CHECK(ctx.split_tensors_count == 4);
  std::fclose(file);
}

TEST_CASE("gguf parse_kv loads tokenizer metadata and arrays") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::metadata_string_equals;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  const char * tokens[] = {"hello", "world"};
  const float scores[] = {0.1f, 0.2f};
  const int32_t types[] = {1, 2};
  const char * merges[] = {"h e", "w o"};
  const uint8_t charsmap[] = {1, 2, 3};
  REQUIRE(write_kv_string(file, "tokenizer.ggml.model", "gpt2"));
  REQUIRE(write_kv_string(file, "tokenizer.ggml.pre", "default"));
  REQUIRE(write_kv_array_string(file, "tokenizer.ggml.tokens", tokens, 2));
  REQUIRE(write_kv_array_f32(file, "tokenizer.ggml.scores", scores, 2));
  REQUIRE(write_kv_array_i32(file, "tokenizer.ggml.token_type", types, 2));
  REQUIRE(write_kv_u32(file, "tokenizer.ggml.token_type_count", 3));
  REQUIRE(write_kv_array_string(file, "tokenizer.ggml.merges", merges, 2));
  REQUIRE(write_kv_array_u8(file, "tokenizer.ggml.precompiled_charsmap", charsmap, 3));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.bos_token_id", 1));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.eos_token_id", 2));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.eot_token_id", 3));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.eom_token_id", 4));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.unknown_token_id", 5));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.seperator_token_id", 6));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.padding_token_id", 7));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.cls_token_id", 8));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.mask_token_id", 9));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.prefix_token_id", 10));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.suffix_token_id", 11));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.middle_token_id", 12));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.fim_pre_token_id", 13));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.fim_suf_token_id", 14));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.fim_mid_token_id", 15));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.fim_pad_token_id", 16));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.fim_rep_token_id", 17));
  REQUIRE(write_kv_i32(file, "tokenizer.ggml.fim_sep_token_id", 18));
  REQUIRE(write_kv_bool(file, "tokenizer.ggml.add_bos_token", true));
  REQUIRE(write_kv_bool(file, "tokenizer.ggml.add_eos_token", false));
  REQUIRE(write_kv_bool(file, "tokenizer.ggml.add_sep_token", true));
  REQUIRE(write_kv_bool(file, "tokenizer.ggml.add_space_prefix", true));
  REQUIRE(write_kv_bool(file, "tokenizer.ggml.remove_extra_whitespaces", false));
  REQUIRE(write_kv_bool(file, "tokenizer.ggml.escape_whitespaces", true));
  REQUIRE(write_kv_bool(file, "tokenizer.ggml.treat_whitespace_as_suffix", true));
  REQUIRE(write_kv_bool(file, "tokenizer.ggml.ignore_merges", false));
  std::rewind(file);

  reader r{file};
  context ctx{};
  emel::model::data local_model{};
  int32_t err = EMEL_OK;
  CHECK(parse_kv(r, ctx, local_model, 34, err));
  CHECK(local_model.vocab_data.n_tokens == 2);
  CHECK(local_model.vocab_data.n_token_types == 3);
  CHECK(local_model.vocab_data.entries[0].score == doctest::Approx(0.1f));
  CHECK(local_model.vocab_data.entries[1].type == 2);
  CHECK(std::strncmp(local_model.vocab_data.tokenizer_model.data(), "gpt2", 4) == 0);
  CHECK(std::strncmp(local_model.vocab_data.tokenizer_pre.data(), "default", 7) == 0);
  CHECK(local_model.vocab_data.n_merges == 2);
  CHECK(local_model.vocab_data.precompiled_charsmap_size == 3);
  CHECK(local_model.vocab_data.bos_id == 1);
  CHECK(local_model.vocab_data.eos_id == 2);
  CHECK(local_model.vocab_data.eot_id == 3);
  CHECK(local_model.vocab_data.eom_id == 4);
  CHECK(local_model.vocab_data.unk_id == 5);
  CHECK(local_model.vocab_data.sep_id == 6);
  CHECK(local_model.vocab_data.pad_id == 7);
  CHECK(local_model.vocab_data.cls_id == 8);
  CHECK(local_model.vocab_data.mask_id == 9);
  CHECK(local_model.vocab_data.prefix_id == 10);
  CHECK(local_model.vocab_data.suffix_id == 11);
  CHECK(local_model.vocab_data.middle_id == 12);
  CHECK(local_model.vocab_data.fim_pre_id == 13);
  CHECK(local_model.vocab_data.fim_suf_id == 14);
  CHECK(local_model.vocab_data.fim_mid_id == 15);
  CHECK(local_model.vocab_data.fim_pad_id == 16);
  CHECK(local_model.vocab_data.fim_rep_id == 17);
  CHECK(local_model.vocab_data.fim_sep_id == 18);
  CHECK(local_model.vocab_data.add_bos);
  CHECK(!local_model.vocab_data.add_eos);
  CHECK(local_model.vocab_data.add_sep);
  CHECK(local_model.vocab_data.add_space_prefix);
  CHECK(!local_model.vocab_data.remove_extra_whitespaces);
  CHECK(local_model.vocab_data.escape_whitespaces);
  CHECK(local_model.vocab_data.treat_whitespace_as_suffix);
  CHECK(!local_model.vocab_data.ignore_merges);
  std::fclose(file);
}

TEST_CASE("gguf parse_kv loads general metadata fields") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::metadata_string_equals;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  const char * tags[] = {"tag-a", "tag-b"};
  const char * languages[] = {"en", "fr"};
  REQUIRE(write_kv_string(file, "general.name", "test-model"));
  REQUIRE(write_kv_string(file, "general.author", "tester"));
  REQUIRE(write_kv_string(file, "general.version", "1.0"));
  REQUIRE(write_kv_string(file, "general.organization", "org"));
  REQUIRE(write_kv_string(file, "general.finetune", "finetune"));
  REQUIRE(write_kv_string(file, "general.basename", "base"));
  REQUIRE(write_kv_string(file, "general.description", "desc"));
  REQUIRE(write_kv_string(file, "general.quantized_by", "quant"));
  REQUIRE(write_kv_string(file, "general.size_label", "small"));
  REQUIRE(write_kv_string(file, "general.license", "mit"));
  REQUIRE(write_kv_string(file, "general.license.name", "MIT"));
  REQUIRE(write_kv_string(file, "general.license.link", "license"));
  REQUIRE(write_kv_string(file, "general.url", "url"));
  REQUIRE(write_kv_string(file, "general.doi", "doi"));
  REQUIRE(write_kv_string(file, "general.uuid", "uuid"));
  REQUIRE(write_kv_string(file, "general.repo_url", "repo"));
  REQUIRE(write_kv_string(file, "general.source.url", "source-url"));
  REQUIRE(write_kv_string(file, "general.source.doi", "source-doi"));
  REQUIRE(write_kv_string(file, "general.source.uuid", "source-uuid"));
  REQUIRE(write_kv_string(file, "general.source.repo_url", "source-repo"));
  REQUIRE(write_kv_string(file, "general.source.huggingface.repository", "hf-repo"));
  REQUIRE(write_kv_array_string(file, "general.tags", tags, 2));
  REQUIRE(write_kv_array_string(file, "general.languages", languages, 2));
  REQUIRE(write_kv_u32(file, "general.base_model.count", 1));
  REQUIRE(write_kv_string(file, "general.base_model.0.name", "base-name"));
  REQUIRE(write_kv_string(file, "general.base_model.0.author", "base-author"));
  REQUIRE(write_kv_string(file, "general.base_model.0.version", "base-ver"));
  REQUIRE(write_kv_string(file, "general.base_model.0.organization", "base-org"));
  REQUIRE(write_kv_string(file, "general.base_model.0.description", "base-desc"));
  REQUIRE(write_kv_string(file, "general.base_model.0.url", "base-url"));
  REQUIRE(write_kv_string(file, "general.base_model.0.doi", "base-doi"));
  REQUIRE(write_kv_string(file, "general.base_model.0.uuid", "base-uuid"));
  REQUIRE(write_kv_string(file, "general.base_model.0.repo_url", "base-repo"));
  REQUIRE(write_kv_u32(file, "general.dataset.count", 1));
  REQUIRE(write_kv_string(file, "general.dataset.0.name", "data-name"));
  REQUIRE(write_kv_string(file, "general.dataset.0.author", "data-author"));
  REQUIRE(write_kv_string(file, "general.dataset.0.version", "data-ver"));
  REQUIRE(write_kv_string(file, "general.dataset.0.organization", "data-org"));
  REQUIRE(write_kv_string(file, "general.dataset.0.description", "data-desc"));
  REQUIRE(write_kv_string(file, "general.dataset.0.url", "data-url"));
  REQUIRE(write_kv_string(file, "general.dataset.0.doi", "data-doi"));
  REQUIRE(write_kv_string(file, "general.dataset.0.uuid", "data-uuid"));
  REQUIRE(write_kv_string(file, "general.dataset.0.repo_url", "data-repo"));
  std::rewind(file);

  reader r{file};
  context ctx{};
  emel::model::data local_model{};
  int32_t err = EMEL_OK;
  CHECK(parse_kv(r, ctx, local_model, 43, err));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.name,
                               "test-model", 10));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.author,
                               "tester", 6));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.version,
                               "1.0", 3));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.organization,
                               "org", 3));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.finetune,
                               "finetune", 8));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.basename,
                               "base", 4));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.description,
                               "desc", 4));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.quantized_by,
                               "quant", 5));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.size_label,
                               "small", 5));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.license,
                               "mit", 3));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.license_name,
                               "MIT", 3));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.license_link,
                               "license", 7));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.url,
                               "url", 3));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.doi,
                               "doi", 3));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.uuid,
                               "uuid", 4));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.repo_url,
                               "repo", 4));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.source_url,
                               "source-url", 10));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.source_doi,
                               "source-doi", 10));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.source_uuid,
                               "source-uuid", 11));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.source_repo_url,
                               "source-repo", 11));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.source_hf_repo,
                               "hf-repo", 7));
  CHECK(local_model.meta.general_data.tag_count == 2);
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.tags[0],
                               "tag-a", 5));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.tags[1],
                               "tag-b", 5));
  CHECK(local_model.meta.general_data.language_count == 2);
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.languages[0],
                               "en", 2));
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.languages[1],
                               "fr", 2));
  CHECK(local_model.meta.general_data.base_model_count == 1);
  CHECK(metadata_string_equals(local_model.meta, local_model.meta.general_data.base_models[0].name,
                               "base-name", 9));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.base_models[0].author,
      "base-author", 11));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.base_models[0].version,
      "base-ver", 8));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.base_models[0].organization,
      "base-org", 8));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.base_models[0].description,
      "base-desc", 9));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.base_models[0].url,
      "base-url", 8));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.base_models[0].doi,
      "base-doi", 8));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.base_models[0].uuid,
      "base-uuid", 9));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.base_models[0].repo_url,
      "base-repo", 9));
  CHECK(local_model.meta.general_data.dataset_count == 1);
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.datasets[0].name,
      "data-name", 9));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.datasets[0].author,
      "data-author", 11));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.datasets[0].version,
      "data-ver", 8));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.datasets[0].organization,
      "data-org", 8));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.datasets[0].description,
      "data-desc", 9));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.datasets[0].url,
      "data-url", 8));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.datasets[0].doi,
      "data-doi", 8));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.datasets[0].uuid,
      "data-uuid", 9));
  CHECK(metadata_string_equals(
      local_model.meta, local_model.meta.general_data.datasets[0].repo_url,
      "data-repo", 9));
  std::fclose(file);
}

TEST_CASE("gguf parse_hparams loads core fields") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_hparams;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_kv_string(file, emel::parser::gguf::k_key_architecture, "llama"));
  REQUIRE(write_kv_u32(file, "llama.block_count", 2));
  REQUIRE(write_kv_u32(file, "llama.context_length", 2048));
  REQUIRE(write_kv_u32(file, "llama.embedding_length", 4096));
  REQUIRE(write_kv_u32(file, "llama.attention.head_count", 32));
  REQUIRE(write_kv_u32(file, "llama.attention.head_count_kv", 8));
  REQUIRE(write_kv_u32(file, "llama.rope.dimension_count", 128));
  std::rewind(file);

  reader r{file};
  context ctx{};
  emel::model::data local_model{};
  int32_t err = EMEL_OK;
  CHECK(parse_kv(r, ctx, local_model, 7, err));
  emel::parser::event::parse_model request{};
  request.model = &local_model;
  request.format_ctx = &ctx;
  CHECK(parse_hparams(request, &err));
  CHECK(local_model.params.n_layer == 2);
  CHECK(local_model.params.n_ctx == 2048);
  CHECK(local_model.params.n_embd == 4096);
  CHECK(local_model.params.n_head == 32);
  CHECK(local_model.params.n_head_kv == 8);
  CHECK(local_model.params.n_rot == 128);
  std::fclose(file);
}

TEST_CASE("gguf load_streamed supports split weights") {
  using emel::parser::gguf::map_parser;
  using emel::parser::gguf::load_streamed;

#if defined(_WIN32)
  return;
#else
  char base_path[] = "/tmp/emel_split_XXXXXX";
  const int fd = mkstemp(base_path);
  REQUIRE(fd != -1);
  close(fd);
  const std::string base = base_path;
  unlink(base_path);
  const std::string path0 = base + "-00001-of-00002.gguf";
  const std::string path1 = base + "-00002-of-00002.gguf";
  REQUIRE(write_minimal_split_gguf(path0.c_str(), 2, 0, 1.0f));
  REQUIRE(write_minimal_split_gguf(path1.c_str(), 2, 1, 2.0f));

  emel::model::data local_model{};
  emel::parser::gguf::context ctx{};
  emel::model::loader::event::load request{local_model};
  request.model_path = path0;
  request.format_ctx = &ctx;
  int32_t err = EMEL_OK;
  CHECK(map_parser(request, &err));
  std::array<uint8_t, 512> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();
  emel::model::weight_loader::event::load_weights load_req{
    .request_mmap = false,
    .request_direct_io = false,
    .check_tensors = false,
    .no_alloc = false,
    .mmap_supported = false,
    .direct_io_supported = false,
    .buffer_allocator_sm = nullptr,
    .map_mmap = nullptr,
    .load_streamed = load_streamed,
    .progress_callback = nullptr,
    .progress_user_data = nullptr,
    .loader_request = &request
  };
  uint64_t bytes_done = 0;
  uint64_t bytes_total = 0;
  CHECK(load_streamed(load_req, &bytes_done, &bytes_total, &err));
  CHECK(bytes_total == local_model.weights_size);
  CHECK(bytes_done == local_model.weights_size);
  const float * t0 = static_cast<const float *>(local_model.tensors[0].data);
  const float * t1 = static_cast<const float *>(local_model.tensors[1].data);
  CHECK(t0 != nullptr);
  CHECK(t1 != nullptr);
  CHECK(t0[0] == doctest::Approx(1.0f));
  CHECK(t1[0] == doctest::Approx(2.0f));
#endif
}

TEST_CASE("gguf parse_tensors rejects oversize name") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_tensors;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::tensor_type;

  emel::model::data model{};
  context ctx{};
  int32_t err = EMEL_OK;
  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);

  const uint64_t name_len = emel::parser::gguf::k_max_key_length;
  REQUIRE(std::fwrite(&name_len, 1, sizeof(name_len), file) == sizeof(name_len));
  std::array<char, emel::parser::gguf::k_max_key_length> name = {};
  std::memset(name.data(), 'a', name.size());
  REQUIRE(std::fwrite(name.data(), 1, name.size(), file) == name.size());
  REQUIRE(write_u32(file, 1));
  REQUIRE(write_i64(file, 32));
  const int32_t type_raw = static_cast<int32_t>(tensor_type::k_f32);
  REQUIRE(std::fwrite(&type_raw, 1, sizeof(type_raw), file) == sizeof(type_raw));
  REQUIRE(write_u64(file, 0));
  std::rewind(file);

  reader r{file};
  CHECK(!parse_tensors(r, ctx, model, 1, 0, 0, 0, err));
  std::fclose(file);
}

TEST_CASE("gguf helper functions cover branch paths") {
  using emel::parser::gguf::add_overflow_u64;
  using emel::parser::gguf::align_up_u64;
  using emel::parser::gguf::key_equals;
  using emel::parser::gguf::key_has_suffix;
  using emel::parser::gguf::mul_overflow_u64;

  CHECK(align_up_u64(64, 32) == 64);
  CHECK(align_up_u64(65, 32) == 96);
  CHECK(align_up_u64(17, 0) == 17);

  uint64_t out = 0;
  CHECK(!mul_overflow_u64(2, 3, out));
  CHECK(out == 6u);
  CHECK(mul_overflow_u64(std::numeric_limits<uint64_t>::max(), 2, out));

  CHECK(!add_overflow_u64(2, 3, out));
  CHECK(out == 5u);
  CHECK(add_overflow_u64(std::numeric_limits<uint64_t>::max(), 1, out));

  CHECK(key_equals("abc", 3, "abc"));
  CHECK(!key_equals("abc", 2, "abc"));

  uint64_t prefix_len = 0;
  CHECK(key_has_suffix("foo.bar", 7, ".bar", prefix_len));
  CHECK(prefix_len == 3);
  CHECK(!key_has_suffix("foo", 3, ".bar", prefix_len));
  CHECK(!key_has_suffix("foo.baz", 7, ".bar", prefix_len));
}

TEST_CASE("gguf helper functions cover indexed and prefix checks") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::key_has_suffix_primary;
  using emel::parser::gguf::parse_indexed_key;
  using emel::parser::gguf::prefix_is_primary_arch;

  context ctx{};
  uint64_t prefix_len = 0;
  const char * block_key = "llama.block_count";
  CHECK(key_has_suffix_primary(ctx, block_key, std::strlen(block_key),
                               ".block_count", prefix_len));
  CHECK(prefix_len == 5);
  CHECK(prefix_is_primary_arch(ctx, "llama.block_count", 5));
  CHECK_FALSE(prefix_is_primary_arch(ctx, "llama.block_count", 6));

  std::memcpy(ctx.architecture.data(), "llama", 5);
  ctx.architecture_len = 5;
  CHECK(key_has_suffix_primary(ctx, block_key, std::strlen(block_key),
                               ".block_count", prefix_len));
  const char * other_block = "other.block_count";
  CHECK_FALSE(key_has_suffix_primary(ctx, other_block, std::strlen(other_block),
                                     ".block_count", prefix_len));

  uint32_t index = 0;
  const char * indexed_key = "general.base_model.12.name";
  CHECK(parse_indexed_key(indexed_key, std::strlen(indexed_key),
                          "general.base_model.", ".name", index));
  CHECK(index == 12);
  const char * no_index = "general.base_model.name";
  CHECK_FALSE(parse_indexed_key(no_index, std::strlen(no_index),
                                "general.base_model.", ".name", index));
  const char * bad_digit = "general.base_model.1a.name";
  CHECK_FALSE(parse_indexed_key(bad_digit, std::strlen(bad_digit),
                                "general.base_model.", ".name", index));
  const char * long_index = "general.base_model.12345678901.name";
  CHECK_FALSE(parse_indexed_key(long_index, std::strlen(long_index),
                                "general.base_model.", ".name", index));
  const char * bad_suffix = "general.base_model.0.names";
  CHECK_FALSE(parse_indexed_key(bad_suffix, std::strlen(bad_suffix),
                                "general.base_model.", ".name", index));
  const char * wrong_prefix = "other.base_model.0.name";
  CHECK_FALSE(parse_indexed_key(wrong_prefix, std::strlen(wrong_prefix),
                                "general.base_model.", ".name", index));
}

TEST_CASE("gguf read_and_discard_string handles short reads") {
  using emel::parser::gguf::read_and_discard_string;
  using emel::parser::gguf::reader;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    reader r{file};
    CHECK_FALSE(read_and_discard_string(r));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 4;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char str[] = {'t', 'e', 's', 't'};
    REQUIRE(std::fwrite(str, 1, sizeof(str), file) == sizeof(str));
    std::rewind(file);
    reader r{file};
    CHECK(read_and_discard_string(r));
    std::fclose(file);
  }
}

TEST_CASE("gguf read_string and skip_value cover branches") {
  using emel::parser::gguf::reader;
  using emel::parser::gguf::skip_value;
  using emel::parser::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    std::array<uint8_t, 128> zeros = {};
    REQUIRE(std::fwrite(zeros.data(), 1, zeros.size(), file) == zeros.size());
    std::rewind(file);
    reader r{file};
    CHECK(skip_value(r, value_type::k_u8, 4));
    CHECK(skip_value(r, value_type::k_i8, 4));
    CHECK(skip_value(r, value_type::k_bool, 4));
    CHECK(skip_value(r, value_type::k_u16, 2));
    CHECK(skip_value(r, value_type::k_i16, 2));
    CHECK(skip_value(r, value_type::k_u32, 2));
    CHECK(skip_value(r, value_type::k_i32, 2));
    CHECK(skip_value(r, value_type::k_f32, 2));
    CHECK(skip_value(r, value_type::k_u64, 1));
    CHECK(skip_value(r, value_type::k_i64, 1));
    CHECK(skip_value(r, value_type::k_f64, 1));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 5;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char str[] = {'h', 'e', 'l', 'l', 'o'};
    REQUIRE(std::fwrite(str, 1, sizeof(str), file) == sizeof(str));
    std::rewind(file);
    reader r{file};
    uint64_t out_len = 0;
    std::array<char, 8> out = {};
    CHECK(r.read_string(out.data(), out.size(), out_len));
    CHECK(out_len == len);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 8;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    std::array<char, 8> str = {'1','2','3','4','5','6','7','8'};
    REQUIRE(std::fwrite(str.data(), 1, str.size(), file) == str.size());
    std::rewind(file);
    reader r{file};
    uint64_t out_len = 0;
    std::array<char, 4> out = {};
    CHECK(!r.read_string(out.data(), out.size(), out_len));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 4;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char str[] = {'t', 'e', 's', 't'};
    REQUIRE(std::fwrite(str, 1, sizeof(str), file) == sizeof(str));
    std::rewind(file);
    reader r{file};
    uint64_t out_len = 0;
    CHECK(r.read_string(nullptr, 0, out_len));
    CHECK(out_len == len);
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_tensors validates counts") {
  emel::model::data model{};
  emel::parser::gguf::context ctx{};
  int32_t err = EMEL_OK;
  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  emel::parser::gguf::reader r{file};
  CHECK(!emel::parser::gguf::parse_tensors(r, ctx, model, -1, 0, 0, 0, err));
  std::fclose(file);
}

TEST_CASE("gguf parse_tensors rejects invalid entries") {
  using emel::parser::gguf::parse_tensors;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::tensor_type;

  emel::model::data model{};
  emel::parser::gguf::context ctx{};
  int32_t err = EMEL_OK;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, ""));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, 1));
    const int32_t type_raw = static_cast<int32_t>(tensor_type::k_f32);
    REQUIRE(std::fwrite(&type_raw, 1, sizeof(type_raw), file) == sizeof(type_raw));
    REQUIRE(write_u64(file, 0));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, 0, 0, 0, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, emel::parser::gguf::k_max_dims + 1));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, 0, 0, 0, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, -1));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, 0, 0, 0, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, 1));
    const int32_t type_raw = static_cast<int32_t>(tensor_type::k_count) + 1;
    REQUIRE(std::fwrite(&type_raw, 1, sizeof(type_raw), file) == sizeof(type_raw));
    REQUIRE(write_u64(file, 0));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, 0, 0, 0, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, 31));
    const int32_t type_raw = static_cast<int32_t>(tensor_type::k_q4_0);
    REQUIRE(std::fwrite(&type_raw, 1, sizeof(type_raw), file) == sizeof(type_raw));
    REQUIRE(write_u64(file, 0));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, 0, 0, 0, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, 32));
    const int32_t type_raw = static_cast<int32_t>(tensor_type::k_f32);
    REQUIRE(std::fwrite(&type_raw, 1, sizeof(type_raw), file) == sizeof(type_raw));
    REQUIRE(write_u64(file, 8));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, 0, 0, 0, err));
    std::fclose(file);
  }
}

TEST_CASE("gguf map_parser handles invalid inputs") {
  emel::model::data model{};
  emel::parser::gguf::context ctx{};

  {
    emel::model::loader::event::load request{model};
    int32_t err = EMEL_OK;
    CHECK(!emel::parser::gguf::map_parser(request, &err));
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    emel::model::loader::event::load request{model};
    request.format_ctx = &ctx;
    request.model_path = "";
    int32_t err = EMEL_OK;
    CHECK(!emel::parser::gguf::map_parser(request, &err));
    CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const char bad_magic[4] = {'B', 'A', 'D', '!'};
    REQUIRE(std::fwrite(bad_magic, 1, sizeof(bad_magic), file) == sizeof(bad_magic));
    std::rewind(file);

    emel::model::loader::event::load request{model};
    request.format_ctx = &ctx;
    request.file_handle = file;
    int32_t err = EMEL_OK;
    CHECK(!emel::parser::gguf::map_parser(request, &err));
    CHECK(err == EMEL_ERR_FORMAT_UNSUPPORTED);
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_architecture and hparams handle invalid ctx") {
  emel::model::data model{};
  emel::parser::gguf::context ctx{};
  emel::parser::event::parse_model parse_request{};
  parse_request.model = &model;
  parse_request.format_ctx = &ctx;

  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::parse_architecture(parse_request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  std::memcpy(ctx.architecture.data(), "llama", 5);
  ctx.architecture_len = static_cast<uint32_t>(model.architecture_name.size());
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::parse_architecture(parse_request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  ctx.architecture_len = 5;
  err = EMEL_OK;
  CHECK(emel::parser::gguf::parse_architecture(parse_request, &err));
  CHECK(std::strcmp(model.architecture_name.data(), "llama") == 0);

  err = EMEL_OK;
  CHECK(!emel::parser::gguf::parse_hparams(parse_request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  ctx.block_count = 3;
  model.params.n_ctx = 1;
  model.params.n_embd = 1;
  err = EMEL_OK;
  CHECK(emel::parser::gguf::parse_hparams(parse_request, &err));
  CHECK(model.n_layers == 3);
}

TEST_CASE("gguf map_tensors and map_layers validate counts") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  int32_t err = EMEL_OK;

  CHECK(!emel::parser::gguf::map_tensors({.model = nullptr}, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::parser::event::parse_model parse_request{};
  parse_request.model = &model;
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::map_tensors(parse_request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  model.n_tensors = 1;
  err = EMEL_OK;
  CHECK(emel::parser::gguf::map_tensors(parse_request, &err));

  err = EMEL_OK;
  CHECK(!emel::parser::gguf::map_layers(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);

  model.n_layers = 2;
  err = EMEL_OK;
  CHECK(emel::parser::gguf::map_layers(request, &err));
}

TEST_CASE("gguf parse_kv handles i32 alignment and string arrays") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, emel::parser::gguf::k_key_alignment));
    const int32_t type = static_cast<int32_t>(value_type::k_i32);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    const int32_t alignment = 64;
    REQUIRE(std::fwrite(&alignment, 1, sizeof(alignment), file) == sizeof(alignment));
    std::rewind(file);

    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(parse_kv(r, ctx, model, 1, err));
    CHECK(ctx.alignment == 64);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "strings"));
    const int32_t type = static_cast<int32_t>(value_type::k_array);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    const int32_t elem_type = static_cast<int32_t>(value_type::k_string);
    REQUIRE(std::fwrite(&elem_type, 1, sizeof(elem_type), file) == sizeof(elem_type));
    const uint64_t count = 2;
    REQUIRE(write_u64(file, count));
    REQUIRE(write_string(file, "a"));
    REQUIRE(write_string(file, "b"));
    std::rewind(file);

    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(parse_kv(r, ctx, model, 1, err));
    std::fclose(file);
  }
}

TEST_CASE("gguf map_parser rejects long path") {
  emel::model::data model{};
  emel::parser::gguf::context ctx{};
  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;

  std::array<char, emel::parser::gguf::k_max_path_length + 1> buf = {};
  std::memset(buf.data(), 'a', buf.size());
  request.model_path = std::string_view{buf.data(), buf.size()};

  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gguf parse_tensors rejects oversize tensor count") {
  emel::model::data model{};
  emel::parser::gguf::context ctx{};
  int32_t err = EMEL_OK;
  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  emel::parser::gguf::reader r{file};
  CHECK(!emel::parser::gguf::parse_tensors(r, ctx, model,
    emel::model::data::k_max_tensors + 1, 0, 0, 0, err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
  std::fclose(file);
}

TEST_CASE("gguf parse_kv handles architecture type errors") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_string(file, emel::parser::gguf::k_key_architecture));
  const int32_t type = static_cast<int32_t>(value_type::k_u32);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  const uint32_t value = 7;
  REQUIRE(write_u32(file, value));
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(!parse_kv(r, ctx, model, 1, err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
  std::fclose(file);
}

TEST_CASE("gguf parse_header rejects zero low bits") {
  using emel::parser::gguf::parse_header;
  using emel::parser::gguf::reader;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  const char magic[4] = {'G', 'G', 'U', 'F'};
  REQUIRE(std::fwrite(magic, 1, sizeof(magic), file) == sizeof(magic));
  const uint32_t version = 0x00010000u;
  REQUIRE(std::fwrite(&version, 1, sizeof(version), file) == sizeof(version));
  const int64_t n_tensors = 0;
  const int64_t n_kv = 0;
  REQUIRE(std::fwrite(&n_tensors, 1, sizeof(n_tensors), file) == sizeof(n_tensors));
  REQUIRE(std::fwrite(&n_kv, 1, sizeof(n_kv), file) == sizeof(n_kv));
  std::rewind(file);

  reader r{file};
  emel::parser::gguf::context ctx{};
  int64_t out_tensors = 0;
  int64_t out_kv = 0;
  CHECK(!parse_header(r, ctx, out_tensors, out_kv));
  std::fclose(file);
}

TEST_CASE("gguf parse_kv reports invalid architecture string length") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_string(file, emel::parser::gguf::k_key_architecture));
  const int32_t type = static_cast<int32_t>(value_type::k_string);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  const uint64_t len = emel::parser::gguf::k_max_architecture;
  REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
  std::array<char, emel::parser::gguf::k_max_architecture> name = {};
  REQUIRE(std::fwrite(name.data(), 1, name.size(), file) == name.size());
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(!parse_kv(r, ctx, model, 1, err));
  CHECK(err == EMEL_ERR_PARSE_FAILED);
  std::fclose(file);
}

TEST_CASE("gguf parse_kv handles array string read failure") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_string(file, "bad.strings"));
  const int32_t type = static_cast<int32_t>(value_type::k_array);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  const int32_t elem_type = static_cast<int32_t>(value_type::k_string);
  REQUIRE(std::fwrite(&elem_type, 1, sizeof(elem_type), file) == sizeof(elem_type));
  const uint64_t count = 1;
  REQUIRE(write_u64(file, count));
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(!parse_kv(r, ctx, model, 1, err));
  CHECK(err == EMEL_ERR_PARSE_FAILED);
  std::fclose(file);
}

TEST_CASE("gguf parse_kv ignores oversize block_count prefix") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  constexpr size_t prefix_len = 70;
  const char suffix[] = ".block_count";
  constexpr size_t suffix_len = sizeof(suffix) - 1;
  std::array<char, 96> key = {};
  std::memset(key.data(), 'a', key.size());
  std::memcpy(key.data() + prefix_len, suffix, suffix_len);
  const uint64_t key_len = prefix_len + suffix_len;
  REQUIRE(std::fwrite(&key_len, 1, sizeof(key_len), file) == sizeof(key_len));
  REQUIRE(std::fwrite(key.data(), 1, key_len, file) == key_len);
  const int32_t type = static_cast<int32_t>(value_type::k_u32);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  REQUIRE(write_u32(file, 3));
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(parse_kv(r, ctx, model, 1, err));
  CHECK(!ctx.pending_block_count_valid);
  std::fclose(file);
}

#if !defined(_WIN32)
TEST_CASE("gguf map_mmap handles invalid requests") {
  emel::model::data model{};
  emel::parser::gguf::context ctx{};

  emel::model::weight_loader::event::load_weights load_req{};
  uint64_t done = 0;
  uint64_t total = 0;
  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::model::loader::event::load request{model};
  request.format_ctx = nullptr;
  load_req.loader_request = &request;
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  request.format_ctx = &ctx;
  request.model_path = "";
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  ctx.mapped_data = &ctx;
  ctx.mapped_size = 8;
  ctx.mapped_splits[0] = &ctx;
  ctx.mapped_sizes[0] = 8;
  ctx.mapped_count = 1;
  ctx.data_offset = 8;
  ctx.data_size = 16;
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
  ctx.mapped_data = nullptr;
  ctx.mapped_size = 0;
  ctx.mapped_splits[0] = nullptr;
  ctx.mapped_sizes[0] = 0;
  ctx.mapped_count = 0;
}
#endif

TEST_CASE("gguf load_streamed reports invalid requests") {
  emel::model::data model{};
  emel::parser::gguf::context ctx{};

  emel::model::weight_loader::event::load_weights load_req{};
  uint64_t done = 0;
  uint64_t total = 0;
  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::model::loader::event::load request{model};
  request.format_ctx = nullptr;
  load_req.loader_request = &request;
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  request.format_ctx = &ctx;
  request.weights_buffer = nullptr;
  request.weights_buffer_size = 0;
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gguf load_streamed handles fseek and fread failures") {
  emel::model::data model{};
  emel::parser::gguf::context ctx{};

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  ctx.file = file;
  ctx.data_offset = 0;
  ctx.data_size = 8;

  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;
  std::array<uint8_t, 8> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };

  std::fclose(file);
  uint64_t done = 0;
  uint64_t total = 0;
  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_IO);

  ctx.file = std::tmpfile();
  REQUIRE(ctx.file != nullptr);
  std::array<uint8_t, 4> small_data = {1, 2, 3, 4};
  REQUIRE(std::fwrite(small_data.data(), 1, small_data.size(), ctx.file) == small_data.size());
  std::rewind(ctx.file);
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_IO);
  std::fclose(ctx.file);
  ctx.file = nullptr;
}

TEST_CASE("gguf reader rejects null file") {
  using emel::parser::gguf::read_key;
  using emel::parser::gguf::reader;

  reader r{};
  uint32_t value = 0;
  CHECK_FALSE(r.read(value));
  CHECK_FALSE(r.skip(4));
  uint64_t out_len = 0;
  std::array<char, 4> out = {};
  CHECK_FALSE(r.read_string(out.data(), out.size(), out_len));
  CHECK_FALSE(read_key(r, out.data(), out.size(), out_len));
}

TEST_CASE("gguf reader handles short strings") {
  using emel::parser::gguf::read_key;
  using emel::parser::gguf::reader;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 4;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char data[2] = {'a', 'b'};
    REQUIRE(std::fwrite(data, 1, sizeof(data), file) == sizeof(data));
    std::rewind(file);
    reader r{file};
    uint64_t out_len = 0;
    std::array<char, 8> out = {};
    CHECK(!r.read_string(out.data(), out.size(), out_len));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const uint64_t len = 4;
    REQUIRE(std::fwrite(&len, 1, sizeof(len), file) == sizeof(len));
    const char data[2] = {'a', 'b'};
    REQUIRE(std::fwrite(data, 1, sizeof(data), file) == sizeof(data));
    std::rewind(file);
    reader r{file};
    uint64_t out_len = 0;
    std::array<char, 8> out = {};
    CHECK(!read_key(r, out.data(), out.size(), out_len));
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_u32_value handles read failures") {
  using emel::parser::gguf::parse_u32_value;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  reader r{file};
  uint32_t out = 0;
  CHECK(!parse_u32_value(r, value_type::k_i32, out));
  std::rewind(file);
  CHECK(!parse_u32_value(r, value_type::k_u16, out));
  std::fclose(file);
}

TEST_CASE("gguf parse_header handles short reads") {
  using emel::parser::gguf::parse_header;
  using emel::parser::gguf::reader;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    reader r{file};
    emel::parser::gguf::context ctx{};
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    CHECK(!parse_header(r, ctx, n_tensors, n_kv));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const char magic[4] = {'G', 'G', 'U', 'F'};
    REQUIRE(std::fwrite(magic, 1, sizeof(magic), file) == sizeof(magic));
    std::rewind(file);
    reader r{file};
    emel::parser::gguf::context ctx{};
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    CHECK(!parse_header(r, ctx, n_tensors, n_kv));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const char magic[4] = {'G', 'G', 'U', 'F'};
    REQUIRE(std::fwrite(magic, 1, sizeof(magic), file) == sizeof(magic));
    const uint32_t version = emel::parser::gguf::k_gguf_version;
    REQUIRE(std::fwrite(&version, 1, sizeof(version), file) == sizeof(version));
    std::rewind(file);
    reader r{file};
    emel::parser::gguf::context ctx{};
    int64_t n_tensors = 0;
    int64_t n_kv = 0;
    CHECK(!parse_header(r, ctx, n_tensors, n_kv));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    const char magic[4] = {'G', 'G', 'U', 'F'};
    REQUIRE(std::fwrite(magic, 1, sizeof(magic), file) == sizeof(magic));
    const uint32_t version = emel::parser::gguf::k_gguf_version;
    REQUIRE(std::fwrite(&version, 1, sizeof(version), file) == sizeof(version));
    const int64_t n_tensors = 1;
    REQUIRE(std::fwrite(&n_tensors, 1, sizeof(n_tensors), file) == sizeof(n_tensors));
    std::rewind(file);
    reader r{file};
    emel::parser::gguf::context ctx{};
    int64_t out_tensors = 0;
    int64_t out_kv = 0;
    CHECK(!parse_header(r, ctx, out_tensors, out_kv));
    std::fclose(file);
  }
}

TEST_CASE("gguf store_name rejects overflow conditions") {
  using emel::parser::gguf::store_name;

  emel::model::data model{};
  uint32_t offset = 0;
  model.name_bytes_used = static_cast<uint32_t>(model.name_storage.size()) + 1;
  CHECK(!store_name(model, "a", 1, offset));

  model.name_bytes_used = static_cast<uint32_t>(model.name_storage.size() - 1);
  CHECK(!store_name(model, "ab", 2, offset));
}

TEST_CASE("gguf helper functions cover zero multiplication") {
  using emel::parser::gguf::mul_overflow_u64;

  uint64_t out = 1;
  CHECK(!mul_overflow_u64(0, 10, out));
  CHECK(out == 0);
  CHECK(!mul_overflow_u64(10, 0, out));
  CHECK(out == 0);
}

TEST_CASE("gguf parse_kv handles read failures") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, model, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "key"));
    std::rewind(file);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, model, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "key"));
    const int32_t type = static_cast<int32_t>(value_type::k_array);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    std::rewind(file);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, model, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "key"));
    const int32_t type = static_cast<int32_t>(value_type::k_array);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    const int32_t elem_type = static_cast<int32_t>(value_type::k_u32);
    REQUIRE(std::fwrite(&elem_type, 1, sizeof(elem_type), file) == sizeof(elem_type));
    std::rewind(file);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, model, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_kv reports split metadata parse failures") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, emel::parser::gguf::k_key_split_count));
    const int32_t type = static_cast<int32_t>(value_type::k_f32);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    std::rewind(file);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, model, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, emel::parser::gguf::k_key_split_no));
    const int32_t type = static_cast<int32_t>(value_type::k_f32);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    std::rewind(file);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, model, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, emel::parser::gguf::k_key_split_tensors));
    const int32_t type = static_cast<int32_t>(value_type::k_f32);
    REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
    std::rewind(file);
    reader r{file};
    context ctx{};
    int32_t err = EMEL_OK;
    CHECK(!parse_kv(r, ctx, model, 1, err));
    CHECK(err == EMEL_ERR_PARSE_FAILED);
    std::fclose(file);
  }
}

TEST_CASE("gguf parse_kv reports block_count parse failures") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_string(file, "llama.block_count"));
  const int32_t type = static_cast<int32_t>(value_type::k_f32);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  std::rewind(file);
  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(!parse_kv(r, ctx, model, 1, err));
  CHECK(err == EMEL_ERR_PARSE_FAILED);
  std::fclose(file);
}

TEST_CASE("gguf parse_kv reports alignment parse failures") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_string(file, emel::parser::gguf::k_key_alignment));
  const int32_t type = static_cast<int32_t>(value_type::k_f32);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  std::rewind(file);

  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(!parse_kv(r, ctx, model, 1, err));
  CHECK(err == EMEL_ERR_PARSE_FAILED);
  std::fclose(file);
}

TEST_CASE("gguf parse_kv reports unknown skip failures") {
  using emel::parser::gguf::context;
  using emel::parser::gguf::parse_kv;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::value_type;

  std::FILE * file = std::tmpfile();
  REQUIRE(file != nullptr);
  REQUIRE(write_string(file, "unknown"));
  const int32_t type = static_cast<int32_t>(value_type::k_string);
  REQUIRE(std::fwrite(&type, 1, sizeof(type), file) == sizeof(type));
  std::rewind(file);
  reader r{file};
  context ctx{};
  int32_t err = EMEL_OK;
  CHECK(!parse_kv(r, ctx, model, 1, err));
  CHECK(err == EMEL_ERR_PARSE_FAILED);
  std::fclose(file);
}

TEST_CASE("gguf parse_tensors handles read failures") {
  using emel::parser::gguf::parse_tensors;
  using emel::parser::gguf::reader;
  using emel::parser::gguf::tensor_type;

  emel::model::data model{};
  emel::parser::gguf::context ctx{};
  int32_t err = EMEL_OK;

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, 0, 0, 0, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, 0, 0, 0, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, 32));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, 0, 0, 0, err));
    std::fclose(file);
  }

  {
    std::FILE * file = std::tmpfile();
    REQUIRE(file != nullptr);
    REQUIRE(write_string(file, "tensor"));
    REQUIRE(write_u32(file, 1));
    REQUIRE(write_i64(file, 32));
    const int32_t type_raw = static_cast<int32_t>(tensor_type::k_f32);
    REQUIRE(std::fwrite(&type_raw, 1, sizeof(type_raw), file) == sizeof(type_raw));
    std::rewind(file);
    reader r{file};
    CHECK(!parse_tensors(r, ctx, model, 1, 0, 0, 0, err));
    std::fclose(file);
  }
}

TEST_CASE("gguf compute_tensor_size rejects invalid type") {
  using emel::parser::gguf::compute_tensor_size;
  using emel::parser::gguf::tensor_type;

  uint64_t out = 0;
  std::array<int64_t, emel::parser::gguf::k_max_dims> dims = {1, 1, 1, 1};
  const auto invalid = static_cast<tensor_type>(static_cast<int32_t>(tensor_type::k_count));
  CHECK(!compute_tensor_size(dims, invalid, out));
}

TEST_CASE("gguf map_parser reports file open failure") {
  emel::model::data model{};
  emel::parser::gguf::context ctx{};
  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;

  char path[128] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  std::remove(path);
  request.model_path = path;

  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_ERR_IO);
}

TEST_CASE("gguf validate_structure handles null context") {
  emel::model::data model{};
  emel::model::loader::event::load request{model};
  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::validate_structure(request, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gguf parse_architecture handles null model") {
  emel::parser::gguf::context ctx{};
  emel::parser::event::parse_model parse_request{
    .model = nullptr,
    .format_ctx = &ctx
  };
  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::parse_architecture(parse_request, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gguf map_architecture accepts empty whitelist") {
  emel::model::data model{};
  std::memcpy(model.architecture_name.data(), "llama", 6);
  emel::parser::event::parse_model parse_request{
    .model = &model,
    .architectures = nullptr,
    .n_architectures = 0
  };
  int32_t err = EMEL_OK;
  CHECK(emel::parser::gguf::map_architecture(parse_request, &err));
  CHECK(err == EMEL_OK);
}

TEST_CASE("gguf validate_structure rejects empty weights") {
  emel::model::data model{};
  emel::parser::gguf::context ctx{};
  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;
  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::validate_structure(request, &err));
  CHECK(err == EMEL_ERR_MODEL_INVALID);
}

#if !defined(_WIN32)
TEST_CASE("gguf map_mmap reports file open failure") {
  emel::model::data model{};
  emel::parser::gguf::context ctx{};
  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;

  char path[128] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  std::remove(path);
  request.model_path = path;

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_IO);
}
#endif

TEST_CASE("gguf load_streamed reports path errors") {
  emel::model::data model{};
  emel::parser::gguf::context ctx{};
  ctx.data_size = 0;

  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;
  std::array<uint8_t, 1> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };

  uint64_t done = 0;
  uint64_t total = 0;
  int32_t err = EMEL_OK;
  request.model_path = "";
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  char path[128] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  std::remove(path);
  request.model_path = path;
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_IO);
}

TEST_CASE("gguf load_streamed handles owned files") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  int32_t err = EMEL_OK;

  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);
  ctx.file = nullptr;

  std::array<uint8_t, 128> buffer = {};
  request.weights_buffer = buffer.data();
  request.weights_buffer_size = buffer.size();

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };
  uint64_t done = 0;
  uint64_t total = 0;
  CHECK(emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_OK);
  CHECK(done == 128);
  CHECK(total == 128);

  char short_path[1024] = {};
  CHECK(make_temp_path(short_path, sizeof(short_path)));
  std::FILE * short_file = std::fopen(short_path, "wb");
  REQUIRE(short_file != nullptr);
  const uint32_t marker = 0xabcdef01u;
  REQUIRE(std::fwrite(&marker, 1, sizeof(marker), short_file) == sizeof(marker));
  std::fclose(short_file);

  std::array<uint8_t, 16> small_buffer = {};
  request.model_path = short_path;
  request.weights_buffer = small_buffer.data();
  request.weights_buffer_size = small_buffer.size();
  ctx.file = nullptr;
  ctx.data_offset = 0;
  ctx.data_size = 16;
  model.weights_size = 16;
  model.weights_split_count = 1;
  model.weights_split_sizes[0] = 16;
  model.weights_split_offsets[0] = 0;
  ctx.split_count = 1;
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::load_streamed(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_IO);

  std::remove(short_path);

  std::remove(path);
}

#if !defined(_WIN32)
TEST_CASE("gguf init_mappings maps files") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;

  int32_t err = EMEL_OK;
  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };
  CHECK(emel::parser::gguf::init_mappings(load_req, &err));
  CHECK(err == EMEL_OK);
  CHECK(ctx.mapped_count == request.model_data.weights_split_count);

  std::remove(path);
}

TEST_CASE("gguf clean_up_weights unmaps unused ranges") {
  const long page = sysconf(_SC_PAGESIZE);
  REQUIRE(page > 0);
  const size_t map_size = static_cast<size_t>(page) * 3;
  void * mapping = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  REQUIRE(mapping != MAP_FAILED);

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  ctx.mapped_splits[0] = mapping;
  ctx.mapped_sizes[0] = map_size;
  ctx.mapped_count = 1;

  model.weights_split_count = 1;
  model.weights_split_offsets[0] = 0;
  model.weights_split_sizes[0] = map_size;
  model.weights_size = map_size;
  model.n_tensors = 1;
  model.tensors[0].file_index = 0;
  model.tensors[0].file_offset = static_cast<uint64_t>(page);
  model.tensors[0].data_size = static_cast<uint64_t>(page);

  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };
  int32_t err = EMEL_OK;
  CHECK(emel::parser::gguf::clean_up_weights(load_req, &err));
  CHECK(err == EMEL_OK);
}

TEST_CASE("gguf init_mappings reports invalid requests") {
  emel::model::weight_loader::event::load_weights load_req{};
  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::init_mappings(load_req, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::model::data model{};
  emel::model::loader::event::load request{model};
  load_req.loader_request = &request;
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::init_mappings(load_req, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gguf validate_weights handles request errors") {
  emel::model::weight_loader::event::load_weights load_req{};
  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::validate_weights(load_req, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::model::data model{};
  emel::parser::gguf::context ctx{};
  emel::model::loader::event::load request{model};
  request.format_ctx = &ctx;
  request.check_tensors = false;
  load_req.loader_request = &request;
  load_req.check_tensors = false;
  err = EMEL_OK;
  CHECK(emel::parser::gguf::validate_weights(load_req, &err));
  CHECK(err == EMEL_OK);
}

TEST_CASE("gguf clean_up_weights reports invalid requests") {
  emel::model::weight_loader::event::load_weights load_req{};
  int32_t err = EMEL_OK;
  CHECK(!emel::parser::gguf::clean_up_weights(load_req, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);

  emel::model::data model{};
  emel::model::loader::event::load request{model};
  load_req.loader_request = &request;
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::clean_up_weights(load_req, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("gguf map_mmap reports progress cancellation") {
  char path[1024] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  CHECK(write_minimal_gguf(path));

  emel::model::data model = {};
  emel::parser::gguf::context ctx = {};
  emel::model::loader::event::load request{model};
  request.model_path = path;
  request.format_ctx = &ctx;
  int32_t err = EMEL_OK;
  CHECK(emel::parser::gguf::map_parser(request, &err));
  CHECK(err == EMEL_OK);

  emel::model::weight_loader::event::load_weights load_req{
    .loader_request = &request
  };
  load_req.progress_callback = [](float, void *) {
    return false;
  };
  uint64_t done = 0;
  uint64_t total = 0;
  err = EMEL_OK;
  CHECK(!emel::parser::gguf::map_mmap(load_req, &done, &total, &err));
  CHECK(err == EMEL_ERR_BACKEND);

  std::remove(path);
}
#endif

TEST_CASE("gguf validate_row_data covers quant types") {
  using namespace emel::parser::gguf;

  CHECK(!validate_row_data(tensor_type::k_f32, nullptr, 4));

  const float nan_value = std::numeric_limits<float>::quiet_NaN();
  CHECK(!validate_row_data(tensor_type::k_f32, &nan_value, sizeof(nan_value)));

  const uint16_t f16_inf = 0x7c00u;
  CHECK(!validate_row_data(tensor_type::k_f16, &f16_inf, sizeof(f16_inf)));

  const uint16_t bf16_inf = 0x7f80u;
  CHECK(!validate_row_data(tensor_type::k_bf16, &bf16_inf, sizeof(bf16_inf)));

  block_mxfp4 mxfp4_invalid{};
  mxfp4_invalid.e = 0xffu;
  CHECK(!validate_row_data(tensor_type::k_mxfp4, &mxfp4_invalid, sizeof(mxfp4_invalid)));

  block_q4_0 q4_0{};
  CHECK(validate_row_data(tensor_type::k_q4_0, &q4_0, sizeof(q4_0)));

  block_q4_1 q4_1{};
  CHECK(validate_row_data(tensor_type::k_q4_1, &q4_1, sizeof(q4_1)));

  block_q5_0 q5_0{};
  CHECK(validate_row_data(tensor_type::k_q5_0, &q5_0, sizeof(q5_0)));

  block_q5_1 q5_1{};
  CHECK(validate_row_data(tensor_type::k_q5_1, &q5_1, sizeof(q5_1)));

  block_q8_0 q8_0{};
  CHECK(validate_row_data(tensor_type::k_q8_0, &q8_0, sizeof(q8_0)));

  block_q8_1 q8_1{};
  CHECK(validate_row_data(tensor_type::k_q8_1, &q8_1, sizeof(q8_1)));

  block_mxfp4 mxfp4{};
  CHECK(validate_row_data(tensor_type::k_mxfp4, &mxfp4, sizeof(mxfp4)));

  block_q2_k q2_k{};
  CHECK(validate_row_data(tensor_type::k_q2_k, &q2_k, sizeof(q2_k)));

  block_q3_k q3_k{};
  CHECK(validate_row_data(tensor_type::k_q3_k, &q3_k, sizeof(q3_k)));

  block_q4_k q4_k{};
  CHECK(validate_row_data(tensor_type::k_q4_k, &q4_k, sizeof(q4_k)));

  block_q5_k q5_k{};
  CHECK(validate_row_data(tensor_type::k_q5_k, &q5_k, sizeof(q5_k)));

  block_q6_k q6_k{};
  CHECK(validate_row_data(tensor_type::k_q6_k, &q6_k, sizeof(q6_k)));

  block_q8_k q8_k{};
  q8_k.d = 0.0f;
  CHECK(validate_row_data(tensor_type::k_q8_k, &q8_k, sizeof(q8_k)));

  block_tq1_0 tq1_0{};
  CHECK(validate_row_data(tensor_type::k_tq1_0, &tq1_0, sizeof(tq1_0)));

  block_tq2_0 tq2_0{};
  CHECK(validate_row_data(tensor_type::k_tq2_0, &tq2_0, sizeof(tq2_0)));

  block_iq1_s iq1_s{};
  CHECK(validate_row_data(tensor_type::k_iq1_s, &iq1_s, sizeof(iq1_s)));

  block_iq1_m iq1_m{};
  CHECK(validate_row_data(tensor_type::k_iq1_m, &iq1_m, sizeof(iq1_m)));

  block_iq2_xxs iq2_xxs{};
  CHECK(validate_row_data(tensor_type::k_iq2_xxs, &iq2_xxs, sizeof(iq2_xxs)));

  block_iq2_xs iq2_xs{};
  CHECK(validate_row_data(tensor_type::k_iq2_xs, &iq2_xs, sizeof(iq2_xs)));

  block_iq2_s iq2_s{};
  CHECK(validate_row_data(tensor_type::k_iq2_s, &iq2_s, sizeof(iq2_s)));

  block_iq3_xxs iq3_xxs{};
  CHECK(validate_row_data(tensor_type::k_iq3_xxs, &iq3_xxs, sizeof(iq3_xxs)));

  block_iq3_s iq3_s{};
  CHECK(validate_row_data(tensor_type::k_iq3_s, &iq3_s, sizeof(iq3_s)));

  block_iq4_xs iq4_xs{};
  CHECK(validate_row_data(tensor_type::k_iq4_xs, &iq4_xs, sizeof(iq4_xs)));

  block_iq4_nl iq4_nl{};
  CHECK(validate_row_data(tensor_type::k_iq4_nl, &iq4_nl, sizeof(iq4_nl)));

  int8_t i8 = 0;
  CHECK(validate_row_data(tensor_type::k_i8, &i8, sizeof(i8)));
  int16_t i16 = 0;
  CHECK(validate_row_data(tensor_type::k_i16, &i16, sizeof(i16)));
  int32_t i32 = 0;
  CHECK(validate_row_data(tensor_type::k_i32, &i32, sizeof(i32)));
  int64_t i64 = 0;
  CHECK(validate_row_data(tensor_type::k_i64, &i64, sizeof(i64)));
}

TEST_CASE("gguf parse_i32_value handles multiple input types") {
  using namespace emel::parser::gguf;

  char path[128] = {};
  CHECK(make_temp_path(path, sizeof(path)));
  std::FILE * file = std::fopen(path, "wb+");
  REQUIRE(file != nullptr);

  const uint32_t small_u32 = 5;
  const uint32_t big_u32 =
    static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) + 1u;
  const uint16_t small_u16 = 7;
  const int8_t bool_i8 = -1;

  REQUIRE(std::fwrite(&small_u32, 1, sizeof(small_u32), file) == sizeof(small_u32));
  REQUIRE(std::fwrite(&big_u32, 1, sizeof(big_u32), file) == sizeof(big_u32));
  REQUIRE(std::fwrite(&small_u16, 1, sizeof(small_u16), file) == sizeof(small_u16));
  REQUIRE(std::fwrite(&bool_i8, 1, sizeof(bool_i8), file) == sizeof(bool_i8));
  std::rewind(file);

  reader r{file};
  int32_t out = 0;
  CHECK(parse_i32_value(r, value_type::k_u32, out));
  CHECK(out == 5);
  CHECK(!parse_i32_value(r, value_type::k_u32, out));
  CHECK(parse_i32_value(r, value_type::k_u16, out));
  CHECK(out == 7);
  bool bout = false;
  CHECK(parse_bool_value(r, value_type::k_i8, bout));
  CHECK(bout);

  std::fclose(file);
  std::remove(path);
}
