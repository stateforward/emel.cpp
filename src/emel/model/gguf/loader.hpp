#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/parser/events.hpp"
#include "emel/model/weight_loader/events.hpp"

namespace emel::model::gguf {

inline constexpr uint32_t k_gguf_version = 3;
inline constexpr uint32_t k_default_alignment = 32;
inline constexpr uint32_t k_max_architecture = 64;
inline constexpr uint32_t k_max_key_length = 256;
inline constexpr uint32_t k_max_path_length = 1024;

inline constexpr char k_magic[] = "GGUF";
inline constexpr char k_key_architecture[] = "general.architecture";
inline constexpr char k_key_alignment[] = "general.alignment";
inline constexpr char k_key_split_count[] = "split.count";
inline constexpr char k_key_split_no[] = "split.no";
inline constexpr char k_key_split_tensors[] = "split.tensors.count";
inline constexpr char k_suffix_block_count[] = ".block_count";

enum class value_type : int32_t {
  k_u8 = 0,
  k_i8 = 1,
  k_u16 = 2,
  k_i16 = 3,
  k_u32 = 4,
  k_i32 = 5,
  k_f32 = 6,
  k_bool = 7,
  k_string = 8,
  k_array = 9,
  k_u64 = 10,
  k_i64 = 11,
  k_f64 = 12,
  k_count = 13
};

enum class tensor_type : int32_t {
  k_f32 = 0,
  k_f16 = 1,
  k_q4_0 = 2,
  k_q4_1 = 3,
  k_q5_0 = 6,
  k_q5_1 = 7,
  k_q8_0 = 8,
  k_q8_1 = 9,
  k_q2_k = 10,
  k_q3_k = 11,
  k_q4_k = 12,
  k_q5_k = 13,
  k_q6_k = 14,
  k_q8_k = 15,
  k_iq2_xxs = 16,
  k_iq2_xs = 17,
  k_iq3_xxs = 18,
  k_iq1_s = 19,
  k_iq4_nl = 20,
  k_iq3_s = 21,
  k_iq2_s = 22,
  k_iq4_xs = 23,
  k_i8 = 24,
  k_i16 = 25,
  k_i32 = 26,
  k_i64 = 27,
  k_f64 = 28,
  k_iq1_m = 29,
  k_bf16 = 30,
  k_tq1_0 = 34,
  k_tq2_0 = 35,
  k_mxfp4 = 39,
  k_count = 40
};

inline constexpr int32_t k_max_dims = 4;
inline constexpr int32_t k_qk_k = 256;
inline constexpr int32_t k_k_scale_size = 12;
inline constexpr int32_t k_qk4_0 = 32;
inline constexpr int32_t k_qk4_1 = 32;
inline constexpr int32_t k_qk5_0 = 32;
inline constexpr int32_t k_qk5_1 = 32;
inline constexpr int32_t k_qk8_0 = 32;
inline constexpr int32_t k_qk8_1 = 32;
inline constexpr int32_t k_qk4_nl = 32;
inline constexpr int32_t k_qk_mxfp4 = 32;

struct context {
  std::FILE * file = nullptr;
  bool owns_file = false;
  uint32_t version = 0;
  uint32_t alignment = k_default_alignment;
  uint64_t data_offset = 0;
  uint64_t data_size = 0;
  uint16_t split_count = 1;
  uint16_t split_no = 0;
  uint16_t split_tensors_count = 0;
  std::array<char, k_max_architecture> architecture = {};
  uint32_t architecture_len = 0;
  uint32_t block_count = 0;
  std::array<char, k_max_architecture> pending_arch = {};
  uint32_t pending_arch_len = 0;
  uint32_t pending_block_count = 0;
  bool pending_block_count_valid = false;
  const void * mapped_data = nullptr;
  uint64_t mapped_size = 0;
};

struct reader {
  std::FILE * file = nullptr;

  bool read_raw(void * dst, const size_t size) const {
    if (file == nullptr) {
      return false;
    }
    return std::fread(dst, 1, size, file) == size;
  }

  bool skip(const uint64_t size) const {
    if (file == nullptr) {
      return false;
    }
    return std::fseek(file, static_cast<long>(size), SEEK_CUR) == 0;
  }

  template <class T>
  bool read(T & out) const {
    return read_raw(&out, sizeof(T));
  }

  bool read_string(char * out, const uint64_t capacity, uint64_t & out_len) const {
    uint64_t len = 0;
    if (!read(len)) {
      return false;
    }
    out_len = len;
    if (capacity == 0) {
      return skip(len);
    }
    if (len >= capacity) {
      return false;
    }
    if (!read_raw(out, static_cast<size_t>(len))) {
      return false;
    }
    out[len] = '\0';
    return true;
  }
};

inline bool read_key(const reader & r, char * out, const uint64_t capacity, uint64_t & out_len) {
  uint64_t len = 0;
  if (!r.read(len)) {
    return false;
  }
  out_len = len;
  if (capacity == 0) {
    return r.skip(len);
  }
  if (len >= capacity) {
    if (!r.skip(len)) {
      return false;  // GCOVR_EXCL_LINE
    }
    out[0] = '\0';
    out_len = 0;
    return true;
  }
  if (!r.read_raw(out, static_cast<size_t>(len))) {
    return false;
  }
  out[len] = '\0';
  return true;
}

inline uint64_t align_up_u64(const uint64_t value, const uint64_t alignment) noexcept {
  if (alignment == 0) {
    return value;
  }
  const uint64_t rem = value % alignment;
  if (rem == 0) {
    return value;
  }
  return value + (alignment - rem);
}

inline bool mul_overflow_u64(const uint64_t lhs, const uint64_t rhs, uint64_t & out) noexcept {
  if (lhs == 0 || rhs == 0) {
    out = 0;
    return false;
  }
  if (lhs > std::numeric_limits<uint64_t>::max() / rhs) {
    return true;
  }
  out = lhs * rhs;
  return false;
}

inline bool add_overflow_u64(const uint64_t lhs, const uint64_t rhs, uint64_t & out) noexcept {
  if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
    return true;
  }
  out = lhs + rhs;
  return false;
}

inline bool key_equals(const char * key, const uint64_t len, const char * literal) noexcept {
  const size_t lit_len = std::strlen(literal);
  if (len != lit_len) {
    return false;
  }
  return std::memcmp(key, literal, len) == 0;
}

inline bool key_has_suffix(const char * key, const uint64_t len, const char * suffix,
                           uint64_t & prefix_len) noexcept {
  const size_t suffix_len = std::strlen(suffix);
  if (len < suffix_len) {
    return false;
  }
  const uint64_t start = len - suffix_len;
  if (std::memcmp(key + start, suffix, suffix_len) != 0) {
    return false;
  }
  prefix_len = start;
  return true;
}

inline bool read_and_discard_string(const reader & r) {
  uint64_t len = 0;
  if (!r.read(len)) {
    return false;
  }
  return r.skip(len);
}

inline bool skip_value(const reader & r, const value_type type, const uint64_t count) {
  switch (type) {
    case value_type::k_u8:
    case value_type::k_i8:
    case value_type::k_bool:
      return r.skip(count);
    case value_type::k_u16:
    case value_type::k_i16:
      return r.skip(count * sizeof(uint16_t));
    case value_type::k_u32:
    case value_type::k_i32:
    case value_type::k_f32:
      return r.skip(count * sizeof(uint32_t));
    case value_type::k_u64:
    case value_type::k_i64:
    case value_type::k_f64:
      return r.skip(count * sizeof(uint64_t));
    case value_type::k_string:
      for (uint64_t i = 0; i < count; ++i) {
        if (!read_and_discard_string(r)) {
          return false;
        }
      }
      return true;
    default:
      return false;
  }
}

inline bool parse_u32_value(const reader & r, const value_type type, uint32_t & out) {
  if (type == value_type::k_u32) {
    return r.read(out);
  }
  if (type == value_type::k_i32) {
    int32_t tmp = 0;
    if (!r.read(tmp)) {
      return false;
    }
    if (tmp < 0) {
      return false;
    }
    out = static_cast<uint32_t>(tmp);
    return true;
  }
  if (type == value_type::k_u16) {
    uint16_t tmp = 0;
    if (!r.read(tmp)) {
      return false;
    }
    out = tmp;
    return true;
  }
  return false;
}

inline int32_t blck_size_for(const tensor_type type) noexcept {
  switch (type) {
    case tensor_type::k_q4_0: return k_qk4_0;
    case tensor_type::k_q4_1: return k_qk4_1;
    case tensor_type::k_q5_0: return k_qk5_0;
    case tensor_type::k_q5_1: return k_qk5_1;
    case tensor_type::k_q8_0: return k_qk8_0;
    case tensor_type::k_q8_1: return k_qk8_1;
    case tensor_type::k_q2_k:
    case tensor_type::k_q3_k:
    case tensor_type::k_q4_k:
    case tensor_type::k_q5_k:
    case tensor_type::k_q6_k:
    case tensor_type::k_q8_k:
    case tensor_type::k_iq2_xxs:
    case tensor_type::k_iq2_xs:
    case tensor_type::k_iq3_xxs:
    case tensor_type::k_iq1_s:
    case tensor_type::k_iq3_s:
    case tensor_type::k_iq2_s:
    case tensor_type::k_iq4_xs:
    case tensor_type::k_iq1_m:
    case tensor_type::k_tq1_0:
    case tensor_type::k_tq2_0:
      return k_qk_k;
    case tensor_type::k_iq4_nl:
      return k_qk4_nl;
    case tensor_type::k_mxfp4:
      return k_qk_mxfp4;
    case tensor_type::k_f32:
    case tensor_type::k_f16:
    case tensor_type::k_i8:
    case tensor_type::k_i16:
    case tensor_type::k_i32:
    case tensor_type::k_i64:
    case tensor_type::k_f64:
    case tensor_type::k_bf16:
      return 1;
    default:
      return 0;
  }
}

inline uint32_t type_size_for(const tensor_type type) noexcept {
  switch (type) {
    case tensor_type::k_f32: return 4;
    case tensor_type::k_f16: return 2;
    case tensor_type::k_i8: return 1;
    case tensor_type::k_i16: return 2;
    case tensor_type::k_i32: return 4;
    case tensor_type::k_i64: return 8;
    case tensor_type::k_f64: return 8;
    case tensor_type::k_bf16: return 2;
    case tensor_type::k_q4_0: return 2 + k_qk4_0 / 2;
    case tensor_type::k_q4_1: return 4 + k_qk4_1 / 2;
    case tensor_type::k_q5_0: return 2 + 4 + k_qk5_0 / 2;
    case tensor_type::k_q5_1: return 4 + 4 + k_qk5_1 / 2;
    case tensor_type::k_q8_0: return 2 + k_qk8_0;
    case tensor_type::k_q8_1: return 4 + k_qk8_1;
    case tensor_type::k_mxfp4: return 1 + k_qk_mxfp4 / 2;
    case tensor_type::k_q2_k: return 4 + k_qk_k / 16 + k_qk_k / 4;
    case tensor_type::k_q3_k: return 2 + k_qk_k / 4 + k_qk_k / 8 + 12;
    case tensor_type::k_q4_k: return 4 + k_k_scale_size + k_qk_k / 2;
    case tensor_type::k_q5_k: return 4 + k_k_scale_size + k_qk_k / 2 + k_qk_k / 8;
    case tensor_type::k_q6_k: return 2 + k_qk_k / 16 + 3 * k_qk_k / 4;
    case tensor_type::k_q8_k: return 4 + k_qk_k + (k_qk_k / 16) * 2;
    case tensor_type::k_iq2_xxs: return 2 + (k_qk_k / 8) * 2;
    case tensor_type::k_iq2_xs: return 2 + (k_qk_k / 8) * 2 + k_qk_k / 32;
    case tensor_type::k_iq3_xxs: return 2 + 3 * (k_qk_k / 8);
    case tensor_type::k_iq3_s: return 2 + 13 * (k_qk_k / 32) + (k_qk_k / 64);
    case tensor_type::k_iq2_s: return 2 + k_qk_k / 4 + k_qk_k / 16;
    case tensor_type::k_iq1_s: return 2 + k_qk_k / 8 + k_qk_k / 16;
    case tensor_type::k_iq1_m: return k_qk_k / 8 + k_qk_k / 16 + k_qk_k / 32;
    case tensor_type::k_iq4_nl: return 2 + k_qk4_nl / 2;
    case tensor_type::k_iq4_xs: return 2 + 2 + k_qk_k / 64 + k_qk_k / 2;
    case tensor_type::k_tq1_0: return 2 + k_qk_k / 64 + (k_qk_k - 4 * (k_qk_k / 64)) / 5;
    case tensor_type::k_tq2_0: return 2 + k_qk_k / 4;
    default:
      return 0;
  }
}

inline bool compute_tensor_size(
    const std::array<int64_t, k_max_dims> & dims, const tensor_type type,
    uint64_t & out_size) {
  const int32_t block = blck_size_for(type);
  const uint32_t type_size = type_size_for(type);
  if (block <= 0 || type_size == 0) {
    return false;
  }
  if (dims[0] < 0 || dims[1] < 0 || dims[2] < 0 || dims[3] < 0) {
    return false;
  }
  if (dims[0] % block != 0) {
    return false;
  }
  uint64_t elements = 1;
  for (int32_t i = 0; i < k_max_dims; ++i) {
    uint64_t value = static_cast<uint64_t>(dims[i]);
    if (mul_overflow_u64(elements, value, elements)) {
      return false;  // GCOVR_EXCL_LINE
    }
  }
  const uint64_t blocks = elements / static_cast<uint64_t>(block);
  uint64_t bytes = 0;
  if (mul_overflow_u64(blocks, type_size, bytes)) {
    return false;
  }
  out_size = bytes;
  return true;
}

inline void reset_context(context & ctx) {
#if !defined(_WIN32)
  if (ctx.mapped_data != nullptr && ctx.mapped_size > 0) {
    (void)munmap(const_cast<void *>(ctx.mapped_data), static_cast<size_t>(ctx.mapped_size));
  }
#endif
  if (ctx.owns_file && ctx.file != nullptr) {
    std::fclose(ctx.file);
  }
  ctx = {};
  ctx.alignment = k_default_alignment;
  ctx.split_count = 1;
}

inline void reset_model_data(emel::model::data & model) {
  model.n_layers = 0;
  model.n_tensors = 0;
  model.name_bytes_used = 0;
  model.weights_data = nullptr;
  model.weights_size = 0;
  model.weights_mapped = false;
  model.architecture_name[0] = '\0';
}

inline context * get_context(const void * ptr) {
  return static_cast<context *>(const_cast<void *>(ptr));
}

inline bool store_name(
    emel::model::data & model, const char * name, const uint64_t name_len,
    uint32_t & out_offset) {
  if (name_len == 0) {
    return false;
  }
  const uint64_t total = name_len + 1;
  if (model.name_bytes_used > static_cast<uint32_t>(model.name_storage.size())) {
    return false;
  }
  const uint64_t remaining = model.name_storage.size() - model.name_bytes_used;
  if (total > remaining) {
    return false;
  }
  out_offset = model.name_bytes_used;
  std::memcpy(model.name_storage.data() + model.name_bytes_used, name, name_len);
  model.name_storage[model.name_bytes_used + name_len] = '\0';
  model.name_bytes_used += static_cast<uint32_t>(total);
  return true;
}

inline bool parse_header(
    const reader & r, context & ctx, int64_t & n_tensors, int64_t & n_kv) {
  char magic[5] = {};
  if (!r.read_raw(magic, 4)) {
    return false;
  }
  if (std::memcmp(magic, k_magic, 4) != 0) {
    return false;
  }
  if (!r.read(ctx.version)) {
    return false;
  }
  if ((ctx.version & 0x0000FFFFu) == 0u) {
    return false;
  }
  if (ctx.version == 1 || ctx.version > k_gguf_version) {
    return false;
  }
  if (!r.read(n_tensors)) {
    return false;
  }
  if (!r.read(n_kv)) {
    return false;
  }
  return true;
}

inline bool parse_kv(
    const reader & r, context & ctx, const int64_t n_kv, int32_t & out_error) {
  char key[k_max_key_length] = {};
  for (int64_t i = 0; i < n_kv; ++i) {
    uint64_t key_len = 0;
    if (!read_key(r, key, sizeof(key), key_len)) {
      out_error = EMEL_ERR_PARSE_FAILED;
      return false;
    }
    int32_t type_raw = -1;
    if (!r.read(type_raw)) {
      out_error = EMEL_ERR_PARSE_FAILED;
      return false;
    }
    bool is_array = false;
    uint64_t count = 1;
    value_type type = static_cast<value_type>(type_raw);
    if (type == value_type::k_array) {
      is_array = true;
      int32_t arr_type_raw = -1;
      if (!r.read(arr_type_raw)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      type = static_cast<value_type>(arr_type_raw);
      if (!r.read(count)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
    }
    if (is_array) {
      if (!skip_value(r, type, count)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }

    if (key_equals(key, key_len, k_key_architecture)) {
      if (type != value_type::k_string) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      uint64_t len = 0;
      if (!r.read_string(ctx.architecture.data(), ctx.architecture.size(), len)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      ctx.architecture_len = static_cast<uint32_t>(len);
      if (ctx.pending_block_count_valid &&
          ctx.pending_arch_len == ctx.architecture_len &&
          std::memcmp(ctx.pending_arch.data(), ctx.architecture.data(), ctx.architecture_len) == 0) {
        ctx.block_count = ctx.pending_block_count;
      }
      continue;
    }

    if (key_equals(key, key_len, k_key_alignment)) {
      uint32_t alignment = 0;
      if (!parse_u32_value(r, type, alignment)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      ctx.alignment = alignment;
      continue;
    }

    if (key_equals(key, key_len, k_key_split_count)) {
      uint32_t split = 0;
      if (!parse_u32_value(r, type, split)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      ctx.split_count = static_cast<uint16_t>(split);
      continue;
    }

    if (key_equals(key, key_len, k_key_split_no)) {
      uint32_t split = 0;
      if (!parse_u32_value(r, type, split)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      ctx.split_no = static_cast<uint16_t>(split);
      continue;
    }

    if (key_equals(key, key_len, k_key_split_tensors)) {
      uint32_t split = 0;
      if (!parse_u32_value(r, type, split)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      ctx.split_tensors_count = static_cast<uint16_t>(split);
      continue;
    }

    uint64_t prefix_len = 0;
    if (key_has_suffix(key, key_len, k_suffix_block_count, prefix_len)) {
      uint32_t block_count = 0;
      if (!parse_u32_value(r, type, block_count)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      if (ctx.architecture_len > 0 &&
          prefix_len == ctx.architecture_len &&
          std::memcmp(key, ctx.architecture.data(), ctx.architecture_len) == 0) {
        ctx.block_count = block_count;
      } else if (prefix_len < ctx.pending_arch.size()) {
        std::memcpy(ctx.pending_arch.data(), key, prefix_len);
        ctx.pending_arch_len = static_cast<uint32_t>(prefix_len);
        ctx.pending_block_count = block_count;
        ctx.pending_block_count_valid = true;
      }
      continue;
    }

    if (!skip_value(r, type, count)) {
      out_error = EMEL_ERR_PARSE_FAILED;
      return false;
    }
  }
  return true;
}

inline bool parse_tensors(
    const reader & r, context & ctx, emel::model::data & model,
    const int64_t n_tensors, int32_t & out_error) {
  if (n_tensors < 0 || n_tensors > emel::model::data::k_max_tensors) {
    out_error = EMEL_ERR_MODEL_INVALID;
    return false;
  }
  model.n_tensors = static_cast<uint32_t>(n_tensors);

  uint64_t expected_offset = 0;
  for (int64_t i = 0; i < n_tensors; ++i) {
    char name[k_max_key_length] = {};
    uint64_t name_len = 0;
    if (!r.read_string(name, sizeof(name), name_len)) {
      out_error = EMEL_ERR_PARSE_FAILED;
      return false;
    }
    uint32_t name_offset = 0;
    if (!store_name(model, name, name_len, name_offset)) {
      out_error = EMEL_ERR_MODEL_INVALID;
      return false;
    }
    uint32_t n_dims = 0;
    if (!r.read(n_dims)) {
      out_error = EMEL_ERR_PARSE_FAILED;
      return false;
    }
    if (n_dims > k_max_dims) {
      out_error = EMEL_ERR_MODEL_INVALID;
      return false;
    }
    std::array<int64_t, k_max_dims> dims = {1, 1, 1, 1};
    for (uint32_t d = 0; d < n_dims; ++d) {
      int64_t dim = 0;
      if (!r.read(dim)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      if (dim < 0) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      dims[d] = dim;
    }
    int32_t type_raw = 0;
    if (!r.read(type_raw)) {
      out_error = EMEL_ERR_PARSE_FAILED;
      return false;
    }
    if (type_raw < 0 || type_raw >= static_cast<int32_t>(tensor_type::k_count)) {
      out_error = EMEL_ERR_MODEL_INVALID;
      return false;
    }
    const tensor_type type = static_cast<tensor_type>(type_raw);
    uint64_t tensor_bytes = 0;
    if (!compute_tensor_size(dims, type, tensor_bytes)) {
      out_error = EMEL_ERR_MODEL_INVALID;
      return false;
    }
    uint64_t offset = 0;
    if (!r.read(offset)) {
      out_error = EMEL_ERR_PARSE_FAILED;
      return false;
    }
    if (offset != expected_offset) {
      out_error = EMEL_ERR_MODEL_INVALID;
      return false;
    }
    emel::model::data::tensor_record & record = model.tensors[static_cast<size_t>(i)];
    record.name_offset = name_offset;
    record.name_length = static_cast<uint32_t>(name_len);
    record.type = type_raw;
    record.n_dims = static_cast<int32_t>(n_dims);
    record.dims = dims;
    record.data_offset = offset;
    record.data_size = tensor_bytes;
    record.data = nullptr;

    uint64_t padded = align_up_u64(tensor_bytes, ctx.alignment);
  if (add_overflow_u64(expected_offset, padded, expected_offset)) {
      out_error = EMEL_ERR_MODEL_INVALID;  // GCOVR_EXCL_LINE
      return false;  // GCOVR_EXCL_LINE
  }
  }
  ctx.data_size = expected_offset;
  model.weights_size = expected_offset;
  return true;
}

inline bool map_parser(const emel::model::loader::event::load & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  context * ctx = get_context(ev.format_ctx);
  if (ctx == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  reset_context(*ctx);
  reset_model_data(ev.model_data);

  if (ev.file_handle != nullptr) {
    ctx->file = static_cast<std::FILE *>(ev.file_handle);
    ctx->owns_file = false;
  } else {
    if (ev.model_path.empty() || ev.model_path.size() >= k_max_path_length) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_INVALID_ARGUMENT;
      }
      return false;
    }
    char path[k_max_path_length] = {};
    std::memcpy(path, ev.model_path.data(), ev.model_path.size());
    path[ev.model_path.size()] = '\0';
    ctx->file = std::fopen(path, "rb");
    if (ctx->file == nullptr) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_IO;
      }
      return false;
    }
    ctx->owns_file = true;
  }

  reader r{ctx->file};
  int64_t n_tensors = 0;
  int64_t n_kv = 0;
  if (!parse_header(r, *ctx, n_tensors, n_kv)) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_FORMAT_UNSUPPORTED;
    }
    reset_context(*ctx);
    return false;
  }
  int32_t parse_error = EMEL_OK;
  if (!parse_kv(r, *ctx, n_kv, parse_error)) {
    if (err_out != nullptr) {
      *err_out = parse_error;
    }
    reset_context(*ctx);
    return false;
  }
  if (!parse_tensors(r, *ctx, ev.model_data, n_tensors, parse_error)) {
    if (err_out != nullptr) {
      *err_out = parse_error;
    }
    reset_context(*ctx);
    return false;
  }

  const long pos = std::ftell(ctx->file);
  if (pos < 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
    }
    reset_context(*ctx);
    return false;  // GCOVR_EXCL_LINE
  }
  const uint64_t aligned = align_up_u64(static_cast<uint64_t>(pos), ctx->alignment);
  if (std::fseek(ctx->file, static_cast<long>(aligned), SEEK_SET) != 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
    }
    reset_context(*ctx);
    return false;  // GCOVR_EXCL_LINE
  }
  ctx->data_offset = aligned;
  return true;
}

inline bool parse_architecture(const emel::model::parser::event::parse_model & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (ev.model == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  context * ctx = get_context(ev.format_ctx);
  if (ctx == nullptr || ctx->architecture_len == 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  if (ctx->architecture_len >= ev.model->architecture_name.size()) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  std::memcpy(ev.model->architecture_name.data(), ctx->architecture.data(), ctx->architecture_len);
  ev.model->architecture_name[ctx->architecture_len] = '\0';
  return true;
}

inline bool map_architecture(const emel::model::parser::event::parse_model & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (ev.model == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  if (ev.architectures == nullptr || ev.n_architectures <= 0) {
    return true;
  }
  const auto * list = static_cast<const char * const *>(ev.architectures);
  const char * name = ev.model->architecture_name.data();
  for (int32_t i = 0; i < ev.n_architectures; ++i) {
    if (list[i] != nullptr && std::strcmp(list[i], name) == 0) {
      return true;
    }
  }
  if (err_out != nullptr) {
    *err_out = EMEL_ERR_MODEL_INVALID;
  }
  return false;
}

inline bool parse_hparams(const emel::model::parser::event::parse_model & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (ev.model == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  context * ctx = get_context(ev.format_ctx);
  if (ctx == nullptr || ctx->block_count == 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  ev.model->n_layers = static_cast<int32_t>(ctx->block_count);
  return true;
}

inline bool parse_vocab(const emel::model::parser::event::parse_model &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  return true;
}

inline bool map_tensors(const emel::model::parser::event::parse_model & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (ev.model == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  if (ev.model->n_tensors == 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  return true;
}

inline bool map_layers(const emel::model::loader::event::load & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (ev.model_data.n_layers <= 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  return true;
}

inline bool validate_structure(const emel::model::loader::event::load & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  const context * ctx = get_context(ev.format_ctx);
  if (ctx == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  if (ev.model_data.n_tensors == 0 || ev.model_data.weights_size == 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  if (ctx->data_offset == 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  return true;
}

inline bool validate_architecture(
    const emel::model::loader::event::load & ev, int32_t * err_out) {
  emel::model::parser::event::parse_model request{
    .model = &ev.model_data,
    .architectures = ev.architectures,
    .n_architectures = ev.n_architectures,
    .format_ctx = ev.format_ctx
  };
  return map_architecture(request, err_out);
}

#if defined(_WIN32)
inline bool map_mmap(
    const emel::model::weight_loader::event::load_weights &,
    uint64_t *, uint64_t *, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_ERR_FORMAT_UNSUPPORTED;
  }
  return false;
}
#else
inline bool map_mmap(
    const emel::model::weight_loader::event::load_weights & ev,
    uint64_t * bytes_done, uint64_t * bytes_total, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  const emel::model::loader::event::load * request = ev.loader_request;
  if (request == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  context * ctx = get_context(request->format_ctx);
  if (ctx == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  if (ctx->mapped_data == nullptr) {
    if (request->model_path.empty() || request->model_path.size() >= k_max_path_length) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_INVALID_ARGUMENT;
      }
      return false;
    }
    char path[k_max_path_length] = {};
    std::memcpy(path, request->model_path.data(), request->model_path.size());
    path[request->model_path.size()] = '\0';
    std::FILE * file = std::fopen(path, "rb");
    if (file == nullptr) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_IO;
      }
      return false;
    }
    if (std::fseek(file, 0, SEEK_END) != 0) {
      std::fclose(file);
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
      }
      return false;  // GCOVR_EXCL_LINE
    }
    const long file_size = std::ftell(file);
    if (file_size < 0) {
      std::fclose(file);
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
      }
      return false;  // GCOVR_EXCL_LINE
    }
    std::rewind(file);
    const int fd = fileno(file);
    void * data = mmap(nullptr, static_cast<size_t>(file_size), PROT_READ, MAP_PRIVATE, fd, 0);
    std::fclose(file);
    if (data == MAP_FAILED) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
      }
      return false;  // GCOVR_EXCL_LINE
    }
    ctx->mapped_data = data;
    ctx->mapped_size = static_cast<uint64_t>(file_size);
  }
  if (ctx->mapped_size < ctx->data_offset ||
      ctx->mapped_size - ctx->data_offset < ctx->data_size) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }

  const uint8_t * base =
    static_cast<const uint8_t *>(ctx->mapped_data) + ctx->data_offset;
  request->model_data.weights_data = base;
  request->model_data.weights_size = ctx->data_size;
  request->model_data.weights_mapped = true;
  for (uint32_t i = 0; i < request->model_data.n_tensors; ++i) {
    auto & record = request->model_data.tensors[i];
    record.data = base + record.data_offset;
  }
  if (bytes_total != nullptr) {
    *bytes_total = ctx->data_size;
  }
  if (bytes_done != nullptr) {
    *bytes_done = ctx->data_size;
  }
  return true;
}
#endif

inline bool load_streamed(
    const emel::model::weight_loader::event::load_weights & ev,
    uint64_t * bytes_done, uint64_t * bytes_total, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  const emel::model::loader::event::load * request = ev.loader_request;
  if (request == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  context * ctx = get_context(request->format_ctx);
  if (ctx == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  if (request->weights_buffer == nullptr || request->weights_buffer_size < ctx->data_size) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  std::FILE * file = ctx->file;
  bool owns_file = false;
  if (file == nullptr) {
    if (request->model_path.empty() || request->model_path.size() >= k_max_path_length) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_INVALID_ARGUMENT;
      }
      return false;
    }
    char path[k_max_path_length] = {};
    std::memcpy(path, request->model_path.data(), request->model_path.size());
    path[request->model_path.size()] = '\0';
    file = std::fopen(path, "rb");
    if (file == nullptr) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_IO;
      }
      return false;
    }
    owns_file = true;
  }
  if (std::fseek(file, static_cast<long>(ctx->data_offset), SEEK_SET) != 0) {
    if (owns_file) {
      std::fclose(file);
    }
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
    }
    return false;  // GCOVR_EXCL_LINE
  }
  if (ctx->data_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    if (owns_file) {
      std::fclose(file);
    }
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;  // GCOVR_EXCL_LINE
    }
    return false;  // GCOVR_EXCL_LINE
  }
  const size_t bytes_to_read = static_cast<size_t>(ctx->data_size);
  if (std::fread(request->weights_buffer, 1, bytes_to_read, file) != bytes_to_read) {
    if (owns_file) {
      std::fclose(file);
    }
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
    }
    return false;  // GCOVR_EXCL_LINE
  }
  if (owns_file) {
    std::fclose(file);
  }
  const auto * base = static_cast<const uint8_t *>(request->weights_buffer);
  request->model_data.weights_data = base;
  request->model_data.weights_size = ctx->data_size;
  request->model_data.weights_mapped = false;
  for (uint32_t i = 0; i < request->model_data.n_tensors; ++i) {
    auto & record = request->model_data.tensors[i];
    record.data = base + record.data_offset;
  }
  if (bytes_total != nullptr) {
    *bytes_total = ctx->data_size;
  }
  if (bytes_done != nullptr) {
    *bytes_done = ctx->data_size;
  }
  return true;
}

}  // namespace emel::model::gguf
