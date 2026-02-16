#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
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

using fp16 = uint16_t;

struct block_q4_0 {
  fp16 d;
  uint8_t qs[k_qk4_0 / 2];
};

struct block_q4_1 {
  fp16 d;
  fp16 m;
  uint8_t qs[k_qk4_1 / 2];
};

struct block_mxfp4 {
  uint8_t e;
  uint8_t qs[k_qk_mxfp4 / 2];
};

struct block_q5_0 {
  fp16 d;
  uint8_t qh[4];
  uint8_t qs[k_qk5_0 / 2];
};

struct block_q5_1 {
  fp16 d;
  fp16 m;
  uint8_t qh[4];
  uint8_t qs[k_qk5_1 / 2];
};

struct block_q8_0 {
  fp16 d;
  int8_t qs[k_qk8_0];
};

struct block_q8_1 {
  fp16 d;
  fp16 s;
  int8_t qs[k_qk8_1];
};

struct block_tq1_0 {
  uint8_t qs[(k_qk_k - 4 * (k_qk_k / 64)) / 5];
  uint8_t qh[k_qk_k / 64];
  fp16 d;
};

struct block_tq2_0 {
  uint8_t qs[k_qk_k / 4];
  fp16 d;
};

struct block_q2_k {
  uint8_t scales[k_qk_k / 16];
  uint8_t qs[k_qk_k / 4];
  fp16 d;
  fp16 dmin;
};

struct block_q3_k {
  uint8_t hmask[k_qk_k / 8];
  uint8_t qs[k_qk_k / 4];
  uint8_t scales[12];
  fp16 d;
};

struct block_q4_k {
  fp16 d;
  fp16 dmin;
  uint8_t scales[k_k_scale_size];
  uint8_t qs[k_qk_k / 2];
};

struct block_q5_k {
  fp16 d;
  fp16 dmin;
  uint8_t scales[k_k_scale_size];
  uint8_t qh[k_qk_k / 8];
  uint8_t qs[k_qk_k / 2];
};

struct block_q6_k {
  uint8_t ql[k_qk_k / 2];
  uint8_t qh[k_qk_k / 4];
  int8_t scales[k_qk_k / 16];
  fp16 d;
};

struct block_q8_k {
  float d;
  int8_t qs[k_qk_k];
  int16_t bsums[k_qk_k / 16];
};

struct block_iq2_xxs {
  fp16 d;
  uint16_t qs[k_qk_k / 8];
};

struct block_iq2_xs {
  fp16 d;
  uint16_t qs[k_qk_k / 8];
  uint8_t scales[k_qk_k / 32];
};

struct block_iq2_s {
  fp16 d;
  uint8_t qs[k_qk_k / 4];
  uint8_t qh[k_qk_k / 32];
  uint8_t scales[k_qk_k / 32];
};

struct block_iq3_xxs {
  fp16 d;
  uint8_t qs[3 * k_qk_k / 8];
};

inline constexpr int32_t k_iq3s_n_scale = k_qk_k / 64;

struct block_iq3_s {
  fp16 d;
  uint8_t qs[k_qk_k / 4];
  uint8_t qh[k_qk_k / 32];
  uint8_t signs[k_qk_k / 8];
  uint8_t scales[k_iq3s_n_scale];
};

struct block_iq1_s {
  fp16 d;
  uint8_t qs[k_qk_k / 8];
  uint16_t qh[k_qk_k / 32];
};

struct block_iq1_m {
  uint8_t qs[k_qk_k / 8];
  uint8_t qh[k_qk_k / 16];
  uint8_t scales[k_qk_k / 32];
};

union iq1m_scale_t {
  fp16 f16;
  uint16_t u16;
};

struct block_iq4_nl {
  fp16 d;
  uint8_t qs[k_qk4_nl / 2];
};

struct block_iq4_xs {
  fp16 d;
  uint16_t scales_h;
  uint8_t scales_l[k_qk_k / 64];
  uint8_t qs[k_qk_k / 2];
};

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
  std::array<const void *, emel::model::data::k_max_split_files> mapped_splits = {};
  std::array<uint64_t, emel::model::data::k_max_split_files> mapped_sizes = {};
  uint16_t mapped_count = 0;
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

inline bool parse_i32_value(const reader & r, const value_type type, int32_t & out) {
  if (type == value_type::k_i32) {
    return r.read(out);
  }
  if (type == value_type::k_u32) {
    uint32_t tmp = 0;
    if (!r.read(tmp)) {
      return false;
    }
    if (tmp > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }
    out = static_cast<int32_t>(tmp);
    return true;
  }
  if (type == value_type::k_u16) {
    uint16_t tmp = 0;
    if (!r.read(tmp)) {
      return false;
    }
    out = static_cast<int32_t>(tmp);
    return true;
  }
  return false;
}

inline bool parse_bool_value(const reader & r, const value_type type, bool & out) {
  if (type == value_type::k_bool || type == value_type::k_u8) {
    uint8_t tmp = 0;
    if (!r.read(tmp)) {
      return false;
    }
    out = tmp != 0;
    return true;
  }
  if (type == value_type::k_i8) {
    int8_t tmp = 0;
    if (!r.read(tmp)) {
      return false;
    }
    out = tmp != 0;
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

inline bool is_inf_f16(const uint16_t value) noexcept {
  return (value & 0x7c00u) == 0x7c00u && (value & 0x03ffu) == 0;
}

inline bool is_nan_f16(const uint16_t value) noexcept {
  return (value & 0x7c00u) == 0x7c00u && (value & 0x03ffu) != 0;
}

inline bool validate_fp16_value(const uint16_t value) noexcept {
  return !is_inf_f16(value) && !is_nan_f16(value);
}

inline bool validate_float_value(const float value) noexcept {
  return std::isfinite(value);
}

inline bool validate_double_value(const double value) noexcept {
  return std::isfinite(value);
}

inline bool validate_e8m0_value(const uint8_t value) noexcept {
  return value != 0xffu;
}

template <typename block_type>
inline bool validate_block_d_f16(const void * data, const size_t blocks) noexcept {
  const block_type * q = static_cast<const block_type *>(data);
  for (size_t i = 0; i < blocks; ++i) {
    if (!validate_fp16_value(q[i].d)) {
      return false;
    }
  }
  return true;
}

template <typename block_type>
inline bool validate_block_dm_f16(const void * data, const size_t blocks) noexcept {
  const block_type * q = static_cast<const block_type *>(data);
  for (size_t i = 0; i < blocks; ++i) {
    if (!validate_fp16_value(q[i].d) || !validate_fp16_value(q[i].m)) {
      return false;
    }
  }
  return true;
}

template <typename block_type>
inline bool validate_block_ddmin_f16(const void * data, const size_t blocks) noexcept {
  const block_type * q = static_cast<const block_type *>(data);
  for (size_t i = 0; i < blocks; ++i) {
    if (!validate_fp16_value(q[i].d) || !validate_fp16_value(q[i].dmin)) {
      return false;
    }
  }
  return true;
}

template <typename block_type>
inline bool validate_block_e8m0(const void * data, const size_t blocks) noexcept {
  const block_type * q = static_cast<const block_type *>(data);
  for (size_t i = 0; i < blocks; ++i) {
    if (!validate_e8m0_value(q[i].e)) {
      return false;
    }
  }
  return true;
}

inline bool validate_block_iq1_m(const void * data, const size_t blocks) noexcept {
  const block_iq1_m * q = static_cast<const block_iq1_m *>(data);
  for (size_t i = 0; i < blocks; ++i) {
    const uint16_t * sc = reinterpret_cast<const uint16_t *>(q[i].scales);
    iq1m_scale_t scale{};
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0u) | ((sc[2] >> 4) & 0x0f00u) |
                (sc[3] & 0xf000u);
    if (!validate_fp16_value(scale.f16)) {
      return false;
    }
  }
  return true;
}

inline bool validate_row_data(const tensor_type type, const void * data,
                              const uint64_t nbytes) noexcept {
  if (data == nullptr && nbytes > 0) {
    return false;
  }
  if (type < tensor_type::k_f32 || type >= tensor_type::k_count) {
    return false;
  }
  const uint32_t type_size = type_size_for(type);
  if (type_size == 0) {
    return false;
  }
  if ((nbytes % type_size) != 0) {
    return false;
  }
  const size_t count = static_cast<size_t>(nbytes / type_size);
  switch (type) {
    case tensor_type::k_f32: {
      const float * values = static_cast<const float *>(data);
      for (size_t i = 0; i < count; ++i) {
        if (!validate_float_value(values[i])) {
          return false;
        }
      }
    } break;
    case tensor_type::k_f64: {
      const double * values = static_cast<const double *>(data);
      for (size_t i = 0; i < count; ++i) {
        if (!validate_double_value(values[i])) {
          return false;
        }
      }
    } break;
    case tensor_type::k_f16: {
      const uint16_t * values = static_cast<const uint16_t *>(data);
      for (size_t i = 0; i < count; ++i) {
        if (!validate_fp16_value(values[i])) {
          return false;
        }
      }
    } break;
    case tensor_type::k_bf16: {
      const uint16_t * values = static_cast<const uint16_t *>(data);
      for (size_t i = 0; i < count; ++i) {
        const uint16_t abs = values[i] & 0x7fffu;
        if (abs > 0x7f80u || abs == 0x7f80u) {
          return false;
        }
      }
    } break;
    case tensor_type::k_q4_0:
      return validate_block_d_f16<block_q4_0>(data, count);
    case tensor_type::k_q4_1:
      return validate_block_dm_f16<block_q4_1>(data, count);
    case tensor_type::k_q5_0:
      return validate_block_d_f16<block_q5_0>(data, count);
    case tensor_type::k_q5_1:
      return validate_block_dm_f16<block_q5_1>(data, count);
    case tensor_type::k_q8_0:
      return validate_block_d_f16<block_q8_0>(data, count);
    case tensor_type::k_q8_1: {
      const block_q8_1 * q = static_cast<const block_q8_1 *>(data);
      for (size_t i = 0; i < count; ++i) {
        if (!validate_fp16_value(q[i].d) || !validate_fp16_value(q[i].s)) {
          return false;
        }
      }
    } break;
    case tensor_type::k_mxfp4:
      return validate_block_e8m0<block_mxfp4>(data, count);
    case tensor_type::k_q2_k:
      return validate_block_ddmin_f16<block_q2_k>(data, count);
    case tensor_type::k_q3_k:
      return validate_block_d_f16<block_q3_k>(data, count);
    case tensor_type::k_q4_k:
      return validate_block_ddmin_f16<block_q4_k>(data, count);
    case tensor_type::k_q5_k:
      return validate_block_ddmin_f16<block_q5_k>(data, count);
    case tensor_type::k_q6_k:
      return validate_block_d_f16<block_q6_k>(data, count);
    case tensor_type::k_q8_k: {
      const block_q8_k * q = static_cast<const block_q8_k *>(data);
      for (size_t i = 0; i < count; ++i) {
        if (!validate_float_value(q[i].d)) {
          return false;
        }
      }
    } break;
    case tensor_type::k_tq1_0:
      return validate_block_d_f16<block_tq1_0>(data, count);
    case tensor_type::k_tq2_0:
      return validate_block_d_f16<block_tq2_0>(data, count);
    case tensor_type::k_iq1_s:
      return validate_block_d_f16<block_iq1_s>(data, count);
    case tensor_type::k_iq1_m:
      return validate_block_iq1_m(data, count);
    case tensor_type::k_iq2_xxs:
      return validate_block_d_f16<block_iq2_xxs>(data, count);
    case tensor_type::k_iq2_xs:
      return validate_block_d_f16<block_iq2_xs>(data, count);
    case tensor_type::k_iq2_s:
      return validate_block_d_f16<block_iq2_s>(data, count);
    case tensor_type::k_iq3_xxs:
      return validate_block_d_f16<block_iq3_xxs>(data, count);
    case tensor_type::k_iq3_s:
      return validate_block_d_f16<block_iq3_s>(data, count);
    case tensor_type::k_iq4_xs:
      return validate_block_d_f16<block_iq4_xs>(data, count);
    case tensor_type::k_iq4_nl:
      return validate_block_d_f16<block_iq4_nl>(data, count);
    case tensor_type::k_i8:
    case tensor_type::k_i16:
    case tensor_type::k_i32:
    case tensor_type::k_i64:
      break;
    default:
      break;
  }
  return true;
}

inline bool validate_tensor_data(const emel::model::data & model, int32_t * err_out) noexcept {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  for (uint32_t i = 0; i < model.n_tensors; ++i) {
    const auto & record = model.tensors[i];
    if (record.data_size == 0) {
      continue;
    }
    if (record.data == nullptr) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;
      }
      return false;
    }
    if (record.file_index >= model.weights_split_count) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;
      }
      return false;
    }
    const uint64_t split_size = model.weights_split_sizes[record.file_index];
    if (record.file_offset > split_size || record.data_size > split_size - record.file_offset) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;
      }
      return false;
    }
    if (record.type < 0 || record.type >= static_cast<int32_t>(tensor_type::k_count)) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;
      }
      return false;
    }
    const tensor_type type = static_cast<tensor_type>(record.type);
    if (!validate_row_data(type, record.data, record.data_size)) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;
      }
      return false;
    }
  }
  return true;
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
  for (uint16_t i = 0; i < ctx.mapped_count; ++i) {
    if (ctx.mapped_splits[i] != nullptr && ctx.mapped_sizes[i] > 0) {
      (void)munmap(const_cast<void *>(ctx.mapped_splits[i]),
                   static_cast<size_t>(ctx.mapped_sizes[i]));
    }
  }
  if (ctx.mapped_count == 0 && ctx.mapped_data != nullptr && ctx.mapped_size > 0) {
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
  model.params = {};
  model.vocab_data = {};
  model.weights_split_count = 1;
  model.weights_split_sizes = {};
  model.weights_split_offsets = {};
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
    const reader & r, context & ctx, emel::model::data & model,
    const int64_t n_kv, int32_t & out_error) {
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
      if (key_equals(key, key_len, "tokenizer.ggml.tokens")) {
        if (type != value_type::k_string ||
            count > static_cast<uint64_t>(emel::model::data::k_max_vocab_tokens)) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        model.vocab_data.n_tokens = static_cast<uint32_t>(count);
        for (uint64_t idx = 0; idx < count; ++idx) {
          uint64_t token_len = 0;
          uint32_t offset = model.vocab_data.token_bytes_used;
          if (!r.read_string(model.vocab_data.token_storage.data() + offset,
                             model.vocab_data.token_storage.size() - offset, token_len)) {
            out_error = EMEL_ERR_PARSE_FAILED;
            return false;
          }
          if (token_len == 0) {
            out_error = EMEL_ERR_MODEL_INVALID;
            return false;
          }
          model.vocab_data.entries[idx].text_offset = offset;
          model.vocab_data.entries[idx].text_length = static_cast<uint32_t>(token_len);
          model.vocab_data.token_bytes_used =
            static_cast<uint32_t>(offset + token_len + 1);
        }
        continue;
      }
      if (key_equals(key, key_len, "tokenizer.ggml.scores")) {
        if (type != value_type::k_f32 ||
            count > static_cast<uint64_t>(emel::model::data::k_max_vocab_tokens)) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (model.vocab_data.n_tokens < count) {
          model.vocab_data.n_tokens = static_cast<uint32_t>(count);
        }
        for (uint64_t idx = 0; idx < count; ++idx) {
          float score = 0.0f;
          if (!r.read(score)) {
            out_error = EMEL_ERR_PARSE_FAILED;
            return false;
          }
          model.vocab_data.entries[idx].score = score;
        }
        continue;
      }
      if (key_equals(key, key_len, "tokenizer.ggml.token_type")) {
        if (type != value_type::k_i32 ||
            count > static_cast<uint64_t>(emel::model::data::k_max_vocab_tokens)) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (model.vocab_data.n_tokens < count) {
          model.vocab_data.n_tokens = static_cast<uint32_t>(count);
        }
        for (uint64_t idx = 0; idx < count; ++idx) {
          int32_t token_type = 0;
          if (!r.read(token_type)) {
            out_error = EMEL_ERR_PARSE_FAILED;
            return false;
          }
          model.vocab_data.entries[idx].type = token_type;
        }
        continue;
      }
      if (key_equals(key, key_len, "tokenizer.ggml.merges")) {
        if (type != value_type::k_string ||
            count > static_cast<uint64_t>(emel::model::data::k_max_merges)) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        model.vocab_data.n_merges = static_cast<uint32_t>(count);
        for (uint64_t idx = 0; idx < count; ++idx) {
          uint64_t merge_len = 0;
          uint32_t offset = model.vocab_data.merge_bytes_used;
          if (!r.read_string(model.vocab_data.merge_storage.data() + offset,
                             model.vocab_data.merge_storage.size() - offset, merge_len)) {
            out_error = EMEL_ERR_PARSE_FAILED;
            return false;
          }
          if (merge_len == 0) {
            out_error = EMEL_ERR_MODEL_INVALID;
            return false;
          }
          model.vocab_data.merge_offsets[idx] = offset;
          model.vocab_data.merge_lengths[idx] = static_cast<uint32_t>(merge_len);
          model.vocab_data.merge_bytes_used =
            static_cast<uint32_t>(offset + merge_len + 1);
        }
        continue;
      }
      if (key_equals(key, key_len, "tokenizer.ggml.precompiled_charsmap")) {
        if (type != value_type::k_u8 ||
            count > static_cast<uint64_t>(
              emel::model::data::k_max_precompiled_charsmap_bytes)) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        model.vocab_data.precompiled_charsmap_size = static_cast<uint32_t>(count);
        if (!r.read_raw(model.vocab_data.precompiled_charsmap.data(),
                        static_cast<size_t>(count))) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
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

    if (key_equals(key, key_len, "tokenizer.ggml.model")) {
      if (type != value_type::k_string) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      uint64_t len = 0;
      if (!r.read_string(model.vocab_data.tokenizer_model.data(),
                         model.vocab_data.tokenizer_model.size(), len)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }

    if (key_equals(key, key_len, "tokenizer.ggml.pre")) {
      if (type != value_type::k_string) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      uint64_t len = 0;
      if (!r.read_string(model.vocab_data.tokenizer_pre.data(),
                         model.vocab_data.tokenizer_pre.size(), len)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
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

    uint64_t prefix_len = 0;
    if (key_has_suffix(key, key_len, ".context_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_ctx = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix(key, key_len, ".embedding_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_embd = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix(key, key_len, ".embedding_length_out", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_embd_out = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix(key, key_len, ".feed_forward_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_ff = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix(key, key_len, ".attention.head_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_head = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix(key, key_len, ".attention.head_count_kv", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_head_kv = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix(key, key_len, ".rope.dimension_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_rot = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix(key, key_len, ".rope.freq_base", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_freq_base = value;
      continue;
    }
    if (key_has_suffix(key, key_len, ".rope.scale_linear", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scale_linear = value;
      continue;
    }
    if (key_has_suffix(key, key_len, ".rope.scaling.factor", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scaling_factor = value;
      continue;
    }
    if (key_has_suffix(key, key_len, ".rope.scaling.attn_factor", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scaling_attn_factor = value;
      continue;
    }
    if (key_has_suffix(key, key_len, ".rope.scaling.original_context_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scaling_orig_ctx_len = static_cast<int32_t>(value);
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

    if (key_equals(key, key_len, "tokenizer.ggml.token_type_count")) {
      uint32_t count_value = 0;
      if (!parse_u32_value(r, type, count_value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.n_token_types = count_value;
      continue;
    }

    if (key_equals(key, key_len, "tokenizer.ggml.bos_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.bos_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.eos_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.eos_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.eot_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.eot_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.eom_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.eom_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.unknown_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.unk_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.seperator_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.sep_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.padding_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.pad_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.cls_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.cls_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.mask_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.mask_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.prefix_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.prefix_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.suffix_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.suffix_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.middle_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.middle_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.fim_pre_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.fim_pre_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.fim_suf_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.fim_suf_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.fim_mid_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.fim_mid_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.fim_pad_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.fim_pad_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.fim_rep_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.fim_rep_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.fim_sep_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.fim_sep_id = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.add_bos_token")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.add_bos = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.add_eos_token")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.add_eos = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.add_sep_token")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.add_sep = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.add_space_prefix")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.add_space_prefix = value;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.remove_extra_whitespaces")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.remove_extra_whitespaces = value;
      continue;
    }

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

inline bool parse_kv_skip(
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
    if (!skip_value(r, type, count)) {
      out_error = EMEL_ERR_PARSE_FAILED;
      return false;
    }
  }
  return true;
}

inline bool parse_tensors(
    const reader & r, context & ctx, emel::model::data & model,
    const int64_t n_tensors, const uint32_t start_index, const uint16_t file_index,
    const uint64_t base_offset, int32_t & out_error) {
  if (n_tensors < 0 || n_tensors > emel::model::data::k_max_tensors) {
    out_error = EMEL_ERR_MODEL_INVALID;
    return false;
  }
  if (start_index > emel::model::data::k_max_tensors ||
      start_index + static_cast<uint32_t>(n_tensors) > emel::model::data::k_max_tensors) {
    out_error = EMEL_ERR_MODEL_INVALID;
    return false;
  }
  model.n_tensors = std::max(model.n_tensors, start_index + static_cast<uint32_t>(n_tensors));

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
    emel::model::data::tensor_record & record =
      model.tensors[static_cast<size_t>(start_index + static_cast<uint32_t>(i))];
    record.name_offset = name_offset;
    record.name_length = static_cast<uint32_t>(name_len);
    record.type = type_raw;
    record.n_dims = static_cast<int32_t>(n_dims);
    record.dims = dims;
    record.data_offset = base_offset + offset;
    record.file_offset = offset;
    record.data_size = tensor_bytes;
    record.data = nullptr;
    record.file_index = file_index;

    uint64_t padded = align_up_u64(tensor_bytes, ctx.alignment);
  if (add_overflow_u64(expected_offset, padded, expected_offset)) {
      out_error = EMEL_ERR_MODEL_INVALID;  // GCOVR_EXCL_LINE
      return false;  // GCOVR_EXCL_LINE
  }
  }
  ctx.data_size = expected_offset;
  return true;
}

inline bool has_split_suffix(const std::string_view path, size_t & prefix_len) {
  constexpr size_t k_digits = 5;
  constexpr size_t k_suffix = 1 + k_digits + 4 + k_digits + 5;
  if (path.size() < k_suffix) {
    return false;
  }
  const size_t start = path.size() - k_suffix;
  if (path[start] != '-') {
    return false;
  }
  for (size_t i = 0; i < k_digits; ++i) {
    if (!std::isdigit(static_cast<unsigned char>(path[start + 1 + i]))) {
      return false;
    }
  }
  if (std::memcmp(path.data() + start + 1 + k_digits, "-of-", 4) != 0) {
    return false;
  }
  for (size_t i = 0; i < k_digits; ++i) {
    if (!std::isdigit(static_cast<unsigned char>(path[start + 1 + k_digits + 4 + i]))) {
      return false;
    }
  }
  if (std::memcmp(path.data() + path.size() - 5, ".gguf", 5) != 0) {
    return false;
  }
  prefix_len = start;
  return true;
}

inline bool format_split_path(const std::string_view path, const uint16_t index,
                              const uint16_t count, char * out, const size_t capacity) {
  if (out == nullptr || capacity == 0) {
    return false;
  }
  size_t prefix_len = 0;
  if (!has_split_suffix(path, prefix_len)) {
    if (path.size() < 5 || std::memcmp(path.data() + path.size() - 5, ".gguf", 5) != 0) {
      return false;
    }
    prefix_len = path.size() - 5;
  }
  const int written = std::snprintf(out, capacity, "%.*s-%05u-of-%05u.gguf",
                                    static_cast<int>(prefix_len), path.data(),
                                    static_cast<unsigned>(index + 1),
                                    static_cast<unsigned>(count));
  return written > 0 && static_cast<size_t>(written) < capacity;
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
  if (!parse_kv(r, *ctx, ev.model_data, n_kv, parse_error)) {
    if (err_out != nullptr) {
      *err_out = parse_error;
    }
    reset_context(*ctx);
    return false;
  }
  if (!parse_tensors(r, *ctx, ev.model_data, n_tensors, 0, 0, 0, parse_error)) {
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
  ev.model_data.weights_split_count = ctx->split_count;
  if (ctx->split_count == 0 || ctx->split_count > emel::model::data::k_max_split_files) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    reset_context(*ctx);
    return false;
  }
  ev.model_data.weights_split_sizes[0] = ctx->data_size;
  ev.model_data.weights_split_offsets[0] = ctx->data_offset;
  uint64_t global_offset = ctx->data_size;

  if (ctx->split_count > 1) {
    if (ev.model_path.empty()) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_INVALID_ARGUMENT;
      }
      reset_context(*ctx);
      return false;
    }
    uint32_t start_index = ev.model_data.n_tensors;
    for (uint16_t split_idx = 1; split_idx < ctx->split_count; ++split_idx) {
      char split_path[k_max_path_length] = {};
      if (!format_split_path(ev.model_path, split_idx, ctx->split_count,
                             split_path, sizeof(split_path))) {
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_INVALID_ARGUMENT;
        }
        reset_context(*ctx);
        return false;
      }
      context split_ctx{};
      split_ctx.file = std::fopen(split_path, "rb");
      if (split_ctx.file == nullptr) {
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_IO;
        }
        reset_context(*ctx);
        return false;
      }
      reader split_reader{split_ctx.file};
      int64_t split_tensors = 0;
      int64_t split_kv = 0;
      if (!parse_header(split_reader, split_ctx, split_tensors, split_kv)) {
        std::fclose(split_ctx.file);
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_FORMAT_UNSUPPORTED;
        }
        reset_context(*ctx);
        return false;
      }
      if (!parse_kv_skip(split_reader, split_ctx, split_kv, parse_error)) {
        std::fclose(split_ctx.file);
        if (err_out != nullptr) {
          *err_out = parse_error;
        }
        reset_context(*ctx);
        return false;
      }
      if (!parse_tensors(split_reader, split_ctx, ev.model_data, split_tensors,
                         start_index, split_idx, global_offset, parse_error)) {
        std::fclose(split_ctx.file);
        if (err_out != nullptr) {
          *err_out = parse_error;
        }
        reset_context(*ctx);
        return false;
      }
      const long split_pos = std::ftell(split_ctx.file);
      if (split_pos < 0) {
        std::fclose(split_ctx.file);
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
        }
        reset_context(*ctx);
        return false;  // GCOVR_EXCL_LINE
      }
      const uint64_t split_aligned =
        align_up_u64(static_cast<uint64_t>(split_pos), split_ctx.alignment);
      if (std::fseek(split_ctx.file, static_cast<long>(split_aligned), SEEK_SET) != 0) {
        std::fclose(split_ctx.file);
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
        }
        reset_context(*ctx);
        return false;  // GCOVR_EXCL_LINE
      }
      split_ctx.data_offset = split_aligned;
      ev.model_data.weights_split_sizes[split_idx] = split_ctx.data_size;
      ev.model_data.weights_split_offsets[split_idx] = split_ctx.data_offset;
      global_offset += split_ctx.data_size;
      start_index = ev.model_data.n_tensors;
      std::fclose(split_ctx.file);
    }
  }
  ev.model_data.weights_size = global_offset;
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
  ev.model->params.n_layer = static_cast<int32_t>(ctx->block_count);
  if (ev.model->vocab_data.n_tokens > 0) {
    ev.model->params.n_vocab = static_cast<int32_t>(ev.model->vocab_data.n_tokens);
  }
  if (ev.model->params.n_ctx <= 0 || ev.model->params.n_embd <= 0 ||
      ev.model->params.n_layer <= 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  return true;
}

inline bool parse_vocab(const emel::model::parser::event::parse_model & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (ev.model == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  if (ev.model->vocab_data.n_tokens == 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  for (uint32_t i = 0; i < ev.model->vocab_data.n_tokens; ++i) {
    if (ev.model->vocab_data.entries[i].text_length == 0) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;
      }
      return false;
    }
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
  const uint16_t split_count = request->model_data.weights_split_count;
  if (split_count == 0 || split_count > emel::model::data::k_max_split_files) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  if (split_count == 1 && request->model_data.weights_split_sizes[0] == 0) {
    request->model_data.weights_split_sizes[0] = ctx->data_size;
    request->model_data.weights_split_offsets[0] = ctx->data_offset;
    request->model_data.weights_size = ctx->data_size;
  }
  if (ev.no_alloc) {
    if (bytes_total != nullptr) {
      *bytes_total = request->model_data.weights_size;
    }
    if (bytes_done != nullptr) {
      *bytes_done = 0;
    }
    return true;
  }
  if (ctx->mapped_count == 0) {
    for (uint16_t split_idx = 0; split_idx < split_count; ++split_idx) {
      std::FILE * file = nullptr;
      bool owns_file = false;
      if (split_idx == 0 && ctx->file != nullptr && request->model_path.empty()) {
        file = ctx->file;
      } else {
        if (request->model_path.empty() || request->model_path.size() >= k_max_path_length) {
          if (err_out != nullptr) {
            *err_out = EMEL_ERR_INVALID_ARGUMENT;
          }
          return false;
        }
        char path[k_max_path_length] = {};
        if (split_count == 1) {
          std::memcpy(path, request->model_path.data(), request->model_path.size());
          path[request->model_path.size()] = '\0';
        } else if (!format_split_path(request->model_path, split_idx, split_count,
                                      path, sizeof(path))) {
          if (err_out != nullptr) {
            *err_out = EMEL_ERR_INVALID_ARGUMENT;
          }
          return false;
        }
        file = std::fopen(path, "rb");
        if (file == nullptr) {
          if (err_out != nullptr) {
            *err_out = EMEL_ERR_IO;
          }
          return false;
        }
        owns_file = true;
      }
      if (std::fseek(file, 0, SEEK_END) != 0) {
        if (owns_file) {
          std::fclose(file);
        }
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
        }
        return false;  // GCOVR_EXCL_LINE
      }
      const long file_size = std::ftell(file);
      if (file_size < 0) {
        if (owns_file) {
          std::fclose(file);
        }
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
        }
        return false;  // GCOVR_EXCL_LINE
      }
      std::rewind(file);
      const int fd = fileno(file);
      void * data = mmap(nullptr, static_cast<size_t>(file_size), PROT_READ, MAP_PRIVATE, fd, 0);
      if (owns_file) {
        std::fclose(file);
      }
      if (data == MAP_FAILED) {
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
        }
        return false;  // GCOVR_EXCL_LINE
      }
      ctx->mapped_splits[split_idx] = data;
      ctx->mapped_sizes[split_idx] = static_cast<uint64_t>(file_size);
    }
    ctx->mapped_count = split_count;
    ctx->mapped_data = ctx->mapped_splits[0];
    ctx->mapped_size = ctx->mapped_sizes[0];
  }
  for (uint16_t split_idx = 0; split_idx < split_count; ++split_idx) {
    const uint64_t mapped_size = ctx->mapped_sizes[split_idx];
    const uint64_t data_offset = request->model_data.weights_split_offsets[split_idx];
    const uint64_t data_size = request->model_data.weights_split_sizes[split_idx];
    if (mapped_size < data_offset || mapped_size - data_offset < data_size) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;
      }
      return false;
    }
  }

  request->model_data.weights_data =
    static_cast<const uint8_t *>(ctx->mapped_splits[0]) +
    request->model_data.weights_split_offsets[0];
  request->model_data.weights_mapped = true;
  for (uint32_t i = 0; i < request->model_data.n_tensors; ++i) {
    auto & record = request->model_data.tensors[i];
    const uint16_t file_index = record.file_index;
    record.data = static_cast<const uint8_t *>(ctx->mapped_splits[file_index]) +
      request->model_data.weights_split_offsets[file_index] + record.file_offset;
  }
  if (ev.check_tensors) {
    if (!validate_tensor_data(request->model_data, err_out)) {
      return false;
    }
  }
  if (bytes_total != nullptr) {
    *bytes_total = request->model_data.weights_size;
  }
  if (bytes_done != nullptr) {
    *bytes_done = request->model_data.weights_size;
  }
  if (ev.progress_callback != nullptr) {
    (void)ev.progress_callback(1.0f, ev.progress_user_data);
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
  const uint16_t split_count = request->model_data.weights_split_count;
  if (split_count == 0 || split_count > emel::model::data::k_max_split_files) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  if (split_count == 1 && request->model_data.weights_split_sizes[0] == 0) {
    request->model_data.weights_split_sizes[0] = ctx->data_size;
    request->model_data.weights_split_offsets[0] = ctx->data_offset;
    request->model_data.weights_size = ctx->data_size;
  }
  if (ev.no_alloc) {
    if (bytes_total != nullptr) {
      *bytes_total = request->model_data.weights_size;
    }
    if (bytes_done != nullptr) {
      *bytes_done = 0;
    }
    return true;
  }
  if (request->weights_buffer == nullptr ||
      request->weights_buffer_size < request->model_data.weights_size) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_INVALID_ARGUMENT;
    }
    return false;
  }
  uint64_t total_size = request->model_data.weights_size;
  if (bytes_total != nullptr) {
    *bytes_total = total_size;
  }
  uint64_t done = 0;
  constexpr size_t k_chunk_size = 1024 * 1024;
  uint64_t base_offset = 0;
  for (uint16_t split_idx = 0; split_idx < split_count; ++split_idx) {
    std::FILE * file = ctx->file;
    bool owns_file = false;
    if (split_idx > 0 || file == nullptr) {
      if (request->model_path.empty() || request->model_path.size() >= k_max_path_length) {
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_INVALID_ARGUMENT;
        }
        return false;
      }
      char path[k_max_path_length] = {};
      if (split_count == 1) {
        std::memcpy(path, request->model_path.data(), request->model_path.size());
        path[request->model_path.size()] = '\0';
      } else if (!format_split_path(request->model_path, split_idx, split_count,
                                    path, sizeof(path))) {
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_INVALID_ARGUMENT;
        }
        return false;
      }
      file = std::fopen(path, "rb");
      if (file == nullptr) {
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_IO;
        }
        return false;
      }
      owns_file = true;
    }
    const uint64_t data_offset = request->model_data.weights_split_offsets[split_idx];
    const uint64_t data_size = request->model_data.weights_split_sizes[split_idx];
    if (data_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
      if (owns_file) {
        std::fclose(file);
      }
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;  // GCOVR_EXCL_LINE
      }
      return false;  // GCOVR_EXCL_LINE
    }
    if (std::fseek(file, static_cast<long>(data_offset), SEEK_SET) != 0) {
      if (owns_file) {
        std::fclose(file);
      }
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
      }
      return false;  // GCOVR_EXCL_LINE
    }
    uint8_t * dst =
      static_cast<uint8_t *>(request->weights_buffer) + base_offset;
    uint64_t remaining = data_size;
    while (remaining > 0) {
      const size_t chunk = static_cast<size_t>(
        std::min<uint64_t>(remaining, k_chunk_size));
      if (std::fread(dst, 1, chunk, file) != chunk) {
        if (owns_file) {
          std::fclose(file);
        }
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
        }
        return false;  // GCOVR_EXCL_LINE
      }
      dst += chunk;
      remaining -= chunk;
      done += chunk;
      if (bytes_done != nullptr) {
        *bytes_done = done;
      }
      if (ev.progress_callback != nullptr && total_size > 0) {
        const float progress =
          static_cast<float>(done) / static_cast<float>(total_size);
        if (!ev.progress_callback(progress, ev.progress_user_data)) {
          if (owns_file) {
            std::fclose(file);
          }
          if (err_out != nullptr) {
            *err_out = EMEL_ERR_BACKEND;
          }
          return false;
        }
      }
    }
    if (owns_file) {
      std::fclose(file);
    }
    base_offset += data_size;
  }
  const auto * base = static_cast<const uint8_t *>(request->weights_buffer);
  request->model_data.weights_data = base;
  request->model_data.weights_size = total_size;
  request->model_data.weights_mapped = false;
  for (uint32_t i = 0; i < request->model_data.n_tensors; ++i) {
    auto & record = request->model_data.tensors[i];
    record.data = base + record.data_offset;
  }
  if (ev.check_tensors) {
    if (!validate_tensor_data(request->model_data, err_out)) {
      return false;
    }
  }
  if (bytes_total != nullptr) {
    *bytes_total = total_size;
  }
  if (bytes_done != nullptr) {
    *bytes_done = done;
  }
  return true;
}

}  // namespace emel::model::gguf
