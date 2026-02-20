#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__linux__)
#include <fcntl.h>
#endif

#include "emel/emel.h"
#include "emel/model/data.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/parser/actions.hpp"
#include "emel/parser/events.hpp"
#include "emel/parser/gguf/context.hpp"
#include "emel/model/weight_loader/events.hpp"

namespace emel::parser::gguf {

inline constexpr uint32_t k_gguf_version = 3;
inline constexpr uint32_t k_max_key_length = 256;
inline constexpr uint32_t k_max_path_length = 1024;
inline constexpr uint32_t k_direct_io_alignment = 4096;
inline constexpr uint32_t k_direct_io_chunk_size = 256 * 1024;

inline constexpr char k_magic[] = "GGUF";
inline constexpr char k_key_architecture[] = "general.architecture";
inline constexpr char k_key_general_type[] = "general.type";
inline constexpr char k_key_general_quant_version[] = "general.quantization_version";
inline constexpr char k_key_general_file_type[] = "general.file_type";
inline constexpr char k_key_general_sampling_sequence[] = "general.sampling.sequence";
inline constexpr char k_key_general_sampling_top_k[] = "general.sampling.top_k";
inline constexpr char k_key_general_sampling_top_p[] = "general.sampling.top_p";
inline constexpr char k_key_general_sampling_min_p[] = "general.sampling.min_p";
inline constexpr char k_key_general_sampling_xtc_prob[] = "general.sampling.xtc_probability";
inline constexpr char k_key_general_sampling_xtc_threshold[] = "general.sampling.xtc_threshold";
inline constexpr char k_key_general_sampling_temp[] = "general.sampling.temp";
inline constexpr char k_key_general_sampling_penalty_last_n[] = "general.sampling.penalty_last_n";
inline constexpr char k_key_general_sampling_penalty_repeat[] = "general.sampling.penalty_repeat";
inline constexpr char k_key_general_sampling_mirostat[] = "general.sampling.mirostat";
inline constexpr char k_key_general_sampling_mirostat_tau[] = "general.sampling.mirostat_tau";
inline constexpr char k_key_general_sampling_mirostat_eta[] = "general.sampling.mirostat_eta";
inline constexpr char k_key_general_name[] = "general.name";
inline constexpr char k_key_general_author[] = "general.author";
inline constexpr char k_key_general_version[] = "general.version";
inline constexpr char k_key_general_organization[] = "general.organization";
inline constexpr char k_key_general_finetune[] = "general.finetune";
inline constexpr char k_key_general_basename[] = "general.basename";
inline constexpr char k_key_general_description[] = "general.description";
inline constexpr char k_key_general_quantized_by[] = "general.quantized_by";
inline constexpr char k_key_general_size_label[] = "general.size_label";
inline constexpr char k_key_general_license[] = "general.license";
inline constexpr char k_key_general_license_name[] = "general.license.name";
inline constexpr char k_key_general_license_link[] = "general.license.link";
inline constexpr char k_key_general_url[] = "general.url";
inline constexpr char k_key_general_doi[] = "general.doi";
inline constexpr char k_key_general_uuid[] = "general.uuid";
inline constexpr char k_key_general_repo_url[] = "general.repo_url";
inline constexpr char k_key_general_source_url[] = "general.source.url";
inline constexpr char k_key_general_source_doi[] = "general.source.doi";
inline constexpr char k_key_general_source_uuid[] = "general.source.uuid";
inline constexpr char k_key_general_source_repo_url[] = "general.source.repo_url";
inline constexpr char k_key_general_source_hf_repo[] = "general.source.huggingface.repository";
inline constexpr char k_key_general_base_model_count[] = "general.base_model.count";
inline constexpr char k_key_general_dataset_count[] = "general.dataset.count";
inline constexpr char k_key_general_tags[] = "general.tags";
inline constexpr char k_key_general_languages[] = "general.languages";
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

inline bool is_little_endian() noexcept {
  const uint16_t value = 0x1;
  return *reinterpret_cast<const uint8_t *>(&value) == 1;
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

inline bool prefix_is_primary_arch(const context & ctx, const char * key,
                                   const uint64_t prefix_len) noexcept {
  if (ctx.architecture_len > 0) {
    return prefix_len == ctx.architecture_len &&
           std::memcmp(key, ctx.architecture.data(), ctx.architecture_len) == 0;
  }
  for (uint64_t i = 0; i < prefix_len; ++i) {
    if (key[i] == '.') {
      return false;
    }
  }
  return true;
}

inline bool key_has_suffix_primary(const context & ctx, const char * key, const uint64_t len,
                                   const char * suffix, uint64_t & prefix_len) noexcept {
  if (!key_has_suffix(key, len, suffix, prefix_len)) {
    return false;
  }
  return prefix_is_primary_arch(ctx, key, prefix_len);
}

inline bool parse_indexed_key(const char * key, const uint64_t len, const char * prefix,
                              const char * suffix, uint32_t & index) noexcept {
  const size_t prefix_len = std::strlen(prefix);
  const size_t suffix_len = std::strlen(suffix);
  if (len <= prefix_len + suffix_len) {
    return false;
  }
  if (std::memcmp(key, prefix, prefix_len) != 0) {
    return false;
  }
  const uint64_t suffix_start = len - suffix_len;
  if (std::memcmp(key + suffix_start, suffix, suffix_len) != 0) {
    return false;
  }
  const uint64_t idx_len = len - prefix_len - suffix_len;
  if (idx_len == 0 || idx_len > 10) {
    return false;
  }
  uint32_t value = 0;
  for (uint64_t i = 0; i < idx_len; ++i) {
    const char c = key[prefix_len + i];
    if (c < '0' || c > '9') {
      return false;
    }
    value = value * 10u + static_cast<uint32_t>(c - '0');
  }
  index = value;
  return true;
}

inline bool metadata_string_equals(const emel::model::data::metadata & meta,
                                   const emel::model::data::metadata::string_view & view,
                                   const char * text, const uint64_t len) noexcept {
  if (view.length != len) {
    return false;
  }
  if (view.length == 0) {
    return len == 0;
  }
  if (view.offset + view.length > meta.blob_bytes_used) {
    return false;
  }
  return std::memcmp(meta.blob.data() + view.offset, text,
                     static_cast<size_t>(len)) == 0;
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

inline bool store_metadata_string(emel::model::data::metadata & meta, const char * src,
                                  const uint64_t len,
                                  emel::model::data::metadata::string_view & out) {
  if (len == 0) {
    out.offset = 0;
    out.length = 0;
    return true;
  }
  if (len > emel::model::data::k_max_metadata_blob_bytes) {
    return false;
  }
  if (meta.blob_bytes_used > emel::model::data::k_max_metadata_blob_bytes - len) {
    return false;
  }
  const uint32_t offset = meta.blob_bytes_used;
  std::memcpy(meta.blob.data() + offset, src, static_cast<size_t>(len));
  meta.blob_bytes_used = offset + static_cast<uint32_t>(len);
  out.offset = offset;
  out.length = static_cast<uint32_t>(len);
  return true;
}

inline bool read_metadata_string(const reader & r, const value_type type,
                                 emel::model::data::metadata & meta,
                                 emel::model::data::metadata::string_view & out) {
  if (type != value_type::k_string) {
    return false;
  }
  uint64_t len = 0;
  if (!r.read(len)) {
    return false;
  }
  if (len == 0) {
    out.offset = 0;
    out.length = 0;
    return true;
  }
  if (len > emel::model::data::k_max_metadata_blob_bytes) {
    return false;
  }
  if (meta.blob_bytes_used > emel::model::data::k_max_metadata_blob_bytes - len) {
    return false;
  }
  const uint32_t offset = meta.blob_bytes_used;
  if (!r.read_raw(meta.blob.data() + offset, static_cast<size_t>(len))) {
    return false;
  }
  meta.blob_bytes_used = offset + static_cast<uint32_t>(len);
  out.offset = offset;
  out.length = static_cast<uint32_t>(len);
  return true;
}

inline bool read_metadata_string_array(const reader & r, const value_type type,
                                       emel::model::data::metadata & meta,
                                       std::array<emel::model::data::metadata::string_view,
                                                  emel::model::data::k_max_metadata_list> & out,
                                       uint32_t & out_count) {
  if (type != value_type::k_array) {
    return false;
  }
  value_type elem_type = value_type::k_count;
  if (!r.read(elem_type)) {
    return false;
  }
  if (elem_type != value_type::k_string) {
    return false;
  }
  uint64_t count = 0;
  if (!r.read(count)) {
    return false;
  }
  if (count > out.size()) {
    return false;
  }
  out_count = static_cast<uint32_t>(count);
  for (uint64_t i = 0; i < count; ++i) {
    if (!read_metadata_string(r, value_type::k_string, meta, out[i])) {
      return false;
    }
  }
  return true;
}

inline bool read_u32_array(const reader & r, const value_type type,
                           std::array<uint32_t, emel::model::data::k_max_metadata_arrays> & out,
                           uint32_t & out_count) {
  if (type != value_type::k_array) {
    return false;
  }
  value_type elem_type = value_type::k_count;
  if (!r.read(elem_type)) {
    return false;
  }
  if (elem_type != value_type::k_u32 && elem_type != value_type::k_i32 &&
      elem_type != value_type::k_u16) {
    return false;
  }
  uint64_t count = 0;
  if (!r.read(count)) {
    return false;
  }
  if (count > out.size()) {
    return false;
  }
  out_count = static_cast<uint32_t>(count);
  for (uint64_t i = 0; i < count; ++i) {
    uint32_t value = 0;
    if (!parse_u32_value(r, elem_type, value)) {
      return false;
    }
    out[i] = value;
  }
  return true;
}

template <size_t N>
inline bool read_i32_array(const reader & r, const value_type type,
                           std::array<int32_t, N> & out, int32_t & out_count) {
  if (type != value_type::k_array) {
    return false;
  }
  value_type elem_type = value_type::k_count;
  if (!r.read(elem_type)) {
    return false;
  }
  if (elem_type != value_type::k_i32 && elem_type != value_type::k_u32 &&
      elem_type != value_type::k_u16) {
    return false;
  }
  uint64_t count = 0;
  if (!r.read(count)) {
    return false;
  }
  if (count > out.size()) {
    return false;
  }
  out_count = static_cast<int32_t>(count);
  for (uint64_t i = 0; i < count; ++i) {
    int32_t value = 0;
    if (!parse_i32_value(r, elem_type, value)) {
      return false;
    }
    out[i] = value;
  }
  return true;
}

template <size_t N>
inline bool read_f32_array(const reader & r, const value_type type,
                           std::array<float, N> & out, uint32_t & out_count) {
  if (type != value_type::k_array) {
    return false;
  }
  value_type elem_type = value_type::k_count;
  if (!r.read(elem_type)) {
    return false;
  }
  if (elem_type != value_type::k_f32) {
    return false;
  }
  uint64_t count = 0;
  if (!r.read(count)) {
    return false;
  }
  if (count > out.size()) {
    return false;
  }
  out_count = static_cast<uint32_t>(count);
  for (uint64_t i = 0; i < count; ++i) {
    float value = 0.0f;
    if (!r.read(value)) {
      return false;
    }
    out[i] = value;
  }
  return true;
}

template <size_t N>
inline bool read_bool_array(const reader & r, const value_type type,
                            std::array<uint8_t, N> & out, uint32_t & out_count) {
  if (type != value_type::k_array) {
    return false;
  }
  value_type elem_type = value_type::k_count;
  if (!r.read(elem_type)) {
    return false;
  }
  if (elem_type != value_type::k_bool && elem_type != value_type::k_u8 &&
      elem_type != value_type::k_i8) {
    return false;
  }
  uint64_t count = 0;
  if (!r.read(count)) {
    return false;
  }
  if (count > out.size()) {
    return false;
  }
  out_count = static_cast<uint32_t>(count);
  for (uint64_t i = 0; i < count; ++i) {
    bool value = false;
    if (!parse_bool_value(r, elem_type, value)) {
      return false;
    }
    out[i] = static_cast<uint8_t>(value ? 1 : 0);
  }
  return true;
}

inline bool read_metadata_string_array_values(
    const reader & r, const value_type elem_type, const uint64_t count,
    emel::model::data::metadata & meta,
    std::array<emel::model::data::metadata::string_view,
               emel::model::data::k_max_metadata_list> & out,
    uint32_t & out_count) {
  if (elem_type != value_type::k_string) {
    return false;
  }
  if (count > out.size()) {
    return false;
  }
  out_count = static_cast<uint32_t>(count);
  for (uint64_t i = 0; i < count; ++i) {
    if (!read_metadata_string(r, value_type::k_string, meta, out[i])) {
      return false;
    }
  }
  return true;
}

inline bool read_u32_array_values(
    const reader & r, const value_type elem_type, const uint64_t count,
    std::array<uint32_t, emel::model::data::k_max_metadata_arrays> & out,
    uint32_t & out_count) {
  if (elem_type != value_type::k_u32 && elem_type != value_type::k_i32 &&
      elem_type != value_type::k_u16) {
    return false;
  }
  if (count > out.size()) {
    return false;
  }
  out_count = static_cast<uint32_t>(count);
  for (uint64_t i = 0; i < count; ++i) {
    uint32_t value = 0;
    if (!parse_u32_value(r, elem_type, value)) {
      return false;
    }
    out[i] = value;
  }
  return true;
}

template <size_t N>
inline bool read_i32_array_values(const reader & r, const value_type elem_type,
                                  const uint64_t count, std::array<int32_t, N> & out,
                                  int32_t & out_count) {
  if (elem_type != value_type::k_i32 && elem_type != value_type::k_u32 &&
      elem_type != value_type::k_u16) {
    return false;
  }
  if (count > out.size()) {
    return false;
  }
  out_count = static_cast<int32_t>(count);
  for (uint64_t i = 0; i < count; ++i) {
    int32_t value = 0;
    if (!parse_i32_value(r, elem_type, value)) {
      return false;
    }
    out[i] = value;
  }
  return true;
}

template <size_t N>
inline bool read_f32_array_values(const reader & r, const value_type elem_type,
                                  const uint64_t count, std::array<float, N> & out,
                                  uint32_t & out_count) {
  if (elem_type != value_type::k_f32) {
    return false;
  }
  if (count > out.size()) {
    return false;
  }
  out_count = static_cast<uint32_t>(count);
  for (uint64_t i = 0; i < count; ++i) {
    float value = 0.0f;
    if (!r.read(value)) {
      return false;
    }
    out[i] = value;
  }
  return true;
}

template <size_t N>
inline bool read_bool_array_values(const reader & r, const value_type elem_type,
                                   const uint64_t count, std::array<uint8_t, N> & out,
                                   uint32_t & out_count) {
  if (elem_type != value_type::k_bool && elem_type != value_type::k_u8 &&
      elem_type != value_type::k_i8) {
    return false;
  }
  if (count > out.size()) {
    return false;
  }
  out_count = static_cast<uint32_t>(count);
  for (uint64_t i = 0; i < count; ++i) {
    bool value = false;
    if (!parse_bool_value(r, elem_type, value)) {
      return false;
    }
    out[i] = static_cast<uint8_t>(value ? 1 : 0);
  }
  return true;
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

inline std::string_view metadata_view(
    const emel::model::data::metadata & meta,
    const emel::model::data::metadata::string_view & view) {
  if (view.length == 0) {
    return {};
  }
  if (view.offset + view.length > meta.blob_bytes_used) {
    return {};
  }
  return std::string_view(meta.blob.data() + view.offset, view.length);
}

template <size_t N>
inline std::string_view array_view(const std::array<char, N> & data) {
  size_t len = 0;
  while (len < N && data[len] != '\0') {
    ++len;
  }
  return std::string_view(data.data(), len);
}

inline bool token_type_is_special(const int32_t type) {
  return type == 2 || type == 3 || type == 4;
}

inline int32_t find_token_id_by_text(
    const emel::model::data::vocab & vocab,
    const std::string_view token) {
  if (token.empty()) {
    return -1;
  }
  for (uint32_t i = 0; i < vocab.n_tokens; ++i) {
    const auto & entry = vocab.entries[i];
    if (entry.text_length != token.size()) {
      continue;
    }
    if (entry.text_length == 0) {
      continue;
    }
    const char * text = vocab.token_storage.data() + entry.text_offset;
    if (std::memcmp(text, token.data(), token.size()) == 0) {
      return static_cast<int32_t>(i);
    }
  }
  return -1;
}

inline void set_vocab_flag(
    emel::model::data::vocab & vocab,
    std::array<uint8_t, emel::model::data::vocab::k_attr_flag_bytes> & flags,
    const int32_t id,
    const bool value) {
  if (id < 0 || id >= static_cast<int32_t>(vocab.n_tokens)) {
    return;
  }
  const uint32_t idx = static_cast<uint32_t>(id);
  const uint32_t byte = idx >> 3;
  const uint8_t mask = static_cast<uint8_t>(1u << (idx & 7u));
  if (value) {
    flags[byte] = static_cast<uint8_t>(flags[byte] | mask);
  } else {
    flags[byte] = static_cast<uint8_t>(flags[byte] & static_cast<uint8_t>(~mask));
  }
}

inline bool contains_any(
    const std::string & value,
    const std::string_view * tokens,
    const size_t count) {
  for (size_t i = 0; i < count; ++i) {
    const std::string_view token = tokens[i];
    if (token.empty()) {
      continue;
    }
    if (value.find(token) != std::string::npos) {
      return true;
    }
  }
  return false;
}

inline void finalize_vocab_attrs(emel::model::data & model, const context &ctx) {
  emel::model::data::vocab & vocab = model.vocab_data;
  if (vocab.n_tokens == 0) {
    return;
  }
  std::fill(vocab.lstrip_flags.begin(), vocab.lstrip_flags.end(), 0);
  std::fill(vocab.rstrip_flags.begin(), vocab.rstrip_flags.end(), 0);
  bool has_token_type = false;
  for (uint32_t i = 0; i < vocab.n_tokens; ++i) {
    if (vocab.entries[i].type != 0) {
      has_token_type = true;
      break;
    }
  }
  if (!has_token_type) {
    for (uint32_t i = 0; i < vocab.n_tokens; ++i) {
      vocab.entries[i].type = 1;
    }
  }

  std::string model_name(metadata_view(model.meta, model.meta.general_data.name));
  std::string tokenizer_model(array_view(vocab.tokenizer_model));
  std::string tokenizer_pre(array_view(vocab.tokenizer_pre));
  std::string general_arch(array_view(model.architecture_name));

  auto to_lower_ascii = [](const char c) {
    if (c >= 'A' && c <= 'Z') {
      return static_cast<char>(c - 'A' + 'a');
    }
    return c;
  };
  std::transform(model_name.begin(), model_name.end(), model_name.begin(), to_lower_ascii);
  std::transform(tokenizer_model.begin(), tokenizer_model.end(), tokenizer_model.begin(),
                 to_lower_ascii);
  std::transform(tokenizer_pre.begin(), tokenizer_pre.end(), tokenizer_pre.begin(), to_lower_ascii);
  std::transform(general_arch.begin(), general_arch.end(), general_arch.begin(), to_lower_ascii);

  auto model_id_from = [](const std::string_view model) {
    using TokenizerModel = emel::model::data::TokenizerModel;
    if (model.empty() || model == "none" || model == "no_vocab") {
      return TokenizerModel::NONE;
    }
    if (model == "llama") {
      return TokenizerModel::SPM;
    }
    if (model == "gpt2") {
      return TokenizerModel::BPE;
    }
    if (model == "bert") {
      return TokenizerModel::WPM;
    }
    if (model == "t5") {
      return TokenizerModel::UGM;
    }
    if (model == "rwkv") {
      return TokenizerModel::RWKV;
    }
    if (model == "plamo2") {
      return TokenizerModel::PLAMO2;
    }
    return TokenizerModel::UNKNOWN;
  };

  auto pre_id_from = [](const std::string_view pre) {
    using TokenizerPre = emel::model::data::TokenizerPre;
    if (pre.empty() || pre == "default") {
      return TokenizerPre::DEFAULT;
    }
    if (pre == "llama3" || pre == "llama-v3" || pre == "llama-bpe" ||
        pre == "falcon3" || pre == "falcon-h1" || pre == "pixtral" ||
        pre == "midm-2.0" || pre == "lfm2") {
      return TokenizerPre::LLAMA3;
    }
    if (pre == "jais-2") {
      return TokenizerPre::JAIS2;
    }
    if (pre == "dbrx") {
      return TokenizerPre::DBRX;
    }
    if (pre == "smaug-bpe") {
      return TokenizerPre::SMAUG;
    }
    if (pre == "deepseek-llm") {
      return TokenizerPre::DEEPSEEK_LLM;
    }
    if (pre == "deepseek-coder") {
      return TokenizerPre::DEEPSEEK_CODER;
    }
    if (pre == "deepseek-v3") {
      return TokenizerPre::DEEPSEEK3_LLM;
    }
    if (pre == "youtu") {
      return TokenizerPre::YOUTU;
    }
    if (pre == "falcon") {
      return TokenizerPre::FALCON;
    }
    if (pre == "mpt") {
      return TokenizerPre::MPT;
    }
    if (pre == "starcoder") {
      return TokenizerPre::STARCODER;
    }
    if (pre == "gpt-2" || pre == "phi-2" || pre == "jina-es" || pre == "jina-de" ||
        pre == "gigachat" || pre == "jina-v2-es" || pre == "jina-v2-de" ||
        pre == "a.x-4.0" || pre == "mellum" || pre == "modern-bert" ||
        pre == "jina-v1-en" || pre == "jina-v2-code" || pre == "roberta-bpe" ||
        pre == "exaone4") {
      return TokenizerPre::GPT2;
    }
    if (pre == "jais") {
      return TokenizerPre::JAIS;
    }
    if (pre == "refact") {
      return TokenizerPre::REFACT;
    }
    if (pre == "command-r") {
      return TokenizerPre::COMMAND_R;
    }
    if (pre == "qwen2" || pre == "deepseek-r1-qwen" || pre == "kormo" ||
        pre == "megrez") {
      return TokenizerPre::QWEN2;
    }
    if (pre == "qwen35") {
      return TokenizerPre::QWEN35;
    }
    if (pre == "stablelm2") {
      return TokenizerPre::STABLELM2;
    }
    if (pre == "olmo") {
      return TokenizerPre::OLMO;
    }
    if (pre == "poro-chat") {
      return TokenizerPre::PORO;
    }
    if (pre == "glm4" || pre == "chatglm-bpe") {
      return TokenizerPre::CHATGLM4;
    }
    if (pre == "viking") {
      return TokenizerPre::VIKING;
    }
    if (pre == "tekken") {
      return TokenizerPre::TEKKEN;
    }
    if (pre == "smollm") {
      return TokenizerPre::SMOLLM;
    }
    if (pre == "codeshell") {
      return TokenizerPre::CODESHELL;
    }
    if (pre == "bloom") {
      return TokenizerPre::BLOOM;
    }
    if (pre == "gpt3-finnish") {
      return TokenizerPre::GPT3_FINNISH;
    }
    if (pre == "exaone") {
      return TokenizerPre::EXAONE;
    }
    if (pre == "exaone-moe") {
      return TokenizerPre::EXAONE_MOE;
    }
    if (pre == "chameleon") {
      return TokenizerPre::CHAMELEON;
    }
    if (pre == "minerva-7b") {
      return TokenizerPre::MINERVA;
    }
    if (pre == "gpt-4o" || pre == "llama4") {
      return TokenizerPre::GPT4O;
    }
    if (pre == "tiny_aya") {
      return TokenizerPre::TINY_AYA;
    }
    if (pre == "superbpe") {
      return TokenizerPre::SUPERBPE;
    }
    if (pre == "trillion") {
      return TokenizerPre::TRILLION;
    }
    if (pre == "granite-docling") {
      return TokenizerPre::GRANITE_DOCLING;
    }
    if (pre == "bailingmoe" || pre == "bailingmoe2" || pre == "llada-moe") {
      return TokenizerPre::BAILINGMOE;
    }
    if (pre == "seed-coder") {
      return TokenizerPre::SEED_CODER;
    }
    if (pre == "hunyuan") {
      return TokenizerPre::HUNYUAN;
    }
    if (pre == "hunyuan-dense") {
      return TokenizerPre::HUNYUAN_DENSE;
    }
    if (pre == "joyai-llm") {
      return TokenizerPre::JOYAI_LLM;
    }
    if (pre == "kimi-k2") {
      return TokenizerPre::KIMI_K2;
    }
    if (pre == "grok-2") {
      return TokenizerPre::GROK_2;
    }
    if (pre == "afmoe") {
      return TokenizerPre::AFMOE;
    }
    if (pre == "minimax-m2") {
      return TokenizerPre::MINIMAX_M2;
    }
    if (pre == "solar-open") {
      return TokenizerPre::SOLAR_OPEN;
    }
    return TokenizerPre::UNKNOWN;
  };

  vocab.tokenizer_model_id = model_id_from(tokenizer_model);
  vocab.tokenizer_pre_id = pre_id_from(tokenizer_pre);

  auto set_if_missing = [](bool has_key, int32_t &field, int32_t value) {
    if (!has_key) {
      field = value;
    }
  };

  using TokenizerModel = emel::model::data::TokenizerModel;
  switch (vocab.tokenizer_model_id) {
    case TokenizerModel::SPM:
      set_if_missing(ctx.has_bos_id, vocab.bos_id, 1);
      set_if_missing(ctx.has_eos_id, vocab.eos_id, 2);
      set_if_missing(ctx.has_unk_id, vocab.unk_id, 0);
      set_if_missing(ctx.has_sep_id, vocab.sep_id, -1);
      set_if_missing(ctx.has_pad_id, vocab.pad_id, -1);
      set_if_missing(ctx.has_mask_id, vocab.mask_id, -1);
      if (!ctx.has_add_space_prefix) {
        vocab.add_space_prefix = true;
      }
      if (!ctx.has_add_bos) {
        vocab.add_bos = true;
      }
      if (!ctx.has_add_eos) {
        vocab.add_eos = false;
      }
      break;
    case TokenizerModel::WPM:
      set_if_missing(ctx.has_bos_id, vocab.bos_id, 101);
      set_if_missing(ctx.has_eos_id, vocab.eos_id, -1);
      set_if_missing(ctx.has_unk_id, vocab.unk_id, 100);
      set_if_missing(ctx.has_sep_id, vocab.sep_id, 102);
      set_if_missing(ctx.has_pad_id, vocab.pad_id, 0);
      set_if_missing(ctx.has_mask_id, vocab.mask_id, 103);
      if (!ctx.has_add_space_prefix) {
        vocab.add_space_prefix = false;
      }
      if (!ctx.has_add_bos) {
        vocab.add_bos = true;
      }
      if (!ctx.has_add_eos) {
        vocab.add_eos = false;
      }
      if (!ctx.has_add_sep) {
        vocab.add_sep = true;
      }
      break;
    case TokenizerModel::UGM:
      set_if_missing(ctx.has_bos_id, vocab.bos_id, -1);
      set_if_missing(ctx.has_eos_id, vocab.eos_id, 1);
      set_if_missing(ctx.has_unk_id, vocab.unk_id, 2);
      set_if_missing(ctx.has_sep_id, vocab.sep_id, -1);
      set_if_missing(ctx.has_pad_id, vocab.pad_id, 0);
      set_if_missing(ctx.has_mask_id, vocab.mask_id, -1);
      if (!ctx.has_add_bos) {
        vocab.add_bos = false;
      }
      if (!ctx.has_add_eos) {
        vocab.add_eos = true;
      }
      break;
    case TokenizerModel::RWKV:
      set_if_missing(ctx.has_bos_id, vocab.bos_id, -1);
      set_if_missing(ctx.has_eos_id, vocab.eos_id, -1);
      set_if_missing(ctx.has_unk_id, vocab.unk_id, -1);
      set_if_missing(ctx.has_sep_id, vocab.sep_id, -1);
      set_if_missing(ctx.has_pad_id, vocab.pad_id, -1);
      set_if_missing(ctx.has_mask_id, vocab.mask_id, -1);
      if (!ctx.has_add_space_prefix) {
        vocab.add_space_prefix = false;
      }
      if (!ctx.has_add_bos) {
        vocab.add_bos = false;
      }
      if (!ctx.has_add_eos) {
        vocab.add_eos = false;
      }
      break;
    case TokenizerModel::PLAMO2:
      set_if_missing(ctx.has_bos_id, vocab.bos_id, 1);
      set_if_missing(ctx.has_eos_id, vocab.eos_id, 2);
      set_if_missing(ctx.has_unk_id, vocab.unk_id, 0);
      set_if_missing(ctx.has_sep_id, vocab.sep_id, -1);
      set_if_missing(ctx.has_pad_id, vocab.pad_id, 3);
      set_if_missing(ctx.has_mask_id, vocab.mask_id, -1);
      break;
    case TokenizerModel::BPE:
      set_if_missing(ctx.has_bos_id, vocab.bos_id, 11);
      set_if_missing(ctx.has_eos_id, vocab.eos_id, 11);
      set_if_missing(ctx.has_unk_id, vocab.unk_id, -1);
      set_if_missing(ctx.has_sep_id, vocab.sep_id, -1);
      set_if_missing(ctx.has_pad_id, vocab.pad_id, -1);
      set_if_missing(ctx.has_mask_id, vocab.mask_id, -1);
      if (!ctx.has_add_space_prefix) {
        vocab.add_space_prefix = false;
      }
      if (!ctx.has_add_bos) {
        const bool add_bos =
            vocab.tokenizer_pre_id == emel::model::data::TokenizerPre::LLAMA3 ||
            vocab.tokenizer_pre_id == emel::model::data::TokenizerPre::TEKKEN ||
            vocab.tokenizer_pre_id == emel::model::data::TokenizerPre::CHAMELEON;
        vocab.add_bos = add_bos;
      }
      if (!ctx.has_add_eos) {
        vocab.add_eos = false;
      }
      if (!ctx.has_add_sep) {
        const bool add_sep = (tokenizer_pre == "jina-v1-en" ||
                              tokenizer_pre == "jina-v2-code" ||
                              tokenizer_pre == "roberta-bpe");
        vocab.add_sep = add_sep;
      }
      if (!ctx.has_ignore_merges) {
        const bool ignore_merges =
            vocab.tokenizer_pre_id == emel::model::data::TokenizerPre::LLAMA3 ||
            vocab.tokenizer_pre_id == emel::model::data::TokenizerPre::YOUTU ||
            vocab.tokenizer_pre_id == emel::model::data::TokenizerPre::TEKKEN;
        vocab.ignore_merges = ignore_merges;
      }
      break;
    case TokenizerModel::NONE:
    case TokenizerModel::UNKNOWN:
    default:
      break;
  }

  const std::array<std::string_view, 3> jina_tokens = {
    "jina-v2-de", "jina-v2-es", "jina-v2-code",
  };
  const std::array<std::string_view, 2> jina_arch = {
    "nomic-bert-moe", "jina-bert-v3",
  };
  const std::array<std::string_view, 2> phi_tokens = {
    "phi-3", "phi3",
  };
  const std::array<std::string_view, 1> modern_bert_tokens = {
    "modern-bert",
  };

  if (contains_any(tokenizer_pre, jina_tokens.data(), jina_tokens.size()) ||
      contains_any(general_arch, jina_arch.data(), jina_arch.size())) {
    const int32_t mask_id = find_token_id_by_text(vocab, "<mask>");
    if (mask_id >= 0) {
      set_vocab_flag(vocab, vocab.lstrip_flags, mask_id, true);
    }
  } else if (contains_any(model_name, phi_tokens.data(), phi_tokens.size())) {
    for (uint32_t i = 0; i < vocab.n_tokens; ++i) {
      if (token_type_is_special(vocab.entries[i].type)) {
        set_vocab_flag(vocab, vocab.rstrip_flags, static_cast<int32_t>(i), true);
      }
    }
    set_vocab_flag(vocab, vocab.rstrip_flags, find_token_id_by_text(vocab, "</s>"), true);
    set_vocab_flag(vocab, vocab.rstrip_flags, find_token_id_by_text(vocab, "<unk>"), false);
    set_vocab_flag(vocab, vocab.rstrip_flags, find_token_id_by_text(vocab, "<s>"), false);
    set_vocab_flag(vocab, vocab.rstrip_flags, find_token_id_by_text(vocab, "<|endoftext|>"), false);
  } else if (contains_any(model_name, modern_bert_tokens.data(), modern_bert_tokens.size())) {
    const int32_t mask_id = find_token_id_by_text(vocab, "[MASK]");
    if (mask_id >= 0) {
      set_vocab_flag(vocab, vocab.lstrip_flags, mask_id, true);
    }
  }
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

inline bool validate_split_metadata(const context & ctx, const int64_t n_tensors,
                                    int32_t & out_error) noexcept {
  if (n_tensors < 0) {
    out_error = EMEL_ERR_MODEL_INVALID;
    return false;
  }
  if (ctx.split_count == 0 || ctx.split_count > emel::model::data::k_max_split_files) {
    out_error = EMEL_ERR_MODEL_INVALID;
    return false;
  }
  if (ctx.split_no >= ctx.split_count) {
    out_error = EMEL_ERR_MODEL_INVALID;
    return false;
  }
  if (ctx.split_count == 1 && ctx.split_no != 0) {
    out_error = EMEL_ERR_MODEL_INVALID;
    return false;
  }
  if (ctx.split_tensors_count != 0 &&
      ctx.split_tensors_count != static_cast<uint16_t>(n_tensors)) {
    out_error = EMEL_ERR_MODEL_INVALID;
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
    uint64_t prefix_len = 0;
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
      if (key_equals(key, key_len, k_key_general_tags)) {
        if (type != value_type::k_string ||
            count > static_cast<uint64_t>(model.meta.general_data.tags.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_metadata_string_array_values(r, type, count, model.meta,
                                               model.meta.general_data.tags,
                                               model.meta.general_data.tag_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_equals(key, key_len, k_key_general_languages)) {
        if (type != value_type::k_string ||
            count > static_cast<uint64_t>(model.meta.general_data.languages.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_metadata_string_array_values(r, type, count, model.meta,
                                               model.meta.general_data.languages,
                                               model.meta.general_data.language_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_equals(key, key_len, "tokenizer.chat_templates")) {
        if (type != value_type::k_string ||
            count > static_cast<uint64_t>(
              model.meta.tokenizer_data.chat_template_names.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_metadata_string_array_values(
              r, type, count, model.meta,
              model.meta.tokenizer_data.chat_template_names,
              model.meta.tokenizer_data.chat_template_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        for (uint32_t idx = 0; idx < model.meta.tokenizer_data.chat_template_count; ++idx) {
          model.meta.tokenizer_data.chat_template_values[idx].offset = 0;
          model.meta.tokenizer_data.chat_template_values[idx].length = 0;
        }
        continue;
      }
      if (key_equals(key, key_len, "imatrix.datasets")) {
        if (type != value_type::k_string ||
            count > static_cast<uint64_t>(model.meta.imatrix_data.datasets.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_metadata_string_array_values(r, type, count, model.meta,
                                               model.meta.imatrix_data.datasets,
                                               model.meta.imatrix_data.dataset_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_equals(key, key_len, "adapter.alora.invocation_tokens")) {
        if (type != value_type::k_u32 && type != value_type::k_i32 &&
            type != value_type::k_u16) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (count > static_cast<uint64_t>(
              model.meta.adapter_data.alora_invocation_tokens.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_u32_array_values(r, type, count,
                                   model.meta.adapter_data.alora_invocation_tokens,
                                   model.meta.adapter_data.alora_invocation_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_has_suffix_primary(ctx, key, key_len, ".rope.dimension_sections", prefix_len)) {
        if (type != value_type::k_i32 && type != value_type::k_u32 &&
            type != value_type::k_u16) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (count > model.params.rope_dimension_sections.size()) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_i32_array_values(r, type, count,
                                   model.params.rope_dimension_sections,
                                   model.params.rope_dimension_sections_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_has_suffix_primary(ctx, key, key_len, ".classifier.output_labels", prefix_len)) {
        if (type != value_type::k_string ||
            count > static_cast<uint64_t>(model.meta.classifier_data.labels.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_metadata_string_array_values(r, type, count, model.meta,
                                               model.meta.classifier_data.labels,
                                               model.meta.classifier_data.label_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_equals(key, key_len, "clip.vision.image_mean")) {
        if (type != value_type::k_f32 ||
            count > static_cast<uint64_t>(model.meta.clip_vision_data.image_mean.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_f32_array_values(r, type, count, model.meta.clip_vision_data.image_mean,
                                   model.meta.clip_vision_data.image_mean_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_equals(key, key_len, "clip.vision.image_std")) {
        if (type != value_type::k_f32 ||
            count > static_cast<uint64_t>(model.meta.clip_vision_data.image_std.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_f32_array_values(r, type, count, model.meta.clip_vision_data.image_std,
                                   model.meta.clip_vision_data.image_std_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_equals(key, key_len, "clip.vision.wa_layer_indexes")) {
        if (type != value_type::k_u32 && type != value_type::k_i32 &&
            type != value_type::k_u16) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (count > static_cast<uint64_t>(
              model.meta.clip_vision_data.wa_layer_indexes.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_u32_array_values(r, type, count,
                                   model.meta.clip_vision_data.wa_layer_indexes,
                                   model.meta.clip_vision_data.wa_layer_index_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_equals(key, key_len, "clip.vision.is_deepstack_layers")) {
        if (type != value_type::k_bool && type != value_type::k_u8 &&
            type != value_type::k_i8) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (count > static_cast<uint64_t>(
              model.meta.clip_vision_data.deepstack_layers.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_bool_array_values(r, type, count,
                                    model.meta.clip_vision_data.deepstack_layers,
                                    model.meta.clip_vision_data.deepstack_layer_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_has_suffix_primary(ctx, key, key_len, ".attention.sliding_window_pattern",
                                 prefix_len)) {
        if (type != value_type::k_bool && type != value_type::k_u8 &&
            type != value_type::k_i8) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (count > static_cast<uint64_t>(
              model.params.attention_sliding_window_pattern_flags.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_bool_array_values(r, type, count,
                                    model.params.attention_sliding_window_pattern_flags,
                                    model.params.attention_sliding_window_pattern_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_has_suffix_primary(ctx, key, key_len, ".swiglu_clamp_exp", prefix_len)) {
        if (type != value_type::k_f32 ||
            count > static_cast<uint64_t>(model.params.swiglu_clamp_exp.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_f32_array_values(r, type, count, model.params.swiglu_clamp_exp,
                                   model.params.swiglu_clamp_exp_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_has_suffix_primary(ctx, key, key_len, ".swiglu_clamp_shexp", prefix_len)) {
        if (type != value_type::k_f32 ||
            count > static_cast<uint64_t>(model.params.swiglu_clamp_shexp.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_f32_array_values(r, type, count, model.params.swiglu_clamp_shexp,
                                   model.params.swiglu_clamp_shexp_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_equals(key, key_len, "xielu.alpha_p")) {
        if (type != value_type::k_f32 ||
            count > static_cast<uint64_t>(model.meta.xielu_data.alpha_p.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_f32_array_values(r, type, count, model.meta.xielu_data.alpha_p,
                                   model.meta.xielu_data.alpha_p_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_equals(key, key_len, "xielu.alpha_n")) {
        if (type != value_type::k_f32 ||
            count > static_cast<uint64_t>(model.meta.xielu_data.alpha_n.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_f32_array_values(r, type, count, model.meta.xielu_data.alpha_n,
                                   model.meta.xielu_data.alpha_n_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_equals(key, key_len, "xielu.beta")) {
        if (type != value_type::k_f32 ||
            count > static_cast<uint64_t>(model.meta.xielu_data.beta.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_f32_array_values(r, type, count, model.meta.xielu_data.beta,
                                   model.meta.xielu_data.beta_count)) {
          out_error = EMEL_ERR_PARSE_FAILED;
          return false;
        }
        continue;
      }
      if (key_equals(key, key_len, "xielu.eps")) {
        if (type != value_type::k_f32 ||
            count > static_cast<uint64_t>(model.meta.xielu_data.eps.size())) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!read_f32_array_values(r, type, count, model.meta.xielu_data.eps,
                                   model.meta.xielu_data.eps_count)) {
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

    if (key_equals(key, key_len, k_key_general_type)) {
      if (!read_metadata_string(r, type, model.meta, model.meta.general_data.type)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_quant_version)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.quantization_version = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, k_key_general_file_type)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.file_type = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, k_key_general_sampling_sequence)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.sampling_data.sequence)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_sampling_top_k)) {
      int32_t value = 0;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.sampling_data.top_k = value;
      continue;
    }
    if (key_equals(key, key_len, k_key_general_sampling_top_p)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.sampling_data.top_p = value;
      continue;
    }
    if (key_equals(key, key_len, k_key_general_sampling_min_p)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.sampling_data.min_p = value;
      continue;
    }
    if (key_equals(key, key_len, k_key_general_sampling_xtc_prob)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.sampling_data.xtc_probability = value;
      continue;
    }
    if (key_equals(key, key_len, k_key_general_sampling_xtc_threshold)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.sampling_data.xtc_threshold = value;
      continue;
    }
    if (key_equals(key, key_len, k_key_general_sampling_temp)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.sampling_data.temp = value;
      continue;
    }
    if (key_equals(key, key_len, k_key_general_sampling_penalty_last_n)) {
      int32_t value = 0;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.sampling_data.penalty_last_n = value;
      continue;
    }
    if (key_equals(key, key_len, k_key_general_sampling_penalty_repeat)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.sampling_data.penalty_repeat = value;
      continue;
    }
    if (key_equals(key, key_len, k_key_general_sampling_mirostat)) {
      int32_t value = 0;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.sampling_data.mirostat = value;
      continue;
    }
    if (key_equals(key, key_len, k_key_general_sampling_mirostat_tau)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.sampling_data.mirostat_tau = value;
      continue;
    }
    if (key_equals(key, key_len, k_key_general_sampling_mirostat_eta)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.sampling_data.mirostat_eta = value;
      continue;
    }
    if (key_equals(key, key_len, k_key_general_name)) {
      if (!read_metadata_string(r, type, model.meta, model.meta.general_data.name)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_author)) {
      if (!read_metadata_string(r, type, model.meta, model.meta.general_data.author)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_version)) {
      if (!read_metadata_string(r, type, model.meta, model.meta.general_data.version)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_organization)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.organization)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_finetune)) {
      if (!read_metadata_string(r, type, model.meta, model.meta.general_data.finetune)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_basename)) {
      if (!read_metadata_string(r, type, model.meta, model.meta.general_data.basename)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_description)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.description)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_quantized_by)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.quantized_by)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_size_label)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.size_label)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_license)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.license)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_license_name)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.license_name)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_license_link)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.license_link)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_url)) {
      if (!read_metadata_string(r, type, model.meta, model.meta.general_data.url)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_doi)) {
      if (!read_metadata_string(r, type, model.meta, model.meta.general_data.doi)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_uuid)) {
      if (!read_metadata_string(r, type, model.meta, model.meta.general_data.uuid)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_repo_url)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.repo_url)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_source_url)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.source_url)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_source_doi)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.source_doi)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_source_uuid)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.source_uuid)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_source_repo_url)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.source_repo_url)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_source_hf_repo)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.source_hf_repo)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, k_key_general_base_model_count)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      if (value > model.meta.general_data.base_models.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      model.meta.general_data.base_model_count = value;
      continue;
    }
    if (key_equals(key, key_len, k_key_general_dataset_count)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      if (value > model.meta.general_data.datasets.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      model.meta.general_data.dataset_count = value;
      continue;
    }
    uint32_t indexed_id = 0;
    if (parse_indexed_key(key, key_len, "general.base_model.", ".name", indexed_id)) {
      if (indexed_id >= model.meta.general_data.base_models.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.base_models[indexed_id].name)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.base_model_count =
        std::max(model.meta.general_data.base_model_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.base_model.", ".author", indexed_id)) {
      if (indexed_id >= model.meta.general_data.base_models.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.base_models[indexed_id].author)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.base_model_count =
        std::max(model.meta.general_data.base_model_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.base_model.", ".version", indexed_id)) {
      if (indexed_id >= model.meta.general_data.base_models.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.base_models[indexed_id].version)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.base_model_count =
        std::max(model.meta.general_data.base_model_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.base_model.", ".organization", indexed_id)) {
      if (indexed_id >= model.meta.general_data.base_models.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(
            r, type, model.meta,
            model.meta.general_data.base_models[indexed_id].organization)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.base_model_count =
        std::max(model.meta.general_data.base_model_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.base_model.", ".description", indexed_id)) {
      if (indexed_id >= model.meta.general_data.base_models.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(
            r, type, model.meta,
            model.meta.general_data.base_models[indexed_id].description)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.base_model_count =
        std::max(model.meta.general_data.base_model_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.base_model.", ".url", indexed_id)) {
      if (indexed_id >= model.meta.general_data.base_models.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.base_models[indexed_id].url)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.base_model_count =
        std::max(model.meta.general_data.base_model_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.base_model.", ".doi", indexed_id)) {
      if (indexed_id >= model.meta.general_data.base_models.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.base_models[indexed_id].doi)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.base_model_count =
        std::max(model.meta.general_data.base_model_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.base_model.", ".uuid", indexed_id)) {
      if (indexed_id >= model.meta.general_data.base_models.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.base_models[indexed_id].uuid)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.base_model_count =
        std::max(model.meta.general_data.base_model_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.base_model.", ".repo_url", indexed_id)) {
      if (indexed_id >= model.meta.general_data.base_models.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(
            r, type, model.meta,
            model.meta.general_data.base_models[indexed_id].repo_url)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.base_model_count =
        std::max(model.meta.general_data.base_model_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.dataset.", ".name", indexed_id)) {
      if (indexed_id >= model.meta.general_data.datasets.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.datasets[indexed_id].name)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.dataset_count =
        std::max(model.meta.general_data.dataset_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.dataset.", ".author", indexed_id)) {
      if (indexed_id >= model.meta.general_data.datasets.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.datasets[indexed_id].author)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.dataset_count =
        std::max(model.meta.general_data.dataset_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.dataset.", ".version", indexed_id)) {
      if (indexed_id >= model.meta.general_data.datasets.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.datasets[indexed_id].version)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.dataset_count =
        std::max(model.meta.general_data.dataset_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.dataset.", ".organization", indexed_id)) {
      if (indexed_id >= model.meta.general_data.datasets.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(
            r, type, model.meta,
            model.meta.general_data.datasets[indexed_id].organization)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.dataset_count =
        std::max(model.meta.general_data.dataset_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.dataset.", ".description", indexed_id)) {
      if (indexed_id >= model.meta.general_data.datasets.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(
            r, type, model.meta,
            model.meta.general_data.datasets[indexed_id].description)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.dataset_count =
        std::max(model.meta.general_data.dataset_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.dataset.", ".url", indexed_id)) {
      if (indexed_id >= model.meta.general_data.datasets.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.datasets[indexed_id].url)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.dataset_count =
        std::max(model.meta.general_data.dataset_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.dataset.", ".doi", indexed_id)) {
      if (indexed_id >= model.meta.general_data.datasets.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.datasets[indexed_id].doi)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.dataset_count =
        std::max(model.meta.general_data.dataset_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.dataset.", ".uuid", indexed_id)) {
      if (indexed_id >= model.meta.general_data.datasets.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.datasets[indexed_id].uuid)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.dataset_count =
        std::max(model.meta.general_data.dataset_count, indexed_id + 1);
      continue;
    }
    if (parse_indexed_key(key, key_len, "general.dataset.", ".repo_url", indexed_id)) {
      if (indexed_id >= model.meta.general_data.datasets.size()) {
        out_error = EMEL_ERR_MODEL_INVALID;
        return false;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.general_data.datasets[indexed_id].repo_url)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.general_data.dataset_count =
        std::max(model.meta.general_data.dataset_count, indexed_id + 1);
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
      ctx.has_tokenizer_model = true;
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
      ctx.has_tokenizer_pre = true;
      continue;
    }

    if (key_equals(key, key_len, "tokenizer.huggingface.json")) {
      if (!read_metadata_string(r, type, model.meta, model.meta.tokenizer_data.hf_json)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.rwkv.world")) {
      if (!read_metadata_string(r, type, model.meta, model.meta.tokenizer_data.rwkv_world)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.chat_template")) {
      if (!read_metadata_string(
            r, type, model.meta, model.meta.tokenizer_data.chat_template)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_len > 24 && std::memcmp(key, "tokenizer.chat_template.", 24) == 0) {
      const char * name = key + 24;
      const uint64_t name_len = key_len - 24;
      uint32_t slot = model.meta.tokenizer_data.chat_template_count;
      for (uint32_t idx = 0; idx < model.meta.tokenizer_data.chat_template_count; ++idx) {
        if (metadata_string_equals(model.meta,
                                   model.meta.tokenizer_data.chat_template_names[idx],
                                   name, name_len)) {
          slot = idx;
          break;
        }
      }
      if (slot == model.meta.tokenizer_data.chat_template_count) {
        if (slot >= model.meta.tokenizer_data.chat_template_names.size()) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        if (!store_metadata_string(
              model.meta, name, name_len,
              model.meta.tokenizer_data.chat_template_names[slot])) {
          out_error = EMEL_ERR_MODEL_INVALID;
          return false;
        }
        model.meta.tokenizer_data.chat_template_count++;
      }
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.tokenizer_data.chat_template_values[slot])) {
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
      model.meta.general_data.alignment = static_cast<int32_t>(alignment);
      continue;
    }

    if (key_has_suffix_primary(ctx, key, key_len, ".context_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_ctx = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".vocab_size", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_vocab = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".embedding_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_embd = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".embedding_length_out", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_embd_out = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".features_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_features = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".feed_forward_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_ff = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".leading_dense_block_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_leading_dense_block = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".expert_feed_forward_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_expert_ff = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".expert_shared_feed_forward_length",
                               prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_expert_shared_ff = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".expert_chunk_feed_forward_length",
                               prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_expert_chunk_ff = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".use_parallel_residual", prefix_len)) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.use_parallel_residual = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".tensor_data_layout", prefix_len)) {
      if (!read_metadata_string(r, type, model.meta,
                                model.meta.llm_strings_data.tensor_data_layout)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".expert_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_expert = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".expert_used_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_expert_used = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".expert_shared_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_expert_shared = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".expert_group_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_expert_group = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".expert_group_used_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_expert_group_used = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".expert_weights_scale", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.expert_weights_scale = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".expert_weights_norm", prefix_len)) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.expert_weights_norm = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".expert_gating_func", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.expert_gating_func = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".expert_group_scale", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.expert_group_scale = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".experts_per_group", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.experts_per_group = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".moe_every_n_layers", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.moe_every_n_layers = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".nextn_predict_layers", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.nextn_predict_layers = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".n_deepstack_layers", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_deepstack_layers = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".pooling_type", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.pooling_type = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".logit_scale", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.logit_scale = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".decoder_start_token_id", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.decoder_start_token_id = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".decoder_block_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.decoder_block_count = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attn_logit_softcapping", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attn_logit_softcapping = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".router_logit_softcapping", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.router_logit_softcapping = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".final_logit_softcapping", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.final_logit_softcapping = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".swin_norm", prefix_len)) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.swin_norm = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".rescale_every_n_layers", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rescale_every_n_layers = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".time_mix_extra_dim", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.time_mix_extra_dim = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".time_decay_extra_dim", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.time_decay_extra_dim = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".residual_scale", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.residual_scale = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".embedding_scale", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.embedding_scale = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".token_shift_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.token_shift_count = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".interleave_moe_layer_step", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.interleave_moe_layer_step = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".full_attention_interval", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.full_attention_interval = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".activation_sparsity_scale", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.activation_sparsity_scale = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".altup.active_idx", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.altup_active_idx = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".altup.num_inputs", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.altup_num_inputs = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".embedding_length_per_layer_input",
                               prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.embd_length_per_layer_input = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".dense_2_feat_in", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.dense_2_feat_in = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".dense_2_feat_out", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.dense_2_feat_out = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".dense_3_feat_in", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.dense_3_feat_in = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".dense_3_feat_out", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.dense_3_feat_out = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.head_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_head = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.head_count_kv", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_head_kv = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.max_alibi_bias", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_max_alibi_bias = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.clamp_kqv", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_clamp_kqv = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.key_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_key_length = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.value_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_value_length = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.layer_norm_epsilon", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_layer_norm_epsilon = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.layer_norm_rms_epsilon",
                               prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_layer_norm_rms_epsilon = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.group_norm_epsilon",
                               prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_group_norm_epsilon = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.group_norm_groups",
                               prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_group_norm_groups = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.causal", prefix_len)) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_causal = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.q_lora_rank", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_q_lora_rank = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.kv_lora_rank", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_kv_lora_rank = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.decay_lora_rank", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_decay_lora_rank = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.iclr_lora_rank", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_iclr_lora_rank = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len,
                               ".attention.value_residual_mix_lora_rank", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_value_residual_mix_lora_rank = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.gate_lora_rank", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_gate_lora_rank = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.relative_buckets_count",
                               prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_relative_buckets_count = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.sliding_window", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_sliding_window = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.sliding_window_pattern",
                               prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_sliding_window_pattern = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.scale", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_scale = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.output_scale", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_output_scale = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.temperature_length",
                               prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_temperature_length = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.temperature_scale",
                               prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_temperature_scale = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.key_length_mla", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_key_length_mla = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.value_length_mla",
                               prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_value_length_mla = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.indexer.head_count",
                               prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_indexer_head_count = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.indexer.key_length",
                               prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_indexer_key_length = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.indexer.top_k",
                               prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_indexer_top_k = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".attention.shared_kv_layers",
                               prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.attention_shared_kv_layers = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".rope.dimension_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.n_rot = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".rope.freq_base", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_freq_base = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".rope.freq_base_swa", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_freq_base_swa = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".rope.scale_linear", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scale_linear = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".rope.scaling.type", prefix_len)) {
      if (!read_metadata_string(r, type, model.meta, model.meta.rope_data.scaling_type)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".rope.scaling.factor", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scaling_factor = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".rope.scaling.attn_factor", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scaling_attn_factor = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len,
                               ".rope.scaling.original_context_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scaling_orig_ctx_len = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".rope.scaling.finetuned", prefix_len)) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scaling_finetuned = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len,
                               ".rope.scaling.yarn_log_multiplier", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scaling_yarn_log_multiplier = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len,
                               ".rope.scaling.yarn_ext_factor", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scaling_yarn_ext_factor = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len,
                               ".rope.scaling.yarn_attn_factor", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scaling_yarn_attn_factor = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len,
                               ".rope.scaling.yarn_beta_fast", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scaling_yarn_beta_fast = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len,
                               ".rope.scaling.yarn_beta_slow", prefix_len)) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.rope_scaling_yarn_beta_slow = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".ssm.conv_kernel", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.ssm_conv_kernel = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".ssm.inner_size", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.ssm_inner_size = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".ssm.state_size", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.ssm_state_size = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".ssm.time_step_rank", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.ssm_time_step_rank = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".ssm.group_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.ssm_group_count = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".ssm.dt_b_c_rms", prefix_len)) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.ssm_dt_b_c_rms = value;
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".kda.head_dim", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.kda_head_dim = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".wkv.head_size", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.wkv_head_size = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".posnet.embedding_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.posnet_embd = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".posnet.block_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.posnet_block_count = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".convnext.embedding_length", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.convnext_embd = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".convnext.block_count", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.convnext_block_count = static_cast<int32_t>(value);
      continue;
    }
    if (key_has_suffix_primary(ctx, key, key_len, ".shortconv.l_cache", prefix_len)) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.params.shortconv_l_cache = static_cast<int32_t>(value);
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

    if (key_equals(key, key_len, "adapter.type")) {
      if (!read_metadata_string(r, type, model.meta, model.meta.adapter_data.type)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, "adapter.lora.alpha")) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.adapter_data.lora_alpha = value;
      continue;
    }
    if (key_equals(key, key_len, "adapter.lora.task_name")) {
      if (!read_metadata_string(
            r, type, model.meta, model.meta.adapter_data.lora_task_name)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, "adapter.lora.prompt_prefix")) {
      if (!read_metadata_string(
            r, type, model.meta, model.meta.adapter_data.lora_prompt_prefix)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, "imatrix.chunk_count")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.imatrix_data.chunk_count = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "imatrix.chunk_size")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.imatrix_data.chunk_size = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.has_vision_encoder")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_data.has_vision_encoder = value;
      continue;
    }
    if (key_equals(key, key_len, "clip.has_audio_encoder")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_data.has_audio_encoder = value;
      continue;
    }
    if (key_equals(key, key_len, "clip.has_llava_projector")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_data.has_llava_projector = value;
      continue;
    }
    if (key_equals(key, key_len, "clip.projector_type")) {
      if (!read_metadata_string(r, type, model.meta, model.meta.clip_data.projector_type)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, "clip.use_gelu")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_data.use_gelu = value;
      continue;
    }
    if (key_equals(key, key_len, "clip.use_silu")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_data.use_silu = value;
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.projector_type")) {
      if (!read_metadata_string(
            r, type, model.meta, model.meta.clip_vision_data.projector_type)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.image_size")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.image_size = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.image_min_pixels")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.image_min_pixels = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.image_max_pixels")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.image_max_pixels = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.preproc_image_size")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.preproc_image_size = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.patch_size")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.patch_size = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.embedding_length")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.embedding_length = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.feed_forward_length")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.feed_forward_length = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.projection_dim")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.projection_dim = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.block_count")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.block_count = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.spatial_merge_size")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.spatial_merge_size = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.n_wa_pattern")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.n_wa_pattern = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.window_size")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.window_size = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.attention.head_count")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.attention_head_count = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.attention.layer_norm_epsilon")) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.attention_layer_norm_epsilon = value;
      continue;
    }
    if (key_equals(key, key_len, "clip.vision.projector.scale_factor")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_vision_data.projector_scale_factor = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.audio.projector_type")) {
      if (!read_metadata_string(
            r, type, model.meta, model.meta.clip_audio_data.projector_type)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      continue;
    }
    if (key_equals(key, key_len, "clip.audio.num_mel_bins")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_audio_data.num_mel_bins = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.audio.embedding_length")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_audio_data.embedding_length = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.audio.feed_forward_length")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_audio_data.feed_forward_length = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.audio.projection_dim")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_audio_data.projection_dim = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.audio.block_count")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_audio_data.block_count = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.audio.attention.head_count")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_audio_data.attention_head_count = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "clip.audio.attention.layer_norm_epsilon")) {
      float value = 0.0f;
      if (type != value_type::k_f32 || !r.read(value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_audio_data.attention_layer_norm_epsilon = value;
      continue;
    }
    if (key_equals(key, key_len, "clip.audio.projector.stack_factor")) {
      uint32_t value = 0;
      if (!parse_u32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.clip_audio_data.projector_stack_factor = static_cast<int32_t>(value);
      continue;
    }
    if (key_equals(key, key_len, "diffusion.shift_logits")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.meta.diffusion_data.shift_logits = value;
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
      ctx.has_bos_id = true;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.eos_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.eos_id = value;
      ctx.has_eos_id = true;
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
      ctx.has_unk_id = true;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.seperator_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.sep_id = value;
      ctx.has_sep_id = true;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.padding_token_id")) {
      int32_t value = -1;
      if (!parse_i32_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.pad_id = value;
      ctx.has_pad_id = true;
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
      ctx.has_mask_id = true;
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
      ctx.has_add_bos = true;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.add_eos_token")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.add_eos = value;
      ctx.has_add_eos = true;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.add_sep_token")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.add_sep = value;
      ctx.has_add_sep = true;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.add_space_prefix")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.add_space_prefix = value;
      ctx.has_add_space_prefix = true;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.remove_extra_whitespaces")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.remove_extra_whitespaces = value;
      ctx.has_remove_extra_whitespaces = true;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.escape_whitespaces")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.escape_whitespaces = value;
      ctx.has_escape_whitespaces = true;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.treat_whitespace_as_suffix")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.treat_whitespace_as_suffix = value;
      ctx.has_treat_whitespace_as_suffix = true;
      continue;
    }
    if (key_equals(key, key_len, "tokenizer.ggml.ignore_merges")) {
      bool value = false;
      if (!parse_bool_value(r, type, value)) {
        out_error = EMEL_ERR_PARSE_FAILED;
        return false;
      }
      model.vocab_data.ignore_merges = value;
      ctx.has_ignore_merges = true;
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

inline bool has_gguf_extension(const std::string_view path) {
  size_t prefix_len = 0;
  if (has_split_suffix(path, prefix_len)) {
    return true;
  }
  if (path.size() < 5) {
    return false;
  }
  return std::memcmp(path.data() + path.size() - 5, ".gguf", 5) == 0;
}

inline bool match_magic(std::FILE * file, bool & checked) {
  checked = false;
  if (file == nullptr) {
    return false;
  }
  const long offset = std::ftell(file);
  if (offset < 0) {
    return false;
  }
  if (std::fseek(file, 0, SEEK_SET) != 0) {
    return false;
  }
  checked = true;
  std::array<char, sizeof(k_magic) - 1> magic = {};
  const size_t read = std::fread(magic.data(), 1, magic.size(), file);
  (void)std::fseek(file, offset, SEEK_SET);
  if (read != magic.size()) {
    return false;
  }
  return std::memcmp(magic.data(), k_magic, magic.size()) == 0;
}

inline bool can_handle(const emel::model::loader::event::load & ev) {
  if (ev.file_handle != nullptr) {
    auto * file = static_cast<std::FILE *>(ev.file_handle);
    bool checked = false;
    const bool magic_ok = match_magic(file, checked);
    if (checked) {
      return magic_ok;
    }
  }
  if (ev.model_path.empty()) {
    return false;
  }
  return has_gguf_extension(ev.model_path);
}

inline bool map_parser(const emel::model::loader::event::load & ev, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  if (!is_little_endian()) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_FORMAT_UNSUPPORTED;
    }
    return false;
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
  if (n_tensors < 0 || n_kv < 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
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
  if (!validate_split_metadata(*ctx, n_tensors, parse_error)) {
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
      if (!validate_split_metadata(split_ctx, split_tensors, parse_error) ||
          split_ctx.split_count != ctx->split_count ||
          split_ctx.split_no != split_idx) {
        std::fclose(split_ctx.file);
        if (err_out != nullptr) {
          *err_out = parse_error == EMEL_OK ? EMEL_ERR_MODEL_INVALID : parse_error;
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

inline bool parse_architecture(const emel::parser::event::parse_model & ev, int32_t * err_out) {
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

inline bool map_architecture(const emel::parser::event::parse_model & ev, int32_t * err_out) {
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

inline bool parse_hparams(const emel::parser::event::parse_model & ev, int32_t * err_out) {
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

inline bool parse_vocab(const emel::parser::event::parse_model & ev, int32_t * err_out) {
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
  if (ctx == nullptr) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  finalize_vocab_attrs(*ev.model, *ctx);
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

inline bool map_tensors(const emel::parser::event::parse_model & ev, int32_t * err_out) {
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
  if (ev.model_data.n_layers <= 0 && ev.model_data.params.n_layer > 0) {
    ev.model_data.n_layers = ev.model_data.params.n_layer;
  }
  if (ev.model_data.n_layers <= 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  if (ev.model_data.params.n_layer > 0 &&
      ev.model_data.params.n_layer != ev.model_data.n_layers) {
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
  const uint16_t split_count = ev.model_data.weights_split_count;
  if (split_count == 0 || split_count > emel::model::data::k_max_split_files) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  if (ev.model_data.n_tensors == 0 || ev.model_data.weights_size == 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  if (split_count == 1) {
    if (ev.model_data.weights_split_sizes[0] == 0 && ev.model_data.weights_size > 0) {
      ev.model_data.weights_split_sizes[0] = ev.model_data.weights_size;
    }
    if (ev.model_data.weights_split_offsets[0] == 0 && ctx->data_offset > 0) {
      ev.model_data.weights_split_offsets[0] = ctx->data_offset;
    }
  }
  if (ctx->data_offset == 0) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  uint64_t total_weights = 0;
  bool sizes_known = true;
  for (uint16_t split_idx = 0; split_idx < split_count; ++split_idx) {
    const uint64_t split_size = ev.model_data.weights_split_sizes[split_idx];
    if (split_size == 0) {
      sizes_known = false;
      break;
    }
    if (add_overflow_u64(total_weights, split_size, total_weights)) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;
      }
      return false;
    }
  }
  if (sizes_known && total_weights != ev.model_data.weights_size) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  const uint64_t weights_size = ev.model_data.weights_size;
  for (uint32_t i = 0; i < ev.model_data.n_tensors; ++i) {
    const auto & record = ev.model_data.tensors[i];
    if (record.data_size == 0) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;
      }
      return false;
    }
    if (record.file_index >= split_count) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;
      }
      return false;
    }
    uint64_t end = 0;
    if (add_overflow_u64(record.data_offset, record.data_size, end) ||
        end > weights_size) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;
      }
      return false;
    }
    if (sizes_known) {
      const uint64_t split_size = ev.model_data.weights_split_sizes[record.file_index];
      if (add_overflow_u64(record.file_offset, record.data_size, end) ||
          end > split_size) {
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_MODEL_INVALID;
        }
        return false;
      }
    }
  }
  return true;
}

inline bool validate_architecture(
    const emel::model::loader::event::load & ev, int32_t * err_out) {
  emel::parser::event::parse_model request{
    .model = &ev.model_data,
    .architectures = ev.architectures,
    .n_architectures = ev.n_architectures,
    .format_ctx = ev.format_ctx
  };
  return map_architecture(request, err_out);
}

#if defined(_WIN32)
inline bool ensure_mapped(
    const emel::model::loader::event::load &, context &, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_ERR_FORMAT_UNSUPPORTED;
  }
  return false;
}
#else
inline bool ensure_mapped(
    const emel::model::loader::event::load & request, context & ctx, int32_t * err_out) {
  if (err_out != nullptr) {
    *err_out = EMEL_OK;
  }
  const uint16_t split_count = request.model_data.weights_split_count;
  if (split_count == 0 || split_count > emel::model::data::k_max_split_files) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_MODEL_INVALID;
    }
    return false;
  }
  if (split_count == 1 && request.model_data.weights_split_sizes[0] == 0) {
    request.model_data.weights_split_sizes[0] = ctx.data_size;
    request.model_data.weights_split_offsets[0] = ctx.data_offset;
    request.model_data.weights_size = ctx.data_size;
  }
  if (ctx.mapped_count != 0) {
    return true;
  }
  for (uint16_t split_idx = 0; split_idx < split_count; ++split_idx) {
    std::FILE * file = nullptr;
    bool owns_file = false;
    if (split_idx == 0 && ctx.file != nullptr && request.model_path.empty()) {
      file = ctx.file;
    } else {
      if (request.model_path.empty() || request.model_path.size() >= k_max_path_length) {
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_INVALID_ARGUMENT;
        }
        return false;
      }
      char path[k_max_path_length] = {};
      if (split_count == 1) {
        std::memcpy(path, request.model_path.data(), request.model_path.size());
        path[request.model_path.size()] = '\0';
      } else if (!format_split_path(request.model_path, split_idx, split_count,
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
    ctx.mapped_splits[split_idx] = data;
    ctx.mapped_sizes[split_idx] = static_cast<uint64_t>(file_size);
  }
  ctx.mapped_count = split_count;
  ctx.mapped_data = ctx.mapped_splits[0];
  ctx.mapped_size = ctx.mapped_sizes[0];
  return true;
}
#endif

inline bool init_mappings(const emel::model::weight_loader::event::load_weights & ev,
                          int32_t * err_out) {
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
#if defined(_WIN32)
  if (err_out != nullptr) {
    *err_out = EMEL_ERR_FORMAT_UNSUPPORTED;
  }
  return false;
#else
  return ensure_mapped(*request, *ctx, err_out);
#endif
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
  if (!ensure_mapped(*request, *ctx, err_out)) {
    return false;
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
    if (!ev.progress_callback(1.0f, ev.progress_user_data)) {
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_BACKEND;
      }
      return false;
    }
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
  const bool use_direct_io = ev.request_direct_io && ev.direct_io_supported;
#if !defined(__linux__)
  if (use_direct_io) {
    if (err_out != nullptr) {
      *err_out = EMEL_ERR_FORMAT_UNSUPPORTED;
    }
    return false;
  }
#endif
  const bool wants_upload =
    ev.upload_begin != nullptr || ev.upload_chunk != nullptr || ev.upload_end != nullptr;
  if (wants_upload &&
      (ev.upload_begin == nullptr || ev.upload_chunk == nullptr || ev.upload_end == nullptr)) {
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
  if (wants_upload) {
    int32_t upload_err = EMEL_OK;
    if (!ev.upload_begin(ev.upload_ctx, total_size, &upload_err) || upload_err != EMEL_OK) {
      if (err_out != nullptr) {
        *err_out = upload_err == EMEL_OK ? EMEL_ERR_BACKEND : upload_err;
      }
      return false;
    }
  }
  uint64_t done = 0;
  constexpr size_t k_chunk_size = 1024 * 1024;
  uint64_t base_offset = 0;
  for (uint16_t split_idx = 0; split_idx < split_count; ++split_idx) {
    std::FILE * file = ctx->file;
    bool owns_file = false;
    char path[k_max_path_length] = {};
    if (split_idx > 0 || file == nullptr || use_direct_io) {
      if (request->model_path.empty() || request->model_path.size() >= k_max_path_length) {
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_INVALID_ARGUMENT;
        }
        return false;
      }
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
    }
    if (!use_direct_io && (split_idx > 0 || file == nullptr)) {
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
      if (owns_file && file != nullptr) {
        std::fclose(file);
      }
      if (err_out != nullptr) {
        *err_out = EMEL_ERR_MODEL_INVALID;  // GCOVR_EXCL_LINE
      }
      return false;  // GCOVR_EXCL_LINE
    }
    uint8_t * dst =
      static_cast<uint8_t *>(request->weights_buffer) + base_offset;
    uint64_t remaining = data_size;
    uint64_t split_offset = 0;
#if defined(__linux__)
    if (use_direct_io) {
      const int fd = ::open(path, O_RDONLY | O_DIRECT);
      if (fd < 0) {
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_IO;
        }
        return false;
      }
      alignas(k_direct_io_alignment) std::array<uint8_t, k_direct_io_chunk_size> io_buffer = {};
      uint64_t file_offset = data_offset;
      while (remaining > 0) {
        const uint64_t aligned_offset =
          file_offset & ~(static_cast<uint64_t>(k_direct_io_alignment) - 1u);
        const uint64_t prefix = file_offset - aligned_offset;
        const uint64_t max_payload = k_direct_io_chunk_size - prefix;
        const uint64_t payload = std::min<uint64_t>(remaining, max_payload);
        const uint64_t aligned_size = align_up_u64(prefix + payload, k_direct_io_alignment);
        const ssize_t bytes =
          ::pread(fd, io_buffer.data(), aligned_size, static_cast<off_t>(aligned_offset));
        if (bytes != static_cast<ssize_t>(aligned_size)) {
          ::close(fd);
          if (err_out != nullptr) {
            *err_out = EMEL_ERR_IO;
          }
          return false;
        }
        std::memcpy(dst, io_buffer.data() + prefix, static_cast<size_t>(payload));
        if (wants_upload) {
          int32_t upload_err = EMEL_OK;
          if (!ev.upload_chunk(ev.upload_ctx, dst, payload, base_offset + split_offset,
                               &upload_err) ||
              upload_err != EMEL_OK) {
            ::close(fd);
            if (err_out != nullptr) {
              *err_out = upload_err == EMEL_OK ? EMEL_ERR_BACKEND : upload_err;
            }
            return false;
          }
        }
        dst += payload;
        file_offset += payload;
        remaining -= payload;
        split_offset += payload;
        done += payload;
        if (bytes_done != nullptr) {
          *bytes_done = done;
        }
        if (ev.progress_callback != nullptr && total_size > 0) {
          const float progress =
            static_cast<float>(done) / static_cast<float>(total_size);
          if (!ev.progress_callback(progress, ev.progress_user_data)) {
            ::close(fd);
            if (err_out != nullptr) {
              *err_out = EMEL_ERR_BACKEND;
            }
            return false;
          }
        }
      }
      ::close(fd);
    } else {
#endif
      if (std::fseek(file, static_cast<long>(data_offset), SEEK_SET) != 0) {
        if (owns_file) {
          std::fclose(file);
        }
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_IO;  // GCOVR_EXCL_LINE
        }
        return false;  // GCOVR_EXCL_LINE
      }
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
        if (wants_upload) {
          int32_t upload_err = EMEL_OK;
          if (!ev.upload_chunk(ev.upload_ctx, dst, chunk, base_offset + split_offset,
                               &upload_err) ||
              upload_err != EMEL_OK) {
            if (owns_file) {
              std::fclose(file);
            }
            if (err_out != nullptr) {
              *err_out = upload_err == EMEL_OK ? EMEL_ERR_BACKEND : upload_err;
            }
            return false;
          }
        }
        dst += chunk;
        remaining -= chunk;
        split_offset += chunk;
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
#if defined(__linux__)
    }
#endif
    base_offset += data_size;
  }
  if (wants_upload) {
    int32_t upload_err = EMEL_OK;
    if (!ev.upload_end(ev.upload_ctx, &upload_err) || upload_err != EMEL_OK) {
      if (err_out != nullptr) {
        *err_out = upload_err == EMEL_OK ? EMEL_ERR_BACKEND : upload_err;
      }
      return false;
    }
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

inline bool validate_weights(const emel::model::weight_loader::event::load_weights & ev,
                             int32_t * err_out) {
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
  if (!ev.check_tensors) {
    return true;
  }
  return validate_tensor_data(request->model_data, err_out);
}

inline bool clean_up_weights(const emel::model::weight_loader::event::load_weights & ev,
                             int32_t * err_out) {
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
#if defined(_WIN32)
  (void)ctx;
  return true;
#else
  if (ctx->mapped_count == 0) {
    return true;
  }
  const uint16_t split_count = request->model_data.weights_split_count;
  for (uint16_t split_idx = 0; split_idx < split_count; ++split_idx) {
    const uint64_t mapped_size = ctx->mapped_sizes[split_idx];
    if (mapped_size == 0 || ctx->mapped_splits[split_idx] == nullptr) {
      continue;
    }
    uint64_t min_offset = mapped_size;
    uint64_t max_offset = 0;
    const uint64_t base_offset = request->model_data.weights_split_offsets[split_idx];
    for (uint32_t i = 0; i < request->model_data.n_tensors; ++i) {
      const auto & record = request->model_data.tensors[i];
      if (record.data_size == 0 || record.file_index != split_idx) {
        continue;
      }
      uint64_t start = 0;
      if (add_overflow_u64(base_offset, record.file_offset, start)) {
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_MODEL_INVALID;
        }
        return false;
      }
      uint64_t end = 0;
      if (add_overflow_u64(start, record.data_size, end)) {
        if (err_out != nullptr) {
          *err_out = EMEL_ERR_MODEL_INVALID;
        }
        return false;
      }
      min_offset = std::min(min_offset, start);
      max_offset = std::max(max_offset, end);
    }
    if (min_offset == mapped_size) {
      continue;
    }
    if (min_offset > 0) {
      (void)munmap(const_cast<void *>(ctx->mapped_splits[split_idx]),
                   static_cast<size_t>(min_offset));
    }
    if (max_offset < mapped_size) {
      uint8_t * base = static_cast<uint8_t *>(
        const_cast<void *>(ctx->mapped_splits[split_idx]));
      (void)munmap(base + max_offset, static_cast<size_t>(mapped_size - max_offset));
    }
  }
  return true;
#endif
}

namespace action {

struct run_parse_architecture {
  void operator()(emel::parser::action::context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    int32_t err = EMEL_OK;
    const bool ok = emel::parser::gguf::parse_architecture(ctx.request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_PARSE_FAILED;
      }
      ctx.phase_error = err;
    }
  }
};

struct run_map_architecture {
  void operator()(emel::parser::action::context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    int32_t err = EMEL_OK;
    const bool ok = emel::parser::gguf::map_architecture(ctx.request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_MODEL_INVALID;
      }
      ctx.phase_error = err;
    }
  }
};

struct run_parse_hparams {
  void operator()(emel::parser::action::context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    int32_t err = EMEL_OK;
    const bool ok = emel::parser::gguf::parse_hparams(ctx.request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_PARSE_FAILED;
      }
      ctx.phase_error = err;
    }
  }
};

struct run_parse_vocab {
  void operator()(emel::parser::action::context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    int32_t err = EMEL_OK;
    const bool ok = emel::parser::gguf::parse_vocab(ctx.request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_PARSE_FAILED;
      }
      ctx.phase_error = err;
    }
  }
};

struct run_map_tensors {
  void operator()(emel::parser::action::context & ctx) const noexcept {
    ctx.phase_error = EMEL_OK;
    int32_t err = EMEL_OK;
    const bool ok = emel::parser::gguf::map_tensors(ctx.request, &err);
    if (!ok || err != EMEL_OK) {
      if (err == EMEL_OK) {
        err = EMEL_ERR_BACKEND;
      }
      ctx.phase_error = err;
    }
  }
};

inline constexpr run_parse_architecture run_parse_architecture{};
inline constexpr run_map_architecture run_map_architecture{};
inline constexpr run_parse_hparams run_parse_hparams{};
inline constexpr run_parse_vocab run_parse_vocab{};
inline constexpr run_map_tensors run_map_tensors{};

}  // namespace action

}  // namespace emel::parser::gguf
