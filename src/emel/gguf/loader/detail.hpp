#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>

#include "emel/error/error.hpp"
#include "emel/gguf/loader/errors.hpp"
#include "emel/gguf/loader/events.hpp"

namespace emel::gguf::loader::detail {

namespace constants {

inline constexpr std::string_view magic = "GGUF";
inline constexpr std::string_view general_alignment = "general.alignment";
inline constexpr uint32_t version = 3u;
inline constexpr uint32_t default_alignment = 32u;
inline constexpr uint32_t max_tensor_dims = 4u;

inline constexpr uint32_t gguf_type_uint8 = 0u;
inline constexpr uint32_t gguf_type_int8 = 1u;
inline constexpr uint32_t gguf_type_uint16 = 2u;
inline constexpr uint32_t gguf_type_int16 = 3u;
inline constexpr uint32_t gguf_type_uint32 = 4u;
inline constexpr uint32_t gguf_type_int32 = 5u;
inline constexpr uint32_t gguf_type_float32 = 6u;
inline constexpr uint32_t gguf_type_bool = 7u;
inline constexpr uint32_t gguf_type_string = 8u;
inline constexpr uint32_t gguf_type_array = 9u;
inline constexpr uint32_t gguf_type_uint64 = 10u;
inline constexpr uint32_t gguf_type_int64 = 11u;
inline constexpr uint32_t gguf_type_float64 = 12u;
inline constexpr uint32_t gguf_type_count = 13u;

}  // namespace constants

struct ggml_type_layout {
  uint16_t block_size = 0u;
  uint16_t type_size = 0u;
};

inline constexpr ggml_type_layout ggml_layout(const uint32_t type) noexcept {
  switch (type) {
    case 0u: return {1u, 4u};       // F32
    case 1u: return {1u, 2u};       // F16
    case 2u: return {32u, 18u};     // Q4_0
    case 3u: return {32u, 20u};     // Q4_1
    case 6u: return {32u, 22u};     // Q5_0
    case 7u: return {32u, 24u};     // Q5_1
    case 8u: return {32u, 34u};     // Q8_0
    case 9u: return {32u, 36u};     // Q8_1
    case 10u: return {256u, 84u};   // Q2_K
    case 11u: return {256u, 110u};  // Q3_K
    case 12u: return {256u, 144u};  // Q4_K
    case 13u: return {256u, 176u};  // Q5_K
    case 14u: return {256u, 210u};  // Q6_K
    case 15u: return {256u, 292u};  // Q8_K
    case 16u: return {256u, 66u};   // IQ2_XXS
    case 17u: return {256u, 74u};   // IQ2_XS
    case 18u: return {256u, 98u};   // IQ3_XXS
    case 19u: return {256u, 50u};   // IQ1_S
    case 20u: return {32u, 18u};    // IQ4_NL
    case 21u: return {256u, 110u};  // IQ3_S
    case 22u: return {256u, 82u};   // IQ2_S
    case 23u: return {256u, 136u};  // IQ4_XS
    case 24u: return {1u, 1u};      // I8
    case 25u: return {1u, 2u};      // I16
    case 26u: return {1u, 4u};      // I32
    case 27u: return {1u, 8u};      // I64
    case 28u: return {1u, 8u};      // F64
    case 29u: return {256u, 56u};   // IQ1_M
    case 30u: return {1u, 2u};      // BF16
    case 34u: return {256u, 54u};   // TQ1_0
    case 35u: return {256u, 66u};   // TQ2_0
    case 41u: return {128u, 18u};   // Q1_0_g128
    case 39u: return {32u, 17u};    // MXFP4
    default: return {};
  }
}

inline constexpr uint8_t gguf_scalar_size(const uint32_t type) noexcept {
  switch (type) {
    case constants::gguf_type_uint8:
    case constants::gguf_type_int8:
    case constants::gguf_type_bool: return 1u;
    case constants::gguf_type_uint16:
    case constants::gguf_type_int16: return 2u;
    case constants::gguf_type_uint32:
    case constants::gguf_type_int32:
    case constants::gguf_type_float32: return 4u;
    case constants::gguf_type_uint64:
    case constants::gguf_type_int64:
    case constants::gguf_type_float64: return 8u;
    default: return 0u;
  }
}

inline emel::error::type cast_loader_error(const error err) noexcept {
  return emel::error::cast(err);
}

inline bool add_u64(const uint64_t lhs, const uint64_t rhs, uint64_t & out) noexcept {
  if (std::numeric_limits<uint64_t>::max() - lhs < rhs) {
    return false;
  }

  out = lhs + rhs;
  return true;
}

inline bool multiply_u64(const uint64_t lhs, const uint64_t rhs, uint64_t & out) noexcept {
  if (lhs == 0u || rhs == 0u) {
    out = 0u;
    return true;
  }

  if (std::numeric_limits<uint64_t>::max() / lhs < rhs) {
    return false;
  }

  out = lhs * rhs;
  return true;
}

inline bool pad_to_alignment(const uint64_t value,
                             const uint32_t alignment,
                             uint64_t & out) noexcept {
  const uint64_t alignment_u64 = static_cast<uint64_t>(alignment);
  const uint64_t remainder = value % alignment_u64;

  if (remainder == 0u) {
    out = value;
    return true;
  }

  return add_u64(value, alignment_u64 - remainder, out);
}

struct bounded_reader {
  std::span<const uint8_t> bytes = {};
  uint64_t offset = 0u;

  explicit bounded_reader(std::span<const uint8_t> bytes_in) noexcept : bytes(bytes_in) {}

  bool can_read(const uint64_t count) const noexcept {
    const uint64_t size = static_cast<uint64_t>(bytes.size());
    return count <= size && offset <= size - count;
  }

  template <class value_type>
  bool read_scalar(value_type & out) noexcept {
    using unsigned_type = std::make_unsigned_t<value_type>;

    if (!can_read(sizeof(value_type))) {
      return false;
    }

    unsigned_type value = 0u;
    for (size_t i = 0; i < sizeof(value_type); ++i) {
      const unsigned_type byte =
          static_cast<unsigned_type>(bytes[static_cast<size_t>(offset) + i]);
      value |= byte << (i * 8u);
    }

    out = static_cast<value_type>(value);
    offset += sizeof(value_type);
    return true;
  }

  bool read_string(std::string_view & out) noexcept {
    uint64_t length = 0u;

    if (!read_scalar(length) || !can_read(length)) {
      return false;
    }

    out = std::string_view{
      reinterpret_cast<const char *>(bytes.data() + static_cast<size_t>(offset)),
      static_cast<size_t>(length),
    };
    offset += length;
    return true;
  }

  bool skip(const uint64_t count) noexcept {
    if (!can_read(count)) {
      return false;
    }

    offset += count;
    return true;
  }

  bool align_to(const uint32_t alignment) noexcept {
    uint64_t aligned = 0u;

    if (!pad_to_alignment(offset, alignment, aligned)) {
      return false;
    }

    if (aligned > bytes.size()) {
      return false;
    }

    offset = aligned;
    return true;
  }
};

inline bool valid_gguf_type(const uint32_t type) noexcept {
  return type < constants::gguf_type_count;
}

inline bool valid_ggml_type(const uint32_t type) noexcept {
  const ggml_type_layout layout = ggml_layout(type);
  return layout.block_size != 0u && layout.type_size != 0u;
}

inline uint64_t required_kv_arena_bytes(const requirements & requirements_in) noexcept;

inline bool read_header(bounded_reader & reader,
                        uint64_t & tensor_count_out,
                        uint64_t & kv_count_out) noexcept {
  if (!reader.can_read(constants::magic.size())) {
    return false;
  }

  const std::string_view magic{
    reinterpret_cast<const char *>(reader.bytes.data() + static_cast<size_t>(reader.offset)),
    constants::magic.size(),
  };
  reader.offset += constants::magic.size();

  uint32_t version = 0u;
  if (magic != constants::magic ||
      !reader.read_scalar(version) ||
      !reader.read_scalar(tensor_count_out) ||
      !reader.read_scalar(kv_count_out)) {
    return false;
  }

  return version == constants::version;
}

inline emel::error::type scan_value_payload(bounded_reader & reader,
                                            const uint32_t type,
                                            const std::string_view key,
                                            uint32_t & alignment_out,
                                            uint64_t & serialized_size_out) noexcept {
  const uint64_t value_start = reader.offset;

  if (!valid_gguf_type(type)) {
    return cast_loader_error(error::model_invalid);
  }

  if (type == constants::gguf_type_string) {
    std::string_view value = {};
    if (!reader.read_string(value)) {
      return cast_loader_error(error::parse_failed);
    }
    serialized_size_out = reader.offset - value_start;
    return cast_loader_error(error::none);
  }

  if (type == constants::gguf_type_array) {
    uint32_t array_type = 0u;
    uint64_t count = 0u;

    if (!reader.read_scalar(array_type) || !reader.read_scalar(count)) {
      return cast_loader_error(error::parse_failed);
    }

    if (!valid_gguf_type(array_type) || array_type == constants::gguf_type_array) {
      return cast_loader_error(error::model_invalid);
    }

    if (array_type == constants::gguf_type_string) {
      for (uint64_t i = 0u; i < count; ++i) {
        std::string_view value = {};
        if (!reader.read_string(value)) {
          return cast_loader_error(error::parse_failed);
        }
      }
    } else {
      const uint8_t element_size = gguf_scalar_size(array_type);
      if (element_size == 0u) {
        return cast_loader_error(error::model_invalid);
      }

      uint64_t payload_size = 0u;
      if (!multiply_u64(count, element_size, payload_size)) {
        return cast_loader_error(error::capacity);
      }

      if (!reader.skip(payload_size)) {
        return cast_loader_error(error::parse_failed);
      }
    }

    serialized_size_out = reader.offset - value_start;
    return cast_loader_error(error::none);
  }

  const uint8_t scalar_size = gguf_scalar_size(type);
  if (scalar_size == 0u) {
    return cast_loader_error(error::model_invalid);
  }

  if (key == constants::general_alignment) {
    if (type != constants::gguf_type_uint32) {
      return cast_loader_error(error::model_invalid);
    }

    if (!reader.read_scalar(alignment_out)) {
      return cast_loader_error(error::parse_failed);
    }

    if (alignment_out == 0u || !std::has_single_bit(alignment_out)) {
      return cast_loader_error(error::model_invalid);
    }
  } else if (!reader.skip(scalar_size)) {
    return cast_loader_error(error::parse_failed);
  }

  serialized_size_out = reader.offset - value_start;
  return cast_loader_error(error::none);
}

inline emel::error::type compute_tensor_data_size(const std::array<uint64_t, 4> & dims,
                                                  const uint32_t n_dims,
                                                  const uint32_t type,
                                                  uint64_t & data_size_out) noexcept {
  if (!valid_ggml_type(type)) {
    return cast_loader_error(error::model_invalid);
  }

  const ggml_type_layout layout = ggml_layout(type);
  const uint64_t first_dim = n_dims == 0u ? 1u : dims[0];
  if ((first_dim % layout.block_size) != 0u) {
    return cast_loader_error(error::model_invalid);
  }

  uint64_t element_count = 1u;
  for (uint32_t dim_index = 0u; dim_index < n_dims; ++dim_index) {
    if (!multiply_u64(element_count, dims[dim_index], element_count)) {
      return cast_loader_error(error::capacity);
    }
  }

  if (!multiply_u64(element_count / layout.block_size, layout.type_size, data_size_out)) {
    return cast_loader_error(error::capacity);
  }

  return cast_loader_error(error::none);
}

inline bool copy_bytes(std::span<uint8_t> dst,
                       const std::span<const uint8_t> src,
                       const uint64_t offset) noexcept {
  if (offset > dst.size() || src.size() > dst.size() - static_cast<size_t>(offset)) {
    return false;
  }

  for (size_t i = 0; i < src.size(); ++i) {
    dst[static_cast<size_t>(offset) + i] = src[i];
  }

  return true;
}

inline emel::error::type probe_requirements(const std::span<const uint8_t> & file_image,
                                            requirements & requirements_out) noexcept {
  requirements_out = {};

  bounded_reader reader{file_image};
  uint64_t tensor_count = 0u;
  uint64_t kv_count = 0u;
  uint32_t alignment = constants::default_alignment;
  uint64_t expected_tensor_offset = 0u;

  if (!read_header(reader, tensor_count, kv_count)) {
    return file_image.size() < constants::magic.size()
             ? cast_loader_error(error::parse_failed)
             : cast_loader_error(error::model_invalid);
  }

  if (tensor_count > std::numeric_limits<uint32_t>::max() ||
      kv_count > std::numeric_limits<uint32_t>::max()) {
    return cast_loader_error(error::capacity);
  }

  requirements_out.tensor_count = static_cast<uint32_t>(tensor_count);
  requirements_out.kv_count = static_cast<uint32_t>(kv_count);

  for (uint64_t i = 0u; i < kv_count; ++i) {
    std::string_view key = {};
    uint32_t type = 0u;
    uint64_t value_size = 0u;
    uint32_t maybe_alignment = alignment;

    if (!reader.read_string(key) || !reader.read_scalar(type)) {
      return cast_loader_error(error::parse_failed);
    }

    if (key.size() > std::numeric_limits<uint32_t>::max()) {
      return cast_loader_error(error::capacity);
    }

    const emel::error::type value_err =
        scan_value_payload(reader, type, key, maybe_alignment, value_size);
    if (value_err != cast_loader_error(error::none)) {
      return value_err;
    }

    if (value_size > std::numeric_limits<uint32_t>::max()) {
      return cast_loader_error(error::capacity);
    }

    if (requirements_out.max_key_bytes < key.size()) {
      requirements_out.max_key_bytes = static_cast<uint32_t>(key.size());
    }

    if (requirements_out.max_value_bytes < value_size) {
      requirements_out.max_value_bytes = static_cast<uint32_t>(value_size);
    }

    alignment = maybe_alignment;
  }

  for (uint64_t i = 0u; i < tensor_count; ++i) {
    std::string_view name = {};
    uint32_t n_dims = 0u;
    std::array<uint64_t, 4> dims = {1u, 1u, 1u, 1u};
    uint32_t type = 0u;
    uint64_t tensor_data_offset = 0u;
    uint64_t tensor_data_size = 0u;

    if (!reader.read_string(name) || !reader.read_scalar(n_dims)) {
      return cast_loader_error(error::parse_failed);
    }

    if (name.size() > std::numeric_limits<uint32_t>::max()) {
      return cast_loader_error(error::capacity);
    }

    if (n_dims > constants::max_tensor_dims) {
      return cast_loader_error(error::model_invalid);
    }

    for (uint32_t dim_index = 0u; dim_index < n_dims; ++dim_index) {
      if (!reader.read_scalar(dims[dim_index])) {
        return cast_loader_error(error::parse_failed);
      }
    }

    if (!reader.read_scalar(type) || !reader.read_scalar(tensor_data_offset)) {
      return cast_loader_error(error::parse_failed);
    }

    const emel::error::type size_err =
        compute_tensor_data_size(dims, n_dims, type, tensor_data_size);
    if (size_err != cast_loader_error(error::none)) {
      return size_err;
    }

    if (tensor_data_offset != expected_tensor_offset) {
      return cast_loader_error(error::parse_failed);
    }

    uint64_t padded_size = 0u;
    if (!pad_to_alignment(tensor_data_size, alignment, padded_size) ||
        !add_u64(expected_tensor_offset, padded_size, expected_tensor_offset)) {
      return cast_loader_error(error::capacity);
    }
  }

  if (!reader.align_to(alignment)) {
    return cast_loader_error(error::parse_failed);
  }

  uint64_t required_file_size = 0u;
  if (!add_u64(reader.offset, expected_tensor_offset, required_file_size)) {
    return cast_loader_error(error::capacity);
  }

  if (required_file_size > file_image.size()) {
    return cast_loader_error(error::parse_failed);
  }

  return cast_loader_error(error::none);
}

inline emel::error::type parse_bound_storage(
    const std::span<const uint8_t> & file_image,
    const std::span<uint8_t> kv_arena,
    const std::span<kv_entry> kv_entries,
    const std::span<emel::model::data::tensor_record> tensors,
    const requirements & requirements_in) noexcept {
  if (kv_entries.size() < requirements_in.kv_count ||
      tensors.size() < requirements_in.tensor_count ||
      kv_arena.size() < required_kv_arena_bytes(requirements_in)) {
    return cast_loader_error(error::capacity);
  }

  bounded_reader reader{file_image};
  uint64_t tensor_count = 0u;
  uint64_t kv_count = 0u;
  uint32_t alignment = constants::default_alignment;
  uint64_t arena_cursor = 0u;
  uint64_t expected_tensor_offset = 0u;

  if (!read_header(reader, tensor_count, kv_count)) {
    return file_image.size() < constants::magic.size()
             ? cast_loader_error(error::parse_failed)
             : cast_loader_error(error::model_invalid);
  }

  if (tensor_count != requirements_in.tensor_count || kv_count != requirements_in.kv_count) {
    return cast_loader_error(error::invalid_request);
  }

  for (uint64_t i = 0u; i < kv_count; ++i) {
    std::string_view key = {};
    uint32_t type = 0u;
    uint32_t maybe_alignment = alignment;
    uint64_t value_start = 0u;
    uint64_t value_size = 0u;

    if (!reader.read_string(key) || !reader.read_scalar(type)) {
      return cast_loader_error(error::parse_failed);
    }

    value_start = reader.offset;
    const emel::error::type value_err =
        scan_value_payload(reader, type, key, maybe_alignment, value_size);
    if (value_err != cast_loader_error(error::none)) {
      return value_err;
    }

    if (key.size() > requirements_in.max_key_bytes || value_size > requirements_in.max_value_bytes) {
      return cast_loader_error(error::capacity);
    }

    const uint64_t entry_offset = arena_cursor;
    if (!add_u64(arena_cursor, key.size(), arena_cursor) ||
        !add_u64(arena_cursor, value_size, arena_cursor)) {
      return cast_loader_error(error::capacity);
    }

    if (arena_cursor > kv_arena.size()) {
      return cast_loader_error(error::capacity);
    }

    const std::span<const uint8_t> key_bytes{
      reinterpret_cast<const uint8_t *>(key.data()),
      key.size(),
    };
    const std::span<const uint8_t> value_bytes{
      file_image.data() + static_cast<size_t>(value_start),
      static_cast<size_t>(value_size),
    };

    if (!copy_bytes(kv_arena, key_bytes, entry_offset) ||
        !copy_bytes(kv_arena, value_bytes, entry_offset + key.size())) {
      return cast_loader_error(error::capacity);
    }

    kv_entries[static_cast<size_t>(i)] = kv_entry{
      .key_offset = static_cast<uint32_t>(entry_offset),
      .key_length = static_cast<uint32_t>(key.size()),
      .value_offset = static_cast<uint32_t>(entry_offset + key.size()),
      .value_length = static_cast<uint32_t>(value_size),
      .value_type = type,
    };
    alignment = maybe_alignment;
  }

  for (uint64_t i = 0u; i < tensor_count; ++i) {
    std::string_view name = {};
    uint32_t n_dims = 0u;
    std::array<uint64_t, 4> dims = {1u, 1u, 1u, 1u};
    uint32_t type = 0u;
    uint64_t tensor_data_offset = 0u;
    uint64_t tensor_data_size = 0u;

    if (!reader.read_string(name) || !reader.read_scalar(n_dims)) {
      return cast_loader_error(error::parse_failed);
    }

    if (name.size() > std::numeric_limits<uint32_t>::max()) {
      return cast_loader_error(error::capacity);
    }

    if (n_dims > constants::max_tensor_dims) {
      return cast_loader_error(error::model_invalid);
    }

    for (uint32_t dim_index = 0u; dim_index < n_dims; ++dim_index) {
      if (!reader.read_scalar(dims[dim_index])) {
        return cast_loader_error(error::parse_failed);
      }
    }

    if (!reader.read_scalar(type) || !reader.read_scalar(tensor_data_offset)) {
      return cast_loader_error(error::parse_failed);
    }

    const emel::error::type size_err =
        compute_tensor_data_size(dims, n_dims, type, tensor_data_size);
    if (size_err != cast_loader_error(error::none)) {
      return size_err;
    }

    if (tensor_data_offset != expected_tensor_offset) {
      return cast_loader_error(error::parse_failed);
    }

    uint64_t padded_size = 0u;
    if (!pad_to_alignment(tensor_data_size, alignment, padded_size) ||
        !add_u64(expected_tensor_offset, padded_size, expected_tensor_offset)) {
      return cast_loader_error(error::capacity);
    }

    const uint64_t name_offset =
        static_cast<uint64_t>(reinterpret_cast<const uint8_t *>(name.data()) - file_image.data());
    if (name_offset > std::numeric_limits<uint32_t>::max()) {
      return cast_loader_error(error::capacity);
    }

    auto & tensor = tensors[static_cast<size_t>(i)];
    tensor = {};
    tensor.name_offset = static_cast<uint32_t>(name_offset);
    tensor.name_length = static_cast<uint32_t>(name.size());
    tensor.type = static_cast<int32_t>(type);
    tensor.n_dims = static_cast<int32_t>(n_dims);
    for (uint32_t dim_index = 0u; dim_index < constants::max_tensor_dims; ++dim_index) {
      tensor.dims[dim_index] = static_cast<int64_t>(dims[dim_index]);
    }
    tensor.data_offset = tensor_data_offset;
    tensor.data_size = tensor_data_size;
    tensor.file_index = 0u;
  }

  if (!reader.align_to(alignment)) {
    return cast_loader_error(error::parse_failed);
  }

  const uint64_t data_section_offset = reader.offset;
  if (data_section_offset > file_image.size()) {
    return cast_loader_error(error::parse_failed);
  }

  uint64_t required_file_size = 0u;
  if (!add_u64(data_section_offset, expected_tensor_offset, required_file_size)) {
    return cast_loader_error(error::capacity);
  }

  if (required_file_size > file_image.size()) {
    return cast_loader_error(error::parse_failed);
  }

  for (size_t i = 0; i < requirements_in.tensor_count; ++i) {
    auto & tensor = tensors[i];
    uint64_t absolute_offset = 0u;
    uint64_t tensor_end = 0u;

    if (!add_u64(data_section_offset, tensor.data_offset, absolute_offset) ||
        !add_u64(absolute_offset, tensor.data_size, tensor_end)) {
      return cast_loader_error(error::capacity);
    }

    if (tensor_end > file_image.size()) {
      return cast_loader_error(error::parse_failed);
    }

    tensor.file_offset = absolute_offset;
    tensor.data = file_image.data() + static_cast<size_t>(absolute_offset);
  }

  return cast_loader_error(error::none);
}

inline uint64_t required_kv_arena_bytes(const requirements & requirements_in) noexcept {
  uint64_t entry_bytes = 0u;
  uint64_t arena_bytes = 0u;

  if (!add_u64(static_cast<uint64_t>(requirements_in.max_key_bytes),
               static_cast<uint64_t>(requirements_in.max_value_bytes),
               entry_bytes)) {
    return std::numeric_limits<uint64_t>::max();
  }

  if (!multiply_u64(static_cast<uint64_t>(requirements_in.kv_count), entry_bytes, arena_bytes)) {
    return std::numeric_limits<uint64_t>::max();
  }

  return arena_bytes;
}

}  // namespace emel::gguf::loader::detail
