#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <span>
#include <string_view>

#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/model/data.hpp"

namespace emel::model::detail {

struct kv_binding {
  std::span<const uint8_t> arena = {};
  std::span<const emel::gguf::loader::kv_entry> entries = {};
};

inline uint32_t read_u32_le(const std::span<const uint8_t> bytes) noexcept {
  if (bytes.size() < sizeof(uint32_t)) {
    return 0u;
  }

  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8u) |
         (static_cast<uint32_t>(bytes[2]) << 16u) |
         (static_cast<uint32_t>(bytes[3]) << 24u);
}

inline uint64_t read_u64_le(const std::span<const uint8_t> bytes) noexcept {
  if (bytes.size() < sizeof(uint64_t)) {
    return 0u;
  }

  return static_cast<uint64_t>(bytes[0]) |
         (static_cast<uint64_t>(bytes[1]) << 8u) |
         (static_cast<uint64_t>(bytes[2]) << 16u) |
         (static_cast<uint64_t>(bytes[3]) << 24u) |
         (static_cast<uint64_t>(bytes[4]) << 32u) |
         (static_cast<uint64_t>(bytes[5]) << 40u) |
         (static_cast<uint64_t>(bytes[6]) << 48u) |
         (static_cast<uint64_t>(bytes[7]) << 56u);
}

template <size_t k_array_size>
inline void copy_name(std::array<char, k_array_size> & dst,
                      const std::string_view value) noexcept {
  static_assert(k_array_size > 0u);
  dst.fill('\0');
  const size_t copy_len = std::min(value.size(), k_array_size - 1u);
  if (copy_len > 0u) {
    std::memcpy(dst.data(), value.data(), copy_len);
  }
}

inline std::string_view kv_key_view(const kv_binding & binding,
                                    const emel::gguf::loader::kv_entry & entry) noexcept {
  if (static_cast<size_t>(entry.key_offset) + static_cast<size_t>(entry.key_length) >
      binding.arena.size()) {
    return {};
  }

  return std::string_view{
      reinterpret_cast<const char *>(binding.arena.data() + entry.key_offset),
      entry.key_length,
  };
}

inline std::span<const uint8_t> kv_value_view(
    const kv_binding & binding,
    const emel::gguf::loader::kv_entry & entry) noexcept {
  if (static_cast<size_t>(entry.value_offset) + static_cast<size_t>(entry.value_length) >
      binding.arena.size()) {
    return {};
  }

  return std::span<const uint8_t>{binding.arena.data() + entry.value_offset, entry.value_length};
}

inline const emel::gguf::loader::kv_entry * find_kv_entry(const kv_binding & binding,
                                                          const std::string_view key) noexcept {
  for (const auto & entry : binding.entries) {
    if (kv_key_view(binding, entry) == key) {
      return &entry;
    }
  }
  return nullptr;
}

inline const emel::gguf::loader::kv_entry * find_kv_entry_any(
    const kv_binding & binding,
    const std::initializer_list<std::string_view> keys) noexcept {
  for (const std::string_view key : keys) {
    if (const auto * entry = find_kv_entry(binding, key); entry != nullptr) {
      return entry;
    }
  }
  return nullptr;
}

inline bool decode_integer_value(const kv_binding & binding,
                                 const emel::gguf::loader::kv_entry & entry,
                                 uint64_t & value_out) noexcept {
  const std::span<const uint8_t> bytes = kv_value_view(binding, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  switch (entry.value_type) {
    case constants::gguf_type_uint8:
      if (bytes.size() != 1u) {
        return false;
      }
      value_out = bytes[0];
      return true;
    case constants::gguf_type_int8:
      if (bytes.size() != 1u) {
        return false;
      }
      value_out = static_cast<uint64_t>(static_cast<int8_t>(bytes[0]));
      return true;
    case constants::gguf_type_uint16:
    case constants::gguf_type_int16:
      if (bytes.size() != sizeof(uint16_t)) {
        return false;
      }
      value_out = static_cast<uint64_t>(bytes[0]) |
                  (static_cast<uint64_t>(bytes[1]) << 8u);
      return true;
    case constants::gguf_type_uint32:
    case constants::gguf_type_int32:
      if (bytes.size() != sizeof(uint32_t)) {
        return false;
      }
      value_out = read_u32_le(bytes);
      return true;
    case constants::gguf_type_uint64:
    case constants::gguf_type_int64:
      if (bytes.size() != sizeof(uint64_t)) {
        return false;
      }
      value_out = read_u64_le(bytes);
      return true;
    default:
      return false;
  }
}

inline bool decode_signed_integer_value(const kv_binding & binding,
                                        const emel::gguf::loader::kv_entry & entry,
                                        int64_t & value_out) noexcept {
  const std::span<const uint8_t> bytes = kv_value_view(binding, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  switch (entry.value_type) {
    case constants::gguf_type_uint8:
      if (bytes.size() != 1u) {
        return false;
      }
      value_out = bytes[0];
      return true;
    case constants::gguf_type_int8:
      if (bytes.size() != 1u) {
        return false;
      }
      value_out = static_cast<int8_t>(bytes[0]);
      return true;
    case constants::gguf_type_uint16:
      if (bytes.size() != sizeof(uint16_t)) {
        return false;
      }
      value_out = static_cast<int64_t>(static_cast<uint16_t>(bytes[0]) |
                                       (static_cast<uint16_t>(bytes[1]) << 8u));
      return true;
    case constants::gguf_type_int16:
      if (bytes.size() != sizeof(int16_t)) {
        return false;
      }
      value_out = static_cast<int16_t>(static_cast<uint16_t>(bytes[0]) |
                                       (static_cast<uint16_t>(bytes[1]) << 8u));
      return true;
    case constants::gguf_type_uint32:
      if (bytes.size() != sizeof(uint32_t)) {
        return false;
      }
      value_out = static_cast<int64_t>(read_u32_le(bytes));
      return true;
    case constants::gguf_type_int32:
      if (bytes.size() != sizeof(int32_t)) {
        return false;
      }
      value_out = static_cast<int32_t>(read_u32_le(bytes));
      return true;
    case constants::gguf_type_uint64:
      if (bytes.size() != sizeof(uint64_t)) {
        return false;
      }
      if (read_u64_le(bytes) > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
      }
      value_out = static_cast<int64_t>(read_u64_le(bytes));
      return true;
    case constants::gguf_type_int64:
      if (bytes.size() != sizeof(int64_t)) {
        return false;
      }
      value_out = static_cast<int64_t>(read_u64_le(bytes));
      return true;
    default:
      return false;
  }
}

inline bool decode_bool_value(const kv_binding & binding,
                              const emel::gguf::loader::kv_entry & entry,
                              bool & value_out) noexcept {
  const std::span<const uint8_t> bytes = kv_value_view(binding, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  if (entry.value_type != constants::gguf_type_bool || bytes.size() != 1u) {
    return false;
  }

  value_out = bytes[0] != 0u;
  return true;
}

inline bool decode_string_value(const kv_binding & binding,
                                const emel::gguf::loader::kv_entry & entry,
                                std::string_view & value_out) noexcept {
  const std::span<const uint8_t> bytes = kv_value_view(binding, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  if (entry.value_type != constants::gguf_type_string || bytes.size() < sizeof(uint64_t)) {
    return false;
  }

  const uint64_t length = read_u64_le(bytes.first(sizeof(uint64_t)));
  if (length > bytes.size() - sizeof(uint64_t)) {
    return false;
  }

  value_out = std::string_view{
      reinterpret_cast<const char *>(bytes.data() + sizeof(uint64_t)),
      static_cast<size_t>(length),
  };
  return true;
}

struct array_header {
  uint32_t element_type = 0u;
  uint64_t count = 0u;
  std::span<const uint8_t> payload = {};
};

inline bool decode_array_header(const kv_binding & binding,
                                const emel::gguf::loader::kv_entry & entry,
                                array_header & header_out) noexcept {
  const std::span<const uint8_t> bytes = kv_value_view(binding, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  if (entry.value_type != constants::gguf_type_array ||
      bytes.size() < sizeof(uint32_t) + sizeof(uint64_t)) {
    return false;
  }

  header_out.element_type = read_u32_le(bytes.first(sizeof(uint32_t)));
  header_out.count = read_u64_le(bytes.subspan(sizeof(uint32_t), sizeof(uint64_t)));
  header_out.payload = bytes.subspan(sizeof(uint32_t) + sizeof(uint64_t));
  return true;
}

inline size_t scalar_array_element_size(const uint32_t element_type) noexcept {
  namespace constants = emel::gguf::loader::detail::constants;

  switch (element_type) {
    case constants::gguf_type_uint8:
    case constants::gguf_type_int8:
    case constants::gguf_type_bool:
      return 1u;
    case constants::gguf_type_uint16:
    case constants::gguf_type_int16:
      return 2u;
    case constants::gguf_type_uint32:
    case constants::gguf_type_int32:
    case constants::gguf_type_float32:
      return 4u;
    case constants::gguf_type_uint64:
    case constants::gguf_type_int64:
    case constants::gguf_type_float64:
      return 8u;
    default:
      return 0u;
  }
}

inline bool decode_string_array_count(const kv_binding & binding,
                                      const emel::gguf::loader::kv_entry & entry,
                                      uint32_t & count_out) noexcept {
  array_header header = {};
  namespace constants = emel::gguf::loader::detail::constants;

  if (!decode_array_header(binding, entry, header) ||
      header.element_type != constants::gguf_type_string ||
      header.count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }

  count_out = static_cast<uint32_t>(header.count);
  return true;
}

inline bool decode_string_array_element(const kv_binding & binding,
                                        const emel::gguf::loader::kv_entry & entry,
                                        const uint32_t index,
                                        std::string_view & value_out) noexcept {
  array_header header = {};
  namespace constants = emel::gguf::loader::detail::constants;

  if (!decode_array_header(binding, entry, header) ||
      header.element_type != constants::gguf_type_string ||
      index >= header.count) {
    return false;
  }

  size_t cursor = 0u;
  for (uint64_t current = 0u; current <= static_cast<uint64_t>(index); ++current) {
    if (cursor + sizeof(uint64_t) > header.payload.size()) {
      return false;
    }
    const uint64_t length = read_u64_le(header.payload.subspan(cursor, sizeof(uint64_t)));
    cursor += sizeof(uint64_t);
    if (length > header.payload.size() - cursor) {
      return false;
    }
    if (current == static_cast<uint64_t>(index)) {
      value_out = std::string_view{
          reinterpret_cast<const char *>(header.payload.data() + cursor),
          static_cast<size_t>(length),
      };
      return true;
    }
    cursor += static_cast<size_t>(length);
  }

  return false;
}

template <class Fn>
inline bool visit_string_array_elements(const kv_binding & binding,
                                        const emel::gguf::loader::kv_entry & entry,
                                        Fn && fn) noexcept {
  array_header header = {};
  namespace constants = emel::gguf::loader::detail::constants;

  if (!decode_array_header(binding, entry, header) ||
      header.element_type != constants::gguf_type_string ||
      header.count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }

  size_t cursor = 0u;
  for (uint32_t index = 0u; index < static_cast<uint32_t>(header.count); ++index) {
    if (cursor + sizeof(uint64_t) > header.payload.size()) {
      return false;
    }

    const uint64_t length = read_u64_le(header.payload.subspan(cursor, sizeof(uint64_t)));
    cursor += sizeof(uint64_t);
    if (length > header.payload.size() - cursor) {
      return false;
    }

    const std::string_view value{
        reinterpret_cast<const char *>(header.payload.data() + cursor),
        static_cast<size_t>(length),
    };
    if (!fn(index, value)) {
      return false;
    }

    cursor += static_cast<size_t>(length);
  }

  return cursor == header.payload.size();
}

inline bool decode_uint_array_element(const kv_binding & binding,
                                      const emel::gguf::loader::kv_entry & entry,
                                      const uint32_t index,
                                      uint64_t & value_out) noexcept {
  array_header header = {};
  if (!decode_array_header(binding, entry, header) || index >= header.count) {
    return false;
  }

  const size_t element_size = scalar_array_element_size(header.element_type);
  if (element_size == 0u || header.payload.size() != header.count * element_size) {
    return false;
  }

  const std::span<const uint8_t> bytes =
      header.payload.subspan(static_cast<size_t>(index) * element_size, element_size);
  namespace constants = emel::gguf::loader::detail::constants;

  switch (header.element_type) {
    case constants::gguf_type_uint8:
      value_out = bytes[0];
      return true;
    case constants::gguf_type_int8:
      value_out = static_cast<uint64_t>(static_cast<int8_t>(bytes[0]));
      return true;
    case constants::gguf_type_uint16:
    case constants::gguf_type_int16:
      value_out = static_cast<uint64_t>(bytes[0]) |
                  (static_cast<uint64_t>(bytes[1]) << 8u);
      return true;
    case constants::gguf_type_uint32:
    case constants::gguf_type_int32:
      value_out = read_u32_le(bytes);
      return true;
    case constants::gguf_type_uint64:
    case constants::gguf_type_int64:
      value_out = read_u64_le(bytes);
      return true;
    default:
      return false;
  }
}

inline bool decode_first_nonzero_uint_array_value(const kv_binding & binding,
                                                  const emel::gguf::loader::kv_entry & entry,
                                                  uint64_t & value_out) noexcept {
  array_header header = {};
  if (!decode_array_header(binding, entry, header) || header.count == 0u) {
    return false;
  }

  for (uint64_t index = 0u; index < header.count; ++index) {
    uint64_t candidate = 0u;
    if (!decode_uint_array_element(binding, entry, static_cast<uint32_t>(index), candidate)) {
      return false;
    }
    if (candidate != 0u) {
      value_out = candidate;
      return true;
    }
  }

  return false;
}

inline bool decode_first_uint_array_value(const kv_binding & binding,
                                          const emel::gguf::loader::kv_entry & entry,
                                          uint64_t & value_out) noexcept {
  array_header header = {};
  if (!decode_array_header(binding, entry, header) || header.count == 0u) {
    return false;
  }

  return decode_uint_array_element(binding, entry, 0u, value_out);
}

inline bool decode_float_array_element(const kv_binding & binding,
                                       const emel::gguf::loader::kv_entry & entry,
                                       const uint32_t index,
                                       float & value_out) noexcept {
  array_header header = {};
  if (!decode_array_header(binding, entry, header) || index >= header.count) {
    return false;
  }

  const size_t element_size = scalar_array_element_size(header.element_type);
  if (element_size == 0u || header.payload.size() != header.count * element_size) {
    return false;
  }

  const std::span<const uint8_t> bytes =
      header.payload.subspan(static_cast<size_t>(index) * element_size, element_size);
  namespace constants = emel::gguf::loader::detail::constants;

  switch (header.element_type) {
    case constants::gguf_type_float32:
      if (bytes.size() != sizeof(float)) {
        return false;
      }
      std::memcpy(&value_out, bytes.data(), sizeof(float));
      return true;
    case constants::gguf_type_float64: {
      if (bytes.size() != sizeof(double)) {
        return false;
      }
      double value = 0.0;
      std::memcpy(&value, bytes.data(), sizeof(double));
      value_out = static_cast<float>(value);
      return true;
    }
    default:
      return false;
  }
}

inline bool decode_flag_array_copy(const kv_binding & binding,
                                   const emel::gguf::loader::kv_entry & entry,
                                   std::span<uint8_t> dst,
                                   uint32_t & count_out) noexcept {
  array_header header = {};
  namespace constants = emel::gguf::loader::detail::constants;

  if (!decode_array_header(binding, entry, header) ||
      header.count > static_cast<uint64_t>(dst.size())) {
    return false;
  }

  const size_t element_size = scalar_array_element_size(header.element_type);
  if (element_size == 0u || header.payload.size() != header.count * element_size) {
    return false;
  }

  for (uint64_t index = 0u; index < header.count; ++index) {
    const std::span<const uint8_t> bytes =
        header.payload.subspan(static_cast<size_t>(index) * element_size, element_size);
    uint64_t value = 0u;
    switch (header.element_type) {
      case constants::gguf_type_bool:
      case constants::gguf_type_uint8:
        value = bytes[0];
        break;
      case constants::gguf_type_int8:
        value = static_cast<uint64_t>(static_cast<int8_t>(bytes[0]));
        break;
      case constants::gguf_type_uint16:
      case constants::gguf_type_int16:
        value = static_cast<uint64_t>(bytes[0]) |
                (static_cast<uint64_t>(bytes[1]) << 8u);
        break;
      case constants::gguf_type_uint32:
      case constants::gguf_type_int32:
        value = read_u32_le(bytes);
        break;
      case constants::gguf_type_uint64:
      case constants::gguf_type_int64:
        value = read_u64_le(bytes);
        break;
      default:
        return false;
    }

    dst[static_cast<size_t>(index)] = value != 0u ? 1u : 0u;
  }

  count_out = static_cast<uint32_t>(header.count);
  return true;
}

inline bool decode_byte_array_copy(const kv_binding & binding,
                                   const emel::gguf::loader::kv_entry & entry,
                                   std::span<uint8_t> dst,
                                   uint32_t & bytes_copied_out) noexcept {
  array_header header = {};
  namespace constants = emel::gguf::loader::detail::constants;

  if (!decode_array_header(binding, entry, header) ||
      (header.element_type != constants::gguf_type_uint8 &&
       header.element_type != constants::gguf_type_int8) ||
      header.count > static_cast<uint64_t>(dst.size()) ||
      header.payload.size() != static_cast<size_t>(header.count)) {
    return false;
  }

  if (header.count > 0u) {
    std::memcpy(dst.data(), header.payload.data(), static_cast<size_t>(header.count));
  }
  bytes_copied_out = static_cast<uint32_t>(header.count);
  return true;
}

struct hparam_loader {
  const kv_binding & binding;

  bool assign_i32(const std::string_view key, int32_t & field) const noexcept {
    const auto * entry = find_kv_entry(binding, key);
    if (entry == nullptr) {
      return true;
    }

    uint64_t value = 0u;
    if (!decode_integer_value(binding, *entry, value) ||
        value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }

    field = static_cast<int32_t>(value);
    return true;
  }

  bool assign_i32_or_first_array_value(const std::string_view key, int32_t & field) const noexcept {
    const auto * entry = find_kv_entry(binding, key);
    if (entry == nullptr) {
      return true;
    }

    uint64_t value = 0u;
    const bool ok = decode_integer_value(binding, *entry, value) ||
                    decode_first_uint_array_value(binding, *entry, value);
    if (!ok || value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }

    field = static_cast<int32_t>(value);
    return true;
  }

  bool assign_first_nonzero_i32_from_array(const std::string_view key,
                                           int32_t & field) const noexcept {
    const auto * entry = find_kv_entry(binding, key);
    if (entry == nullptr) {
      return false;
    }

    uint64_t value = 0u;
    if (!decode_first_nonzero_uint_array_value(binding, *entry, value) ||
        value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }

    field = static_cast<int32_t>(value);
    return true;
  }

  bool assign_f32(const std::string_view key, float & field) const noexcept {
    const auto * entry = find_kv_entry(binding, key);
    if (entry == nullptr) {
      return true;
    }

    const std::span<const uint8_t> bytes = kv_value_view(binding, *entry);
    namespace constants = emel::gguf::loader::detail::constants;

    if (entry->value_type == constants::gguf_type_float32 && bytes.size() == sizeof(float)) {
      std::memcpy(&field, bytes.data(), sizeof(float));
      return true;
    }

    if (entry->value_type == constants::gguf_type_float64 && bytes.size() == sizeof(double)) {
      double value = 0.0;
      std::memcpy(&value, bytes.data(), sizeof(double));
      field = static_cast<float>(value);
      return true;
    }

    return false;
  }

  bool copy_flag_array(const std::string_view key,
                       std::span<uint8_t> dst,
                       uint32_t & count_out) const noexcept {
    const auto * entry = find_kv_entry(binding, key);
    return entry != nullptr && decode_flag_array_copy(binding, *entry, dst, count_out);
  }
};

}  // namespace emel::model::detail
