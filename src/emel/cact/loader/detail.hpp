#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

#include "emel/cact/loader/errors.hpp"
#include "emel/cact/loader/events.hpp"
#include "emel/error/error.hpp"

namespace emel::cact::loader::detail {

namespace constants {

inline constexpr uint32_t tag = 0x05E12A83u;
inline constexpr uint32_t alignment = 64u;
inline constexpr uint32_t header_u32_fields = 29u;
inline constexpr size_t header_bytes =
    header_u32_fields * sizeof(uint32_t) + sizeof(float);
inline constexpr size_t record_bytes = 44u;
inline constexpr uint32_t max_engram_orders = 4u;
inline constexpr uint32_t max_engram_sites = 4u;
inline constexpr uint32_t max_tensor_dims = 4u;
inline constexpr uint32_t dtype_fp16 = 1u;
inline constexpr uint32_t dtype_fp32 = 2u;
inline constexpr uint32_t dtype_cq = 3u;
inline constexpr uint32_t dtype_raw = 4u;
inline constexpr uint32_t ternary_record_bits = 5u;
inline constexpr uint32_t max_tensor_count = 1'000'000u;

} // namespace constants

inline emel::error::type cast_loader_error(const error err) noexcept {
  return emel::error::cast(err);
}

inline bool add_u64(const uint64_t lhs, const uint64_t rhs,
                    uint64_t &out) noexcept {
  if (std::numeric_limits<uint64_t>::max() - lhs < rhs) {
    return false;
  }
  out = lhs + rhs;
  return true;
}

inline bool multiply_u64(const uint64_t lhs, const uint64_t rhs,
                         uint64_t &out) noexcept {
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

// Bounded little-endian byte-stream reader. Shared, non-branching cursor
// arithmetic only; validation outcomes are surfaced as bool returns consumed
// by the calling scan routine, never as internal control-flow selection.
struct bounded_reader {
  std::span<const uint8_t> bytes = {};
  uint64_t offset = 0u;

  explicit bounded_reader(std::span<const uint8_t> bytes_in) noexcept
      : bytes(bytes_in) {}

  bool can_read(const uint64_t count) const noexcept {
    const uint64_t size = static_cast<uint64_t>(bytes.size());
    return count <= size && offset <= size - count;
  }

  bool read_u32(uint32_t &out) noexcept {
    if (!can_read(sizeof(uint32_t))) {
      return false;
    }
    uint32_t value = 0u;
    for (size_t i = 0; i < sizeof(uint32_t); ++i) {
      value |= static_cast<uint32_t>(bytes[static_cast<size_t>(offset) + i])
               << (i * 8u);
    }
    out = value;
    offset += sizeof(uint32_t);
    return true;
  }

  bool read_u16(uint16_t &out) noexcept {
    if (!can_read(sizeof(uint16_t))) {
      return false;
    }
    uint16_t value = 0u;
    for (size_t i = 0; i < sizeof(uint16_t); ++i) {
      value = static_cast<uint16_t>(
          value | (static_cast<uint16_t>(bytes[static_cast<size_t>(offset) + i])
                   << (i * 8u)));
    }
    out = value;
    offset += sizeof(uint16_t);
    return true;
  }

  bool read_u8(uint8_t &out) noexcept {
    if (!can_read(sizeof(uint8_t))) {
      return false;
    }
    out = bytes[static_cast<size_t>(offset)];
    offset += sizeof(uint8_t);
    return true;
  }

  bool read_u64(uint64_t &out) noexcept {
    if (!can_read(sizeof(uint64_t))) {
      return false;
    }
    uint64_t value = 0u;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
      value |= static_cast<uint64_t>(bytes[static_cast<size_t>(offset) + i])
               << (i * 8u);
    }
    out = value;
    offset += sizeof(uint64_t);
    return true;
  }

  bool read_f32(float &out) noexcept {
    uint32_t bits = 0u;
    if (!read_u32(bits)) {
      return false;
    }
    out = std::bit_cast<float>(bits);
    return true;
  }
};

// Decodes the fixed 120-byte `.cact` header plus codebook into `geometry_out`.
// Data-plane iteration only (fixed field count, fixed codebook length); no
// behavior selection - every rejection path is a bounded-read failure
// surfaced as a single error code for guards to dispatch on.
inline emel::error::type read_header(bounded_reader &reader,
                                     geometry &geometry_out) noexcept {
  uint32_t tag = 0u;
  uint32_t num_tensors = 0u;
  uint32_t codebook_len = 0u;

  if (!reader.read_u32(tag) || !reader.read_u32(num_tensors) ||
      !reader.read_u32(codebook_len)) {
    return cast_loader_error(error::parse_failed);
  }
  if (tag != constants::tag) {
    return cast_loader_error(error::model_invalid);
  }
  if (codebook_len != k_codebook_len) {
    return cast_loader_error(error::model_invalid);
  }
  if (num_tensors == 0u || num_tensors > constants::max_tensor_count) {
    return cast_loader_error(error::model_invalid);
  }

  geometry_out.num_tensors = num_tensors;

  if (!reader.read_u32(geometry_out.kv_window) ||
      !reader.read_u32(geometry_out.kv_bits) ||
      !reader.read_u32(geometry_out.vocab_size) ||
      !reader.read_u32(geometry_out.d_model) ||
      !reader.read_u32(geometry_out.num_heads) ||
      !reader.read_u32(geometry_out.num_kv_heads) ||
      !reader.read_u32(geometry_out.num_layers) ||
      !reader.read_u32(geometry_out.head_dim) ||
      !reader.read_u32(geometry_out.max_seq_len) ||
      !reader.read_u32(geometry_out.hada_n) ||
      !reader.read_u32(geometry_out.mhc_lanes) ||
      !reader.read_u32(geometry_out.engram_slots) ||
      !reader.read_u32(geometry_out.engram_sub_dim) ||
      !reader.read_u32(geometry_out.num_engram_tables) ||
      !reader.read_u32(geometry_out.engram_conv_taps) ||
      !reader.read_u32(geometry_out.engram_conv_dilation) ||
      !reader.read_u32(geometry_out.num_engram_orders)) {
    return cast_loader_error(error::parse_failed);
  }

  if (geometry_out.num_engram_orders > constants::max_engram_orders) {
    return cast_loader_error(error::model_invalid);
  }

  for (uint32_t i = 0u; i < constants::max_engram_orders; ++i) {
    if (!reader.read_u32(geometry_out.engram_orders[i])) {
      return cast_loader_error(error::parse_failed);
    }
  }

  if (!reader.read_u32(geometry_out.num_engram_sites)) {
    return cast_loader_error(error::parse_failed);
  }
  if (geometry_out.num_engram_sites > constants::max_engram_sites) {
    return cast_loader_error(error::model_invalid);
  }

  for (uint32_t i = 0u; i < constants::max_engram_sites; ++i) {
    if (!reader.read_u32(geometry_out.engram_sites[i])) {
      return cast_loader_error(error::parse_failed);
    }
  }

  if (!reader.read_f32(geometry_out.rope_theta)) {
    return cast_loader_error(error::parse_failed);
  }

  if (reader.offset != constants::header_bytes) {
    return cast_loader_error(error::internal_error);
  }

  for (uint32_t i = 0u; i < k_codebook_len; ++i) {
    if (!reader.read_f32(geometry_out.codebook[i])) {
      return cast_loader_error(error::parse_failed);
    }
  }

  return cast_loader_error(error::none);
}

// Computes the expected CQ blob byte size (packed indices + per-group fp16
// L2 norms) for a [out, in] logical matrix padded to a multiple of `group`.
// Pure numeric computation on the already-chosen CQ path; no path selection.
inline bool compute_cq_expected_bytes(const uint32_t out_rows,
                                      const uint32_t in_dim,
                                      const uint32_t group, const uint32_t bits,
                                      uint64_t &expected_bytes_out) noexcept {
  if (group == 0u || out_rows == 0u) {
    return false;
  }

  const uint64_t in_pad =
      ((static_cast<uint64_t>(in_dim) + group - 1u) / group) * group;
  const uint64_t packed_bits =
      bits == constants::ternary_record_bits ? 2u : bits;
  if (in_pad % 8u != 0u) {
    return false;
  }

  const uint64_t packed_row_bytes = (in_pad * packed_bits) / 8u;
  uint64_t packed_total = 0u;
  if (!multiply_u64(static_cast<uint64_t>(out_rows), packed_row_bytes,
                    packed_total)) {
    return false;
  }

  const uint64_t groups_per_row = in_pad / group;
  uint64_t norms_per_row_bytes = 0u;
  if (!multiply_u64(groups_per_row, 2u, norms_per_row_bytes)) {
    return false;
  }
  uint64_t norms_total = 0u;
  if (!multiply_u64(static_cast<uint64_t>(out_rows), norms_per_row_bytes,
                    norms_total)) {
    return false;
  }

  return add_u64(packed_total, norms_total, expected_bytes_out);
}

// Locates and bounds-checks the directory span within `file_image` for the
// given tensor count. Shared by probe (bounds-only) and parse (bounds +
// populate) so the offset/size arithmetic is not duplicated.
inline emel::error::type
locate_directory(const std::span<const uint8_t> &file_image,
                 const uint32_t num_tensors,
                 uint64_t &directory_offset_out) noexcept {
  const uint64_t codebook_bytes =
      static_cast<uint64_t>(k_codebook_len) * sizeof(float);
  if (!add_u64(constants::header_bytes, codebook_bytes, directory_offset_out)) {
    return cast_loader_error(error::capacity);
  }

  uint64_t directory_bytes = 0u;
  if (!multiply_u64(static_cast<uint64_t>(num_tensors), constants::record_bytes,
                    directory_bytes)) {
    return cast_loader_error(error::capacity);
  }

  uint64_t directory_end = 0u;
  if (!add_u64(directory_offset_out, directory_bytes, directory_end) ||
      directory_end > file_image.size()) {
    return cast_loader_error(error::parse_failed);
  }

  return cast_loader_error(error::none);
}

// Reads and validates one 44-byte directory record from `reader` against
// `file_image` bounds/alignment/CQ-size invariants, writing the decoded
// fields into `view_out`. Shared by probe_geometry (discards the view) and
// parse_directory (keeps it), so the per-record validation lives in exactly
// one place.
inline emel::error::type
scan_directory_record(bounded_reader &reader,
                      const std::span<const uint8_t> &file_image,
                      tensor_view &view_out) noexcept {
  uint8_t dtype = 0u;
  uint8_t ndim = 0u;
  uint16_t pad = 0u;
  std::array<uint32_t, 4> shape = {0u, 0u, 0u, 0u};
  uint64_t offset = 0u;
  uint64_t nbytes = 0u;
  uint32_t group = 0u;
  uint32_t bits = 0u;

  if (!reader.read_u8(dtype) || !reader.read_u8(ndim) ||
      !reader.read_u16(pad) || !reader.read_u32(shape[0]) ||
      !reader.read_u32(shape[1]) || !reader.read_u32(shape[2]) ||
      !reader.read_u32(shape[3]) || !reader.read_u64(offset) ||
      !reader.read_u64(nbytes) || !reader.read_u32(group) ||
      !reader.read_u32(bits)) {
    return cast_loader_error(error::parse_failed);
  }

  if (dtype != constants::dtype_fp16 && dtype != constants::dtype_fp32 &&
      dtype != constants::dtype_cq && dtype != constants::dtype_raw) {
    return cast_loader_error(error::model_invalid);
  }
  if (ndim > constants::max_tensor_dims) {
    return cast_loader_error(error::model_invalid);
  }
  if (offset % constants::alignment != 0u) {
    return cast_loader_error(error::model_invalid);
  }

  uint64_t tensor_end = 0u;
  if (!add_u64(offset, nbytes, tensor_end) || tensor_end > file_image.size()) {
    return cast_loader_error(error::parse_failed);
  }

  if (dtype == constants::dtype_cq) {
    if (ndim != 2u) {
      return cast_loader_error(error::model_invalid);
    }
    uint64_t expected_bytes = 0u;
    if (!compute_cq_expected_bytes(shape[0], shape[1], group, bits,
                                   expected_bytes) ||
        expected_bytes != nbytes) {
      return cast_loader_error(error::model_invalid);
    }
  }

  view_out.dtype = dtype;
  view_out.ndim = ndim;
  view_out.shape = shape;
  view_out.offset = offset;
  view_out.nbytes = nbytes;
  view_out.group = group;
  view_out.bits = bits;
  view_out.data = file_image.data() + static_cast<size_t>(offset);
  return cast_loader_error(error::none);
}

// Parses the nameless tensor directory into `tensors_out`, validating each
// record's shape/offset/nbytes/group/bits and 64-byte blob alignment against
// `file_image`. This is bulk data-plane iteration over a bounded record
// count (geometry.num_tensors, already capped in read_header); each record's
// outcome is folded into a single accumulated error code, so control flow
// never leaves this single transition's action.
inline emel::error::type
parse_directory(const std::span<const uint8_t> &file_image,
                const geometry &geometry_in,
                const std::span<tensor_view> tensors_out) noexcept {
  if (tensors_out.size() < geometry_in.num_tensors) {
    return cast_loader_error(error::capacity);
  }

  uint64_t directory_offset = 0u;
  const emel::error::type locate_err =
      locate_directory(file_image, geometry_in.num_tensors, directory_offset);
  if (locate_err != cast_loader_error(error::none)) {
    return locate_err;
  }

  bounded_reader reader{file_image};
  reader.offset = directory_offset;

  for (uint32_t i = 0u; i < geometry_in.num_tensors; ++i) {
    const emel::error::type record_err =
        scan_directory_record(reader, file_image, tensors_out[i]);
    if (record_err != cast_loader_error(error::none)) {
      return record_err;
    }
  }

  return cast_loader_error(error::none);
}

// Full probe pass: header/codebook decode plus a directory scan that only
// validates bounds/alignment (does not populate tensor views - that is
// parse's job so probe stays a read-only capability check).
inline emel::error::type
probe_geometry(const std::span<const uint8_t> &file_image,
               geometry &geometry_out) noexcept {
  geometry_out = {};

  bounded_reader reader{file_image};
  if (!reader.can_read(constants::header_bytes)) {
    return cast_loader_error(error::parse_failed);
  }

  const emel::error::type header_err = read_header(reader, geometry_out);
  if (header_err != cast_loader_error(error::none)) {
    return header_err;
  }

  uint64_t directory_offset = 0u;
  const emel::error::type locate_err =
      locate_directory(file_image, geometry_out.num_tensors, directory_offset);
  if (locate_err != cast_loader_error(error::none)) {
    return locate_err;
  }

  bounded_reader dir_reader{file_image};
  dir_reader.offset = directory_offset;

  for (uint32_t i = 0u; i < geometry_out.num_tensors; ++i) {
    tensor_view discarded = {};
    const emel::error::type record_err =
        scan_directory_record(dir_reader, file_image, discarded);
    if (record_err != cast_loader_error(error::none)) {
      return record_err;
    }
  }

  return cast_loader_error(error::none);
}

} // namespace emel::cact::loader::detail
