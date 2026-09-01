#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "doctest/doctest.h"

#include "emel/cact/loader/any.hpp"
#include "emel/cact/loader/detail.hpp"
#include "emel/cact/loader/guards.hpp"
#include "emel/cact/loader/sm.hpp"

namespace {

constexpr uint32_t k_tag = 0x05E12A83u;
constexpr uint32_t k_codebook_len = emel::cact::loader::k_codebook_len;
constexpr size_t k_header_bytes = emel::cact::loader::constants::header_bytes;
constexpr uint32_t k_alignment = emel::cact::loader::constants::alignment;

// Pinned fixture geometry from /shared/effortless/train/route-w4-qat.cact,
// verified against the Python exporter (`needle/model/export.py`, `<29If`).
constexpr uint32_t k_fixture_num_tensors = 405u;
constexpr uint32_t k_fixture_kv_window = 704u;
constexpr uint32_t k_fixture_kv_bits = 8u;
constexpr uint32_t k_fixture_vocab = 8192u;
constexpr uint32_t k_fixture_d_model = 512u;
constexpr uint32_t k_fixture_num_heads = 8u;
constexpr uint32_t k_fixture_num_kv_heads = 4u;
constexpr uint32_t k_fixture_num_layers = 27u;
constexpr uint32_t k_fixture_head_dim = 64u;
constexpr uint32_t k_fixture_max_seq_len = 2048u;
constexpr uint32_t k_fixture_hada_n = 512u;
constexpr uint32_t k_fixture_mhc_lanes = 4u;
constexpr float k_fixture_rope_theta = 100000.0f;

template <class value_type>
void append_scalar(std::vector<uint8_t> &bytes, const value_type value) {
  for (size_t i = 0; i < sizeof(value_type); ++i) {
    bytes.push_back(static_cast<uint8_t>(
        (static_cast<uint64_t>(value) >> (i * 8u)) & 0xffu));
  }
}

template <class value_type>
void write_scalar(std::vector<uint8_t> &bytes, const size_t offset,
                  const value_type value) {
  REQUIRE(offset + sizeof(value_type) <= bytes.size());
  for (size_t i = 0; i < sizeof(value_type); ++i) {
    bytes[offset + i] = static_cast<uint8_t>(
        (static_cast<uint64_t>(value) >> (i * 8u)) & 0xffu);
  }
}

void append_f32(std::vector<uint8_t> &bytes, const float value) {
  uint32_t raw = 0u;
  static_assert(sizeof(raw) == sizeof(value));
  __builtin_memcpy(&raw, &value, sizeof(raw));
  append_scalar<uint32_t>(bytes, raw);
}

// Builds a minimal valid single-tensor `.cact` image: header, 28-float
// codebook, one FP16 vector tensor record, 64-byte aligned blob.
std::vector<uint8_t> make_valid_cact_file() {
  std::vector<uint8_t> bytes;

  const uint32_t num_tensors = 1u;
  const uint32_t shape0 = 16u;
  const uint64_t nbytes = static_cast<uint64_t>(shape0) * 2u;

  append_scalar<uint32_t>(bytes, k_tag);
  append_scalar<uint32_t>(bytes, num_tensors);
  append_scalar<uint32_t>(bytes, k_codebook_len);
  append_scalar<uint32_t>(bytes, 704u);  // kv_window
  append_scalar<uint32_t>(bytes, 8u);    // kv_bits
  append_scalar<uint32_t>(bytes, 8192u); // vocab
  append_scalar<uint32_t>(bytes, 512u);  // d_model
  append_scalar<uint32_t>(bytes, 8u);    // num_heads
  append_scalar<uint32_t>(bytes, 4u);    // num_kv_heads
  append_scalar<uint32_t>(bytes, 27u);   // num_layers
  append_scalar<uint32_t>(bytes, 64u);   // head_dim
  append_scalar<uint32_t>(bytes, 2048u); // max_seq_len
  append_scalar<uint32_t>(bytes, 512u);  // hada_n
  append_scalar<uint32_t>(bytes, 4u);    // mhc_lanes
  append_scalar<uint32_t>(bytes, 8192u); // engram_slots
  append_scalar<uint32_t>(bytes, 128u);  // engram_sub_dim
  append_scalar<uint32_t>(bytes, 4u);    // num_engram_tables
  append_scalar<uint32_t>(bytes, 4u);    // engram_conv_taps
  append_scalar<uint32_t>(bytes, 3u);    // engram_conv_dilation
  append_scalar<uint32_t>(bytes, 2u);    // num_engram_orders
  append_scalar<uint32_t>(bytes, 2u);    // engram_orders[0]
  append_scalar<uint32_t>(bytes, 3u);    // engram_orders[1]
  append_scalar<uint32_t>(bytes, 0u);    // engram_orders[2]
  append_scalar<uint32_t>(bytes, 0u);    // engram_orders[3]
  append_scalar<uint32_t>(bytes, 2u);    // num_engram_sites
  append_scalar<uint32_t>(bytes, 2u);    // engram_sites[0]
  append_scalar<uint32_t>(bytes, 15u);   // engram_sites[1]
  append_scalar<uint32_t>(bytes, 0u);    // engram_sites[2]
  append_scalar<uint32_t>(bytes, 0u);    // engram_sites[3]
  append_f32(bytes, 100000.0f);          // rope_theta
  CHECK(bytes.size() == k_header_bytes);

  for (uint32_t i = 0u; i < k_codebook_len; ++i) {
    append_f32(bytes, static_cast<float>(i) * 0.125f - 1.0f);
  }

  const uint64_t directory_offset = bytes.size();
  const uint64_t blob_offset =
      ((directory_offset + emel::cact::loader::constants::record_bytes +
        k_alignment - 1u) /
       k_alignment) *
      k_alignment;

  append_scalar<uint8_t>(bytes, static_cast<uint8_t>(1u)); // dtype FP16
  append_scalar<uint8_t>(bytes, static_cast<uint8_t>(1u)); // ndim
  append_scalar<uint16_t>(bytes, static_cast<uint16_t>(0u));
  append_scalar<uint32_t>(bytes, shape0);
  append_scalar<uint32_t>(bytes, 0u);
  append_scalar<uint32_t>(bytes, 0u);
  append_scalar<uint32_t>(bytes, 0u);
  append_scalar<uint64_t>(bytes, blob_offset);
  append_scalar<uint64_t>(bytes, nbytes);
  append_scalar<uint32_t>(bytes, 0u); // group
  append_scalar<uint32_t>(bytes, 0u); // bits

  bytes.resize(static_cast<size_t>(blob_offset), 0u);
  bytes.resize(static_cast<size_t>(blob_offset + nbytes), 0x5au);
  return bytes;
}

std::vector<uint8_t> make_bad_tag_cact_file() {
  std::vector<uint8_t> bytes = make_valid_cact_file();
  bytes[0] ^= 0xffu;
  return bytes;
}

std::vector<uint8_t> make_truncated_directory_cact_file() {
  std::vector<uint8_t> bytes = make_valid_cact_file();
  bytes.resize(k_header_bytes + k_codebook_len * sizeof(float) + 8u);
  return bytes;
}

std::vector<uint8_t> make_bad_offset_cact_file() {
  std::vector<uint8_t> bytes = make_valid_cact_file();
  // Record's u64 offset field starts 20 bytes into the record.
  const size_t record_offset = k_header_bytes + k_codebook_len * sizeof(float);
  const size_t offset_field = record_offset + 20u;
  // Point the blob past the end of the file (keeps 64-byte alignment).
  const uint64_t bogus = (static_cast<uint64_t>(bytes.size()) + k_alignment) &
                         ~static_cast<uint64_t>(k_alignment - 1u);
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    bytes[offset_field + i] = static_cast<uint8_t>((bogus >> (i * 8u)) & 0xffu);
  }
  return bytes;
}

std::vector<uint8_t> make_unaligned_offset_cact_file() {
  std::vector<uint8_t> bytes = make_valid_cact_file();
  const size_t record_offset = k_header_bytes + k_codebook_len * sizeof(float);
  const size_t offset_field = record_offset + 20u;
  bytes[offset_field] |= 0x01u; // break the 64-byte alignment invariant
  return bytes;
}

constexpr size_t record_offset(const size_t index = 0u) {
  return k_header_bytes + k_codebook_len * sizeof(float) +
         index * emel::cact::loader::constants::record_bytes;
}

std::vector<uint8_t> make_single_tensor_cact_file(
    const uint8_t dtype, const uint8_t ndim,
    const std::array<uint32_t, 4> &shape, const uint64_t nbytes,
    const uint32_t group = 0u, const uint32_t bits = 0u) {
  std::vector<uint8_t> bytes = make_valid_cact_file();
  const size_t record = record_offset();
  write_scalar<uint8_t>(bytes, record, dtype);
  write_scalar<uint8_t>(bytes, record + 1u, ndim);
  for (size_t i = 0u; i < shape.size(); ++i) {
    write_scalar<uint32_t>(bytes, record + 4u + i * sizeof(uint32_t), shape[i]);
  }
  write_scalar<uint64_t>(bytes, record + 28u, nbytes);
  write_scalar<uint32_t>(bytes, record + 36u, group);
  write_scalar<uint32_t>(bytes, record + 40u, bits);
  const uint64_t offset = 320u;
  write_scalar<uint64_t>(bytes, record + 20u, offset);
  bytes.resize(static_cast<size_t>(offset + nbytes), 0x5au);
  return bytes;
}

std::vector<uint8_t> make_metadata_overlap_cact_file() {
  std::vector<uint8_t> bytes = make_valid_cact_file();
  write_scalar<uint64_t>(bytes, record_offset() + 20u, 256u);
  return bytes;
}

std::vector<uint8_t> make_overlapping_tensor_ranges_cact_file() {
  std::vector<uint8_t> bytes = make_valid_cact_file();
  write_scalar<uint32_t>(bytes, sizeof(uint32_t), 2u);

  const size_t first_record = record_offset();
  const size_t second_record = record_offset(1u);
  const uint64_t offset = 384u;
  const uint64_t nbytes = 32u;

  bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(second_record),
               emel::cact::loader::constants::record_bytes, 0u);
  write_scalar<uint64_t>(bytes, first_record + 20u, offset);

  write_scalar<uint8_t>(bytes, second_record, static_cast<uint8_t>(1u));
  write_scalar<uint8_t>(bytes, second_record + 1u, static_cast<uint8_t>(1u));
  write_scalar<uint32_t>(bytes, second_record + 4u, 16u);
  write_scalar<uint64_t>(bytes, second_record + 20u, offset);
  write_scalar<uint64_t>(bytes, second_record + 28u, nbytes);
  bytes.resize(static_cast<size_t>(offset + nbytes), 0x5au);
  return bytes;
}

std::vector<uint8_t> make_valid_raw_payload() {
  std::vector<uint8_t> payload;
  append_scalar<uint32_t>(payload, 1u); // token count
  append_scalar<uint32_t>(payload, 0u); // pad id
  append_scalar<uint32_t>(payload, 0u); // eos id
  append_scalar<uint32_t>(payload, 0u); // bos id
  append_scalar<uint32_t>(payload, 0u); // unk id
  append_scalar<uint8_t>(payload, 1u);  // add dummy prefix
  append_scalar<uint8_t>(payload, 1u);  // byte fallback
  append_scalar<uint16_t>(payload, 0u); // reserved
  append_f32(payload, 0.0f);            // score
  append_scalar<uint8_t>(payload, 0u);  // normal token
  append_scalar<uint16_t>(payload, 1u); // token byte length
  append_scalar<uint8_t>(payload, static_cast<uint8_t>('x'));
  return payload;
}

std::vector<uint8_t> make_raw_cact_file(const size_t payload_size) {
  const std::vector<uint8_t> valid_payload = make_valid_raw_payload();
  std::vector<uint8_t> bytes = make_single_tensor_cact_file(
      4u, 0u, {0u, 0u, 0u, 0u}, payload_size);
  const uint64_t offset = 320u;
  const size_t copied = std::min(payload_size, valid_payload.size());
  for (size_t i = 0u; i < copied; ++i) {
    bytes[static_cast<size_t>(offset) + i] = valid_payload[i];
  }
  return bytes;
}

struct callback_state {
  uint32_t probe_done_count = 0u;
  uint32_t probe_error_count = 0u;
  uint32_t bind_done_count = 0u;
  uint32_t bind_error_count = 0u;
  uint32_t parse_done_count = 0u;
  uint32_t parse_error_count = 0u;
  emel::cact::loader::geometry probe_geometry = {};
  emel::error::type probe_error =
      emel::error::cast(emel::cact::loader::error::none);
  emel::error::type bind_error =
      emel::error::cast(emel::cact::loader::error::none);
  emel::error::type parse_error =
      emel::error::cast(emel::cact::loader::error::none);
};

callback_state *g_callback_state = nullptr;

struct callback_scope {
  explicit callback_scope(callback_state &state) noexcept {
    g_callback_state = &state;
  }

  ~callback_scope() { g_callback_state = nullptr; }
};

void on_probe_done(const emel::cact::loader::events::probe_done &ev) {
  if (g_callback_state == nullptr) {
    return;
  }
  ++g_callback_state->probe_done_count;
  g_callback_state->probe_geometry = ev.geometry_out;
}

void on_probe_error(const emel::cact::loader::events::probe_error &ev) {
  if (g_callback_state == nullptr) {
    return;
  }
  ++g_callback_state->probe_error_count;
  g_callback_state->probe_error = ev.err;
}

void on_bind_done(const emel::cact::loader::events::bind_done &) {
  if (g_callback_state != nullptr) {
    ++g_callback_state->bind_done_count;
  }
}

void on_bind_error(const emel::cact::loader::events::bind_error &ev) {
  if (g_callback_state == nullptr) {
    return;
  }
  ++g_callback_state->bind_error_count;
  g_callback_state->bind_error = ev.err;
}

void on_parse_done(const emel::cact::loader::events::parse_done &) {
  if (g_callback_state != nullptr) {
    ++g_callback_state->parse_done_count;
  }
}

void on_parse_error(const emel::cact::loader::events::parse_error &ev) {
  if (g_callback_state == nullptr) {
    return;
  }
  ++g_callback_state->parse_error_count;
  g_callback_state->parse_error = ev.err;
}

const emel::cact::loader::event::probe_done_fn k_probe_done_cb =
    emel::cact::loader::event::probe_done_fn::from<&on_probe_done>();
const emel::cact::loader::event::probe_error_fn k_probe_error_cb =
    emel::cact::loader::event::probe_error_fn::from<&on_probe_error>();
const emel::cact::loader::event::bind_done_fn k_bind_done_cb =
    emel::cact::loader::event::bind_done_fn::from<&on_bind_done>();
const emel::cact::loader::event::bind_error_fn k_bind_error_cb =
    emel::cact::loader::event::bind_error_fn::from<&on_bind_error>();
const emel::cact::loader::event::parse_done_fn k_parse_done_cb =
    emel::cact::loader::event::parse_done_fn::from<&on_parse_done>();
const emel::cact::loader::event::parse_error_fn k_parse_error_cb =
    emel::cact::loader::event::parse_error_fn::from<&on_parse_error>();

std::vector<uint8_t> read_file_bytes(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  CHECK(input.good());

  input.seekg(0, std::ios::end);
  const std::streamsize size = input.tellg();
  CHECK(size > 0);
  input.seekg(0, std::ios::beg);

  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  input.read(reinterpret_cast<char *>(bytes.data()), size);
  CHECK(input.good());
  return bytes;
}

std::filesystem::path repo_relative(const char *relative) {
  return std::filesystem::path{EMEL_TEST_REPO_ROOT} / relative;
}

struct fixture_record {
  uint32_t index = 0u;
  uint32_t dtype = 0u;
  uint32_t ndim = 0u;
  std::array<uint32_t, 4> shape = {0u, 0u, 0u, 0u};
  uint64_t offset = 0u;
  uint64_t nbytes = 0u;
  uint32_t group = 0u;
  uint32_t bits = 0u;
};

// Parses the committed parity CSV produced by scripts/gen_cact_directory_csv.py
// from the pinned fixture (raw struct dump, same source of truth as the Python
// exporter's `read_export`).
std::vector<fixture_record>
read_directory_fixture(const std::filesystem::path &path) {
  std::ifstream input(path);
  CHECK(input.good());

  std::vector<fixture_record> records;
  std::string line;
  CHECK(static_cast<bool>(std::getline(input, line))); // header row

  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }
    fixture_record record = {};
    const int parsed = std::sscanf(
        line.c_str(), "%u,%u,%u,%u,%u,%u,%u,%llu,%llu,%u,%u", &record.index,
        &record.dtype, &record.ndim, &record.shape[0], &record.shape[1],
        &record.shape[2], &record.shape[3],
        reinterpret_cast<unsigned long long *>(&record.offset),
        reinterpret_cast<unsigned long long *>(&record.nbytes), &record.group,
        &record.bits);
    CHECK(parsed == 11);
    records.push_back(record);
  }
  return records;
}

} // namespace

TEST_CASE("cact loader probe bind parse lifecycle on a minimal image") {
  emel::cact::loader::sm machine{};
  callback_state state = {};
  callback_scope scope{state};

  const std::vector<uint8_t> file_bytes = make_valid_cact_file();
  emel::cact::loader::geometry geometry = {};
  const emel::cact::loader::event::probe probe{
      std::span<const uint8_t>{file_bytes},
      geometry,
      k_probe_done_cb,
      k_probe_error_cb,
  };

  CHECK(machine.process_event(probe));
  CHECK(state.probe_done_count == 1u);
  CHECK(state.probe_error_count == 0u);
  CHECK(machine.is(stateforward::sml::state<emel::cact::loader::state_probed>));
  CHECK(geometry.num_tensors == 1u);
  CHECK(geometry.kv_window == 704u);
  CHECK(geometry.rope_theta == doctest::Approx(100000.0f));

  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  const emel::cact::loader::event::bind_storage bind{
      std::span<emel::cact::loader::tensor_view>{tensors},
      k_bind_done_cb,
      k_bind_error_cb,
  };

  CHECK(machine.process_event(bind));
  CHECK(state.bind_done_count == 1u);
  CHECK(state.bind_error_count == 0u);
  CHECK(machine.is(stateforward::sml::state<emel::cact::loader::state_bound>));

  const emel::cact::loader::event::parse parse{
      std::span<const uint8_t>{file_bytes},
      k_parse_done_cb,
      k_parse_error_cb,
  };

  CHECK(machine.process_event(parse));
  CHECK(state.parse_done_count == 1u);
  CHECK(state.parse_error_count == 0u);
  CHECK(machine.is(stateforward::sml::state<emel::cact::loader::state_parsed>));

  CHECK(tensors[0].dtype == emel::cact::loader::constants::dtype_fp16);
  CHECK(tensors[0].ndim == 1u);
  CHECK(tensors[0].shape[0] == 16u);
  CHECK(tensors[0].nbytes == 32u);
  CHECK(tensors[0].offset % emel::cact::loader::constants::alignment == 0u);
  CHECK(tensors[0].data == file_bytes.data() + tensors[0].offset);
  CHECK(tensors[0].data[0] == 0x5au);
}

TEST_CASE("cact loader probe rejects invalid request inputs") {
  emel::cact::loader::sm machine{};
  callback_state state = {};
  callback_scope scope{state};

  emel::cact::loader::geometry geometry = {};
  const emel::cact::loader::event::probe probe{
      std::span<const uint8_t>{},
      geometry,
      k_probe_done_cb,
      k_probe_error_cb,
  };

  CHECK_FALSE(machine.process_event(probe));
  CHECK(state.probe_done_count == 0u);
  CHECK(state.probe_error_count == 1u);
  CHECK(state.probe_error ==
        emel::error::cast(emel::cact::loader::error::invalid_request));
  CHECK(
      machine.is(stateforward::sml::state<emel::cact::loader::state_errored>));
}

TEST_CASE("cact loader probe classifies malformed images") {
  SUBCASE("bad tag maps to model_invalid") {
    emel::cact::loader::sm machine{};
    callback_state state = {};
    callback_scope scope{state};
    const std::vector<uint8_t> file_bytes = make_bad_tag_cact_file();
    emel::cact::loader::geometry geometry = {};
    const emel::cact::loader::event::probe probe{
        std::span<const uint8_t>{file_bytes},
        geometry,
        k_probe_done_cb,
        k_probe_error_cb,
    };

    CHECK_FALSE(machine.process_event(probe));
    CHECK(state.probe_error ==
          emel::error::cast(emel::cact::loader::error::model_invalid));
    CHECK(machine.is(
        stateforward::sml::state<emel::cact::loader::state_errored>));
  }

  SUBCASE("truncated directory maps to parse_failed") {
    emel::cact::loader::sm machine{};
    callback_state state = {};
    callback_scope scope{state};
    const std::vector<uint8_t> file_bytes =
        make_truncated_directory_cact_file();
    emel::cact::loader::geometry geometry = {};
    const emel::cact::loader::event::probe probe{
        std::span<const uint8_t>{file_bytes},
        geometry,
        k_probe_done_cb,
        k_probe_error_cb,
    };

    CHECK_FALSE(machine.process_event(probe));
    CHECK(state.probe_error ==
          emel::error::cast(emel::cact::loader::error::parse_failed));
    CHECK(machine.is(
        stateforward::sml::state<emel::cact::loader::state_errored>));
  }

  SUBCASE("out-of-bounds blob offset maps to parse_failed") {
    emel::cact::loader::sm machine{};
    callback_state state = {};
    callback_scope scope{state};
    const std::vector<uint8_t> file_bytes = make_bad_offset_cact_file();
    emel::cact::loader::geometry geometry = {};
    const emel::cact::loader::event::probe probe{
        std::span<const uint8_t>{file_bytes},
        geometry,
        k_probe_done_cb,
        k_probe_error_cb,
    };

    CHECK_FALSE(machine.process_event(probe));
    CHECK(state.probe_error ==
          emel::error::cast(emel::cact::loader::error::parse_failed));
    CHECK(machine.is(
        stateforward::sml::state<emel::cact::loader::state_errored>));
  }

  SUBCASE("unaligned blob offset maps to model_invalid") {
    emel::cact::loader::sm machine{};
    callback_state state = {};
    callback_scope scope{state};
    const std::vector<uint8_t> file_bytes = make_unaligned_offset_cact_file();
    emel::cact::loader::geometry geometry = {};
    const emel::cact::loader::event::probe probe{
        std::span<const uint8_t>{file_bytes},
        geometry,
        k_probe_done_cb,
        k_probe_error_cb,
    };

    CHECK_FALSE(machine.process_event(probe));
    CHECK(state.probe_error ==
          emel::error::cast(emel::cact::loader::error::model_invalid));
    CHECK(machine.is(
        stateforward::sml::state<emel::cact::loader::state_errored>));
  }
}

TEST_CASE("cact loader rejects tensor payloads inside metadata before bind") {
  emel::cact::loader::sm machine{};
  callback_state state = {};
  callback_scope scope{state};

  const std::vector<uint8_t> file_bytes = make_metadata_overlap_cact_file();
  emel::cact::loader::geometry geometry = {};
  const emel::cact::loader::event::probe probe{
      std::span<const uint8_t>{file_bytes},
      geometry,
      k_probe_done_cb,
      k_probe_error_cb,
  };

  CHECK_FALSE(machine.process_event(probe));
  CHECK(state.probe_done_count == 0u);
  CHECK(state.probe_error_count == 1u);
  CHECK(state.probe_error ==
        emel::error::cast(emel::cact::loader::error::model_invalid));
  CHECK(geometry.num_tensors == 0u);
  CHECK(
      machine.is(stateforward::sml::state<emel::cact::loader::state_errored>));

  std::array<emel::cact::loader::tensor_view, 1u> tensors = {};
  tensors[0].data = reinterpret_cast<const uint8_t *>(0x1u);
  const emel::cact::loader::event::bind_storage bind{
      std::span<emel::cact::loader::tensor_view>{tensors},
      k_bind_done_cb,
      k_bind_error_cb,
  };
  CHECK_FALSE(machine.process_event(bind));
  CHECK(state.bind_done_count == 0u);
  CHECK(state.bind_error_count == 1u);
  CHECK(tensors[0].data == reinterpret_cast<const uint8_t *>(0x1u));
}

TEST_CASE("cact loader validates exact tensor byte counts before bind") {
  struct malformed_case {
    const char *name;
    uint8_t dtype;
    uint8_t ndim;
    std::array<uint32_t, 4> shape;
    uint64_t nbytes;
    uint32_t group;
    uint32_t bits;
  };

  constexpr std::array<malformed_case, 11u> cases = {{
      {"fp16 zero", 1u, 1u, {16u, 0u, 0u, 0u}, 0u, 0u, 0u},
      {"fp16 undersized", 1u, 1u, {16u, 0u, 0u, 0u}, 30u, 0u, 0u},
      {"fp16 oversized", 1u, 1u, {16u, 0u, 0u, 0u}, 34u, 0u, 0u},
      {"fp32 zero", 2u, 2u, {4u, 8u, 0u, 0u}, 0u, 0u, 0u},
      {"fp32 undersized", 2u, 2u, {4u, 8u, 0u, 0u}, 124u, 0u, 0u},
      {"fp32 oversized", 2u, 2u, {4u, 8u, 0u, 0u}, 132u, 0u, 0u},
      {"raw zero", 4u, 0u, {0u, 0u, 0u, 0u}, 0u, 0u, 0u},
      {"raw shaped", 4u, 1u, {1u, 0u, 0u, 0u}, 1u, 0u, 0u},
      {"cq zero", 3u, 2u, {2u, 128u, 0u, 0u}, 0u, 128u, 4u},
      {"cq undersized", 3u, 2u, {2u, 128u, 0u, 0u}, 130u, 128u, 4u},
      {"cq oversized", 3u, 2u, {2u, 128u, 0u, 0u}, 134u, 128u, 4u},
  }};

  for (const malformed_case &test : cases) {
    CAPTURE(std::string{test.name});
    emel::cact::loader::sm machine{};
    callback_state state = {};
    callback_scope scope{state};
    const std::vector<uint8_t> file_bytes = make_single_tensor_cact_file(
        test.dtype, test.ndim, test.shape, test.nbytes, test.group, test.bits);
    emel::cact::loader::geometry geometry = {};
    geometry.num_tensors = 77u;
    const emel::cact::loader::event::probe probe{
        std::span<const uint8_t>{file_bytes},
        geometry,
        k_probe_done_cb,
        k_probe_error_cb,
    };

    CHECK_FALSE(machine.process_event(probe));
    CHECK(state.probe_done_count == 0u);
    CHECK(state.probe_error_count == 1u);
    CHECK(state.probe_error ==
          emel::error::cast(emel::cact::loader::error::model_invalid));
    CHECK(geometry.num_tensors == 77u);
    CHECK(machine.is(
        stateforward::sml::state<emel::cact::loader::state_errored>));
  }
}

TEST_CASE("cact loader validates RAW payload byte count from its schema") {
  const size_t expected_bytes = make_valid_raw_payload().size();
  for (const size_t nbytes : {expected_bytes - 1u, expected_bytes + 1u}) {
    CAPTURE(nbytes);
    emel::cact::loader::sm machine{};
    callback_state state = {};
    callback_scope scope{state};
    const std::vector<uint8_t> file_bytes = make_raw_cact_file(nbytes);
    emel::cact::loader::geometry geometry = {};
    const emel::cact::loader::event::probe probe{
        std::span<const uint8_t>{file_bytes},
        geometry,
        k_probe_done_cb,
        k_probe_error_cb,
    };

    CHECK_FALSE(machine.process_event(probe));
    CHECK(state.probe_done_count == 0u);
    CHECK(state.probe_error_count == 1u);
    CHECK(state.probe_error ==
          emel::error::cast(emel::cact::loader::error::model_invalid));
    CHECK(geometry.num_tensors == 0u);
  }

  emel::cact::loader::sm machine{};
  callback_state state = {};
  callback_scope scope{state};
  const std::vector<uint8_t> file_bytes = make_raw_cact_file(expected_bytes);
  emel::cact::loader::geometry geometry = {};
  const emel::cact::loader::event::probe probe{
      std::span<const uint8_t>{file_bytes},
      geometry,
      k_probe_done_cb,
      k_probe_error_cb,
  };
  CHECK(machine.process_event(probe));
  CHECK(state.probe_done_count == 1u);
  CHECK(state.probe_error_count == 0u);
  CHECK(geometry.num_tensors == 1u);
}

TEST_CASE("cact loader rejects overlapping tensor payload ranges") {
  emel::cact::loader::sm machine{};
  callback_state state = {};
  callback_scope scope{state};
  const std::vector<uint8_t> file_bytes =
      make_overlapping_tensor_ranges_cact_file();
  emel::cact::loader::geometry geometry = {};
  const emel::cact::loader::event::probe probe{
      std::span<const uint8_t>{file_bytes},
      geometry,
      k_probe_done_cb,
      k_probe_error_cb,
  };

  CHECK_FALSE(machine.process_event(probe));
  CHECK(state.probe_done_count == 0u);
  CHECK(state.probe_error_count == 1u);
  CHECK(state.probe_error ==
        emel::error::cast(emel::cact::loader::error::model_invalid));
  CHECK(geometry.num_tensors == 0u);
}

TEST_CASE("cact loader bind rejects undersized or unbound storage") {
  emel::cact::loader::sm machine{};
  callback_state state = {};
  callback_scope scope{state};

  const std::vector<uint8_t> file_bytes = make_valid_cact_file();
  emel::cact::loader::geometry geometry = {};
  const emel::cact::loader::event::probe probe{
      std::span<const uint8_t>{file_bytes},
      geometry,
      k_probe_done_cb,
      k_probe_error_cb,
  };
  CHECK(machine.process_event(probe));

  SUBCASE("empty storage span maps to invalid_request") {
    const emel::cact::loader::event::bind_storage bind{
        std::span<emel::cact::loader::tensor_view>{},
        k_bind_done_cb,
        k_bind_error_cb,
    };
    CHECK_FALSE(machine.process_event(bind));
    CHECK(state.bind_error ==
          emel::error::cast(emel::cact::loader::error::invalid_request));
    CHECK(machine.is(
        stateforward::sml::state<emel::cact::loader::state_errored>));
  }
}

TEST_CASE("cact loader parse before bind is rejected") {
  emel::cact::loader::sm machine{};
  callback_state state = {};
  callback_scope scope{state};

  const std::vector<uint8_t> file_bytes = make_valid_cact_file();
  emel::cact::loader::geometry geometry = {};
  const emel::cact::loader::event::probe probe{
      std::span<const uint8_t>{file_bytes},
      geometry,
      k_probe_done_cb,
      k_probe_error_cb,
  };
  CHECK(machine.process_event(probe));

  const emel::cact::loader::event::parse parse{
      std::span<const uint8_t>{file_bytes},
      k_parse_done_cb,
      k_parse_error_cb,
  };
  CHECK_FALSE(machine.process_event(parse));
  CHECK(state.parse_done_count == 0u);
  CHECK(state.parse_error ==
        emel::error::cast(emel::cact::loader::error::invalid_request));
  CHECK(
      machine.is(stateforward::sml::state<emel::cact::loader::state_errored>));
}

TEST_CASE("cact loader parses the pinned route-w4-qat fixture with directory "
          "parity") {
  const std::filesystem::path model_path =
      repo_relative("tests/models/route-w4-qat.cact");
  REQUIRE(std::filesystem::exists(model_path));
  const std::filesystem::path csv_path =
      repo_relative("tests/fixtures/cact/route-w4-qat.directory.csv");
  REQUIRE(std::filesystem::exists(csv_path));

  const std::vector<uint8_t> file_bytes = read_file_bytes(model_path);
  const std::vector<fixture_record> expected = read_directory_fixture(csv_path);
  REQUIRE(expected.size() == k_fixture_num_tensors);

  emel::cact::loader::sm machine{};
  callback_state state = {};
  callback_scope scope{state};

  emel::cact::loader::geometry geometry = {};
  const emel::cact::loader::event::probe probe{
      std::span<const uint8_t>{file_bytes},
      geometry,
      k_probe_done_cb,
      k_probe_error_cb,
  };
  CHECK(machine.process_event(probe));
  CHECK(state.probe_done_count == 1u);

  CHECK(geometry.num_tensors == k_fixture_num_tensors);
  CHECK(geometry.kv_window == k_fixture_kv_window);
  CHECK(geometry.kv_bits == k_fixture_kv_bits);
  CHECK(geometry.vocab_size == k_fixture_vocab);
  CHECK(geometry.d_model == k_fixture_d_model);
  CHECK(geometry.num_heads == k_fixture_num_heads);
  CHECK(geometry.num_kv_heads == k_fixture_num_kv_heads);
  CHECK(geometry.num_layers == k_fixture_num_layers);
  CHECK(geometry.head_dim == k_fixture_head_dim);
  CHECK(geometry.max_seq_len == k_fixture_max_seq_len);
  CHECK(geometry.hada_n == k_fixture_hada_n);
  CHECK(geometry.mhc_lanes == k_fixture_mhc_lanes);
  CHECK(geometry.engram_slots == 8192u);
  CHECK(geometry.engram_sub_dim == 128u);
  CHECK(geometry.num_engram_tables == 4u);
  CHECK(geometry.engram_conv_taps == 4u);
  CHECK(geometry.engram_conv_dilation == 3u);
  CHECK(geometry.num_engram_orders == 2u);
  CHECK(geometry.engram_orders[0] == 2u);
  CHECK(geometry.engram_orders[1] == 3u);
  CHECK(geometry.num_engram_sites == 2u);
  CHECK(geometry.engram_sites[0] == 2u);
  CHECK(geometry.engram_sites[1] == 15u);
  CHECK(geometry.rope_theta == doctest::Approx(k_fixture_rope_theta));

  std::vector<emel::cact::loader::tensor_view> tensors(geometry.num_tensors);
  const emel::cact::loader::event::bind_storage bind{
      std::span<emel::cact::loader::tensor_view>{tensors},
      k_bind_done_cb,
      k_bind_error_cb,
  };
  CHECK(machine.process_event(bind));

  const emel::cact::loader::event::parse parse{
      std::span<const uint8_t>{file_bytes},
      k_parse_done_cb,
      k_parse_error_cb,
  };
  CHECK(machine.process_event(parse));
  CHECK(state.parse_done_count == 1u);
  CHECK(machine.is(stateforward::sml::state<emel::cact::loader::state_parsed>));

  for (size_t i = 0; i < expected.size(); ++i) {
    const fixture_record &want = expected[i];
    const emel::cact::loader::tensor_view &got = tensors[i];
    CAPTURE(i);
    CHECK(got.dtype == want.dtype);
    CHECK(got.ndim == want.ndim);
    CHECK(got.shape[0] == want.shape[0]);
    CHECK(got.shape[1] == want.shape[1]);
    CHECK(got.shape[2] == want.shape[2]);
    CHECK(got.shape[3] == want.shape[3]);
    CHECK(got.offset == want.offset);
    CHECK(got.nbytes == want.nbytes);
    CHECK(got.group == want.group);
    CHECK(got.bits == want.bits);
    CHECK(got.data == file_bytes.data() + want.offset);
  }
}
