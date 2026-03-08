#include <array>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include "doctest/doctest.h"

#include "emel/gguf/loader/guards.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/model/data.hpp"

namespace {

constexpr uint32_t k_gguf_version = 3u;
constexpr uint32_t k_gguf_type_uint32 = 4u;
constexpr uint32_t k_gguf_type_string = 8u;
constexpr uint32_t k_gguf_type_array = 9u;
constexpr uint32_t k_ggml_type_f32 = 0u;
constexpr uint32_t k_valid_alignment = 32u;
constexpr uint64_t k_tensor_data_bytes = 32u;

template <class value_type>
void append_scalar(std::vector<uint8_t> & bytes, const value_type value) {
  using unsigned_type = std::make_unsigned_t<value_type>;
  const unsigned_type raw = static_cast<unsigned_type>(value);

  for (size_t i = 0; i < sizeof(value_type); ++i) {
    bytes.push_back(static_cast<uint8_t>((raw >> (i * 8u)) & 0xffu));
  }
}

void append_bytes(std::vector<uint8_t> & bytes, const std::string_view text) {
  bytes.insert(bytes.end(), text.begin(), text.end());
}

void append_string(std::vector<uint8_t> & bytes, const std::string_view text) {
  append_scalar<uint64_t>(bytes, static_cast<uint64_t>(text.size()));
  append_bytes(bytes, text);
}

void append_kv_u32(std::vector<uint8_t> & bytes,
                   const std::string_view key,
                   const uint32_t value) {
  append_string(bytes, key);
  append_scalar<uint32_t>(bytes, k_gguf_type_uint32);
  append_scalar<uint32_t>(bytes, value);
}

void append_kv_string_array(std::vector<uint8_t> & bytes,
                            const std::string_view key,
                            const std::span<const std::string_view> values) {
  append_string(bytes, key);
  append_scalar<uint32_t>(bytes, k_gguf_type_array);
  append_scalar<uint32_t>(bytes, k_gguf_type_string);
  append_scalar<uint64_t>(bytes, static_cast<uint64_t>(values.size()));

  for (const std::string_view value : values) {
    append_string(bytes, value);
  }
}

void append_tensor_info(std::vector<uint8_t> & bytes,
                        const std::string_view name,
                        const std::span<const uint64_t> dims,
                        const uint32_t type,
                        const uint64_t offset) {
  append_string(bytes, name);
  append_scalar<uint32_t>(bytes, static_cast<uint32_t>(dims.size()));

  for (const uint64_t dim : dims) {
    append_scalar<uint64_t>(bytes, dim);
  }

  append_scalar<uint32_t>(bytes, type);
  append_scalar<uint64_t>(bytes, offset);
}

void align_bytes(std::vector<uint8_t> & bytes, const uint32_t alignment) {
  while ((bytes.size() % alignment) != 0u) {
    bytes.push_back(0u);
  }
}

std::vector<uint8_t> make_valid_gguf_file() {
  std::vector<uint8_t> bytes;
  const std::array<std::string_view, 2> tokens = {"hi", "world"};
  const std::array<uint64_t, 2> dims = {2u, 3u};

  append_bytes(bytes, "GGUF");
  append_scalar<uint32_t>(bytes, k_gguf_version);
  append_scalar<uint64_t>(bytes, 1u);
  append_scalar<uint64_t>(bytes, 2u);
  append_kv_u32(bytes, "general.alignment", k_valid_alignment);
  append_kv_string_array(bytes, "tokenizer.tokens", std::span<const std::string_view>{tokens});
  append_tensor_info(bytes, "weights.f32", std::span<const uint64_t>{dims}, k_ggml_type_f32, 0u);
  align_bytes(bytes, k_valid_alignment);
  bytes.resize(bytes.size() + k_tensor_data_bytes, 0u);
  return bytes;
}

std::vector<uint8_t> make_bad_magic_gguf_file() {
  std::vector<uint8_t> bytes = make_valid_gguf_file();
  bytes[0] = 'F';
  return bytes;
}

std::vector<uint8_t> make_bad_version_gguf_file() {
  std::vector<uint8_t> bytes = make_valid_gguf_file();
  bytes[4] = 0x04u;
  return bytes;
}

std::vector<uint8_t> make_truncated_gguf_file() {
  std::vector<uint8_t> bytes = make_valid_gguf_file();
  bytes.pop_back();
  return bytes;
}

struct callback_state {
  uint32_t probe_done_count = 0;
  uint32_t probe_error_count = 0;
  uint32_t bind_done_count = 0;
  uint32_t bind_error_count = 0;
  uint32_t parse_done_count = 0;
  uint32_t parse_error_count = 0;
  emel::gguf::loader::requirements probe_requirements = {};
  emel::error::type probe_error = emel::error::cast(emel::gguf::loader::error::none);
  emel::error::type bind_error = emel::error::cast(emel::gguf::loader::error::none);
  emel::error::type parse_error = emel::error::cast(emel::gguf::loader::error::none);
};

callback_state * g_callback_state = nullptr;

struct callback_scope {
  explicit callback_scope(callback_state & state) noexcept {
    g_callback_state = &state;
  }

  ~callback_scope() {
    g_callback_state = nullptr;
  }
};

void on_probe_done(const emel::gguf::loader::events::probe_done & ev) {
  if (g_callback_state == nullptr) {
    return;
  }

  ++g_callback_state->probe_done_count;
  g_callback_state->probe_requirements = ev.requirements_out;
}

void on_probe_error(const emel::gguf::loader::events::probe_error & ev) {
  if (g_callback_state == nullptr) {
    return;
  }

  ++g_callback_state->probe_error_count;
  g_callback_state->probe_error = ev.err;
}

void on_bind_done(const emel::gguf::loader::events::bind_done &) {
  if (g_callback_state != nullptr) {
    ++g_callback_state->bind_done_count;
  }
}

void on_bind_error(const emel::gguf::loader::events::bind_error & ev) {
  if (g_callback_state == nullptr) {
    return;
  }

  ++g_callback_state->bind_error_count;
  g_callback_state->bind_error = ev.err;
}

void on_parse_done(const emel::gguf::loader::events::parse_done &) {
  if (g_callback_state != nullptr) {
    ++g_callback_state->parse_done_count;
  }
}

void on_parse_error(const emel::gguf::loader::events::parse_error & ev) {
  if (g_callback_state == nullptr) {
    return;
  }

  ++g_callback_state->parse_error_count;
  g_callback_state->parse_error = ev.err;
}

}  // namespace

TEST_CASE("gguf loader probe bind parse lifecycle") {
  emel::gguf::loader::sm machine{};
  callback_state state = {};
  callback_scope scope{state};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb =
      emel::gguf::loader::event::probe_done_fn::from<&on_probe_done>();
  const emel::gguf::loader::event::probe_error_fn probe_error_cb =
      emel::gguf::loader::event::probe_error_fn::from<&on_probe_error>();
  const emel::gguf::loader::event::bind_done_fn bind_done_cb =
      emel::gguf::loader::event::bind_done_fn::from<&on_bind_done>();
  const emel::gguf::loader::event::bind_error_fn bind_error_cb =
      emel::gguf::loader::event::bind_error_fn::from<&on_bind_error>();
  const emel::gguf::loader::event::parse_done_fn parse_done_cb =
      emel::gguf::loader::event::parse_done_fn::from<&on_parse_done>();
  const emel::gguf::loader::event::parse_error_fn parse_error_cb =
      emel::gguf::loader::event::parse_error_fn::from<&on_parse_error>();

  const std::vector<uint8_t> file_bytes = make_valid_gguf_file();
  emel::gguf::loader::requirements req = {};
  const emel::gguf::loader::event::probe probe{
    std::span<const uint8_t>{file_bytes},
    req,
    probe_done_cb,
    probe_error_cb,
  };

  CHECK(machine.process_event(probe));
  CHECK(state.probe_done_count == 1u);
  CHECK(state.probe_error_count == 0u);
  CHECK(state.probe_error == emel::error::cast(emel::gguf::loader::error::none));
  CHECK(req.tensor_count == 1u);
  CHECK(req.kv_count == 2u);
  CHECK(req.max_key_bytes == 17u);
  CHECK(req.max_value_bytes == 35u);

  std::vector<uint8_t> kv_arena(
      emel::gguf::loader::detail::required_kv_arena_bytes(req), 0u);
  std::vector<emel::gguf::loader::kv_entry> kv_entries(req.kv_count);
  std::vector<emel::model::data::tensor_record> tensors(req.tensor_count);
  const emel::gguf::loader::event::bind_storage bind{
    std::span<uint8_t>{kv_arena},
    std::span<emel::gguf::loader::kv_entry>{kv_entries},
    std::span<emel::model::data::tensor_record>{tensors},
    bind_done_cb,
    bind_error_cb,
  };

  CHECK(machine.process_event(bind));
  CHECK(state.bind_done_count == 1u);
  CHECK(state.bind_error_count == 0u);
  CHECK(state.bind_error == emel::error::cast(emel::gguf::loader::error::none));

  const emel::gguf::loader::event::parse parse{
    std::span<const uint8_t>{file_bytes},
    parse_done_cb,
    parse_error_cb,
  };

  CHECK(machine.process_event(parse));
  CHECK(state.parse_done_count == 1u);
  CHECK(state.parse_error_count == 0u);
  CHECK(state.parse_error == emel::error::cast(emel::gguf::loader::error::none));
}

TEST_CASE("gguf loader probe rejects invalid request inputs") {
  emel::gguf::loader::sm machine{};
  callback_state state = {};
  callback_scope scope{state};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb =
      emel::gguf::loader::event::probe_done_fn::from<&on_probe_done>();
  const emel::gguf::loader::event::probe_error_fn probe_error_cb =
      emel::gguf::loader::event::probe_error_fn::from<&on_probe_error>();

  emel::gguf::loader::requirements req = {};
  const emel::gguf::loader::event::probe probe{
    std::span<const uint8_t>{},
    req,
    probe_done_cb,
    probe_error_cb,
  };

  CHECK_FALSE(machine.process_event(probe));
  CHECK(state.probe_done_count == 0u);
  CHECK(state.probe_error_count == 1u);
  CHECK(state.probe_error == emel::error::cast(emel::gguf::loader::error::invalid_request));
}

TEST_CASE("gguf loader probe classifies malformed file images") {
  emel::gguf::loader::sm machine{};
  callback_state state = {};
  callback_scope scope{state};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb =
      emel::gguf::loader::event::probe_done_fn::from<&on_probe_done>();
  const emel::gguf::loader::event::probe_error_fn probe_error_cb =
      emel::gguf::loader::event::probe_error_fn::from<&on_probe_error>();

  auto probe_error_for = [&](const std::vector<uint8_t> & file_bytes) {
    state = {};
    emel::gguf::loader::requirements req = {};
    const emel::gguf::loader::event::probe probe{
      std::span<const uint8_t>{file_bytes},
      req,
      probe_done_cb,
      probe_error_cb,
    };

    CHECK_FALSE(machine.process_event(probe));
    CHECK(state.probe_done_count == 0u);
    CHECK(state.probe_error_count == 1u);
    return state.probe_error;
  };

  CHECK(probe_error_for(make_bad_magic_gguf_file()) ==
        emel::error::cast(emel::gguf::loader::error::model_invalid));
  CHECK(probe_error_for(make_bad_version_gguf_file()) ==
        emel::error::cast(emel::gguf::loader::error::model_invalid));
  CHECK(probe_error_for(make_truncated_gguf_file()) ==
        emel::error::cast(emel::gguf::loader::error::parse_failed));
}

TEST_CASE("gguf loader explicit error guard classification") {
  emel::gguf::loader::action::context ctx = {};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb =
      emel::gguf::loader::event::probe_done_fn::from<&on_probe_done>();
  const emel::gguf::loader::event::probe_error_fn probe_error_cb =
      emel::gguf::loader::event::probe_error_fn::from<&on_probe_error>();
  const emel::gguf::loader::event::bind_done_fn bind_done_cb =
      emel::gguf::loader::event::bind_done_fn::from<&on_bind_done>();
  const emel::gguf::loader::event::bind_error_fn bind_error_cb =
      emel::gguf::loader::event::bind_error_fn::from<&on_bind_error>();
  const emel::gguf::loader::event::parse_done_fn parse_done_cb =
      emel::gguf::loader::event::parse_done_fn::from<&on_parse_done>();
  const emel::gguf::loader::event::parse_error_fn parse_error_cb =
      emel::gguf::loader::event::parse_error_fn::from<&on_parse_error>();

  std::array<uint8_t, 4> file_bytes = {};
  emel::gguf::loader::requirements req = {};
  emel::gguf::loader::event::probe probe{
    std::span<const uint8_t>{file_bytes},
    req,
    probe_done_cb,
    probe_error_cb,
  };
  emel::gguf::loader::event::probe_ctx probe_ctx = {};
  emel::gguf::loader::event::probe_runtime probe_runtime{probe, probe_ctx};

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::none);
  CHECK(emel::gguf::loader::guard::probe_error_none{}(probe_runtime, ctx));

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::invalid_request);
  CHECK(emel::gguf::loader::guard::probe_error_invalid_request{}(probe_runtime, ctx));

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::model_invalid);
  CHECK(emel::gguf::loader::guard::probe_error_model_invalid{}(probe_runtime, ctx));

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::capacity);
  CHECK(emel::gguf::loader::guard::probe_error_capacity{}(probe_runtime, ctx));

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::parse_failed);
  CHECK(emel::gguf::loader::guard::probe_error_parse_failed{}(probe_runtime, ctx));

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::internal_error);
  CHECK(emel::gguf::loader::guard::probe_error_internal_error{}(probe_runtime, ctx));

  probe_ctx.err = emel::error::cast(emel::gguf::loader::error::untracked);
  CHECK(emel::gguf::loader::guard::probe_error_untracked{}(probe_runtime, ctx));

  probe_ctx.err = 0x7fff;
  CHECK(emel::gguf::loader::guard::probe_error_unknown{}(probe_runtime, ctx));

  std::array<uint8_t, 8> kv_arena = {};
  std::array<emel::gguf::loader::kv_entry, 1> kv_entries = {};
  std::array<emel::model::data::tensor_record, 1> tensors = {};
  emel::gguf::loader::event::bind_storage bind{
    std::span<uint8_t>{kv_arena},
    std::span<emel::gguf::loader::kv_entry>{kv_entries},
    std::span<emel::model::data::tensor_record>{tensors},
    bind_done_cb,
    bind_error_cb,
  };
  emel::gguf::loader::event::bind_ctx bind_ctx = {};
  emel::gguf::loader::event::bind_runtime bind_runtime{bind, bind_ctx};

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::none);
  CHECK(emel::gguf::loader::guard::bind_error_none{}(bind_runtime, ctx));

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::invalid_request);
  CHECK(emel::gguf::loader::guard::bind_error_invalid_request{}(bind_runtime, ctx));

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::model_invalid);
  CHECK(emel::gguf::loader::guard::bind_error_model_invalid{}(bind_runtime, ctx));

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::capacity);
  CHECK(emel::gguf::loader::guard::bind_error_capacity{}(bind_runtime, ctx));

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::parse_failed);
  CHECK(emel::gguf::loader::guard::bind_error_parse_failed{}(bind_runtime, ctx));

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::internal_error);
  CHECK(emel::gguf::loader::guard::bind_error_internal_error{}(bind_runtime, ctx));

  bind_ctx.err = emel::error::cast(emel::gguf::loader::error::untracked);
  CHECK(emel::gguf::loader::guard::bind_error_untracked{}(bind_runtime, ctx));

  bind_ctx.err = 0x7fff;
  CHECK(emel::gguf::loader::guard::bind_error_unknown{}(bind_runtime, ctx));

  emel::gguf::loader::event::parse parse{
    std::span<const uint8_t>{file_bytes},
    parse_done_cb,
    parse_error_cb,
  };
  emel::gguf::loader::event::parse_ctx parse_ctx = {};
  emel::gguf::loader::event::parse_runtime parse_runtime{parse, parse_ctx};

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::none);
  CHECK(emel::gguf::loader::guard::parse_error_none{}(parse_runtime, ctx));

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::invalid_request);
  CHECK(emel::gguf::loader::guard::parse_error_invalid_request{}(parse_runtime, ctx));

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::model_invalid);
  CHECK(emel::gguf::loader::guard::parse_error_model_invalid{}(parse_runtime, ctx));

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::capacity);
  CHECK(emel::gguf::loader::guard::parse_error_capacity{}(parse_runtime, ctx));

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::parse_failed);
  CHECK(emel::gguf::loader::guard::parse_error_parse_failed{}(parse_runtime, ctx));

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::internal_error);
  CHECK(emel::gguf::loader::guard::parse_error_internal_error{}(parse_runtime, ctx));

  parse_ctx.err = emel::error::cast(emel::gguf::loader::error::untracked);
  CHECK(emel::gguf::loader::guard::parse_error_untracked{}(parse_runtime, ctx));

  parse_ctx.err = 0x7fff;
  CHECK(emel::gguf::loader::guard::parse_error_unknown{}(parse_runtime, ctx));
}
