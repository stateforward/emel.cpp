#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "emel/error/error.hpp"
#include "emel/gguf/loader/detail.hpp"
#include "emel/gguf/loader/errors.hpp"
#include "emel/gguf/loader/events.hpp"
#include "emel/gguf/loader/sm.hpp"
#include "emel/generator/errors.hpp"
#include "emel/generator/events.hpp"
#include "emel/generator/sm.hpp"
#include "emel/logits/sampler/events.hpp"
#include "emel/model/data.hpp"
#include "emel/model/detail.hpp"
#include "emel/model/loader/errors.hpp"
#include "emel/model/loader/events.hpp"
#include "emel/model/loader/sm.hpp"
#include "emel/model/weight_loader/errors.hpp"
#include "emel/model/weight_loader/events.hpp"
#include "emel/model/weight_loader/sm.hpp"
#include "emel/text/conditioner/sm.hpp"
#include "emel/text/formatter/format.hpp"
#include "emel/text/tokenizer/sm.hpp"

namespace {

constexpr std::string_view k_prompt = "hello";
constexpr int32_t k_max_tokens = 1;
constexpr size_t k_output_capacity = 4096u;
constexpr std::array<std::string_view, 9> k_supported_qwen_primary_template_markers = {
    "<|im_start|>",
    "<|im_end|>",
    "messages[0].role == 'system'",
    "message.role == \"user\"",
    "message.role == \"assistant\"",
    "add_generation_prompt",
    "enable_thinking",
    "<tool_call>",
    "tool_response",
};
constexpr std::string_view k_im_start = "<|im_start|>";
constexpr std::string_view k_im_end = "<|im_end|>\n";
constexpr std::string_view k_assistant_generation_prefix = "<|im_start|>assistant\n";
constexpr std::string_view k_message_separator = "\n";
int k_supported_qwen_formatter_sentinel = 0;

struct formatter_binding {
  void * formatter_ctx = nullptr;
  emel::text::formatter::format_fn format_prompt = emel::text::formatter::format_raw;
  bool supported = false;
};

struct gguf_capture {
  bool probe_done = false;
  bool probe_error = false;
  bool bind_done = false;
  bool bind_error = false;
  bool parse_done = false;
  bool parse_error = false;
  emel::gguf::loader::requirements requirements = {};
  emel::error::type err = emel::error::cast(emel::gguf::loader::error::none);
};

struct weight_capture {
  bool bind_done = false;
  bool bind_error = false;
  bool plan_done = false;
  bool plan_error = false;
  bool apply_done = false;
  bool apply_error = false;
  uint32_t effect_count = 0u;
  emel::error::type err = emel::error::cast(emel::model::weight_loader::error::none);
};

struct load_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::model::loader::error::none);
  uint64_t bytes_total = 0u;
  uint64_t bytes_done = 0u;
  bool used_mmap = false;
};

struct initialize_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::generator::error::none);
};

struct generation_capture {
  bool done = false;
  bool error = false;
  emel::error::type err = emel::error::cast(emel::generator::error::none);
  int32_t tokens_generated = 0;
  size_t output_length = 0u;
};

struct generation_result {
  std::array<char, k_output_capacity> output = {};
  int32_t tokens_generated = 0;
  size_t output_length = 0u;
};

struct emel_fixture {
  std::unique_ptr<emel::model::data> model_data = std::make_unique<emel::model::data>();
  std::vector<uint8_t> file_bytes = {};
  std::vector<uint8_t> kv_arena = {};
  std::vector<emel::gguf::loader::kv_entry> kv_entries = {};
  std::vector<emel::model::weight_loader::effect_request> effect_requests = {};
  std::vector<emel::model::weight_loader::effect_result> effect_results = {};
  emel::gguf::loader::sm gguf_loader = {};
  emel::model::weight_loader::sm weight_loader = {};
  emel::model::loader::sm model_loader = {};
  gguf_capture gguf = {};
  weight_capture weight = {};
  load_capture load = {};
  formatter_binding formatter = {};
};

struct emel_session {
  std::unique_ptr<emel::model::data> model_data = std::make_unique<emel::model::data>();
  emel::text::tokenizer::sm tokenizer = {};
  emel::text::conditioner::sm conditioner = {};
  std::unique_ptr<emel::generator::sm> generator = {};
  formatter_binding formatter = {};
  initialize_capture initialize = {};
  generation_capture generation = {};
};

template <size_t k_array_size>
void copy_name(std::array<char, k_array_size> & dst, const std::string_view value) {
  static_assert(k_array_size > 0u, "copy_name requires non-empty destination");
  dst.fill('\0');
  const size_t copy_len = std::min(value.size(), k_array_size - 1u);
  if (copy_len > 0u) {
    std::memcpy(dst.data(), value.data(), copy_len);
  }
}

uint32_t read_u32_le(const std::span<const uint8_t> bytes) {
  uint32_t value = 0u;
  for (size_t index = 0u; index < sizeof(uint32_t); ++index) {
    value |= static_cast<uint32_t>(bytes[index]) << (index * 8u);
  }
  return value;
}

uint64_t read_u64_le(const std::span<const uint8_t> bytes) {
  uint64_t value = 0u;
  for (size_t index = 0u; index < sizeof(uint64_t); ++index) {
    value |= static_cast<uint64_t>(bytes[index]) << (index * 8u);
  }
  return value;
}

bool read_file_bytes(const std::string & path, std::vector<uint8_t> & out) {
  out.clear();
  std::FILE * file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return false;
  }

  const bool seek_end_ok = std::fseek(file, 0, SEEK_END) == 0;
  const long file_size = seek_end_ok ? std::ftell(file) : -1L;
  const bool seek_start_ok = file_size >= 0L && std::fseek(file, 0, SEEK_SET) == 0;
  if (!seek_end_ok || file_size < 0L || !seek_start_ok) {
    std::fclose(file);
    return false;
  }

  out.resize(static_cast<size_t>(file_size));
  const size_t read_size = out.empty() ? 0u : std::fread(out.data(), 1u, out.size(), file);
  std::fclose(file);
  return read_size == out.size();
}

bool qwen_role_supported(const std::string_view role) noexcept {
  return role == "system" || role == "user" || role == "assistant";
}

bool qwen_messages_supported(
    const std::span<const emel::text::formatter::chat_message> messages) noexcept {
  if (messages.empty()) {
    return false;
  }

  for (const auto & message : messages) {
    if (!qwen_role_supported(message.role)) {
      return false;
    }
  }
  return true;
}

bool append_bytes(const std::string_view bytes,
                  char * output,
                  const size_t output_capacity,
                  size_t & offset) noexcept {
  if (offset + bytes.size() > output_capacity) {
    return false;
  }
  if (!bytes.empty()) {
    std::memcpy(output + offset, bytes.data(), bytes.size());
  }
  offset += bytes.size();
  return true;
}

bool format_supported_qwen_contract(void *,
                                    const emel::text::formatter::format_request & request,
                                    int32_t * error_out) noexcept {
  if (error_out != nullptr) {
    *error_out = emel::text::formatter::error_code(emel::text::formatter::error::none);
  }
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0u;
  }
  if ((request.output == nullptr && request.output_capacity > 0u) ||
      !qwen_messages_supported(request.messages) ||
      !request.add_generation_prompt ||
      request.enable_thinking) {
    if (error_out != nullptr) {
      *error_out = emel::text::formatter::error_code(
          emel::text::formatter::error::invalid_request);
    }
    return false;
  }

  size_t output_length = 0u;
  for (const auto & message : request.messages) {
    if (!append_bytes(k_im_start, request.output, request.output_capacity, output_length) ||
        !append_bytes(message.role, request.output, request.output_capacity, output_length) ||
        !append_bytes(k_message_separator,
                      request.output,
                      request.output_capacity,
                      output_length) ||
        !append_bytes(message.content, request.output, request.output_capacity, output_length) ||
        !append_bytes(k_im_end, request.output, request.output_capacity, output_length)) {
      if (error_out != nullptr) {
        *error_out = emel::text::formatter::error_code(
            emel::text::formatter::error::invalid_request);
      }
      return false;
    }
  }

  if (!append_bytes(k_assistant_generation_prefix,
                    request.output,
                    request.output_capacity,
                    output_length)) {
    if (error_out != nullptr) {
      *error_out = emel::text::formatter::error_code(
          emel::text::formatter::error::invalid_request);
    }
    return false;
  }

  if (request.output_length_out != nullptr) {
    *request.output_length_out = output_length;
  }
  return true;
}

bool template_matches_supported_qwen_contract(const std::string_view primary_template) noexcept {
  for (const std::string_view marker : k_supported_qwen_primary_template_markers) {
    if (primary_template.find(marker) == std::string_view::npos) {
      return false;
    }
  }
  return true;
}

std::span<const emel::text::formatter::chat_message> single_user_messages(
    std::array<emel::text::formatter::chat_message, 1> & storage,
    const std::string_view text) noexcept {
  storage[0] = emel::text::formatter::chat_message{
      .role = "user",
      .content = text,
  };
  return std::span<const emel::text::formatter::chat_message>{storage};
}

size_t formatted_capacity_upper_bound(
    const std::span<const emel::text::formatter::chat_message> messages,
    const bool add_generation_prompt) noexcept {
  size_t capacity = 0u;
  for (const auto & message : messages) {
    capacity += k_im_start.size() + message.role.size() + k_message_separator.size() +
        message.content.size() + k_im_end.size();
  }
  if (add_generation_prompt) {
    capacity += k_assistant_generation_prefix.size();
  }
  return capacity;
}

bool format_single_user_prompt(const formatter_binding & binding,
                               const std::string_view text,
                               std::string & output) {
  if (!binding.supported) {
    output.clear();
    return false;
  }

  std::array<emel::text::formatter::chat_message, 1> message_storage = {};
  const auto messages = single_user_messages(message_storage, text);
  output.resize(formatted_capacity_upper_bound(messages, true));
  size_t output_length = 0u;
  int32_t error_out = emel::text::formatter::error_code(emel::text::formatter::error::none);
  const emel::text::formatter::format_request request{
      .messages = messages,
      .add_generation_prompt = true,
      .enable_thinking = false,
      .output = output.data(),
      .output_capacity = output.size(),
      .output_length_out = &output_length,
  };
  if (!binding.format_prompt(binding.formatter_ctx, request, &error_out)) {
    output.clear();
    return false;
  }
  output.resize(output_length);
  return true;
}

emel::text::tokenizer::preprocessor::preprocessor_kind generation_preprocessor_variant(
    const emel::model::data & model_data) {
  using preprocessor_kind = emel::text::tokenizer::preprocessor::preprocessor_kind;
  using tokenizer_model = emel::model::data::tokenizer_model;

  switch (model_data.vocab_data.tokenizer_model_id) {
    case tokenizer_model::SPM:
      return preprocessor_kind::spm;
    case tokenizer_model::BPE:
      return preprocessor_kind::bpe;
    case tokenizer_model::WPM:
      return preprocessor_kind::wpm;
    case tokenizer_model::UGM:
      return preprocessor_kind::ugm;
    case tokenizer_model::RWKV:
      return preprocessor_kind::rwkv;
    case tokenizer_model::PLAMO2:
      return preprocessor_kind::plamo2;
    case tokenizer_model::NONE:
    case tokenizer_model::UNKNOWN:
    default:
      return preprocessor_kind::fallback;
  }
}

emel::text::encoders::encoder_kind generation_encoder_variant(
    const emel::model::data & model_data) {
  using encoder_kind = emel::text::encoders::encoder_kind;
  using tokenizer_model = emel::model::data::tokenizer_model;

  switch (model_data.vocab_data.tokenizer_model_id) {
    case tokenizer_model::SPM:
      return encoder_kind::spm;
    case tokenizer_model::BPE:
      return encoder_kind::bpe;
    case tokenizer_model::WPM:
      return encoder_kind::wpm;
    case tokenizer_model::UGM:
      return encoder_kind::ugm;
    case tokenizer_model::RWKV:
      return encoder_kind::rwkv;
    case tokenizer_model::PLAMO2:
      return encoder_kind::plamo2;
    case tokenizer_model::NONE:
    case tokenizer_model::UNKNOWN:
    default:
      return encoder_kind::fallback;
  }
}

template <class fixture_type>
void reset_gguf_capture(fixture_type & fixture) {
  fixture.gguf = {};
}

void reset_weight_capture(emel_fixture & fixture) { fixture.weight = {}; }
void reset_load_capture(emel_fixture & fixture) { fixture.load = {}; }
void reset_initialize_capture(emel_session & session) { session.initialize = {}; }
void reset_generation_capture(emel_session & session) { session.generation = {}; }

template <class fixture_type>
void on_probe_done_impl(void * owner, const emel::gguf::loader::events::probe_done & ev) {
  auto & fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.probe_done = true;
  fixture.gguf.probe_error = false;
  fixture.gguf.requirements = ev.requirements_out;
}

template <class fixture_type>
void on_probe_error_impl(void * owner, const emel::gguf::loader::events::probe_error & ev) {
  auto & fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.probe_error = true;
  fixture.gguf.err = ev.err;
}

template <class fixture_type>
void on_bind_done_impl(void * owner, const emel::gguf::loader::events::bind_done &) {
  auto & fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.bind_done = true;
  fixture.gguf.bind_error = false;
}

template <class fixture_type>
void on_bind_error_impl(void * owner, const emel::gguf::loader::events::bind_error & ev) {
  auto & fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.bind_error = true;
  fixture.gguf.err = ev.err;
}

template <class fixture_type>
void on_parse_done_impl(void * owner, const emel::gguf::loader::events::parse_done &) {
  auto & fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.parse_done = true;
  fixture.gguf.parse_error = false;
}

template <class fixture_type>
void on_parse_error_impl(void * owner, const emel::gguf::loader::events::parse_error & ev) {
  auto & fixture = *static_cast<fixture_type *>(owner);
  fixture.gguf.parse_error = true;
  fixture.gguf.err = ev.err;
}

void on_probe_done(void * owner, const emel::gguf::loader::events::probe_done & ev) {
  on_probe_done_impl<emel_fixture>(owner, ev);
}

void on_probe_error(void * owner, const emel::gguf::loader::events::probe_error & ev) {
  on_probe_error_impl<emel_fixture>(owner, ev);
}

void on_bind_done(void * owner, const emel::gguf::loader::events::bind_done & ev) {
  on_bind_done_impl<emel_fixture>(owner, ev);
}

void on_bind_error(void * owner, const emel::gguf::loader::events::bind_error & ev) {
  on_bind_error_impl<emel_fixture>(owner, ev);
}

void on_parse_done(void * owner, const emel::gguf::loader::events::parse_done & ev) {
  on_parse_done_impl<emel_fixture>(owner, ev);
}

void on_parse_error(void * owner, const emel::gguf::loader::events::parse_error & ev) {
  on_parse_error_impl<emel_fixture>(owner, ev);
}

void on_weight_bind_done(void * owner, const emel::model::weight_loader::events::bind_done &) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.weight.bind_done = true;
  fixture.weight.bind_error = false;
}

void on_weight_bind_error(void * owner,
                          const emel::model::weight_loader::events::bind_error & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.weight.bind_error = true;
  fixture.weight.err = ev.err;
}

void on_weight_plan_done(void * owner,
                         const emel::model::weight_loader::events::plan_done & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.weight.plan_done = true;
  fixture.weight.plan_error = false;
  fixture.weight.effect_count = ev.effect_count;
}

void on_weight_plan_error(void * owner,
                          const emel::model::weight_loader::events::plan_error & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.weight.plan_error = true;
  fixture.weight.err = ev.err;
}

void on_weight_apply_done(void * owner, const emel::model::weight_loader::events::apply_done &) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.weight.apply_done = true;
  fixture.weight.apply_error = false;
}

void on_weight_apply_error(void * owner,
                           const emel::model::weight_loader::events::apply_error & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.weight.apply_error = true;
  fixture.weight.err = ev.err;
}

void on_load_done(void * owner, const emel::model::loader::events::load_done & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.load.done = true;
  fixture.load.error = false;
  fixture.load.err = emel::error::cast(emel::model::loader::error::none);
  fixture.load.bytes_total = ev.bytes_total;
  fixture.load.bytes_done = ev.bytes_done;
  fixture.load.used_mmap = ev.used_mmap;
}

void on_load_error(void * owner, const emel::model::loader::events::load_error & ev) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  fixture.load.error = true;
  fixture.load.err = ev.err;
}

void on_initialize_done(void * owner, const emel::generator::events::initialize_done &) {
  auto & session = *static_cast<emel_session *>(owner);
  session.initialize.done = true;
  session.initialize.error = false;
  session.initialize.err = emel::error::cast(emel::generator::error::none);
}

void on_initialize_error(void * owner, const emel::generator::events::initialize_error & ev) {
  auto & session = *static_cast<emel_session *>(owner);
  session.initialize.error = true;
  session.initialize.err = ev.err;
}

void on_generation_done(void * owner, const emel::generator::events::generation_done & ev) {
  auto & session = *static_cast<emel_session *>(owner);
  session.generation.done = true;
  session.generation.error = false;
  session.generation.err = emel::error::cast(emel::generator::error::none);
  session.generation.tokens_generated = ev.tokens_generated;
  session.generation.output_length = ev.output_length;
}

void on_generation_error(void * owner, const emel::generator::events::generation_error & ev) {
  auto & session = *static_cast<emel_session *>(owner);
  session.generation.error = true;
  session.generation.err = ev.err;
  session.generation.tokens_generated = ev.tokens_generated;
  session.generation.output_length = ev.output_length;
}

bool tokenizer_bind_dispatch(void * tokenizer_sm,
                             const emel::text::tokenizer::event::bind & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

bool tokenizer_tokenize_dispatch(void * tokenizer_sm,
                                 const emel::text::tokenizer::event::tokenize & ev) {
  return static_cast<emel::text::tokenizer::sm *>(tokenizer_sm)->process_event(ev);
}

emel::error::type map_gguf_error(const emel::error::type err) {
  using gguf_error = emel::gguf::loader::error;
  using model_error = emel::model::loader::error;

  switch (err) {
    case emel::error::cast(gguf_error::none):
      return emel::error::cast(model_error::none);
    case emel::error::cast(gguf_error::invalid_request):
      return emel::error::cast(model_error::invalid_request);
    case emel::error::cast(gguf_error::model_invalid):
      return emel::error::cast(model_error::model_invalid);
    case emel::error::cast(gguf_error::capacity):
      return emel::error::cast(model_error::backend_error);
    case emel::error::cast(gguf_error::parse_failed):
      return emel::error::cast(model_error::parse_failed);
    case emel::error::cast(gguf_error::internal_error):
      return emel::error::cast(model_error::internal_error);
    case emel::error::cast(gguf_error::untracked):
    default:
      return emel::error::cast(model_error::untracked);
  }
}

emel::error::type map_weight_loader_error(const emel::error::type err) {
  using model_error = emel::model::loader::error;
  using weight_error = emel::model::weight_loader::error;

  switch (err) {
    case emel::error::cast(weight_error::none):
      return emel::error::cast(model_error::none);
    case emel::error::cast(weight_error::invalid_request):
      return emel::error::cast(model_error::invalid_request);
    case emel::error::cast(weight_error::capacity):
    case emel::error::cast(weight_error::backend_error):
    case emel::error::cast(weight_error::out_of_memory):
      return emel::error::cast(model_error::backend_error);
    case emel::error::cast(weight_error::model_invalid):
      return emel::error::cast(model_error::model_invalid);
    case emel::error::cast(weight_error::internal_error):
      return emel::error::cast(model_error::internal_error);
    case emel::error::cast(weight_error::untracked):
    default:
      return emel::error::cast(model_error::untracked);
  }
}

template <class fixture_type>
std::string_view kv_key_view(const fixture_type & fixture,
                             const emel::gguf::loader::kv_entry & entry) {
  if (static_cast<size_t>(entry.key_offset) + static_cast<size_t>(entry.key_length) >
      fixture.kv_arena.size()) {
    return {};
  }

  return std::string_view{
      reinterpret_cast<const char *>(fixture.kv_arena.data() + entry.key_offset),
      entry.key_length,
  };
}

template <class fixture_type>
std::span<const uint8_t> kv_value_view(const fixture_type & fixture,
                                       const emel::gguf::loader::kv_entry & entry) {
  if (static_cast<size_t>(entry.value_offset) + static_cast<size_t>(entry.value_length) >
      fixture.kv_arena.size()) {
    return {};
  }

  return std::span<const uint8_t>{fixture.kv_arena.data() + entry.value_offset, entry.value_length};
}

template <class fixture_type>
const emel::gguf::loader::kv_entry * find_kv_entry(const fixture_type & fixture,
                                                   const std::string_view key) {
  for (const auto & entry : fixture.kv_entries) {
    if (kv_key_view(fixture, entry) == key) {
      return &entry;
    }
  }
  return nullptr;
}

template <class fixture_type>
bool decode_integer_value(const fixture_type & fixture,
                          const emel::gguf::loader::kv_entry & entry,
                          uint64_t & value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
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
      if (bytes.size() != 2u) {
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

template <class fixture_type>
bool decode_float_value(const fixture_type & fixture,
                        const emel::gguf::loader::kv_entry & entry,
                        float & value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  if (entry.value_type == constants::gguf_type_float32) {
    if (bytes.size() != sizeof(float)) {
      return false;
    }
    std::memcpy(&value_out, bytes.data(), sizeof(float));
    return true;
  }

  if (entry.value_type == constants::gguf_type_float64) {
    if (bytes.size() != sizeof(double)) {
      return false;
    }
    double value = 0.0;
    std::memcpy(&value, bytes.data(), sizeof(double));
    value_out = static_cast<float>(value);
    return true;
  }

  return false;
}

template <class fixture_type>
bool decode_string_value(const fixture_type & fixture,
                         const emel::gguf::loader::kv_entry & entry,
                         std::string_view & value_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
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

template <class fixture_type>
bool decode_string_array_count(const fixture_type & fixture,
                               const emel::gguf::loader::kv_entry & entry,
                               uint32_t & count_out) {
  const std::span<const uint8_t> bytes = kv_value_view(fixture, entry);
  namespace constants = emel::gguf::loader::detail::constants;

  if (entry.value_type != constants::gguf_type_array ||
      bytes.size() < sizeof(uint32_t) + sizeof(uint64_t)) {
    return false;
  }

  const uint32_t element_type = read_u32_le(bytes.first(sizeof(uint32_t)));
  if (element_type != constants::gguf_type_string) {
    return false;
  }

  const uint64_t count = read_u64_le(bytes.subspan(sizeof(uint32_t), sizeof(uint64_t)));
  if (count > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }

  count_out = static_cast<uint32_t>(count);
  return true;
}

bool copy_tensor_names(const std::span<const uint8_t> file_image, emel::model::data & model_data) {
  model_data.name_bytes_used = 0u;

  for (uint32_t index = 0u; index < model_data.n_tensors; ++index) {
    auto & tensor = model_data.tensors[index];
    const size_t name_offset = static_cast<size_t>(tensor.name_offset);
    const size_t name_length = static_cast<size_t>(tensor.name_length);
    if (name_offset + name_length > file_image.size() ||
        model_data.name_bytes_used + name_length > model_data.name_storage.size()) {
      return false;
    }

    const uint32_t copied_offset = model_data.name_bytes_used;
    if (name_length > 0u) {
      std::memcpy(model_data.name_storage.data() + copied_offset,
                  file_image.data() + name_offset,
                  name_length);
    }

    model_data.name_bytes_used += static_cast<uint32_t>(name_length);
    tensor.name_offset = copied_offset;
  }
  return true;
}

emel::error::type populate_model_metadata(const emel_fixture & fixture,
                                          emel::model::data & model_data) {
  const auto * architecture_entry = find_kv_entry(fixture, "general.architecture");
  if (architecture_entry == nullptr) {
#ifndef NDEBUG
    std::fprintf(stderr, "populate_model_metadata: missing general.architecture\n");
#endif
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  std::string_view architecture = {};
  if (!decode_string_value(fixture, *architecture_entry, architecture) ||
      architecture != "qwen3" ||
      architecture.size() >= model_data.architecture_name.size()) {
#ifndef NDEBUG
    std::fprintf(stderr,
                 "populate_model_metadata: unexpected architecture '%.*s'\n",
                 static_cast<int>(architecture.size()),
                 architecture.data());
#endif
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  copy_name(model_data.architecture_name, architecture);

  const auto assign_i32 = [&](const std::string_view key, int32_t & field) {
    const auto * entry = find_kv_entry(fixture, key);
    if (entry == nullptr) {
      return false;
    }

    uint64_t value = 0u;
    if (!decode_integer_value(fixture, *entry, value) ||
        value > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
      return false;
    }

    field = static_cast<int32_t>(value);
    return true;
  };

  const auto assign_f32 = [&](const std::string_view key, float & field) {
    const auto * entry = find_kv_entry(fixture, key);
    if (entry == nullptr) {
      return false;
    }

    float value = 0.0f;
    if (!decode_float_value(fixture, *entry, value)) {
      return false;
    }

    field = value;
    return true;
  };

  int32_t qwen3_key_length = 0;
  int32_t qwen3_value_length = 0;
  if (!assign_i32("qwen3.context_length", model_data.params.n_ctx) ||
      !assign_i32("qwen3.embedding_length", model_data.params.n_embd) ||
      !assign_i32("qwen3.feed_forward_length", model_data.params.n_ff) ||
      !assign_i32("qwen3.attention.head_count", model_data.params.n_head) ||
      !assign_i32("qwen3.attention.head_count_kv", model_data.params.n_head_kv) ||
      !assign_i32("qwen3.attention.key_length", qwen3_key_length) ||
      !assign_i32("qwen3.attention.value_length", qwen3_value_length) ||
      !assign_i32("qwen3.block_count", model_data.params.n_layer) ||
      !assign_f32("qwen3.attention.layer_norm_rms_epsilon",
                  model_data.params.attention_layer_norm_rms_epsilon) ||
      !assign_f32("qwen3.rope.freq_base", model_data.params.rope_freq_base)) {
#ifndef NDEBUG
    std::fprintf(stderr, "populate_model_metadata: missing required qwen3 metadata\n");
#endif
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  model_data.params.attention_key_length = qwen3_key_length;
  model_data.params.attention_value_length = qwen3_value_length;
  if (model_data.params.n_embd_out == 0) {
    model_data.params.n_embd_out = model_data.params.n_embd;
  }
  if (model_data.params.n_rot == 0) {
    model_data.params.n_rot = qwen3_key_length;
  }
  if (qwen3_key_length <= 0 || qwen3_value_length <= 0) {
#ifndef NDEBUG
    std::fprintf(stderr,
                 "populate_model_metadata: invalid qwen3 key/value lengths (%d, %d)\n",
                 qwen3_key_length,
                 qwen3_value_length);
#endif
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const auto * tokens_entry = find_kv_entry(fixture, "tokenizer.tokens");
  if (tokens_entry != nullptr) {
    uint32_t token_count = 0u;
    if (!decode_string_array_count(fixture, *tokens_entry, token_count) ||
        token_count > static_cast<uint32_t>(emel::model::data::k_max_vocab_tokens)) {
#ifndef NDEBUG
      std::fprintf(stderr, "populate_model_metadata: invalid tokenizer.tokens array\n");
#endif
      return emel::error::cast(emel::model::loader::error::model_invalid);
    }
    model_data.vocab_data.n_tokens = token_count;
    model_data.params.n_vocab = static_cast<int32_t>(token_count);
  }
  return emel::error::cast(emel::model::loader::error::none);
}

formatter_binding resolve_formatter_binding(const emel_fixture & fixture) {
  const auto * entry = find_kv_entry(fixture, "tokenizer.chat_template");
  if (entry == nullptr) {
    return {};
  }

  std::string_view primary_template = {};
  if (!decode_string_value(fixture, *entry, primary_template)) {
    return {};
  }

  uint32_t named_template_count = 0u;
  for (const auto & candidate : fixture.kv_entries) {
    const std::string_view key = kv_key_view(fixture, candidate);
    if (key.starts_with("tokenizer.chat_template.") && key != "tokenizer.chat_template") {
      named_template_count += 1u;
    }
  }

  if (named_template_count != 0u || !template_matches_supported_qwen_contract(primary_template)) {
    return {};
  }

  return formatter_binding{
      .formatter_ctx = &k_supported_qwen_formatter_sentinel,
      .format_prompt = format_supported_qwen_contract,
      .supported = true,
  };
}

emel::error::type run_emel_parse_model(void * owner,
                                       const emel::model::loader::event::load & req) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  if (req.file_image == nullptr || req.file_size == 0u) {
    return emel::error::cast(emel::model::loader::error::invalid_request);
  }

  const std::span<const uint8_t> file_image{
      static_cast<const uint8_t *>(req.file_image),
      static_cast<size_t>(req.file_size),
  };

  reset_gguf_capture(fixture);
  emel::gguf::loader::requirements requirements = {};
  const emel::gguf::loader::event::probe_done_fn probe_done_cb{&fixture, on_probe_done};
  const emel::gguf::loader::event::probe_error_fn probe_error_cb{&fixture, on_probe_error};
  const emel::gguf::loader::event::probe probe_ev{
      file_image,
      requirements,
      probe_done_cb,
      probe_error_cb,
  };
  if (!fixture.gguf_loader.process_event(probe_ev) || !fixture.gguf.probe_done ||
      fixture.gguf.probe_error) {
    return map_gguf_error(fixture.gguf.err);
  }

  if (requirements.tensor_count > static_cast<uint32_t>(emel::model::data::k_max_tensors)) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  const uint64_t arena_bytes =
      emel::gguf::loader::detail::required_kv_arena_bytes(requirements);
  if (arena_bytes == std::numeric_limits<uint64_t>::max()) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }

  fixture.kv_arena.resize(static_cast<size_t>(arena_bytes));
  fixture.kv_entries.resize(requirements.kv_count);

  reset_gguf_capture(fixture);
  const emel::gguf::loader::event::bind_done_fn bind_done_cb{&fixture, on_bind_done};
  const emel::gguf::loader::event::bind_error_fn bind_error_cb{&fixture, on_bind_error};
  const emel::gguf::loader::event::bind_storage bind_ev{
      std::span<uint8_t>{fixture.kv_arena},
      std::span<emel::gguf::loader::kv_entry>{fixture.kv_entries},
      std::span<emel::model::data::tensor_record>{req.model_data.tensors.data(),
                                                  requirements.tensor_count},
      bind_done_cb,
      bind_error_cb,
  };
  if (!fixture.gguf_loader.process_event(bind_ev) || !fixture.gguf.bind_done ||
      fixture.gguf.bind_error) {
    return map_gguf_error(fixture.gguf.err);
  }

  reset_gguf_capture(fixture);
  const emel::gguf::loader::event::parse_done_fn parse_done_cb{&fixture, on_parse_done};
  const emel::gguf::loader::event::parse_error_fn parse_error_cb{&fixture, on_parse_error};
  const emel::gguf::loader::event::parse parse_ev{
      file_image,
      parse_done_cb,
      parse_error_cb,
  };
  if (!fixture.gguf_loader.process_event(parse_ev) || !fixture.gguf.parse_done ||
      fixture.gguf.parse_error) {
    return map_gguf_error(fixture.gguf.err);
  }

  req.model_data.n_tensors = requirements.tensor_count;
  if (!copy_tensor_names(file_image, req.model_data)) {
    return emel::error::cast(emel::model::loader::error::backend_error);
  }
  return populate_model_metadata(fixture, req.model_data);
}

emel::error::type run_emel_load_weights(void * owner,
                                        const emel::model::loader::event::load & req,
                                        uint64_t & bytes_total,
                                        uint64_t & bytes_done,
                                        bool & used_mmap) {
  auto & fixture = *static_cast<emel_fixture *>(owner);
  if (req.model_data.n_tensors == 0u) {
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }

  fixture.effect_requests.resize(req.model_data.n_tensors);
  fixture.effect_results.resize(req.model_data.n_tensors);

  reset_weight_capture(fixture);
  emel::model::weight_loader::event::bind_storage bind_ev{
      std::span<emel::model::data::tensor_record>{req.model_data.tensors.data(),
                                                  req.model_data.n_tensors},
  };
  bind_ev.on_done = {&fixture, on_weight_bind_done};
  bind_ev.on_error = {&fixture, on_weight_bind_error};
  if (!fixture.weight_loader.process_event(bind_ev) || !fixture.weight.bind_done ||
      fixture.weight.bind_error) {
    return map_weight_loader_error(fixture.weight.err);
  }

  reset_weight_capture(fixture);
  emel::model::weight_loader::event::plan_load plan_ev{
      std::span<emel::model::weight_loader::effect_request>{fixture.effect_requests},
  };
  plan_ev.on_done = {&fixture, on_weight_plan_done};
  plan_ev.on_error = {&fixture, on_weight_plan_error};
  if (!fixture.weight_loader.process_event(plan_ev) || !fixture.weight.plan_done ||
      fixture.weight.plan_error) {
    return map_weight_loader_error(fixture.weight.err);
  }

  const uint32_t effect_count = fixture.weight.effect_count;
  for (uint32_t index = 0u; index < effect_count; ++index) {
    fixture.effect_results[index] = emel::model::weight_loader::effect_result{
        .kind = fixture.effect_requests[index].kind,
        .handle = fixture.effect_requests[index].target,
        .err = emel::error::cast(emel::model::weight_loader::error::none),
    };
  }

  reset_weight_capture(fixture);
  emel::model::weight_loader::event::apply_effect_results apply_ev{
      std::span<const emel::model::weight_loader::effect_result>{fixture.effect_results.data(),
                                                                 effect_count},
  };
  apply_ev.on_done = {&fixture, on_weight_apply_done};
  apply_ev.on_error = {&fixture, on_weight_apply_error};
  if (!fixture.weight_loader.process_event(apply_ev) || !fixture.weight.apply_done ||
      fixture.weight.apply_error) {
    return map_weight_loader_error(fixture.weight.err);
  }

  req.model_data.weights_data = req.file_image;
  req.model_data.weights_size = req.file_size;
  req.model_data.weights_mapped = false;
  req.model_data.weights_split_count = 1u;
  req.model_data.weights_split_offsets[0] = 0u;
  req.model_data.weights_split_sizes[0] = req.file_size;
  bytes_total = req.file_size;
  bytes_done = req.file_size;
  used_mmap = false;
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type run_emel_map_layers(void *, const emel::model::loader::event::load & req) {
  int32_t max_block_index = -1;
  for (uint32_t index = 0u; index < req.model_data.n_tensors; ++index) {
    int32_t block_index = -1;
    if (emel::model::try_parse_block_index(
            emel::model::tensor_name_view(req.model_data, req.model_data.tensors[index]),
            block_index) &&
        block_index > max_block_index) {
      max_block_index = block_index;
    }
  }

  if (max_block_index >= 0) {
    req.model_data.n_layers = max_block_index + 1;
    return emel::error::cast(emel::model::loader::error::none);
  }
  if (req.model_data.params.n_layer > 0) {
    req.model_data.n_layers = req.model_data.params.n_layer;
    return emel::error::cast(emel::model::loader::error::none);
  }
#ifndef NDEBUG
  std::fprintf(stderr, "run_emel_map_layers: unable to derive layer count\n");
#endif
  return emel::error::cast(emel::model::loader::error::model_invalid);
}

emel::error::type run_emel_validate_structure(void *,
                                              const emel::model::loader::event::load & req) {
  if (req.model_data.n_tensors == 0u || req.model_data.n_layers <= 0 ||
      req.model_data.weights_data == nullptr || req.model_data.weights_size == 0u) {
#ifndef NDEBUG
    std::fprintf(stderr,
                 "run_emel_validate_structure: tensors=%u layers=%d weights=%p size=%llu\n",
                 req.model_data.n_tensors,
                 req.model_data.n_layers,
                 req.model_data.weights_data,
                 static_cast<unsigned long long>(req.model_data.weights_size));
#endif
    return emel::error::cast(emel::model::loader::error::model_invalid);
  }
  return emel::error::cast(emel::model::loader::error::none);
}

emel::error::type run_emel_validate_architecture(void *,
                                                 const emel::model::loader::event::load & req) {
  const emel::error::type err = emel::model::validate_execution_contract(req.model_data);
#ifndef NDEBUG
  if (err != emel::error::cast(emel::model::loader::error::none)) {
    std::fprintf(stderr, "run_emel_validate_architecture: err=%d\n", err);
  }
#endif
  return err;
}

bool prepare_emel_fixture(emel_fixture & fixture, const std::string & model_path) {
  if (!read_file_bytes(model_path, fixture.file_bytes)) {
#ifndef NDEBUG
    std::fprintf(stderr, "prepare_emel_fixture: read_file_bytes failed\n");
#endif
    return false;
  }

  reset_load_capture(fixture);
  emel::model::loader::event::parse_model_fn parse_model{&fixture, run_emel_parse_model};
  emel::model::loader::event::load load_ev{*fixture.model_data, parse_model};
  load_ev.model_path = model_path;
  load_ev.file_image = fixture.file_bytes.data();
  load_ev.file_size = fixture.file_bytes.size();
  load_ev.load_weights = {&fixture, run_emel_load_weights};
  load_ev.map_layers = {nullptr, run_emel_map_layers};
  load_ev.validate_structure = {nullptr, run_emel_validate_structure};
  load_ev.validate_architecture_impl = {nullptr, run_emel_validate_architecture};
  load_ev.on_done = {&fixture, on_load_done};
  load_ev.on_error = {&fixture, on_load_error};
  if (!fixture.model_loader.process_event(load_ev) || !fixture.load.done || fixture.load.error) {
    fixture.formatter = resolve_formatter_binding(fixture);
#ifndef NDEBUG
    std::fprintf(stderr,
                 "prepare_emel_fixture: model_loader failed done=%d error=%d err=%d\n",
                 fixture.load.done ? 1 : 0,
                 fixture.load.error ? 1 : 0,
                 fixture.load.err);
#endif
    return false;
  }

  fixture.formatter = resolve_formatter_binding(fixture);
#ifndef NDEBUG
  if (!fixture.formatter.supported) {
    std::fprintf(stderr, "prepare_emel_fixture: formatter unsupported\n");
  }
#endif
  return fixture.formatter.supported;
}

bool initialize_emel_session(emel_session & session) {
  if (session.generator == nullptr) {
    return false;
  }

  std::string formatted_prompt = {};
  if (!format_single_user_prompt(session.formatter, k_prompt, formatted_prompt)) {
    return false;
  }

  const int32_t prompt_capacity =
      std::max<int32_t>(32, static_cast<int32_t>(formatted_prompt.size()) + 8);
  const int32_t decode_capacity = std::max<int32_t>(4, k_max_tokens);
  const int32_t block_capacity = std::max<int32_t>(8, prompt_capacity + decode_capacity);

  reset_initialize_capture(session);
  emel::error::type error_out = emel::error::cast(emel::generator::error::none);
  emel::generator::event::initialize request{
      &session.tokenizer,
      tokenizer_bind_dispatch,
      tokenizer_tokenize_dispatch,
      std::span<emel::logits::sampler::fn>{},
  };
  request.preprocessor_variant = generation_preprocessor_variant(*session.model_data);
  request.encoder_variant = generation_encoder_variant(*session.model_data);
  request.add_special = false;
  request.parse_special = false;
  request.selection_mode = emel::generator::selection_mode::preselected_argmax;
  request.max_prompt_tokens = prompt_capacity;
  request.max_generated_tokens = decode_capacity;
  request.max_blocks = block_capacity;
  request.block_tokens = 16;
  request.strip_leading_space = false;
  request.error_out = &error_out;
  request.on_done = {&session, on_initialize_done};
  request.on_error = {&session, on_initialize_error};

  const bool accepted = session.generator->process_event(request);
  return accepted && session.initialize.done && !session.initialize.error &&
         error_out == emel::error::cast(emel::generator::error::none);
}

bool run_emel_generate(emel_session & session, generation_result & result_out) {
  if (session.generator == nullptr) {
    return false;
  }

  result_out = {};
  reset_generation_capture(session);
  emel::error::type error_out = emel::error::cast(emel::generator::error::none);
  std::array<emel::text::formatter::chat_message, 1> message_storage = {};
  emel::generator::event::generate request{
      single_user_messages(message_storage, k_prompt),
      k_max_tokens,
      std::span<char>{result_out.output},
      result_out.output_length,
  };
  request.add_generation_prompt = true;
  request.enable_thinking = false;
  request.error_out = &error_out;
  request.on_done = {&session, on_generation_done};
  request.on_error = {&session, on_generation_error};
  const bool accepted = session.generator->process_event(request);
  if (!accepted || !session.generation.done || session.generation.error ||
      error_out != emel::error::cast(emel::generator::error::none)) {
    return false;
  }

  result_out.tokens_generated = session.generation.tokens_generated;
  result_out.output_length = session.generation.output_length;
  return result_out.tokens_generated > 0;
}

}  // namespace

int main(int argc, char ** argv) {
  if (argc != 2) {
    return 0;
  }

#ifndef NDEBUG
  const auto fail_step = [](const char * step) {
    std::fprintf(stderr, "emel_probe failed at %s\n", step);
    return 1;
  };
#endif

  auto fixture = std::make_unique<emel_fixture>();
  if (!prepare_emel_fixture(*fixture, argv[1])) {
#ifndef NDEBUG
    return fail_step("prepare_emel_fixture");
#else
    return 1;
#endif
  }

  if (!emel::model::detail::load_vocab_from_gguf(
          emel::model::detail::kv_binding{
              .arena = std::span<const uint8_t>{fixture->kv_arena.data(), fixture->kv_arena.size()},
              .entries = std::span<const emel::gguf::loader::kv_entry>{fixture->kv_entries.data(),
                                                                       fixture->kv_entries.size()},
          },
          fixture->model_data->vocab_data)) {
#ifndef NDEBUG
    return fail_step("load_vocab_from_gguf");
#else
    return 1;
#endif
  }
  fixture->model_data->params.n_vocab =
      static_cast<int32_t>(fixture->model_data->vocab_data.n_tokens);

  auto session = std::make_unique<emel_session>();
  session->model_data = std::move(fixture->model_data);
  session->formatter = fixture->formatter;
  session->generator = std::make_unique<emel::generator::sm>(*session->model_data,
                                                             session->conditioner,
                                                             session->formatter.formatter_ctx,
                                                             session->formatter.format_prompt);

  if (!initialize_emel_session(*session)) {
#ifndef NDEBUG
    return fail_step("initialize_emel_session");
#else
    return 1;
#endif
  }

  generation_result result{};
  if (!run_emel_generate(*session, result)) {
#ifndef NDEBUG
    return fail_step("run_emel_generate");
#else
    return 1;
#endif
  }

  volatile size_t sink = result.output_length + static_cast<size_t>(result.tokens_generated);
  return sink == 0u ? 1 : 0;
}
