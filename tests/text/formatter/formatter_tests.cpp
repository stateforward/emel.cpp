#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/text/formatter/format.hpp"

namespace {

bool store_metadata_string(emel::model::data::metadata & metadata,
                           const std::string_view value,
                           emel::model::data::metadata::string_view & out) {
  out = {};
  if (value.empty()) {
    return true;
  }

  if (metadata.blob_bytes_used + value.size() > metadata.blob.size()) {
    return false;
  }

  out.offset = metadata.blob_bytes_used;
  out.length = static_cast<uint32_t>(value.size());
  std::memcpy(metadata.blob.data() + metadata.blob_bytes_used, value.data(), value.size());
  metadata.blob_bytes_used += static_cast<uint32_t>(value.size());
  return true;
}

}  // namespace

TEST_CASE("formatter_format_raw_handles_invalid_and_empty_inputs") {
  int32_t err = 0;
  size_t out_len = 7;
  const std::array messages = {
      emel::text::formatter::chat_message{.role = "user", .content = "x"},
  };

  emel::text::formatter::format_request bad_req = {};
  bad_req.messages = messages;
  bad_req.output = nullptr;
  bad_req.output_capacity = 1;
  bad_req.output_length_out = &out_len;
  CHECK_FALSE(emel::text::formatter::format_raw(nullptr, bad_req, &err));
  CHECK(err == (1 << 0));
  CHECK(out_len == 0);

  err = 0;
  out_len = 9;
  emel::text::formatter::format_request empty_req = {};
  empty_req.messages = {};
  empty_req.output = nullptr;
  empty_req.output_capacity = 0;
  empty_req.output_length_out = &out_len;
  CHECK(emel::text::formatter::format_raw(nullptr, empty_req, &err));
  CHECK(err == 0);
  CHECK(out_len == 0);
}

TEST_CASE("formatter_contract_carries_structured_messages_and_options") {
  int32_t err = 0;
  size_t out_len = 0;
  std::array<char, 32> output = {};
  const std::array messages = {
      emel::text::formatter::chat_message{.role = "system", .content = "alpha"},
      emel::text::formatter::chat_message{.role = "user", .content = "beta"},
  };

  emel::text::formatter::format_request request = {};
  request.messages = messages;
  request.add_generation_prompt = true;
  request.enable_thinking = false;
  request.output = output.data();
  request.output_capacity = output.size();
  request.output_length_out = &out_len;

  CHECK(emel::text::formatter::format_raw(nullptr, request, &err));
  CHECK(err == 0);
  CHECK(out_len == std::string_view{"alphabeta"}.size());
  CHECK(std::string_view{output.data(), out_len} == "alphabeta");
  CHECK(request.messages[0].role == "system");
  CHECK(request.messages[1].content == "beta");
  CHECK(request.add_generation_prompt);
  CHECK_FALSE(request.enable_thinking);
}

TEST_CASE("formatter_supported_qwen_contract_formats_chatml_and_rejects_unsupported_shapes") {
  int32_t err = 0;
  size_t out_len = 0;
  std::array<char, 256> output = {};
  const std::array messages = {
      emel::text::formatter::chat_message{.role = "system", .content = "policy"},
      emel::text::formatter::chat_message{.role = "user", .content = "hello"},
      emel::text::formatter::chat_message{.role = "assistant", .content = "world"},
  };

  emel::text::formatter::format_request request = {};
  request.messages = messages;
  request.add_generation_prompt = true;
  request.enable_thinking = false;
  request.output = output.data();
  request.output_capacity = output.size();
  request.output_length_out = &out_len;

  CHECK(emel::text::formatter::format_supported_qwen_contract(nullptr, request, &err));
  CHECK(err == 0);
  CHECK(std::string_view{output.data(), out_len} ==
        "<|im_start|>system\npolicy<|im_end|>\n"
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\nworld<|im_end|>\n"
        "<|im_start|>assistant\n");

  const std::array unsupported_messages = {
      emel::text::formatter::chat_message{.role = "tool", .content = "call"},
  };
  request.messages = unsupported_messages;
  out_len = 0;
  CHECK_FALSE(emel::text::formatter::format_supported_qwen_contract(nullptr, request, &err));
  CHECK(err == emel::text::formatter::error_code(
                   emel::text::formatter::error::invalid_request));

  request.messages = messages;
  request.add_generation_prompt = false;
  CHECK_FALSE(emel::text::formatter::request_matches_contract(
      emel::text::formatter::contract_kind::supported_qwen_contract,
      request.messages,
      request.add_generation_prompt,
      request.enable_thinking));
}

TEST_CASE("formatter_supported_qwen_tool_contract_formats_standard_assistant_generation_prefix") {
  int32_t err = 0;
  size_t out_len = 0;
  std::array<char, 256> output = {};
  const std::array messages = {
      emel::text::formatter::chat_message{.role = "system", .content = "policy"},
      emel::text::formatter::chat_message{.role = "user", .content = "hello"},
  };

  emel::text::formatter::format_request request = {};
  request.messages = messages;
  request.add_generation_prompt = true;
  request.enable_thinking = false;
  request.output = output.data();
  request.output_capacity = output.size();
  request.output_length_out = &out_len;

  CHECK(emel::text::formatter::format_supported_qwen_tool_contract(
      nullptr, request, &err));
  CHECK(err == 0);
  CHECK(std::string_view{output.data(), out_len} ==
        "<|im_start|>system\npolicy<|im_end|>\n"
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n");
}

TEST_CASE("formatter_supported_contract_formats_chatml_with_bos_and_rejects_invalid_shapes") {
  int32_t err = 0;
  size_t out_len = 17;
  std::array<char, 256> output = {};
  const std::array messages = {
      emel::text::formatter::chat_message{.role = "system", .content = "policy"},
      emel::text::formatter::chat_message{.role = "user", .content = "hello"},
  };

  emel::text::formatter::format_request request = {};
  request.messages = messages;
  request.add_generation_prompt = true;
  request.enable_thinking = false;
  request.output = output.data();
  request.output_capacity = output.size();
  request.output_length_out = &out_len;

  CHECK(emel::text::formatter::format_supported_contract(nullptr, request, &err));
  CHECK(err == 0);
  CHECK(std::string_view{output.data(), out_len} ==
        "<|startoftext|>"
        "<|im_start|>system\npolicy<|im_end|>\n"
        "<|im_start|>user\nhello<|im_end|>\n"
        "<|im_start|>assistant\n");

  request.output = nullptr;
  request.output_capacity = 1;
  CHECK_FALSE(emel::text::formatter::format_supported_contract(nullptr, request, &err));
  CHECK(err == emel::text::formatter::error_code(
                   emel::text::formatter::error::invalid_request));

  request.output = output.data();
  request.output_capacity = output.size();
  request.messages = {};
  CHECK_FALSE(emel::text::formatter::format_supported_contract(nullptr, request, &err));

  const std::array invalid_messages = {
      emel::text::formatter::chat_message{.role = "user", .content = "hello"},
      emel::text::formatter::chat_message{.role = "system", .content = "late"},
  };
  request.messages = invalid_messages;
  CHECK_FALSE(emel::text::formatter::format_supported_contract(nullptr, request, &err));

  request.messages = messages;
  request.add_generation_prompt = false;
  CHECK_FALSE(emel::text::formatter::format_supported_contract(nullptr, request, &err));

  request.add_generation_prompt = true;
  request.enable_thinking = true;
  CHECK_FALSE(emel::text::formatter::format_supported_contract(nullptr, request, &err));

  request.enable_thinking = false;
  request.output_capacity = 4;
  CHECK_FALSE(emel::text::formatter::format_supported_contract(nullptr, request, &err));
}

TEST_CASE("formatter_request_contract_helpers_cover_supported_and_unsupported_cases") {
  const std::array system_user_messages = {
      emel::text::formatter::chat_message{.role = "system", .content = "policy"},
      emel::text::formatter::chat_message{.role = "user", .content = "hello"},
  };
  const std::array qwen_messages = {
      emel::text::formatter::chat_message{.role = "assistant", .content = "reply"},
  };
  const std::array invalid_messages = {
      emel::text::formatter::chat_message{.role = "tool", .content = "call"},
  };

  CHECK(emel::text::formatter::role_supported("system"));
  CHECK(emel::text::formatter::role_supported("user"));
  CHECK_FALSE(emel::text::formatter::role_supported("assistant"));
  CHECK(emel::text::formatter::qwen_role_supported("assistant"));
  CHECK_FALSE(emel::text::formatter::qwen_role_supported("tool"));

  CHECK(emel::text::formatter::messages_supported(system_user_messages));
  CHECK_FALSE(emel::text::formatter::messages_supported(qwen_messages));
  CHECK_FALSE(emel::text::formatter::messages_supported(invalid_messages));
  CHECK_FALSE(emel::text::formatter::messages_supported(
      std::span<const emel::text::formatter::chat_message>{}));

  CHECK(emel::text::formatter::qwen_messages_supported(system_user_messages));
  CHECK(emel::text::formatter::qwen_messages_supported(qwen_messages));
  CHECK_FALSE(emel::text::formatter::qwen_messages_supported(invalid_messages));
  CHECK_FALSE(emel::text::formatter::qwen_messages_supported(
      std::span<const emel::text::formatter::chat_message>{}));

  CHECK(emel::text::formatter::request_matches_contract(
      emel::text::formatter::contract_kind::raw,
      invalid_messages,
      false,
      true));
  CHECK(emel::text::formatter::request_matches_contract(
      emel::text::formatter::contract_kind::supported_contract,
      system_user_messages,
      true,
      false));
  CHECK_FALSE(emel::text::formatter::request_matches_contract(
      emel::text::formatter::contract_kind::supported_contract,
      qwen_messages,
      true,
      false));
  CHECK(emel::text::formatter::request_matches_contract(
      emel::text::formatter::contract_kind::supported_qwen_contract,
      qwen_messages,
      true,
      false));
  CHECK(emel::text::formatter::request_matches_contract(
      emel::text::formatter::contract_kind::supported_qwen_tool_contract,
      qwen_messages,
      true,
      false));
  CHECK_FALSE(emel::text::formatter::request_matches_contract(
      emel::text::formatter::contract_kind::supported_qwen_contract,
      qwen_messages,
      true,
      true));
  CHECK_FALSE(emel::text::formatter::request_matches_contract(
      emel::text::formatter::contract_kind::unsupported_template,
      system_user_messages,
      true,
      false));
}

TEST_CASE("formatter_template_matching_and_metadata_helpers_cover_edge_cases") {
  std::string supported_template = {};
  for (const auto marker : emel::text::formatter::k_supported_primary_template_markers) {
    supported_template.append(marker);
    supported_template.push_back('\n');
  }
  CHECK(emel::text::formatter::template_matches_supported_contract(supported_template));
  CHECK_FALSE(emel::text::formatter::template_matches_supported_contract("missing"));

  std::string supported_qwen_template = {};
  for (const auto marker : emel::text::formatter::k_supported_qwen_primary_template_markers) {
    supported_qwen_template.append(marker);
    supported_qwen_template.push_back('\n');
  }
  CHECK(emel::text::formatter::template_matches_supported_qwen_contract(
      supported_qwen_template));
  CHECK_FALSE(emel::text::formatter::template_matches_supported_qwen_contract("missing"));

  std::string supported_qwen_tool_template = {};
  for (const auto marker :
       emel::text::formatter::k_supported_qwen_tool_markers) {
    supported_qwen_tool_template.append(marker);
    supported_qwen_tool_template.push_back('\n');
  }
  CHECK(emel::text::formatter::template_matches_supported_qwen_tool_contract(
      supported_qwen_tool_template));
  CHECK_FALSE(emel::text::formatter::template_matches_supported_qwen_tool_contract(
      "missing"));

  emel::model::data::metadata metadata = {};
  emel::model::data::metadata::string_view stored = {};
  REQUIRE(store_metadata_string(metadata, "hello", stored));
  CHECK(emel::text::formatter::metadata_string_view(metadata, stored) == "hello");

  emel::model::data::metadata::string_view empty = {};
  CHECK(emel::text::formatter::metadata_string_view(metadata, empty).empty());

  emel::model::data::metadata::string_view invalid = {
      .offset = static_cast<uint32_t>(metadata.blob.size() - 1u),
      .length = 4u,
  };
  CHECK(emel::text::formatter::metadata_string_view(metadata, invalid).empty());

  size_t offset = 0;
  std::array<char, 8> buffer = {};
  CHECK(emel::text::formatter::append_bytes("ab", buffer.data(), buffer.size(), offset));
  CHECK(offset == 2u);
  CHECK_FALSE(emel::text::formatter::append_bytes(
      "toolong", buffer.data(), 4u, offset));
}

TEST_CASE("formatter_resolve_model_binding_uses_embedded_qwen_primary_template_even_with_named_variants") {
  auto model = std::make_unique<emel::model::data>();
  constexpr std::string_view k_supported_qwen_template =
      "{% if messages[0].role == 'system' %}\n"
      "<|im_start|>system\n"
      "{% endif %}\n"
      "{% if message.role == \"user\" %}\n"
      "{% endif %}\n"
      "{% if message.role == \"assistant\" %}\n"
      "{% endif %}\n"
      "{% if add_generation_prompt %}\n"
      "{% if enable_thinking %}\n"
      "<tool_call>\n"
      "tool_response\n"
      "<|im_end|>\n"
      "<|im_start|>\n";

  REQUIRE(store_metadata_string(
      model->meta, k_supported_qwen_template, model->meta.tokenizer_data.chat_template));
  auto binding = emel::text::formatter::resolve_model_binding(
      *model, nullptr, emel::text::formatter::format_raw);
  CHECK(binding.contract == emel::text::formatter::contract_kind::supported_qwen_contract);
  CHECK(binding.contract_text == emel::text::formatter::k_supported_qwen_contract);

  model->meta.tokenizer_data.chat_template_count = 1u;
  binding = emel::text::formatter::resolve_model_binding(
      *model, nullptr, emel::text::formatter::format_raw);
  CHECK(binding.contract == emel::text::formatter::contract_kind::supported_qwen_contract);
  CHECK(binding.contract_text == emel::text::formatter::k_supported_qwen_contract);

  model = std::make_unique<emel::model::data>();
  constexpr std::string_view k_supported_qwen_tool_template =
      "{%- if tools %}\n"
      "{% if messages[0].role == 'system' %}\n"
      "<|im_start|>system\n"
      "{% endif %}\n"
      "ns.multi_step_tool\n"
      "{% if (message.role == \"user\") %}\n"
      "{% endif %}\n"
      "{%- elif message.role == \"assistant\" %}\n"
      "{% endif %}\n"
      "message.tool_calls\n"
      "message.role == \"tool\"\n"
      "<tool_call>\n"
      "tool_response\n"
      "<|im_end|>\n"
      "<|im_start|>\n"
      "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n";
  REQUIRE(store_metadata_string(
      model->meta,
      k_supported_qwen_tool_template,
      model->meta.tokenizer_data.chat_template));
  binding = emel::text::formatter::resolve_model_binding(
      *model, nullptr, emel::text::formatter::format_raw);
  CHECK(binding.contract ==
        emel::text::formatter::contract_kind::supported_qwen_tool_contract);
  CHECK(binding.contract_text ==
        emel::text::formatter::k_supported_qwen_tool_contract);

  model = std::make_unique<emel::model::data>();
  binding = emel::text::formatter::resolve_model_binding(
      *model, nullptr, emel::text::formatter::format_raw);
  CHECK(binding.contract == emel::text::formatter::contract_kind::raw);
}

TEST_CASE("formatter_resolve_model_binding_covers_supported_raw_and_fallback_variants") {
  auto model = std::make_unique<emel::model::data>();
  constexpr std::string_view k_supported_template =
      "{{- bos_token -}}\n"
      "keep_past_thinking\n"
      "messages[0][\"role\"] == \"system\"\n"
      "\"List of tools: [\"\n"
      "message[\"role\"] == \"assistant\"\n"
      "</think>\n"
      "add_generation_prompt\n"
      "<|im_start|>assistant\\n\n"
      "<|im_start|>system\\n\n";
  REQUIRE(store_metadata_string(
      model->meta, k_supported_template, model->meta.tokenizer_data.chat_template));

  auto binding = emel::text::formatter::resolve_model_binding(
      *model,
      &emel::text::formatter::k_supported_formatter_sentinel,
      emel::text::formatter::format_raw);
  CHECK(binding.contract == emel::text::formatter::contract_kind::supported_contract);
  CHECK(binding.contract_text == emel::text::formatter::k_supported_contract);

  model = std::make_unique<emel::model::data>();
  REQUIRE(store_metadata_string(
      model->meta, "unknown template", model->meta.tokenizer_data.chat_template));
  binding = emel::text::formatter::resolve_model_binding(
      *model,
      &emel::text::formatter::k_supported_formatter_sentinel,
      emel::text::formatter::format_raw);
  CHECK(binding.contract == emel::text::formatter::contract_kind::unsupported_template);
  CHECK(binding.format_prompt == emel::text::formatter::format_raw);

  model = std::make_unique<emel::model::data>();
  binding = emel::text::formatter::resolve_model_binding(
      *model,
      &emel::text::formatter::k_supported_formatter_sentinel,
      emel::text::formatter::format_raw);
  CHECK(binding.contract == emel::text::formatter::contract_kind::raw);
  CHECK(binding.formatter_ctx == &emel::text::formatter::k_supported_formatter_sentinel);
  CHECK(binding.format_prompt == emel::text::formatter::format_raw);
}
