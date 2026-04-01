#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>

#include "llama.h"

#include "emel/text/formatter/format.hpp"

namespace emel::tools::generation_formatter_contract {

enum class support_kind : uint8_t {
  no_template,
  supported_contract,
  unsupported_template,
};

inline constexpr std::string_view k_supported_contract =
    "source=tokenizer.chat_template support=supported_contract "
    "shape=structured_chat_messages_v1 roles=system,user tools=none "
    "add_generation_prompt=true enable_thinking=false keep_past_thinking=false "
    "bos=<|startoftext|>";
inline constexpr std::string_view k_supported_qwen_contract =
    "source=tokenizer.chat_template support=supported_contract "
    "shape=structured_chat_messages_v1 roles=system,user,assistant tools=none "
    "add_generation_prompt=true enable_thinking=false bos=none";

inline constexpr std::string_view k_no_template_contract =
    "source=tokenizer.chat_template support=no_template "
    "shape=structured_chat_messages_v1 roles=system,user tools=none "
    "add_generation_prompt=true enable_thinking=false keep_past_thinking=false "
    "bos=<|startoftext|>";

inline constexpr std::string_view k_unsupported_template_contract =
    "source=tokenizer.chat_template support=unsupported_template "
    "shape=structured_chat_messages_v1 roles=system,user tools=none "
    "add_generation_prompt=true enable_thinking=false keep_past_thinking=false "
    "bos=<|startoftext|>";

inline constexpr std::array<std::string_view, 9> k_supported_primary_template_markers = {
    "{{- bos_token -}}",
    "keep_past_thinking",
    "messages[0][\"role\"] == \"system\"",
    "\"List of tools: [\"",
    "message[\"role\"] == \"assistant\"",
    "</think>",
    "add_generation_prompt",
    "<|im_start|>assistant\\n",
    "<|im_start|>system\\n",
};

inline constexpr std::array<std::string_view, 9> k_supported_qwen_primary_template_markers = {
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

struct formatter_binding {
  void * formatter_ctx = nullptr;
  emel::text::formatter::format_fn format_prompt =
      emel::text::formatter::format_raw;
  support_kind support = support_kind::no_template;
  std::string_view contract = k_no_template_contract;
};

struct reference_formatter_info {
  std::string primary_template = {};
  uint32_t named_template_count = 0u;
  support_kind support = support_kind::no_template;
  std::string_view contract = k_no_template_contract;
};

inline constexpr std::string_view k_bos = "<|startoftext|>";
inline constexpr std::string_view k_im_start = "<|im_start|>";
inline constexpr std::string_view k_im_end = "<|im_end|>\n";
inline constexpr std::string_view k_assistant_generation_prefix =
    "<|im_start|>assistant\n";
inline constexpr std::string_view k_message_separator = "\n";
inline int k_supported_formatter_sentinel = 0;
inline int k_supported_qwen_formatter_sentinel = 0;

inline bool qwen_role_supported(const std::string_view role) noexcept {
  return role == "system" || role == "user" || role == "assistant";
}

inline bool role_supported(const std::string_view role) noexcept {
  return role == "system" || role == "user";
}

inline bool qwen_messages_supported(
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

inline bool messages_supported(
    const std::span<const emel::text::formatter::chat_message> messages) noexcept {
  if (messages.empty()) {
    return false;
  }

  bool saw_user = false;
  bool allow_system = true;
  for (const auto & message : messages) {
    if (!role_supported(message.role)) {
      return false;
    }
    if (message.role == "system") {
      if (!allow_system) {
        return false;
      }
      continue;
    }
    saw_user = true;
    allow_system = false;
  }
  return saw_user;
}

inline bool append_bytes(const std::string_view bytes,
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

inline bool format_supported_qwen_contract(void *,
                                           const emel::text::formatter::format_request & request,
                                           int32_t * error_out) noexcept {
  if (error_out != nullptr) {
    *error_out = emel::text::formatter::error_code(
        emel::text::formatter::error::none);
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

inline bool format_supported_contract(void *,
                                      const emel::text::formatter::format_request & request,
                                      int32_t * error_out) noexcept {
  if (error_out != nullptr) {
    *error_out = emel::text::formatter::error_code(
        emel::text::formatter::error::none);
  }
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0u;
  }
  if ((request.output == nullptr && request.output_capacity > 0u) ||
      !messages_supported(request.messages) ||
      !request.add_generation_prompt ||
      request.enable_thinking) {
    if (error_out != nullptr) {
      *error_out = emel::text::formatter::error_code(
          emel::text::formatter::error::invalid_request);
    }
    return false;
  }

  size_t output_length = 0u;
  if (!append_bytes(k_bos, request.output, request.output_capacity, output_length)) {
    if (error_out != nullptr) {
      *error_out = emel::text::formatter::error_code(
          emel::text::formatter::error::invalid_request);
    }
    return false;
  }
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

inline bool template_matches_supported_contract(
    const std::string_view primary_template) noexcept {
  for (const std::string_view marker : k_supported_primary_template_markers) {
    if (primary_template.find(marker) == std::string_view::npos) {
      return false;
    }
  }
  return true;
}

inline bool template_matches_supported_qwen_contract(
    const std::string_view primary_template) noexcept {
  for (const std::string_view marker : k_supported_qwen_primary_template_markers) {
    if (primary_template.find(marker) == std::string_view::npos) {
      return false;
    }
  }
  return true;
}

inline formatter_binding resolve_primary_template_binding(
    const std::string_view primary_template,
    const uint32_t named_template_count) noexcept {
  if (primary_template.empty()) {
    return formatter_binding{};
  }
  if (named_template_count != 0u) {
    return formatter_binding{
        .formatter_ctx = nullptr,
        .format_prompt = emel::text::formatter::format_raw,
        .support = support_kind::unsupported_template,
        .contract = k_unsupported_template_contract,
    };
  }
  if (template_matches_supported_contract(primary_template)) {
    return formatter_binding{
        .formatter_ctx = &k_supported_formatter_sentinel,
        .format_prompt = format_supported_contract,
        .support = support_kind::supported_contract,
        .contract = k_supported_contract,
    };
  }
  if (template_matches_supported_qwen_contract(primary_template)) {
    return formatter_binding{
        .formatter_ctx = &k_supported_qwen_formatter_sentinel,
        .format_prompt = format_supported_qwen_contract,
        .support = support_kind::supported_contract,
        .contract = k_supported_qwen_contract,
    };
  }
  return formatter_binding{
      .formatter_ctx = nullptr,
      .format_prompt = emel::text::formatter::format_raw,
      .support = support_kind::unsupported_template,
      .contract = k_unsupported_template_contract,
  };
}

inline bool binding_supported(const formatter_binding & binding) noexcept {
  return binding.support == support_kind::supported_contract;
}

inline bool reference_binding_supported(
    const reference_formatter_info & formatter) noexcept {
  return formatter.support == support_kind::supported_contract;
}

inline uint32_t count_reference_named_chat_templates(
    const llama_model * model) noexcept {
  if (model == nullptr) {
    return 0u;
  }

  constexpr std::string_view k_primary_key = "tokenizer.chat_template";
  constexpr std::string_view k_named_prefix = "tokenizer.chat_template.";
  uint32_t named_template_count = 0u;
  const int32_t metadata_count = llama_model_meta_count(model);
  for (int32_t index = 0; index < metadata_count; ++index) {
    char key_storage[256] = {};
    const int32_t key_length =
        llama_model_meta_key_by_index(model, index, key_storage, sizeof(key_storage));
    if (key_length <= 0) {
      continue;
    }

    const std::string_view key{key_storage, static_cast<size_t>(key_length)};
    if (key.starts_with(k_named_prefix) && key != k_primary_key) {
      named_template_count += 1u;
    }
  }
  return named_template_count;
}

inline reference_formatter_info resolve_reference_formatter_info(
    const llama_model * model) noexcept {
  reference_formatter_info formatter = {};
  formatter.named_template_count = count_reference_named_chat_templates(model);

  const char * primary_template =
      model != nullptr ? llama_model_chat_template(model, nullptr) : nullptr;
  if (primary_template == nullptr || primary_template[0] == '\0') {
    return formatter;
  }

  formatter.primary_template = primary_template;
  if (formatter.named_template_count != 0u) {
    formatter.support = support_kind::unsupported_template;
    formatter.contract = k_unsupported_template_contract;
    return formatter;
  }

  if (template_matches_supported_contract(formatter.primary_template)) {
    formatter.support = support_kind::supported_contract;
    formatter.contract = k_supported_contract;
    return formatter;
  }

  if (template_matches_supported_qwen_contract(formatter.primary_template)) {
    formatter.support = support_kind::supported_contract;
    formatter.contract = k_supported_qwen_contract;
    return formatter;
  }

  formatter.support = support_kind::unsupported_template;
  formatter.contract = k_unsupported_template_contract;
  return formatter;
}

inline std::span<const emel::text::formatter::chat_message>
single_user_messages(std::array<emel::text::formatter::chat_message, 1> & storage,
                     const std::string_view text) noexcept {
  storage[0] = emel::text::formatter::chat_message{
      .role = "user",
      .content = text,
  };
  return std::span<const emel::text::formatter::chat_message>{storage};
}

inline size_t formatted_capacity_upper_bound(
    const formatter_binding & binding,
    const std::span<const emel::text::formatter::chat_message> messages,
    const bool add_generation_prompt) noexcept {
  size_t capacity = 0u;
  if (binding.contract == k_supported_contract || binding.contract == k_no_template_contract) {
    capacity += k_bos.size();
  }
  for (const auto & message : messages) {
    capacity += k_im_start.size() + message.role.size() + k_message_separator.size() +
        message.content.size() + k_im_end.size();
  }
  if (add_generation_prompt) {
    capacity += k_assistant_generation_prefix.size();
  }
  return capacity;
}

inline bool format_single_user_prompt(const formatter_binding & binding,
                                      const std::string_view text,
                                      std::string & output) {
  if (!binding_supported(binding)) {
    output.clear();
    return false;
  }

  std::array<emel::text::formatter::chat_message, 1> message_storage = {};
  const auto messages = single_user_messages(message_storage, text);
  output.resize(formatted_capacity_upper_bound(binding, messages, true));
  size_t output_length = 0u;
  int32_t error_out = emel::text::formatter::error_code(
      emel::text::formatter::error::none);
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

inline bool format_reference_single_user_prompt(
    const reference_formatter_info & formatter,
    const std::string_view text,
    std::string & output) {
  if (!reference_binding_supported(formatter) || formatter.primary_template.empty()) {
    output.clear();
    return false;
  }

  std::string prompt_text{text};
  const llama_chat_message message = {
      "user",
      prompt_text.c_str(),
  };

  int32_t output_capacity =
      std::max<int32_t>(64, static_cast<int32_t>(text.size()) * 4 + 64);
  for (;;) {
    output.resize(static_cast<size_t>(output_capacity));
    const int32_t output_length = llama_chat_apply_template(formatter.primary_template.c_str(),
                                                            &message,
                                                            1u,
                                                            true,
                                                            output.data(),
                                                            output_capacity);
    if (output_length < 0) {
      output.clear();
      return false;
    }
    if (output_length <= output_capacity) {
      output.resize(static_cast<size_t>(output_length));
      return true;
    }
    output_capacity = output_length;
  }
}

}  // namespace emel::tools::generation_formatter_contract
