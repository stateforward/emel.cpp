#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

#include "emel/model/data.hpp"

namespace emel::text::formatter {

enum class error : int32_t {
  none = 0u,
  invalid_request = (1u << 0),
};

constexpr int32_t error_code(const error value) noexcept {
  return static_cast<int32_t>(value);
}

struct chat_message {
  std::string_view role = {};
  std::string_view content = {};
};

struct format_request {
  std::span<const chat_message> messages = {};
  bool add_generation_prompt = false;
  bool enable_thinking = false;
  char * output = nullptr;
  size_t output_capacity = 0;
  size_t * output_length_out = nullptr;
};

using format_fn = bool (*)(void * formatter_ctx,
                           const format_request & request,
                           int32_t * error_out);

enum class contract_kind : uint8_t {
  raw = 0u,
  supported_contract = 1u,
  supported_qwen_contract = 2u,
  supported_qwen_tool_contract = 3u,
  unsupported_template = 4u,
};

struct binding {
  void * formatter_ctx = nullptr;
  format_fn format_prompt = nullptr;
  contract_kind contract = contract_kind::raw;
  std::string_view contract_text = {};
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
inline constexpr std::string_view k_supported_qwen_tool_contract =
    "source=tokenizer.chat_template support=supported_contract "
    "shape=structured_chat_messages_v1 roles=system,user,assistant tools=tool_xml "
    "add_generation_prompt=true enable_thinking=false bos=none";
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
inline constexpr std::array<std::string_view, 12> k_supported_qwen_tool_markers = {
    "{%- if tools %}",
    "<|im_start|>",
    "<|im_end|>",
    "messages[0].role == 'system'",
    "ns.multi_step_tool",
    "(message.role == \"user\")",
    "{%- elif message.role == \"assistant\" %}",
    "message.tool_calls",
    "message.role == \"tool\"",
    "<tool_call>",
    "tool_response",
    "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n",
};

inline constexpr std::string_view k_bos = "<|startoftext|>";
inline constexpr std::string_view k_im_start = "<|im_start|>";
inline constexpr std::string_view k_im_end = "<|im_end|>\n";
inline constexpr std::string_view k_assistant_generation_prefix =
    "<|im_start|>assistant\n";
inline constexpr std::string_view k_message_separator = "\n";
inline int k_supported_formatter_sentinel = 0;
inline int k_supported_qwen_formatter_sentinel = 0;

inline bool format_raw(void *,
                       const format_request & request,
                       int32_t * error_out) noexcept {
  if (error_out != nullptr) {
    *error_out = error_code(error::none);
  }
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0;
  }
  if (request.output == nullptr && request.output_capacity > 0) {
    if (error_out != nullptr) {
      *error_out = error_code(error::invalid_request);
    }
    return false;
  }

  size_t output_length = 0;
  for (const auto & message : request.messages) {
    output_length += message.content.size();
  }

  if (output_length > request.output_capacity) {
    if (error_out != nullptr) {
      *error_out = error_code(error::invalid_request);
    }
    return false;
  }

  size_t write_offset = 0;
  for (const auto & message : request.messages) {
    if (!message.content.empty()) {
      std::memcpy(request.output + write_offset,
                  message.content.data(),
                  message.content.size());
    }
    write_offset += message.content.size();
  }

  if (request.output_length_out != nullptr) {
    *request.output_length_out = output_length;
  }
  return true;
}

inline bool role_supported(const std::string_view role) noexcept {
  return role == "system" || role == "user";
}

inline bool qwen_role_supported(const std::string_view role) noexcept {
  return role == "system" || role == "user" || role == "assistant";
}

inline bool messages_supported(
    const std::span<const chat_message> messages) noexcept {
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

inline bool qwen_messages_supported(
    const std::span<const chat_message> messages) noexcept {
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

inline bool format_supported_contract(void *,
                                      const format_request & request,
                                      int32_t * error_out) noexcept {
  if (error_out != nullptr) {
    *error_out = error_code(error::none);
  }
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0u;
  }
  if ((request.output == nullptr && request.output_capacity > 0u) ||
      !messages_supported(request.messages) ||
      !request.add_generation_prompt ||
      request.enable_thinking) {
    if (error_out != nullptr) {
      *error_out = error_code(error::invalid_request);
    }
    return false;
  }

  size_t output_length = 0u;
  if (!append_bytes(k_bos, request.output, request.output_capacity, output_length)) {
    if (error_out != nullptr) {
      *error_out = error_code(error::invalid_request);
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
        *error_out = error_code(error::invalid_request);
      }
      return false;
    }
  }

  if (!append_bytes(k_assistant_generation_prefix,
                    request.output,
                    request.output_capacity,
                    output_length)) {
    if (error_out != nullptr) {
      *error_out = error_code(error::invalid_request);
    }
    return false;
  }

  if (request.output_length_out != nullptr) {
    *request.output_length_out = output_length;
  }
  return true;
}

inline bool format_supported_qwen_contract(void *,
                                           const format_request & request,
                                           int32_t * error_out) noexcept;
inline bool format_supported_qwen_tool_contract(
    void *,
    const format_request & request,
    int32_t * error_out) noexcept;

inline bool format_supported_qwen_contract_with_generation_prefix(
    const format_request & request,
    const std::string_view generation_prefix,
    int32_t * error_out) noexcept {
  if (error_out != nullptr) {
    *error_out = error_code(error::none);
  }
  if (request.output_length_out != nullptr) {
    *request.output_length_out = 0u;
  }
  if ((request.output == nullptr && request.output_capacity > 0u) ||
      !qwen_messages_supported(request.messages) ||
      !request.add_generation_prompt ||
      request.enable_thinking) {
    if (error_out != nullptr) {
      *error_out = error_code(error::invalid_request);
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
        *error_out = error_code(error::invalid_request);
      }
      return false;
    }
  }

  if (!append_bytes(generation_prefix,
                    request.output,
                    request.output_capacity,
                    output_length)) {
    if (error_out != nullptr) {
      *error_out = error_code(error::invalid_request);
    }
    return false;
  }

  if (request.output_length_out != nullptr) {
    *request.output_length_out = output_length;
  }
  return true;
}

inline bool format_supported_qwen_contract(void *,
                                           const format_request & request,
                                           int32_t * error_out) noexcept {
  return format_supported_qwen_contract_with_generation_prefix(
      request, k_assistant_generation_prefix, error_out);
}

inline bool format_supported_qwen_tool_contract(
    void *,
    const format_request & request,
    int32_t * error_out) noexcept {
  return format_supported_qwen_contract_with_generation_prefix(
      request, k_assistant_generation_prefix, error_out);
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

inline bool template_matches_supported_qwen_tool_contract(
    const std::string_view primary_template) noexcept {
  for (const std::string_view marker : k_supported_qwen_tool_markers) {
    if (primary_template.find(marker) == std::string_view::npos) {
      return false;
    }
  }
  return true;
}

inline std::string_view metadata_string_view(
    const emel::model::data::metadata & metadata,
    const emel::model::data::metadata::string_view & view) noexcept {
  if (view.length == 0u) {
    return {};
  }
  if (static_cast<size_t>(view.offset) + static_cast<size_t>(view.length) > metadata.blob.size()) {
    return {};
  }
  return std::string_view{metadata.blob.data() + view.offset, view.length};
}

inline bool request_matches_contract(
    const contract_kind contract,
    const std::span<const chat_message> messages,
    const bool add_generation_prompt,
    const bool enable_thinking) noexcept {
  switch (contract) {
    case contract_kind::raw:
      return true;
    case contract_kind::supported_contract:
      return messages_supported(messages) && add_generation_prompt && !enable_thinking;
    case contract_kind::supported_qwen_contract:
    case contract_kind::supported_qwen_tool_contract:
      return qwen_messages_supported(messages) && add_generation_prompt && !enable_thinking;
    case contract_kind::unsupported_template:
      return false;
  }

  return false;
}

inline binding resolve_model_binding(const emel::model::data & model,
                                     void * fallback_ctx,
                                     format_fn fallback_prompt) noexcept {
  const std::string_view primary_template =
      metadata_string_view(model.meta, model.meta.tokenizer_data.chat_template);
  const uint32_t named_template_count = model.meta.tokenizer_data.chat_template_count;
  static_cast<void>(named_template_count);

  if (primary_template.empty()) {
    return binding{
        .formatter_ctx = fallback_ctx,
        .format_prompt = fallback_prompt,
        .contract = contract_kind::raw,
        .contract_text = {},
    };
  }

  if (template_matches_supported_contract(primary_template)) {
    return binding{
        .formatter_ctx = &k_supported_formatter_sentinel,
        .format_prompt = format_supported_contract,
        .contract = contract_kind::supported_contract,
        .contract_text = k_supported_contract,
    };
  }

  if (template_matches_supported_qwen_tool_contract(primary_template)) {
    return binding{
        .formatter_ctx = &k_supported_qwen_formatter_sentinel,
        .format_prompt = format_supported_qwen_tool_contract,
        .contract = contract_kind::supported_qwen_tool_contract,
        .contract_text = k_supported_qwen_tool_contract,
    };
  }

  if (template_matches_supported_qwen_contract(primary_template)) {
    return binding{
        .formatter_ctx = &k_supported_qwen_formatter_sentinel,
        .format_prompt = format_supported_qwen_contract,
        .contract = contract_kind::supported_qwen_contract,
        .contract_text = k_supported_qwen_contract,
    };
  }

  return binding{
      .formatter_ctx = nullptr,
      .format_prompt = fallback_prompt,
      .contract = contract_kind::unsupported_template,
      .contract_text = k_unsupported_template_contract,
  };
}

}  // namespace emel::text::formatter
