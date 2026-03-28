#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

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

}  // namespace emel::text::formatter
