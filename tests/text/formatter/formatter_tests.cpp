#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/text/formatter/format.hpp"

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
