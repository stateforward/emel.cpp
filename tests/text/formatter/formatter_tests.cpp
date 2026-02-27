#include <cstddef>
#include <cstdint>

#include <doctest/doctest.h>

#include "emel/emel.h"
#include "emel/text/formatter/format.hpp"

TEST_CASE("formatter_format_raw_handles_invalid_and_empty_inputs") {
  int32_t err = EMEL_OK;
  size_t out_len = 7;

  emel::text::formatter::format_request bad_req = {};
  bad_req.input = "x";
  bad_req.output = nullptr;
  bad_req.output_capacity = 1;
  bad_req.output_length_out = &out_len;
  CHECK_FALSE(emel::text::formatter::format_raw(nullptr, bad_req, &err));
  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(out_len == 0);

  err = EMEL_OK;
  out_len = 9;
  emel::text::formatter::format_request empty_req = {};
  empty_req.input = "";
  empty_req.output = nullptr;
  empty_req.output_capacity = 0;
  empty_req.output_length_out = &out_len;
  CHECK(emel::text::formatter::format_raw(nullptr, empty_req, &err));
  CHECK(err == EMEL_OK);
  CHECK(out_len == 0);
}
