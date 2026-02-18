#include <array>
#include <boost/sml.hpp>
#include <doctest/doctest.h>

#include "emel/decoder/sm.hpp"
#include "emel/emel.h"

TEST_CASE("decoder_sm_rejects_invalid_decode_request") {
  emel::decoder::sm machine{};
  int32_t err = EMEL_OK;

  CHECK_FALSE(machine.process_event(emel::decoder::event::decode{
    .token_ids = nullptr,
    .n_tokens = 0,
    .n_ubatch = 1,
    .error_out = &err,
  }));

  CHECK(err == EMEL_ERR_INVALID_ARGUMENT);
  CHECK(machine.is(boost::sml::state<emel::decoder::initialized>));
  CHECK(machine.last_error() == EMEL_ERR_INVALID_ARGUMENT);
}

TEST_CASE("decoder_sm_accepts_valid_request_and_resets") {
  emel::decoder::sm machine{};
  std::array<int32_t, 1> tokens = {{1}};
  int32_t err = EMEL_OK;

  (void)machine.process_event(emel::decoder::event::decode{
    .token_ids = tokens.data(),
    .n_tokens = 1,
    .n_ubatch = 1,
    .error_out = &err,
  });

  CHECK(machine.is(boost::sml::state<emel::decoder::initialized>));
}
