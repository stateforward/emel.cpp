#include "doctest/doctest.h"

#include "emel/error/error.hpp"
#include "emel/logits/validator/errors.hpp"
#include "emel/logits/validator/sm.hpp"

namespace {

struct build_buffers {
  float logits[4] = {1.0F, 4.0F, 2.0F, -1.0F};
  int32_t ids[4] = {};
  float scores[4] = {};
  int32_t count = 0;
  emel::error::type err = emel::error::cast(emel::logits::validator::error::none);
};

}  // namespace

TEST_CASE("validator builds candidates and normalizes scores") {
  emel::logits::validator::sm machine{};
  build_buffers buffers{};

  emel::logits::validator::event::build request{
      buffers.logits[0],
      4,
      buffers.ids[0],
      buffers.scores[0],
      4,
      buffers.count,
      buffers.err};

  CHECK(machine.process_event(request));
  CHECK(buffers.err == emel::error::cast(emel::logits::validator::error::none));
  CHECK(buffers.count == 4);
  CHECK(buffers.ids[0] == 0);
  CHECK(buffers.ids[3] == 3);
  CHECK(buffers.scores[1] == doctest::Approx(0.0F));
  CHECK(buffers.scores[0] == doctest::Approx(-3.0F));
  CHECK(buffers.scores[2] == doctest::Approx(-2.0F));
  CHECK(buffers.scores[3] == doctest::Approx(-5.0F));
}

TEST_CASE("validator reports invalid request when capacity is too small") {
  emel::logits::validator::sm machine{};
  build_buffers buffers{};
  buffers.count = 99;
  buffers.err = emel::error::cast(emel::logits::validator::error::backend_error);

  emel::logits::validator::event::build request{
      buffers.logits[0],
      4,
      buffers.ids[0],
      buffers.scores[0],
      2,
      buffers.count,
      buffers.err};

  CHECK(!machine.process_event(request));
  CHECK(buffers.err == emel::error::cast(emel::logits::validator::error::invalid_request));
  CHECK(buffers.count == 0);
}

TEST_CASE("validator reports invalid request when vocab size is not positive") {
  emel::logits::validator::sm machine{};
  build_buffers buffers{};
  buffers.count = 13;
  buffers.err = emel::error::cast(emel::logits::validator::error::backend_error);

  emel::logits::validator::event::build request{
      buffers.logits[0],
      0,
      buffers.ids[0],
      buffers.scores[0],
      4,
      buffers.count,
      buffers.err};

  CHECK(!machine.process_event(request));
  CHECK(buffers.err == emel::error::cast(emel::logits::validator::error::invalid_request));
  CHECK(buffers.count == 0);
}
