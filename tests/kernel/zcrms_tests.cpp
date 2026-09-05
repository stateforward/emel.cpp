#include <array>
#include <doctest/doctest.h>

#include "emel/kernel/zcrms/sm.hpp"

namespace {

using emel::kernel::zcrms::event::dispatch_result;

} // namespace

TEST_CASE("zcrms norm rows match the reference (1+scale)*x/rms") {
  // scale = fp16([0.5, -0.25, 1.0]); expectations from the JAX `_zcrms`
  // formula evaluated in f64-backed numpy with fp16-decoded scales.
  const std::array<float, 6> input{1.0f, 2.0f, 3.0f, 0.5f, -1.0f, 4.0f};
  const std::array<float, 3> scale{0.5f, -0.25f, 1.0f};
  std::array<float, 6> output{};
  const emel::kernel::zcrms::event::norm_rows_request request{
      .input = input, .scale = scale, .rows = 2u, .dim = 3u, .output = output};
  emel::kernel::zcrms::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::zcrms::event::execute_norm_rows{request, result}));
  CHECK(output[0] == doctest::Approx(0.6943650245666504f));
  CHECK(output[1] == doctest::Approx(0.6943650245666504f));
  CHECK(output[2] == doctest::Approx(2.7774600982666016f));
  CHECK(output[3] == doctest::Approx(0.31277158856391907f));
  CHECK(output[4] == doctest::Approx(-0.31277158856391907f));
  CHECK(output[5] == doctest::Approx(3.3362302780151367f));
}

TEST_CASE("zcrms unit rows match the reference rms_unit") {
  const std::array<float, 6> input{1.0f, 2.0f, 3.0f, 0.5f, -1.0f, 4.0f};
  std::array<float, 6> output{};
  const emel::kernel::zcrms::event::unit_rows_request request{
      .input = input, .rows = 2u, .dim = 3u, .output = output};
  emel::kernel::zcrms::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::zcrms::event::execute_unit_rows{request, result}));
  CHECK(output[0] == doctest::Approx(0.462909996509552f));
  CHECK(output[1] == doctest::Approx(0.925819993019104f));
  CHECK(output[2] == doctest::Approx(1.3887300491333008f));
  CHECK(output[3] == doctest::Approx(0.20851439237594604f));
  CHECK(output[4] == doctest::Approx(-0.4170287847518921f));
  CHECK(output[5] == doctest::Approx(1.6681151390075684f));
}

TEST_CASE("zcrms guard rejects undersized spans") {
  const std::array<float, 2> input{1.0f, 2.0f};
  const std::array<float, 3> scale{};
  std::array<float, 6> output{};
  const emel::kernel::zcrms::event::norm_rows_request request{
      .input = input, .scale = scale, .rows = 2u, .dim = 3u, .output = output};
  emel::kernel::zcrms::sm machine;
  emel::kernel::zcrms::event::dispatch_result result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::zcrms::event::execute_norm_rows{request, result}));
  CHECK(machine.is(stateforward::sml::state<emel::kernel::zcrms::state_ready>));
}
