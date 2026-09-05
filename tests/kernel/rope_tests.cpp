#include <array>
#include <doctest/doctest.h>

#include "emel/kernel/rope/sm.hpp"

namespace {

using emel::kernel::rope::event::dispatch_result;

} // namespace

TEST_CASE("rope precompute matches the reference frequency tables") {
  // head_dim=4, theta=100: freqs = [1, 1/10]; row for position 2.
  std::array<float, 6> cos_table{};
  std::array<float, 6> sin_table{};
  const emel::kernel::rope::event::precompute_request request{
      .theta = 100.0f,
      .head_dim = 4u,
      .positions = 3u,
      .cos_out = cos_table,
      .sin_out = sin_table};
  emel::kernel::rope::sm machine;
  dispatch_result result{};
  REQUIRE(machine.process_event(
      emel::kernel::rope::event::execute_precompute{request, result}));
  CHECK(cos_table[0] == doctest::Approx(1.0f));
  CHECK(sin_table[0] == doctest::Approx(0.0f));
  CHECK(cos_table[4] == doctest::Approx(-0.41614681482315063f));
  CHECK(cos_table[5] == doctest::Approx(0.9800665974617004f));
  CHECK(sin_table[4] == doctest::Approx(0.9092974066734314f));
  CHECK(sin_table[5] == doctest::Approx(0.19866932928562164f));
}

TEST_CASE(
    "rope apply rotates interleaved halves per the reference apply_rope") {
  std::array<float, 6> cos_table{};
  std::array<float, 6> sin_table{};
  const emel::kernel::rope::event::precompute_request precompute{
      .theta = 100.0f,
      .head_dim = 4u,
      .positions = 3u,
      .cos_out = cos_table,
      .sin_out = sin_table};
  emel::kernel::rope::sm machine;
  dispatch_result precompute_result{};
  REQUIRE(machine.process_event(emel::kernel::rope::event::execute_precompute{
      precompute, precompute_result}));

  std::array<float, 4> rows{1.0f, 2.0f, 3.0f, 4.0f};
  const emel::kernel::rope::event::apply_rows_request apply{
      .cos_table = cos_table,
      .sin_table = sin_table,
      .position = 2u,
      .head_count = 1u,
      .head_dim = 4u,
      .rows = rows};
  dispatch_result apply_result{};
  REQUIRE(machine.process_event(
      emel::kernel::rope::event::execute_apply_rows{apply, apply_result}));
  CHECK(rows[0] == doctest::Approx(-3.1440389156341553f));
  CHECK(rows[1] == doctest::Approx(1.1654558181762695f));
  CHECK(rows[2] == doctest::Approx(-0.3391430974006653f));
  CHECK(rows[3] == doctest::Approx(4.317605018615723f));
}

TEST_CASE("rope guard rejects short tables") {
  std::array<float, 2> cos_table{};
  std::array<float, 2> sin_table{};
  std::array<float, 4> rows{};
  const emel::kernel::rope::event::apply_rows_request apply{
      .cos_table = cos_table,
      .sin_table = sin_table,
      .position = 2u,
      .head_count = 1u,
      .head_dim = 4u,
      .rows = rows};
  emel::kernel::rope::sm machine;
  dispatch_result result{};
  CHECK_FALSE(machine.process_event(
      emel::kernel::rope::event::execute_apply_rows{apply, result}));
  CHECK(machine.is(stateforward::sml::state<emel::kernel::rope::state_ready>));
}
