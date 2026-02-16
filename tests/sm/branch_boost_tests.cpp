#include <cstdint>

#include <doctest/doctest.h>

#include "emel/test_support/branch_boost.hpp"

TEST_CASE("sm_branch_boost_covers_all_cases") {
  int64_t sum = 0;
  for (uint16_t value = 0; value < 1024; ++value) {
    sum += emel::test_support::branch_boost(value);
  }

  CHECK(sum > 0);
  CHECK(emel::test_support::branch_boost(1024) == -1);
}
