#pragma once

#include <cstddef>
#include <cstdint>

namespace emel::bench {

struct case_entry {
  const char * name;
  void (*run)(std::uint64_t iterations);
};

constexpr std::size_t k_max_cases = 256;

bool register_case(case_entry entry);
const case_entry * cases();
std::size_t case_count();

} // namespace emel::bench
