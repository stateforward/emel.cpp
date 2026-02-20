#include "bench/bench_registry.hpp"

namespace emel::bench {
namespace {
case_entry k_cases[k_max_cases] = {};
std::size_t k_count = 0;
}

bool register_case(case_entry entry) {
  if (entry.name == nullptr || entry.run == nullptr) {
    return false;
  }
  if (k_count >= k_max_cases) {
    return false;
  }
  k_cases[k_count++] = entry;
  return true;
}

const case_entry * cases() {
  return k_cases;
}

std::size_t case_count() {
  return k_count;
}

} // namespace emel::bench
