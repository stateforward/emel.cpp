#include "bench_cases.hpp"

#include "emel/kernel/aarch64/sm.hpp"

#include "kernel/bench_common.hpp"

namespace emel::bench {

void append_emel_kernel_aarch64_cases(std::vector<result> & results, const config & cfg) {
  emel::kernel::aarch64::sm aarch_machine{};
  auto exec = [&](const auto & ev) {
    return aarch_machine.process_event(ev);
  };
  append_emel_backend_cases(results, cfg, "aarch64", exec);
}

void append_reference_kernel_aarch64_cases(std::vector<result> & results, const config & cfg) {
  append_reference_backend_cases(results, cfg, "aarch64");
}

}  // namespace emel::bench
