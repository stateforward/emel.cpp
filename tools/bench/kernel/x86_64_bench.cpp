#include "bench_cases.hpp"

#include "emel/kernel/x86_64/sm.hpp"

#include "kernel/bench_common.hpp"

namespace emel::bench {

void append_emel_kernel_x86_64_cases(std::vector<result> & results, const config & cfg) {
  emel::kernel::x86_64::sm x86_machine{};
  auto exec = [&](const auto & ev) {
    return x86_machine.process_event(ev);
  };
  append_emel_backend_cases(results, cfg, "x86_64", exec);
}

void append_reference_kernel_x86_64_cases(std::vector<result> & results, const config & cfg) {
  append_reference_backend_cases(results, cfg, "x86_64");
}

}  // namespace emel::bench
