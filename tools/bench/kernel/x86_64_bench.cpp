#include "bench_cases.hpp"

#include "emel/kernel/x86_64/context.hpp"
#include "emel/kernel/x86_64/detail.hpp"

#include "kernel/bench_common.hpp"

namespace emel::bench {

void append_emel_kernel_x86_64_cases(std::vector<result> & results, const config & cfg) {
  const emel::kernel::x86_64::action::context x86_ctx{};
  auto exec = [&](const auto & ev) {
    return emel::kernel::x86_64::detail::execute_request(ev, x86_ctx);
  };
  append_emel_backend_cases(results, cfg, "x86_64", exec);
}

void append_reference_kernel_x86_64_cases(std::vector<result> & results, const config & cfg) {
  append_reference_backend_cases(results, cfg, "x86_64");
}

}  // namespace emel::bench
