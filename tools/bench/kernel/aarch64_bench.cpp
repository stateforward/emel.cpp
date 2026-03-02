#include "bench_cases.hpp"

#include "emel/kernel/aarch64/context.hpp"
#include "emel/kernel/aarch64/detail.hpp"

#include "kernel/bench_common.hpp"

namespace emel::bench {

void append_emel_kernel_aarch64_cases(std::vector<result> & results, const config & cfg) {
  const emel::kernel::aarch64::action::context aarch_ctx{};
  auto exec = [&](const auto & ev) {
    return emel::kernel::aarch64::detail::execute_request(ev, aarch_ctx);
  };
  append_emel_backend_cases(results, cfg, "aarch64", exec);
}

void append_reference_kernel_aarch64_cases(std::vector<result> & results, const config & cfg) {
  append_reference_backend_cases(results, cfg, "aarch64");
}

}  // namespace emel::bench
