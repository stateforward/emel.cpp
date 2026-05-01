#include "parity_runner.hpp"
#include "parity_engine.hpp"

#include <cstdio>

namespace emel::paritychecker {

int run_parity(const parity_options & opts) {
  const engine_adapter * engine = find_engine(opts.mode);
  if (engine == nullptr || engine->run == nullptr) {
    std::fprintf(stderr, "unsupported parity mode: %u\n", static_cast<unsigned>(opts.mode));
    return 1;
  }
  return engine->run(opts);
}

}  // namespace emel::paritychecker
