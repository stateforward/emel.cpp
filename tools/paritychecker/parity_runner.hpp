#pragma once

#include <string>

namespace emel::paritychecker {

struct parity_options {
  std::string model_path;
  std::string text;
  bool add_special = false;
  bool parse_special = false;
  bool dump_tokens = false;
};

int run_parity(const parity_options & opts);

}  // namespace emel::paritychecker
