#pragma once

#include <cstdint>
#include <string>

namespace emel::paritychecker {

enum class parity_mode : uint8_t {
  tokenizer = 0,
  gbnf_parser = 1,
  kernel = 2,
};

struct parity_options {
  parity_mode mode = parity_mode::tokenizer;
  std::string model_path;
  std::string text;
  bool add_special = false;
  bool parse_special = false;
  bool dump = false;
};

int run_parity(const parity_options & opts);

}  // namespace emel::paritychecker
