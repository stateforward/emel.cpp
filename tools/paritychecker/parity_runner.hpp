#pragma once

#include <cstdint>
#include <string>

namespace emel::paritychecker {

enum class parity_mode : uint8_t {
  tokenizer = 0,
  gbnf_parser = 1,
  kernel = 2,
  jinja = 3,
  generation = 4,
};

struct parity_options {
  parity_mode mode = parity_mode::tokenizer;
  std::string model_path;
  std::string text;
  std::string write_generation_baseline_path;
  int32_t max_tokens = 32;
  bool add_special = false;
  bool parse_special = false;
  bool dump = false;
  bool attribution = false;
};

int run_parity(const parity_options & opts);

}  // namespace emel::paritychecker
