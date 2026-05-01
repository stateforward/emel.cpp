#include "parity_engine.hpp"
#include "parity_engines.hpp"

namespace emel::paritychecker {

namespace {

constexpr engine_adapter k_tokenizer_engine{
    .mode = parity_mode::tokenizer,
    .name = "tokenizer",
    .run = engines::run_tokenizer,
};

constexpr engine_adapter k_gbnf_parser_engine{
    .mode = parity_mode::gbnf_parser,
    .name = "gbnf_parser",
    .run = engines::run_gbnf_parser,
};

constexpr engine_adapter k_kernel_engine{
    .mode = parity_mode::kernel,
    .name = "kernel",
    .run = engines::run_kernel,
};

constexpr engine_adapter k_jinja_engine{
    .mode = parity_mode::jinja,
    .name = "jinja",
    .run = engines::run_jinja,
};

constexpr engine_adapter k_generation_engine{
    .mode = parity_mode::generation,
    .name = "generation",
    .run = engines::run_generation,
};

}  // namespace

const engine_adapter * find_engine(const parity_mode mode) {
  switch (mode) {
    case parity_mode::tokenizer:
      return &k_tokenizer_engine;
    case parity_mode::gbnf_parser:
      return &k_gbnf_parser_engine;
    case parity_mode::kernel:
      return &k_kernel_engine;
    case parity_mode::jinja:
      return &k_jinja_engine;
    case parity_mode::generation:
      return &k_generation_engine;
    default:
      return nullptr;
  }
}

}  // namespace emel::paritychecker
