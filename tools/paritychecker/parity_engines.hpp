#pragma once

#include "parity_runner.hpp"

namespace emel::paritychecker::engines {

int run_tokenizer(const parity_options & opts);
int run_gbnf_parser(const parity_options & opts);
int run_kernel(const parity_options & opts);
int run_jinja(const parity_options & opts);
int run_generation(const parity_options & opts);

}  // namespace emel::paritychecker::engines
