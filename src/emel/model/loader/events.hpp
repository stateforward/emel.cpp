#pragma once

#include <string_view>

#include "emel/model/data.hpp"

namespace emel::model::loader::event {

struct load {
  emel::model::data & model_data;
  std::string_view model_path = {};
  bool request_mmap = true;
  bool request_direct_io = false;
};
struct mapping_parser_done {};
struct unsupported_format_error {};
struct layers_mapped {};
struct structure_validated {};
struct architecture_validated {};

}  // namespace emel::model::loader::event
