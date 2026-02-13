#pragma once

#include <cstdint>
#include <string_view>

namespace emel::encoder::event {

enum class backend_type : uint8_t {
  merging = 0,
  searching = 1,
  scanning = 2,
};

struct encode {
  std::string_view text = {};
};

struct pretokenizing_done {};
struct pretokenizing_error {};

struct algorithm_selected {
  backend_type backend = backend_type::merging;
};

struct algorithm_step_done {};
struct algorithm_step_error {};

struct emission_done {};
struct emission_error {};

struct postrules_done {};
struct postrules_error {};

struct tokenized_done {};
struct tokenized_error {};

}  // namespace emel::encoder::event

namespace emel::encoder::events {

struct encoding_done {};
struct encoding_error {};

}  // namespace emel::encoder::events
