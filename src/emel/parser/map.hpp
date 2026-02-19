#pragma once

#include <cstddef>
#include <cstdint>

#include "emel/model/loader/events.hpp"
#include "emel/parser/events.hpp"

namespace emel::parser {

enum class kind : uint8_t {
  gguf = 0,
  count = 1
};

using map_parser_fn =
  bool (*)(const emel::model::loader::event::load &,
           int32_t * err_out);
using can_handle_fn =
  bool (*)(const emel::model::loader::event::load &);
using dispatch_parse_fn =
  bool (*)(void * parser_sm, const emel::parser::event::parse_model &);

struct entry {
  kind kind_id = kind::gguf;
  void * parser_sm = nullptr;
  map_parser_fn map_parser = nullptr;
  can_handle_fn can_handle = nullptr;
  dispatch_parse_fn dispatch_parse = nullptr;
};

struct map {
  const entry * entries = nullptr;
  size_t count = 0;
};

struct selection {
  const entry * entry = nullptr;
  kind kind_id = kind::count;
};

inline bool kind_valid(const kind id) noexcept {
  return static_cast<size_t>(id) < static_cast<size_t>(kind::count);
}

inline selection select(const map * parser_map,
                        const emel::model::loader::event::load & ev) {
  selection result{};
  if (parser_map == nullptr || parser_map->entries == nullptr || parser_map->count == 0) {
    return result;
  }
  for (size_t i = 0; i < parser_map->count; ++i) {
    const entry & candidate = parser_map->entries[i];
    if (!kind_valid(candidate.kind_id)) {
      continue;
    }
    if (candidate.can_handle == nullptr) {
      continue;
    }
    if (!candidate.can_handle(ev)) {
      continue;
    }
    result.entry = &candidate;
    result.kind_id = candidate.kind_id;
    return result;
  }
  return result;
}

}  // namespace emel::parser
