#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "emel/encoder/context.hpp"
#include "emel/encoder/types.hpp"

namespace emel::encoder::plamo2::action {

struct table_row {
  int32_t piece_length = 0;
  int32_t token_id = 0;
  int32_t score = 0;
  int32_t piece_id = 0;
};

struct path_entry {
  int32_t token_length = 0;
  int32_t token_id = 0;
  int32_t num_tokens = 0;
};

struct context : emel::encoder::action::context {
  bool plamo2_tables_ready = false;
  const emel::model::data::vocab *plamo2_vocab = nullptr;
  std::array<int32_t, 256> byte_tokens = {};
  std::unordered_map<int64_t, int32_t> suffix_map = {};
  std::vector<table_row> table = {};
  std::array<int64_t, emel::encoder::detail::k_max_encode_bytes + 1> scores = {};
  std::array<path_entry, emel::encoder::detail::k_max_encode_bytes + 1> paths = {};
  std::array<uint32_t, emel::encoder::detail::k_max_encode_bytes> cpts = {};
};

}  // namespace emel::encoder::plamo2::action
