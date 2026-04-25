#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "emel/text/encoders/context.hpp"
#include "emel/text/encoders/errors.hpp"
#include "emel/text/encoders/events.hpp"
#include "emel/text/encoders/types.hpp"

namespace emel::text::encoders::plamo2::action {

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

struct context : emel::text::encoders::action::context {
  bool plamo2_tables_ready = false;
  const emel::model::data::vocab *plamo2_vocab = nullptr;
  std::array<int32_t, 256> byte_tokens = {};
  std::unordered_map<int64_t, int32_t> suffix_map = {};
  std::vector<table_row> table = {};
  std::array<int64_t, emel::text::encoders::detail::k_max_encode_bytes + 1> scores = {};
  std::array<path_entry, emel::text::encoders::detail::k_max_encode_bytes + 1> paths = {};
  std::array<uint32_t, emel::text::encoders::detail::k_max_encode_bytes> cpts = {};
};

}  // namespace emel::text::encoders::plamo2::action

namespace emel::text::encoders::plamo2::runtime {

struct encode_runtime {
  const emel::text::encoders::event::encode_runtime & event_;
  mutable int32_t data_len = 0;
  mutable int32_t emit_result_error =
    emel::text::encoders::error::to_emel(emel::text::encoders::error::code::ok);
  mutable int32_t emit_result_token_count = 0;
};

}  // namespace emel::text::encoders::plamo2::runtime
