#pragma once

#include <cstdint>

#include "emel/gbnf/sampler/candidate_parser/events.hpp"
#include "emel/error/error.hpp"
#include "emel/gbnf/sampler/accept_parser/events.hpp"
#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/matcher_parser/events.hpp"
#include "emel/gbnf/sampler/token_parser/events.hpp"

namespace emel::gbnf::sampler::event {

struct sample {
  int32_t & candidate_ids;
  float & candidate_scores;
  int32_t & candidate_count;
  int32_t & selected_token_out;
  emel::error::type & error_out;
};

struct sample_ctx {
  emel::error::type err = emel::error::cast(error::none);
  int32_t read_index = 0;
  int32_t write_index = 0;
  int32_t current_token_id = -1;
  emel::gbnf::sampler::candidate_parser::events::candidate_kind candidate_kind =
      emel::gbnf::sampler::candidate_parser::events::candidate_kind::unknown;
  emel::gbnf::sampler::token_parser::events::token_kind token_kind =
      emel::gbnf::sampler::token_parser::events::token_kind::unknown;
  emel::gbnf::sampler::matcher_parser::events::match_result match_result =
      emel::gbnf::sampler::matcher_parser::events::match_result::unknown;
  bool candidate_allowed = false;
  emel::gbnf::sampler::accept_parser::events::accept_result accept_result =
      emel::gbnf::sampler::accept_parser::events::accept_result::unknown;
};

struct sample_runtime {
  const sample & request;
  sample_ctx & ctx;
};

}  // namespace emel::gbnf::sampler::event

namespace emel::gbnf::sampler::events {

struct sample_done {
  int32_t candidate_count = 0;
  int32_t selected_token = -1;
};

struct sample_error {
  emel::error::type err = emel::error::cast(error::none);
};

}  // namespace emel::gbnf::sampler::events
