#pragma once

#include <cstdint>
#include <string_view>

#include "emel/callback.hpp"
#include "emel/error/error.hpp"
#include "emel/gbnf/detail.hpp"
#include "emel/gbnf/sampler/accept_parser/events.hpp"
#include "emel/gbnf/sampler/candidate_parser/events.hpp"
#include "emel/gbnf/sampler/errors.hpp"
#include "emel/gbnf/sampler/matcher_parser/events.hpp"
#include "emel/gbnf/sampler/token_parser/events.hpp"

namespace emel::gbnf::sampler::events {

struct apply_done;
struct apply_error;
struct accept_done;
struct accept_error;

}  // namespace emel::gbnf::sampler::events

namespace emel::gbnf::sampler::event {

struct apply {
  const emel::gbnf::grammar & grammar;
  std::string_view text = {};
  uint32_t start_rule_id = 0;
  const ::emel::callback<bool(const ::emel::gbnf::sampler::events::apply_done &)> & on_done;
  const ::emel::callback<bool(const ::emel::gbnf::sampler::events::apply_error &)> & on_error;
};

struct accept {
  const emel::gbnf::grammar & grammar;
  uint32_t token_id = 0;
  const ::emel::callback<bool(const ::emel::gbnf::sampler::events::accept_done &)> & on_done;
  const ::emel::callback<bool(const ::emel::gbnf::sampler::events::accept_error &)> & on_error;
};

struct apply_ctx {
  bool candidate_allowed = false;
  emel::gbnf::sampler::candidate_parser::events::candidate_kind candidate_kind =
      emel::gbnf::sampler::candidate_parser::events::candidate_kind::unknown;
  emel::gbnf::sampler::token_parser::events::token_kind token_kind =
      emel::gbnf::sampler::token_parser::events::token_kind::unknown;
  emel::gbnf::sampler::matcher_parser::events::match_result match_result =
      emel::gbnf::sampler::matcher_parser::events::match_result::unknown;
  emel::error::type err = emel::error::cast(error::none);
};

struct accept_ctx {
  bool accepted = false;
  emel::gbnf::sampler::accept_parser::events::accept_result accept_result =
      emel::gbnf::sampler::accept_parser::events::accept_result::unknown;
  emel::error::type err = emel::error::cast(error::none);
};

struct apply_runtime {
  const apply & request;
  apply_ctx & ctx;
};

struct accept_runtime {
  const accept & request;
  accept_ctx & ctx;
};

}  // namespace emel::gbnf::sampler::event

namespace emel::gbnf::sampler::events {

struct apply_done {
  const emel::gbnf::grammar & grammar;
  bool allowed = false;
};

struct apply_error {
  const emel::gbnf::grammar & grammar;
  int32_t err = 0;
};

struct accept_done {
  const emel::gbnf::grammar & grammar;
  bool accepted = false;
};

struct accept_error {
  const emel::gbnf::grammar & grammar;
  int32_t err = 0;
};

}  // namespace emel::gbnf::sampler::events
