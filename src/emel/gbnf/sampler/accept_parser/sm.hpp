#pragma once
// benchmark: scaffold

#include "emel/gbnf/sampler/accept_parser/actions.hpp"
#include "emel/gbnf/sampler/accept_parser/guards.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::sampler::accept_parser {

struct deciding {};
struct parsed {};
struct parse_failed {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<parsed> <= *sml::state<deciding> + sml::completion<sampler::event::sample_runtime>
                 [ guard::token_accepted_by_grammar{} ]
                 / action::consume_accepted

      , sml::state<parsed> <= sml::state<deciding> + sml::completion<sampler::event::sample_runtime>
                 [ guard::token_rejected_by_grammar{} ]
                 / action::consume_rejected

      , sml::state<parse_failed> <= sml::state<deciding> + sml::completion<sampler::event::sample_runtime>
                 [ guard::parse_failed{} ]
                 / action::dispatch_parse_failed

      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<parsed>
      , sml::X <= sml::state<parse_failed>

      //------------------------------------------------------------------------------//
      , sml::state<parse_failed> <= sml::state<deciding> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<parse_failed> <= sml::state<parsed> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<parse_failed> <= sml::state<parse_failed> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : emel::sm<model> {
  using model_type = model;
};

}  // namespace emel::gbnf::sampler::accept_parser
