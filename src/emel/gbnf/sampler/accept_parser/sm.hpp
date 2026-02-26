#pragma once
// benchmark: scaffold

#include "emel/gbnf/sampler/accept_parser/actions.hpp"
#include "emel/gbnf/sampler/accept_parser/guards.hpp"
#include "emel/sm.hpp"

namespace emel::gbnf::sampler::accept_parser {

struct deciding {};
struct parsed {};
struct parse_failed {};
struct unexpected_event {};

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    // clang-format off
    return sml::make_transition_table(
      //------------------------------------------------------------------------------//
        sml::state<parsed> <= *sml::state<deciding> + sml::completion<sampler::event::accept_runtime>
                 [ guard::valid_accept_token{} ]
                 / action::consume_accepted

      , sml::state<parse_failed> <= sml::state<deciding> + sml::completion<sampler::event::accept_runtime>
                 [ guard::parse_failed{} ]
                 / action::dispatch_parse_failed

      //------------------------------------------------------------------------------//
      , sml::X <= sml::state<parsed>
      , sml::X <= sml::state<parse_failed>

      //------------------------------------------------------------------------------//
      , sml::state<unexpected_event> <= sml::state<deciding> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<parsed> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<parse_failed> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
      , sml::state<unexpected_event> <= sml::state<unexpected_event> + sml::unexpected_event<sml::_>
                 / action::on_unexpected
    );
    // clang-format on
  }
};

struct sm : emel::sm<model> {
  using model_type = model;
};

}  // namespace emel::gbnf::sampler::accept_parser
