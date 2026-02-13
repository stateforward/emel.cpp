#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/generator/actions.hpp"
#include "emel/generator/events.hpp"
#include "emel/generator/guards.hpp"
#include "emel/model/data.hpp"
#include "emel/tokenizer/sm.hpp"

namespace emel::generator {

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct initialized {};
    struct tokenizing_prompt {};
    struct prefilling {};
    struct decoding {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::generate> = sml::state<tokenizing_prompt>,

      sml::state<tokenizing_prompt> + sml::event<event::prompt_tokenized_done> = sml::state<prefilling>,
      sml::state<tokenizing_prompt> + sml::event<event::prompt_tokenized_error> = sml::state<errored>,

      sml::state<prefilling> + sml::event<event::prefill_done> = sml::state<decoding>,
      sml::state<prefilling> + sml::event<event::prefill_error> = sml::state<errored>,

      sml::state<decoding> + sml::event<event::decode_step_done>[guard::should_continue_decode] =
        sml::state<decoding>,
      sml::state<decoding> + sml::event<event::stop_condition_met> = sml::state<done>,
      sml::state<decoding> + sml::event<event::decode_step_error> = sml::state<errored>
    );
  }
};

struct sm : emel::sm<model> {
  explicit sm(emel::model::data & model_data) : model_(model_data) {}

 private:
  emel::model::data & model_;
  emel::tokenizer::sm tokenizer_ = {};
  int32_t status_code = 0;
  int32_t tokens_generated = 0;
  int32_t max_tokens = 0;
};

}  // namespace emel::generator
