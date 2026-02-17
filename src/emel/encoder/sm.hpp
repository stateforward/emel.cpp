#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/encoder/actions.hpp"
#include "emel/encoder/events.hpp"
#include "emel/encoder/guards.hpp"

namespace emel::encoder {

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct initialized {};
    struct pretokenizing {};
    struct selecting_algorithm {};
    struct merging {};
    struct searching {};
    struct scanning {};
    struct emitting_tokens {};
    struct applying_backend_postrules {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::encode> / action::on_encode_requested =
          sml::state<pretokenizing>,

      sml::state<pretokenizing> + sml::event<event::pretokenizing_done> /
          action::on_pretokenizing_done = sml::state<selecting_algorithm>,
      sml::state<pretokenizing> + sml::event<event::pretokenizing_error> /
          action::on_pretokenizing_error = sml::state<errored>,

      sml::state<selecting_algorithm> + sml::event<event::algorithm_selected>[guard::use_merging] =
          sml::state<merging>,
      sml::state<selecting_algorithm> +
          sml::event<event::algorithm_selected>[guard::use_searching] = sml::state<searching>,
      sml::state<selecting_algorithm> + sml::event<event::algorithm_selected>[guard::use_scanning] =
          sml::state<scanning>,

      sml::state<merging> + sml::event<event::algorithm_step_done> / action::on_algorithm_step_done =
          sml::state<emitting_tokens>,
      sml::state<merging> + sml::event<event::algorithm_step_error> /
          action::on_algorithm_step_error = sml::state<errored>,

      sml::state<searching> + sml::event<event::algorithm_step_done> /
          action::on_algorithm_step_done = sml::state<emitting_tokens>,
      sml::state<searching> + sml::event<event::algorithm_step_error> /
          action::on_algorithm_step_error = sml::state<errored>,

      sml::state<scanning> + sml::event<event::algorithm_step_done> / action::on_algorithm_step_done =
          sml::state<emitting_tokens>,
      sml::state<scanning> + sml::event<event::algorithm_step_error> /
          action::on_algorithm_step_error = sml::state<errored>,

      sml::state<emitting_tokens> + sml::event<event::emission_done> / action::on_emission_done =
          sml::state<applying_backend_postrules>,
      sml::state<emitting_tokens> + sml::event<event::emission_error> / action::on_emission_error =
          sml::state<errored>,

      sml::state<applying_backend_postrules> + sml::event<event::postrules_done> /
          action::on_postrules_done = sml::state<done>,
      sml::state<applying_backend_postrules> + sml::event<event::postrules_error> /
          action::on_postrules_error = sml::state<errored>,

      sml::state<done> + sml::event<event::tokenized_done> / action::dispatch_encoding_done_to_owner =
          sml::state<done>,
      sml::state<errored> + sml::event<event::tokenized_error> /
          action::dispatch_encoding_error_to_owner = sml::state<errored>
    );
  }
};

struct sm : emel::sm<model> {
  using emel::sm<model>::sm;

 private:
  // Runtime fields for algorithm sessions (SPM/BPE/WPM/UGM/RWKV/PLaMo2).
  event::backend_type selected_backend = event::backend_type::merging;
  uint32_t fragment_index = 0;
  uint32_t fragment_count = 0;
  uint32_t algorithm_step_count = 0;

  uint32_t merge_queue_size = 0;
  uint32_t merges_applied = 0;
  uint32_t search_frontier_size = 0;
  uint32_t scan_cursor = 0;

  bool parse_special = false;
  bool add_special = false;
  bool had_unknown_fallback = false;
  bool had_byte_fallback = false;

  int32_t n_tokens = 0;
};

}  // namespace emel::encoder
