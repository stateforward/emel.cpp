#pragma once

#include <cstdint>

#include "emel/sm.hpp"
#include "emel/encoder/sm.hpp"
#include "emel/tokenizer/actions.hpp"
#include "emel/tokenizer/events.hpp"
#include "emel/tokenizer/guards.hpp"

namespace emel::tokenizer {

struct model {
  auto operator()() const {
    namespace sml = boost::sml;

    struct initialized {};
    struct partitioning_special {};
    struct selecting_backend {};
    struct applying_special_prefix {};
    struct encoding_fragments {};
    struct applying_special_suffix {};
    struct finalizing {};
    struct done {};
    struct errored {};

    return sml::make_transition_table(
      *sml::state<initialized> + sml::event<event::tokenize>[guard::can_tokenize] /
          action::on_tokenize_requested = sml::state<partitioning_special>,

      sml::state<partitioning_special> + sml::event<event::partitioning_special_done> /
          action::on_partitioning_special_done = sml::state<selecting_backend>,
      sml::state<partitioning_special> + sml::event<event::partitioning_special_error> /
          action::on_partitioning_special_error = sml::state<errored>,

      sml::state<selecting_backend> + sml::event<event::selecting_backend_done> /
          action::on_selecting_backend_done = sml::state<applying_special_prefix>,
      sml::state<selecting_backend> + sml::event<event::selecting_backend_error> /
          action::on_selecting_backend_error = sml::state<errored>,

      sml::state<applying_special_prefix> + sml::event<event::applying_special_prefix_done> /
          action::on_applying_special_prefix_done = sml::state<encoding_fragments>,
      sml::state<applying_special_prefix> + sml::event<event::applying_special_prefix_error> /
          action::on_applying_special_prefix_error = sml::state<errored>,

      sml::state<encoding_fragments> + sml::event<emel::encoder::event::tokenized_done> /
          action::on_encoding_fragment_done = sml::state<encoding_fragments>,
      sml::state<encoding_fragments> + sml::event<emel::encoder::event::tokenized_error> /
          action::on_encoding_fragment_error = sml::state<errored>,

      sml::state<encoding_fragments> + sml::event<event::encoding_fragment_done>[guard::has_more_fragments] /
          action::on_encoding_fragment_done = sml::state<encoding_fragments>,
      sml::state<encoding_fragments> + sml::event<event::encoding_fragment_done>[guard::no_more_fragments] /
          action::on_encoding_fragment_done = sml::state<applying_special_suffix>,
      sml::state<encoding_fragments> + sml::event<event::encoding_fragment_error> /
          action::on_encoding_fragment_error = sml::state<errored>,

      sml::state<applying_special_suffix> + sml::event<event::applying_special_suffix_done> /
          action::on_applying_special_suffix_done = sml::state<finalizing>,
      sml::state<applying_special_suffix> + sml::event<event::applying_special_suffix_error> /
          action::on_applying_special_suffix_error = sml::state<errored>,

      sml::state<finalizing> + sml::event<event::finalizing_done> / action::on_finalizing_done =
          sml::state<done>,
      sml::state<finalizing> + sml::event<event::finalizing_error> /
          action::on_finalizing_error = sml::state<errored>
    );
  }
};

struct sm : emel::sm<model> {
  using emel::sm<model>::sm;

 private:
  // Tokenizer metadata/state mirrored from llama.cpp vocab internals.
  uint32_t n_token_types = 0;
  int32_t max_token_len = 0;

  int32_t special_bos_id = 1;
  int32_t special_eos_id = 2;
  int32_t special_eot_id = -1;
  int32_t special_eom_id = -1;
  int32_t special_unk_id = 0;
  int32_t special_sep_id = -1;
  int32_t special_pad_id = -1;
  int32_t special_mask_id = -1;
  int32_t linefeed_id = 13;

  int32_t special_fim_pre_id = -1;
  int32_t special_fim_suf_id = -1;
  int32_t special_fim_mid_id = -1;
  int32_t special_fim_pad_id = -1;
  int32_t special_fim_rep_id = -1;
  int32_t special_fim_sep_id = -1;

  bool add_space_prefix = false;
  bool add_bos = false;
  bool add_eos = false;
  bool add_sep = false;
  bool ignore_merges = false;
  bool clean_spaces = false;
  bool remove_extra_whitespaces = false;
  bool escape_whitespaces = true;
  bool treat_whitespace_as_suffix = false;

  uint32_t cache_special_tokens_count = 0;
  uint32_t cache_token_to_piece_count = 0;
  uint32_t bpe_rank_count = 0;
  uint32_t precompiled_charsmap_size = 0;

  enum class vocab_type : uint8_t {
    none = 0,
    spm = 1,
    bpe = 2,
    wpm = 3,
    ugm = 4,
    rwkv = 5,
    plamo2 = 6,
  };

  enum class vocab_pre_type : uint8_t {
    default_pre = 0,
  };

  vocab_type type = vocab_type::spm;
  vocab_pre_type pre_type = vocab_pre_type::default_pre;

  emel::encoder::sm encoder = {};
  int32_t n_tokens = 0;
};

}  // namespace emel::tokenizer
