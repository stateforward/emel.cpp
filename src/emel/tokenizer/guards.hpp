#pragma once

#include "emel/tokenizer/actions.hpp"

namespace emel::tokenizer::guard {

struct can_tokenize {
  bool operator()(const event::tokenize &ev) const noexcept {
    if (ev.vocab == nullptr || ev.token_ids_out == nullptr ||
        ev.token_count_out == nullptr) {
      return false;
    }
    return ev.token_capacity > 0;
  }
};

struct phase_ok {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.phase_error == EMEL_OK;
  }
};

struct phase_failed {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.phase_error != EMEL_OK;
  }
};

struct has_special_tokens {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.special_cache.count > 0;
  }
};

struct no_special_tokens {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.special_cache.count == 0;
  }
};

struct has_capacity {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.token_count < ctx.token_capacity;
  }
};

struct no_capacity {
  bool operator()(const action::context &ctx) const noexcept {
    return !has_capacity{}(ctx);
  }
};

struct should_add_bos {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.add_special && ctx.vocab != nullptr && ctx.vocab->add_bos;
  }
};

struct no_prefix {
  bool operator()(const action::context &ctx) const noexcept {
    return !should_add_bos{}(ctx);
  }
};

struct bos_id_valid {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.vocab != nullptr && ctx.vocab->bos_id >= 0;
  }
};

struct bos_id_invalid {
  bool operator()(const action::context &ctx) const noexcept {
    return !bos_id_valid{}(ctx);
  }
};

struct bos_ready {
  bool operator()(const action::context &ctx) const noexcept {
    return should_add_bos{}(ctx) && bos_id_valid{}(ctx) && has_capacity{}(ctx);
  }
};

struct bos_no_capacity {
  bool operator()(const action::context &ctx) const noexcept {
    return should_add_bos{}(ctx) && bos_id_valid{}(ctx) && no_capacity{}(ctx);
  }
};

struct bos_invalid_id {
  bool operator()(const action::context &ctx) const noexcept {
    return should_add_bos{}(ctx) && bos_id_invalid{}(ctx);
  }
};

struct should_add_sep {
  bool operator()(const action::context &ctx) const noexcept {
    if (!ctx.add_special || ctx.vocab == nullptr) {
      return false;
    }
    return ctx.model_slot == action::encoder_slot::wpm && ctx.vocab->add_sep;
  }
};

struct should_add_eos {
  bool operator()(const action::context &ctx) const noexcept {
    if (!ctx.add_special || ctx.vocab == nullptr) {
      return false;
    }
    return ctx.model_slot != action::encoder_slot::wpm && ctx.vocab->add_eos;
  }
};

struct no_suffix {
  bool operator()(const action::context &ctx) const noexcept {
    return !should_add_sep{}(ctx) && !should_add_eos{}(ctx);
  }
};

struct sep_id_valid {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.vocab != nullptr && ctx.vocab->sep_id >= 0;
  }
};

struct sep_id_invalid {
  bool operator()(const action::context &ctx) const noexcept {
    return !sep_id_valid{}(ctx);
  }
};

struct sep_ready {
  bool operator()(const action::context &ctx) const noexcept {
    return should_add_sep{}(ctx) && sep_id_valid{}(ctx) && has_capacity{}(ctx);
  }
};

struct sep_no_capacity {
  bool operator()(const action::context &ctx) const noexcept {
    return should_add_sep{}(ctx) && sep_id_valid{}(ctx) && no_capacity{}(ctx);
  }
};

struct sep_invalid_id {
  bool operator()(const action::context &ctx) const noexcept {
    return should_add_sep{}(ctx) && sep_id_invalid{}(ctx);
  }
};

struct eos_id_valid {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.vocab != nullptr && ctx.vocab->eos_id >= 0;
  }
};

struct eos_id_invalid {
  bool operator()(const action::context &ctx) const noexcept {
    return !eos_id_valid{}(ctx);
  }
};

struct eos_ready {
  bool operator()(const action::context &ctx) const noexcept {
    return should_add_eos{}(ctx) && eos_id_valid{}(ctx) && has_capacity{}(ctx);
  }
};

struct eos_no_capacity {
  bool operator()(const action::context &ctx) const noexcept {
    return should_add_eos{}(ctx) && eos_id_valid{}(ctx) && no_capacity{}(ctx);
  }
};

struct eos_invalid_id {
  bool operator()(const action::context &ctx) const noexcept {
    return should_add_eos{}(ctx) && eos_id_invalid{}(ctx);
  }
};

struct has_more_fragments {
  bool operator()(const action::context &ctx) const noexcept {
    return ctx.fragment_index < ctx.fragment_count;
  }
};

struct no_more_fragments {
  bool operator()(const action::context &ctx) const noexcept {
    return !has_more_fragments{}(ctx);
  }
};

struct fragment_is_token {
  bool operator()(const action::context &ctx) const noexcept {
    if (!has_more_fragments{}(ctx)) {
      return false;
    }
    return ctx.fragments[ctx.fragment_index].kind ==
           action::fragment_kind::token;
  }
};

struct fragment_is_raw {
  bool operator()(const action::context &ctx) const noexcept {
    if (!has_more_fragments{}(ctx)) {
      return false;
    }
    return ctx.fragments[ctx.fragment_index].kind ==
           action::fragment_kind::raw_text;
  }
};

struct more_fragments_no_capacity {
  bool operator()(const action::context &ctx) const noexcept {
    return has_more_fragments{}(ctx) && no_capacity{}(ctx);
  }
};

struct more_fragments_token {
  bool operator()(const action::context &ctx) const noexcept {
    return has_more_fragments{}(ctx) && fragment_is_token{}(ctx);
  }
};

struct more_fragments_raw {
  bool operator()(const action::context &ctx) const noexcept {
    return has_more_fragments{}(ctx) && fragment_is_raw{}(ctx);
  }
};

} // namespace emel::tokenizer::guard
