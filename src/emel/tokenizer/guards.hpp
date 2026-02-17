#pragma once

#include "emel/tokenizer/actions.hpp"

namespace emel::tokenizer::guard {

struct can_tokenize {
  bool operator()(const event::tokenize & ev) const {
    if (ev.vocab == nullptr || ev.token_ids_out == nullptr || ev.token_count_out == nullptr) {
      return false;
    }
    return ev.token_capacity > 0;
  }
};

struct has_request {
  template <class Event>
  bool operator()(const Event & ev) const {
    return ev.request != nullptr;
  }
};

struct has_text {
  bool operator()(const event::tokenize & ev) const {
    return !ev.text.empty();
  }
  template <class Event>
  bool operator()(const Event & ev) const {
    return ev.request != nullptr && !ev.request->text.empty();
  }
};

struct no_text {
  template <class Event>
  bool operator()(const Event & ev) const {
    return !has_text{}(ev);
  }
};

struct has_special_tokens {
  template <class Event>
  bool operator()(const Event &, const action::context & ctx) const {
    return ctx.special_token_count > 0;
  }
};

struct no_special_tokens {
  template <class Event>
  bool operator()(const Event & ev, const action::context & ctx) const {
    return !has_special_tokens{}(ev, ctx);
  }
};

struct has_capacity {
  template <class Event>
  bool operator()(const Event & ev, const action::context & ctx) const {
    const event::tokenize * request = detail::request_from(ev);
    if (request == nullptr) {
      return false;
    }
    return ctx.token_count < request->token_capacity;
  }
};

struct no_capacity {
  template <class Event>
  bool operator()(const Event & ev, const action::context & ctx) const {
    return !has_capacity{}(ev, ctx);
  }
};

struct should_add_bos {
  template <class Event>
  bool operator()(const Event & ev) const {
    const event::tokenize * request = detail::request_from(ev);
    if (request == nullptr || request->vocab == nullptr) {
      return false;
    }
    return request->add_special && request->vocab->add_bos;
  }
};

struct no_prefix {
  template <class Event>
  bool operator()(const Event & ev) const {
    return !should_add_bos{}(ev);
  }
};

struct bos_id_valid {
  template <class Event>
  bool operator()(const Event & ev) const {
    const event::tokenize * request = detail::request_from(ev);
    if (request == nullptr || request->vocab == nullptr) {
      return false;
    }
    return request->vocab->bos_id >= 0;
  }
};

struct bos_id_invalid {
  template <class Event>
  bool operator()(const Event & ev) const {
    return !bos_id_valid{}(ev);
  }
};

struct should_add_sep {
  template <class Event>
  bool operator()(const Event & ev) const {
    const event::tokenize * request = detail::request_from(ev);
    if (request == nullptr || request->vocab == nullptr) {
      return false;
    }
    if (!request->add_special) {
      return false;
    }
    const auto model_type = detail::detect_model(*request->vocab);
    return model_type == detail::tokenizer_model::wpm && request->vocab->add_sep;
  }
};

struct should_add_eos {
  template <class Event>
  bool operator()(const Event & ev) const {
    const event::tokenize * request = detail::request_from(ev);
    if (request == nullptr || request->vocab == nullptr) {
      return false;
    }
    if (!request->add_special) {
      return false;
    }
    const auto model_type = detail::detect_model(*request->vocab);
    return model_type != detail::tokenizer_model::wpm && request->vocab->add_eos;
  }
};

struct no_suffix {
  template <class Event>
  bool operator()(const Event & ev) const {
    return !should_add_sep{}(ev) && !should_add_eos{}(ev);
  }
};

struct sep_id_valid {
  template <class Event>
  bool operator()(const Event & ev) const {
    const event::tokenize * request = detail::request_from(ev);
    if (request == nullptr || request->vocab == nullptr) {
      return false;
    }
    return request->vocab->sep_id >= 0;
  }
};

struct sep_id_invalid {
  template <class Event>
  bool operator()(const Event & ev) const {
    return !sep_id_valid{}(ev);
  }
};

struct eos_id_valid {
  template <class Event>
  bool operator()(const Event & ev) const {
    const event::tokenize * request = detail::request_from(ev);
    if (request == nullptr || request->vocab == nullptr) {
      return false;
    }
    return request->vocab->eos_id >= 0;
  }
};

struct eos_id_invalid {
  template <class Event>
  bool operator()(const Event & ev) const {
    return !eos_id_valid{}(ev);
  }
};

struct has_more_fragments {
  template <class Event>
  bool operator()(const Event &, const action::context & ctx) const {
    return ctx.fragment_index < ctx.fragment_count;
  }
};

struct no_more_fragments {
  template <class Event>
  bool operator()(const Event & ev, const action::context & ctx) const {
    return !has_more_fragments{}(ev, ctx);
  }
};

}  // namespace emel::tokenizer::guard
