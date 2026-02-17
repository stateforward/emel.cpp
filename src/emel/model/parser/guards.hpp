#pragma once

#include "emel/emel.h"
#include "emel/model/parser/events.hpp"

namespace emel::model::parser::guard {

struct no_error {
  template <class Event>
  bool operator()(const Event & ev) const {
    return ev.err == EMEL_OK;
  }
};

struct has_error {
  template <class Event>
  bool operator()(const Event & ev) const {
    return ev.err != EMEL_OK;
  }
};

struct can_parse_architecture {
  bool operator()(const event::parse_model & ev) const { return ev.parse_architecture != nullptr; }
};

struct cannot_parse_architecture {
  bool operator()(const event::parse_model & ev) const { return !can_parse_architecture{}(ev); }
};

struct can_map_architecture {
  bool operator()(const events::parse_architecture_done & ev) const {
    return ev.request != nullptr && ev.request->map_architecture != nullptr;
  }
};

struct cannot_map_architecture {
  bool operator()(const events::parse_architecture_done & ev) const {
    return !can_map_architecture{}(ev);
  }
};

struct can_parse_hparams {
  bool operator()(const events::map_architecture_done & ev) const {
    return ev.request != nullptr && ev.request->parse_hparams != nullptr;
  }
};

struct cannot_parse_hparams {
  bool operator()(const events::map_architecture_done & ev) const {
    return !can_parse_hparams{}(ev);
  }
};

struct can_parse_vocab {
  bool operator()(const events::parse_hparams_done & ev) const {
    return ev.request != nullptr && ev.request->parse_vocab != nullptr;
  }
};

struct cannot_parse_vocab {
  bool operator()(const events::parse_hparams_done & ev) const {
    return !can_parse_vocab{}(ev);
  }
};

struct skip_map_tensors {
  bool operator()(const events::parse_vocab_done & ev) const {
    return ev.request != nullptr && !ev.request->map_tensors;
  }
};

struct can_map_tensors {
  bool operator()(const events::parse_vocab_done & ev) const {
    return ev.request != nullptr && ev.request->map_tensors &&
           ev.request->map_tensors_impl != nullptr;
  }
};

struct cannot_map_tensors {
  bool operator()(const events::parse_vocab_done & ev) const {
    if (ev.request == nullptr) {
      return true;
    }
    return ev.request->map_tensors && ev.request->map_tensors_impl == nullptr;
  }
};

}  // namespace emel::model::parser::guard
