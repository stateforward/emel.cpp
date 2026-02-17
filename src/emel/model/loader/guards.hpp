#pragma once

#include <type_traits>

#include "emel/emel.h"
#include "emel/model/loader/events.hpp"

namespace emel::model::loader::guard {

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

struct has_arch_validate {
  bool operator()(const events::structure_validated & ev) const {
    return ev.request != nullptr && ev.request->validate_architecture;
  }
};

struct no_arch_validate {
  bool operator()(const events::structure_validated & ev) const {
    return ev.request != nullptr && !ev.request->validate_architecture;
  }
};

struct should_load_weights {
  bool operator()(const events::parsing_done & ev) const {
    return ev.request != nullptr && !ev.request->vocab_only;
  }
};

struct skip_weights {
  bool operator()(const events::parsing_done & ev) const {
    return ev.request != nullptr && ev.request->vocab_only;
  }
};

struct can_map_parser {
  bool operator()(const event::load & ev) const { return ev.map_parser != nullptr; }
};

struct cannot_map_parser {
  bool operator()(const event::load & ev) const { return !can_map_parser{}(ev); }
};

struct can_parse {
  bool operator()(const events::mapping_parser_done & ev) const {
    if (ev.request == nullptr) {
      return false;
    }
    return ev.request->dispatch_parse_model != nullptr &&
           ev.request->parser_sm != nullptr &&
           ev.request->loader_sm != nullptr &&
           ev.request->dispatch_parsing_done != nullptr &&
           ev.request->dispatch_parsing_error != nullptr;
  }
};

struct cannot_parse {
  bool operator()(const events::mapping_parser_done & ev) const { return !can_parse{}(ev); }
};

struct can_load_weights {
  bool operator()(const events::parsing_done & ev) const {
    if (ev.request == nullptr) {
      return false;
    }
    return ev.request->dispatch_load_weights != nullptr &&
           ev.request->weight_loader_sm != nullptr &&
           ev.request->loader_sm != nullptr &&
           ev.request->dispatch_loading_done != nullptr &&
           ev.request->dispatch_loading_error != nullptr;
  }
};

struct cannot_load_weights {
  bool operator()(const events::parsing_done & ev) const { return !can_load_weights{}(ev); }
};

struct can_map_layers {
  bool operator()(const events::loading_done & ev) const {
    return ev.request != nullptr && ev.request->map_layers != nullptr;
  }
};

struct cannot_map_layers {
  bool operator()(const events::loading_done & ev) const { return !can_map_layers{}(ev); }
};

struct can_validate_structure {
  template <class Event>
  bool operator()(const Event & ev) const {
    const event::load * request = nullptr;
    if constexpr (std::is_same_v<Event, events::layers_mapped>) {
      request = ev.request;
    } else if constexpr (std::is_same_v<Event, events::parsing_done>) {
      request = ev.request;
    } else {
      return false;
    }
    if (request == nullptr) {
      return false;
    }
    if (!request->check_tensors) {
      return true;
    }
    return request->validate_structure != nullptr;
  }
};

struct cannot_validate_structure {
  template <class Event>
  bool operator()(const Event & ev) const {
    return !can_validate_structure{}(ev);
  }
};

struct can_validate_architecture {
  bool operator()(const events::structure_validated & ev) const {
    if (ev.request == nullptr) {
      return false;
    }
    if (!ev.request->validate_architecture) {
      return true;
    }
    return ev.request->validate_architecture_impl != nullptr;
  }
};

struct cannot_validate_architecture {
  bool operator()(const events::structure_validated & ev) const {
    return !can_validate_architecture{}(ev);
  }
};

struct should_load_weights_and_can_load {
  bool operator()(const events::parsing_done & ev) const {
    return should_load_weights{}(ev) && can_load_weights{}(ev);
  }
};

struct should_load_weights_and_cannot_load {
  bool operator()(const events::parsing_done & ev) const {
    return should_load_weights{}(ev) && cannot_load_weights{}(ev);
  }
};

struct skip_weights_and_can_validate_structure {
  bool operator()(const events::parsing_done & ev) const {
    return skip_weights{}(ev) && can_validate_structure{}(ev);
  }
};

struct skip_weights_and_cannot_validate_structure {
  bool operator()(const events::parsing_done & ev) const {
    return skip_weights{}(ev) && cannot_validate_structure{}(ev);
  }
};

struct has_arch_validate_and_can_validate_architecture {
  bool operator()(const events::structure_validated & ev) const {
    return has_arch_validate{}(ev) && can_validate_architecture{}(ev);
  }
};

struct has_arch_validate_and_cannot_validate_architecture {
  bool operator()(const events::structure_validated & ev) const {
    return has_arch_validate{}(ev) && cannot_validate_architecture{}(ev);
  }
};

}  // namespace emel::model::loader::guard
