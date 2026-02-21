#pragma once

#include "emel/sm.hpp"
#include "emel/tokenizer/preprocessor/actions.hpp"
#include "emel/tokenizer/preprocessor/model.hpp"

namespace emel::tokenizer::preprocessor::plamo2 {

struct plamo2_tag {};
struct model : emel::tokenizer::preprocessor::detail::model<plamo2_tag> {};

struct sm : public emel::sm<model> {
  using base_type = emel::sm<model>;

  sm() : base_type(context_) {}

  bool process_event(const event::preprocess & ev) {
    namespace sml = boost::sml;

    const bool accepted = base_type::process_event(ev);
    const bool ok = this->is(sml::state<detail::done>);
    const int32_t err =
        ok ? EMEL_OK
           : (context_.last_error != EMEL_OK ? context_.last_error
                                             : EMEL_ERR_BACKEND);

    if (ev.fragment_count_out != nullptr) {
      *ev.fragment_count_out = context_.fragment_count;
    }
    if (ev.error_out != nullptr) {
      *ev.error_out = err;
    }
    if (ok) {
      if (ev.dispatch_done != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_done(ev.owner_sm,
                         events::preprocess_done{&ev, context_.fragment_count});
      }
    } else {
      if (ev.dispatch_error != nullptr && ev.owner_sm != nullptr) {
        ev.dispatch_error(ev.owner_sm, events::preprocess_error{&ev, err});
      }
    }

    action::clear_request(context_);
    return accepted && ok;
  }

  using base_type::process_event;
  using base_type::visit_current_states;

  int32_t last_error() const noexcept { return context_.last_error; }
  size_t fragment_count() const noexcept { return context_.fragment_count; }

 private:
  action::context context_{};
};

}  // namespace emel::tokenizer::preprocessor::plamo2
